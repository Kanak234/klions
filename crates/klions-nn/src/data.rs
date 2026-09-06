//! Datasets — EIR-014 … EIR-017, FR-AI-012, FR-AI-013, NFR-S-005, NFR-S-006.
//!
//! Loaders are hardened against malformed input: declared dimensions are
//! validated against the actual file length *before* any allocation, so a
//! corrupt header can never cause an over-read or an unbounded allocation.

use std::path::{Component, Path, PathBuf};

use klions_tensor::kernels as k;
use klions_tensor::{Rng, Tensor, TensorResult};

#[derive(Clone, Debug)]
pub enum DatasetError {
    NotFound(String),
    BadMagic { path: String, found: u32 },
    Truncated { path: String, need: usize, have: usize },
    CountMismatch { features: usize, labels: usize },
    MalformedCsv { path: String, line: usize, reason: String },
    BadLabelColumn { path: String, column: usize, width: usize },
    Empty(String),
    PathTraversal(String),
    Io(String),
}

impl DatasetError {
    pub fn code(&self) -> &'static str {
        use klions_diagnostics::codes;
        match self {
            DatasetError::NotFound(_) => codes::DATASET_NOT_FOUND,
            DatasetError::BadMagic { .. } => codes::IDX_BAD_MAGIC,
            DatasetError::Truncated { .. } => codes::IDX_TRUNCATED,
            DatasetError::CountMismatch { .. } => codes::SAMPLE_COUNT_MISMATCH,
            DatasetError::MalformedCsv { .. } => codes::CSV_MALFORMED,
            DatasetError::BadLabelColumn { .. } => codes::CSV_BAD_LABEL_COLUMN,
            DatasetError::Empty(_) => codes::DATASET_EMPTY,
            DatasetError::PathTraversal(_) => codes::PATH_TRAVERSAL,
            DatasetError::Io(_) => codes::IO_ERROR,
        }
    }

    pub fn help(&self) -> Option<String> {
        match self {
            DatasetError::NotFound(p) => Some(format!(
                "check the path; it is resolved relative to the source file, not the shell's \
                 working directory. Looked for: {}",
                p
            )),
            DatasetError::BadMagic { .. } => Some(
                "IDX files begin with a four-byte magic number: two zero bytes, a type byte, \
                 then the number of dimensions"
                    .into(),
            ),
            DatasetError::Truncated { .. } => {
                Some("the header declares more data than the file contains; it is truncated or corrupt".into())
            }
            DatasetError::CountMismatch { .. } => {
                Some("the feature file and the label file must describe the same number of samples".into())
            }
            DatasetError::BadLabelColumn { width, .. } => {
                Some(format!("this file has {} columns, indexed from 0", width))
            }
            DatasetError::PathTraversal(_) => {
                Some("dataset paths may not escape the project root with `..`".into())
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::NotFound(p) => write!(f, "dataset file not found: {}", p),
            DatasetError::BadMagic { path, found } => write!(
                f,
                "{} is not a valid IDX file (magic number 0x{:08x})",
                path, found
            ),
            DatasetError::Truncated { path, need, have } => write!(
                f,
                "{} is truncated: the header declares {} bytes of data but the file holds {}",
                path, need, have
            ),
            DatasetError::CountMismatch { features, labels } => write!(
                f,
                "sample count mismatch: the feature file has {} samples, the label file has {}",
                features, labels
            ),
            DatasetError::MalformedCsv { path, line, reason } => {
                write!(f, "{}:{}: malformed CSV — {}", path, line, reason)
            }
            DatasetError::BadLabelColumn { path, column, width } => write!(
                f,
                "{}: label column {} is out of range for a file {} columns wide",
                path, column, width
            ),
            DatasetError::Empty(p) => write!(f, "dataset {} contains no samples", p),
            DatasetError::PathTraversal(p) => {
                write!(f, "dataset path escapes the project root: {}", p)
            }
            DatasetError::Io(m) => write!(f, "{}", m),
        }
    }
}

pub type DataResult<T> = Result<T, DatasetError>;

/// A dataset held once in memory (FR-AI-013). Batches are views over it.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// `[samples, features...]`
    pub features: Tensor,
    /// `[samples]` of class indices, or `[samples, k]` once one-hot.
    pub labels: Tensor,
    pub name: String,
    /// Permutation applied by `shuffle()`; identity otherwise.
    order: Vec<usize>,
    pub batch_size: usize,
    pub classes: Option<usize>,
}

impl Dataset {
    pub fn new(features: Tensor, labels: Tensor, name: impl Into<String>) -> DataResult<Dataset> {
        let n = features.dim(0);
        let m = labels.dim(0);
        // EIR-016: validate the counts and name both.
        if n != m {
            return Err(DatasetError::CountMismatch { features: n, labels: m });
        }
        if n == 0 {
            return Err(DatasetError::Empty(name.into()));
        }
        Ok(Dataset {
            features,
            labels,
            name: name.into(),
            order: (0..n).collect(),
            batch_size: 32,
            classes: None,
        })
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    /// Width of one flattened sample — checked against the model (FR-TYP-013).
    pub fn feature_width(&self) -> usize {
        self.features.numel() / self.features.dim(0).max(1)
    }
    pub fn label_width(&self) -> usize {
        self.labels.numel() / self.labels.dim(0).max(1)
    }

    pub fn batch(&self, n: usize) -> Dataset {
        let mut d = self.clone();
        d.batch_size = n.max(1);
        d
    }

    /// FR-AI-012 / FR-RT-010: shuffling draws from the global generator.
    pub fn shuffle(&self, rng: &mut Rng) -> Dataset {
        let mut d = self.clone();
        rng.shuffle(&mut d.order);
        d
    }

    /// Split into (first, second) by fraction, without reordering.
    pub fn split(&self, fraction: f32) -> (Dataset, Dataset) {
        let cut = ((self.len() as f32) * fraction.clamp(0.0, 1.0)) as usize;
        let mut a = self.clone();
        let mut b = self.clone();
        a.order = self.order[..cut].to_vec();
        b.order = self.order[cut..].to_vec();
        (a, b)
    }

    /// Scale features into [0, 1] by their observed maximum.
    pub fn normalize(&self) -> TensorResult<Dataset> {
        let mut d = self.clone();
        let v = self.features.to_vec();
        let max = v.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        if max > 0.0 {
            d.features = k::scalar_mul(&self.features, 1.0 / max)?;
        }
        Ok(d)
    }

    /// Standardize to zero mean and unit variance.
    pub fn standardize(&self) -> TensorResult<Dataset> {
        let mut d = self.clone();
        let v = self.features.to_vec();
        let n = v.len().max(1) as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let sd = var.sqrt().max(1e-8);
        d.features = k::unary_op(&self.features, |x| (x - mean) / sd)?;
        Ok(d)
    }

    /// FR-AI-012: expand integer labels into one-hot rows.
    pub fn one_hot(&self, classes: usize) -> TensorResult<Dataset> {
        let mut d = self.clone();
        if self.labels.rank() > 1 && self.labels.dim(1) == classes {
            d.classes = Some(classes);
            return Ok(d); // already one-hot
        }
        let flat = self.labels.reshape(&[self.labels.dim(0) as isize])?;
        d.labels = k::one_hot(&flat, classes)?;
        d.classes = Some(classes);
        Ok(d)
    }

    /// Number of batches at the current batch size.
    pub fn batch_count(&self) -> usize {
        (self.len() + self.batch_size - 1) / self.batch_size.max(1)
    }

    /// FR-AI-013: materialize exactly one batch, never the whole set again.
    pub fn get_batch(&self, index: usize) -> TensorResult<(Tensor, Tensor)> {
        let start = index * self.batch_size;
        let end = (start + self.batch_size).min(self.len());
        self.gather(&self.order[start..end])
    }

    /// Gather the given sample indices into a contiguous batch.
    pub fn gather(&self, indices: &[usize]) -> TensorResult<(Tensor, Tensor)> {
        let fw = self.feature_width();
        let lw = self.label_width();
        let fv = self.features.to_vec();
        let lv = self.labels.to_vec();
        let mut fb = Vec::with_capacity(indices.len() * fw);
        let mut lb = Vec::with_capacity(indices.len() * lw);
        for &i in indices {
            fb.extend_from_slice(&fv[i * fw..(i + 1) * fw]);
            lb.extend_from_slice(&lv[i * lw..(i + 1) * lw]);
        }
        let n = indices.len();
        let x = Tensor::from_vec(fb, vec![n, fw])?;
        // Preserve the label *rank*, not just the width. A [n, 1] label column
        // must stay rank-2 so it lines up with a single-output model; collapsing
        // it to [n] would fail the loss shape check for no good reason.
        let y = if self.labels.rank() > 1 {
            Tensor::from_vec(lb, vec![n, lw])?
        } else {
            Tensor::from_vec(lb, vec![n])?
        };
        Ok((x, y))
    }

    /// One sample as a tensor shaped like the source rows.
    pub fn get_sample(&self, i: usize) -> TensorResult<Tensor> {
        let idx = *self.order.get(i).unwrap_or(&0);
        let fw = self.feature_width();
        let v = self.features.to_vec();
        let mut shape = self.features.shape.clone();
        shape[0] = 1;
        Tensor::from_vec(v[idx * fw..(idx + 1) * fw].to_vec(), shape)
    }

    pub fn get_label(&self, i: usize) -> TensorResult<f32> {
        let idx = *self.order.get(i).unwrap_or(&0);
        let lw = self.label_width();
        let v = self.labels.to_vec();
        if lw == 1 {
            return Ok(v[idx]);
        }
        // One-hot: report the class index.
        let row = &v[idx * lw..(idx + 1) * lw];
        let mut best = 0usize;
        for (j, &x) in row.iter().enumerate() {
            if x > row[best] {
                best = j;
            }
        }
        Ok(best as f32)
    }

    pub fn describe(&self) -> String {
        format!(
            "Dataset {} — {} samples, feature width {}, label width {}",
            self.name,
            self.len(),
            self.feature_width(),
            self.label_width()
        )
    }
}

// ============================ IDX (EIR-014) ============================

/// Read an IDX file. NFR-S-005: dimensions are validated against the actual
/// file length before any allocation is attempted.
pub fn read_idx(path: &Path) -> DataResult<(Vec<f32>, Vec<usize>)> {
    let display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => DatasetError::NotFound(display.clone()),
        _ => DatasetError::Io(format!("{}: {}", display, e)),
    })?;

    if bytes.len() < 4 {
        return Err(DatasetError::Truncated {
            path: display,
            need: 4,
            have: bytes.len(),
        });
    }
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    // Layout: 0x0000 TT DD — TT is the element type, DD the dimension count.
    let zero = (magic >> 16) & 0xffff;
    let type_code = (magic >> 8) & 0xff;
    let ndims = (magic & 0xff) as usize;
    if zero != 0 || ndims == 0 || ndims > 4 {
        return Err(DatasetError::BadMagic { path: display, found: magic });
    }
    let elem_size = match type_code {
        0x08 | 0x09 => 1usize, // unsigned byte / signed byte
        0x0B => 2,             // short
        0x0C | 0x0D => 4,      // int / float
        0x0E => 8,             // double
        _ => return Err(DatasetError::BadMagic { path: display, found: magic }),
    };

    let header = 4 + 4 * ndims;
    if bytes.len() < header {
        return Err(DatasetError::Truncated {
            path: display,
            need: header,
            have: bytes.len(),
        });
    }
    let mut dims = Vec::with_capacity(ndims);
    let mut count: usize = 1;
    for i in 0..ndims {
        let o = 4 + 4 * i;
        let d = u32::from_be_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        // Overflow-checked product: a corrupt header must not wrap.
        count = match count.checked_mul(d) {
            Some(c) => c,
            None => {
                return Err(DatasetError::Truncated {
                    path: display,
                    need: usize::MAX,
                    have: bytes.len(),
                })
            }
        };
        dims.push(d);
    }
    let need = header + count * elem_size;
    if bytes.len() < need {
        // NFR-S-005: refuse before allocating anything.
        return Err(DatasetError::Truncated { path: display, need, have: bytes.len() });
    }

    let mut out = Vec::with_capacity(count);
    let body = &bytes[header..need];
    match type_code {
        0x08 => out.extend(body.iter().map(|&b| b as f32)),
        0x09 => out.extend(body.iter().map(|&b| b as i8 as f32)),
        0x0B => {
            for c in body.chunks_exact(2) {
                out.push(i16::from_be_bytes([c[0], c[1]]) as f32);
            }
        }
        0x0C => {
            for c in body.chunks_exact(4) {
                out.push(i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
        }
        0x0D => {
            for c in body.chunks_exact(4) {
                out.push(f32::from_be_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        0x0E => {
            for c in body.chunks_exact(8) {
                out.push(f64::from_be_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]) as f32);
            }
        }
        _ => unreachable!("type code validated above"),
    }
    Ok((out, dims))
}

/// Load an IDX feature/label pair as a Dataset.
pub fn from_idx(images: &Path, labels: &Path) -> DataResult<Dataset> {
    let (fv, fdims) = read_idx(images)?;
    let (lv, ldims) = read_idx(labels)?;
    let n = *fdims.first().unwrap_or(&0);
    let m = *ldims.first().unwrap_or(&0);
    if n != m {
        return Err(DatasetError::CountMismatch { features: n, labels: m });
    }
    let width: usize = fdims[1..].iter().product::<usize>().max(1);
    let features = Tensor::from_vec(fv, vec![n, width])
        .map_err(|e| DatasetError::Io(e.to_string()))?;
    let labels_t =
        Tensor::from_vec(lv, vec![m]).map_err(|e| DatasetError::Io(e.to_string()))?;
    let name = images
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "idx".to_string());
    Dataset::new(features, labels_t, name)
}

// ============================ CSV (EIR-015) ============================

/// RFC 4180 CSV with an optional header row and a caller-specified label column.
pub fn from_csv(path: &Path, label_column: usize, has_header: bool) -> DataResult<Dataset> {
    let display = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => DatasetError::NotFound(display.clone()),
        _ => DatasetError::Io(format!("{}: {}", display, e)),
    })?;

    let rows = parse_csv(&text, &display)?;
    let rows = if has_header && !rows.is_empty() { &rows[1..] } else { &rows[..] };
    if rows.is_empty() {
        return Err(DatasetError::Empty(display));
    }
    let width = rows[0].len();
    if label_column >= width {
        return Err(DatasetError::BadLabelColumn {
            path: display,
            column: label_column,
            width,
        });
    }

    let mut features = Vec::with_capacity(rows.len() * (width - 1));
    let mut labels = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(DatasetError::MalformedCsv {
                path: display,
                line: i + 1 + has_header as usize,
                reason: format!("expected {} fields, found {}", width, row.len()),
            });
        }
        for (j, cell) in row.iter().enumerate() {
            let v: f32 = cell.trim().parse().map_err(|_| DatasetError::MalformedCsv {
                path: display.clone(),
                line: i + 1 + has_header as usize,
                reason: format!("field {} is not a number: `{}`", j, cell.trim()),
            })?;
            if j == label_column {
                labels.push(v);
            } else {
                features.push(v);
            }
        }
    }
    let n = rows.len();
    let fw = width - 1;
    let features = Tensor::from_vec(features, vec![n, fw])
        .map_err(|e| DatasetError::Io(e.to_string()))?;
    let labels_t =
        Tensor::from_vec(labels, vec![n]).map_err(|e| DatasetError::Io(e.to_string()))?;
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "csv".to_string());
    Dataset::new(features, labels_t, name)
}

/// RFC 4180 field splitting, including quoted fields and doubled quotes.
fn parse_csv(text: &str, path: &str) -> DataResult<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut line = 1usize;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        field.push('"');
                    } else {
                        in_quotes = false;
                    }
                }
                '\n' => {
                    line += 1;
                    field.push('\n');
                }
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => in_quotes = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                if row.len() > 1 || !row[0].trim().is_empty() {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
                line += 1;
            }
            _ => field.push(c),
        }
    }
    if in_quotes {
        return Err(DatasetError::MalformedCsv {
            path: path.to_string(),
            line,
            reason: "unterminated quoted field".to_string(),
        });
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        if row.len() > 1 || !row[0].trim().is_empty() {
            rows.push(row);
        }
    }
    Ok(rows)
}

// ============================ path handling ============================

/// EIR-017: resolve relative to the source file's directory.
/// NFR-S-006: reject traversal beyond the project root.
pub fn resolve_dataset_path(source_dir: &Path, raw: &str, root: Option<&Path>) -> DataResult<PathBuf> {
    let p = Path::new(raw);
    let joined = if p.is_absolute() { p.to_path_buf() } else { source_dir.join(p) };

    // Anchor before normalizing. Normalizing a relative path first would let
    // `..` components walk past its start and vanish, so `../../../etc/passwd`
    // would collapse to `etc/passwd` and then look like a path inside the
    // project. Anchoring first keeps every `..` meaningful.
    let normalized = normalize(&anchor(&joined));

    if let Some(root) = root {
        let root_abs = normalize(&anchor(root));
        if !normalized.starts_with(&root_abs) && !p.is_absolute() {
            return Err(DatasetError::PathTraversal(raw.to_string()));
        }
    }
    Ok(normalized)
}

/// Make a path absolute by joining it to the current directory. Purely
/// lexical, so it works for paths that do not exist yet.
fn anchor(p: &Path) -> PathBuf {
    if p.is_absolute() {
        return p.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(p),
        Err(_) => p.to_path_buf(),
    }
}

/// Lexical normalization: no filesystem access, so it works on missing paths.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("klions_test_{}_{}", std::process::id(), name));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    fn idx_bytes(dims: &[u32], data: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8, 0u8, 0x08u8, dims.len() as u8];
        for d in dims {
            v.extend_from_slice(&d.to_be_bytes());
        }
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn idx_round_trips() {
        let p = tmp("ok.idx", &idx_bytes(&[2, 3], &[1, 2, 3, 4, 5, 6]));
        let (v, dims) = read_idx(&p).unwrap();
        assert_eq!(dims, vec![2, 3]);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn idx_rejects_bad_magic() {
        let p = tmp("bad.idx", &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 1]);
        assert!(matches!(read_idx(&p), Err(DatasetError::BadMagic { .. })));
        std::fs::remove_file(p).ok();
    }

    /// NFR-S-005: a header claiming more data than exists must be refused
    /// before allocation, not produce an over-read.
    #[test]
    fn idx_rejects_truncated_payload() {
        let p = tmp("short.idx", &idx_bytes(&[1000, 1000], &[1, 2, 3]));
        match read_idx(&p) {
            Err(DatasetError::Truncated { need, have, .. }) => {
                assert_eq!(need, 12 + 1_000_000);
                assert!(have < need);
            }
            other => panic!("expected Truncated, got {:?}", other.map(|(v, _)| v.len())),
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn idx_rejects_overflowing_dimensions() {
        let mut v = vec![0u8, 0u8, 0x08u8, 4u8];
        for _ in 0..4 {
            v.extend_from_slice(&u32::MAX.to_be_bytes());
        }
        let p = tmp("overflow.idx", &v);
        assert!(read_idx(&p).is_err());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn csv_parses_with_header_and_label_column() {
        let p = tmp("d.csv", b"a,b,label\n1,2,0\n3,4,1\n");
        let d = from_csv(&p, 2, true).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.feature_width(), 2);
        assert_eq!(d.features.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(d.labels.to_vec(), vec![0.0, 1.0]);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn csv_reports_bad_label_column() {
        let p = tmp("d2.csv", b"1,2\n3,4\n");
        assert!(matches!(
            from_csv(&p, 9, false),
            Err(DatasetError::BadLabelColumn { .. })
        ));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn csv_reports_ragged_rows() {
        let p = tmp("d3.csv", b"1,2,3\n4,5\n");
        assert!(matches!(
            from_csv(&p, 0, false),
            Err(DatasetError::MalformedCsv { .. })
        ));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn dataset_batching_is_exact() {
        let f = Tensor::from_vec((0..20).map(|i| i as f32).collect(), vec![10, 2]).unwrap();
        let l = Tensor::from_vec((0..10).map(|i| (i % 2) as f32).collect(), vec![10]).unwrap();
        let d = Dataset::new(f, l, "t").unwrap().batch(3);
        assert_eq!(d.batch_count(), 4);
        let (x, y) = d.get_batch(0).unwrap();
        assert_eq!(x.shape, vec![3, 2]);
        assert_eq!(y.shape, vec![3]);
        let (xl, _) = d.get_batch(3).unwrap();
        assert_eq!(xl.shape, vec![1, 2]); // final partial batch
    }

    #[test]
    fn one_hot_widens_labels() {
        let f = Tensor::zeros(&[3, 2]).unwrap();
        let l = Tensor::from_vec(vec![0.0, 2.0, 1.0], vec![3]).unwrap();
        let d = Dataset::new(f, l, "t").unwrap().one_hot(3).unwrap();
        assert_eq!(d.label_width(), 3);
        assert_eq!(
            d.labels.to_vec(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn split_partitions_without_overlap() {
        let f = Tensor::zeros(&[10, 1]).unwrap();
        let l = Tensor::zeros(&[10]).unwrap();
        let d = Dataset::new(f, l, "t").unwrap();
        let (a, b) = d.split(0.3);
        assert_eq!((a.len(), b.len()), (3, 7));
    }

    #[test]
    fn count_mismatch_names_both_counts() {
        let f = Tensor::zeros(&[5, 2]).unwrap();
        let l = Tensor::zeros(&[3]).unwrap();
        match Dataset::new(f, l, "t") {
            Err(DatasetError::CountMismatch { features, labels }) => {
                assert_eq!((features, labels), (5, 3));
            }
            _ => panic!("expected a count mismatch"),
        }
    }

    /// NFR-S-006: `..` may not escape the project root.
    #[test]
    fn path_traversal_is_rejected() {
        let root = Path::new("/project");
        let src = Path::new("/project/src");
        assert!(resolve_dataset_path(src, "../data/x.idx", Some(root)).is_ok());
        assert!(resolve_dataset_path(src, "../../etc/passwd", Some(root)).is_err());
    }

    /// A relative source directory must still resolve against an absolute
    /// project root. Getting this wrong rejected every legitimate path as
    /// traversal, which a clean-room build caught.
    #[test]
    fn a_relative_source_dir_resolves_against_an_absolute_root() {
        let cwd = std::env::current_dir().unwrap();
        let src = Path::new("examples");
        let root = cwd.as_path();
        assert!(
            resolve_dataset_path(src, "data/x.csv", Some(root)).is_ok(),
            "a path inside the project must be accepted"
        );
        assert!(
            resolve_dataset_path(src, "../../../etc/passwd", Some(root)).is_err(),
            "a path escaping the project must still be rejected"
        );
    }

    #[test]
    fn paths_resolve_relative_to_the_source_file() {
        let p = resolve_dataset_path(Path::new("/a/b"), "./data/t.idx", None).unwrap();
        assert_eq!(p, PathBuf::from("/a/b/data/t.idx"));
    }
}
