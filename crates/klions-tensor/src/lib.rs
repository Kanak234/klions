//! KLIONS tensor engine — FR-TEN-001 … FR-TEN-014.
//!
//! Dense, row-major, strided `f32` tensors of rank 0…4. No BLAS, no external
//! numerical library (DC-007): every kernel here is first-party.

use std::cell::RefCell;
use std::rc::Rc;

pub mod error;
pub mod kernels;
pub mod rng;

pub use error::{TensorError, TensorResult};
pub use rng::Rng;

/// FR-TEN-001 maximum rank.
pub const MAX_RANK: usize = 4;
/// FR-TEN-010 parallelism threshold.
pub const PARALLEL_THRESHOLD: usize = 65_536;

/// DC-006: the device field exists from the outset; only `Cpu` is accepted.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Device {
    #[default]
    Cpu,
}

impl Device {
    pub fn parse(s: &str) -> TensorResult<Device> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" | "rocm" | "vulkan" => Err(TensorError::UnsupportedDevice {
                requested: s.to_string(),
                release: "0.3.0",
            }),
            other => Err(TensorError::UnsupportedDevice {
                requested: other.to_string(),
                release: "0.3.0",
            }),
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
        }
    }
}

/// Shared, reference-counted element buffer (FR-RT-009, DC-005).
pub type Buffer = Rc<RefCell<Vec<f32>>>;

pub fn new_buffer(v: Vec<f32>) -> Buffer {
    Rc::new(RefCell::new(v))
}

/// FR-TEN-002: the tensor descriptor.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Shared data buffer — views alias it (FR-TEN-007).
    pub buf: Buffer,
    /// Element offset of this view into `buf`.
    pub offset: usize,
    pub shape: Vec<usize>,
    /// Strides in elements, not bytes.
    pub strides: Vec<usize>,
    pub device: Device,
    /// Autodiff bookkeeping (FR-AI-001 … 003).
    pub requires_grad: bool,
    pub grad: Option<Buffer>,
    /// Index of the producing node on the active tape, if any.
    pub node: Option<usize>,
}

impl Tensor {
    // ---------------- construction (FR-TEN-003) ----------------

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> TensorResult<Tensor> {
        let n: usize = shape.iter().product();
        if shape.len() > MAX_RANK {
            return Err(TensorError::BadRank { rank: shape.len(), max: MAX_RANK });
        }
        if n != data.len() {
            return Err(TensorError::ReshapeSize { elements: data.len(), shape: shape.clone() });
        }
        let strides = contiguous_strides(&shape);
        Ok(Tensor {
            buf: new_buffer(data),
            offset: 0,
            shape,
            strides,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    pub fn scalar(v: f32) -> Tensor {
        Tensor::from_vec(vec![v], vec![]).unwrap()
    }

    pub fn zeros(shape: &[usize]) -> TensorResult<Tensor> {
        let n = check_alloc(shape)?;
        Tensor::from_vec(vec![0.0; n], shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> TensorResult<Tensor> {
        Tensor::full(shape, 1.0)
    }

    pub fn full(shape: &[usize], v: f32) -> TensorResult<Tensor> {
        let n = check_alloc(shape)?;
        Tensor::from_vec(vec![v; n], shape.to_vec())
    }

    pub fn eye(n: usize) -> TensorResult<Tensor> {
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            d[i * n + i] = 1.0;
        }
        Tensor::from_vec(d, vec![n, n])
    }

    pub fn arange(start: f32, end: f32, step: f32) -> TensorResult<Tensor> {
        if step == 0.0 {
            return Err(TensorError::BadArgument("arange step must be non-zero".into()));
        }
        let mut d = Vec::new();
        let mut x = start;
        while (step > 0.0 && x < end) || (step < 0.0 && x > end) {
            d.push(x);
            x += step;
        }
        let n = d.len();
        Tensor::from_vec(d, vec![n])
    }

    pub fn rand_uniform(shape: &[usize], lo: f32, hi: f32, rng: &mut Rng) -> TensorResult<Tensor> {
        let n = check_alloc(shape)?;
        let d = (0..n).map(|_| rng.uniform(lo, hi)).collect();
        Tensor::from_vec(d, shape.to_vec())
    }

    pub fn rand_normal(shape: &[usize], mean: f32, std: f32, rng: &mut Rng) -> TensorResult<Tensor> {
        let n = check_alloc(shape)?;
        let d = (0..n).map(|_| rng.normal(mean, std)).collect();
        Tensor::from_vec(d, shape.to_vec())
    }

    /// FR-AI-011: Glorot (Xavier) uniform.
    pub fn glorot_uniform(fan_in: usize, fan_out: usize, rng: &mut Rng) -> TensorResult<Tensor> {
        let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
        Tensor::rand_uniform(&[fan_in, fan_out], -limit, limit, rng)
    }

    // ---------------- shape queries ----------------

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn dim(&self, i: usize) -> usize {
        self.shape.get(i).copied().unwrap_or(1)
    }
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }
    pub fn shape_str(&self) -> String {
        format!(
            "[{}]",
            self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        )
    }

    // ---------------- element access ----------------

    /// Materialize into a contiguous `Vec<f32>` in logical order.
    pub fn to_vec(&self) -> Vec<f32> {
        let b = self.buf.borrow();
        if self.is_contiguous() && self.offset == 0 && b.len() == self.numel() {
            return b.clone();
        }
        let n = self.numel();
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; self.rank()];
        for _ in 0..n {
            let mut off = self.offset;
            for (k, i) in idx.iter().enumerate() {
                off += i * self.strides[k];
            }
            out.push(b[off]);
            bump_index(&mut idx, &self.shape);
        }
        out
    }

    /// A contiguous tensor with the same logical contents.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }
        let mut t = Tensor::from_vec(self.to_vec(), self.shape.clone()).unwrap();
        t.device = self.device;
        t
    }

    pub fn item(&self) -> TensorResult<f32> {
        if self.numel() != 1 {
            return Err(TensorError::BadArgument(format!(
                "item() expects a single-element tensor, this one has shape {}",
                self.shape_str()
            )));
        }
        Ok(self.to_vec()[0])
    }

    /// FR-TEN-014: bounds-checked indexing that names axis, index, and bound.
    pub fn get(&self, index: &[usize]) -> TensorResult<f32> {
        let off = self.offset_of(index)?;
        Ok(self.buf.borrow()[off])
    }

    pub fn set(&self, index: &[usize], v: f32) -> TensorResult<()> {
        let off = self.offset_of(index)?;
        self.buf.borrow_mut()[off] = v;
        Ok(())
    }

    fn offset_of(&self, index: &[usize]) -> TensorResult<usize> {
        if index.len() != self.rank() {
            return Err(TensorError::BadRank { rank: index.len(), max: self.rank() });
        }
        let mut off = self.offset;
        for (axis, &i) in index.iter().enumerate() {
            if i >= self.shape[axis] {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index: i as isize,
                    bound: self.shape[axis],
                });
            }
            off += i * self.strides[axis];
        }
        Ok(off)
    }

    // ---------------- shape operations (FR-TEN-007, 008) ----------------

    /// FR-TEN-008: one `-1` dimension is inferred from the element count.
    pub fn reshape(&self, dims: &[isize]) -> TensorResult<Tensor> {
        let total = self.numel();
        let mut wild = None;
        let mut known = 1usize;
        for (i, &d) in dims.iter().enumerate() {
            if d == -1 {
                if wild.is_some() {
                    return Err(TensorError::BadArgument(
                        "reshape accepts at most one `-1` dimension".into(),
                    ));
                }
                wild = Some(i);
            } else if d < 0 {
                return Err(TensorError::BadArgument(format!(
                    "reshape dimension must be non-negative or -1, found {}",
                    d
                )));
            } else {
                known *= d as usize;
            }
        }
        let mut shape: Vec<usize> = dims
            .iter()
            .map(|&d| if d == -1 { 0 } else { d as usize })
            .collect();
        if let Some(i) = wild {
            if known == 0 || total % known != 0 {
                return Err(TensorError::ReshapeSize { elements: total, shape: shape.clone() });
            }
            shape[i] = total / known;
        }
        let want: usize = shape.iter().product();
        if want != total {
            return Err(TensorError::ReshapeSize { elements: total, shape });
        }
        if shape.len() > MAX_RANK {
            return Err(TensorError::BadRank { rank: shape.len(), max: MAX_RANK });
        }
        // A view is possible only when the source is contiguous.
        if self.is_contiguous() {
            Ok(Tensor {
                buf: Rc::clone(&self.buf),
                offset: self.offset,
                strides: contiguous_strides(&shape),
                shape,
                device: self.device,
                requires_grad: false,
                grad: None,
                node: None,
            })
        } else {
            Tensor::from_vec(self.to_vec(), shape)
        }
    }

    pub fn flatten(&self) -> TensorResult<Tensor> {
        self.reshape(&[self.numel() as isize])
    }

    /// Swap two axes, producing a view (FR-TEN-007).
    pub fn transpose(&self, a: usize, b: usize) -> TensorResult<Tensor> {
        let r = self.rank();
        if a >= r || b >= r {
            return Err(TensorError::BadAxis { axis: a.max(b), rank: r });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(a, b);
        strides.swap(a, b);
        Ok(Tensor {
            buf: Rc::clone(&self.buf),
            offset: self.offset,
            shape,
            strides,
            device: self.device,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    /// Transpose the last two axes — the common matrix case.
    pub fn t(&self) -> TensorResult<Tensor> {
        if self.rank() < 2 {
            return Ok(self.clone());
        }
        self.transpose(self.rank() - 2, self.rank() - 1)
    }

    pub fn permute(&self, order: &[usize]) -> TensorResult<Tensor> {
        let r = self.rank();
        if order.len() != r {
            return Err(TensorError::BadArgument(format!(
                "permute needs {} axes, got {}",
                r,
                order.len()
            )));
        }
        let mut seen = vec![false; r];
        for &a in order {
            if a >= r {
                return Err(TensorError::BadAxis { axis: a, rank: r });
            }
            if seen[a] {
                return Err(TensorError::BadArgument(format!("axis {} repeated in permute", a)));
            }
            seen[a] = true;
        }
        let shape = order.iter().map(|&a| self.shape[a]).collect();
        let strides = order.iter().map(|&a| self.strides[a]).collect();
        Ok(Tensor {
            buf: Rc::clone(&self.buf),
            offset: self.offset,
            shape,
            strides,
            device: self.device,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    pub fn squeeze(&self, axis: Option<usize>) -> TensorResult<Tensor> {
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        match axis {
            Some(a) => {
                if a >= self.rank() {
                    return Err(TensorError::BadAxis { axis: a, rank: self.rank() });
                }
                if self.shape[a] != 1 {
                    return Err(TensorError::BadArgument(format!(
                        "cannot squeeze axis {} of size {}",
                        a, self.shape[a]
                    )));
                }
                for (i, (&d, &s)) in self.shape.iter().zip(&self.strides).enumerate() {
                    if i != a {
                        shape.push(d);
                        strides.push(s);
                    }
                }
            }
            None => {
                for (&d, &s) in self.shape.iter().zip(&self.strides) {
                    if d != 1 {
                        shape.push(d);
                        strides.push(s);
                    }
                }
            }
        }
        Ok(Tensor {
            buf: Rc::clone(&self.buf),
            offset: self.offset,
            shape,
            strides,
            device: self.device,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    pub fn unsqueeze(&self, axis: usize) -> TensorResult<Tensor> {
        if axis > self.rank() {
            return Err(TensorError::BadAxis { axis, rank: self.rank() });
        }
        if self.rank() + 1 > MAX_RANK {
            return Err(TensorError::BadRank { rank: self.rank() + 1, max: MAX_RANK });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let s = if axis < self.strides.len() {
            self.strides[axis] * self.shape[axis]
        } else {
            1
        };
        shape.insert(axis, 1);
        strides.insert(axis, s);
        Ok(Tensor {
            buf: Rc::clone(&self.buf),
            offset: self.offset,
            shape,
            strides,
            device: self.device,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    /// Half-open slice along one axis, producing a view.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> TensorResult<Tensor> {
        if axis >= self.rank() {
            return Err(TensorError::BadAxis { axis, rank: self.rank() });
        }
        if end > self.shape[axis] || start > end {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index: end as isize,
                bound: self.shape[axis],
            });
        }
        let mut shape = self.shape.clone();
        shape[axis] = end - start;
        Ok(Tensor {
            buf: Rc::clone(&self.buf),
            offset: self.offset + start * self.strides[axis],
            shape,
            strides: self.strides.clone(),
            device: self.device,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    /// Select one index along axis 0, dropping that axis.
    pub fn row(&self, i: usize) -> TensorResult<Tensor> {
        self.slice(0, i, i + 1)?.squeeze(Some(0))
    }

    // ---------------- printing (FR-TEN-013) ----------------

    pub fn format(&self) -> String {
        let data = self.to_vec();
        let header = format!("tensor<f32, {}>", 
            self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "));
        if self.rank() == 0 {
            return format!("{} {}", header, fmt_f32(data[0]));
        }
        let body = format_nested(&data, &self.shape, 0, &mut 0);
        format!("{}\n{}", header, body)
    }
}

fn format_nested(data: &[f32], shape: &[usize], depth: usize, cursor: &mut usize) -> String {
    let indent = "  ".repeat(depth);
    if shape.len() == 1 {
        let n = shape[0];
        let mut parts = Vec::new();
        // FR-TEN-013: elide the interior of any axis longer than 8.
        if n > 8 {
            for k in 0..4 {
                parts.push(fmt_f32(data[*cursor + k]));
            }
            parts.push("...".to_string());
            for k in n - 4..n {
                parts.push(fmt_f32(data[*cursor + k]));
            }
        } else {
            for k in 0..n {
                parts.push(fmt_f32(data[*cursor + k]));
            }
        }
        *cursor += n;
        return format!("{}[{}]", indent, parts.join(", "));
    }
    let n = shape[0];
    let inner: usize = shape[1..].iter().product();
    let mut rows = Vec::new();
    if n > 8 {
        for k in 0..4 {
            let mut c = *cursor + k * inner;
            rows.push(format_nested(data, &shape[1..], depth + 1, &mut c));
        }
        rows.push(format!("{}  ...", indent));
        for k in n - 4..n {
            let mut c = *cursor + k * inner;
            rows.push(format_nested(data, &shape[1..], depth + 1, &mut c));
        }
    } else {
        for k in 0..n {
            let mut c = *cursor + k * inner;
            rows.push(format_nested(data, &shape[1..], depth + 1, &mut c));
        }
    }
    *cursor += n * inner;
    format!("{}[\n{}\n{}]", indent, rows.join(",\n"), indent)
}

pub fn fmt_f32(v: f32) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if v == v.trunc() && v.abs() < 1e7 {
        format!("{:.1}", v)
    } else if v.abs() >= 1e-4 && v.abs() < 1e6 {
        format!("{:.4}", v)
    } else {
        format!("{:.4e}", v)
    }
}

// ---------------- free functions ----------------

pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

pub fn bump_index(idx: &mut [usize], shape: &[usize]) {
    for k in (0..idx.len()).rev() {
        idx[k] += 1;
        if idx[k] < shape[k] {
            return;
        }
        idx[k] = 0;
    }
}

/// FR-TYP-007: right-aligned NumPy broadcasting.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for i in 0..r {
        let ad = if i < r - a.len() { 1 } else { a[i - (r - a.len())] };
        let bd = if i < r - b.len() { 1 } else { b[i - (r - b.len())] };
        out[i] = if ad == bd {
            ad
        } else if ad == 1 {
            bd
        } else if bd == 1 {
            ad
        } else {
            return None;
        };
    }
    Some(out)
}

/// The first axis (from the left, in the broadcast frame) where two shapes
/// are incompatible — used by FR-TYP-009 / FR-ERR-006 diagnostics.
pub fn first_bad_axis(a: &[usize], b: &[usize]) -> Option<usize> {
    let r = a.len().max(b.len());
    for i in 0..r {
        let ad = if i < r - a.len() { 1 } else { a[i - (r - a.len())] };
        let bd = if i < r - b.len() { 1 } else { b[i - (r - b.len())] };
        if ad != bd && ad != 1 && bd != 1 {
            return Some(i);
        }
    }
    None
}

thread_local! {
    /// FR-ERR-008: ceiling checked before any allocation is attempted.
    static MEMORY_CEILING: RefCell<usize> = RefCell::new(default_ceiling());
}

fn default_ceiling() -> usize {
    // 75 % of MemAvailable, per FR-ERR-008.
    let avail = read_mem_available().unwrap_or(2 * 1024 * 1024 * 1024);
    (avail as f64 * 0.75) as usize
}

fn read_mem_available() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

pub fn set_memory_ceiling(bytes: usize) {
    MEMORY_CEILING.with(|c| *c.borrow_mut() = bytes);
}
pub fn memory_ceiling() -> usize {
    MEMORY_CEILING.with(|c| *c.borrow())
}

/// FR-ERR-008: validate an allocation *before* attempting it.
pub fn check_alloc(shape: &[usize]) -> TensorResult<usize> {
    if shape.len() > MAX_RANK {
        return Err(TensorError::BadRank { rank: shape.len(), max: MAX_RANK });
    }
    let mut n: usize = 1;
    for &d in shape {
        n = match n.checked_mul(d) {
            Some(v) => v,
            None => {
                return Err(TensorError::AllocationCeiling {
                    shape: shape.to_vec(),
                    bytes: usize::MAX,
                    ceiling: memory_ceiling(),
                })
            }
        };
    }
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let ceiling = memory_ceiling();
    if bytes > ceiling {
        return Err(TensorError::AllocationCeiling { shape: shape.to_vec(), bytes, ceiling });
    }
    Ok(n)
}

/// FR-TEN-010: worker count, settable by `KLIONS_THREADS`.
pub fn thread_count() -> usize {
    if let Ok(v) = std::env::var("KLIONS_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
