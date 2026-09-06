//! Model serialization — FR-AI-017.
//!
//! A documented, versioned binary format with a magic number and a shape
//! manifest, so a mismatched file fails loudly instead of loading garbage.
//!
//! ```text
//!   offset  size  field
//!   0       6     magic "KLIONM"
//!   6       2     format version (u16, little-endian)
//!   8       4     parameter count N (u32)
//!   then, per parameter:
//!        4        name length L (u32)
//!        L        name bytes (UTF-8)
//!        1        rank R (u8)
//!        4*R      dimensions (u32 each)
//!        4*prod   f32 elements, little-endian
//! ```

use std::path::Path;

use crate::Model;
use klions_tensor::Tensor;

pub const MAGIC: &[u8; 6] = b"KLIONM";
pub const FORMAT_VERSION: u16 = 1;

#[derive(Debug)]
pub enum ModelIoError {
    Io(String),
    BadMagic,
    UnsupportedVersion(u16),
    Manifest(String),
}

impl ModelIoError {
    pub fn code(&self) -> &'static str {
        use klions_diagnostics::codes;
        match self {
            ModelIoError::Io(_) => codes::IO_ERROR,
            ModelIoError::BadMagic => codes::MODEL_FILE_BAD_MAGIC,
            ModelIoError::UnsupportedVersion(_) => codes::MODEL_FILE_VERSION,
            ModelIoError::Manifest(_) => codes::MODEL_SHAPE_MANIFEST,
        }
    }
}

impl std::fmt::Display for ModelIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelIoError::Io(m) => write!(f, "{}", m),
            ModelIoError::BadMagic => {
                write!(f, "not a KLIONS model file (magic number does not match)")
            }
            ModelIoError::UnsupportedVersion(v) => write!(
                f,
                "model file format version {} is not supported by this toolchain (expected {})",
                v, FORMAT_VERSION
            ),
            ModelIoError::Manifest(m) => write!(f, "model shape manifest mismatch: {}", m),
        }
    }
}

pub type ModelIoResult<T> = Result<T, ModelIoError>;

pub fn save(model: &Model, path: &Path) -> ModelIoResult<()> {
    let params = model.params();
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(params.len() as u32).to_le_bytes());

    for p in &params {
        let name = p.name.as_bytes();
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name);
        let t = p.value.borrow();
        out.push(t.rank() as u8);
        for d in &t.shape {
            out.extend_from_slice(&(*d as u32).to_le_bytes());
        }
        for v in t.to_vec() {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &out)
        .map_err(|e| ModelIoError::Io(format!("{}: {}", path.display(), e)))
}

pub fn load_into(model: &Model, path: &Path) -> ModelIoResult<()> {
    let bytes = std::fs::read(path)
        .map_err(|e| ModelIoError::Io(format!("{}: {}", path.display(), e)))?;
    if bytes.len() < 12 || &bytes[..6] != MAGIC {
        return Err(ModelIoError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[6], bytes[7]]);
    if version != FORMAT_VERSION {
        return Err(ModelIoError::UnsupportedVersion(version));
    }
    let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let params = model.params();
    if count != params.len() {
        return Err(ModelIoError::Manifest(format!(
            "the file holds {} parameter tensors, this model has {}",
            count,
            params.len()
        )));
    }

    let mut o = 12usize;
    for p in &params {
        let need = |o: usize, n: usize, len: usize| -> ModelIoResult<()> {
            if o + n > len {
                Err(ModelIoError::Manifest("file ends mid-parameter".into()))
            } else {
                Ok(())
            }
        };
        need(o, 4, bytes.len())?;
        let nl = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        o += 4;
        need(o, nl, bytes.len())?;
        let name = String::from_utf8_lossy(&bytes[o..o + nl]).to_string();
        o += nl;
        if name != p.name {
            return Err(ModelIoError::Manifest(format!(
                "expected parameter `{}`, the file holds `{}`",
                p.name, name
            )));
        }
        need(o, 1, bytes.len())?;
        let rank = bytes[o] as usize;
        o += 1;
        need(o, 4 * rank, bytes.len())?;
        let mut shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let b = o + 4 * i;
            shape.push(u32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
                as usize);
        }
        o += 4 * rank;
        let want = p.shape();
        if shape != want {
            return Err(ModelIoError::Manifest(format!(
                "parameter `{}` has shape {:?} in the file but {:?} in the model",
                name, shape, want
            )));
        }
        let n: usize = shape.iter().product();
        need(o, 4 * n, bytes.len())?;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let b = o + 4 * i;
            data.push(f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]));
        }
        o += 4 * n;
        let t = Tensor::from_vec(data, shape)
            .map_err(|e| ModelIoError::Manifest(e.to_string()))?;
        *p.value.borrow_mut() = t;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, Layer, Model};
    use klions_tensor::Rng;

    fn build(seed: u64) -> Model {
        let mut rng = Rng::new(seed);
        let mut m = Model::new("Net");
        m.push("h", Layer::dense("h", 4, 5, Activation::ReLU, &mut rng).unwrap());
        m.push("o", Layer::dense("o", 5, 2, Activation::None, &mut rng).unwrap());
        m
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("klions_model_{}_{}.klm", std::process::id(), name));
        p
    }

    #[test]
    fn save_then_load_restores_parameters() {
        let a = build(1);
        let b = build(2); // different weights
        let p = tmp("rt");
        save(&a, &p).unwrap();
        load_into(&b, &p).unwrap();
        for (pa, pb) in a.params().iter().zip(b.params()) {
            assert_eq!(pa.value.borrow().to_vec(), pb.value.borrow().to_vec());
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn bad_magic_is_rejected() {
        let p = tmp("bad");
        std::fs::write(&p, b"NOTAMODELFILE___").unwrap();
        assert!(matches!(load_into(&build(1), &p), Err(ModelIoError::BadMagic)));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn shape_manifest_mismatch_is_rejected() {
        let a = build(1);
        let p = tmp("shape");
        save(&a, &p).unwrap();

        let mut rng = Rng::new(9);
        let mut other = Model::new("Other");
        other.push("h", Layer::dense("h", 7, 5, Activation::ReLU, &mut rng).unwrap());
        other.push("o", Layer::dense("o", 5, 2, Activation::None, &mut rng).unwrap());
        assert!(matches!(load_into(&other, &p), Err(ModelIoError::Manifest(_))));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn version_byte_is_checked() {
        let a = build(1);
        let p = tmp("ver");
        save(&a, &p).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[6] = 99;
        std::fs::write(&p, &bytes).unwrap();
        assert!(matches!(
            load_into(&build(1), &p),
            Err(ModelIoError::UnsupportedVersion(_))
        ));
        std::fs::remove_file(p).ok();
    }
}
