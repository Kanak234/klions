//! Tensor-layer errors. These map onto the K1xxx `ShapeError` and K5xxx
//! `DeviceError` ranges (FR-ERR-005, FR-ERR-006).

use klions_diagnostics::codes;

pub type TensorResult<T> = Result<T, TensorError>;

#[derive(Clone, Debug)]
pub enum TensorError {
    /// FR-ERR-006: operation, both shapes, and the disagreeing axis.
    ShapeMismatch {
        op: String,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        axis: Option<usize>,
    },
    MatMulShape {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    ReshapeSize {
        elements: usize,
        shape: Vec<usize>,
    },
    BadRank {
        rank: usize,
        max: usize,
    },
    BadAxis {
        axis: usize,
        rank: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: isize,
        bound: usize,
    },
    AllocationCeiling {
        shape: Vec<usize>,
        bytes: usize,
        ceiling: usize,
    },
    UnsupportedDevice {
        requested: String,
        release: &'static str,
    },
    BadArgument(String),
}

impl TensorError {
    pub fn code(&self) -> &'static str {
        match self {
            TensorError::ShapeMismatch { .. } => codes::SHAPE_MISMATCH,
            TensorError::MatMulShape { .. } => codes::MATMUL_SHAPE,
            TensorError::ReshapeSize { .. } => codes::RESHAPE_SIZE,
            TensorError::BadRank { .. } => codes::BAD_RANK,
            TensorError::BadAxis { .. } => codes::BAD_AXIS,
            TensorError::IndexOutOfBounds { .. } => codes::INDEX_ERROR,
            TensorError::AllocationCeiling { .. } => codes::ALLOCATION_CEILING,
            TensorError::UnsupportedDevice { .. } => codes::UNSUPPORTED_DEVICE,
            TensorError::BadArgument(_) => codes::BAD_OPERAND_TYPE,
        }
    }

    pub fn help(&self) -> Option<String> {
        match self {
            TensorError::ShapeMismatch { .. } => Some(
                "element-wise operations need identical shapes, or shapes that broadcast: \
                 dimensions are compared from the right, and a dimension of 1 is stretched"
                    .into(),
            ),
            TensorError::MatMulShape { lhs, rhs } => Some(format!(
                "`@` needs the inner dimensions to agree: [.., m, k] @ [.., k, n]; \
                 here k is {} on the left but {} on the right",
                lhs.get(lhs.len().saturating_sub(1)).copied().unwrap_or(0),
                rhs.get(rhs.len().saturating_sub(2)).copied().unwrap_or(0)
            )),
            TensorError::ReshapeSize { elements, .. } => Some(format!(
                "the new shape must have exactly {} elements; use -1 to infer one dimension",
                elements
            )),
            TensorError::BadRank { max, .. } => {
                Some(format!("v0.1.0 supports tensors of rank 0 through {}", max))
            }
            TensorError::BadAxis { rank, .. } => {
                Some(format!("this tensor has rank {}, so axes run 0..{}", rank, rank))
            }
            TensorError::IndexOutOfBounds { .. } => {
                Some("indices are zero-based and must be less than the axis length".into())
            }
            TensorError::AllocationCeiling { ceiling, .. } => Some(format!(
                "the ceiling is {}; raise it with `klions.toml` or reduce the batch size",
                human_bytes(*ceiling)
            )),
            TensorError::UnsupportedDevice { release, .. } => Some(format!(
                "v0.1.0 executes on the CPU only; GPU devices are scheduled for release {}",
                release
            )),
            TensorError::BadArgument(_) => None,
        }
    }

    /// The `note:` line, where one adds information beyond the message.
    pub fn note(&self) -> Option<String> {
        match self {
            TensorError::ShapeMismatch { lhs, rhs, axis: Some(a), .. } => {
                let r = lhs.len().max(rhs.len());
                let ld = if *a < r - lhs.len() { 1 } else { lhs[a - (r - lhs.len())] };
                let rd = if *a < r - rhs.len() { 1 } else { rhs[a - (r - rhs.len())] };
                Some(format!("axis {} disagrees: {} vs {}", a, ld, rd))
            }
            TensorError::AllocationCeiling { bytes, .. } => {
                Some(format!("the request was {}", human_bytes(*bytes)))
            }
            _ => None,
        }
    }
}

fn shape_str(s: &[usize]) -> String {
    format!("[{}]", s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

pub fn human_bytes(b: usize) -> String {
    const K: f64 = 1024.0;
    let b = b as f64;
    if b < K {
        format!("{} B", b as usize)
    } else if b < K * K {
        format!("{:.1} KiB", b / K)
    } else if b < K * K * K {
        format!("{:.1} MiB", b / (K * K))
    } else {
        format!("{:.2} GiB", b / (K * K * K))
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { op, lhs, rhs, .. } => write!(
                f,
                "shape mismatch in `{}`: left is {}, right is {}",
                op,
                shape_str(lhs),
                shape_str(rhs)
            ),
            TensorError::MatMulShape { lhs, rhs } => write!(
                f,
                "cannot matrix-multiply {} by {}",
                shape_str(lhs),
                shape_str(rhs)
            ),
            TensorError::ReshapeSize { elements, shape } => write!(
                f,
                "cannot reshape {} elements into {}",
                elements,
                shape_str(shape)
            ),
            TensorError::BadRank { rank, max } => {
                write!(f, "tensor rank {} exceeds the maximum of {}", rank, max)
            }
            TensorError::BadAxis { axis, rank } => {
                write!(f, "axis {} is out of range for a rank-{} tensor", axis, rank)
            }
            TensorError::IndexOutOfBounds { axis, index, bound } => write!(
                f,
                "index {} is out of bounds on axis {} (length {})",
                index, axis, bound
            ),
            TensorError::AllocationCeiling { shape, bytes, .. } => write!(
                f,
                "allocating a tensor of shape {} would need {}, exceeding the memory ceiling",
                shape_str(shape),
                human_bytes(*bytes)
            ),
            TensorError::UnsupportedDevice { requested, .. } => {
                write!(f, "device `{}` is not supported in v0.1.0", requested)
            }
            TensorError::BadArgument(m) => write!(f, "{}", m),
        }
    }
}
