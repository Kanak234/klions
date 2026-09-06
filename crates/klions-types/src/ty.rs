//! The KLIONS type lattice — FR-TYP-001, FR-TYP-005 … FR-TYP-008.

use klions_ast::Dim;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    I32,
    I64,
    F32,
    F64,
    Bool,
    Str,
    Void,
    /// `tensor<T, D...>`. Dimensions participate in compile-time checking.
    Tensor {
        elem: Box<Ty>,
        dims: Vec<Dim>,
    },
    Array(Box<Ty>),
    Function {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },
    Result(Box<Ty>, Box<Ty>),
    Model,
    Dataset,
    Optimizer,
    Layer,
    /// The `EvalResult` returned by `model.evaluate(...)`.
    Metrics,
    /// A named error type, e.g. `TrainingError`.
    ErrorType(String),
    /// Unresolved — suppresses cascading diagnostics.
    Unknown,
}

impl Ty {
    pub fn tensor(dims: Vec<Dim>) -> Ty {
        Ty::Tensor { elem: Box::new(Ty::F32), dims }
    }
    pub fn tensor_fixed(dims: &[usize]) -> Ty {
        Ty::tensor(dims.iter().map(|&d| Dim::Fixed(d)).collect())
    }
    /// A tensor whose rank is known but whose extents are not.
    pub fn tensor_wild(rank: usize) -> Ty {
        Ty::tensor(vec![Dim::Wild; rank])
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Ty::I32 | Ty::I64 | Ty::F32 | Ty::F64)
    }
    pub fn is_int(&self) -> bool {
        matches!(self, Ty::I32 | Ty::I64)
    }
    pub fn is_float(&self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }
    pub fn is_tensor(&self) -> bool {
        matches!(self, Ty::Tensor { .. })
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, Ty::Unknown)
    }
    pub fn dims(&self) -> Option<&[Dim]> {
        match self {
            Ty::Tensor { dims, .. } => Some(dims),
            _ => None,
        }
    }
    pub fn rank(&self) -> Option<usize> {
        self.dims().map(|d| d.len())
    }

    /// Assignability. `Unknown` unifies with anything so one error does not
    /// cascade into a dozen more.
    pub fn accepts(&self, other: &Ty) -> bool {
        if self.is_unknown() || other.is_unknown() {
            return true;
        }
        match (self, other) {
            (Ty::Tensor { elem: e1, dims: d1 }, Ty::Tensor { elem: e2, dims: d2 }) => {
                if e1 != e2 {
                    return false;
                }
                if d1.is_empty() || d2.is_empty() {
                    return true; // rank not pinned down
                }
                if d1.len() != d2.len() {
                    return false;
                }
                d1.iter().zip(d2).all(|(a, b)| dims_compatible(*a, *b))
            }
            (Ty::Result(a1, e1), Ty::Result(a2, e2)) => a1.accepts(a2) && e1.accepts(e2),
            (Ty::Array(a), Ty::Array(b)) => a.accepts(b),
            (Ty::ErrorType(_), Ty::ErrorType(_)) => true,
            _ => self == other,
        }
    }

    pub fn display(&self) -> String {
        match self {
            Ty::I32 => "i32".into(),
            Ty::I64 => "i64".into(),
            Ty::F32 => "f32".into(),
            Ty::F64 => "f64".into(),
            Ty::Bool => "bool".into(),
            Ty::Str => "string".into(),
            Ty::Void => "void".into(),
            Ty::Tensor { elem, dims } => {
                if dims.is_empty() {
                    format!("tensor<{}>", elem.display())
                } else {
                    format!(
                        "tensor<{}, {}>",
                        elem.display(),
                        dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
                    )
                }
            }
            Ty::Array(t) => format!("[{}]", t.display()),
            Ty::Function { params, ret } => format!(
                "fn({}) -> {}",
                params.iter().map(|p| p.display()).collect::<Vec<_>>().join(", "),
                ret.display()
            ),
            Ty::Result(a, b) => format!("Result<{}, {}>", a.display(), b.display()),
            Ty::Model => "Model".into(),
            Ty::Dataset => "Dataset".into(),
            Ty::Optimizer => "Optimizer".into(),
            Ty::Layer => "Layer".into(),
            Ty::Metrics => "Metrics".into(),
            Ty::ErrorType(n) => n.clone(),
            Ty::Unknown => "?".into(),
        }
    }

    /// The shape as a printable list, for FR-TYP-009 diagnostics.
    pub fn shape_str(&self) -> String {
        match self.dims() {
            Some(d) => format!(
                "[{}]",
                d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
            ),
            None => self.display(),
        }
    }
}

/// Two dimensions agree if they are equal or either is unknown.
pub fn dims_compatible(a: Dim, b: Dim) -> bool {
    match (a, b) {
        (Dim::Wild, _) | (_, Dim::Wild) => true,
        (Dim::Fixed(x), Dim::Fixed(y)) => x == y,
    }
}

/// One dimension broadcasts against another if they agree or either is 1.
pub fn dims_broadcast(a: Dim, b: Dim) -> bool {
    match (a, b) {
        (Dim::Wild, _) | (_, Dim::Wild) => true,
        (Dim::Fixed(x), Dim::Fixed(y)) => x == y || x == 1 || y == 1,
    }
}

fn merge(a: Dim, b: Dim) -> Dim {
    match (a, b) {
        (Dim::Fixed(1), other) | (other, Dim::Fixed(1)) => other,
        (Dim::Fixed(x), Dim::Fixed(_)) => Dim::Fixed(x),
        _ => Dim::Wild,
    }
}

/// FR-TYP-007: right-aligned NumPy broadcasting over static dimensions.
/// Returns the result shape, or the first disagreeing axis.
pub fn broadcast(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>, usize> {
    if a.is_empty() {
        return Ok(b.to_vec());
    }
    if b.is_empty() {
        return Ok(a.to_vec());
    }
    let r = a.len().max(b.len());
    let mut out = Vec::with_capacity(r);
    for i in 0..r {
        let ad = if i < r - a.len() { Dim::Fixed(1) } else { a[i - (r - a.len())] };
        let bd = if i < r - b.len() { Dim::Fixed(1) } else { b[i - (r - b.len())] };
        if !dims_broadcast(ad, bd) {
            return Err(i);
        }
        out.push(merge(ad, bd));
    }
    Ok(out)
}

/// FR-TYP-006: `A: [.., m, k] @ B: [.., k, n]` yields `[.., m, n]`.
#[derive(Debug)]
pub enum MatMulError {
    RankTooLow { side: &'static str, rank: usize },
    Inner { k_lhs: Dim, k_rhs: Dim },
    Batch { axis: usize },
}

pub fn matmul_shape(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>, MatMulError> {
    if a.len() < 2 {
        return Err(MatMulError::RankTooLow { side: "left", rank: a.len() });
    }
    if b.len() < 2 {
        return Err(MatMulError::RankTooLow { side: "right", rank: b.len() });
    }
    let (m, k1) = (a[a.len() - 2], a[a.len() - 1]);
    let (k2, n) = (b[b.len() - 2], b[b.len() - 1]);
    if !dims_compatible(k1, k2) {
        return Err(MatMulError::Inner { k_lhs: k1, k_rhs: k2 });
    }
    // Leading batch axes broadcast.
    let abatch = &a[..a.len() - 2];
    let bbatch = &b[..b.len() - 2];
    let batch = broadcast(abatch, bbatch).map_err(|axis| MatMulError::Batch { axis })?;
    let mut out = batch;
    out.push(m);
    out.push(n);
    Ok(out)
}

/// Convert an AST type annotation into a `Ty`.
pub fn from_ast(t: &klions_ast::TypeExpr) -> Ty {
    use klions_ast::TypeExprKind as K;
    match &t.kind {
        K::Auto => Ty::Unknown,
        K::Named(n) => named(n),
        K::Tensor { elem, dims } => Ty::Tensor {
            elem: Box::new(named(elem)),
            dims: dims.clone(),
        },
        K::Result(a, b) => Ty::Result(Box::new(from_ast(a)), Box::new(from_ast(b))),
        K::Function(ps, r) => Ty::Function {
            params: ps.iter().map(from_ast).collect(),
            ret: Box::new(from_ast(r)),
        },
    }
}

pub fn named(n: &str) -> Ty {
    match n {
        "i32" => Ty::I32,
        "i64" => Ty::I64,
        "f32" => Ty::F32,
        "f64" => Ty::F64,
        "bool" => Ty::Bool,
        "string" => Ty::Str,
        "void" => Ty::Void,
        "auto" => Ty::Unknown,
        "Model" => Ty::Model,
        "Dataset" => Ty::Dataset,
        "Optimizer" => Ty::Optimizer,
        "Layer" => Ty::Layer,
        "Metrics" => Ty::Metrics,
        "tensor" => Ty::Tensor { elem: Box::new(Ty::F32), dims: vec![] },
        other if other.ends_with("Error") => Ty::ErrorType(other.to_string()),
        _ => Ty::Unknown,
    }
}

/// Error types the standard library can produce.
pub const ERROR_TYPES: &[&str] = &[
    "TrainingError",
    "DatasetError",
    "ShapeError",
    "ModelError",
    "DeviceError",
    "Error",
];

#[cfg(test)]
mod tests {
    use super::*;
    use klions_ast::Dim::*;

    #[test]
    fn broadcast_right_aligns() {
        assert_eq!(
            broadcast(&[Fixed(2), Fixed(3)], &[Fixed(3)]).unwrap(),
            vec![Fixed(2), Fixed(3)]
        );
        assert_eq!(
            broadcast(&[Fixed(4), Fixed(1)], &[Fixed(3)]).unwrap(),
            vec![Fixed(4), Fixed(3)]
        );
    }

    #[test]
    fn broadcast_reports_the_bad_axis() {
        assert_eq!(broadcast(&[Fixed(2), Fixed(3)], &[Fixed(2), Fixed(4)]), Err(1));
    }

    #[test]
    fn wildcards_defer_to_run_time() {
        assert!(broadcast(&[Wild, Fixed(3)], &[Fixed(2), Fixed(3)]).is_ok());
        assert!(dims_compatible(Wild, Fixed(99)));
    }

    #[test]
    fn matmul_contracts_the_inner_dimension() {
        let out = matmul_shape(&[Fixed(64), Fixed(784)], &[Fixed(784), Fixed(128)]).unwrap();
        assert_eq!(out, vec![Fixed(64), Fixed(128)]);
    }

    #[test]
    fn matmul_rejects_mismatched_inner_dimension() {
        assert!(matches!(
            matmul_shape(&[Fixed(64), Fixed(784)], &[Fixed(128), Fixed(10)]),
            Err(MatMulError::Inner { .. })
        ));
    }

    #[test]
    fn batched_matmul_keeps_the_batch_axis() {
        let out =
            matmul_shape(&[Fixed(8), Fixed(4), Fixed(5)], &[Fixed(8), Fixed(5), Fixed(2)]).unwrap();
        assert_eq!(out, vec![Fixed(8), Fixed(4), Fixed(2)]);
    }

    #[test]
    fn tensor_types_display_readably() {
        assert_eq!(
            Ty::tensor_fixed(&[64, 784]).display(),
            "tensor<f32, 64, 784>"
        );
        assert_eq!(Ty::tensor(vec![Wild, Fixed(10)]).display(), "tensor<f32, _, 10>");
    }

    #[test]
    fn unknown_absorbs_everything() {
        assert!(Ty::Unknown.accepts(&Ty::I32));
        assert!(Ty::I32.accepts(&Ty::Unknown));
        assert!(!Ty::I32.accepts(&Ty::F32));
    }
}
