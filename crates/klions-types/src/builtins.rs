//! Signatures for the built-in surface: free functions, tensor and model
//! methods, associated constructors, and layer parameters.
//!
//! Keeping these in one table means the checker, the LSP's completion list,
//! and the runtime's dispatch all agree about what exists.

use crate::ty::Ty;
use klions_ast::Dim;
use klions_diagnostics::Span;

pub const ACTIVATIONS: &[&str] = &["none", "relu", "sigmoid", "tanh", "softmax"];
pub const LOSSES: &[&str] = &["mse", "mae", "cross_entropy", "binary_cross_entropy"];
pub const OPTIMIZERS: &[&str] = &["SGD", "Adam", "RMSProp"];
pub const METRICS: &[&str] = &["loss", "accuracy"];

/// Free functions callable without a receiver.
pub const BUILTIN_NAMES: &[&str] = &[
    "print", "println", "format", "len", "shape", "rank", "argmax", "argmin", "sum", "mean",
    "max", "min", "abs", "sqrt", "exp", "ln", "pow", "floor", "ceil", "round", "zeros", "ones",
    "full", "eye", "arange", "rand", "randn", "tensor", "one_hot", "time_ms", "assert",
    "int", "float", "str",
];

/// How deeply an array type nests: `[[f64]]` is 2. Used to give a tensor
/// built from an array literal the right rank at compile time.
pub fn nesting_depth(t: &Ty) -> usize {
    match t {
        Ty::Array(inner) => 1 + nesting_depth(inner),
        _ => 0,
    }
}

pub struct LayerSpec {
    pub name: &'static str,
    /// (parameter name, required)
    pub params: &'static [(&'static str, bool)],
}

pub fn layer_spec(kind: &str) -> Option<LayerSpec> {
    let params: &'static [(&'static str, bool)] = match kind {
        "Dense" => &[("inputs", true), ("outputs", true), ("activation", false)],
        "Dropout" => &[("p", true)],
        "ReLU" | "Sigmoid" | "Tanh" | "Softmax" | "Flatten" => &[],
        _ => return None,
    };
    Some(LayerSpec { name: Box::leak(kind.to_string().into_boxed_str()), params })
}

pub struct TrainOptionSpec {
    pub name: &'static str,
    pub ty: Ty,
    pub help: &'static str,
}

pub fn train_option_spec(key: &str) -> Option<TrainOptionSpec> {
    Some(match key {
        "epochs" => TrainOptionSpec {
            name: "epochs",
            ty: Ty::I64,
            help: "the number of passes over the dataset, e.g. `epochs = 10`",
        },
        "batch_size" => TrainOptionSpec {
            name: "batch_size",
            ty: Ty::I64,
            help: "samples per gradient step, e.g. `batch_size = 32`",
        },
        "optimizer" => TrainOptionSpec {
            name: "optimizer",
            ty: Ty::Optimizer,
            help: "an optimizer value, e.g. `optimizer = Adam(learning_rate=0.001)`",
        },
        "loss" => TrainOptionSpec {
            name: "loss",
            ty: Ty::Str,
            help: "a loss name, e.g. `loss = \"cross_entropy\"`",
        },
        "metrics" => TrainOptionSpec {
            name: "metrics",
            ty: Ty::Array(Box::new(Ty::Str)),
            help: "a list of metric names, e.g. `metrics = [\"accuracy\"]`",
        },
        "validation_split" => TrainOptionSpec {
            name: "validation_split",
            ty: Ty::F64,
            help: "the held-out fraction, e.g. `validation_split = 0.1`",
        },
        "verbose" => TrainOptionSpec {
            name: "verbose",
            ty: Ty::Bool,
            help: "whether to print per-epoch progress, e.g. `verbose = true`",
        },
        _ => return None,
    })
}

/// Type of a bare identifier that names a builtin (rare; mostly optimizers).
pub fn builtin_ty(name: &str) -> Option<Ty> {
    match name {
        "Adam" | "SGD" | "RMSProp" => Some(Ty::Optimizer),
        _ => None,
    }
}

pub fn builtin_call_ty(name: &str, args: &[(Ty, Span)]) -> Option<Ty> {
    let a0 = args.first().map(|(t, _)| t.clone());
    Some(match name {
        "print" | "println" => Ty::Void,
        "format" | "str" => Ty::Str,
        "assert" => Ty::Void,
        "len" | "rank" => Ty::I64,
        "time_ms" => Ty::I64,
        "int" => Ty::I64,
        "float" => Ty::F64,
        "shape" => Ty::Array(Box::new(Ty::I64)),
        "argmax" | "argmin" => Ty::I64,
        // Reductions collapse a tensor to a scalar.
        "sum" | "mean" | "max" | "min" => match a0 {
            Some(Ty::Tensor { .. }) => Ty::F32,
            Some(t) if t.is_numeric() => t,
            _ => Ty::F32,
        },
        // Element-wise maths preserve shape.
        "abs" | "sqrt" | "exp" | "ln" | "floor" | "ceil" | "round" => match a0 {
            Some(t @ Ty::Tensor { .. }) => t,
            Some(t) if t.is_numeric() => t,
            _ => Ty::F64,
        },
        "pow" => match a0 {
            Some(t @ Ty::Tensor { .. }) => t,
            _ => Ty::F64,
        },
        // `tensor([[1, 2], [3, 4]])` takes its rank from how deeply the array
        // literal nests, not from the argument count. Getting this wrong made
        // the checker disagree with the interpreter about `@`.
        "tensor" => match args.first().map(|(t, _)| t) {
            Some(t @ Ty::Array(_)) => Ty::tensor(vec![Dim::Wild; nesting_depth(t)]),
            _ => Ty::tensor(vec![Dim::Wild; args.len().max(1)]),
        },
        // `zeros(2, 3)` pins the rank; `zeros([2, 3])` cannot, because the
        // length of that list is a run-time value.
        "zeros" | "ones" | "rand" | "randn" | "full" => {
            match args.first().map(|(t, _)| t) {
                Some(Ty::Array(_)) if args.len() == 1 => {
                    Ty::Tensor { elem: Box::new(Ty::F32), dims: vec![] }
                }
                _ => Ty::tensor(vec![Dim::Wild; args.len().max(1)]),
            }
        }
        "eye" => Ty::tensor(vec![Dim::Wild, Dim::Wild]),
        "arange" => Ty::tensor(vec![Dim::Wild]),
        "one_hot" => Ty::tensor(vec![Dim::Wild, Dim::Wild]),
        "Adam" | "SGD" | "RMSProp" => Ty::Optimizer,
        _ => return None,
    })
}

pub enum MethodLookup {
    Found(Ty),
    ArityMismatch { want: usize, got: usize },
    NotFound,
}

/// Methods on tensors, models, datasets, and metrics.
pub fn method_ty(recv: &Ty, name: &str, args: &[(Ty, Span)]) -> MethodLookup {
    let n = args.len();
    let found = |t: Ty| MethodLookup::Found(t);
    let arity = |want: usize| MethodLookup::ArityMismatch { want, got: n };

    match recv {
        Ty::Unknown => MethodLookup::Found(Ty::Unknown),

        Ty::Tensor { elem, dims } => match name {
            "shape" => {
                if n > 0 { arity(0) } else { found(Ty::Array(Box::new(Ty::I64))) }
            }
            "rank" | "len" | "numel" => {
                if n > 0 { arity(0) } else { found(Ty::I64) }
            }
            "item" => {
                if n > 0 { arity(0) } else { found((**elem).clone()) }
            }
            "sum" | "mean" | "max" | "min" => {
                if n == 0 {
                    found(Ty::F32)
                } else {
                    // With an axis argument the rank drops by one.
                    found(match dims.len() {
                        0 => Ty::Tensor { elem: elem.clone(), dims: vec![] },
                        r => Ty::tensor(vec![Dim::Wild; r - 1]),
                    })
                }
            }
            "argmax" | "argmin" => {
                if n == 0 {
                    found(Ty::I64)
                } else {
                    found(Ty::tensor(vec![Dim::Wild; dims.len().saturating_sub(1)]))
                }
            }
            "reshape" => {
                if n == 0 {
                    arity(1)
                } else {
                    found(Ty::tensor(vec![Dim::Wild; n]))
                }
            }
            "flatten" => {
                if n > 0 { arity(0) } else { found(Ty::tensor(vec![Dim::Wild])) }
            }
            "t" | "transpose" => {
                let mut d = dims.clone();
                if d.len() >= 2 {
                    let n = d.len();
                    d.swap(n - 2, n - 1);
                }
                found(Ty::Tensor { elem: elem.clone(), dims: d })
            }
            "relu" | "sigmoid" | "tanh" | "softmax" | "exp" | "ln" | "sqrt" | "abs" => {
                found(recv.clone())
            }
            "to_vec" => found(Ty::Array(Box::new(Ty::F32))),
            "clone" => found(recv.clone()),
            _ => MethodLookup::NotFound,
        },

        Ty::Model => match name {
            "predict" => {
                if n != 1 {
                    arity(1)
                } else {
                    found(Ty::tensor(vec![Dim::Wild, Dim::Wild]))
                }
            }
            "evaluate" => {
                if n != 1 {
                    arity(1)
                } else {
                    found(Ty::Metrics)
                }
            }
            "save" => {
                if n != 1 {
                    arity(1)
                } else {
                    found(Ty::Result(
                        Box::new(Ty::Void),
                        Box::new(Ty::ErrorType("ModelError".into())),
                    ))
                }
            }
            "load" => {
                if n != 1 {
                    arity(1)
                } else {
                    found(Ty::Result(
                        Box::new(Ty::Void),
                        Box::new(Ty::ErrorType("ModelError".into())),
                    ))
                }
            }
            "summary" => found(Ty::Str),
            "parameter_count" => found(Ty::I64),
            _ => MethodLookup::NotFound,
        },

        Ty::Dataset => match name {
            "batch" | "shuffle" | "normalize" | "standardize" | "one_hot" => found(Ty::Dataset),
            "split" => found(Ty::Dataset),
            "len" | "size" => found(Ty::I64),
            "features" | "labels" => found(Ty::tensor(vec![Dim::Wild, Dim::Wild])),
            "feature_width" | "label_width" => found(Ty::I64),
            "sample" => {
                if n != 1 {
                    arity(1)
                } else {
                    found(Ty::tensor(vec![Dim::Wild, Dim::Wild]))
                }
            }
            "label" => {
                if n != 1 {
                    arity(1)
                } else {
                    found(Ty::F32)
                }
            }
            _ => MethodLookup::NotFound,
        },

        Ty::Metrics => match name {
            "loss" | "accuracy" => found(Ty::F32),
            "samples" => found(Ty::I64),
            _ => MethodLookup::NotFound,
        },

        Ty::Optimizer => match name {
            "learning_rate" => found(Ty::F32),
            "name" => found(Ty::Str),
            _ => MethodLookup::NotFound,
        },

        Ty::Str => match name {
            "len" => found(Ty::I64),
            _ => MethodLookup::NotFound,
        },

        Ty::Array(inner) => match name {
            "len" => found(Ty::I64),
            "get" => found((**inner).clone()),
            _ => MethodLookup::NotFound,
        },

        Ty::Result(ok, _) => match name {
            "unwrap" => found((**ok).clone()),
            "is_ok" | "is_err" => found(Ty::Bool),
            _ => MethodLookup::NotFound,
        },

        _ => MethodLookup::NotFound,
    }
}

/// Field access (no call parentheses).
pub fn field_ty(recv: &Ty, name: &str) -> Option<Ty> {
    match recv {
        Ty::Unknown => Some(Ty::Unknown),
        Ty::Metrics => match name {
            "loss" | "accuracy" => Some(Ty::F32),
            "samples" => Some(Ty::I64),
            _ => None,
        },
        // Method references without a call resolve to their result type so
        // that `x.shape` reads naturally alongside `x.shape()`.
        _ => match method_ty(recv, name, &[]) {
            MethodLookup::Found(t) => Some(t),
            _ => None,
        },
    }
}

pub fn members_of(recv: &Ty) -> Vec<&'static str> {
    match recv {
        Ty::Tensor { .. } => vec![
            "shape", "rank", "len", "numel", "item", "sum", "mean", "max", "min", "argmax",
            "argmin", "reshape", "flatten", "t", "transpose", "relu", "sigmoid", "tanh",
            "softmax", "exp", "ln", "sqrt", "abs", "to_vec", "clone",
        ],
        Ty::Model => vec![
            "predict",
            "evaluate",
            "save",
            "load",
            "summary",
            "parameter_count",
        ],
        Ty::Dataset => vec![
            "batch", "shuffle", "normalize", "standardize", "one_hot", "split", "len", "size",
            "features", "labels", "feature_width", "label_width", "sample", "label",
        ],
        Ty::Metrics => vec!["loss", "accuracy", "samples"],
        Ty::Optimizer => vec!["learning_rate", "name"],
        Ty::Result(_, _) => vec!["unwrap", "is_ok", "is_err"],
        Ty::Array(_) => vec!["len", "get"],
        Ty::Str => vec!["len"],
        _ => vec![],
    }
}

pub fn method_signature(recv: &Ty, name: &str) -> String {
    match (recv, name) {
        (Ty::Model, "predict") => "signature: model.predict(input: tensor) -> tensor".into(),
        (Ty::Model, "evaluate") => "signature: model.evaluate(data: Dataset) -> Metrics".into(),
        (Ty::Model, "save") => {
            "signature: model.save(path: string) -> Result<void, ModelError>".into()
        }
        (Ty::Model, "load") => {
            "signature: model.load(path: string) -> Result<void, ModelError>".into()
        }
        (Ty::Dataset, "sample") => "signature: data.sample(index: i64) -> tensor".into(),
        (Ty::Dataset, "label") => "signature: data.label(index: i64) -> f32".into(),
        (Ty::Tensor { .. }, "reshape") => {
            "signature: t.reshape(dims...) -> tensor; use -1 to infer one dimension".into()
        }
        _ => format!("check the reference for `{}`", name),
    }
}

/// `Type::function(...)` constructors.
pub fn path_call_ty(base: &str, name: &str, args: &[(Ty, Span)]) -> Option<Ty> {
    let _ = args;
    Some(match (base, name) {
        ("Dataset", "from_idx") => Ty::Result(
            Box::new(Ty::Dataset),
            Box::new(Ty::ErrorType("DatasetError".into())),
        ),
        ("Dataset", "from_csv") => Ty::Result(
            Box::new(Ty::Dataset),
            Box::new(Ty::ErrorType("DatasetError".into())),
        ),
        ("Tensor", "zeros") | ("Tensor", "ones") | ("Tensor", "rand") | ("Tensor", "randn") => {
            Ty::tensor(vec![Dim::Wild; args.len().max(1)])
        }
        ("Tensor", "eye") => Ty::tensor(vec![Dim::Wild, Dim::Wild]),
        ("Tensor", "from_array") => match args.first().map(|(t, _)| t) {
            Some(t @ Ty::Array(_)) => Ty::tensor(vec![Dim::Wild; nesting_depth(t)]),
            _ => Ty::Tensor { elem: Box::new(Ty::F32), dims: vec![] },
        },
        ("Optimizer", "adam") | ("Optimizer", "sgd") | ("Optimizer", "rmsprop") => Ty::Optimizer,
        _ => return None,
    })
}

pub fn path_ty(base: &str, name: &str) -> Option<Ty> {
    path_call_ty(base, name, &[])
}

pub fn path_members(base: &str) -> Vec<&'static str> {
    match base {
        "Dataset" => vec!["from_idx", "from_csv"],
        "Tensor" => vec!["zeros", "ones", "rand", "randn", "eye", "from_array"],
        "Optimizer" => vec!["adam", "sgd", "rmsprop"],
        _ => vec![],
    }
}
