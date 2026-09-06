//! The built-in surface at run time — FR-RT-008, FR-AI-016, EIR-001 … EIR-006.
//!
//! Mirrors the signature table in `klions-types::builtins`. The checker and
//! the interpreter are separate code, so each entry here is exercised by a
//! test in `tests/programs` to keep them from drifting apart.

use std::rc::Rc;

use klions_autodiff as ad;
use klions_diagnostics::{codes, Span};
use klions_nn::data::Dataset;
use klions_nn::loss;
use klions_tensor::kernels as k;
use klions_tensor::Tensor;

use crate::value::{RuntimeError, Value};
use crate::Interpreter;

type R = Result<Value, RuntimeError>;

fn arity(name: &str, want: usize, got: usize, span: Span) -> RuntimeError {
    RuntimeError::new(
        codes::WRONG_ARG_COUNT,
        span,
        format!(
            "`{}` takes {} argument{}, {} given",
            name,
            want,
            if want == 1 { "" } else { "s" },
            got
        ),
    )
}

fn dims_from(args: &[Value], span: Span) -> Result<Vec<usize>, RuntimeError> {
    // Accept either `zeros(2, 3)` or `zeros([2, 3])`.
    if args.len() == 1 {
        if let Value::Array(items) = &args[0] {
            let mut out = Vec::with_capacity(items.len());
            for v in items.iter() {
                out.push(v.as_int(span)?.max(0) as usize);
            }
            return Ok(out);
        }
    }
    let mut out = Vec::with_capacity(args.len());
    for v in args {
        let n = v.as_int(span)?;
        if n < 0 {
            return Err(RuntimeError::new(
                codes::NEGATIVE_DIMENSION,
                span,
                format!("tensor dimension must be non-negative, found {}", n),
            ));
        }
        out.push(n as usize);
    }
    Ok(out)
}

/// FR-RT-008: `{}` placeholders, with `{{` and `}}` escapes.
pub fn format_args(fmt: &str, args: &[Value], span: Span) -> Result<String, RuntimeError> {
    let chars: Vec<char> = fmt.chars().collect();
    let mut out = String::with_capacity(fmt.len());
    let mut i = 0usize;
    let mut next = 0usize;
    while i < chars.len() {
        match chars[i] {
            '{' if i + 1 < chars.len() && chars[i + 1] == '{' => {
                out.push('{');
                i += 2;
            }
            '}' if i + 1 < chars.len() && chars[i + 1] == '}' => {
                out.push('}');
                i += 2;
            }
            '{' => {
                let mut j = i + 1;
                while j < chars.len() && chars[j] != '}' {
                    j += 1;
                }
                let spec: String = chars[i + 1..j.min(chars.len())].iter().collect();
                let v = args.get(next).ok_or_else(|| {
                    RuntimeError::new(
                        codes::FORMAT_ARG_COUNT,
                        span,
                        format!(
                            "format string needs at least {} arguments, {} supplied",
                            next + 1,
                            args.len()
                        ),
                    )
                })?;
                out.push_str(&apply_spec(v, &spec));
                next += 1;
                i = j + 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Supports `{}` and `{:.N}` precision, which is what training output needs.
fn apply_spec(v: &Value, spec: &str) -> String {
    if let Some(rest) = spec.strip_prefix(':') {
        if let Some(p) = rest.strip_prefix('.') {
            if let Ok(places) = p.parse::<usize>() {
                return match v {
                    Value::Float(f) => format!("{:.*}", places, f),
                    Value::Int(i) => format!("{:.*}", places, *i as f64),
                    Value::Tensor(t) if t.numel() == 1 => {
                        format!("{:.*}", places, t.to_vec()[0])
                    }
                    other => other.display(),
                };
            }
        }
    }
    v.display()
}

// ============================ free functions ============================

pub fn call(interp: &mut Interpreter, name: &str, args: Vec<Value>, span: Span) -> R {
    match name {
        "print" | "println" => {
            let text = match args.first() {
                None => String::new(),
                Some(Value::Str(f)) => format_args(f, &args[1..], span)?,
                Some(v) if args.len() == 1 => v.display(),
                Some(v) => {
                    let mut s = v.display();
                    for a in &args[1..] {
                        s.push(' ');
                        s.push_str(&a.display());
                    }
                    s
                }
            };
            let nl = if name == "println" { "\n" } else { "" };
            write!(interp.out, "{}{}", text, nl).map_err(|e| {
                RuntimeError::new(codes::IO_ERROR, span, format!("write failed: {}", e))
            })?;
            Ok(Value::Void)
        }
        "format" => {
            let f = args
                .first()
                .ok_or_else(|| arity("format", 1, 0, span))?
                .as_str(span)?;
            Ok(Value::Str(format_args(&f, &args[1..], span)?))
        }
        "assert" => {
            let ok = args
                .first()
                .ok_or_else(|| arity("assert", 1, 0, span))?
                .truthy(span)?;
            if !ok {
                let msg = match args.get(1) {
                    Some(Value::Str(s)) => format_args(s, &args[2..], span)?,
                    Some(v) => v.display(),
                    None => "assertion failed".to_string(),
                };
                return Err(RuntimeError::new(codes::ASSERTION_FAILED, span, msg));
            }
            Ok(Value::Void)
        }
        "len" => {
            let v = args.first().ok_or_else(|| arity("len", 1, 0, span))?;
            Ok(Value::Int(match v {
                Value::Array(a) => a.len() as i64,
                Value::Str(s) => s.chars().count() as i64,
                Value::Tensor(t) => t.dim(0) as i64,
                Value::Dataset(d) => d.len() as i64,
                other => {
                    return Err(RuntimeError::new(
                        codes::BAD_OPERAND_TYPE,
                        span,
                        format!("`len` is not defined on `{}`", other.type_name()),
                    ))
                }
            }))
        }
        "shape" => {
            let t = args
                .first()
                .ok_or_else(|| arity("shape", 1, 0, span))?
                .as_tensor(span)?;
            Ok(Value::Array(Rc::new(
                t.shape.iter().map(|&d| Value::Int(d as i64)).collect(),
            )))
        }
        "rank" => {
            let t = args
                .first()
                .ok_or_else(|| arity("rank", 1, 0, span))?
                .as_tensor(span)?;
            Ok(Value::Int(t.rank() as i64))
        }
        "zeros" | "ones" | "rand" | "randn" | "full" | "eye" | "arange" | "tensor" => {
            construct(interp, name, &args, span)
        }
        "one_hot" => {
            if args.len() != 2 {
                return Err(arity("one_hot", 2, args.len(), span));
            }
            let t = args[0].as_tensor(span)?;
            let c = args[1].as_int(span)? as usize;
            let out = k::one_hot(&t, c).map_err(|e| RuntimeError::from_tensor(e, span))?;
            Ok(Value::Tensor(Rc::new(out)))
        }
        "sum" | "mean" | "max" | "min" | "argmax" | "argmin" => reduce(name, &args, span),
        "abs" | "sqrt" | "exp" | "ln" | "floor" | "ceil" | "round" | "relu" | "sigmoid"
        | "tanh" | "softmax" => elementwise(name, &args, span),
        "pow" => {
            if args.len() != 2 {
                return Err(arity("pow", 2, args.len(), span));
            }
            match &args[0] {
                Value::Tensor(t) => {
                    let p = args[1].as_float(span)? as f32;
                    let out = k::unary_op(t, |x| x.powf(p))
                        .map_err(|e| RuntimeError::from_tensor(e, span))?;
                    Ok(Value::Tensor(Rc::new(out)))
                }
                _ => Ok(Value::Float(
                    args[0].as_float(span)?.powf(args[1].as_float(span)?),
                )),
            }
        }
        "int" => Ok(Value::Int(
            args.first().ok_or_else(|| arity("int", 1, 0, span))?.as_float(span)? as i64,
        )),
        "float" => Ok(Value::Float(
            args.first().ok_or_else(|| arity("float", 1, 0, span))?.as_float(span)?,
        )),
        "str" => Ok(Value::Str(
            args.first().map(|v| v.display()).unwrap_or_default(),
        )),
        "time_ms" => Ok(Value::Int(klions_platform::now_ms())),
        "Adam" | "SGD" | "RMSProp" => optimizer_ctor(name, &args, span),
        other => Err(RuntimeError::new(
            codes::UNDEFINED_NAME,
            span,
            format!("cannot find function `{}`", other),
        )),
    }
}

fn construct(interp: &mut Interpreter, name: &str, args: &[Value], span: Span) -> R {
    let t = match name {
        "zeros" => Tensor::zeros(&dims_from(args, span)?),
        "ones" => Tensor::ones(&dims_from(args, span)?),
        "rand" => {
            let d = dims_from(args, span)?;
            Tensor::rand_uniform(&d, 0.0, 1.0, &mut interp.rng)
        }
        "randn" => {
            let d = dims_from(args, span)?;
            Tensor::rand_normal(&d, 0.0, 1.0, &mut interp.rng)
        }
        "full" => {
            if args.is_empty() {
                return Err(arity("full", 2, 0, span));
            }
            let v = args[args.len() - 1].as_float(span)? as f32;
            let d = dims_from(&args[..args.len() - 1], span)?;
            Tensor::full(&d, v)
        }
        "eye" => {
            let n = args.first().ok_or_else(|| arity("eye", 1, 0, span))?.as_int(span)?;
            Tensor::eye(n.max(0) as usize)
        }
        "arange" => {
            let (start, end, step) = match args.len() {
                1 => (0.0, args[0].as_float(span)? as f32, 1.0),
                2 => (
                    args[0].as_float(span)? as f32,
                    args[1].as_float(span)? as f32,
                    1.0,
                ),
                3 => (
                    args[0].as_float(span)? as f32,
                    args[1].as_float(span)? as f32,
                    args[2].as_float(span)? as f32,
                ),
                n => return Err(arity("arange", 2, n, span)),
            };
            Tensor::arange(start, end, step)
        }
        "tensor" => {
            // `tensor([1, 2, 3])` or `tensor([[1, 2], [3, 4]])`
            let v = args.first().ok_or_else(|| arity("tensor", 1, 0, span))?;
            return build_from_nested(v, span);
        }
        _ => unreachable!(),
    }
    .map_err(|e| RuntimeError::from_tensor(e, span))?;
    Ok(Value::Tensor(Rc::new(t)))
}

fn build_from_nested(v: &Value, span: Span) -> R {
    fn walk(
        v: &Value,
        depth: usize,
        shape: &mut Vec<usize>,
        out: &mut Vec<f32>,
        span: Span,
    ) -> Result<(), RuntimeError> {
        match v {
            Value::Array(items) => {
                if shape.len() == depth {
                    shape.push(items.len());
                } else if shape[depth] != items.len() {
                    return Err(RuntimeError::new(
                        codes::SHAPE_MISMATCH,
                        span,
                        format!(
                            "ragged nested array: expected {} elements at depth {}, found {}",
                            shape[depth],
                            depth,
                            items.len()
                        ),
                    )
                    .with_help("every row of a nested array must have the same length"));
                }
                for it in items.iter() {
                    walk(it, depth + 1, shape, out, span)?;
                }
                Ok(())
            }
            other => {
                out.push(other.as_float(span)? as f32);
                Ok(())
            }
        }
    }
    let mut shape = Vec::new();
    let mut data = Vec::new();
    walk(v, 0, &mut shape, &mut data, span)?;
    let t = Tensor::from_vec(data, shape).map_err(|e| RuntimeError::from_tensor(e, span))?;
    Ok(Value::Tensor(Rc::new(t)))
}

fn reduce(name: &str, args: &[Value], span: Span) -> R {
    let v = args.first().ok_or_else(|| arity(name, 1, 0, span))?;
    let axis = args.get(1).map(|a| a.as_int(span)).transpose()?;
    let t = match v {
        Value::Tensor(t) => Rc::clone(t),
        Value::Array(items) => {
            let mut d = Vec::with_capacity(items.len());
            for it in items.iter() {
                d.push(it.as_float(span)? as f32);
            }
            let n = d.len();
            Rc::new(
                Tensor::from_vec(d, vec![n]).map_err(|e| RuntimeError::from_tensor(e, span))?,
            )
        }
        other => {
            return Err(RuntimeError::new(
                codes::BAD_OPERAND_TYPE,
                span,
                format!("`{}` is not defined on `{}`", name, other.type_name()),
            ))
        }
    };
    let out = match (name, axis) {
        ("sum", None) => k::sum_all(&t),
        ("mean", None) => k::mean_all(&t),
        ("max", None) => k::max_all(&t),
        ("min", None) => k::min_all(&t),
        ("argmax", None) => k::argmax_all(&t),
        ("argmin", None) => {
            let d = t.to_vec();
            let mut best = 0usize;
            for (i, &x) in d.iter().enumerate() {
                if x < d[best] {
                    best = i;
                }
            }
            Ok(Tensor::scalar(best as f32))
        }
        ("sum", Some(a)) => k::sum_axis(&t, a as usize, false),
        ("mean", Some(a)) => k::mean_axis(&t, a as usize, false),
        ("max", Some(a)) => k::max_axis(&t, a as usize, false),
        ("min", Some(a)) => k::min_axis(&t, a as usize, false),
        ("argmax", Some(a)) => k::argmax_axis(&t, a as usize, false),
        ("argmin", Some(a)) => {
            let neg = k::neg(&t).map_err(|e| RuntimeError::from_tensor(e, span))?;
            k::argmax_axis(&neg, a as usize, false)
        }
        _ => unreachable!(),
    }
    .map_err(|e| RuntimeError::from_tensor(e, span))?;

    if out.numel() == 1 {
        let x = out.to_vec()[0];
        return Ok(if name.starts_with("arg") {
            Value::Int(x as i64)
        } else {
            Value::Float(x as f64)
        });
    }
    Ok(Value::Tensor(Rc::new(out)))
}

fn elementwise(name: &str, args: &[Value], span: Span) -> R {
    let v = args.first().ok_or_else(|| arity(name, 1, 0, span))?;
    if let Value::Tensor(t) = v {
        let out = match name {
            "abs" => k::abs(t),
            "sqrt" => k::sqrt(t),
            "exp" => k::exp(t),
            "ln" => k::ln(t),
            "relu" => k::relu(t),
            "sigmoid" => k::sigmoid(t),
            "tanh" => k::tanh(t),
            "softmax" => k::softmax_last(t),
            "floor" => k::unary_op(t, f32::floor),
            "ceil" => k::unary_op(t, f32::ceil),
            "round" => k::unary_op(t, f32::round),
            _ => unreachable!(),
        }
        .map_err(|e| RuntimeError::from_tensor(e, span))?;
        return Ok(Value::Tensor(Rc::new(out)));
    }
    let x = v.as_float(span)?;
    Ok(Value::Float(match name {
        "abs" => x.abs(),
        "sqrt" => x.sqrt(),
        "exp" => x.exp(),
        "ln" => x.ln(),
        "floor" => x.floor(),
        "ceil" => x.ceil(),
        "round" => x.round(),
        "relu" => x.max(0.0),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        "softmax" => 1.0,
        _ => unreachable!(),
    }))
}

fn optimizer_ctor(name: &str, args: &[Value], span: Span) -> R {
    use klions_nn::optim::*;
    // Positional first argument is the learning rate; named arguments were
    // already flattened by the caller, so only the rate is read positionally.
    let lr = args.first().map(|v| v.as_float(span)).transpose()?;
    let kind = match name {
        "Adam" => {
            let mut o = default_adam();
            if let (OptimizerKind::Adam { learning_rate, .. }, Some(v)) = (&mut o, lr) {
                *learning_rate = v as f32;
            }
            o
        }
        "SGD" => {
            let mut o = default_sgd();
            if let (OptimizerKind::Sgd { learning_rate, momentum }, Some(v)) = (&mut o, lr) {
                *learning_rate = v as f32;
                if let Some(m) = args.get(1) {
                    *momentum = m.as_float(span)? as f32;
                }
            }
            o
        }
        "RMSProp" => {
            let mut o = default_rmsprop();
            if let (OptimizerKind::RmsProp { learning_rate, .. }, Some(v)) = (&mut o, lr) {
                *learning_rate = v as f32;
            }
            o
        }
        _ => unreachable!(),
    };
    Ok(Value::Optimizer(Rc::new(kind)))
}

// ============================ associated functions ============================

pub fn path_call(
    interp: &mut Interpreter,
    base: &str,
    name: &str,
    args: Vec<Value>,
    span: Span,
) -> R {
    match (base, name) {
        ("Dataset", "from_idx") => {
            if args.len() != 2 {
                return Err(arity("Dataset::from_idx", 2, args.len(), span));
            }
            let images = interp.resolve_path(&args[0].as_str(span)?, span)?;
            let labels = interp.resolve_path(&args[1].as_str(span)?, span)?;
            match klions_nn::data::from_idx(&images, &labels) {
                Ok(d) => Ok(Value::Ok(Box::new(Value::Dataset(Rc::new(d))))),
                Err(e) => Ok(Value::Err(Box::new(Value::Str(
                    RuntimeError::from_dataset(e, span).message,
                )))),
            }
        }
        ("Dataset", "from_csv") => {
            if args.is_empty() {
                return Err(arity("Dataset::from_csv", 2, 0, span));
            }
            let path = interp.resolve_path(&args[0].as_str(span)?, span)?;
            let col = args.get(1).map(|v| v.as_int(span)).transpose()?.unwrap_or(0) as usize;
            let header = match args.get(2) {
                Some(v) => v.truthy(span)?,
                None => detect_header(&path),
            };
            match klions_nn::data::from_csv(&path, col, header) {
                Ok(d) => Ok(Value::Ok(Box::new(Value::Dataset(Rc::new(d))))),
                Err(e) => Ok(Value::Err(Box::new(Value::Str(
                    RuntimeError::from_dataset(e, span).message,
                )))),
            }
        }
        ("Tensor", ctor) if matches!(ctor, "zeros" | "ones" | "rand" | "randn" | "eye") => {
            construct(interp, ctor, &args, span)
        }
        ("Tensor", "from_array") => {
            let v = args.first().ok_or_else(|| arity("Tensor::from_array", 1, 0, span))?;
            build_from_nested(v, span)
        }
        ("Optimizer", "adam") => optimizer_ctor("Adam", &args, span),
        ("Optimizer", "sgd") => optimizer_ctor("SGD", &args, span),
        ("Optimizer", "rmsprop") => optimizer_ctor("RMSProp", &args, span),
        _ => Err(RuntimeError::new(
            codes::UNDEFINED_NAME,
            span,
            format!("`{}::{}` does not exist", base, name),
        )),
    }
}

/// A first row that does not parse as numbers is treated as a header.
fn detect_header(path: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Some(first) = text.lines().next() else {
        return false;
    };
    first
        .split(',')
        .any(|c| c.trim().parse::<f64>().is_err() && !c.trim().is_empty())
}

// ============================ methods ============================

pub fn method(
    interp: &mut Interpreter,
    recv: &Value,
    name: &str,
    args: Vec<Value>,
    name_span: Span,
    span: Span,
) -> R {
    match recv {
        Value::Tensor(t) => tensor_method(t, name, &args, span),
        Value::Model(m) => model_method(interp, m, name, &args, span),
        Value::Dataset(d) => dataset_method(interp, d, name, &args, span),
        Value::Metrics(e) => match name {
            "loss" => Ok(Value::Float(e.loss as f64)),
            "accuracy" => Ok(Value::Float(e.accuracy as f64)),
            "samples" => Ok(Value::Int(e.samples as i64)),
            _ => no_method(recv, name, name_span),
        },
        Value::Optimizer(o) => match name {
            "learning_rate" => Ok(Value::Float(o.learning_rate() as f64)),
            "name" => Ok(Value::Str(o.name().to_string())),
            _ => no_method(recv, name, name_span),
        },
        Value::Array(items) => match name {
            "len" => Ok(Value::Int(items.len() as i64)),
            "get" => {
                let i = args.first().ok_or_else(|| arity("get", 1, 0, span))?.as_int(span)?;
                crate::value::index(recv, &[i], span)
            }
            _ => no_method(recv, name, name_span),
        },
        Value::Str(s) => match name {
            "len" => Ok(Value::Int(s.chars().count() as i64)),
            _ => no_method(recv, name, name_span),
        },
        Value::Ok(v) => match name {
            "unwrap" => Ok((**v).clone()),
            "is_ok" => Ok(Value::Bool(true)),
            "is_err" => Ok(Value::Bool(false)),
            _ => no_method(recv, name, name_span),
        },
        Value::Err(e) => match name {
            // FR-ERR-002: unwrapping an Err reports the original error text.
            "unwrap" => Err(RuntimeError::new(
                codes::UNWRAP_ON_ERR,
                span,
                format!("called `unwrap` on an error value: {}", e.display()),
            )
            .with_help("handle the error with `match`-free `if r.is_err()`, or propagate it with `?`")),
            "is_ok" => Ok(Value::Bool(false)),
            "is_err" => Ok(Value::Bool(true)),
            _ => no_method(recv, name, name_span),
        },
        other => Err(RuntimeError::new(
            codes::NO_SUCH_FIELD,
            name_span,
            format!("`{}` has no method `{}`", other.type_name(), name),
        )),
    }
}

fn no_method(recv: &Value, name: &str, span: Span) -> R {
    Err(RuntimeError::new(
        codes::NO_SUCH_FIELD,
        span,
        format!("`{}` has no method named `{}`", recv.type_name(), name),
    ))
}

fn tensor_method(t: &Rc<Tensor>, name: &str, args: &[Value], span: Span) -> R {
    let ft = |x: Tensor| Ok(Value::Tensor(Rc::new(x)));
    match name {
        "shape" => Ok(Value::Array(Rc::new(
            t.shape.iter().map(|&d| Value::Int(d as i64)).collect(),
        ))),
        "rank" => Ok(Value::Int(t.rank() as i64)),
        "len" => Ok(Value::Int(t.dim(0) as i64)),
        "numel" => Ok(Value::Int(t.numel() as i64)),
        "item" => Ok(Value::Float(
            t.item().map_err(|e| RuntimeError::from_tensor(e, span))? as f64,
        )),
        "to_vec" => Ok(Value::Array(Rc::new(
            t.to_vec().into_iter().map(|v| Value::Float(v as f64)).collect(),
        ))),
        "clone" => ft(t.contiguous()),
        "reshape" => {
            let dims: Vec<isize> = if args.len() == 1 {
                match &args[0] {
                    Value::Array(items) => {
                        let mut d = Vec::with_capacity(items.len());
                        for v in items.iter() {
                            d.push(v.as_int(span)? as isize);
                        }
                        d
                    }
                    v => vec![v.as_int(span)? as isize],
                }
            } else {
                let mut d = Vec::with_capacity(args.len());
                for v in args {
                    d.push(v.as_int(span)? as isize);
                }
                d
            };
            ft(t.reshape(&dims).map_err(|e| RuntimeError::from_tensor(e, span))?)
        }
        "flatten" => ft(t.flatten().map_err(|e| RuntimeError::from_tensor(e, span))?),
        "t" => ft(t
            .t()
            .map_err(|e| RuntimeError::from_tensor(e, span))?
            .contiguous()),
        "transpose" => {
            let (a, b) = match args.len() {
                0 => (t.rank().saturating_sub(2), t.rank().saturating_sub(1)),
                2 => (args[0].as_int(span)? as usize, args[1].as_int(span)? as usize),
                n => return Err(arity("transpose", 2, n, span)),
            };
            ft(t.transpose(a, b)
                .map_err(|e| RuntimeError::from_tensor(e, span))?
                .contiguous())
        }
        "sum" | "mean" | "max" | "min" | "argmax" | "argmin" => {
            let mut full = vec![Value::Tensor(Rc::clone(t))];
            full.extend(args.iter().cloned());
            reduce(name, &full, span)
        }
        "relu" | "sigmoid" | "tanh" | "softmax" | "exp" | "ln" | "sqrt" | "abs" => {
            elementwise(name, &[Value::Tensor(Rc::clone(t))], span)
        }
        _ => Err(RuntimeError::new(
            codes::NO_SUCH_FIELD,
            span,
            format!("`tensor` has no method named `{}`", name),
        )),
    }
}

fn model_method(
    interp: &mut Interpreter,
    m: &Rc<std::cell::RefCell<klions_nn::Model>>,
    name: &str,
    args: &[Value],
    span: Span,
) -> R {
    match name {
        "predict" => {
            let x = args
                .first()
                .ok_or_else(|| arity("predict", 1, 0, span))?
                .as_tensor(span)?;
            let out = m
                .borrow()
                .predict(&x, &mut interp.rng)
                .map_err(|e| RuntimeError::from_tensor(e, span))?;
            Ok(Value::Tensor(Rc::new(out)))
        }
        "evaluate" => {
            let d = match args.first() {
                Some(Value::Dataset(d)) => Rc::clone(d),
                _ => {
                    return Err(RuntimeError::new(
                        codes::NOT_A_DATASET,
                        span,
                        "`evaluate` takes a Dataset",
                    ))
                }
            };
            let r = crate::train::evaluate(interp, &m.borrow(), &d, span)?;
            Ok(Value::Metrics(Rc::new(r)))
        }
        "summary" => Ok(Value::Str(m.borrow().summary())),
        "parameter_count" => Ok(Value::Int(m.borrow().param_count() as i64)),
        "save" => {
            let p = args
                .first()
                .ok_or_else(|| arity("save", 1, 0, span))?
                .as_str(span)?;
            let path = interp.resolve_path(&p, span)?;
            match klions_nn::serialize::save(&m.borrow(), &path) {
                Ok(()) => Ok(Value::Ok(Box::new(Value::Void))),
                Err(e) => Ok(Value::Err(Box::new(Value::Str(e.to_string())))),
            }
        }
        "load" => {
            let p = args
                .first()
                .ok_or_else(|| arity("load", 1, 0, span))?
                .as_str(span)?;
            let path = interp.resolve_path(&p, span)?;
            match klions_nn::serialize::load_into(&m.borrow(), &path) {
                Ok(()) => Ok(Value::Ok(Box::new(Value::Void))),
                Err(e) => Ok(Value::Err(Box::new(Value::Str(e.to_string())))),
            }
        }
        _ => Err(RuntimeError::new(
            codes::NO_SUCH_FIELD,
            span,
            format!("`Model` has no method named `{}`", name),
        )),
    }
}

fn dataset_method(
    interp: &mut Interpreter,
    d: &Rc<Dataset>,
    name: &str,
    args: &[Value],
    span: Span,
) -> R {
    let wrap = |x: Dataset| Ok(Value::Dataset(Rc::new(x)));
    match name {
        "batch" => {
            let n = args.first().ok_or_else(|| arity("batch", 1, 0, span))?.as_int(span)?;
            if n <= 0 {
                return Err(RuntimeError::new(
                    codes::BAD_HYPERPARAMETER,
                    span,
                    format!("batch size must be at least 1, found {}", n),
                ));
            }
            wrap(d.batch(n as usize))
        }
        "shuffle" => wrap(d.shuffle(&mut interp.rng)),
        "normalize" => wrap(d.normalize().map_err(|e| RuntimeError::from_tensor(e, span))?),
        "standardize" => wrap(
            d.standardize()
                .map_err(|e| RuntimeError::from_tensor(e, span))?,
        ),
        "one_hot" => {
            let c = args.first().ok_or_else(|| arity("one_hot", 1, 0, span))?.as_int(span)?;
            wrap(
                d.one_hot(c.max(1) as usize)
                    .map_err(|e| RuntimeError::from_tensor(e, span))?,
            )
        }
        "split" => {
            let f = args.first().ok_or_else(|| arity("split", 1, 0, span))?.as_float(span)?;
            let (a, _) = d.split(f as f32);
            wrap(a)
        }
        "len" | "size" => Ok(Value::Int(d.len() as i64)),
        "feature_width" => Ok(Value::Int(d.feature_width() as i64)),
        "label_width" => Ok(Value::Int(d.label_width() as i64)),
        "features" => Ok(Value::Tensor(Rc::new(d.features.clone()))),
        "labels" => Ok(Value::Tensor(Rc::new(d.labels.clone()))),
        "sample" => {
            let i = args.first().ok_or_else(|| arity("sample", 1, 0, span))?.as_int(span)?;
            if i < 0 || i as usize >= d.len() {
                return Err(RuntimeError::new(
                    codes::INDEX_ERROR,
                    span,
                    format!("sample index {} is out of range for {} samples", i, d.len()),
                ));
            }
            Ok(Value::Tensor(Rc::new(
                d.get_sample(i as usize)
                    .map_err(|e| RuntimeError::from_tensor(e, span))?,
            )))
        }
        "label" => {
            let i = args.first().ok_or_else(|| arity("label", 1, 0, span))?.as_int(span)?;
            if i < 0 || i as usize >= d.len() {
                return Err(RuntimeError::new(
                    codes::INDEX_ERROR,
                    span,
                    format!("label index {} is out of range for {} samples", i, d.len()),
                ));
            }
            Ok(Value::Float(
                d.get_label(i as usize)
                    .map_err(|e| RuntimeError::from_tensor(e, span))? as f64,
            ))
        }
        _ => Err(RuntimeError::new(
            codes::NO_SUCH_FIELD,
            span,
            format!("`Dataset` has no method named `{}`", name),
        )),
    }
}

// ============================ field access ============================

pub fn field(interp: &mut Interpreter, recv: &Value, name: &str, span: Span) -> R {
    // Zero-argument methods read naturally as fields: `x.shape` == `x.shape()`.
    method(interp, recv, name, Vec::new(), span, span)
}

/// Shared by `train` and `model.evaluate`.
pub fn loss_from_name(name: &str, span: Span) -> Result<loss::Loss, RuntimeError> {
    loss::Loss::parse(name).ok_or_else(|| {
        RuntimeError::new(
            codes::UNKNOWN_LOSS,
            span,
            format!("unknown loss function `{}`", name),
        )
        .with_help(format!("available: {}", loss::LOSS_NAMES.join(", ")))
    })
}

/// Convenience used by the training loop when tracking must be on.
pub fn with_grad<T>(f: impl FnOnce() -> T) -> T {
    let prev = ad::set_tracking(true);
    let out = f();
    ad::set_tracking(prev);
    out
}
