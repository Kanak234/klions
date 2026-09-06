//! Runtime values and their operations — FR-RT-002, FR-RT-006, FR-RT-007.

use std::cell::RefCell;
use std::rc::Rc;

use klions_ast::{BinOp, TypeExpr, TypeExprKind, UnOp};
use klions_diagnostics::{codes, Diagnostic, Span};
use klions_nn::data::{Dataset, DatasetError};
use klions_nn::{EvalResult as NnEval, Model};
use klions_nn::optim::OptimizerKind;
use klions_tensor::kernels as k;
use klions_tensor::{Tensor, TensorError};

#[derive(Clone)]
pub enum Value {
    Void,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// FR-RT-009: reference-counted, so passing a tensor never copies it.
    Tensor(Rc<Tensor>),
    Array(Rc<Vec<Value>>),
    Model(Rc<RefCell<Model>>),
    Dataset(Rc<Dataset>),
    Optimizer(Rc<OptimizerKind>),
    Metrics(Rc<NnEval>),
    Ok(Box<Value>),
    Err(Box<Value>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Void => "void",
            Value::Int(_) => "i64",
            Value::Float(_) => "f64",
            Value::Bool(_) => "bool",
            Value::Str(_) => "string",
            Value::Tensor(_) => "tensor",
            Value::Array(_) => "array",
            Value::Model(_) => "Model",
            Value::Dataset(_) => "Dataset",
            Value::Optimizer(_) => "Optimizer",
            Value::Metrics(_) => "Metrics",
            Value::Ok(_) | Value::Err(_) => "Result",
        }
    }

    pub fn truthy(&self, span: Span) -> Result<bool, RuntimeError> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(RuntimeError::new(
                codes::CONDITION_NOT_BOOL,
                span,
                format!("expected a `bool`, found `{}`", other.type_name()),
            )
            .with_help("KLIONS has no truthiness; write an explicit comparison")),
        }
    }

    pub fn as_int(&self, span: Span) -> Result<i64, RuntimeError> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            Value::Bool(b) => Ok(*b as i64),
            other => Err(RuntimeError::new(
                codes::TYPE_MISMATCH,
                span,
                format!("expected an integer, found `{}`", other.type_name()),
            )),
        }
    }

    pub fn as_float(&self, span: Span) -> Result<f64, RuntimeError> {
        match self {
            Value::Int(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            Value::Tensor(t) if t.numel() == 1 => {
                t.item().map(|v| v as f64).map_err(|e| RuntimeError::from_tensor(e, span))
            }
            other => Err(RuntimeError::new(
                codes::TYPE_MISMATCH,
                span,
                format!("expected a number, found `{}`", other.type_name()),
            )),
        }
    }

    pub fn as_str(&self, span: Span) -> Result<String, RuntimeError> {
        match self {
            Value::Str(s) => Ok(s.clone()),
            other => Err(RuntimeError::new(
                codes::TYPE_MISMATCH,
                span,
                format!("expected a string, found `{}`", other.type_name()),
            )),
        }
    }

    pub fn as_tensor(&self, span: Span) -> Result<Rc<Tensor>, RuntimeError> {
        match self {
            Value::Tensor(t) => Ok(Rc::clone(t)),
            Value::Int(i) => Ok(Rc::new(Tensor::scalar(*i as f32))),
            Value::Float(f) => Ok(Rc::new(Tensor::scalar(*f as f32))),
            other => Err(RuntimeError::new(
                codes::TYPE_MISMATCH,
                span,
                format!("expected a tensor, found `{}`", other.type_name()),
            )),
        }
    }

    /// FR-RT-008: how a value appears inside `{}`.
    pub fn display(&self) -> String {
        match self {
            Value::Void => "void".to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => format_float(*f),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => s.clone(),
            Value::Tensor(t) => {
                if t.numel() == 1 {
                    klions_tensor::fmt_f32(t.to_vec()[0])
                } else if t.rank() <= 1 && t.numel() <= 8 {
                    format!(
                        "[{}]",
                        t.to_vec()
                            .iter()
                            .map(|v| klions_tensor::fmt_f32(*v))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    t.format()
                }
            }
            Value::Array(items) => format!(
                "[{}]",
                items.iter().map(|v| v.display()).collect::<Vec<_>>().join(", ")
            ),
            Value::Model(m) => format!("<model {}>", m.borrow().name),
            Value::Dataset(d) => format!("<dataset {} ({} samples)>", d.name, d.len()),
            Value::Optimizer(o) => format!("<{}>", o.describe()),
            Value::Metrics(m) => format!(
                "loss = {:.4}, accuracy = {:.4}",
                m.loss, m.accuracy
            ),
            Value::Ok(v) => format!("Ok({})", v.display()),
            Value::Err(v) => format!("Err({})", v.display()),
        }
    }
}

pub fn format_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == f.trunc() && f.abs() < 1e15 {
        format!("{:.1}", f)
    } else {
        let s = format!("{}", f);
        s
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

// ---------------- operators ----------------

pub fn unary(op: UnOp, v: &Value, span: Span) -> Result<Value, RuntimeError> {
    match (op, v) {
        (UnOp::Neg, Value::Int(i)) => i
            .checked_neg()
            .map(Value::Int)
            .ok_or_else(|| RuntimeError::overflow("negation", span)),
        (UnOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
        (UnOp::Neg, Value::Tensor(t)) => Ok(Value::Tensor(Rc::new(
            k::neg(t).map_err(|e| RuntimeError::from_tensor(e, span))?,
        ))),
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (op, other) => Err(RuntimeError::new(
            codes::BAD_OPERAND_TYPE,
            span,
            format!("`{}` is not defined on `{}`", op.as_str(), other.type_name()),
        )),
    }
}

pub fn binary(op: BinOp, a: &Value, b: &Value, span: Span) -> Result<Value, RuntimeError> {
    use BinOp::*;

    // Tensor arithmetic, including scalar broadcast.
    if matches!(a, Value::Tensor(_)) || matches!(b, Value::Tensor(_)) {
        if op.is_comparison() {
            return Err(RuntimeError::new(
                codes::BAD_OPERAND_TYPE,
                span,
                format!("`{}` is not defined on tensors", op.as_str()),
            ));
        }
        let at = a.as_tensor(span)?;
        let bt = b.as_tensor(span)?;
        let out = match op {
            Add => k::add(&at, &bt),
            Sub => k::sub(&at, &bt),
            Mul => k::mul(&at, &bt),
            Div => k::div(&at, &bt),
            MatMul => k::matmul(&at, &bt),
            other => {
                return Err(RuntimeError::new(
                    codes::BAD_OPERAND_TYPE,
                    span,
                    format!("`{}` is not defined on tensors", other.as_str()),
                ))
            }
        }
        .map_err(|e| RuntimeError::from_tensor(e, span))?;
        return Ok(Value::Tensor(Rc::new(out)));
    }

    match (a, b) {
        (Value::Int(x), Value::Int(y)) => int_binary(op, *x, *y, span),
        (Value::Float(x), Value::Float(y)) => float_binary(op, *x, *y, span),
        // Mixed int/float arises only from literals the checker has approved.
        (Value::Int(x), Value::Float(y)) => float_binary(op, *x as f64, *y, span),
        (Value::Float(x), Value::Int(y)) => float_binary(op, *x, *y as f64, span),
        (Value::Bool(x), Value::Bool(y)) => match op {
            Eq => Ok(Value::Bool(x == y)),
            Ne => Ok(Value::Bool(x != y)),
            And => Ok(Value::Bool(*x && *y)),
            Or => Ok(Value::Bool(*x || *y)),
            other => Err(RuntimeError::new(
                codes::BAD_OPERAND_TYPE,
                span,
                format!("`{}` is not defined on booleans", other.as_str()),
            )),
        },
        (Value::Str(x), Value::Str(y)) => match op {
            Eq => Ok(Value::Bool(x == y)),
            Ne => Ok(Value::Bool(x != y)),
            Lt => Ok(Value::Bool(x < y)),
            Le => Ok(Value::Bool(x <= y)),
            Gt => Ok(Value::Bool(x > y)),
            Ge => Ok(Value::Bool(x >= y)),
            other => Err(RuntimeError::new(
                codes::BAD_OPERAND_TYPE,
                span,
                format!("`{}` is not defined on strings", other.as_str()),
            )
            .with_help("use `format(\"{}{}\", a, b)` to join strings")),
        },
        (x, y) => Err(RuntimeError::new(
            codes::BAD_OPERAND_TYPE,
            span,
            format!(
                "`{}` is not defined between `{}` and `{}`",
                op.as_str(),
                x.type_name(),
                y.type_name()
            ),
        )),
    }
}

/// FR-RT-006: integer overflow and division by zero are trapped.
fn int_binary(op: BinOp, x: i64, y: i64, span: Span) -> Result<Value, RuntimeError> {
    use BinOp::*;
    Ok(match op {
        Add => Value::Int(x.checked_add(y).ok_or_else(|| RuntimeError::overflow("addition", span))?),
        Sub => Value::Int(x.checked_sub(y).ok_or_else(|| RuntimeError::overflow("subtraction", span))?),
        Mul => Value::Int(
            x.checked_mul(y)
                .ok_or_else(|| RuntimeError::overflow("multiplication", span))?,
        ),
        Div => {
            if y == 0 {
                return Err(RuntimeError::new(
                    codes::DIVISION_BY_ZERO,
                    span,
                    "integer division by zero",
                )
                .with_help(
                    "guard the divisor, or use floats where an infinite result is acceptable",
                ));
            }
            Value::Int(x.checked_div(y).ok_or_else(|| RuntimeError::overflow("division", span))?)
        }
        Rem => {
            if y == 0 {
                return Err(RuntimeError::new(
                    codes::DIVISION_BY_ZERO,
                    span,
                    "integer remainder by zero",
                ));
            }
            Value::Int(x.wrapping_rem(y))
        }
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or | MatMul => {
            return Err(RuntimeError::new(
                codes::BAD_OPERAND_TYPE,
                span,
                format!("`{}` is not defined on integers", op.as_str()),
            ))
        }
    })
}

/// FR-RT-006: float arithmetic follows IEEE 754 — no trapping.
fn float_binary(op: BinOp, x: f64, y: f64, span: Span) -> Result<Value, RuntimeError> {
    use BinOp::*;
    Ok(match op {
        Add => Value::Float(x + y),
        Sub => Value::Float(x - y),
        Mul => Value::Float(x * y),
        Div => Value::Float(x / y),
        Rem => Value::Float(x % y),
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or | MatMul => {
            return Err(RuntimeError::new(
                codes::BAD_OPERAND_TYPE,
                span,
                format!("`{}` is not defined on floats", op.as_str()),
            ))
        }
    })
}

pub fn cast(v: &Value, ty: &TypeExpr, span: Span) -> Result<Value, RuntimeError> {
    let name = match &ty.kind {
        TypeExprKind::Named(n) => n.as_str(),
        TypeExprKind::Tensor { .. } => return Ok(v.clone()),
        _ => {
            return Err(RuntimeError::new(
                codes::BAD_CAST,
                span,
                "only numeric casts are supported",
            ))
        }
    };
    Ok(match name {
        "i32" => Value::Int(v.as_float(span)? as i32 as i64),
        "i64" => Value::Int(v.as_float(span)? as i64),
        "f32" => Value::Float(v.as_float(span)? as f32 as f64),
        "f64" => Value::Float(v.as_float(span)?),
        "bool" => Value::Bool(v.as_float(span)? != 0.0),
        "string" => Value::Str(v.display()),
        other => {
            return Err(RuntimeError::new(
                codes::BAD_CAST,
                span,
                format!("cannot cast to `{}`", other),
            ))
        }
    })
}

pub fn index(base: &Value, idx: &[i64], span: Span) -> Result<Value, RuntimeError> {
    match base {
        Value::Tensor(t) => {
            // Negative indices are rejected explicitly rather than wrapping.
            let mut u = Vec::with_capacity(idx.len());
            for (axis, &i) in idx.iter().enumerate() {
                if i < 0 {
                    return Err(RuntimeError::new(
                        codes::INDEX_ERROR,
                        span,
                        format!("index {} is negative on axis {}", i, axis),
                    )
                    .with_help("indices are zero-based and non-negative"));
                }
                u.push(i as usize);
            }
            if u.len() == t.rank() {
                let v = t.get(&u).map_err(|e| RuntimeError::from_tensor(e, span))?;
                Ok(Value::Float(v as f64))
            } else if u.len() < t.rank() {
                let mut view = (**t).clone();
                for &i in &u {
                    view = view
                        .row(i)
                        .map_err(|e| RuntimeError::from_tensor(e, span))?;
                }
                Ok(Value::Tensor(Rc::new(view)))
            } else {
                Err(RuntimeError::new(
                    codes::BAD_RANK,
                    span,
                    format!(
                        "{} indices given for a rank-{} tensor",
                        u.len(),
                        t.rank()
                    ),
                ))
            }
        }
        Value::Array(items) => {
            let i = *idx.first().unwrap_or(&0);
            if i < 0 || i as usize >= items.len() {
                return Err(RuntimeError::new(
                    codes::INDEX_ERROR,
                    span,
                    format!("index {} is out of bounds for an array of length {}", i, items.len()),
                ));
            }
            Ok(items[i as usize].clone())
        }
        Value::Dataset(d) => {
            let i = *idx.first().unwrap_or(&0);
            if i < 0 || i as usize >= d.len() {
                return Err(RuntimeError::new(
                    codes::INDEX_ERROR,
                    span,
                    format!("index {} is out of bounds for a dataset of {} samples", i, d.len()),
                ));
            }
            let t = d
                .get_sample(i as usize)
                .map_err(|e| RuntimeError::from_tensor(e, span))?;
            Ok(Value::Tensor(Rc::new(t)))
        }
        other => Err(RuntimeError::new(
            codes::NOT_INDEXABLE,
            span,
            format!("`{}` cannot be indexed", other.type_name()),
        )),
    }
}

// ---------------- errors ----------------

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub code: &'static str,
    pub span: Span,
    pub message: String,
    pub help: Option<String>,
    pub note: Option<String>,
    /// Set when the process should exit 130 rather than 101 (FR-RT-011).
    pub interrupted: bool,
}

impl RuntimeError {
    pub fn new(code: &'static str, span: Span, message: impl Into<String>) -> RuntimeError {
        RuntimeError {
            code,
            span,
            message: message.into(),
            help: None,
            note: None,
            interrupted: false,
        }
    }

    pub fn with_help(mut self, h: impl Into<String>) -> RuntimeError {
        self.help = Some(h.into());
        self
    }
    pub fn with_note(mut self, n: impl Into<String>) -> RuntimeError {
        self.note = Some(n.into());
        self
    }

    pub fn overflow(what: &str, span: Span) -> RuntimeError {
        RuntimeError::new(
            codes::INTEGER_OVERFLOW,
            span,
            format!("integer {} overflowed the range of i64", what),
        )
        .with_help("KLIONS traps overflow rather than wrapping silently")
    }

    pub fn interrupted(span: Span) -> RuntimeError {
        let mut e = RuntimeError::new(
            codes::RUNTIME_PANIC,
            span,
            "interrupted",
        );
        e.interrupted = true;
        e
    }

    pub fn propagated(v: Value, span: Span) -> RuntimeError {
        RuntimeError::new(
            codes::UNWRAP_ON_ERR,
            span,
            format!("error propagated by `?`: {}", v.display()),
        )
    }

    pub fn from_tensor(e: TensorError, span: Span) -> RuntimeError {
        let mut r = RuntimeError::new(e.code(), span, e.to_string());
        r.help = e.help();
        r.note = e.note();
        r
    }

    pub fn from_dataset(e: DatasetError, span: Span) -> RuntimeError {
        let mut r = RuntimeError::new(e.code(), span, e.to_string());
        r.help = e.help();
        r
    }

    pub fn to_diagnostic(&self) -> Diagnostic {
        let mut d = Diagnostic::error(self.code, self.span, self.message.clone());
        if let Some(h) = &self.help {
            d = d.with_help(h.clone());
        }
        if let Some(n) = &self.note {
            d = d.with_note(n.clone());
        }
        d
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}
