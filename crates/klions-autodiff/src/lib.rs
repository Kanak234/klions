//! KLIONS reverse-mode automatic differentiation — FR-AI-001 … FR-AI-006.
//!
//! A dynamically-recorded tape. Every differentiable operation performed while
//! tracking is active appends a node recording its parents and the local
//! derivative rule (FR-AI-002). `backward` replays the tape in reverse,
//! accumulating gradients (FR-AI-003).
//!
//! Tracking is suspendable with [`no_grad`] (FR-AI-004), and the tape is
//! cleared at the end of each optimizer step so memory does not grow across a
//! training run (FR-AI-005).

use std::cell::RefCell;
use std::rc::Rc;

use klions_tensor::kernels as k;
use klions_tensor::{Tensor, TensorResult};

pub mod check;

/// Contribution returned by a backward rule: (parent node, gradient).
type Contribution = (usize, Tensor);
type BackwardFn = Box<dyn Fn(&Tensor) -> TensorResult<Vec<Contribution>>>;

struct Node {
    parents: Vec<usize>,
    /// `None` for a leaf: gradient stops here and is collected.
    backward: Option<BackwardFn>,
    shape: Vec<usize>,
    /// Operation name. Kept for `KLIONS_DEBUG_TAPE` inspection and for the
    /// error messages the gradient checker produces on failure.
    #[allow(dead_code)]
    label: &'static str,
}

#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
    enabled: bool,
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape { nodes: Vec::new(), enabled: false });
}

/// Is gradient tracking currently active?
pub fn is_tracking() -> bool {
    TAPE.with(|t| t.borrow().enabled)
}

pub fn set_tracking(on: bool) -> bool {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let prev = t.enabled;
        t.enabled = on;
        prev
    })
}

/// FR-AI-004: run `f` with tracking suspended; no tape nodes are recorded.
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let prev = set_tracking(false);
    let out = f();
    set_tracking(prev);
    out
}

/// FR-AI-005: drop every node. Called after each optimizer step.
pub fn clear_tape() {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        t.nodes.clear();
        t.nodes.shrink_to_fit();
    });
}

pub fn tape_len() -> usize {
    TAPE.with(|t| t.borrow().nodes.len())
}

/// Register a leaf (a parameter or an input we want gradients for).
pub fn leaf(shape: &[usize]) -> usize {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        t.nodes.push(Node {
            parents: Vec::new(),
            backward: None,
            shape: shape.to_vec(),
            label: "leaf",
        });
        t.nodes.len() - 1
    })
}

fn push(label: &'static str, parents: Vec<usize>, shape: Vec<usize>, backward: BackwardFn) -> usize {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        t.nodes.push(Node { parents, backward: Some(backward), shape, label });
        t.nodes.len() - 1
    })
}

/// A tensor plus its position on the tape.
#[derive(Clone, Debug)]
pub struct Var {
    pub value: Tensor,
    pub node: Option<usize>,
}

impl Var {
    /// A constant: participates in the graph but carries no gradient.
    pub fn constant(value: Tensor) -> Var {
        Var { value, node: None }
    }

    /// A differentiable leaf.
    pub fn parameter(value: Tensor) -> Var {
        let node = if is_tracking() { Some(leaf(&value.shape)) } else { None };
        Var { value, node }
    }

    pub fn shape(&self) -> &[usize] {
        &self.value.shape
    }
    pub fn requires_grad(&self) -> bool {
        self.node.is_some()
    }

    fn record(
        &self,
        other: Option<&Var>,
        label: &'static str,
        out: Tensor,
        backward: BackwardFn,
    ) -> Var {
        if !is_tracking() {
            return Var::constant(out);
        }
        let mut parents = Vec::new();
        if let Some(n) = self.node {
            parents.push(n);
        }
        if let Some(o) = other {
            if let Some(n) = o.node {
                parents.push(n);
            }
        }
        if parents.is_empty() {
            return Var::constant(out);
        }
        let shape = out.shape.clone();
        let node = push(label, parents, shape, backward);
        Var { value: out, node: Some(node) }
    }
}

// ============================ differentiable ops ============================

macro_rules! binary_grad {
    ($name:ident, $label:literal, $fwd:path, $da:expr, $db:expr) => {
        pub fn $name(a: &Var, b: &Var) -> TensorResult<Var> {
            let out = $fwd(&a.value, &b.value)?;
            if !is_tracking() || (a.node.is_none() && b.node.is_none()) {
                return Ok(Var::constant(out));
            }
            let (an, bn) = (a.node, b.node);
            let (ash, bsh) = (a.value.shape.clone(), b.value.shape.clone());
            let (av, bv) = (a.value.clone(), b.value.clone());
            let mut parents = Vec::new();
            if let Some(n) = an { parents.push(n); }
            if let Some(n) = bn { parents.push(n); }
            let shape = out.shape.clone();
            let node = push($label, parents, shape, Box::new(move |g: &Tensor| {
                let mut out: Vec<Contribution> = Vec::new();
                if let Some(n) = an {
                    let ga: Tensor = $da(g, &av, &bv)?;
                    out.push((n, k::unbroadcast(&ga, &ash)?));
                }
                if let Some(n) = bn {
                    let gb: Tensor = $db(g, &av, &bv)?;
                    out.push((n, k::unbroadcast(&gb, &bsh)?));
                }
                Ok(out)
            }));
            Ok(Var { value: out, node: Some(node) })
        }
    };
}

// d(a+b)/da = 1, d/db = 1
binary_grad!(add, "add", k::add,
    |g: &Tensor, _a: &Tensor, _b: &Tensor| Ok(g.clone()),
    |g: &Tensor, _a: &Tensor, _b: &Tensor| Ok(g.clone()));

// d(a-b)/da = 1, d/db = -1
binary_grad!(sub, "sub", k::sub,
    |g: &Tensor, _a: &Tensor, _b: &Tensor| Ok(g.clone()),
    |g: &Tensor, _a: &Tensor, _b: &Tensor| k::neg(g));

// d(a*b)/da = b, d/db = a
binary_grad!(mul, "mul", k::mul,
    |g: &Tensor, _a: &Tensor, b: &Tensor| k::mul(g, b),
    |g: &Tensor, a: &Tensor, _b: &Tensor| k::mul(g, a));

// d(a/b)/da = 1/b, d/db = -a/b^2
binary_grad!(div, "div", k::div,
    |g: &Tensor, _a: &Tensor, b: &Tensor| k::div(g, b),
    |g: &Tensor, a: &Tensor, b: &Tensor| {
        let num = k::mul(g, a)?;
        let den = k::mul(b, b)?;
        k::neg(&k::div(&num, &den)?)
    });

/// FR-TEN-005 matmul with its transpose-pair backward rule.
pub fn matmul(a: &Var, b: &Var) -> TensorResult<Var> {
    let out = k::matmul(&a.value, &b.value)?;
    if !is_tracking() || (a.node.is_none() && b.node.is_none()) {
        return Ok(Var::constant(out));
    }
    let (an, bn) = (a.node, b.node);
    let (av, bv) = (a.value.clone(), b.value.clone());
    let (ash, bsh) = (a.value.shape.clone(), b.value.shape.clone());
    let mut parents = Vec::new();
    if let Some(n) = an {
        parents.push(n);
    }
    if let Some(n) = bn {
        parents.push(n);
    }
    let shape = out.shape.clone();
    let node = push(
        "matmul",
        parents,
        shape,
        Box::new(move |g: &Tensor| {
            let mut res: Vec<Contribution> = Vec::new();
            // dL/dA = G @ B^T ; dL/dB = A^T @ G
            if let Some(n) = an {
                let ga = k::matmul(g, &bv.t()?.contiguous())?;
                res.push((n, k::unbroadcast(&ga, &ash)?));
            }
            if let Some(n) = bn {
                let gb = if av.rank() == 3 {
                    // Batched: fold the batch into rows, then contract.
                    let m = av.dim(0) * av.dim(1);
                    let a2 = av.reshape(&[m as isize, av.dim(2) as isize])?;
                    let g2 = g.reshape(&[m as isize, g.dim(g.rank() - 1) as isize])?;
                    k::matmul(&a2.t()?.contiguous(), &g2)?
                } else {
                    k::matmul(&av.t()?.contiguous(), g)?
                };
                res.push((n, k::unbroadcast(&gb, &bsh)?));
            }
            Ok(res)
        }),
    );
    Ok(Var { value: out, node: Some(node) })
}

macro_rules! unary_grad {
    ($name:ident, $label:literal, $fwd:expr, $back:expr) => {
        pub fn $name(a: &Var) -> TensorResult<Var> {
            let f: fn(&Tensor) -> TensorResult<Tensor> = $fwd;
            let out = f(&a.value)?;
            let av = a.value.clone();
            let ov = out.clone();
            let b: fn(&Tensor, &Tensor, &Tensor) -> TensorResult<Tensor> = $back;
            a.record(None, $label, out, Box::new(move |g: &Tensor| {
                Ok(vec![(0usize, b(g, &av, &ov)?)])
            }))
            .fix_parent(a)
        }
    };
}

impl Var {
    /// The macro above emits parent index 0; bind it to the real node id.
    fn fix_parent(self, a: &Var) -> TensorResult<Var> {
        if let (Some(node), Some(pn)) = (self.node, a.node) {
            TAPE.with(|t| {
                let mut t = t.borrow_mut();
                if let Some(n) = t.nodes.get_mut(node) {
                    n.parents = vec![pn];
                }
            });
        }
        Ok(self)
    }
}

unary_grad!(neg, "neg", |x| k::neg(x), |g, _a, _o| k::neg(g));
unary_grad!(exp, "exp", |x| k::exp(x), |g, _a, o| k::mul(g, o));
unary_grad!(ln, "ln", |x| k::ln(x), |g, a, _o| k::div(g, a));
unary_grad!(sqrt, "sqrt", |x| k::sqrt(x), |g, _a, o| {
    let d = k::scalar_mul(o, 2.0)?;
    k::div(g, &d)
});
unary_grad!(abs, "abs", |x| k::abs(x), |g, a, _o| {
    let s = k::unary_op(a, |v| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 })?;
    k::mul(g, &s)
});
unary_grad!(relu, "relu", |x| k::relu(x), |g, a, _o| {
    let m = k::unary_op(a, |v| if v > 0.0 { 1.0 } else { 0.0 })?;
    k::mul(g, &m)
});
unary_grad!(sigmoid, "sigmoid", |x| k::sigmoid(x), |g, _a, o| {
    // s' = s (1 - s)
    let one_minus = k::unary_op(o, |v| 1.0 - v)?;
    let d = k::mul(o, &one_minus)?;
    k::mul(g, &d)
});
unary_grad!(tanh, "tanh", |x| k::tanh(x), |g, _a, o| {
    // t' = 1 - t^2
    let d = k::unary_op(o, |v| 1.0 - v * v)?;
    k::mul(g, &d)
});

/// Softmax over the last axis. Backward uses the Jacobian-vector identity
/// dL/dx = s * (g - sum(g * s)) so no Jacobian is materialized.
pub fn softmax(a: &Var) -> TensorResult<Var> {
    let out = k::softmax_last(&a.value)?;
    let ov = out.clone();
    let axis = out.rank().saturating_sub(1);
    a.record(
        None,
        "softmax",
        out,
        Box::new(move |g: &Tensor| {
            let gs = k::mul(g, &ov)?;
            let s = k::sum_axis(&gs, axis, true)?;
            let inner = k::sub(g, &s)?;
            Ok(vec![(0usize, k::mul(&ov, &inner)?)])
        }),
    )
    .fix_parent(a)
}

/// log-softmax over the last axis (FR-AI-009 stability).
pub fn log_softmax(a: &Var) -> TensorResult<Var> {
    let out = k::log_softmax_last(&a.value)?;
    let ov = out.clone();
    let axis = out.rank().saturating_sub(1);
    a.record(
        None,
        "log_softmax",
        out,
        Box::new(move |g: &Tensor| {
            // dL/dx = g - softmax(x) * sum(g)
            let s = k::sum_axis(g, axis, true)?;
            let p = k::exp(&ov)?;
            let corr = k::mul(&p, &s)?;
            Ok(vec![(0usize, k::sub(g, &corr)?)])
        }),
    )
    .fix_parent(a)
}

pub fn scalar_mul(a: &Var, c: f32) -> TensorResult<Var> {
    let out = k::scalar_mul(&a.value, c)?;
    a.record(
        None,
        "scalar_mul",
        out,
        Box::new(move |g: &Tensor| Ok(vec![(0usize, k::scalar_mul(g, c)?)])),
    )
    .fix_parent(a)
}

pub fn add_scalar(a: &Var, c: f32) -> TensorResult<Var> {
    let out = k::add_scalar(&a.value, c)?;
    a.record(
        None,
        "add_scalar",
        out,
        Box::new(move |g: &Tensor| Ok(vec![(0usize, g.clone())])),
    )
    .fix_parent(a)
}

pub fn powf(a: &Var, p: f32) -> TensorResult<Var> {
    let out = k::unary_op(&a.value, |x| x.powf(p))?;
    let av = a.value.clone();
    a.record(
        None,
        "powf",
        out,
        Box::new(move |g: &Tensor| {
            let d = k::unary_op(&av, |x| p * x.powf(p - 1.0))?;
            Ok(vec![(0usize, k::mul(g, &d)?)])
        }),
    )
    .fix_parent(a)
}

pub fn sum(a: &Var) -> TensorResult<Var> {
    let out = k::sum_all(&a.value)?;
    let shape = a.value.shape.clone();
    a.record(
        None,
        "sum",
        out,
        Box::new(move |g: &Tensor| {
            let v = g.item()?;
            Ok(vec![(0usize, Tensor::full(&shape, v)?)])
        }),
    )
    .fix_parent(a)
}

pub fn mean(a: &Var) -> TensorResult<Var> {
    let out = k::mean_all(&a.value)?;
    let shape = a.value.shape.clone();
    let n = a.value.numel().max(1) as f32;
    a.record(
        None,
        "mean",
        out,
        Box::new(move |g: &Tensor| {
            let v = g.item()? / n;
            Ok(vec![(0usize, Tensor::full(&shape, v)?)])
        }),
    )
    .fix_parent(a)
}

pub fn sum_axis(a: &Var, axis: usize, keepdims: bool) -> TensorResult<Var> {
    let out = k::sum_axis(&a.value, axis, keepdims)?;
    let shape = a.value.shape.clone();
    a.record(
        None,
        "sum_axis",
        out,
        Box::new(move |g: &Tensor| {
            let g2 = if keepdims { g.clone() } else { g.unsqueeze(axis)? };
            let expanded = k::mul(&g2, &Tensor::ones(&shape)?)?;
            Ok(vec![(0usize, expanded)])
        }),
    )
    .fix_parent(a)
}

pub fn mean_axis(a: &Var, axis: usize, keepdims: bool) -> TensorResult<Var> {
    let out = k::mean_axis(&a.value, axis, keepdims)?;
    let shape = a.value.shape.clone();
    let n = a.value.dim(axis).max(1) as f32;
    a.record(
        None,
        "mean_axis",
        out,
        Box::new(move |g: &Tensor| {
            let g2 = if keepdims { g.clone() } else { g.unsqueeze(axis)? };
            let scaled = k::scalar_mul(&g2, 1.0 / n)?;
            let expanded = k::mul(&scaled, &Tensor::ones(&shape)?)?;
            Ok(vec![(0usize, expanded)])
        }),
    )
    .fix_parent(a)
}

pub fn reshape(a: &Var, dims: &[isize]) -> TensorResult<Var> {
    let out = a.value.reshape(dims)?;
    let shape: Vec<isize> = a.value.shape.iter().map(|&d| d as isize).collect();
    a.record(
        None,
        "reshape",
        out,
        Box::new(move |g: &Tensor| Ok(vec![(0usize, g.reshape(&shape)?)])),
    )
    .fix_parent(a)
}

pub fn transpose(a: &Var, i: usize, j: usize) -> TensorResult<Var> {
    let out = a.value.transpose(i, j)?.contiguous();
    a.record(
        None,
        "transpose",
        out,
        Box::new(move |g: &Tensor| Ok(vec![(0usize, g.transpose(i, j)?.contiguous())])),
    )
    .fix_parent(a)
}

/// Dropout — FR-AI-019. The mask is drawn by the caller so the global RNG
/// stays the single source of randomness (FR-RT-010).
pub fn dropout_with_mask(a: &Var, mask: Tensor) -> TensorResult<Var> {
    let out = k::mul(&a.value, &mask)?;
    let m = mask.clone();
    a.record(
        None,
        "dropout",
        out,
        Box::new(move |g: &Tensor| Ok(vec![(0usize, k::mul(g, &m)?)])),
    )
    .fix_parent(a)
}

// ============================ backward pass ============================

/// FR-AI-003: replay the tape in reverse from `root`, accumulating gradients.
/// Returns a table indexed by node id.
pub fn backward(root: usize, seed: Option<Tensor>) -> TensorResult<Vec<Option<Tensor>>> {
    let n_nodes = TAPE.with(|t| t.borrow().nodes.len());
    if root >= n_nodes {
        return Ok(vec![None; n_nodes]);
    }
    let mut grads: Vec<Option<Tensor>> = vec![None; n_nodes];

    let root_shape = TAPE.with(|t| t.borrow().nodes[root].shape.clone());
    grads[root] = Some(match seed {
        Some(s) => s,
        None => Tensor::ones(&root_shape)?,
    });

    // Nodes are appended in topological order, so a reverse index walk is a
    // valid reverse topological traversal. No sort is needed.
    for id in (0..=root).rev() {
        let g = match grads[id].clone() {
            Some(g) => g,
            None => continue,
        };
        let contributions = TAPE.with(|t| {
            let tape = t.borrow();
            let node = &tape.nodes[id];
            match &node.backward {
                None => Ok(Vec::new()), // leaf
                Some(f) => f(&g),
            }
        })?;
        if contributions.is_empty() {
            continue;
        }
        let parents = TAPE.with(|t| t.borrow().nodes[id].parents.clone());
        for (slot, (idx, contrib)) in contributions.into_iter().enumerate() {
            // Unary rules report slot 0; binary rules report real node ids.
            let target = if idx < n_nodes && parents.contains(&idx) {
                idx
            } else {
                match parents.get(slot) {
                    Some(&p) => p,
                    None => continue,
                }
            };
            grads[target] = Some(match grads[target].take() {
                None => contrib,
                Some(existing) => k::add(&existing, &contrib)?,
            });
        }
    }
    Ok(grads)
}

/// Convenience for tests and the `check` module.
pub fn grad_of(var: &Var, wrt: &Var) -> TensorResult<Option<Tensor>> {
    let (Some(root), Some(target)) = (var.node, wrt.node) else {
        return Ok(None);
    };
    let g = backward(root, None)?;
    Ok(g.get(target).cloned().flatten())
}

/// A fresh tape scope: enables tracking, runs `f`, then clears (FR-AI-005).
pub fn with_tape<T>(f: impl FnOnce() -> T) -> T {
    clear_tape();
    let prev = set_tracking(true);
    let out = f();
    set_tracking(prev);
    out
}

/// Report whether every element of a gradient is finite (FR-AI-018).
pub fn all_finite(t: &Tensor) -> bool {
    t.to_vec().iter().all(|v| v.is_finite())
}

/// Shared handle to a trainable parameter.
#[derive(Clone, Debug)]
pub struct Param {
    pub value: Rc<RefCell<Tensor>>,
    pub name: String,
    /// Node id issued during the current forward pass.
    pub node: Rc<RefCell<Option<usize>>>,
}

impl Param {
    pub fn new(name: impl Into<String>, value: Tensor) -> Param {
        Param {
            value: Rc::new(RefCell::new(value)),
            name: name.into(),
            node: Rc::new(RefCell::new(None)),
        }
    }
    /// Register as a leaf for this forward pass and return the tracked Var.
    pub fn var(&self) -> Var {
        let v = self.value.borrow().clone();
        if is_tracking() {
            let id = leaf(&v.shape);
            *self.node.borrow_mut() = Some(id);
            Var { value: v, node: Some(id) }
        } else {
            *self.node.borrow_mut() = None;
            Var::constant(v)
        }
    }
    pub fn shape(&self) -> Vec<usize> {
        self.value.borrow().shape.clone()
    }
    pub fn numel(&self) -> usize {
        self.value.borrow().numel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(d: Vec<f32>, s: Vec<usize>) -> Tensor {
        Tensor::from_vec(d, s).unwrap()
    }

    #[test]
    fn add_gradient_is_one() {
        with_tape(|| {
            let a = Var::parameter(t(vec![1.0, 2.0], vec![2]));
            let b = Var::parameter(t(vec![3.0, 4.0], vec![2]));
            let c = add(&a, &b).unwrap();
            let s = sum(&c).unwrap();
            let g = backward(s.node.unwrap(), None).unwrap();
            assert_eq!(g[a.node.unwrap()].as_ref().unwrap().to_vec(), vec![1.0, 1.0]);
            assert_eq!(g[b.node.unwrap()].as_ref().unwrap().to_vec(), vec![1.0, 1.0]);
        });
    }

    #[test]
    fn mul_gradient_is_the_other_operand() {
        with_tape(|| {
            let a = Var::parameter(t(vec![2.0, 3.0], vec![2]));
            let b = Var::parameter(t(vec![5.0, 7.0], vec![2]));
            let c = mul(&a, &b).unwrap();
            let s = sum(&c).unwrap();
            let g = backward(s.node.unwrap(), None).unwrap();
            assert_eq!(g[a.node.unwrap()].as_ref().unwrap().to_vec(), vec![5.0, 7.0]);
            assert_eq!(g[b.node.unwrap()].as_ref().unwrap().to_vec(), vec![2.0, 3.0]);
        });
    }

    #[test]
    fn gradient_accumulates_on_reuse() {
        // y = x * x, dy/dx = 2x — requires accumulation at the shared leaf.
        with_tape(|| {
            let x = Var::parameter(t(vec![3.0], vec![1]));
            let y = mul(&x, &x).unwrap();
            let s = sum(&y).unwrap();
            let g = backward(s.node.unwrap(), None).unwrap();
            assert_eq!(g[x.node.unwrap()].as_ref().unwrap().to_vec(), vec![6.0]);
        });
    }

    #[test]
    fn broadcast_gradient_sums_back_down() {
        with_tape(|| {
            let a = Var::parameter(t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
            let b = Var::parameter(t(vec![1.0, 1.0, 1.0], vec![3]));
            let c = add(&a, &b).unwrap();
            let s = sum(&c).unwrap();
            let g = backward(s.node.unwrap(), None).unwrap();
            // b was broadcast across 2 rows, so each element sees gradient 2.
            assert_eq!(g[b.node.unwrap()].as_ref().unwrap().to_vec(), vec![2.0, 2.0, 2.0]);
        });
    }

    #[test]
    fn matmul_gradient_shapes_are_right() {
        with_tape(|| {
            let a = Var::parameter(t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
            let b = Var::parameter(t(vec![1.0; 12], vec![3, 4]));
            let c = matmul(&a, &b).unwrap();
            assert_eq!(c.shape(), &[2, 4]);
            let s = sum(&c).unwrap();
            let g = backward(s.node.unwrap(), None).unwrap();
            assert_eq!(g[a.node.unwrap()].as_ref().unwrap().shape, vec![2, 3]);
            assert_eq!(g[b.node.unwrap()].as_ref().unwrap().shape, vec![3, 4]);
        });
    }

    #[test]
    fn no_grad_records_nothing() {
        with_tape(|| {
            let a = Var::parameter(t(vec![1.0], vec![1]));
            let before = tape_len();
            no_grad(|| {
                let b = Var::constant(t(vec![2.0], vec![1]));
                let _ = add(&a, &b).unwrap();
            });
            assert_eq!(tape_len(), before);
        });
    }

    #[test]
    fn clear_tape_frees_nodes() {
        with_tape(|| {
            let a = Var::parameter(t(vec![1.0], vec![1]));
            let _ = exp(&a).unwrap();
            assert!(tape_len() > 0);
        });
        clear_tape();
        assert_eq!(tape_len(), 0);
    }
}
