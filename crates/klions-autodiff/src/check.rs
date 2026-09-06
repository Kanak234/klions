//! Finite-difference gradient verification — FR-AI-006, release gate G-4.
//!
//! Every differentiable operation must agree with central differences to
//! within `1e-3` relative error on `f32`. This module is the harness; the
//! per-operation cases live in the test module at the bottom.

use crate::{backward, set_tracking, with_tape, Var};
use klions_tensor::{Tensor, TensorResult};

/// Tolerance mandated by FR-AI-006.
pub const RELATIVE_TOLERANCE: f32 = 1e-3;
/// Step size. Large enough that `f32` cancellation does not dominate,
/// small enough that the second-order term stays below tolerance.
pub const EPSILON: f32 = 1e-2;

#[derive(Debug)]
pub struct CheckReport {
    pub op: String,
    pub max_relative_error: f32,
    pub worst_index: usize,
    pub analytic: Vec<f32>,
    pub numeric: Vec<f32>,
}

impl CheckReport {
    pub fn passed(&self) -> bool {
        self.max_relative_error <= RELATIVE_TOLERANCE
    }
}

/// Compare the analytic gradient of `f` at `x` against central differences.
///
/// `f` must map a single input Var to a scalar Var (the "loss").
pub fn check_gradient(
    op: &str,
    x: &Tensor,
    f: impl Fn(&Var) -> TensorResult<Var>,
) -> TensorResult<CheckReport> {
    // ---- analytic ----
    let analytic = with_tape(|| -> TensorResult<Vec<f32>> {
        let v = Var::parameter(x.clone());
        let loss = f(&v)?;
        let root = match loss.node {
            Some(r) => r,
            None => return Ok(vec![0.0; x.numel()]),
        };
        let grads = backward(root, None)?;
        let g = grads[v.node.unwrap()]
            .clone()
            .unwrap_or(Tensor::zeros(&x.shape)?);
        Ok(g.to_vec())
    })?;

    // ---- numeric: central differences, tracking off ----
    let prev = set_tracking(false);
    let base = x.to_vec();
    let mut numeric = Vec::with_capacity(base.len());
    for i in 0..base.len() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        // Scale the step with the magnitude of the coordinate.
        let h = EPSILON * base[i].abs().max(1.0);
        plus[i] += h;
        minus[i] -= h;
        let lp = eval_scalar(&f, &plus, &x.shape)?;
        let lm = eval_scalar(&f, &minus, &x.shape)?;
        numeric.push((lp - lm) / (2.0 * h));
    }
    set_tracking(prev);

    // ---- compare ----
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    for i in 0..analytic.len() {
        let (a, n) = (analytic[i], numeric[i]);
        let denom = a.abs().max(n.abs()).max(1e-3);
        let rel = (a - n).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
            worst = i;
        }
    }
    Ok(CheckReport {
        op: op.to_string(),
        max_relative_error: max_rel,
        worst_index: worst,
        analytic,
        numeric,
    })
}

fn eval_scalar(
    f: &impl Fn(&Var) -> TensorResult<Var>,
    data: &[f32],
    shape: &[usize],
) -> TensorResult<f32> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec())?;
    let v = Var::constant(t);
    f(&v)?.value.item()
}

/// Two-input variant, for binary operations.
pub fn check_gradient2(
    op: &str,
    x: &Tensor,
    y: &Tensor,
    f: impl Fn(&Var, &Var) -> TensorResult<Var>,
) -> TensorResult<(CheckReport, CheckReport)> {
    let yc = y.clone();
    let ra = check_gradient(&format!("{} (lhs)", op), x, |v| {
        f(v, &Var::constant(yc.clone()))
    })?;
    let xc = x.clone();
    let rb = check_gradient(&format!("{} (rhs)", op), y, |v| {
        f(&Var::constant(xc.clone()), v)
    })?;
    Ok((ra, rb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as ad;
    use klions_tensor::kernels as k;
    use klions_tensor::Rng;

    fn sample(shape: &[usize], seed: u64) -> Tensor {
        let mut r = Rng::new(seed);
        Tensor::rand_uniform(shape, -1.5, 1.5, &mut r).unwrap()
    }

    fn positive_sample(shape: &[usize], seed: u64) -> Tensor {
        let mut r = Rng::new(seed);
        Tensor::rand_uniform(shape, 0.3, 2.0, &mut r).unwrap()
    }

    fn assert_ok(r: CheckReport) {
        assert!(
            r.passed(),
            "{}: max relative error {:.3e} exceeds {:.1e} at index {} (analytic {:?}, numeric {:?})",
            r.op,
            r.max_relative_error,
            RELATIVE_TOLERANCE,
            r.worst_index,
            r.analytic.get(r.worst_index),
            r.numeric.get(r.worst_index),
        );
    }

    // ---- unary ops ----

    #[test]
    fn grad_neg() {
        assert_ok(check_gradient("neg", &sample(&[3, 4], 1), |v| {
            ad::sum(&ad::neg(v)?)
        }).unwrap());
    }

    #[test]
    fn grad_exp() {
        assert_ok(check_gradient("exp", &sample(&[3, 4], 2), |v| {
            ad::sum(&ad::exp(v)?)
        }).unwrap());
    }

    #[test]
    fn grad_ln() {
        assert_ok(check_gradient("ln", &positive_sample(&[3, 4], 3), |v| {
            ad::sum(&ad::ln(v)?)
        }).unwrap());
    }

    #[test]
    fn grad_sqrt() {
        assert_ok(check_gradient("sqrt", &positive_sample(&[3, 4], 4), |v| {
            ad::sum(&ad::sqrt(v)?)
        }).unwrap());
    }

    #[test]
    fn grad_relu() {
        // Offset away from zero: the derivative is undefined exactly at 0.
        let mut x = sample(&[4, 4], 5);
        x = k::unary_op(&x, |v| if v.abs() < 0.2 { v + 0.5 } else { v }).unwrap();
        assert_ok(check_gradient("relu", &x, |v| ad::sum(&ad::relu(v)?)).unwrap());
    }

    #[test]
    fn grad_sigmoid() {
        assert_ok(check_gradient("sigmoid", &sample(&[3, 4], 6), |v| {
            ad::sum(&ad::sigmoid(v)?)
        }).unwrap());
    }

    #[test]
    fn grad_tanh() {
        assert_ok(check_gradient("tanh", &sample(&[3, 4], 7), |v| {
            ad::sum(&ad::tanh(v)?)
        }).unwrap());
    }

    #[test]
    fn grad_abs() {
        let mut x = sample(&[4, 4], 8);
        x = k::unary_op(&x, |v| if v.abs() < 0.2 { v + 0.6 } else { v }).unwrap();
        assert_ok(check_gradient("abs", &x, |v| ad::sum(&ad::abs(v)?)).unwrap());
    }

    #[test]
    fn grad_softmax() {
        // Weight the output so the gradient is not uniformly zero.
        let w = Tensor::from_vec(vec![0.3, -0.7, 1.1, 0.5], vec![4]).unwrap();
        assert_ok(check_gradient("softmax", &sample(&[3, 4], 9), |v| {
            let s = ad::softmax(v)?;
            ad::sum(&ad::mul(&s, &Var::constant(w.clone()))?)
        }).unwrap());
    }

    #[test]
    fn grad_log_softmax() {
        let w = Tensor::from_vec(vec![0.3, -0.7, 1.1, 0.5], vec![4]).unwrap();
        assert_ok(check_gradient("log_softmax", &sample(&[3, 4], 10), |v| {
            let s = ad::log_softmax(v)?;
            ad::sum(&ad::mul(&s, &Var::constant(w.clone()))?)
        }).unwrap());
    }

    #[test]
    fn grad_powf() {
        assert_ok(check_gradient("powf", &positive_sample(&[3, 3], 11), |v| {
            ad::sum(&ad::powf(v, 3.0)?)
        }).unwrap());
    }

    #[test]
    fn grad_mean() {
        assert_ok(check_gradient("mean", &sample(&[4, 5], 12), |v| ad::mean(v)).unwrap());
    }

    #[test]
    fn grad_sum_axis() {
        assert_ok(check_gradient("sum_axis", &sample(&[4, 5], 13), |v| {
            ad::sum(&ad::sum_axis(v, 1, false)?)
        }).unwrap());
    }

    #[test]
    fn grad_mean_axis() {
        assert_ok(check_gradient("mean_axis", &sample(&[4, 5], 14), |v| {
            ad::sum(&ad::mean_axis(v, 0, false)?)
        }).unwrap());
    }

    #[test]
    fn grad_reshape() {
        assert_ok(check_gradient("reshape", &sample(&[2, 6], 15), |v| {
            let r = ad::reshape(v, &[3, 4])?;
            ad::sum(&ad::mul(&r, &r)?)
        }).unwrap());
    }

    #[test]
    fn grad_transpose() {
        assert_ok(check_gradient("transpose", &sample(&[3, 5], 16), |v| {
            let tr = ad::transpose(v, 0, 1)?;
            ad::sum(&ad::mul(&tr, &tr)?)
        }).unwrap());
    }

    // ---- binary ops ----

    #[test]
    fn grad_add() {
        let (a, b) = check_gradient2("add", &sample(&[3, 4], 20), &sample(&[3, 4], 21), |x, y| {
            ad::sum(&ad::add(x, y)?)
        }).unwrap();
        assert_ok(a);
        assert_ok(b);
    }

    #[test]
    fn grad_sub() {
        let (a, b) = check_gradient2("sub", &sample(&[3, 4], 22), &sample(&[3, 4], 23), |x, y| {
            ad::sum(&ad::sub(x, y)?)
        }).unwrap();
        assert_ok(a);
        assert_ok(b);
    }

    #[test]
    fn grad_mul() {
        let (a, b) = check_gradient2("mul", &sample(&[3, 4], 24), &sample(&[3, 4], 25), |x, y| {
            ad::sum(&ad::mul(x, y)?)
        }).unwrap();
        assert_ok(a);
        assert_ok(b);
    }

    #[test]
    fn grad_div() {
        let (a, b) = check_gradient2(
            "div",
            &sample(&[3, 4], 26),
            &positive_sample(&[3, 4], 27),
            |x, y| ad::sum(&ad::div(x, y)?),
        ).unwrap();
        assert_ok(a);
        assert_ok(b);
    }

    #[test]
    fn grad_matmul() {
        let (a, b) = check_gradient2("matmul", &sample(&[3, 4], 28), &sample(&[4, 5], 29), |x, y| {
            let m = ad::matmul(x, y)?;
            ad::sum(&ad::mul(&m, &m)?)
        }).unwrap();
        assert_ok(a);
        assert_ok(b);
    }

    #[test]
    fn grad_broadcast_add() {
        let (a, b) = check_gradient2("broadcast add", &sample(&[3, 4], 30), &sample(&[4], 31), |x, y| {
            let s = ad::add(x, y)?;
            ad::sum(&ad::mul(&s, &s)?)
        }).unwrap();
        assert_ok(a);
        assert_ok(b);
    }

    /// A composite that exercises the whole tape at once: a dense layer.
    #[test]
    fn grad_dense_layer_composite() {
        let x = sample(&[4, 6], 40);
        let w = sample(&[6, 3], 41);
        let bias = sample(&[3], 42);
        let target = sample(&[4, 3], 43);
        assert_ok(check_gradient("dense composite (wrt W)", &w, |wv| {
            let h = ad::matmul(&Var::constant(x.clone()), wv)?;
            let h = ad::add(&h, &Var::constant(bias.clone()))?;
            let h = ad::relu(&h)?;
            let d = ad::sub(&h, &Var::constant(target.clone()))?;
            ad::mean(&ad::mul(&d, &d)?)
        }).unwrap());
    }
}
