//! Loss functions — FR-AI-009.
//!
//! `cross_entropy` accepts raw logits and applies log-sum-exp internally, so
//! a preceding `Softmax` layer is neither required nor wanted. That choice is
//! the single most common source of silently-wrong beginner models in other
//! stacks; making it impossible here is deliberate.

use klions_autodiff::{self as ad, Var};
use klions_tensor::kernels as k;
use klions_tensor::{Tensor, TensorError, TensorResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    Mse,
    Mae,
    CrossEntropy,
    BinaryCrossEntropy,
}

pub const LOSS_NAMES: &[&str] = &["mse", "mae", "cross_entropy", "binary_cross_entropy"];

impl Loss {
    pub fn parse(s: &str) -> Option<Loss> {
        Some(match s {
            "mse" | "mean_squared_error" => Loss::Mse,
            "mae" | "mean_absolute_error" => Loss::Mae,
            "cross_entropy" | "categorical_cross_entropy" => Loss::CrossEntropy,
            "binary_cross_entropy" | "bce" => Loss::BinaryCrossEntropy,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Loss::Mse => "mse",
            Loss::Mae => "mae",
            Loss::CrossEntropy => "cross_entropy",
            Loss::BinaryCrossEntropy => "binary_cross_entropy",
        }
    }

    /// Does this loss consume raw logits rather than probabilities?
    pub fn wants_logits(&self) -> bool {
        matches!(self, Loss::CrossEntropy)
    }

    /// `pred` is the model output; `target` is a constant.
    pub fn compute(&self, pred: &Var, target: &Tensor) -> TensorResult<Var> {
        check_shapes(self, &pred.value, target)?;
        match self {
            Loss::Mse => {
                let d = ad::sub(pred, &Var::constant(target.clone()))?;
                ad::mean(&ad::mul(&d, &d)?)
            }
            Loss::Mae => {
                let d = ad::sub(pred, &Var::constant(target.clone()))?;
                ad::mean(&ad::abs(&d)?)
            }
            Loss::CrossEntropy => {
                // -mean over the batch of sum_c target_c * log_softmax(logits)_c
                let logp = ad::log_softmax(pred)?;
                let prod = ad::mul(&logp, &Var::constant(target.clone()))?;
                let per_sample = ad::sum_axis(&prod, prod.value.rank() - 1, false)?;
                let m = ad::mean(&per_sample)?;
                ad::neg(&m)
            }
            Loss::BinaryCrossEntropy => {
                // Clamp keeps log() finite when a prediction saturates.
                const EPS: f32 = 1e-7;
                let p = clamp_var(pred, EPS, 1.0 - EPS)?;
                let t = Var::constant(target.clone());
                let log_p = ad::ln(&p)?;
                let one_minus_p = ad::add_scalar(&ad::neg(&p)?, 1.0)?;
                let log_1mp = ad::ln(&one_minus_p)?;
                let one_minus_t = ad::add_scalar(&ad::neg(&t)?, 1.0)?;
                let a = ad::mul(&t, &log_p)?;
                let b = ad::mul(&one_minus_t, &log_1mp)?;
                let s = ad::add(&a, &b)?;
                ad::neg(&ad::mean(&s)?)
            }
        }
    }

    /// Loss value only, no tape (used for validation and evaluation).
    pub fn compute_value(&self, pred: &Tensor, target: &Tensor) -> TensorResult<f32> {
        ad::no_grad(|| {
            let v = Var::constant(pred.clone());
            Ok(self.compute(&v, target)?.value.item()?)
        })
    }
}

/// Clamp with a straight-through gradient: the value is clamped, and the
/// gradient is 1 inside the range and 0 outside it.
///
/// Written as `x * mask + const(clamp(x) - x * mask)`, so the value works out
/// to `clamp(x)` while only the first term carries a derivative.
fn clamp_var(v: &Var, lo: f32, hi: f32) -> TensorResult<Var> {
    let mask = k::unary_op(&v.value, |x| if x > lo && x < hi { 1.0 } else { 0.0 })?;
    let clamped = k::unary_op(&v.value, |x| x.clamp(lo, hi))?;
    let masked_value = k::mul(&v.value, &mask)?;
    let residual = k::sub(&clamped, &masked_value)?;
    let live = ad::mul(v, &Var::constant(mask))?;
    ad::add(&live, &Var::constant(residual))
}

fn check_shapes(loss: &Loss, pred: &Tensor, target: &Tensor) -> TensorResult<()> {
    if pred.shape != target.shape {
        return Err(TensorError::ShapeMismatch {
            op: format!("{} loss", loss.as_str()),
            lhs: pred.shape.clone(),
            rhs: target.shape.clone(),
            axis: klions_tensor::first_bad_axis(&pred.shape, &target.shape),
        });
    }
    Ok(())
}

/// FR-AI-008 `metrics`: accuracy over a batch of logits or probabilities.
pub fn accuracy(pred: &Tensor, target: &Tensor) -> TensorResult<f32> {
    if pred.rank() < 2 {
        // Binary case: threshold at 0.5.
        let p = pred.to_vec();
        let t = target.to_vec();
        let n = p.len().min(t.len());
        if n == 0 {
            return Ok(0.0);
        }
        let hits = (0..n)
            .filter(|&i| (p[i] >= 0.5) == (t[i] >= 0.5))
            .count();
        return Ok(hits as f32 / n as f32);
    }
    let axis = pred.rank() - 1;
    let pi = k::argmax_axis(pred, axis, false)?.to_vec();
    let ti = k::argmax_axis(target, axis, false)?.to_vec();
    let n = pi.len();
    if n == 0 {
        return Ok(0.0);
    }
    let hits = (0..n).filter(|&i| pi[i] == ti[i]).count();
    Ok(hits as f32 / n as f32)
}

pub const METRIC_NAMES: &[&str] = &["loss", "accuracy"];

#[cfg(test)]
mod tests {
    use super::*;

    fn t(d: Vec<f32>, s: Vec<usize>) -> Tensor {
        Tensor::from_vec(d, s).unwrap()
    }

    #[test]
    fn mse_matches_hand_computation() {
        let p = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y = t(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        // ((0)^2 + 1 + 4 + 9) / 4 = 3.5
        assert!((Loss::Mse.compute_value(&p, &y).unwrap() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn mae_matches_hand_computation() {
        let p = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y = t(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        // (0 + 1 + 2 + 3) / 4 = 1.5
        assert!((Loss::Mae.compute_value(&p, &y).unwrap() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_of_a_confident_correct_prediction_is_near_zero() {
        let logits = t(vec![20.0, 0.0, 0.0], vec![1, 3]);
        let target = t(vec![1.0, 0.0, 0.0], vec![1, 3]);
        let v = Loss::CrossEntropy.compute_value(&logits, &target).unwrap();
        assert!(v < 1e-6, "expected near zero, got {}", v);
    }

    #[test]
    fn cross_entropy_of_a_uniform_prediction_is_ln_classes() {
        let logits = t(vec![0.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let target = t(vec![1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let v = Loss::CrossEntropy.compute_value(&logits, &target).unwrap();
        assert!((v - 4.0f32.ln()).abs() < 1e-5, "got {}", v);
    }

    #[test]
    fn cross_entropy_survives_huge_logits() {
        // The whole point of the internal log-sum-exp (FR-AI-009).
        let logits = t(vec![1e8, -1e8, 5e7], vec![1, 3]);
        let target = t(vec![1.0, 0.0, 0.0], vec![1, 3]);
        let v = Loss::CrossEntropy.compute_value(&logits, &target).unwrap();
        assert!(v.is_finite(), "log-sum-exp overflowed: {}", v);
    }

    #[test]
    fn accuracy_counts_argmax_hits() {
        let p = t(vec![0.1, 0.9, 0.8, 0.2, 0.6, 0.4], vec![3, 2]);
        let y = t(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![3, 2]);
        assert!((accuracy(&p, &y).unwrap() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn loss_shape_mismatch_is_reported() {
        let p = t(vec![0.0; 6], vec![2, 3]);
        let y = t(vec![0.0; 4], vec![2, 2]);
        assert!(matches!(
            Loss::Mse.compute_value(&p, &y),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }
}
