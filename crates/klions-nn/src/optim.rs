//! Optimizers — FR-AI-010. Defaults match the published papers.
//!
//! Adam: Kingma & Ba, ICLR 2015 — lr 0.001, beta1 0.9, beta2 0.999, eps 1e-8.
//! RMSProp: Hinton's lecture 6e — decay 0.9, eps 1e-8.
//! SGD: momentum 0.0 unless given.

use klions_autodiff::Param;
use klions_tensor::kernels as k;
use klions_tensor::{Tensor, TensorResult};

#[derive(Clone, Debug)]
pub enum OptimizerKind {
    Sgd { learning_rate: f32, momentum: f32 },
    Adam { learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32 },
    RmsProp { learning_rate: f32, decay: f32, epsilon: f32 },
}

impl OptimizerKind {
    pub fn name(&self) -> &'static str {
        match self {
            OptimizerKind::Sgd { .. } => "SGD",
            OptimizerKind::Adam { .. } => "Adam",
            OptimizerKind::RmsProp { .. } => "RMSProp",
        }
    }
    pub fn learning_rate(&self) -> f32 {
        match self {
            OptimizerKind::Sgd { learning_rate, .. }
            | OptimizerKind::Adam { learning_rate, .. }
            | OptimizerKind::RmsProp { learning_rate, .. } => *learning_rate,
        }
    }
    pub fn describe(&self) -> String {
        match self {
            OptimizerKind::Sgd { learning_rate, momentum } => {
                format!("SGD(learning_rate={}, momentum={})", learning_rate, momentum)
            }
            OptimizerKind::Adam { learning_rate, beta1, beta2, epsilon } => format!(
                "Adam(learning_rate={}, beta1={}, beta2={}, epsilon={})",
                learning_rate, beta1, beta2, epsilon
            ),
            OptimizerKind::RmsProp { learning_rate, decay, epsilon } => format!(
                "RMSProp(learning_rate={}, decay={}, epsilon={})",
                learning_rate, decay, epsilon
            ),
        }
    }
}

pub const OPTIMIZER_NAMES: &[&str] = &["SGD", "Adam", "RMSProp"];

pub fn default_sgd() -> OptimizerKind {
    OptimizerKind::Sgd { learning_rate: 0.01, momentum: 0.0 }
}
pub fn default_adam() -> OptimizerKind {
    OptimizerKind::Adam {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    }
}
pub fn default_rmsprop() -> OptimizerKind {
    OptimizerKind::RmsProp { learning_rate: 0.001, decay: 0.9, epsilon: 1e-8 }
}

/// Per-parameter optimizer state, allocated lazily on the first step.
pub struct Optimizer {
    pub kind: OptimizerKind,
    /// First moment / momentum buffer.
    m: Vec<Option<Tensor>>,
    /// Second moment buffer.
    v: Vec<Option<Tensor>>,
    step_count: u64,
}

impl Optimizer {
    pub fn new(kind: OptimizerKind) -> Optimizer {
        Optimizer { kind, m: Vec::new(), v: Vec::new(), step_count: 0 }
    }

    pub fn steps(&self) -> u64 {
        self.step_count
    }

    fn ensure(&mut self, n: usize) {
        if self.m.len() < n {
            self.m.resize(n, None);
            self.v.resize(n, None);
        }
    }

    /// Apply one update. `grads[i]` corresponds to `params[i]`.
    pub fn step(&mut self, params: &[Param], grads: &[Option<Tensor>]) -> TensorResult<()> {
        self.ensure(params.len());
        self.step_count += 1;
        let t = self.step_count as f32;

        for (i, p) in params.iter().enumerate() {
            let g = match grads.get(i).and_then(|g| g.clone()) {
                Some(g) => g,
                None => continue, // parameter not reached by this loss
            };
            let current = p.value.borrow().clone();
            let updated = match self.kind {
                OptimizerKind::Sgd { learning_rate, momentum } => {
                    if momentum == 0.0 {
                        k::sub(&current, &k::scalar_mul(&g, learning_rate)?)?
                    } else {
                        // v = mu*v + g ; theta = theta - lr*v
                        let prev = match &self.m[i] {
                            Some(b) => b.clone(),
                            None => Tensor::zeros(&g.shape)?,
                        };
                        let buf = k::add(&k::scalar_mul(&prev, momentum)?, &g)?;
                        let out = k::sub(&current, &k::scalar_mul(&buf, learning_rate)?)?;
                        self.m[i] = Some(buf);
                        out
                    }
                }
                OptimizerKind::Adam { learning_rate, beta1, beta2, epsilon } => {
                    let m_prev = match &self.m[i] {
                        Some(b) => b.clone(),
                        None => Tensor::zeros(&g.shape)?,
                    };
                    let v_prev = match &self.v[i] {
                        Some(b) => b.clone(),
                        None => Tensor::zeros(&g.shape)?,
                    };
                    // m = b1*m + (1-b1)*g
                    let m_new = k::add(
                        &k::scalar_mul(&m_prev, beta1)?,
                        &k::scalar_mul(&g, 1.0 - beta1)?,
                    )?;
                    // v = b2*v + (1-b2)*g^2
                    let g2 = k::mul(&g, &g)?;
                    let v_new = k::add(
                        &k::scalar_mul(&v_prev, beta2)?,
                        &k::scalar_mul(&g2, 1.0 - beta2)?,
                    )?;
                    // Bias correction, exactly as in the paper.
                    let mc = 1.0 - beta1.powf(t);
                    let vc = 1.0 - beta2.powf(t);
                    let m_hat = k::scalar_mul(&m_new, 1.0 / mc)?;
                    let v_hat = k::scalar_mul(&v_new, 1.0 / vc)?;
                    let denom = k::add_scalar(&k::sqrt(&v_hat)?, epsilon)?;
                    let update = k::div(&m_hat, &denom)?;
                    let out = k::sub(&current, &k::scalar_mul(&update, learning_rate)?)?;
                    self.m[i] = Some(m_new);
                    self.v[i] = Some(v_new);
                    out
                }
                OptimizerKind::RmsProp { learning_rate, decay, epsilon } => {
                    let v_prev = match &self.v[i] {
                        Some(b) => b.clone(),
                        None => Tensor::zeros(&g.shape)?,
                    };
                    let g2 = k::mul(&g, &g)?;
                    let v_new = k::add(
                        &k::scalar_mul(&v_prev, decay)?,
                        &k::scalar_mul(&g2, 1.0 - decay)?,
                    )?;
                    let denom = k::add_scalar(&k::sqrt(&v_new)?, epsilon)?;
                    let update = k::div(&g, &denom)?;
                    let out = k::sub(&current, &k::scalar_mul(&update, learning_rate)?)?;
                    self.v[i] = Some(v_new);
                    out
                }
            };
            *p.value.borrow_mut() = updated;
        }
        Ok(())
    }

    /// Drop the moment buffers — used when a model is re-trained from scratch.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(v: f32) -> Param {
        Param::new("p", Tensor::from_vec(vec![v], vec![1]).unwrap())
    }
    fn grad(v: f32) -> Option<Tensor> {
        Some(Tensor::from_vec(vec![v], vec![1]).unwrap())
    }

    #[test]
    fn sgd_moves_against_the_gradient() {
        let p = param(1.0);
        let mut o = Optimizer::new(OptimizerKind::Sgd { learning_rate: 0.1, momentum: 0.0 });
        o.step(&[p.clone()], &[grad(2.0)]).unwrap();
        // 1.0 - 0.1 * 2.0 = 0.8
        assert!((p.value.borrow().to_vec()[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sgd_momentum_accumulates() {
        let p = param(0.0);
        let mut o = Optimizer::new(OptimizerKind::Sgd { learning_rate: 1.0, momentum: 0.5 });
        o.step(&[p.clone()], &[grad(1.0)]).unwrap(); // v=1   -> -1
        o.step(&[p.clone()], &[grad(1.0)]).unwrap(); // v=1.5 -> -2.5
        assert!((p.value.borrow().to_vec()[0] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn adam_first_step_is_approximately_the_learning_rate() {
        // With bias correction, step 1 moves by ~lr regardless of gradient scale.
        let p = param(0.0);
        let mut o = Optimizer::new(default_adam());
        o.step(&[p.clone()], &[grad(37.0)]).unwrap();
        let moved = p.value.borrow().to_vec()[0];
        assert!((moved + 0.001).abs() < 1e-5, "moved {}", moved);
    }

    #[test]
    fn adam_defaults_match_the_paper() {
        match default_adam() {
            OptimizerKind::Adam { learning_rate, beta1, beta2, epsilon } => {
                assert_eq!((learning_rate, beta1, beta2), (0.001, 0.9, 0.999));
                assert!((epsilon - 1e-8).abs() < 1e-12);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rmsprop_normalizes_the_step() {
        let p = param(0.0);
        let mut o = Optimizer::new(OptimizerKind::RmsProp {
            learning_rate: 0.1,
            decay: 0.9,
            epsilon: 1e-8,
        });
        o.step(&[p.clone()], &[grad(100.0)]).unwrap();
        // g / sqrt(0.1*g^2) = 1/sqrt(0.1) ≈ 3.162, times lr 0.1
        let moved = -p.value.borrow().to_vec()[0];
        assert!((moved - 0.3162).abs() < 1e-3, "moved {}", moved);
    }

    #[test]
    fn missing_gradient_leaves_the_parameter_alone() {
        let p = param(5.0);
        let mut o = Optimizer::new(default_adam());
        o.step(&[p.clone()], &[None]).unwrap();
        assert_eq!(p.value.borrow().to_vec()[0], 5.0);
    }

    #[test]
    fn a_quadratic_converges_to_its_minimum() {
        // minimize (x - 3)^2 with Adam; gradient is 2(x-3)
        let p = param(0.0);
        let mut o = Optimizer::new(OptimizerKind::Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
        for _ in 0..600 {
            let x = p.value.borrow().to_vec()[0];
            o.step(&[p.clone()], &[grad(2.0 * (x - 3.0))]).unwrap();
        }
        let x = p.value.borrow().to_vec()[0];
        assert!((x - 3.0).abs() < 0.05, "converged to {}", x);
    }
}
