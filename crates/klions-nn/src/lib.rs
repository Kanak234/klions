//! KLIONS neural-network library — FR-AI-007 … FR-AI-019.
//!
//! Layers, losses, optimizers, and datasets. No external ML dependency
//! (DC-007): everything here sits directly on `klions-tensor` and
//! `klions-autodiff`.

pub mod data;
pub mod loss;
pub mod optim;
pub mod serialize;

use klions_autodiff::{self as ad, Param, Var};
use klions_tensor::{Rng, Tensor, TensorError, TensorResult};

/// FR-AI-007: the layer set available in v0.1.0.
#[derive(Clone, Debug)]
pub enum Layer {
    Dense {
        inputs: usize,
        outputs: usize,
        activation: Activation,
        w: Param,
        b: Param,
    },
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Flatten,
    Dropout {
        p: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}

impl Activation {
    pub fn parse(s: &str) -> Option<Activation> {
        Some(match s {
            "none" | "linear" | "identity" => Activation::None,
            "relu" => Activation::ReLU,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Activation::None => "none",
            Activation::ReLU => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Softmax => "softmax",
        }
    }
    pub fn apply(&self, x: &Var) -> TensorResult<Var> {
        match self {
            Activation::None => Ok(x.clone()),
            Activation::ReLU => ad::relu(x),
            Activation::Sigmoid => ad::sigmoid(x),
            Activation::Tanh => ad::tanh(x),
            Activation::Softmax => ad::softmax(x),
        }
    }
    pub const ALL: &'static [&'static str] =
        &["none", "relu", "sigmoid", "tanh", "softmax"];
}

impl Layer {
    /// FR-AI-011: Glorot uniform weights, zero biases.
    pub fn dense(
        name: &str,
        inputs: usize,
        outputs: usize,
        activation: Activation,
        rng: &mut Rng,
    ) -> TensorResult<Layer> {
        if inputs == 0 || outputs == 0 {
            return Err(TensorError::BadArgument(format!(
                "Dense layer needs positive sizes, got inputs={} outputs={}",
                inputs, outputs
            )));
        }
        let w = Tensor::glorot_uniform(inputs, outputs, rng)?;
        let b = Tensor::zeros(&[outputs])?;
        Ok(Layer::Dense {
            inputs,
            outputs,
            activation,
            w: Param::new(format!("{}.weight", name), w),
            b: Param::new(format!("{}.bias", name), b),
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Layer::Dense { .. } => "Dense",
            Layer::ReLU => "ReLU",
            Layer::Sigmoid => "Sigmoid",
            Layer::Tanh => "Tanh",
            Layer::Softmax => "Softmax",
            Layer::Flatten => "Flatten",
            Layer::Dropout { .. } => "Dropout",
        }
    }

    /// Declared input width, where the layer has one (FR-TYP-012).
    pub fn input_size(&self) -> Option<usize> {
        match self {
            Layer::Dense { inputs, .. } => Some(*inputs),
            _ => None,
        }
    }
    pub fn output_size(&self) -> Option<usize> {
        match self {
            Layer::Dense { outputs, .. } => Some(*outputs),
            _ => None,
        }
    }

    pub fn params(&self) -> Vec<Param> {
        match self {
            Layer::Dense { w, b, .. } => vec![w.clone(), b.clone()],
            _ => Vec::new(),
        }
    }

    /// One layer's forward pass. `training` gates Dropout (FR-AI-019).
    pub fn forward(&self, x: &Var, training: bool, rng: &mut Rng) -> TensorResult<Var> {
        match self {
            Layer::Dense { w, b, activation, inputs, .. } => {
                let xin = if x.value.rank() == 1 {
                    ad::reshape(x, &[1, x.value.numel() as isize])?
                } else if x.value.rank() > 2 {
                    ad::reshape(x, &[x.value.dim(0) as isize, -1])?
                } else {
                    x.clone()
                };
                let width = xin.value.dim(xin.value.rank() - 1);
                if width != *inputs {
                    return Err(TensorError::ShapeMismatch {
                        op: "Dense".to_string(),
                        lhs: xin.value.shape.clone(),
                        rhs: vec![*inputs],
                        axis: Some(xin.value.rank() - 1),
                    });
                }
                let h = ad::matmul(&xin, &w.var())?;
                let h = ad::add(&h, &b.var())?;
                activation.apply(&h)
            }
            Layer::ReLU => ad::relu(x),
            Layer::Sigmoid => ad::sigmoid(x),
            Layer::Tanh => ad::tanh(x),
            Layer::Softmax => ad::softmax(x),
            Layer::Flatten => {
                if x.value.rank() <= 2 {
                    Ok(x.clone())
                } else {
                    ad::reshape(x, &[x.value.dim(0) as isize, -1])
                }
            }
            Layer::Dropout { p } => {
                // FR-AI-019: active only while training. Inverted scaling keeps
                // the expected activation constant, so inference needs no change.
                if !training || *p <= 0.0 {
                    return Ok(x.clone());
                }
                let keep = 1.0 - *p;
                let n = x.value.numel();
                let mask: Vec<f32> = (0..n)
                    .map(|_| if rng.bernoulli(keep) { 1.0 / keep } else { 0.0 })
                    .collect();
                let mask = Tensor::from_vec(mask, x.value.shape.clone())?;
                ad::dropout_with_mask(x, mask)
            }
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Layer::Dense { inputs, outputs, activation, .. } => format!(
                "Dense(inputs={}, outputs={}, activation=\"{}\")",
                inputs,
                outputs,
                activation.as_str()
            ),
            Layer::Dropout { p } => format!("Dropout(p={})", p),
            other => format!("{}()", other.kind()),
        }
    }
}

/// A model: an ordered chain of named layers (FR-PAR-006).
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub layers: Vec<(String, Layer)>,
}

impl Model {
    pub fn new(name: impl Into<String>) -> Model {
        Model { name: name.into(), layers: Vec::new() }
    }

    pub fn push(&mut self, name: impl Into<String>, layer: Layer) {
        self.layers.push((name.into(), layer));
    }

    pub fn params(&self) -> Vec<Param> {
        self.layers.iter().flat_map(|(_, l)| l.params()).collect()
    }

    pub fn param_count(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }

    /// The width the first Dense layer expects (FR-TYP-013).
    pub fn input_width(&self) -> Option<usize> {
        self.layers.iter().find_map(|(_, l)| l.input_size())
    }
    pub fn output_width(&self) -> Option<usize> {
        self.layers.iter().rev().find_map(|(_, l)| l.output_size())
    }

    /// FR-AI-016: run the chain in declaration order.
    pub fn forward(&self, x: &Var, training: bool, rng: &mut Rng) -> TensorResult<Var> {
        let mut h = x.clone();
        for (_, layer) in &self.layers {
            h = layer.forward(&h, training, rng)?;
        }
        Ok(h)
    }

    /// Inference on plain tensors, with tracking suspended (FR-AI-004).
    pub fn predict(&self, x: &Tensor, rng: &mut Rng) -> TensorResult<Tensor> {
        ad::no_grad(|| {
            let v = Var::constant(x.clone());
            Ok(self.forward(&v, false, rng)?.value)
        })
    }

    pub fn summary(&self) -> String {
        let mut s = format!("model {} {{\n", self.name);
        for (n, l) in &self.layers {
            let p: usize = l.params().iter().map(|p| p.numel()).sum();
            s.push_str(&format!("    {:<10} {:<52} {:>10} params\n", n, l.describe(), p));
        }
        s.push_str(&format!("}}\ntotal parameters: {}\n", self.param_count()));
        s
    }
}

/// Result of `model.evaluate(dataset)` — FR-AI-016.
#[derive(Clone, Debug, Default)]
pub struct EvalResult {
    pub loss: f32,
    pub accuracy: f32,
    pub samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_shapes_flow_through() {
        let mut rng = Rng::new(42);
        let mut m = Model::new("Net");
        m.push("h", Layer::dense("h", 4, 8, Activation::ReLU, &mut rng).unwrap());
        m.push("o", Layer::dense("o", 8, 3, Activation::None, &mut rng).unwrap());
        let x = Tensor::zeros(&[5, 4]).unwrap();
        let y = m.predict(&x, &mut rng).unwrap();
        assert_eq!(y.shape, vec![5, 3]);
        assert_eq!(m.input_width(), Some(4));
        assert_eq!(m.output_width(), Some(3));
    }

    #[test]
    fn glorot_init_is_in_range_and_bias_is_zero() {
        let mut rng = Rng::new(1);
        let l = Layer::dense("d", 100, 50, Activation::None, &mut rng).unwrap();
        if let Layer::Dense { w, b, .. } = &l {
            let limit = (6.0f32 / 150.0).sqrt();
            for v in w.value.borrow().to_vec() {
                assert!(v.abs() <= limit + 1e-6, "{} outside +/-{}", v, limit);
            }
            assert!(b.value.borrow().to_vec().iter().all(|&v| v == 0.0));
        } else {
            panic!("expected a Dense layer");
        }
    }

    #[test]
    fn param_count_is_right() {
        let mut rng = Rng::new(2);
        let mut m = Model::new("M");
        m.push("a", Layer::dense("a", 784, 128, Activation::ReLU, &mut rng).unwrap());
        m.push("b", Layer::dense("b", 128, 10, Activation::None, &mut rng).unwrap());
        assert_eq!(m.param_count(), 784 * 128 + 128 + 128 * 10 + 10);
    }

    #[test]
    fn dropout_is_inactive_during_inference() {
        let mut rng = Rng::new(3);
        let d = Layer::Dropout { p: 0.9 };
        let x = Tensor::ones(&[10, 10]).unwrap();
        let out = ad::no_grad(|| d.forward(&Var::constant(x.clone()), false, &mut rng)).unwrap();
        assert!(out.value.to_vec().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn dropout_zeroes_some_units_during_training() {
        let mut rng = Rng::new(4);
        let d = Layer::Dropout { p: 0.5 };
        let x = Tensor::ones(&[40, 40]).unwrap();
        let out = ad::no_grad(|| d.forward(&Var::constant(x.clone()), true, &mut rng)).unwrap();
        let zeros = out.value.to_vec().iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 500 && zeros < 1100, "dropped {} of 1600", zeros);
    }

    #[test]
    fn flatten_collapses_trailing_axes() {
        let mut rng = Rng::new(5);
        let f = Layer::Flatten;
        let x = Tensor::zeros(&[7, 1, 28, 28]).unwrap();
        let out = ad::no_grad(|| f.forward(&Var::constant(x), false, &mut rng)).unwrap();
        assert_eq!(out.value.shape, vec![7, 784]);
    }

    #[test]
    fn dense_reports_width_mismatch() {
        let mut rng = Rng::new(6);
        let l = Layer::dense("d", 4, 2, Activation::None, &mut rng).unwrap();
        let x = Tensor::zeros(&[3, 9]).unwrap();
        let e = ad::no_grad(|| l.forward(&Var::constant(x), false, &mut rng));
        assert!(matches!(e, Err(TensorError::ShapeMismatch { .. })));
    }
}
