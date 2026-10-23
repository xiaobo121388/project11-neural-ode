//! Dense layers, manual reverse-mode differentiation and mini-batch SGD.
//!
//! Samples are rows. A layer has weights [input, output] and computes Z = A W + b.
//! MSE averages over all batch elements, including the output dimension.

use anyhow::{Result, ensure};
use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Supported elementwise nonlinearities, all implemented without autodiff.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
    Linear,
}

impl Activation {
    /// Numerically stable scalar activation.
    pub fn value(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                }
            }
            Self::Relu => x.max(0.0),
            Self::Tanh => x.tanh(),
            Self::Linear => x,
        }
    }

    /// Derivative with respect to the preactivation; ReLU uses zero at zero.
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                let a = self.value(x);
                a * (1.0 - a)
            }
            Self::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::Linear => 1.0,
        }
    }
}

/// Fully connected layer with row-major serialized ndarray weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub activation: Activation,
}

/// A sequence of dense layers. No physical equations are stored in this type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Dense>,
}

/// Analytical parameter derivatives for one layer and one batch.
#[derive(Debug, Clone)]
pub struct Gradient {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl Network {
    /// Xavier uniform initialization (He uniform for ReLU); linear output layer.
    pub fn new(sizes: &[usize], hidden: Activation, seed: u64) -> Result<Self> {
        ensure!(
            sizes.len() >= 2 && sizes.len() <= 16,
            "Need 2 to 16 layer widths"
        );
        ensure!(
            sizes.iter().all(|&v| (1..=4096).contains(&v)),
            "Invalid layer width"
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(i, s)| {
                let activation = if i == sizes.len() - 2 {
                    Activation::Linear
                } else {
                    hidden
                };
                let limit = if activation == Activation::Relu {
                    (6.0 / s[0] as f64).sqrt()
                } else {
                    (6.0 / (s[0] + s[1]) as f64).sqrt()
                };
                Dense {
                    weights: Array2::from_shape_fn((s[0], s[1]), |_| {
                        rng.random_range(-limit..limit)
                    }),
                    bias: Array1::zeros(s[1]),
                    activation,
                }
            })
            .collect();
        Ok(Self { layers })
    }

    /// Checks a deserialized model before accepting it for training or inference.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.layers.is_empty() && self.layers.len() < 16,
            "Invalid layer count"
        );
        let mut previous = self.layers[0].weights.nrows();
        for layer in &self.layers {
            ensure!(
                layer.weights.nrows() == previous && previous > 0 && previous <= 4096,
                "Layer input dimension mismatch"
            );
            let width = layer.weights.ncols();
            ensure!(
                width > 0 && width <= 4096 && layer.bias.len() == width,
                "Layer bias dimension mismatch"
            );
            ensure!(
                layer
                    .weights
                    .iter()
                    .chain(layer.bias.iter())
                    .all(|v| v.is_finite()),
                "Non-finite network parameter"
            );
            previous = width;
        }
        Ok(())
    }

    fn check_input(&self, x: &Array2<f64>) -> Result<()> {
        ensure!(!self.layers.is_empty(), "Network has no layers");
        ensure!(
            x.nrows() > 0 && x.ncols() == self.layers[0].weights.nrows(),
            "Input shape mismatch"
        );
        ensure!(x.iter().all(|v| v.is_finite()), "Non-finite network input");
        Ok(())
    }

    /// Batched forward propagation. The network must have passed validation.
    pub fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.check_input(x)?;
        let mut a = x.clone();
        for layer in &self.layers {
            ensure!(
                a.ncols() == layer.weights.nrows() && layer.bias.len() == layer.weights.ncols(),
                "Malformed layer dimensions"
            );
            a = (a.dot(&layer.weights) + &layer.bias).mapv(|z| layer.activation.value(z));
        }
        ensure!(a.iter().all(|v| v.is_finite()), "Non-finite network output");
        Ok(a)
    }

    /// Mean squared error over every scalar output, not a sum over outputs.
    pub fn loss(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<f64> {
        let prediction = self.forward(x)?;
        ensure!(
            prediction.dim() == y.dim() && y.iter().all(|v| v.is_finite()),
            "Target shape or value invalid"
        );
        Ok((&prediction - y).mapv(|v| v * v).sum() / prediction.len() as f64)
    }

    /// Manual backpropagation using the chain rule, without any framework.
    pub fn gradients(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(f64, Vec<Gradient>)> {
        self.check_input(x)?;
        let mut activations = vec![x.clone()];
        let mut preactivations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let input = activations.last().expect("input cache is nonempty");
            ensure!(
                input.ncols() == layer.weights.nrows() && layer.bias.len() == layer.weights.ncols(),
                "Malformed layer dimensions"
            );
            let z = input.dot(&layer.weights) + &layer.bias;
            activations.push(z.mapv(|v| layer.activation.value(v)));
            preactivations.push(z);
        }
        let prediction = activations.last().expect("output cache is nonempty");
        ensure!(
            prediction.dim() == y.dim() && y.iter().all(|v| v.is_finite()),
            "Target shape or value invalid"
        );
        let residual = prediction - y;
        let loss = residual.mapv(|v| v * v).sum() / prediction.len() as f64;
        ensure!(loss.is_finite(), "Training diverged: non-finite loss");
        let mut delta = residual * (2.0 / prediction.len() as f64);
        let mut gradients = Vec::with_capacity(self.layers.len());
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            delta *= &preactivations[i].mapv(|z| layer.activation.derivative(z));
            gradients.push(Gradient {
                weights: activations[i].t().dot(&delta),
                bias: delta.sum_axis(Axis(0)),
            });
            if i > 0 {
                delta = delta.dot(&layer.weights.t());
            }
        }
        gradients.reverse();
        Ok((loss, gradients))
    }

    /// One mini-batch gradient descent update; returns the pre-update batch loss.
    pub fn train_batch(&mut self, x: &Array2<f64>, y: &Array2<f64>, rate: f64) -> Result<f64> {
        ensure!(
            rate.is_finite() && rate > 0.0,
            "Learning rate must be positive and finite"
        );
        let (loss, gradients) = self.gradients(x, y)?;
        for (layer, gradient) in self.layers.iter_mut().zip(gradients) {
            layer.weights.scaled_add(-rate, &gradient.weights);
            layer.bias.scaled_add(-rate, &gradient.bias);
        }
        Ok(loss)
    }

    /// Number of trainable weights plus biases.
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn activation_derivatives_and_extreme_sigmoid() {
        for activation in [
            Activation::Sigmoid,
            Activation::Relu,
            Activation::Tanh,
            Activation::Linear,
        ] {
            for x in [-1.3, 0.4, 2.1] {
                let numeric = (activation.value(x + 1e-6) - activation.value(x - 1e-6)) / 2e-6;
                assert!((numeric - activation.derivative(x)).abs() < 1e-8);
            }
        }
        assert_eq!(Activation::Sigmoid.value(-1000.0), 0.0);
        assert_eq!(Activation::Sigmoid.value(1000.0), 1.0);
    }

    #[test]
    fn every_weight_and_bias_passes_central_difference_check() {
        let x = array![[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]];
        let y = array![[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]];
        for activation in [
            Activation::Sigmoid,
            Activation::Relu,
            Activation::Tanh,
            Activation::Linear,
        ] {
            let mut net = Network::new(&[2, 3, 2], activation, 7).unwrap();
            if activation == Activation::Relu {
                net.layers[0].bias.fill(2.0);
            }
            let (_, grads) = net.gradients(&x, &y).unwrap();
            for (i, grad) in grads.iter().enumerate() {
                for ((r, c), expected) in grad.weights.indexed_iter() {
                    let mut plus = net.clone();
                    let mut minus = net.clone();
                    plus.layers[i].weights[[r, c]] += 1e-6;
                    minus.layers[i].weights[[r, c]] -= 1e-6;
                    let actual = (plus.loss(&x, &y).unwrap() - minus.loss(&x, &y).unwrap()) / 2e-6;
                    assert!((actual - expected).abs() < 1e-7, "weight {i}:{r},{c}");
                }
                for (j, expected) in grad.bias.iter().enumerate() {
                    let mut plus = net.clone();
                    let mut minus = net.clone();
                    plus.layers[i].bias[j] += 1e-6;
                    minus.layers[i].bias[j] -= 1e-6;
                    let actual = (plus.loss(&x, &y).unwrap() - minus.loss(&x, &y).unwrap()) / 2e-6;
                    assert!((actual - expected).abs() < 1e-7, "bias {i}:{j}");
                }
            }
        }
    }

    #[test]
    fn learning_and_initialization_are_reproducible() {
        let x = array![[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8], [-0.1, -0.2]];
        let y = &x * 0.7;
        let mut net = Network::new(&[2, 8, 2], Activation::Tanh, 42).unwrap();
        let same = Network::new(&[2, 8, 2], Activation::Tanh, 42).unwrap();
        assert_eq!(net.layers[0].weights, same.layers[0].weights);
        let before = net.loss(&x, &y).unwrap();
        for _ in 0..600 {
            net.train_batch(&x, &y, 0.03).unwrap();
        }
        assert!(net.loss(&x, &y).unwrap() < before * 0.01);
    }

    #[test]
    fn invalid_shapes_and_values_are_errors() {
        let net = Network::new(&[2, 3, 2], Activation::Tanh, 1).unwrap();
        assert!(net.forward(&array![[1.0, 2.0, 3.0]]).is_err());
        assert!(net.forward(&array![[f64::NAN, 0.0]]).is_err());
        assert!(net.gradients(&array![[1.0, 2.0]], &array![[1.0]]).is_err());
        let mut malformed = net;
        malformed.layers[0].bias = Array1::zeros(1);
        assert!(malformed.validate().is_err());
    }
}
