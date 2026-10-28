use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};

pub const SELU_LAMBDA: f32 = 1.050_701;
pub const SELU_ALPHA: f32 = 1.673_263_2;

fn scale(delta: &mut Array2<f32>, a: &Array2<f32>, f: impl Fn(f32) -> f32) {
    Zip::from(delta).and(a).for_each(|d, &a| *d *= f(a));
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    Linear,
    Relu,
    LeakyRelu { alpha: f32 },
    Elu { alpha: f32 },
    Selu,
    Sigmoid,
    Tanh,
    Softmax,
    Softplus,   
}

impl Activation {
    pub const DEFAULT_LEAKY_ALPHA: f32 = 0.01;
    pub const DEFAULT_ELU_ALPHA: f32 = 1.0;

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::LeakyRelu { alpha } | Self::Elu { alpha }
                if !alpha.is_finite() || *alpha < 0.0 =>
                 Err(format!("Invalid alpha {alpha}: must be finite and >= 0")),
            _ => Ok(()),
        }
    }

    pub fn apply(&self, z: &mut Array2<f32>) {
        match *self {
            Self::Linear => {}
            Self::Relu => z.mapv_inplace(|v| v.max(0.0)),
            Self::LeakyRelu { alpha } => z.mapv_inplace(|v| if v > 0.0 { v } else { alpha * v }),
            Self::Elu { alpha } => z.mapv_inplace(|v| if v > 0.0 { v } else { alpha * v.exp_m1() }),
            Self::Selu => z.mapv_inplace(|v| {
                SELU_LAMBDA * if v > 0.0 { v } else { SELU_ALPHA * v.exp_m1() }
            }),
            Self::Sigmoid => z.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp())),
            Self::Tanh => z.mapv_inplace(f32::tanh),
            Self::Softplus => z.mapv_inplace(|v| v.max(0.0) + (-v.abs()).exp().ln_1p()),
            Self::Softmax => {
                for mut row in z.rows_mut() {
                    let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    row.mapv_inplace(|v| (v - max).exp());
                    let sum = row.sum();
                    row.mapv_inplace(|v| v / sum);
                }
            }
        }
    }

    pub fn backprop(&self, delta: &mut Array2<f32>, a: &Array2<f32>) {
        match *self {
            Self::Linear => {}
            Self::Relu => scale(delta, a, |a| if a > 0.0 { 1.0 } else { 0.0 }),
            Self::LeakyRelu { alpha } => scale(delta, a, |a| if a > 0.0 { 1.0 } else { alpha }),
            Self::Elu { alpha } => scale(delta, a, |a| if a > 0.0 { 1.0 } else { a + alpha }),
            Self::Selu => scale(delta, a, |a| {
                if a > 0.0 { SELU_LAMBDA } else { a + SELU_LAMBDA * SELU_ALPHA }
            }),
            Self::Sigmoid => scale(delta, a, |a| a * (1.0 - a)),
            Self::Tanh => scale(delta, a, |a| 1.0 - a * a),
            Self::Softplus => scale(delta, a, |a| 1.0 - (-a).exp()),
            Self::Softmax => {
                for (mut d, s) in delta.rows_mut().into_iter().zip(a.rows()) {
                    let dot: f32 = d.iter().zip(s.iter()).map(|(g, s)| g * s).sum();
                    Zip::from(&mut d).and(&s).for_each(|d, &s| *d = s * (*d - dot));
                }
            }
        }
    }
}
