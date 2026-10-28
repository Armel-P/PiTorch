use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::nn::activation::Activation;

const EPS: f32 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Loss {
    CrossEntropy,
    Mse,
    Mae,
    Huber { delta: f32 }
}

impl Loss {
    pub const DEFAULT_HUBER_DELTA: f32 = 1.0;

    pub fn validate(&self, last: &Activation) -> Result<(), String> {
        match self {
            Loss::Huber { delta } if !delta.is_finite() || *delta <= 0.0 =>
                Err(format!("invalid huber delta {delta}: must be finite and > 0")),
            Loss::CrossEntropy if !matches!(last, Activation::Softmax | Activation::Sigmoid) =>
                Err("cross-entropy needs a softmax or sigmoid output layer".into()),
            _ => Ok(()),
        }
    }

    pub fn value(&self, out: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let pairs = out.iter().zip(y.iter()).map(|(&o, &t)| (o, t));
        let total: f32 = match *self {
            Loss::CrossEntropy => pairs.map(|(o, t)| -t * (o + EPS).ln()).sum(),
            Loss::Mse => pairs.map(|(o, t)| 0.5 * (o - t).powi(2)).sum(),
            Loss::Mae => pairs.map(|(o, t)| (o - t).abs()).sum(),
            Loss::Huber { delta } => pairs.map(|(o, t)| {
                let r = (o - t).abs();
                if r <= delta { 0.5 * r * r } else { delta * (r - 0.5 * delta) }
            }).sum(),
        };
        total / out.nrows() as f32
    }

    pub fn grad(&self, out: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
        let mut g = out.clone();
        match *self {
            Loss::CrossEntropy => g.zip_mut_with(y, |o, &t| *o = -t / (*o + EPS)),
            Loss::Mse => g.zip_mut_with(y, |o, &t| *o -= t),
            Loss::Mae => g.zip_mut_with(y, |o, &t| {
                let r = *o - t;
                *o = if r > 0.0 { 1.0 } else if r < 0.0 { -1.0 } else { 0.0 };
            }),
            Loss::Huber { delta } => g.zip_mut_with(y, |o, &t| *o = (*o - t).clamp(-delta, delta)),
        }
        g
    }
}
