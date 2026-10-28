use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::nn::activation::Activation;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Init {
    Xavier,
    He,
    LeCun
}

impl Init {
    pub fn recommended(act: &Activation) -> Self {
        match act {
            Activation::Relu | Activation::LeakyRelu { .. } | Activation::Elu { .. }
            | Activation::Softplus => Init::He,
            Activation::Selu => Init::LeCun,
            _ => Init::Xavier,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LayerSpec {
    pub units: usize,
    pub activation: Activation,
    pub init: Init
}

impl LayerSpec {
    pub fn new(units: usize, activation: Activation) -> Self {
        Self { units, activation, init: Init::recommended(&activation) }
    }

    pub fn with_init(mut self, init: Init) -> Self { self.init = init; self }
}

pub struct Layer {
    pub w: Array2<f32>,
    pub b: Array1<f32>,
    pub act: Activation,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, act: Activation, init: Init) -> Self {
        let var = match init {
            Init::Xavier => 2.0 / (n_in + n_out) as f32,
            Init::He => 2.0 / n_in as f32,
            Init::LeCun => 1.0 / n_in as f32,
        };
        let limit = (3.0 * var).sqrt();
        let mut rng = rand::thread_rng();
        Self {
            w: Array2::from_shape_fn((n_in, n_out), |_| rng.gen_range(-limit..limit)),
            b: Array1::zeros(n_out),
            act,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut z = x.dot(&self.w);
        z += &self.b;
        self.act.apply(&mut z);
        z
    }
}
