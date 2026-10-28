use serde::Deserialize;
use std::{fs, path::Path};

use crate::{
    nn::{
        activation::Activation,
        layer::{Init, LayerSpec},
        network::Network
    }
};

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Kind { Linear, Relu, LeakyRelu, Elu, Selu, Sigmoid, Tanh, Softplus, Softmax }

#[derive(Deserialize)]
struct LayerConfig {
    units: usize,
    activation: Kind,
    alpha: Option<f32>,
    init: Option<Init>,
}

#[derive(Deserialize)]
pub struct NetworkConfig {
    n_inputs: usize,
    #[serde(rename = "layer")]
    layers: Vec<LayerConfig>,
}

impl LayerConfig {
    fn to_spec(&self) -> Result<LayerSpec, String> {
        let act = match (self.activation, self.alpha) {
            (Kind::LeakyRelu, a) => Activation::LeakyRelu { alpha: a.unwrap_or(Activation::DEFAULT_LEAKY_ALPHA) },
            (Kind::Elu, a) => Activation::Elu { alpha: a.unwrap_or(Activation::DEFAULT_ELU_ALPHA) },
            (_, Some(_)) => return Err("`alpha` only applies to leaky_relu and elu".into()),
            (Kind::Linear, _) => Activation::Linear,
            (Kind::Relu, _) => Activation::Relu,
            (Kind::Selu, _) => Activation::Selu,
            (Kind::Sigmoid, _) => Activation::Sigmoid,
            (Kind::Tanh, _) => Activation::Tanh,
            (Kind::Softplus, _) => Activation::Softplus,
            (Kind::Softmax, _) => Activation::Softmax,
        };
        let spec = LayerSpec::new(self.units, act);
        Ok(match self.init { Some(i) => spec.with_init(i), None => spec })
    }
}

impl NetworkConfig {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let txt = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        toml::from_str(&txt).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn build(&self) -> Result<Network, String> {
        let specs = self.layers.iter().map(LayerConfig::to_spec).collect::<Result<Vec<_>, _>>()?;
        Network::new(self.n_inputs, &specs)
    }
}
