use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

use crate::{
    nn::{
        activation::Activation,
        layer::{Layer, LayerSpec},
        loss::Loss,
        metrics::{Report}
    }
};

const MAGIC: &[u8; 8] = b"MYTORCH1";

fn take(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let end = count.checked_mul(4)
        .and_then(|n| pos.checked_add(n))
        .filter(|&e| e <= data.len())
        .ok_or("truncated network file")?;
    let v = data[*pos..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    *pos = end;
    Ok(v)
}

#[derive(Serialize, Deserialize)]
pub struct LayerArch {
    pub units: usize,
    pub activation: Activation
}

#[derive(Serialize, Deserialize)]
pub struct Arch {
    pub n_inputs: usize,
    pub layers: Vec<LayerArch>
}

pub struct Grads {
    pub dw: Vec<Array2<f32>>,
    pub db: Vec<Array1<f32>>
}

pub struct Network {
    pub layers: Vec<Layer>
}

impl Network {
    pub fn new(n_inputs: usize, specs: &[LayerSpec]) -> Result<Self, String> {
        if n_inputs == 0 || specs.is_empty() {
            return Err("a network needs inputs and at least one layer".into());
        }
        let mut layers = Vec::with_capacity(specs.len());
        let mut prev = n_inputs;
        for s in specs {
            if s.units == 0 { return Err("a layer cannot have 0 units".into()); }
            s.activation.validate()?;
            layers.push(Layer::new(prev, s.units, s.activation, s.init));
            prev = s.units;
        }
        Ok(Self { layers })
    }

    pub fn check_loss(&self, loss: Loss) -> Result<(), String> {
        loss.validate(&self.layers.last().unwrap().act)
    }

    pub fn predict(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut it = self.layers.iter();
        let mut a = it.next().unwrap().forward(x);
        for l in it { a = l.forward(&a); }
        a
    }

    pub fn forward_cached(&self, x: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut acts: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
        for (i, l) in self.layers.iter().enumerate() {
            let out = l.forward(if i == 0 { x } else { &acts[i - 1] });
            acts.push(out);
        }
        acts
    }

    pub fn backward(&self, x: &Array2<f32>, acts: &[Array2<f32>], y: &Array2<f32>, loss: Loss) -> Grads {
        let n = self.layers.len();
        let m = x.nrows() as f32;
        let last = n - 1;

        let mut delta = match (self.layers[last].act, loss) {
            (Activation::Softmax, Loss::CrossEntropy) => &acts[last] - y, // fused, stable
            (act, _) => {
                let mut d = loss.grad(&acts[last], y);
                act.backprop(&mut d, &acts[last]);
                d
            }
        };

        let (mut dw, mut db) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in (0..n).rev() {
            let input = if i == 0 { x } else { &acts[i - 1] };
            dw.push(input.t().dot(&delta) / m);
            db.push(delta.sum_axis(Axis(0)) / m);
            if i > 0 {
                delta = delta.dot(&self.layers[i].w.t());
                self.layers[i - 1].act.backprop(&mut delta, &acts[i - 1]);
            }
        }
        dw.reverse();
        db.reverse();
        Grads { dw, db }
    }

    pub fn apply(&mut self, g: &Grads, lr: f32) {
        for (l, (dw, db)) in self.layers.iter_mut().zip(g.dw.iter().zip(&g.db)) {
            l.w.scaled_add(-lr, dw);
            l.b.scaled_add(-lr, db);
        }
    }

    pub fn train_batch(&mut self, x: &Array2<f32>, y: &Array2<f32>, lr: f32, loss: Loss) -> f32 {
        let acts = self.forward_cached(x);
        let value = loss.value(acts.last().unwrap(), y);
        let g = self.backward(x, &acts, y, loss);
        self.apply(&g, lr);
        value
    }

    pub fn evaluate(&self, x: &Array2<f32>, y: &Array2<f32>, loss: Loss) -> Report {
        Report::new(&self.predict(x), y, loss)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let arch = Arch {
            n_inputs: self.layers[0].w.nrows(),
            layers: self.layers.iter()
                .map(|l| LayerArch { units: l.w.ncols(), activation: l.act })
                .collect(),
        };
        let header = serde_json::to_vec(&arch).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();

        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(header.len() as u32).to_le_bytes());
        buf.extend_from_slice(&header);

        for l in &self.layers {
            for v in l.w.iter().chain(l.b.iter()) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        fs::write(path, buf).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if data.len() < 12 || &data[..8] != MAGIC {
            return Err(format!("{}: not a my_torch network file", path.display()));
        }

        let hlen = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let header = data.get(12..12usize.saturating_add(hlen)).ok_or("truncated header")?;
        let arch: Arch = serde_json::from_slice(header).map_err(|e| e.to_string())?;

        let mut pos = 12 + hlen;
        let mut prev = arch.n_inputs;
        let mut layers = Vec::with_capacity(arch.layers.len());

        for spec in &arch.layers {
            spec.activation.validate()?;
            let n_w = prev.checked_mul(spec.units).ok_or("invalid layer size")?;
            let w = Array2::from_shape_vec((prev, spec.units), take(&data, &mut pos, n_w)?)
                .map_err(|e| e.to_string())?;
            let b = Array1::from_vec(take(&data, &mut pos, spec.units)?);
            layers.push(Layer { w, b, act: spec.activation });
            prev = spec.units;
        }

        if layers.is_empty() {
            return Err("network has no layers".into());
        }
        if pos != data.len() {
            return Err("unexpected trailing bytes in network file".into());
        }

        Ok(Self { layers })
    }
}
