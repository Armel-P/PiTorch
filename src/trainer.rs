use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use std::io::Write;

use crate::chess::dataset::{self, Sample};
use crate::nn::{loss::Loss, network::Network};

pub struct TrainConfig {
    pub lr: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub val_fraction: f32,
    pub patience: usize,
    pub loss: Loss,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { lr: 0.05, batch_size: 256, epochs: 30, val_fraction: 0.1, patience: 3,
               loss: Loss::CrossEntropy }
    }
}

fn evaluate(net: &Network, samples: &[Sample], idx: &[usize], loss: Loss) -> (f32, f32) {
    let (mut total_loss, mut correct) = (0.0f32, 0usize);
    for chunk in idx.chunks(4096) {
        let (x, y) = dataset::batch(samples, chunk);
        let r = net.evaluate(&x, &y, loss);
        total_loss += r.loss * chunk.len() as f32;
        correct += (0..r.confusion.nrows()).map(|i| r.confusion[(i, i)]).sum::<usize>();
    }
    (total_loss / idx.len() as f32, correct as f32 / idx.len() as f32)
}

pub fn fit(net: &mut Network, samples: &[Sample], cfg: &TrainConfig, log: &mut impl Write)
    -> Result<(), String>
{
    net.check_loss(cfg.loss)?;
    if cfg.batch_size == 0 { return Err("batch size must be > 0".into()); }

    let mut rng = rand::thread_rng();
    let mut idx: Vec<usize> = (0..samples.len()).collect();
    idx.shuffle(&mut rng);
    let n_val = (idx.len() as f32 * cfg.val_fraction) as usize;
    if n_val == 0 || n_val >= idx.len() { return Err("not enough samples for a validation split".into()); }
    let (val_idx, train_idx) = idx.split_at(n_val);
    let mut train_idx = train_idx.to_vec();

    let mut best_loss = f32::INFINITY;
    let mut best: Vec<(Array2<f32>, Array1<f32>)> = Vec::new();
    let mut stale = 0;
    writeln!(log, "epoch,train_loss,val_loss,val_acc").map_err(|e| e.to_string())?;

    for epoch in 1..=cfg.epochs {
        train_idx.shuffle(&mut rng);
        let mut sum = 0.0f32;
        for chunk in train_idx.chunks(cfg.batch_size) {
            let (x, y) = dataset::batch(samples, chunk);
            sum += net.train_batch(&x, &y, cfg.lr, cfg.loss) * chunk.len() as f32;
        }
        let train_loss = sum / train_idx.len() as f32;
        let (val_loss, val_acc) = evaluate(net, samples, val_idx, cfg.loss);
        writeln!(log, "{epoch},{train_loss},{val_loss},{val_acc}").map_err(|e| e.to_string())?;

        if val_loss < best_loss {
            best_loss = val_loss;
            best = net.layers.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
            stale = 0;
        } else {
            stale += 1;
            if stale >= cfg.patience { break; }
        }
    }

    for (l, (w, b)) in net.layers.iter_mut().zip(best) {
        l.w = w;
        l.b = b;
    }
    Ok(())
}
