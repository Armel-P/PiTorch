use ndarray::{Array2, ArrayView1};

use crate::nn::loss::Loss;

pub fn cross_entropy(out: &Array2<f32>, y: &Array2<f32>) -> f32 {
    -(y * &out.mapv(|v| (v + 1e-9).ln())).sum() / out.nrows() as f32
}

fn argmax(r: ArrayView1<f32>) -> usize {
    r.iter()
    .enumerate()
    .max_by(|a, b| a.1.total_cmp(b.1))
    .map(|(i, _)| i)
    .unwrap()
}

pub struct Report {
    pub loss: f32,
    pub accuracy: f32,
    pub confusion: Array2<usize>,
}

impl Report {
    pub fn new(out: &Array2<f32>, y: &Array2<f32>, loss: Loss) -> Self {
        let k = y.ncols();
        let mut confusion = Array2::<usize>::zeros((k, k));
        for (o, t) in out.rows().into_iter().zip(y.rows()) {
            confusion[(argmax(t), argmax(o))] += 1;
        }
        let correct: usize = (0..k).map(|i| confusion[(i, i)]).sum();
        Self {
            loss: loss.value(out, y),
            accuracy: correct as f32 / out.nrows() as f32,
            confusion,
        }
    }

    pub fn recall(&self, c: usize) -> f32 {
        let total: usize = self.confusion.row(c).sum();
        if total == 0 { 0.0 } else { self.confusion[(c, c)] as f32 / total as f32 }
    }

    pub fn precision(&self, c: usize) -> f32 {
        let total: usize = self.confusion.column(c).sum();
        if total == 0 { 0.0 } else { self.confusion[(c, c)] as f32 / total as f32 }
    }
}
