use ndarray::Array2;
use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    fs,
    hash::{Hash, Hasher},
    path::Path,
};

use crate::chess::fen::{self, Position};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Label { Nothing, Check, Checkmate }

impl Label {
    pub const ALL: [Label; 3] = [Label::Nothing, Label::Check, Label::Checkmate];
    pub fn index(self) -> usize { self as usize }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "Nothing" => Some(Label::Nothing),
            "Check" => Some(Label::Check),
            "Checkmate" => Some(Label::Checkmate),
            _ => None,
        }
    }

    pub fn display(self, pos: &Position) -> String {
        match self {
            Label::Nothing => "Nothing".into(),
            Label::Check => format!("Check {}", pos.attacker()),
            Label::Checkmate => format!("Checkmate {}", pos.attacker()),
        }
    }
}

pub struct Sample { pub pos: Position, pub label: Option<Label> }

pub fn parse_line(line: &str) -> Result<Sample, String> {
    let t: Vec<&str> = line.split_whitespace().collect();
    if t.len() < 6 { return Err("expected a 6-field FEN".into()); }
    let pos = fen::parse(&t[..2].join(" "))?; // placement + side to move are all we use
    let label = match t.get(6) {
        Some(w) => Some(Label::parse(w).ok_or_else(|| format!("unknown label '{w}'"))?),
        None => None,
    };
    Ok(Sample { pos, label })
}

pub fn load(path: &Path, dedup: bool) -> Result<Vec<Sample>, String> {
    let txt = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (i, line) in txt.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let s = parse_line(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if dedup {
            let mut h = DefaultHasher::new();
            s.pos.active.hash(&mut h);
            if !seen.insert(h.finish()) { continue; }
        }
        out.push(s);
    }
    Ok(out)
}

pub fn batch(samples: &[Sample], idx: &[usize]) -> (Array2<f32>, Array2<f32>) {
    let mut x = Array2::zeros((idx.len(), fen::N_INPUTS));
    let mut y = Array2::zeros((idx.len(), Label::ALL.len()));
    for (r, &i) in idx.iter().enumerate() {
        samples[i].pos.fill(x.row_mut(r));
        if let Some(l) = samples[i].label { y[(r, l.index())] = 1.0; }
    }
    (x, y)
}
