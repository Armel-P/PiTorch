use ndarray::ArrayViewMut1;

pub const N_INPUTS: usize = 769;

pub struct Position {
    pub active: Vec<u16>,
    pub white_to_move: bool,
}

impl Position {
    pub fn fill(&self, mut row: ArrayViewMut1<f32>) {
        for &i in &self.active { row[i as usize] = 1.0; }
    }
    pub fn attacker(&self) -> &'static str {
        if self.white_to_move { "Black" } else { "White" }
    }
}

pub fn parse(fen: &str) -> Result<Position, String> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().ok_or("empty FEN")?;
    let side = parts.next().ok_or("FEN: missing side to move")?;

    let mut active = Vec::with_capacity(33);
    let (mut rank, mut file) = (0usize, 0usize);
    for c in placement.chars() {
        match c {
            '/' => {
                if file != 8 { return Err(format!("FEN: incomplete row in '{placement}'")); }
                rank += 1;
                file = 0;
            }
            '1'..='8' => file += c.to_digit(10).unwrap() as usize,
            _ => {
                let piece = "PNBRQKpnbrqk".find(c)
                    .ok_or_else(|| format!("FEN: invalid piece '{c}'"))?;
                if rank > 7 || file > 7 { return Err(format!("FEN: board too large '{placement}'")); }
                active.push((piece * 64 + rank * 8 + file) as u16);
                file += 1;
            }
        }
    }
    if rank != 7 || file != 8 { return Err(format!("FEN: incomplete board '{placement}'")); }

    let white_to_move = match side {
        "w" => true,
        "b" => false,
        _ => return Err(format!("FEN: invalid side to move '{side}'")),
    };
    if white_to_move { active.push(768); }
    Ok(Position { active, white_to_move })
}
