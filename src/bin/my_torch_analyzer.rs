use std::{io::{BufWriter, Write}, path::PathBuf, process::exit};

use my_torch::nn::network::Network;
use my_torch::chess::{dataset::{self, Label}, fen};
use my_torch::{metrics::argmax, trainer::{self, TrainConfig}};

const USAGE: &str = "USAGE\n\t./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n\n\
DESCRIPTION\n\t--train\t\tTraining mode: each line is a FEN followed by the expected output.\n\
\t--predict\tPrediction mode: each line is a FEN, optionally followed by an expected output.\n\
\t--save\t\tSave the trained network into SAVEFILE (train mode only).\n\
\tLOADFILE\tFile containing an artificial neural network.\n\tCHESSFILE\tFile containing chessboards.";

enum Mode { Predict, Train { save: Option<PathBuf> } }
struct Args { mode: Mode, load: PathBuf, chess: PathBuf }

fn parse_args() -> Result<Option<Args>, String> {
    let mut it = std::env::args().skip(1).peekable();
    let mut mode = None;
    let mut rest = Vec::new();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => return Ok(None),
            "--predict" => mode = Some(Mode::Predict),
            "--train" => mode = Some(Mode::Train { save: None }),
            "--save" => {
                let path = it.next().ok_or("--save needs a file")?;
                match &mut mode {
                    Some(Mode::Train { save }) => *save = Some(path.into()),
                    _ => return Err("--save only works after --train".into()),
                }
            }
            _ => rest.push(a),
        }
    }
    match (mode, rest.as_slice()) {
        (Some(mode), [load, chess]) => Ok(Some(Args { mode, load: load.into(), chess: chess.into() })),
        _ => Err(USAGE.into()),
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args()? else { println!("{USAGE}"); return Ok(()); };
    let mut net = Network::load(&args.load)?;
    if net.n_inputs() != fen::N_INPUTS || net.n_outputs() != Label::ALL.len() {
        return Err(format!("{}: network shape does not fit the chess analyzer", args.load.display()));
    }

    match args.mode {
        Mode::Predict => {
            let samples = dataset::load(&args.chess, false)?; // keep every line, in order
            let mut w = BufWriter::new(std::io::stdout().lock());
            for (c, chunk) in samples.chunks(4096).enumerate() {
                let idx: Vec<usize> = (c * 4096..c * 4096 + chunk.len()).collect();
                let (x, _) = dataset::batch(&samples, &idx);
                for (row, s) in net.predict(&x).rows().into_iter().zip(chunk) {
                    writeln!(w, "{}", Label::ALL[argmax(row)].display(&s.pos)).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }
        Mode::Train { save } => {
            let samples = dataset::load(&args.chess, true)?;
            if samples.iter().any(|s| s.label.is_none()) {
                return Err("training needs an expected output on every line".into());
            }
            let mut log = std::fs::File::create("training_log.csv").map_err(|e| e.to_string())?;
            trainer::fit(&mut net, &samples, &TrainConfig::default(), &mut log)?;
            net.save(save.as_deref().unwrap_or(args.load.as_path()))
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        exit(84);
    }
}
