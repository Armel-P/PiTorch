use std::{path::Path, process::exit};

use my_torch::nn::config::NetworkConfig;

const USAGE: &str = "USAGE\n\
\t./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n\n\
DESCRIPTION\n\
\tconfig_file_i\tConfiguration file describing a neural network to generate.\n\
\tnb_i\t\tNumber of neural networks to generate from the configuration file.";

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.is_empty() || args.len() % 2 != 0 {
        return Err(USAGE.into());
    }

    for pair in args.chunks(2) {
        let path = Path::new(&pair[0]);
        let nb: usize = pair[1].parse().map_err(|_| format!("invalid count: {}", pair[1]))?;
        let cfg = NetworkConfig::from_file(path)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).ok_or("invalid config name")?;
        for i in 1..=nb {
            cfg.build()?.save(&path.with_file_name(format!("{stem}_{i}.nn")))?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        exit(84);
    }
}
