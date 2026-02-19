use std::env;
use std::path::Path;

use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::calculate_entropy_from_data;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <path_to_parm7> <path_to_nc> [--torsions-only] [--frames N] [--start N] [--stop N] [--python]",
            args[0]
        );
        std::process::exit(1);
    }
    let top_path = Path::new(&args[1]);
    let traj_path = Path::new(&args[2]);

    let mut torsions_only = false;
    let mut frames: Option<usize> = None;
    let mut start: Option<usize> = None;
    let mut stop: Option<usize> = None;
    let mut use_python = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--torsions-only" => {
                torsions_only = true;
                i += 1;
            }
            "--frames" => {
                if i + 1 >= args.len() {
                    eprintln!("--frames requires a value");
                    std::process::exit(1);
                }
                frames = Some(
                    args[i + 1]
                        .parse::<usize>()
                        .expect("invalid --frames value"),
                );
                i += 2;
            }
            "--start" => {
                if i + 1 >= args.len() {
                    eprintln!("--start requires a value");
                    std::process::exit(1);
                }
                start = Some(
                    args[i + 1]
                        .parse::<usize>()
                        .expect("invalid --start value"),
                );
                i += 2;
            }
            "--stop" => {
                if i + 1 >= args.len() {
                    eprintln!("--stop requires a value");
                    std::process::exit(1);
                }
                stop = Some(
                    args[i + 1]
                        .parse::<usize>()
                        .expect("invalid --stop value"),
                );
                i += 2;
            }
            "--python" => {
                use_python = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    if use_python {
        let script = "/gibbs/helmut/code/python_scripts/NN_entropy_calc_rusty.py";
        let frame_arg = frames
            .or(stop)
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-1".to_string());
        let output = std::process::Command::new("python")
            .arg(script)
            .arg(top_path)
            .arg(frame_arg)
            .arg(traj_path)
            .output()
            .expect("failed to run python entropy script");
        if !output.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            std::process::exit(1);
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let entropy = stdout
            .trim()
            .split(',')
            .last()
            .and_then(|s| s.trim().split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
            .expect("failed to parse python entropy output");
        println!("Total entropy = {}", entropy);
        return;
    }

    let mut internal = InternalCoordinates::new(top_path)
        .expect("failed to build BAT list from topology");
    internal
        .calculate_internal_coords(
            traj_path,
            frames.or(stop).unwrap_or(usize::MAX),
            torsions_only,
        )
        .expect("failed to read trajectory or compute BAT coordinates");

    let frames = internal.int_coords.len();
    if frames == 0 {
        eprintln!("No frames read from trajectory.");
        std::process::exit(1);
    }
    let start = start.unwrap_or(0);
    if start >= frames {
        eprintln!("--start is beyond available frames.");
        std::process::exit(1);
    }
    let dim = internal
        .int_coords
        .get(0)
        .map(|row| row.len())
        .unwrap_or(0);
    if dim == 0 {
        eprintln!("No internal coordinates were generated.");
        std::process::exit(1);
    }

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frames - start); dim];
    for frame in &internal.int_coords[start..] {
        if frame.len() != dim {
            eprintln!("Inconsistent internal coordinate dimensions.");
            std::process::exit(1);
        }
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    let used_frames = one_d_data[0].len();
    let entropy = calculate_entropy_from_data(one_d_data, used_frames);
    println!("Total entropy = {}", entropy);
}
