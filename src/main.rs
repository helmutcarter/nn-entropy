use std::env;
use std::fs;
use std::path::Path;

use nn_entropy::bat_library::CoordinateSelection;
use nn_entropy::calculate_entropy_from_data;
use nn_entropy::EntropyExtrapolationConfig;
use nn_entropy::ExtrapolationFitModel;
use nn_entropy::ModelDiagnostics;
use nn_entropy::TauObservable;
use nn_entropy::estimate_tau_from_files;
use nn_entropy::extrapolation_report_to_csv;
use nn_entropy::load_internal_coordinate_data_from_files;
use nn_entropy::run_entropy_extrapolation_from_files;

fn parse_subset_sizes(value: &str) -> Result<Vec<usize>, String> {
    let mut subset_sizes = Vec::new();
    for entry in value.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        subset_sizes.push(
            trimmed
                .parse::<usize>()
                .map_err(|_| format!("invalid subset size '{trimmed}'"))?,
        );
    }
    if subset_sizes.is_empty() {
        return Err("subset size list is empty".to_string());
    }
    Ok(subset_sizes)
}

fn print_model_diagnostics(label: &str, diagnostics: &ModelDiagnostics) {
    println!("{label} ({})", diagnostics.model.as_str());
    if diagnostics.bootstrap_data.bootstrap.replicates > 0 {
        println!(
            "  Bootstrap replicates = {}",
            diagnostics.bootstrap_data.bootstrap.replicates
        );
    } else {
        println!("  Bootstrap replicates = off");
    }
    println!(
        "  Extrapolated entropy = {} +/- {}",
        diagnostics.fit.intercept,
        diagnostics.fit.intercept_std_err
    );
    println!("  Fit slope = {}", diagnostics.fit.slope);
    println!(
        "  Weighted residual sum of squares = {}",
        diagnostics.fit.weighted_residual_sum_squares
    );
    println!("  Subset statistics:");
    for subset in &diagnostics.bootstrap_data.subsets {
        println!(
            "    N = {}  N_eff = {:.6}  x = {:.6}  mean = {:.6}  var = {:.6}",
            subset.subset.raw_samples,
            subset.subset.effective_samples,
            subset.subset.x,
            subset.mean_entropy,
            subset.variance
        );
    }
    if !diagnostics.trailing_subset_fits.is_empty() {
        println!("  Trailing-subset stability:");
        for trailing in &diagnostics.trailing_subset_fits {
            println!(
                "    largest {} subsets: intercept = {} +/- {}",
                trailing.subset_count,
                trailing.fit.intercept,
                trailing.fit.intercept_std_err
            );
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <path_to_parm7> <path_to_nc> [--torsions-only] [--start N] [--stop N] [--tau-observable NAME] [--mdout PATH] [--extrapolate] [--subset-sizes A,B,C] [--subset-count N] [--bootstrap-replicates N] [--block-size N] [--fit-model NAME] [--bootstrap-seed N] [--output-csv PATH]",
            args[0]
        );
        std::process::exit(1);
    }
    let top_path = Path::new(&args[1]);
    let traj_path = Path::new(&args[2]);

    let mut torsions_only = false;
    let mut start: Option<usize> = None;
    let mut stop: Option<usize> = None;
    let mut use_python = false;
    let mut tau_observable: Option<TauObservable> = None;
    let mut mdout_path: Option<String> = None;
    let mut extrapolate = false;
    let mut subset_sizes: Option<Vec<usize>> = None;
    let mut subset_count: usize = 5;
    let mut bootstrap_replicates: Option<usize> = None;
    let mut block_size: Option<usize> = None;
    let mut fit_model = ExtrapolationFitModel::InverseN;
    let mut bootstrap_seed: Option<u64> = None;
    let mut output_csv: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--torsions-only" => {
                torsions_only = true;
                i += 1;
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
            "--tau-observable" => {
                if i + 1 >= args.len() {
                    eprintln!("--tau-observable requires a value");
                    std::process::exit(1);
                }
                tau_observable = Some(
                    TauObservable::parse(&args[i + 1]).unwrap_or_else(|err| {
                        eprintln!("{err}");
                        std::process::exit(1);
                    }),
                );
                i += 2;
            }
            "--mdout" => {
                if i + 1 >= args.len() {
                    eprintln!("--mdout requires a value");
                    std::process::exit(1);
                }
                mdout_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--extrapolate" => {
                extrapolate = true;
                i += 1;
            }
            "--subset-sizes" => {
                if i + 1 >= args.len() {
                    eprintln!("--subset-sizes requires a value");
                    std::process::exit(1);
                }
                subset_sizes = Some(parse_subset_sizes(&args[i + 1]).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    std::process::exit(1);
                }));
                i += 2;
            }
            "--subset-count" => {
                if i + 1 >= args.len() {
                    eprintln!("--subset-count requires a value");
                    std::process::exit(1);
                }
                subset_count = args[i + 1]
                    .parse::<usize>()
                    .expect("invalid --subset-count value");
                i += 2;
            }
            "--bootstrap-replicates" => {
                if i + 1 >= args.len() {
                    eprintln!("--bootstrap-replicates requires a value");
                    std::process::exit(1);
                }
                bootstrap_replicates = args[i + 1]
                    .parse::<usize>()
                    .ok();
                if bootstrap_replicates.is_none() {
                    eprintln!("invalid --bootstrap-replicates value");
                    std::process::exit(1);
                }
                i += 2;
            }
            "--block-size" => {
                if i + 1 >= args.len() {
                    eprintln!("--block-size requires a value");
                    std::process::exit(1);
                }
                block_size = Some(
                    args[i + 1]
                        .parse::<usize>()
                        .expect("invalid --block-size value"),
                );
                i += 2;
            }
            "--fit-model" => {
                if i + 1 >= args.len() {
                    eprintln!("--fit-model requires a value");
                    std::process::exit(1);
                }
                fit_model = ExtrapolationFitModel::parse(&args[i + 1]).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--bootstrap-seed" => {
                if i + 1 >= args.len() {
                    eprintln!("--bootstrap-seed requires a value");
                    std::process::exit(1);
                }
                bootstrap_seed = Some(
                    args[i + 1]
                        .parse::<u64>()
                        .expect("invalid --bootstrap-seed value"),
                );
                i += 2;
            }
            "--output-csv" => {
                if i + 1 >= args.len() {
                    eprintln!("--output-csv requires a value");
                    std::process::exit(1);
                }
                output_csv = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    if use_python {
        if !cfg!(debug_assertions) {
            eprintln!("--python is only available in debug builds.");
            std::process::exit(1);
        }
        let script = "/gibbs/helmut/code/python_scripts/NN_entropy_calc_rusty.py";
        let frame_arg = stop
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

    if extrapolate {
        let selection = if torsions_only {
            CoordinateSelection::DihedralsOnly
        } else {
            CoordinateSelection::All
        };
        let mdout = mdout_path.as_deref().map(Path::new);
        let config = EntropyExtrapolationConfig {
            subset_sizes,
            subset_count,
            fit_model,
            compare_models: true,
            bootstrap_replicates,
            block_size,
            bootstrap_seed,
            tau: None,
        };
        let report = match run_entropy_extrapolation_from_files(
            top_path,
            traj_path,
            start.unwrap_or(0),
            stop,
            selection,
            tau_observable,
            mdout,
            config,
        ) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("Extrapolation failed: {err}");
                std::process::exit(1);
            }
        };

        if let Some(estimate) = &report.tau_estimate {
            println!(
                "Estimated tau ({}) = {} using {} series and {} samples",
                estimate.observable.as_str(),
                estimate.tau,
                estimate.n_series,
                estimate.n_samples
            );
        }
        if let Some(path) = &output_csv {
            let csv = extrapolation_report_to_csv(&report);
            if let Err(err) = fs::write(path, csv) {
                eprintln!("Failed to write CSV output to {path}: {err}");
                std::process::exit(1);
            }
            println!("Wrote convergence CSV to {}", path);
        }

        println!("Total entropy = {}", report.total_entropy);
        print_model_diagnostics("Primary model", &report.primary);
        for comparison in &report.comparisons {
            print_model_diagnostics("Comparison model", comparison);
        }
        return;
    }

    let selection = if torsions_only {
        CoordinateSelection::DihedralsOnly
    } else {
        CoordinateSelection::All
    };
    if let Some(observable) = tau_observable {
        match estimate_tau_from_files(
            top_path,
            traj_path,
            start.unwrap_or(0),
            stop,
            observable,
            mdout_path.as_deref().map(Path::new),
        ) {
            Ok(estimate) => {
                println!(
                    "Estimated tau ({}) = {} using {} series and {} samples",
                    estimate.observable.as_str(),
                    estimate.tau,
                    estimate.n_series,
                    estimate.n_samples
                );
            }
            Err(err) => {
                eprintln!("Tau estimation failed: {err}");
                std::process::exit(1);
            }
        }
    }

    let mut one_d_data = match load_internal_coordinate_data_from_files(
        top_path,
        traj_path,
        stop.unwrap_or(usize::MAX),
        selection,
    ) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("Entropy calculation failed: {err}");
            std::process::exit(1);
        }
    };

    let start = start.unwrap_or(0);
    if one_d_data.is_empty() || start >= one_d_data[0].len() {
        eprintln!("Entropy calculation failed: start is beyond available frames");
        std::process::exit(1);
    }
    for coord in &mut one_d_data {
        *coord = coord[start..].to_vec();
    }
    let used_frames = one_d_data[0].len();
    let entropy = match calculate_entropy_from_data(one_d_data, used_frames) {
        Ok(entropy) => entropy,
        Err(err) => {
            eprintln!("Entropy calculation failed: {err}");
            std::process::exit(1);
        }
    };
    println!("Total entropy = {}", entropy);
}
