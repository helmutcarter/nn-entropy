use std::path::PathBuf;
use std::process::Command;

use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::calculate_entropy_from_data_with_order;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn parse_entropy(stdout: &[u8]) -> f64 {
    String::from_utf8_lossy(stdout)
        .split('=')
        .next_back()
        .expect("missing entropy output")
        .trim()
        .parse()
        .expect("invalid entropy output")
}

#[test]
fn no_periodic_matches_linear_raw_array_estimator() {
    let top = fixture("tests/fixtures/test.parm7");
    let traj = fixture("tests/fixtures/test.nc");
    let output = Command::new(env!("CARGO_BIN_EXE_nn_entropy"))
        .args([
            top.to_str().unwrap(),
            traj.to_str().unwrap(),
            "--no-periodic",
            "--mie-order",
            "2",
        ])
        .output()
        .expect("failed to run CLI");
    assert!(output.status.success());
    let cli_entropy = parse_entropy(&output.stdout);

    let mut internal = InternalCoordinates::new(&top).unwrap();
    internal
        .calculate_internal_coords(&traj, usize::MAX, false)
        .unwrap();
    let frames = internal.int_coords.len();
    let dimensions = internal.int_coords[0].len();
    let mut data = vec![Vec::with_capacity(frames); dimensions];
    for frame in internal.int_coords {
        for (dimension, value) in frame.into_iter().enumerate() {
            data[dimension].push(value);
        }
    }
    let expected = calculate_entropy_from_data_with_order(data, frames, 2).unwrap();
    assert!((cli_entropy - expected).abs() < 1e-12);
}

#[test]
fn periodic_remains_the_default() {
    let top = fixture("tests/fixtures/test.parm7");
    let traj = fixture("tests/fixtures/test.nc");
    let periodic = Command::new(env!("CARGO_BIN_EXE_nn_entropy"))
        .args([top.to_str().unwrap(), traj.to_str().unwrap()])
        .output()
        .unwrap();
    let linear = Command::new(env!("CARGO_BIN_EXE_nn_entropy"))
        .args([
            top.to_str().unwrap(),
            traj.to_str().unwrap(),
            "--no-periodic",
        ])
        .output()
        .unwrap();
    assert!(periodic.status.success() && linear.status.success());
    assert!((parse_entropy(&periodic.stdout) - parse_entropy(&linear.stdout)).abs() > 1.0);
}
