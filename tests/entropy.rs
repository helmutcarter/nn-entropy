use std::path::PathBuf;

use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::calculate_entropy_from_data;

fn test_data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn entropy_matches_reference_for_test_fixture() {
    let top = test_data_path("tests/fixtures/test.parm7");
    let traj = test_data_path("tests/fixtures/test.nc");
    let frames = usize::MAX;

    let mut internal =
        InternalCoordinates::new(&top).expect("failed to build BAT list from topology");
    internal
        .calculate_internal_coords(&traj, frames, false)
        .expect("failed to read trajectory or compute BAT coordinates");

    let frame_count = internal.int_coords.len();
    let dim = internal.int_coords[0].len();

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frame_count); dim];
    for frame in &internal.int_coords {
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    let entropy = calculate_entropy_from_data(one_d_data, frame_count);
    let expected = -44.640002919813014_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn torsions_only_entropy_matches_reference_for_test_fixture() {
    let top = test_data_path("tests/fixtures/test.parm7");
    let traj = test_data_path("tests/fixtures/test.nc");
    let frames = usize::MAX;

    let mut internal =
        InternalCoordinates::new(&top).expect("failed to build BAT list from topology");
    internal
        .calculate_internal_coords(&traj, frames, true)
        .expect("failed to read trajectory or compute BAT coordinates");

    let frame_count = internal.int_coords.len();
    let dim = internal.int_coords[0].len();

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frame_count); dim];
    for frame in &internal.int_coords {
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    let entropy = calculate_entropy_from_data(one_d_data, frame_count);
    let expected = 32.01068807905125_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "torsions-only entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}
