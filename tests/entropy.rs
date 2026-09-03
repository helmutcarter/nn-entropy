use std::path::PathBuf;

use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::{
    CoordinateMetric, calculate_entropy_from_data_with_metrics,
    estimate_coordinate_entropy_with_metrics,
};

fn test_data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn one_d_data_from_fixture(torsions_only: bool) -> (Vec<Vec<f64>>, usize, Vec<CoordinateMetric>) {
    let top = test_data_path("tests/fixtures/test.parm7");
    let traj = test_data_path("tests/fixtures/test.nc");
    let frames = usize::MAX;

    let mut internal =
        InternalCoordinates::new(&top).expect("failed to build BAT list from topology");
    internal
        .calculate_internal_coords(&traj, frames, torsions_only)
        .expect("failed to read trajectory or compute BAT coordinates");

    let frame_count = internal.int_coords.len();
    let dim = internal.int_coords[0].len();
    let metrics = internal.coordinate_metrics();

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frame_count); dim];
    for frame in &internal.int_coords {
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    (one_d_data, frame_count, metrics)
}

#[test]
fn first_order_entropy_matches_periodic_fixture_regression() {
    let (one_d_data, frame_count, metrics) = one_d_data_from_fixture(false);
    let entropy = calculate_entropy_from_data_with_metrics(one_d_data, frame_count, 1, &metrics)
        .expect("first-order entropy calculation failed");
    let expected = -43.2442783659548_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "first-order entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn coordinate_first_order_entropy_sums_to_fixture_total() {
    let (one_d_data, frame_count, metrics) = one_d_data_from_fixture(false);
    let entropies = estimate_coordinate_entropy_with_metrics(one_d_data, frame_count, &metrics)
        .expect("coordinate entropy calculation failed");
    assert_eq!(entropies.len(), metrics.len());
    let sum: f64 = entropies.iter().sum();
    assert!((sum - (-43.2442783659548)).abs() < 1e-9);
}

#[test]
fn second_order_entropy_matches_periodic_fixture_regression() {
    let (one_d_data, frame_count, metrics) = one_d_data_from_fixture(false);
    let entropy = calculate_entropy_from_data_with_metrics(one_d_data, frame_count, 2, &metrics)
        .expect("second-order entropy calculation failed");
    let expected = -130.49745250649198_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "second-order entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn default_entropy_matches_second_order_reference_for_test_fixture() {
    let (one_d_data, frame_count, metrics) = one_d_data_from_fixture(false);
    let entropy = calculate_entropy_from_data_with_metrics(one_d_data, frame_count, 2, &metrics)
        .expect("entropy calculation failed");
    let expected = -130.49745250649198_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn torsions_only_entropy_matches_reference_for_test_fixture() {
    let (one_d_data, frame_count, metrics) = one_d_data_from_fixture(true);
    let entropy = calculate_entropy_from_data_with_metrics(one_d_data, frame_count, 2, &metrics)
        .expect("entropy calculation failed");
    let expected = -10.524094445327933_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "torsions-only entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}
