use std::path::PathBuf;

use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::{
    calculate_entropy_from_data, calculate_entropy_from_data_with_order,
    estimate_coordinate_entropy_rust,
};

fn test_data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn one_d_data_from_fixture(torsions_only: bool) -> (Vec<Vec<f64>>, usize) {
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

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frame_count); dim];
    for frame in &internal.int_coords {
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    (one_d_data, frame_count)
}

#[test]
fn first_order_entropy_matches_external_reference_for_test_fixture() {
    let (one_d_data, frame_count) = one_d_data_from_fixture(false);
    let entropy = calculate_entropy_from_data_with_order(one_d_data, frame_count, 1)
        .expect("first-order entropy calculation failed");
    let expected = -41.043074891730285_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "first-order entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn coordinate_first_order_entropy_matches_external_reference_for_test_fixture() {
    let (one_d_data, frame_count) = one_d_data_from_fixture(false);
    let entropies = estimate_coordinate_entropy_rust(one_d_data, frame_count)
        .expect("coordinate entropy calculation failed");
    let expected = [
        -2.191987290884043,
        -2.0487846062513055,
        -3.1696799251072125,
        -2.2133854743950314,
        -1.393460671737234,
        -1.6466926810259102,
        -1.473717514192777,
        -2.2824391744103885,
        -0.8068605030087515,
        -1.9017447390017441,
        -1.566875588217191,
        -1.7170077099354897,
        -1.0767286727727439,
        -2.212324981754663,
        -2.3159339787698814,
        -1.4020467659639122,
        -2.0492610475080566,
        -1.8253234616647278,
        -1.003624819729084,
        -0.40948034657778365,
        -0.7080474758829705,
        -2.1680865643109426,
        -0.5438524953779607,
        0.4640503371504199,
        -0.4886070200656061,
        -0.7936194264908667,
        -1.0945962573885026,
        -0.37336191468224866,
        -0.6295941217736685,
    ];

    assert_eq!(
        entropies.len(),
        expected.len(),
        "coordinate entropy length mismatch"
    );
    for (idx, (actual, expected)) in entropies.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-9,
            "coordinate {idx} entropy mismatch: got {actual}, expected {expected}, diff {diff}"
        );
    }
}

#[test]
fn second_order_entropy_matches_external_reference_for_test_fixture() {
    let (one_d_data, frame_count) = one_d_data_from_fixture(false);
    let entropy = calculate_entropy_from_data_with_order(one_d_data, frame_count, 2)
        .expect("second-order entropy calculation failed");
    let expected = -44.640002919814606_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "second-order entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn default_entropy_matches_second_order_reference_for_test_fixture() {
    let (one_d_data, frame_count) = one_d_data_from_fixture(false);
    let entropy =
        calculate_entropy_from_data(one_d_data, frame_count).expect("entropy calculation failed");
    let expected = -44.640002919814606_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}

#[test]
fn torsions_only_entropy_matches_reference_for_test_fixture() {
    let (one_d_data, frame_count) = one_d_data_from_fixture(true);
    let entropy =
        calculate_entropy_from_data(one_d_data, frame_count).expect("entropy calculation failed");
    let expected = 32.01068807905125_f64;
    let diff = (entropy - expected).abs();
    assert!(
        diff < 1e-9,
        "torsions-only entropy mismatch: got {entropy}, expected {expected}, diff {diff}"
    );
}
