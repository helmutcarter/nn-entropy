//! The input-validation contract.
//!
//! Every estimator is fallible, and a scientific pipeline depends on bad input
//! being rejected with a diagnosis rather than silently producing a number.
//! These tests pin which inputs are refused and what each error says.

mod common;

use nn_entropy::{
    CoordinateMetric, FiniteSampleConstant, calc_one_d_nn_with_metric,
    calculate_entropy_from_data_with_metrics,
    calculate_entropy_from_data_with_metrics_and_constant, calculate_entropy_from_data_with_order,
    estimate_coordinate_entropy_with_metrics, estimate_coordinate_mie_entropy_with_metrics,
    estimate_coordinate_mutual_information_with_metrics,
};

fn linear(n: usize) -> Vec<CoordinateMetric> {
    vec![CoordinateMetric::Linear; n]
}

fn expect_error(result: Result<f64, String>, needle: &str, what: &str) {
    let error = result.expect_err(&format!("{what}: expected rejection"));
    assert!(
        error.contains(needle),
        "{what}: error {error:?} does not mention {needle:?}"
    );
}

// ---------------------------------------------------------------------------
// Sample count
// ---------------------------------------------------------------------------

#[test]
fn fewer_than_two_frames_is_rejected() {
    // A nearest neighbor needs at least two samples; one frame has none.
    for frames in [0usize, 1] {
        let data = vec![vec![0.0, 1.0, 2.0]];
        expect_error(
            calculate_entropy_from_data_with_order(data, frames, 1),
            "at least two frames",
            &format!("frames_end = {frames}"),
        );
    }
}

#[test]
fn two_frames_is_accepted_as_the_boundary_case() {
    // Exactly two distinct frames is the smallest valid input, so the guard
    // must be `< 2` rather than `<= 2`.
    let value = calculate_entropy_from_data_with_order(vec![vec![0.0, 1.0]], 2, 1)
        .expect("two frames should be accepted");
    assert!(value.is_finite());
}

#[test]
fn a_coordinate_shorter_than_the_frame_count_is_rejected() {
    let data = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0]];
    expect_error(
        calculate_entropy_from_data_with_order(data, 3, 1),
        "expected at least",
        "short coordinate",
    );
}

#[test]
fn empty_data_is_rejected() {
    expect_error(
        calculate_entropy_from_data_with_order(vec![], 4, 1),
        "no coordinate data",
        "empty data",
    );
}

// ---------------------------------------------------------------------------
// Degenerate and non-finite values
// ---------------------------------------------------------------------------

#[test]
fn non_finite_values_are_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let data = vec![vec![0.0, 1.0, bad]];
        expect_error(
            calculate_entropy_from_data_with_order(data, 3, 1),
            "non-finite",
            &format!("value {bad}"),
        );
    }
}

#[test]
fn a_constant_coordinate_is_rejected() {
    // Every sample identical means every nearest-neighbor distance is zero and
    // the log diverges, so this must be refused rather than returning -inf.
    let data = vec![vec![1.0, 1.0, 1.0, 1.0]];
    expect_error(
        calculate_entropy_from_data_with_order(data, 4, 1),
        "two unique values",
        "constant coordinate",
    );

    let error = calc_one_d_nn_with_metric(&[2.5, 2.5, 2.5], CoordinateMetric::Linear)
        .expect_err("constant series should be rejected");
    assert!(error.contains("distinct"), "unexpected error {error:?}");
}

#[test]
fn a_constant_coordinate_is_rejected_even_alongside_valid_ones() {
    // The check is per coordinate, so one degenerate column must fail the whole
    // call rather than being silently skipped.
    let data = vec![vec![0.0, 1.0, 2.0], vec![5.0, 5.0, 5.0]];
    expect_error(
        calculate_entropy_from_data_with_order(data, 3, 2),
        "coordinate 1",
        "degenerate second coordinate",
    );
}

// ---------------------------------------------------------------------------
// Metric metadata
// ---------------------------------------------------------------------------

#[test]
fn a_metric_count_mismatch_is_rejected() {
    let data = vec![vec![0.0, 1.0], vec![1.0, 2.0]];
    expect_error(
        calculate_entropy_from_data_with_metrics(data.clone(), 2, 1, &linear(1)),
        "metrics",
        "too few metrics",
    );
    expect_error(
        calculate_entropy_from_data_with_metrics(data, 2, 1, &linear(3)),
        "metrics",
        "too many metrics",
    );
}

#[test]
fn a_non_positive_or_non_finite_period_is_rejected() {
    let data = vec![vec![0.0, 1.0, 2.0]];
    for period in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        expect_error(
            calculate_entropy_from_data_with_metrics(
                data.clone(),
                3,
                1,
                &[CoordinateMetric::Periodic { period }],
            ),
            "positive",
            &format!("period {period}"),
        );
    }
}

#[test]
fn metric_validation_applies_to_every_per_coordinate_estimator() {
    // The coordinate-wise entry points validate independently of the total, so
    // each must reject the same bad metadata.
    let data = vec![vec![0.0, 1.0, 2.0], vec![2.0, 0.5, 1.0]];
    let bad = [
        CoordinateMetric::Linear,
        CoordinateMetric::Periodic { period: -2.0 },
    ];

    assert!(estimate_coordinate_entropy_with_metrics(data.clone(), 3, &bad).is_err());
    assert!(estimate_coordinate_mutual_information_with_metrics(data.clone(), 3, &bad).is_err());
    assert!(estimate_coordinate_mie_entropy_with_metrics(data.clone(), 3, &bad).is_err());
    assert!(estimate_coordinate_entropy_with_metrics(data, 3, &linear(5)).is_err());
}

// ---------------------------------------------------------------------------
// Expansion order
// ---------------------------------------------------------------------------

#[test]
fn unsupported_expansion_orders_are_rejected() {
    let data = vec![vec![0.1, 0.4, 0.8], vec![1.0, 1.3, 1.9]];
    for order in [0usize, 5, 100] {
        expect_error(
            calculate_entropy_from_data_with_order(data.clone(), 3, order),
            "unsupported MIE order",
            &format!("order {order}"),
        );
    }
}

#[test]
fn an_order_above_the_coordinate_count_falls_back_to_the_joint_entropy() {
    // With fewer coordinates than the requested order the expansion has nothing
    // left to truncate, so it must clamp to the full joint entropy rather than
    // underflowing a coefficient.
    let data = vec![vec![0.1, 0.4, 0.8, 1.1], vec![1.0, 1.3, 1.9, 2.2]];
    let order_two = calculate_entropy_from_data_with_order(data.clone(), 4, 2).unwrap();
    for order in [3usize, 4] {
        let clamped = calculate_entropy_from_data_with_order(data.clone(), 4, order).unwrap();
        assert!(
            (clamped - order_two).abs() < 1e-12,
            "order {order} on two coordinates gave {clamped}, expected {order_two}"
        );
    }

    // Likewise a single coordinate collapses to its own marginal entropy.
    let single = vec![vec![0.1, 0.4, 0.8, 1.1]];
    let first = calculate_entropy_from_data_with_order(single.clone(), 4, 1).unwrap();
    let fourth = calculate_entropy_from_data_with_order(single, 4, 4).unwrap();
    assert!((first - fourth).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Constant selection
// ---------------------------------------------------------------------------

#[test]
fn both_finite_sample_conventions_are_accepted_everywhere() {
    let data = vec![vec![0.1, 0.4, 0.8, 1.1], vec![1.0, 1.3, 1.9, 2.2]];
    let metrics = linear(2);
    for constant in [
        FiniteSampleConstant::PythonCompatibleAsymptotic,
        FiniteSampleConstant::Exact,
    ] {
        for order in 1..=2 {
            let value = calculate_entropy_from_data_with_metrics_and_constant(
                data.clone(),
                4,
                order,
                &metrics,
                constant,
            )
            .expect("valid input should be accepted");
            assert!(value.is_finite(), "{constant:?} order {order} gave {value}");
        }
    }
}

// ---------------------------------------------------------------------------
// Direct nearest-neighbor entry points
// ---------------------------------------------------------------------------

#[test]
fn nearest_neighbor_helpers_reject_mismatched_lengths() {
    use nn_entropy::calc_two_d_nn;
    let error = calc_two_d_nn(&[0.0, 1.0, 2.0], &[0.0, 1.0])
        .expect_err("mismatched lengths should be rejected");
    assert!(error.contains("equal length"), "unexpected error {error:?}");
}

#[test]
fn nearest_neighbor_helpers_reject_a_single_point() {
    let error = calc_one_d_nn_with_metric(&[1.0], CoordinateMetric::Linear)
        .expect_err("a single point should be rejected");
    assert!(
        error.contains("at least two points"),
        "unexpected error {error:?}"
    );
}
