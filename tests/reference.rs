//! Cross-checks of the optimized estimators against brute-force references.
//!
//! `REVIEW.md` asks that optimized implementations be compared against a simple
//! reference. The references here are written from the defining formulas and
//! use O(N^2) scans, so they share no code with the kd-tree search or the
//! aggregated mutual-information expansion they are checking.

mod common;

use common::{
    Rng, brute_force_joint_entropy, brute_force_log_nn_sum, choose, combinations,
    kl_constant_asymptotic,
};
use nn_entropy::{
    CoordinateMetric, calc_four_d_nn_with_metrics, calc_one_d_nn_with_metric,
    calc_three_d_nn_with_metrics, calc_two_d_nn_with_metrics,
    calculate_entropy_from_data_with_metrics,
};

const PERIOD: f64 = std::f64::consts::TAU;

fn linear(n: usize) -> Vec<CoordinateMetric> {
    vec![CoordinateMetric::Linear; n]
}

fn periodic(n: usize) -> Vec<CoordinateMetric> {
    vec![CoordinateMetric::Periodic { period: PERIOD }; n]
}

/// Truncated mutual-information expansion of the given order, evaluated
/// directly from its definition:
///
/// ```text
/// S_m = sum_{d=1}^{m} (-1)^(m-d) * C(n-d-1, m-d) * sum_{|A|=d} S_A
/// ```
///
/// Every joint entropy `S_A` comes from the brute-force reference, and the
/// binomial weights are recomputed here, so this shares nothing with the
/// coefficients baked into `calculate_entropy_from_data_with_metrics`.
fn mie_expansion_reference(data: &[Vec<f64>], metrics: &[CoordinateMetric], order: usize) -> f64 {
    let n = data.len();
    assert!(
        n > order,
        "the expansion coefficients all vanish when n == order; use n > order"
    );

    let mut total = 0.0;
    for subset_size in 1..=order {
        let sign = if (order - subset_size).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let weight = choose(n - subset_size - 1, order - subset_size);
        let mut subset_total = 0.0;
        for subset in combinations(n, subset_size) {
            let coordinates: Vec<&[f64]> = subset.iter().map(|&i| data[i].as_slice()).collect();
            let subset_metrics: Vec<CoordinateMetric> =
                subset.iter().map(|&i| metrics[i]).collect();
            subset_total += brute_force_joint_entropy(&coordinates, &subset_metrics);
        }
        total += sign * weight * subset_total;
    }
    total
}

fn sample_coordinates(seed: u64, n_coords: usize, n_frames: usize, high: f64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    (0..n_coords)
        .map(|_| rng.uniform_vec(n_frames, 0.0, high))
        .collect()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference < tolerance,
        "{what}: got {actual}, expected {expected}, difference {difference} exceeds {tolerance}"
    );
}

// ---------------------------------------------------------------------------
// Nearest-neighbor search vs. brute force
// ---------------------------------------------------------------------------

#[test]
fn one_d_nn_matches_brute_force_for_both_metrics() {
    let mut rng = Rng::new(11);
    // atan2 range, so canonicalization of negative values is exercised.
    let points = rng.uniform_vec(200, -std::f64::consts::PI, std::f64::consts::PI);

    for metrics in [linear(1), periodic(1)] {
        let actual = calc_one_d_nn_with_metric(&points, metrics[0]).unwrap();
        let expected = brute_force_log_nn_sum(&[points.as_slice()], &metrics);
        assert_close(actual, expected, 1e-9, "1D nearest neighbor");
    }
}

#[test]
fn two_d_nn_matches_brute_force_for_every_metric_combination() {
    let mut rng = Rng::new(12);
    let a = rng.uniform_vec(150, -std::f64::consts::PI, std::f64::consts::PI);
    let b = rng.uniform_vec(150, -std::f64::consts::PI, std::f64::consts::PI);

    for metrics in [
        [CoordinateMetric::Linear, CoordinateMetric::Linear],
        [
            CoordinateMetric::Linear,
            CoordinateMetric::Periodic { period: PERIOD },
        ],
        [
            CoordinateMetric::Periodic { period: PERIOD },
            CoordinateMetric::Linear,
        ],
        [
            CoordinateMetric::Periodic { period: PERIOD },
            CoordinateMetric::Periodic { period: PERIOD },
        ],
    ] {
        let actual = calc_two_d_nn_with_metrics(&a, &b, metrics).unwrap();
        let expected = brute_force_log_nn_sum(&[a.as_slice(), b.as_slice()], &metrics);
        assert_close(actual, expected, 1e-9, &format!("2D {metrics:?}"));
    }
}

#[test]
fn three_d_nn_matches_brute_force_with_mixed_metrics() {
    let mut rng = Rng::new(13);
    let a = rng.uniform_vec(120, -std::f64::consts::PI, std::f64::consts::PI);
    let b = rng.uniform_vec(120, -std::f64::consts::PI, std::f64::consts::PI);
    let c = rng.uniform_vec(120, -std::f64::consts::PI, std::f64::consts::PI);
    let metrics = [
        CoordinateMetric::Periodic { period: PERIOD },
        CoordinateMetric::Linear,
        CoordinateMetric::Periodic { period: PERIOD },
    ];

    let actual = calc_three_d_nn_with_metrics(&a, &b, &c, metrics).unwrap();
    let expected = brute_force_log_nn_sum(&[a.as_slice(), b.as_slice(), c.as_slice()], &metrics);
    assert_close(actual, expected, 1e-9, "3D mixed metrics");
}

#[test]
fn four_d_nn_matches_brute_force_with_mixed_metrics() {
    let mut rng = Rng::new(14);
    let a = rng.uniform_vec(100, -std::f64::consts::PI, std::f64::consts::PI);
    let b = rng.uniform_vec(100, -std::f64::consts::PI, std::f64::consts::PI);
    let c = rng.uniform_vec(100, -std::f64::consts::PI, std::f64::consts::PI);
    let d = rng.uniform_vec(100, -std::f64::consts::PI, std::f64::consts::PI);
    let metrics = [
        CoordinateMetric::Linear,
        CoordinateMetric::Periodic { period: PERIOD },
        CoordinateMetric::Periodic { period: PERIOD },
        CoordinateMetric::Linear,
    ];

    let actual = calc_four_d_nn_with_metrics(&a, &b, &c, &d, metrics).unwrap();
    let expected = brute_force_log_nn_sum(
        &[a.as_slice(), b.as_slice(), c.as_slice(), d.as_slice()],
        &metrics,
    );
    assert_close(actual, expected, 1e-9, "4D mixed metrics");
}

#[test]
fn nn_search_matches_brute_force_when_samples_are_duplicated() {
    // Exact ties must resolve to the nearest *distinct* point, which is the
    // documented behavior and the case the kd-tree self-exclusion handles.
    let mut rng = Rng::new(15);
    let mut a = rng.uniform_vec(40, 0.0, 1.0);
    let mut b = rng.uniform_vec(40, 0.0, 1.0);
    for index in 0..12 {
        a.push(a[index]);
        b.push(b[index]);
    }
    // A run of identical samples, beyond the kd-tree's bucket size.
    for _ in 0..10 {
        a.push(0.5);
        b.push(0.5);
    }

    let metrics = linear(2);
    let actual = calc_two_d_nn_with_metrics(&a, &b, [metrics[0], metrics[1]]).unwrap();
    let expected = brute_force_log_nn_sum(&[a.as_slice(), b.as_slice()], &metrics);
    assert_close(actual, expected, 1e-9, "2D with duplicate samples");
}

#[test]
fn periodic_nn_matches_brute_force_when_samples_straddle_the_branch_cut() {
    // Concentrate samples at both ends of the period so most nearest neighbors
    // are found only by wrapping.
    let mut rng = Rng::new(16);
    // Straddle the atan2 branch cut at +/-pi, as a real torsion does.
    let mut points = rng.uniform_vec(80, std::f64::consts::PI - 0.05, std::f64::consts::PI);
    points.extend(rng.uniform_vec(80, -std::f64::consts::PI, -std::f64::consts::PI + 0.05));

    let metric = CoordinateMetric::Periodic { period: PERIOD };
    let actual = calc_one_d_nn_with_metric(&points, metric).unwrap();
    let expected = brute_force_log_nn_sum(&[points.as_slice()], &[metric]);
    assert_close(actual, expected, 1e-9, "periodic branch cut");

    // And the wrapped result must be strictly tighter than the linear one.
    let linear_result = calc_one_d_nn_with_metric(&points, CoordinateMetric::Linear).unwrap();
    assert!(
        actual < linear_result,
        "wrapping should shorten distances: periodic {actual} vs linear {linear_result}"
    );
}

// ---------------------------------------------------------------------------
// MIE expansion coefficients
// ---------------------------------------------------------------------------

#[test]
fn first_order_entropy_matches_sum_of_marginals() {
    let data = sample_coordinates(20, 5, 60, 1.0);
    let metrics = linear(5);
    let n_frames = data[0].len();

    let expected: f64 = data
        .iter()
        .map(|c| brute_force_joint_entropy(&[c.as_slice()], &[CoordinateMetric::Linear]))
        .sum();
    let actual = calculate_entropy_from_data_with_metrics(data, n_frames, 1, &metrics).unwrap();
    assert_close(actual, expected, 1e-9, "order 1");
    let _ = kl_constant_asymptotic(n_frames, 1);
}

#[test]
fn second_order_expansion_matches_inclusion_exclusion_reference() {
    // Six coordinates at order 2: the marginal coefficient is -(n-2) = -4, so
    // unlike an n == order test this actually exercises the weight.
    let data = sample_coordinates(21, 6, 50, 1.0);
    let metrics = linear(6);
    let n_frames = data[0].len();

    let expected = mie_expansion_reference(&data, &metrics, 2);
    let actual = calculate_entropy_from_data_with_metrics(data, n_frames, 2, &metrics).unwrap();
    assert_close(actual, expected, 1e-8, "order 2 expansion");
}

#[test]
fn third_order_expansion_matches_inclusion_exclusion_reference() {
    // n = 6, order 3 => pair weight -(n-3) = -3 and marginal weight
    // (n-2)(n-3)/2 = 6. Both are non-zero only because n > order.
    let data = sample_coordinates(22, 6, 45, 1.0);
    let metrics = linear(6);
    let n_frames = data[0].len();

    let expected = mie_expansion_reference(&data, &metrics, 3);
    let actual = calculate_entropy_from_data_with_metrics(data, n_frames, 3, &metrics).unwrap();
    assert_close(actual, expected, 1e-8, "order 3 expansion");
}

#[test]
fn fourth_order_expansion_matches_inclusion_exclusion_reference() {
    // n = 6, order 4 => triple weight -(n-4) = -2, pair weight
    // (n-3)(n-4)/2 = 3, marginal weight -(n-2)(n-3)(n-4)/6 = -4.
    let data = sample_coordinates(23, 6, 45, 1.0);
    let metrics = linear(6);
    let n_frames = data[0].len();

    let expected = mie_expansion_reference(&data, &metrics, 4);
    let actual = calculate_entropy_from_data_with_metrics(data, n_frames, 4, &metrics).unwrap();
    assert_close(actual, expected, 1e-8, "order 4 expansion");
}

#[test]
fn expansion_coefficients_hold_for_a_seventh_coordinate() {
    // A second arity, so a coefficient that happens to be right at n = 6 but
    // wrong in general cannot slip through.
    let data = sample_coordinates(24, 7, 40, 1.0);
    let metrics = linear(7);
    let n_frames = data[0].len();

    for order in 2..=4 {
        let expected = mie_expansion_reference(&data, &metrics, order);
        let actual =
            calculate_entropy_from_data_with_metrics(data.clone(), n_frames, order, &metrics)
                .unwrap();
        assert_close(actual, expected, 1e-8, &format!("order {order} at n = 7"));
    }
}

#[test]
fn expansion_coefficients_hold_under_periodic_metrics() {
    let mut rng = Rng::new(25);
    let data: Vec<Vec<f64>> = (0..5)
        .map(|_| rng.uniform_vec(45, -std::f64::consts::PI, std::f64::consts::PI))
        .collect();
    let metrics = periodic(5);
    let n_frames = data[0].len();

    for order in 2..=4 {
        let expected = mie_expansion_reference(&data, &metrics, order);
        let actual =
            calculate_entropy_from_data_with_metrics(data.clone(), n_frames, order, &metrics)
                .unwrap();
        assert_close(actual, expected, 1e-8, &format!("periodic order {order}"));
    }
}

#[test]
fn expansion_coefficients_hold_under_mixed_metrics() {
    let mut rng = Rng::new(26);
    let data: Vec<Vec<f64>> = (0..6)
        .map(|_| rng.uniform_vec(40, -std::f64::consts::PI, std::f64::consts::PI))
        .collect();
    let metrics = vec![
        CoordinateMetric::Linear,
        CoordinateMetric::Periodic { period: PERIOD },
        CoordinateMetric::Linear,
        CoordinateMetric::Periodic { period: PERIOD },
        CoordinateMetric::Periodic { period: PERIOD },
        CoordinateMetric::Linear,
    ];
    let n_frames = data[0].len();

    for order in 2..=4 {
        let expected = mie_expansion_reference(&data, &metrics, order);
        let actual =
            calculate_entropy_from_data_with_metrics(data.clone(), n_frames, order, &metrics)
                .unwrap();
        assert_close(actual, expected, 1e-8, &format!("mixed order {order}"));
    }
}
