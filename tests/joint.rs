//! Joint nearest-neighbor entropy at a dimension chosen at run time, and the
//! pruned search over periodic images that makes higher dimensions affordable.
//!
//! Expected values come from the brute-force O(N^2) reference in
//! `tests/common`, from the const-generic search the dynamic one dispatches
//! to, or from closed forms.

mod common;

use common::{Rng, brute_force_log_nn_sum, brute_force_log_nn_sum_windowed, unit_ball_volume};
use nn_entropy::{
    CoordinateMetric, EntropyOptions, FiniteSampleConstant, MAX_JOINT_DIMENSION,
    calc_joint_nn_dynamic, calc_joint_nn_with_window, calculate_entropy, estimate_joint_entropy,
    log_unit_ball_volume,
};

const PERIOD: f64 = std::f64::consts::TAU;
const PERIODIC: CoordinateMetric = CoordinateMetric::Periodic { period: PERIOD };

/// `n` points in `d` dimensions; periodic coordinates cluster near 0 and P so
/// that many nearest neighbors lie across the wrap.
fn sample(rng: &mut Rng, d: usize, n: usize) -> (Vec<Vec<f64>>, Vec<CoordinateMetric>) {
    let metrics: Vec<CoordinateMetric> = (0..d)
        .map(|k| {
            if k % 3 == 2 {
                CoordinateMetric::Linear
            } else {
                PERIODIC
            }
        })
        .collect();
    let data = metrics
        .iter()
        .map(|metric| match metric {
            CoordinateMetric::Periodic { .. } => (0..n)
                .map(|_| (0.4 * rng.normal()).rem_euclid(PERIOD))
                .collect(),
            CoordinateMetric::Linear => rng.normal_vec(n, 0.0, 1.0),
        })
        .collect();
    (data, metrics)
}

fn slices(data: &[Vec<f64>]) -> Vec<&[f64]> {
    data.iter().map(Vec::as_slice).collect()
}

#[test]
fn unit_ball_volume_matches_the_recurrence() {
    for d in 1..=12 {
        let expected = unit_ball_volume(d).ln();
        let actual = log_unit_ball_volume(d);
        if d <= 4 {
            assert_eq!(actual, expected, "d = {d}");
        } else {
            assert!(
                (actual - expected).abs() < 1e-12,
                "d = {d}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn dynamic_dispatch_equals_the_const_generic_search() {
    let mut rng = Rng::new(51);
    let (data, metrics) = sample(&mut rng, 4, 500);
    let columns = slices(&data);
    assert_eq!(
        calc_joint_nn_dynamic(&columns[..1], &metrics[..1], 0).unwrap(),
        calc_joint_nn_with_window::<1>([columns[0]], [metrics[0]], 0).unwrap()
    );
    assert_eq!(
        calc_joint_nn_dynamic(&columns[..2], &metrics[..2], 0).unwrap(),
        calc_joint_nn_with_window::<2>([columns[0], columns[1]], [metrics[0], metrics[1]], 0)
            .unwrap()
    );
    assert_eq!(
        calc_joint_nn_dynamic(&columns, &metrics, 2).unwrap(),
        calc_joint_nn_with_window::<4>(
            columns.clone().try_into().unwrap(),
            metrics.clone().try_into().unwrap(),
            2
        )
        .unwrap()
    );
}

#[test]
fn higher_dimensions_match_brute_force_across_the_wrap() {
    let mut rng = Rng::new(52);
    for d in 5..=MAX_JOINT_DIMENSION {
        let (data, metrics) = sample(&mut rng, d, 400);
        let columns = slices(&data);
        let actual = calc_joint_nn_dynamic(&columns, &metrics, 0).unwrap();
        let expected = brute_force_log_nn_sum(&columns, &metrics);
        assert!(
            (actual - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "d = {d}: {actual} vs {expected}"
        );
    }
}

#[test]
fn higher_dimensions_respect_the_theiler_window() {
    let mut rng = Rng::new(53);
    let (data, metrics) = sample(&mut rng, 6, 300);
    let columns = slices(&data);
    for window in [1, 4] {
        let actual = calc_joint_nn_dynamic(&columns, &metrics, window).unwrap();
        let expected = brute_force_log_nn_sum_windowed(&columns, &metrics, window);
        assert!(
            (actual - expected).abs() < 1e-9 * expected.abs(),
            "window {window}"
        );
    }
}

#[test]
fn one_coordinate_equals_the_first_order_entropy() {
    let mut rng = Rng::new(54);
    let (data, metrics) = sample(&mut rng, 1, 2_000);
    let options = EntropyOptions {
        metrics: Some(&metrics),
        ..Default::default()
    };
    assert_eq!(
        estimate_joint_entropy(&data, 2_000, &options).unwrap(),
        calculate_entropy(&data, 2_000, 1, &options).unwrap()
    );
}

#[test]
fn a_uniform_periodic_box_has_zero_entropy() {
    // On a torus of unit periods a uniform sample has entropy ln 1 = 0, with
    // no boundary bias, so this checks the high-dimensional constant.
    let mut rng = Rng::new(55);
    let d = 6;
    let n = 40_000;
    let data: Vec<Vec<f64>> = (0..d).map(|_| rng.uniform_vec(n, 0.0, 1.0)).collect();
    let metrics = vec![CoordinateMetric::Periodic { period: 1.0 }; d];
    for constant in [
        FiniteSampleConstant::Asymptotic,
        FiniteSampleConstant::Exact,
    ] {
        let options = EntropyOptions {
            metrics: Some(&metrics),
            constant,
            ..Default::default()
        };
        let entropy = estimate_joint_entropy(&data, n, &options).unwrap();
        assert!(entropy.abs() < 0.05, "{constant:?}: {entropy}");
    }
}

#[test]
fn too_many_coordinates_are_refused() {
    let mut rng = Rng::new(56);
    let (data, metrics) = sample(&mut rng, MAX_JOINT_DIMENSION + 1, 50);
    let options = EntropyOptions {
        metrics: Some(&metrics),
        ..Default::default()
    };
    let err = estimate_joint_entropy(&data, 50, &options).unwrap_err();
    assert!(
        err.contains("use the mutual information expansion"),
        "{err}"
    );
}
