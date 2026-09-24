//! Closed-form checks of the estimator against distributions whose differential
//! entropy is known analytically.
//!
//! These are the tests `REVIEW.md` asks for when it says to check analytically
//! tractable limiting cases: every expected value below is a textbook formula
//! evaluated in `tests/common/mod.rs`, derived without reference to the
//! implementation.
//!
//! The Kozachenko-Leonenko estimator is consistent but converges slowly, so the
//! tolerances are finite-sample budgets, not statements about exactness. Each
//! test averages several independent replicates to damp sampling noise, and
//! `estimator_error_shrinks_as_samples_grow` asserts the behavior that actually
//! matters: the error goes down as N goes up.

mod common;

use common::{
    Rng, bivariate_normal_entropy, bivariate_normal_mutual_information, exponential_entropy,
    normal_entropy, uniform_entropy,
};
use nn_entropy::{
    CoordinateMetric, calculate_entropy_from_data_with_metrics,
    estimate_coordinate_mutual_information_with_metrics,
};

const REPLICATES: usize = 5;

fn linear(n: usize) -> Vec<CoordinateMetric> {
    vec![CoordinateMetric::Linear; n]
}

/// Mean estimate over independent replicates, plus the largest single-replicate
/// deviation from the analytic value.
fn replicate_mean<F>(mut sample: F) -> (f64, f64)
where
    F: FnMut(u64) -> f64,
{
    let values: Vec<f64> = (0..REPLICATES).map(|i| sample(1000 + i as u64)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let spread = values
        .iter()
        .map(|v| (v - mean).abs())
        .fold(0.0_f64, f64::max);
    (mean, spread)
}

fn check(label: &str, estimated: f64, exact: f64, spread: f64, tolerance: f64) {
    let error = estimated - exact;
    println!(
        "{label:<46} estimate {estimated:>10.5}  exact {exact:>10.5}  \
         error {error:>+9.5}  spread {spread:>7.5}"
    );
    assert!(
        error.abs() < tolerance,
        "{label}: estimate {estimated} vs exact {exact}, error {error} exceeds {tolerance}"
    );
}

fn entropy_of(data: Vec<Vec<f64>>, order: usize) -> f64 {
    let frames = data[0].len();
    let metrics = linear(data.len());
    calculate_entropy_from_data_with_metrics(data, frames, order, &metrics).unwrap()
}

// ---------------------------------------------------------------------------
// One-dimensional marginals
// ---------------------------------------------------------------------------

#[test]
fn uniform_entropy_matches_log_width() {
    // H = ln(width); the unit interval is the normalization check, H = 0.
    for width in [1.0_f64, 5.0] {
        let (estimate, spread) = replicate_mean(|seed| {
            let mut rng = Rng::new(seed);
            entropy_of(vec![rng.uniform_vec(8000, 0.0, width)], 1)
        });
        check(
            &format!("uniform(width {width})"),
            estimate,
            uniform_entropy(width),
            spread,
            0.02,
        );
    }
}

#[test]
fn normal_entropy_matches_closed_form() {
    // H = 1/2 ln(2 pi e sigma^2).
    for sigma in [1.0_f64, 3.0] {
        let (estimate, spread) = replicate_mean(|seed| {
            let mut rng = Rng::new(seed);
            entropy_of(vec![rng.normal_vec(8000, 0.0, sigma)], 1)
        });
        check(
            &format!("normal(sigma {sigma})"),
            estimate,
            normal_entropy(sigma),
            spread,
            0.02,
        );
    }
}

#[test]
fn exponential_entropy_matches_closed_form() {
    // H = 1 - ln(lambda). A skewed, hard-bounded density, unlike the others.
    for lambda in [1.0_f64, 2.5] {
        let (estimate, spread) = replicate_mean(|seed| {
            let mut rng = Rng::new(seed);
            entropy_of(vec![rng.exponential_vec(8000, lambda)], 1)
        });
        check(
            &format!("exponential(lambda {lambda})"),
            estimate,
            exponential_entropy(lambda),
            spread,
            0.03,
        );
    }
}

#[test]
fn circular_uniform_entropy_matches_log_period() {
    // On a circle of circumference P the uniform entropy is ln(P). This checks
    // the periodic metric and the constant together.
    let period = std::f64::consts::TAU;
    let (estimate, spread) = replicate_mean(|seed| {
        let mut rng = Rng::new(seed);
        let data = vec![rng.uniform_vec(8000, -std::f64::consts::PI, std::f64::consts::PI)];
        let metrics = vec![CoordinateMetric::Periodic { period }];
        calculate_entropy_from_data_with_metrics(data, 8000, 1, &metrics).unwrap()
    });
    check(
        "circular uniform(period tau)",
        estimate,
        period.ln(),
        spread,
        0.02,
    );
}

// ---------------------------------------------------------------------------
// Joint distributions
// ---------------------------------------------------------------------------

#[test]
fn independent_coordinates_are_additive() {
    // For independent coordinates the joint entropy is the sum of the
    // marginals, so order 2 must reproduce sum ln(width_i).
    let widths = [1.0_f64, 2.0, 4.0];
    let (estimate, spread) = replicate_mean(|seed| {
        let mut rng = Rng::new(seed);
        let data: Vec<Vec<f64>> = widths
            .iter()
            .map(|&w| rng.uniform_vec(6000, 0.0, w))
            .collect();
        entropy_of(data, 2)
    });
    let exact: f64 = widths.iter().map(|&w| uniform_entropy(w)).sum();
    check(
        "independent uniforms (order 2)",
        estimate,
        exact,
        spread,
        0.05,
    );
}

#[test]
fn bivariate_normal_joint_entropy_matches_closed_form() {
    // H = ln(2 pi e) + 1/2 ln(1 - rho^2) for unit marginals.
    for rho in [0.0_f64, 0.6, 0.9] {
        let (estimate, spread) = replicate_mean(|seed| {
            let mut rng = Rng::new(seed);
            let (x, y) = rng.correlated_normal_pair(6000, rho);
            // Order 2 on exactly two coordinates is the joint entropy.
            entropy_of(vec![x, y], 2)
        });
        check(
            &format!("bivariate normal joint(rho {rho})"),
            estimate,
            bivariate_normal_entropy(rho),
            spread,
            0.05,
        );
    }
}

#[test]
fn bivariate_normal_mutual_information_matches_closed_form() {
    // I = -1/2 ln(1 - rho^2). Independence must give zero.
    for rho in [0.0_f64, 0.5, 0.8] {
        let (estimate, spread) = replicate_mean(|seed| {
            let mut rng = Rng::new(seed);
            let (x, y) = rng.correlated_normal_pair(6000, rho);
            let metrics = linear(2);
            let mi =
                estimate_coordinate_mutual_information_with_metrics(vec![x, y], 6000, &metrics)
                    .unwrap();
            assert_eq!(mi.len(), 1);
            mi[0]
        });
        check(
            &format!("bivariate normal MI(rho {rho})"),
            estimate,
            bivariate_normal_mutual_information(rho),
            spread,
            0.05,
        );
    }
}

#[test]
fn mutual_information_vanishes_for_independent_coordinates() {
    // I(X;Y) = 0 exactly when X and Y are independent, but the k=1 KL estimator
    // carries a small positive bias at finite N: measured over 20 replicates at
    // N = 4000, the per-pair mean sits between -0.003 and +0.018. The single-
    // replicate spread is ~0.04, so the mean over a handful of replicates is
    // only pinned to roughly +/-0.02 -- hence 12 replicates and a 0.05 band.
    // That band still discriminates sharply: the rho = 0.5 case in
    // `bivariate_normal_mutual_information_matches_closed_form` is 0.144.
    const PAIRS: usize = 6;
    const INDEPENDENT_REPLICATES: usize = 12;
    let mut totals = [0.0_f64; PAIRS];

    for replicate in 0..INDEPENDENT_REPLICATES {
        let mut rng = Rng::new(2000 + replicate as u64);
        let data: Vec<Vec<f64>> = (0..4).map(|_| rng.normal_vec(4000, 0.0, 1.0)).collect();
        let metrics = linear(4);
        let mi = estimate_coordinate_mutual_information_with_metrics(data, 4000, &metrics).unwrap();
        assert_eq!(mi.len(), PAIRS);
        for (pair, value) in mi.iter().enumerate() {
            assert!(
                value.abs() < 0.25,
                "replicate {replicate}, pair {pair}: MI {value} is grossly far from zero"
            );
            totals[pair] += value;
        }
    }

    for (pair, total) in totals.iter().enumerate() {
        let mean = total / INDEPENDENT_REPLICATES as f64;
        println!("independent MI pair {pair}: mean over replicates {mean:>+9.5}");
        assert!(
            mean.abs() < 0.05,
            "pair {pair}: mean MI over {INDEPENDENT_REPLICATES} replicates was {mean}, expected ~0"
        );
    }
}

// ---------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------

#[test]
fn estimator_error_shrinks_as_samples_grow() {
    // Consistency is the property that actually matters: the tolerances above
    // are budgets at a fixed N, but the error must fall as N rises.
    let exact = normal_entropy(1.0);
    let mut errors = Vec::new();
    for n_frames in [500_usize, 2000, 8000] {
        let (estimate, _) = replicate_mean(|seed| {
            let mut rng = Rng::new(seed);
            entropy_of(vec![rng.normal_vec(n_frames, 0.0, 1.0)], 1)
        });
        let error = (estimate - exact).abs();
        println!("normal(sigma 1) N = {n_frames:<6} error {error:.5}");
        errors.push(error);
    }
    assert!(
        errors[2] < errors[0],
        "error did not shrink from N=500 ({:.5}) to N=8000 ({:.5})",
        errors[0],
        errors[2]
    );
    assert!(
        errors[2] < 0.02,
        "error at N=8000 was {:.5}, above the expected budget",
        errors[2]
    );
}
