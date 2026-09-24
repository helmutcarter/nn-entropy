//! Invariance and equivariance properties implied by the underlying theory.
//!
//! Differential entropy is invariant under translation, reflection and
//! relabeling, and shifts by a known amount under scaling. Mutual information
//! is invariant under separate affine maps of each coordinate. These hold
//! exactly for the true quantities, and the nearest-neighbor estimator inherits
//! them exactly (up to floating-point association) because it depends on the
//! samples only through inter-point distances.
//!
//! A tolerance here is therefore a floating-point budget, not a statistical
//! one -- unlike `tests/analytic.rs`, none of these depend on sample size.

mod common;

use common::Rng;
use nn_entropy::{
    CoordinateMetric, calculate_entropy_from_data_with_metrics,
    estimate_coordinate_mutual_information_with_metrics,
};

const PERIOD: f64 = std::f64::consts::TAU;
const FRAMES: usize = 600;
const COORDS: usize = 5;

fn linear(n: usize) -> Vec<CoordinateMetric> {
    vec![CoordinateMetric::Linear; n]
}

fn sample(seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    (0..COORDS)
        .map(|_| rng.normal_vec(FRAMES, 0.0, 1.0))
        .collect()
}

fn entropy(data: Vec<Vec<f64>>, order: usize, metrics: &[CoordinateMetric]) -> f64 {
    calculate_entropy_from_data_with_metrics(data, FRAMES, order, metrics).unwrap()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference < tolerance,
        "{what}: got {actual}, expected {expected}, difference {difference} exceeds {tolerance}"
    );
}

// ---------------------------------------------------------------------------
// Translation, reflection, relabeling
// ---------------------------------------------------------------------------

#[test]
fn entropy_is_invariant_under_translation() {
    let data = sample(1);
    let metrics = linear(COORDS);
    let offsets = [3.0, -7.5, 0.25, 100.0, -0.5];

    for order in 1..=4 {
        let base = entropy(data.clone(), order, &metrics);
        let shifted: Vec<Vec<f64>> = data
            .iter()
            .zip(offsets)
            .map(|(coordinate, offset)| coordinate.iter().map(|v| v + offset).collect())
            .collect();
        let moved = entropy(shifted, order, &metrics);
        assert_close(moved, base, 1e-9, &format!("translation at order {order}"));
    }
}

#[test]
fn entropy_is_invariant_under_reflection() {
    // H(-X) = H(X): the density is mirrored, distances are unchanged.
    let data = sample(2);
    let metrics = linear(COORDS);

    for order in 1..=4 {
        let base = entropy(data.clone(), order, &metrics);
        let mirrored: Vec<Vec<f64>> = data
            .iter()
            .map(|coordinate| coordinate.iter().map(|v| -v).collect())
            .collect();
        assert_close(
            entropy(mirrored, order, &metrics),
            base,
            1e-9,
            &format!("reflection at order {order}"),
        );
    }
}

#[test]
fn entropy_is_invariant_under_coordinate_relabeling() {
    // The expansion is a sum over unordered subsets, so permuting the
    // coordinates must leave the total untouched.
    let data = sample(3);
    let metrics = linear(COORDS);
    let permutation = [3usize, 0, 4, 1, 2];

    for order in 1..=4 {
        let base = entropy(data.clone(), order, &metrics);
        let permuted: Vec<Vec<f64>> = permutation.iter().map(|&i| data[i].clone()).collect();
        assert_close(
            entropy(permuted, order, &metrics),
            base,
            1e-9,
            &format!("coordinate permutation at order {order}"),
        );
    }
}

#[test]
fn entropy_is_invariant_under_frame_reordering() {
    // The estimator treats frames as an unordered sample, so shuffling them
    // must not move the answer. This also guards against any accidental
    // dependence on trajectory ordering.
    let data = sample(4);
    let metrics = linear(COORDS);
    let mut rng = Rng::new(99);

    // Fisher-Yates over frame indices, applied identically to every coordinate.
    let mut order_index: Vec<usize> = (0..FRAMES).collect();
    for i in (1..FRAMES).rev() {
        let j = (rng.uniform() * (i + 1) as f64) as usize;
        order_index.swap(i, j.min(i));
    }
    let shuffled: Vec<Vec<f64>> = data
        .iter()
        .map(|coordinate| order_index.iter().map(|&k| coordinate[k]).collect())
        .collect();

    for order in 1..=4 {
        assert_close(
            entropy(shuffled.clone(), order, &metrics),
            entropy(data.clone(), order, &metrics),
            1e-9,
            &format!("frame reordering at order {order}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Scaling equivariance
// ---------------------------------------------------------------------------

#[test]
fn scaling_one_coordinate_is_exact_at_first_order() {
    // Each first-order term is a 1-D marginal, so scaling coordinate k by c
    // scales every one of its nearest-neighbor distances by exactly c and
    // shifts the total by exactly ln(c).
    let data = sample(5);
    let metrics = linear(COORDS);

    for scale in [2.0_f64, 0.1, 0.5] {
        let base = entropy(data.clone(), 1, &metrics);
        let mut scaled = data.clone();
        scaled[2] = scaled[2].iter().map(|v| v * scale).collect();
        assert_close(
            entropy(scaled, 1, &metrics),
            base + scale.ln(),
            1e-9,
            &format!("scale coordinate 2 by {scale} at order 1"),
        );
    }
}

#[test]
fn anisotropic_rescaling_only_approximately_preserves_higher_order_totals() {
    // The true differential entropy shifts by exactly ln(c) when one coordinate
    // is rescaled, but the estimator does not: stretching a single axis changes
    // which points are nearest in the joint space, so every term of order >= 2
    // is perturbed. This is a real property of the nearest-neighbor estimator,
    // not a defect, and it matters here because BAT coordinates mix units --
    // bonds in angstrom against angles and torsions in radians.
    //
    // Measured deviations at N = 600, 5 coordinates: ~0.05 at order 2 for a
    // factor of 2, growing to ~0.9 at order 4 for a factor of 10. The bound
    // below is a regression guard at moderate anisotropy, not a claim of
    // invariance.
    let data = sample(5);
    let metrics = linear(COORDS);

    for scale in [2.0_f64, 0.5] {
        for order in 2..=4 {
            let base = entropy(data.clone(), order, &metrics);
            let mut scaled = data.clone();
            scaled[2] = scaled[2].iter().map(|v| v * scale).collect();
            let deviation = entropy(scaled, order, &metrics) - (base + scale.ln());
            println!("anisotropic scale {scale} at order {order}: deviation {deviation:+.6}");
            assert!(
                deviation.abs() < 0.25,
                "scale {scale} at order {order}: deviation {deviation} is far larger than \
                 the expected estimator error"
            );
        }
    }
}

#[test]
fn scaling_all_coordinates_shifts_the_total_by_n_log_scale() {
    // Scaling every coordinate by c scales the joint volume element by c^n.
    let data = sample(6);
    let metrics = linear(COORDS);
    let scale = 3.0_f64;

    for order in 1..=4 {
        let base = entropy(data.clone(), order, &metrics);
        let scaled: Vec<Vec<f64>> = data
            .iter()
            .map(|coordinate| coordinate.iter().map(|v| v * scale).collect())
            .collect();
        assert_close(
            entropy(scaled, order, &metrics),
            base + COORDS as f64 * scale.ln(),
            1e-9,
            &format!("scale all coordinates at order {order}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Mutual information
// ---------------------------------------------------------------------------

#[test]
fn mutual_information_is_exactly_invariant_under_a_common_rescaling() {
    // I(aX + b, -aY + d) = I(X, Y) holds exactly for the estimator: a shared
    // scale magnitude scales the joint distances uniformly, so the marginal and
    // joint shifts cancel. Sign flips and offsets change nothing at all.
    let mut rng = Rng::new(7);
    let (x, y) = rng.correlated_normal_pair(FRAMES, 0.7);
    let metrics = linear(2);

    let base = estimate_coordinate_mutual_information_with_metrics(
        vec![x.clone(), y.clone()],
        FRAMES,
        &metrics,
    )
    .unwrap();
    let mapped_x: Vec<f64> = x.iter().map(|v| 4.0 * v - 12.0).collect();
    let mapped_y: Vec<f64> = y.iter().map(|v| -4.0 * v + 3.0).collect();
    let mapped = estimate_coordinate_mutual_information_with_metrics(
        vec![mapped_x, mapped_y],
        FRAMES,
        &metrics,
    )
    .unwrap();

    assert_eq!(base.len(), 1);
    assert_close(
        mapped[0],
        base[0],
        1e-9,
        "common-rescaling mutual information",
    );
    assert!(
        base[0] > 0.2,
        "the correlated pair should carry real mutual information, got {}",
        base[0]
    );
}

#[test]
fn mutual_information_is_approximately_invariant_under_unequal_rescaling() {
    // Mutual information is invariant under separate affine maps in theory, but
    // unequal scales reshape the joint neighbor graph, so the estimator only
    // approximates it. Measured deviation here is ~0.008 against a value of
    // ~0.40, i.e. about 2%.
    let mut rng = Rng::new(7);
    let (x, y) = rng.correlated_normal_pair(FRAMES, 0.7);
    let metrics = linear(2);

    let base = estimate_coordinate_mutual_information_with_metrics(
        vec![x.clone(), y.clone()],
        FRAMES,
        &metrics,
    )
    .unwrap();
    let mapped_x: Vec<f64> = x.iter().map(|v| 4.0 * v - 12.0).collect();
    let mapped_y: Vec<f64> = y.iter().map(|v| -0.25 * v + 3.0).collect();
    let mapped = estimate_coordinate_mutual_information_with_metrics(
        vec![mapped_x, mapped_y],
        FRAMES,
        &metrics,
    )
    .unwrap();

    let deviation = mapped[0] - base[0];
    println!("unequal-rescaling mutual information deviation: {deviation:+.6}");
    assert!(
        deviation.abs() < 0.05,
        "unequal rescaling moved mutual information by {deviation}, far beyond estimator error"
    );
}

// ---------------------------------------------------------------------------
// Periodic coordinates
// ---------------------------------------------------------------------------

#[test]
fn periodic_entropy_is_invariant_under_rotation() {
    // The circle is homogeneous: rotating a periodic coordinate by an arbitrary
    // offset cannot change its entropy. A linear metric would fail this as soon
    // as the rotation pushes samples across the branch cut.
    let mut rng = Rng::new(8);
    let data: Vec<Vec<f64>> = (0..3)
        .map(|_| rng.uniform_vec(FRAMES, -std::f64::consts::PI, std::f64::consts::PI))
        .collect();
    let metrics = vec![CoordinateMetric::Periodic { period: PERIOD }; 3];

    let base = calculate_entropy_from_data_with_metrics(data.clone(), FRAMES, 2, &metrics).unwrap();

    for rotation in [0.3_f64, 1.9, -2.7, PERIOD / 2.0] {
        let rotated: Vec<Vec<f64>> = data
            .iter()
            .map(|coordinate| coordinate.iter().map(|v| v + rotation).collect())
            .collect();
        let moved = calculate_entropy_from_data_with_metrics(rotated, FRAMES, 2, &metrics).unwrap();
        assert_close(moved, base, 1e-9, &format!("rotation by {rotation}"));
    }
}

#[test]
fn periodic_entropy_is_invariant_under_whole_period_shifts() {
    // Canonicalization must fold any number of whole periods back onto itself.
    let mut rng = Rng::new(9);
    let data: Vec<Vec<f64>> = (0..3)
        .map(|_| rng.uniform_vec(FRAMES, -std::f64::consts::PI, std::f64::consts::PI))
        .collect();
    let metrics = vec![CoordinateMetric::Periodic { period: PERIOD }; 3];

    let base = calculate_entropy_from_data_with_metrics(data.clone(), FRAMES, 2, &metrics).unwrap();

    for multiple in [1.0_f64, -1.0, 5.0] {
        let shifted: Vec<Vec<f64>> = data
            .iter()
            .map(|coordinate| coordinate.iter().map(|v| v + multiple * PERIOD).collect())
            .collect();
        let moved = calculate_entropy_from_data_with_metrics(shifted, FRAMES, 2, &metrics).unwrap();
        assert_close(moved, base, 1e-9, &format!("shift by {multiple} periods"));
    }
}

// ---------------------------------------------------------------------------
// Floating-point behavior
// ---------------------------------------------------------------------------

#[test]
fn translation_precision_degrades_gracefully_with_distance_from_the_origin() {
    // Translation is an exact invariance, but computing differences between
    // large coordinates loses absolute precision, so the residual grows roughly
    // in proportion to the offset. Measured at N = 600, 5 coordinates, order 2:
    //
    //     offset 1e3 -> ~3e-9      offset 1e6 -> ~4e-7
    //     offset 1e4 -> ~2e-8      offset 1e8 -> ~3e-4
    //
    // BAT coordinates are bond lengths in angstrom and angles in radians, so
    // production values sit near unity and the exact test above applies; this
    // bounds the damage if a caller ever feeds unwrapped absolute positions.
    let data = sample(10);
    let metrics = linear(COORDS);
    let base = entropy(data.clone(), 2, &metrics);

    for (offset, budget) in [
        (1.0e3_f64, 1e-7),
        (1.0e4, 1e-6),
        (1.0e6, 1e-4),
        (1.0e8, 1e-2),
    ] {
        let shifted: Vec<Vec<f64>> = data
            .iter()
            .map(|coordinate| coordinate.iter().map(|v| v + offset).collect())
            .collect();
        let deviation = entropy(shifted, 2, &metrics) - base;
        println!("translation by {offset:>9.0e}: deviation {deviation:+.3e}");
        assert!(
            deviation.abs() < budget,
            "offset {offset}: deviation {deviation} exceeds budget {budget}"
        );
    }
}

#[test]
fn entropy_is_finite_for_a_very_narrow_distribution() {
    // A tiny spread drives nearest-neighbor distances toward zero, so the
    // logarithms grow large and negative. The result must stay finite and obey
    // the scaling law rather than collapsing to -inf.
    let data = sample(11);
    let metrics = linear(COORDS);
    let base = entropy(data.clone(), 2, &metrics);

    let scale = 1.0e-6_f64;
    let narrow: Vec<Vec<f64>> = data
        .iter()
        .map(|coordinate| coordinate.iter().map(|v| v * scale).collect())
        .collect();
    let narrow_entropy = entropy(narrow, 2, &metrics);

    assert!(
        narrow_entropy.is_finite(),
        "narrow distribution gave {narrow_entropy}"
    );
    assert_close(
        narrow_entropy,
        base + COORDS as f64 * scale.ln(),
        1e-9,
        "scaling law at 1e-6",
    );
}

#[test]
fn repeated_frames_do_not_produce_infinite_entropy() {
    // Exact duplicates measure to the nearest *distinct* point, so a trajectory
    // with repeated frames must stay finite instead of hitting ln(0).
    let mut rng = Rng::new(12);
    let mut data: Vec<Vec<f64>> = (0..3).map(|_| rng.normal_vec(200, 0.0, 1.0)).collect();
    for coordinate in data.iter_mut() {
        let head: Vec<f64> = coordinate[..50].to_vec();
        coordinate.extend(head);
    }
    let frames = data[0].len();
    let metrics = linear(3);

    let value = calculate_entropy_from_data_with_metrics(data, frames, 2, &metrics).unwrap();
    assert!(
        value.is_finite(),
        "duplicated frames produced a non-finite entropy: {value}"
    );
}
