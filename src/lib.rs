//! Configurational entropy of molecules from molecular dynamics trajectories.
//!
//! The crate estimates the entropy of a molecule's internal coordinates
//! (bond lengths, bond angles and torsions) from the frames of a simulation.
//! It reads Amber topologies and NetCDF trajectories, converts them to
//! internal coordinates, and offers several estimators that work on any array
//! of sampled coordinates. A command-line program and Python bindings expose
//! the same estimators; the [user guide](https://helmutcarter.github.io/nn-entropy/) documents them and the
//! methods.
//!
//! # Choosing an estimator
//!
//! - **Boltzmann quasi-harmonic (BQH)**,
//!   [`quasi_harmonic::estimate_quasi_harmonic_entropy`]: each coordinate's own
//!   entropy from its actual distribution, plus a Gaussian correlation term.
//!   The most accurate estimator in the crate on the systems tested so far; see
//!   the [validation results](https://helmutcarter.github.io/nn-entropy/validation.html).
//! - **Nearest-neighbor mutual information expansion (NN MIE)**,
//!   [`calculate_entropy`]: nearest-neighbor entropies of single coordinates
//!   and of pairs (up to sets of four), combined by a truncated expansion.
//!   Makes no assumption about the shape of any distribution, but the sum of
//!   many pairwise terms is noisy.
//! - **Split**, [`split::estimate_split_entropy`]: soft torsions (about
//!   rotatable bonds) by a nearest-neighbor estimate, the stiff coordinates by
//!   a Gaussian, plus a coupling term; for molecules with several coupled
//!   rotatable bonds.
//! - **Quasi-harmonic (QH)**, the same function as BQH: a Gaussian fit to
//!   every coordinate. Fast, but overestimates the entropy of multiwell
//!   torsions.
//! - [`legacy::legacy_entropy`] reproduces the original Python implementation
//!   bit for bit and exists only for comparison with historical results.
//!
//! Every estimator returns a differential entropy in natural-log units
//! (units of k_B). That is the entropy of the coordinates as sampled: it does
//! not include the BAT Jacobian, momentum, or a standard state, so it is
//! meaningful as a difference between ensembles of the same molecule, not as an
//! absolute thermodynamic entropy. [`bat_library::InternalCoordinates::log_jacobian`]
//! supplies the Jacobian term when the Cartesian entropy is wanted.
//!
//! # From a trajectory
//!
//! ```no_run
//! use std::path::Path;
//!
//! use nn_entropy::EntropyOptions;
//! use nn_entropy::bat_library::InternalCoordinates;
//! use nn_entropy::quasi_harmonic::{QuasiHarmonicOptions, estimate_quasi_harmonic_entropy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Heavy-atom bonds, angles, and torsions with phase angles: the defaults.
//! let mut internal = InternalCoordinates::new(Path::new("system.parm7"))?;
//! internal.calculate_internal_coords(Path::new("trajectory.nc"), usize::MAX, false)?;
//!
//! // Every frame (start 0, stride 1), as data[coordinate][frame].
//! let data = internal.columns(0, 1)?;
//! let frames = data[0].len();
//! let metrics = internal.coordinate_metrics(); // torsions are periodic
//! let options = EntropyOptions { metrics: Some(&metrics), ..Default::default() };
//!
//! let entropy = estimate_quasi_harmonic_entropy(
//!     &data,
//!     frames,
//!     &QuasiHarmonicOptions::default(),
//!     &options,
//! )?;
//! println!("BQH entropy: {} k_B", entropy.boltzmann_quasi_harmonic);
//! # Ok(())
//! # }
//! ```
//!
//! # From arrays
//!
//! Any coordinates work, one `Vec<f64>` per coordinate:
//!
//! ```
//! use nn_entropy::{CoordinateMetric, EntropyOptions, calculate_entropy};
//!
//! // An angle sweeping the circle and a coordinate that follows it.
//! let phi: Vec<f64> = (0..2000).map(|i| (i as f64 * 2.399963).rem_euclid(std::f64::consts::TAU)).collect();
//! let x: Vec<f64> = phi.iter().enumerate().map(|(i, p)| p.cos() + 0.1 * ((i * 7919) % 101) as f64 / 101.0).collect();
//!
//! let metrics = [
//!     CoordinateMetric::Periodic { period: std::f64::consts::TAU },
//!     CoordinateMetric::Linear,
//! ];
//! let options = EntropyOptions { metrics: Some(&metrics), ..Default::default() };
//! let first_order = calculate_entropy(&[phi.clone(), x.clone()], 2000, 1, &options)?;
//! let second_order = calculate_entropy(&[phi, x], 2000, 2, &options)?;
//! // The second-order term subtracts the mutual information of the pair.
//! assert!(second_order < first_order);
//! # Ok::<(), String>(())
//! ```
//!
//! # Modules
//!
//! - [`bat_library`]: Amber topology and trajectory reading, solute selection,
//!   and the bond-angle-torsion coordinates, with the options in
//!   [`bat_library::CoordinateOptions`].
//! - [`quasi_harmonic`]: the QH and BQH estimators.
//! - [`split`]: the split (soft/hard) estimator, and [`basins`]: basin
//!   detection for periodic coordinates.
//! - [`legacy`]: the reproduction of the original Python implementation.
//! - [`pyo3_api`]: the Python extension module.

#![warn(missing_docs)]

use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use rayon::prelude::*;
use std::num::NonZero;

pub mod basins;
pub mod bat_library;
pub mod legacy;
pub mod pyo3_api;
pub mod quasi_harmonic;
pub mod split;

/// Metric used for one internal-coordinate dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateMetric {
    /// Ordinary distance on the real line, for bond lengths and angles.
    Linear,
    /// Distance on a circle of circumference `period`, for torsions (period
    /// 2 pi). Values may lie anywhere; they are reduced modulo the period.
    Periodic {
        /// Length of one full turn, in the coordinate's units.
        period: f64,
    },
}

/// Sample-size term used in the Kozachenko-Leonenko entropy constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FiniteSampleConstant {
    /// Large-sample approximation ln(N) + gamma, the convention of the original
    /// Python implementation.
    #[default]
    Asymptotic,
    /// Exact k=1 finite-sample term: psi(N) - psi(1) = H_(N-1).
    Exact,
}

impl CoordinateMetric {
    fn validate(self, index: usize) -> Result<(), String> {
        if let Self::Periodic { period } = self
            && (!period.is_finite() || period <= 0.0)
        {
            return Err(format!(
                "period for coordinate {index} must be finite and positive"
            ));
        }
        Ok(())
    }

    fn canonicalize(self, value: f64) -> f64 {
        match self {
            Self::Linear => value,
            Self::Periodic { period } => value.rem_euclid(period),
        }
    }
}

fn validate_one_d_data(one_d_data: &[Vec<f64>], frames_end: usize) -> Result<(), String> {
    if one_d_data.is_empty() {
        return Err("no coordinate data provided".to_string());
    }
    if frames_end < 2 {
        return Err("need at least two frames for entropy estimation".to_string());
    }
    for (idx, coord) in one_d_data.iter().enumerate() {
        if coord.len() < frames_end {
            return Err(format!(
                "coordinate {idx} has {} frames, expected at least {frames_end}",
                coord.len()
            ));
        }
        let slice = &coord[..frames_end];
        if slice.iter().any(|v| !v.is_finite()) {
            return Err(format!("coordinate {idx} contains non-finite values"));
        }
        let mut unique = slice.to_vec();
        unique.sort_by(|a, b| a.total_cmp(b));
        unique.dedup();
        if unique.len() < 2 {
            return Err(format!(
                "coordinate {idx} must contain at least two unique values"
            ));
        }
    }
    Ok(())
}

fn validate_metrics(metrics: &[CoordinateMetric], dimensions: usize) -> Result<(), String> {
    if metrics.len() != dimensions {
        return Err(format!(
            "received {} coordinate metrics for {dimensions} coordinates",
            metrics.len()
        ));
    }
    for (index, metric) in metrics.iter().copied().enumerate() {
        metric.validate(index)?;
    }
    Ok(())
}

/// ln of the volume of the unit ball in `dimensions` dimensions,
/// pi^(d/2) / Gamma(d/2 + 1). Dimensions 1 to 4 keep their literal forms so the
/// MIE terms are unchanged to the last bit; above that, Gamma is evaluated
/// exactly at its integer or half-integer argument.
pub fn log_unit_ball_volume(dimensions: usize) -> f64 {
    use std::f64::consts::PI;
    match dimensions {
        1 => 2.0_f64.ln(),
        2 => PI.ln(),
        3 => (4.0 * PI / 3.0).ln(),
        4 => (PI.powi(2) / 2.0).ln(),
        d => {
            let ln_factorial = |n: usize| (2..=n).map(|i| (i as f64).ln()).sum::<f64>();
            let ln_gamma = if d % 2 == 0 {
                // Gamma(k + 1) = k!, k = d/2.
                ln_factorial(d / 2)
            } else {
                // Gamma(m + 1/2) = (2m)! sqrt(pi) / (4^m m!), m = (d + 1)/2.
                let m = d.div_ceil(2);
                ln_factorial(2 * m) + 0.5 * PI.ln() - m as f64 * 4.0_f64.ln() - ln_factorial(m)
            };
            0.5 * d as f64 * PI.ln() - ln_gamma
        }
    }
}

fn entropy_constant_with_convention(
    n_frames: usize,
    dimensions: usize,
    convention: FiniteSampleConstant,
    theiler_window: usize,
) -> Result<f64, String> {
    if dimensions == 0 {
        return Err(format!(
            "unsupported nearest-neighbor dimension {dimensions}"
        ));
    }
    let log_unit_ball_volume = log_unit_ball_volume(dimensions);
    validate_theiler_window(n_frames, theiler_window)?;
    // The reference Python implementation (NN_entropy_calc.py) hard-codes
    // gamma as 0.57721566490153, truncated to 14 decimal places. Full double
    // precision differs from it by about 6.6e-15, so the asymptotic constant,
    // and every entropy computed with it, can differ from that implementation
    // in the last few bits.
    use std::f64::consts::EULER_GAMMA as EULER_MASCHERONI;
    if theiler_window == 0 {
        let sample_term = match convention {
            FiniteSampleConstant::Asymptotic => (n_frames as f64).ln() + EULER_MASCHERONI,
            FiniteSampleConstant::Exact => (1..n_frames).map(|i| 1.0 / i as f64).sum(),
        };
        return Ok(sample_term + log_unit_ball_volume);
    }

    // Frame i competes against n_i - 1 eligible neighbors, where n_i is N minus
    // the other frames inside its Theiler window, so the sample term
    // psi(N) - psi(1) becomes the mean over frames of psi(n_i) - psi(1).
    let harmonic: Vec<f64> = std::iter::once(0.0)
        .chain((1..n_frames).scan(0.0, |total, i| {
            *total += 1.0 / i as f64;
            Some(*total)
        }))
        .collect();
    let sample_term_total: f64 = (0..n_frames)
        .map(|frame| {
            let excluded = frame.min(theiler_window) + (n_frames - 1 - frame).min(theiler_window);
            let candidates = n_frames - excluded;
            match convention {
                FiniteSampleConstant::Asymptotic => (candidates as f64).ln() + EULER_MASCHERONI,
                FiniteSampleConstant::Exact => harmonic[candidates - 1],
            }
        })
        .sum();
    Ok(sample_term_total / n_frames as f64 + log_unit_ball_volume)
}

/// Every frame needs at least one neighbor outside its Theiler window, which
/// the middle frames of the series only have when N >= 2w + 2.
fn validate_theiler_window(n_frames: usize, theiler_window: usize) -> Result<(), String> {
    if n_frames < 2 * theiler_window + 2 {
        return Err(format!(
            "Theiler window {theiler_window} needs at least {} frames, got {n_frames}",
            2 * theiler_window + 2
        ));
    }
    Ok(())
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1usize, |acc, i| acc * (n - i) / (i + 1))
}

fn combination_from_rank<const K: usize>(mut rank: usize, n: usize) -> [usize; K] {
    debug_assert!(K <= n);
    debug_assert!(rank < binomial(n, K));

    let mut combination = [0usize; K];
    let mut start = 0;
    for (position, slot) in combination.iter_mut().enumerate() {
        for candidate in start..n {
            let remaining = K - position - 1;
            let combinations_with_candidate = binomial(n - candidate - 1, remaining);
            if rank < combinations_with_candidate {
                *slot = candidate;
                start = candidate + 1;
                break;
            }
            rank -= combinations_with_candidate;
        }
    }
    combination
}

/// Optional settings shared by every entropy estimator.
///
/// `EntropyOptions::default()` treats every coordinate as linear, uses the
/// asymptotic finite-sample constant, and has no Theiler window.
#[derive(Debug, Clone, Copy, Default)]
pub struct EntropyOptions<'a> {
    /// Per-coordinate metrics; `None` treats every coordinate as linear.
    pub metrics: Option<&'a [CoordinateMetric]>,
    /// Sample-size term of the nearest-neighbor entropy constant.
    pub constant: FiniteSampleConstant,
    /// Theiler window `w`: a frame's nearest neighbor must be more than `w`
    /// frames away from it in time. 0 allows every other frame.
    pub theiler_window: usize,
}

/// Validates the input and returns the data truncated to `frames_end` together
/// with one metric per coordinate.
fn resolve_input(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    options: &EntropyOptions,
) -> Result<(Vec<Vec<f64>>, Vec<CoordinateMetric>), String> {
    validate_one_d_data(one_d_data, frames_end)?;
    let metrics = match options.metrics {
        Some(metrics) => {
            validate_metrics(metrics, one_d_data.len())?;
            metrics.to_vec()
        }
        None => vec![CoordinateMetric::Linear; one_d_data.len()],
    };
    let one_d_data = one_d_data
        .iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect();
    Ok((one_d_data, metrics))
}

/// Total entropy from a mutual information expansion truncated at `mie_order`
/// (1 through 4). Order 1 is the sum of the marginal entropies.
pub fn calculate_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    mie_order: usize,
    options: &EntropyOptions,
) -> Result<f64, String> {
    if !(1..=4).contains(&mie_order) {
        return Err(format!(
            "unsupported MIE order {mie_order}; supported orders are 1, 2, 3, and 4"
        ));
    }
    let (one_d_data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let constant = options.constant;
    let theiler_window = options.theiler_window;

    let n_frames = one_d_data[0].len();
    let degrees_freedom = one_d_data.len();
    let one_d_constant = entropy_constant_with_convention(n_frames, 1, constant, theiler_window)?;
    let two_d_constant = entropy_constant_with_convention(n_frames, 2, constant, theiler_window)?;
    let three_d_constant = entropy_constant_with_convention(n_frames, 3, constant, theiler_window)?;
    let four_d_constant = entropy_constant_with_convention(n_frames, 4, constant, theiler_window)?;

    let one_d_distances_total: f64 = one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .try_fold(
            || 0.0,
            |acc, (ic, metric)| {
                Ok::<f64, String>(acc + calc_joint_nn_with_window([ic], [*metric], theiler_window)?)
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok::<f64, String>(a + b))?;

    let one_d_entropy = estimate_entropy_efficient(
        one_d_distances_total,
        (n_frames as f64).recip(),
        one_d_constant * degrees_freedom as f64,
    );

    let effective_order = mie_order.min(degrees_freedom);
    if effective_order == 1 {
        return Ok(one_d_entropy);
    }

    let two_d_degrees_freedom = binomial(degrees_freedom, 2);

    let two_d_distances_total: f64 = (0..two_d_degrees_freedom)
        .into_par_iter()
        .try_fold(
            || 0.0,
            |acc, rank| {
                let [i, j] = combination_from_rank::<2>(rank, degrees_freedom);
                Ok::<f64, String>(
                    acc + calc_joint_nn_with_window(
                        [&one_d_data[i], &one_d_data[j]],
                        [metrics[i], metrics[j]],
                        theiler_window,
                    )?,
                )
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok::<f64, String>(a + b))?;

    let two_d_entropy = estimate_entropy_efficient(
        two_d_distances_total * 2.0,
        (n_frames as f64).recip(),
        two_d_constant * two_d_degrees_freedom as f64,
    );

    if effective_order == 2 {
        return Ok(two_d_entropy - ((degrees_freedom - 2) as f64) * one_d_entropy);
    }

    let three_d_degrees_freedom = binomial(degrees_freedom, 3);

    let three_d_distances_total: f64 = (0..three_d_degrees_freedom)
        .into_par_iter()
        .try_fold(
            || 0.0,
            |acc, rank| {
                let [i, j, k] = combination_from_rank::<3>(rank, degrees_freedom);
                Ok::<f64, String>(
                    acc + calc_joint_nn_with_window(
                        [&one_d_data[i], &one_d_data[j], &one_d_data[k]],
                        [metrics[i], metrics[j], metrics[k]],
                        theiler_window,
                    )?,
                )
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok::<f64, String>(a + b))?;

    let three_d_entropy = estimate_entropy_efficient(
        three_d_distances_total * 3.0,
        (n_frames as f64).recip(),
        three_d_constant * three_d_degrees_freedom as f64,
    );

    let one_d_coefficient = ((degrees_freedom - 2) * (degrees_freedom - 3)) as f64 / 2.0;
    let two_d_coefficient = (degrees_freedom - 3) as f64;

    if effective_order == 3 {
        return Ok(
            three_d_entropy - two_d_coefficient * two_d_entropy + one_d_coefficient * one_d_entropy
        );
    }

    let four_d_degrees_freedom = binomial(degrees_freedom, 4);

    let four_d_distances_total: f64 = (0..four_d_degrees_freedom)
        .into_par_iter()
        .try_fold(
            || 0.0,
            |acc, rank| {
                let [i, j, k, l] = combination_from_rank::<4>(rank, degrees_freedom);
                Ok::<f64, String>(
                    acc + calc_joint_nn_with_window(
                        [
                            &one_d_data[i],
                            &one_d_data[j],
                            &one_d_data[k],
                            &one_d_data[l],
                        ],
                        [metrics[i], metrics[j], metrics[k], metrics[l]],
                        theiler_window,
                    )?,
                )
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok::<f64, String>(a + b))?;

    let four_d_entropy = estimate_entropy_efficient(
        four_d_distances_total * 4.0,
        (n_frames as f64).recip(),
        four_d_constant * four_d_degrees_freedom as f64,
    );

    let one_d_coefficient =
        ((degrees_freedom - 2) * (degrees_freedom - 3) * (degrees_freedom - 4)) as f64 / 6.0;
    let two_d_coefficient = ((degrees_freedom - 3) * (degrees_freedom - 4)) as f64 / 2.0;
    let three_d_coefficient = (degrees_freedom - 4) as f64;

    Ok(
        four_d_entropy - three_d_coefficient * three_d_entropy + two_d_coefficient * two_d_entropy
            - one_d_coefficient * one_d_entropy,
    )
}

/// Each coordinate's share of the MIE entropy truncated at `mie_order` (1 or 2),
/// so the values sum to `calculate_entropy` at the same order.
///
/// Order 1 gives the marginal entropies. Order 2 also splits every pairwise
/// mutual information term evenly between its two coordinates.
pub fn estimate_coordinate_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    mie_order: usize,
    options: &EntropyOptions,
) -> Result<Vec<f64>, String> {
    if !(1..=2).contains(&mie_order) {
        return Err(format!(
            "unsupported per-coordinate MIE order {mie_order}; supported orders are 1 and 2"
        ));
    }
    let (one_d_data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let coordinate_entropies = coordinate_entropy_impl(
        &one_d_data,
        &metrics,
        options.constant,
        options.theiler_window,
    )?;
    let degrees_freedom = coordinate_entropies.len();
    if mie_order == 1 || degrees_freedom == 1 {
        return Ok(coordinate_entropies);
    }

    let pairwise_mutual_information = coordinate_mutual_information_impl(
        &one_d_data,
        &metrics,
        options.constant,
        options.theiler_window,
        &coordinate_entropies,
    )?;
    let mut coordinate_mie_entropy = coordinate_entropies;
    let mut pair_idx = 0;
    for i in 0..degrees_freedom {
        for j in (i + 1)..degrees_freedom {
            let half_pair_mi = 0.5 * pairwise_mutual_information[pair_idx];
            coordinate_mie_entropy[i] -= half_pair_mi;
            coordinate_mie_entropy[j] -= half_pair_mi;
            pair_idx += 1;
        }
    }

    Ok(coordinate_mie_entropy)
}

fn coordinate_entropy_impl(
    one_d_data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    constant: FiniteSampleConstant,
    theiler_window: usize,
) -> Result<Vec<f64>, String> {
    let n_frames: usize = one_d_data[0].len();
    let one_d_constant = entropy_constant_with_convention(n_frames, 1, constant, theiler_window)?;

    one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .map(|(ic, metric)| {
            calc_joint_nn_with_window([ic], [*metric], theiler_window).map(|distance| {
                estimate_entropy_efficient(distance, (n_frames as f64).recip(), one_d_constant)
            })
        })
        .collect()
}

/// Mutual information of every coordinate pair, in lexicographic pair order.
pub fn estimate_coordinate_mutual_information(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    options: &EntropyOptions,
) -> Result<Vec<f64>, String> {
    let (one_d_data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let one_d_entropies = coordinate_entropy_impl(
        &one_d_data,
        &metrics,
        options.constant,
        options.theiler_window,
    )?;
    coordinate_mutual_information_impl(
        &one_d_data,
        &metrics,
        options.constant,
        options.theiler_window,
        &one_d_entropies,
    )
}

fn coordinate_mutual_information_impl(
    one_d_data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    constant: FiniteSampleConstant,
    theiler_window: usize,
    one_d_entropies: &[f64],
) -> Result<Vec<f64>, String> {
    let n_frames: usize = one_d_data[0].len();
    let degrees_freedom: usize = one_d_data.len();
    let two_d_constant = entropy_constant_with_convention(n_frames, 2, constant, theiler_window)?;
    let two_d_degrees_freedom = binomial(degrees_freedom, 2);

    let mutual_information: Vec<f64> = (0..two_d_degrees_freedom)
        .into_par_iter()
        .map(|rank| {
            let [i, j] = combination_from_rank::<2>(rank, degrees_freedom);
            let joint_entropy = estimate_entropy_efficient(
                calc_joint_nn_with_window(
                    [&one_d_data[i], &one_d_data[j]],
                    [metrics[i], metrics[j]],
                    theiler_window,
                )? * 2.0,
                (n_frames as f64).recip(),
                two_d_constant,
            );
            Ok::<f64, String>(one_d_entropies[i] + one_d_entropies[j] - joint_entropy)
        })
        .collect::<Result<Vec<_>, _>>()?;

    assert_eq!(mutual_information.len(), two_d_degrees_freedom); // Just to be safe
    Ok(mutual_information)
}

fn point_cmp<const K: usize>(left: &[f64; K], right: &[f64; K]) -> std::cmp::Ordering {
    for dimension in 0..K {
        let ordering = left[dimension].total_cmp(&right[dimension]);
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// Sum over frames of ln(distance to the nearest other frame) in the K-dimensional
/// joint space of `coordinates`, using each coordinate's metric.
pub fn calc_joint_nn<const K: usize>(
    coordinates: [&[f64]; K],
    metrics: [CoordinateMetric; K],
) -> Result<f64, String> {
    calc_joint_nn_with_window(coordinates, metrics, 0)
}

/// `calc_joint_nn` with a Theiler window: frame i's nearest neighbor is taken
/// only from frames j with |i - j| > `theiler_window`. A window of 0 is
/// identical to `calc_joint_nn`.
///
/// Exact duplicates of a frame's position never count as its neighbor, with or
/// without a window, so the distance is always to the nearest distinct point.
#[allow(clippy::needless_range_loop)]
pub fn calc_joint_nn_with_window<const K: usize>(
    coordinates: [&[f64]; K],
    metrics: [CoordinateMetric; K],
    theiler_window: usize,
) -> Result<f64, String> {
    let points_len = coordinates[0].len();
    if points_len < 2 {
        return Err(format!(
            "need at least two points for {K}D nearest neighbor"
        ));
    }
    if coordinates
        .iter()
        .any(|coordinate| coordinate.len() != points_len)
    {
        return Err(format!(
            "all coordinate series must have equal length for {K}D nearest neighbor"
        ));
    }
    validate_metrics(&metrics, K)?;
    validate_theiler_window(points_len, theiler_window)?;

    let mut points: Vec<[f64; K]> = Vec::with_capacity(points_len);
    for frame_idx in 0..points_len {
        let mut point = [0.0; K];
        for dimension_idx in 0..K {
            let value = coordinates[dimension_idx][frame_idx];
            if !value.is_finite() {
                return Err(format!(
                    "coordinate {dimension_idx} contains non-finite values"
                ));
            }
            point[dimension_idx] = metrics[dimension_idx].canonicalize(value);
        }
        points.push(point);
    }

    // Each distinct position keeps the first and last frame found there, which
    // is all the Theiler test needs: some frame at a position lies outside the
    // window around frame i exactly when the first or the last one does.
    let mut frame_order: Vec<usize> = (0..points_len).collect();
    frame_order.sort_by(|&left, &right| point_cmp(&points[left], &points[right]));
    let mut unique_points: Vec<[f64; K]> = Vec::with_capacity(points_len);
    let mut frame_ranges: Vec<(usize, usize)> = Vec::with_capacity(points_len);
    let mut unique_index_of_frame = vec![0usize; points_len];
    for frame in frame_order {
        let point = points[frame];
        if unique_points
            .last()
            .is_none_or(|last| point_cmp(last, &point).is_ne())
        {
            unique_points.push(point);
            frame_ranges.push((frame, frame));
        }
        let range = frame_ranges.last_mut().unwrap();
        range.0 = range.0.min(frame);
        range.1 = range.1.max(frame);
        unique_index_of_frame[frame] = unique_points.len() - 1;
    }
    if unique_points.len() < 2 {
        return Err(format!(
            "need at least two distinct points for {K}D nearest neighbor"
        ));
    }

    // Only the query's own position and positions whose frames all fall inside
    // its window are ineligible, at most 2w + 1 of them, so the nearest 2w + 2
    // positions always contain the nearest eligible one if it exists.
    let max_query_count = unique_points.len().min(2 * theiler_window + 2);
    let periodic_dimensions: Vec<(usize, f64)> = (0..K)
        .filter_map(|dimension| match metrics[dimension] {
            CoordinateMetric::Periodic { period } => Some((dimension, period)),
            CoordinateMetric::Linear => None,
        })
        .collect();
    let kdtree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice(&unique_points);
    let mut distance_total: f64 = 0.0;
    for (frame, point) in points.into_iter().enumerate() {
        let self_index = unique_index_of_frame[frame];
        let is_eligible = |candidate: usize| {
            let (first, last) = frame_ranges[candidate];
            candidate != self_index
                && (first + theiler_window < frame || last > frame + theiler_window)
        };
        // Squared distance from `image` to its nearest eligible frame.
        let query = |image: &[f64; K]| {
            let mut query_count = max_query_count.min(2);
            loop {
                let neighbors =
                    kdtree.nearest_n::<SquaredEuclidean>(image, NonZero::new(query_count).unwrap());
                let nearest_eligible = neighbors
                    .iter()
                    .filter(|neighbor| is_eligible(neighbor.item as usize))
                    .map(|neighbor| neighbor.distance)
                    .min_by(f64::total_cmp);
                if nearest_eligible.is_some() || query_count == max_query_count {
                    return nearest_eligible;
                }
                query_count = (query_count * 2).min(max_query_count);
            }
        };
        // The unshifted point first; its distance bounds the rest. A periodic
        // image shifted by -P in dimension k is at least P - x_k from every
        // canonical point (all lie in [0, P)), one shifted by +P at least x_k,
        // so a combination of shifts whose summed bound already reaches the
        // best distance cannot improve it and is skipped. The minimum is the
        // same as over every image.
        let mut best = query(&point);
        let mut stack = vec![(0usize, point, 0.0_f64, false)];
        while let Some((depth, image, bound, shifted)) = stack.pop() {
            if best.is_some_and(|best| bound >= best) {
                continue;
            }
            if depth == periodic_dimensions.len() {
                if shifted && let Some(distance) = query(&image) {
                    best = Some(best.map_or(distance, |best| best.min(distance)));
                }
                continue;
            }
            let (dimension, period) = periodic_dimensions[depth];
            let x = point[dimension];
            stack.push((depth + 1, image, bound, shifted));
            let mut below = image;
            below[dimension] -= period;
            stack.push((depth + 1, below, bound + (period - x).powi(2), true));
            let mut above = image;
            above[dimension] += period;
            stack.push((depth + 1, above, bound + x.powi(2), true));
        }
        let result = best.ok_or_else(|| {
                format!(
                    "frame {frame} has no distinct {K}D neighbor outside Theiler window {theiler_window}"
                )
            })?;
        distance_total += result.sqrt().ln();
    }
    Ok(distance_total)
}
/// Largest number of coordinates for a joint nearest-neighbor estimate. The
/// estimator's bias grows quickly with dimension; beyond this, use the MIE.
pub const MAX_JOINT_DIMENSION: usize = 8;

/// [`calc_joint_nn_with_window`] with the dimension chosen at run time, from 1
/// to [`MAX_JOINT_DIMENSION`].
pub fn calc_joint_nn_dynamic(
    coordinates: &[&[f64]],
    metrics: &[CoordinateMetric],
    theiler_window: usize,
) -> Result<f64, String> {
    if coordinates.len() != metrics.len() {
        return Err(format!(
            "received {} coordinate metrics for {} coordinates",
            metrics.len(),
            coordinates.len()
        ));
    }
    macro_rules! dispatch {
        ($($k:literal)*) => {
            match coordinates.len() {
                $($k => calc_joint_nn_with_window::<$k>(
                    coordinates.try_into().unwrap(),
                    metrics.try_into().unwrap(),
                    theiler_window,
                ),)*
                d => Err(format!(
                    "a joint nearest-neighbor estimate supports 1 to {MAX_JOINT_DIMENSION} coordinates, got {d}; use the mutual information expansion instead"
                )),
            }
        };
    }
    dispatch!(1 2 3 4 5 6 7 8)
}

/// Joint nearest-neighbor (Kozachenko-Leonenko) entropy of all the coordinates
/// in `one_d_data` at once, with no expansion: up to [`MAX_JOINT_DIMENSION`]
/// coordinates. For one coordinate it equals [`calculate_entropy`] at order 1.
pub fn estimate_joint_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    options: &EntropyOptions,
) -> Result<f64, String> {
    let (one_d_data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let dimensions = one_d_data.len();
    let columns: Vec<&[f64]> = one_d_data.iter().map(Vec::as_slice).collect();
    let distances = calc_joint_nn_dynamic(&columns, &metrics, options.theiler_window)?;
    let constant = entropy_constant_with_convention(
        frames_end,
        dimensions,
        options.constant,
        options.theiler_window,
    )?;
    Ok(estimate_entropy_efficient(
        distances * dimensions as f64,
        (frames_end as f64).recip(),
        constant,
    ))
}

/// `nn_distance * reciprocal_n_frames + constant`, as a fused multiply-add when
/// the `fma` feature is enabled: a nearest-neighbor entropy from the sum over
/// frames of ln(nearest-neighbor distance), already multiplied by the
/// dimension. The estimators call it; it is public for testing.
pub fn estimate_entropy_efficient(
    nn_distance: f64,
    reciprocal_n_frames: f64,
    constant: f64,
) -> f64 {
    // Reduces number of instructions used, and uses fused multiply add when enabled
    if cfg!(feature = "fma") {
        nn_distance.mul_add(reciprocal_n_frames, constant)
    } else {
        nn_distance * reciprocal_n_frames + constant
    }
}

#[cfg(test)]
mod tests {
    use super::{binomial, combination_from_rank};

    #[test]
    fn binomial_counts_combinations() {
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(6, 4), 15);
        assert_eq!(binomial(4, 5), 0);
    }

    #[test]
    fn combination_ranks_cover_lexicographic_order() {
        let triples = (0..binomial(5, 3))
            .map(|rank| combination_from_rank::<3>(rank, 5))
            .collect::<Vec<_>>();

        assert_eq!(
            triples,
            vec![
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 2, 3],
                [0, 2, 4],
                [0, 3, 4],
                [1, 2, 3],
                [1, 2, 4],
                [1, 3, 4],
                [2, 3, 4],
            ]
        );
    }

    #[test]
    fn combination_ranks_cover_pair_terms() {
        let pairs = (0..binomial(4, 2))
            .map(|rank| combination_from_rank::<2>(rank, 4))
            .collect::<Vec<_>>();

        assert_eq!(pairs, vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
    }

    #[test]
    fn combination_ranks_cover_fourth_order_terms() {
        let fourth_order_terms = (0..binomial(6, 4))
            .map(|rank| combination_from_rank::<4>(rank, 6))
            .collect::<Vec<_>>();

        assert_eq!(fourth_order_terms.len(), 15);
        assert_eq!(fourth_order_terms[0], [0, 1, 2, 3]);
        assert_eq!(fourth_order_terms[14], [2, 3, 4, 5]);
    }
}
