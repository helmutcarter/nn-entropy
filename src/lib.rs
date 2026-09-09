use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::fs::File;
use std::io::ErrorKind;
use std::num::NonZero;

pub mod bat_library;
pub mod pyo3_api;

/// Metric used for one internal-coordinate dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateMetric {
    Linear,
    Periodic { period: f64 },
}

/// Sample-size term used in the Kozachenko-Leonenko entropy constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteSampleConstant {
    /// Historical Python-compatible large-sample approximation: ln(N) + gamma.
    PythonCompatibleAsymptotic,
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

fn entropy_constant(n_frames: usize, dimensions: usize) -> Result<f64, String> {
    entropy_constant_with_convention(
        n_frames,
        dimensions,
        FiniteSampleConstant::PythonCompatibleAsymptotic,
    )
}

fn entropy_constant_with_convention(
    n_frames: usize,
    dimensions: usize,
    convention: FiniteSampleConstant,
) -> Result<f64, String> {
    let log_unit_ball_volume = match dimensions {
        1 => 2.0_f64.ln(),
        2 => std::f64::consts::PI.ln(),
        3 => (4.0 * std::f64::consts::PI / 3.0).ln(),
        4 => (std::f64::consts::PI.powi(2) / 2.0).ln(),
        _ => {
            return Err(format!(
                "unsupported nearest-neighbor dimension {dimensions}"
            ));
        }
    };
    const EULER_MASCHERONI: f64 = 0.57721566490153;
    let sample_term = match convention {
        FiniteSampleConstant::PythonCompatibleAsymptotic => {
            (n_frames as f64).ln() + EULER_MASCHERONI
        }
        FiniteSampleConstant::Exact => (1..n_frames).map(|i| 1.0 / i as f64).sum(),
    };
    Ok(sample_term + log_unit_ball_volume)
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

pub fn calculate_entropy_from_data(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
) -> Result<f64, String> {
    calculate_entropy_from_data_with_order(one_d_data, frames_end, 2)
}

pub fn calculate_entropy_from_data_with_order(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
    mie_order: usize,
) -> Result<f64, String> {
    let metrics = vec![CoordinateMetric::Linear; one_d_data.len()];
    calculate_entropy_from_data_with_metrics(one_d_data, frames_end, mie_order, &metrics)
}

pub fn calculate_entropy_from_data_with_metrics(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
    mie_order: usize,
    metrics: &[CoordinateMetric],
) -> Result<f64, String> {
    calculate_entropy_from_data_with_metrics_and_constant(
        one_d_data,
        frames_end,
        mie_order,
        metrics,
        FiniteSampleConstant::PythonCompatibleAsymptotic,
    )
}

pub fn calculate_entropy_from_data_with_metrics_and_constant(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
    mie_order: usize,
    metrics: &[CoordinateMetric],
    constant: FiniteSampleConstant,
) -> Result<f64, String> {
    if !(1..=4).contains(&mie_order) {
        return Err(format!(
            "unsupported MIE order {mie_order}; supported orders are 1, 2, 3, and 4"
        ));
    }
    validate_one_d_data(&one_d_data, frames_end)?;
    validate_metrics(metrics, one_d_data.len())?;

    let one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect::<Vec<_>>();

    let n_frames = one_d_data[0].len();
    let degrees_freedom = one_d_data.len();
    let one_d_constant = entropy_constant_with_convention(n_frames, 1, constant)?;
    let two_d_constant = entropy_constant_with_convention(n_frames, 2, constant)?;
    let three_d_constant = entropy_constant_with_convention(n_frames, 3, constant)?;
    let four_d_constant = entropy_constant_with_convention(n_frames, 4, constant)?;

    let one_d_distances_total: f64 = one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .try_fold(
            || 0.0,
            |acc, (ic, metric)| Ok::<f64, String>(acc + calc_one_d_nn_with_metric(ic, *metric)?),
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
                    acc + calc_two_d_nn_with_metrics(
                        &one_d_data[i],
                        &one_d_data[j],
                        [metrics[i], metrics[j]],
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
                    acc + calc_three_d_nn_with_metrics(
                        &one_d_data[i],
                        &one_d_data[j],
                        &one_d_data[k],
                        [metrics[i], metrics[j], metrics[k]],
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
                    acc + calc_four_d_nn_with_metrics(
                        &one_d_data[i],
                        &one_d_data[j],
                        &one_d_data[k],
                        &one_d_data[l],
                        [metrics[i], metrics[j], metrics[k], metrics[l]],
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

pub fn estimate_coordinate_entropy_rust(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
) -> Result<Vec<f64>, String> {
    let metrics = vec![CoordinateMetric::Linear; one_d_data.len()];
    estimate_coordinate_entropy_with_metrics(one_d_data, frames_end, &metrics)
}

pub fn estimate_coordinate_entropy_with_metrics(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
    metrics: &[CoordinateMetric],
) -> Result<Vec<f64>, String> {
    validate_one_d_data(&one_d_data, frames_end)?;
    validate_metrics(metrics, one_d_data.len())?;

    let one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect::<Vec<_>>();

    let n_frames: usize = one_d_data[0].len();

    let one_d_constant = entropy_constant(n_frames, 1)?;

    let one_d_distances: Vec<f64> = one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .map(|(ic, metric)| calc_one_d_nn_with_metric(ic, *metric))
        .collect::<Result<Vec<_>, _>>()?;

    let one_d_entropies: Vec<f64> = one_d_distances
        .iter()
        .map(|&distance| {
            estimate_entropy_efficient(distance, (n_frames as f64).recip(), one_d_constant)
        })
        .collect();

    Ok(one_d_entropies)
}

pub fn estimate_coordinate_mutual_information_rust(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
) -> Result<Vec<f64>, String> {
    let metrics = vec![CoordinateMetric::Linear; one_d_data.len()];
    estimate_coordinate_mutual_information_with_metrics(one_d_data, frames_end, &metrics)
}

pub fn estimate_coordinate_mutual_information_with_metrics(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
    metrics: &[CoordinateMetric],
) -> Result<Vec<f64>, String> {
    validate_one_d_data(&one_d_data, frames_end)?;
    validate_metrics(metrics, one_d_data.len())?;

    let one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect::<Vec<_>>();

    let n_frames: usize = one_d_data[0].len();
    let degrees_freedom: usize = one_d_data.len();

    let one_d_constant = entropy_constant(n_frames, 1)?;
    let two_d_constant = entropy_constant(n_frames, 2)?;

    let one_d_entropies = one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .map(|(ic, metric)| {
            calc_one_d_nn_with_metric(ic, *metric).map(|distance| {
                estimate_entropy_efficient(distance, (n_frames as f64).recip(), one_d_constant)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let two_d_degrees_freedom = binomial(degrees_freedom, 2);

    let mutual_information: Vec<f64> = (0..two_d_degrees_freedom)
        .into_par_iter()
        .map(|rank| {
            let [i, j] = combination_from_rank::<2>(rank, degrees_freedom);
            let joint_entropy = estimate_entropy_efficient(
                calc_two_d_nn_with_metrics(
                    &one_d_data[i],
                    &one_d_data[j],
                    [metrics[i], metrics[j]],
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

pub fn estimate_coordinate_mie_entropy_rust(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
) -> Result<Vec<f64>, String> {
    let metrics = vec![CoordinateMetric::Linear; one_d_data.len()];
    estimate_coordinate_mie_entropy_with_metrics(one_d_data, frames_end, &metrics)
}

pub fn estimate_coordinate_mie_entropy_with_metrics(
    one_d_data: Vec<Vec<f64>>,
    frames_end: usize,
    metrics: &[CoordinateMetric],
) -> Result<Vec<f64>, String> {
    validate_one_d_data(&one_d_data, frames_end)?;
    validate_metrics(metrics, one_d_data.len())?;

    let coordinate_entropies =
        estimate_coordinate_entropy_with_metrics(one_d_data.clone(), frames_end, metrics)?;
    let degrees_freedom = coordinate_entropies.len();
    if degrees_freedom == 1 {
        return Ok(coordinate_entropies);
    }

    let pairwise_mutual_information =
        estimate_coordinate_mutual_information_with_metrics(one_d_data, frames_end, metrics)?;
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

pub fn calc_one_d_nn(points: &[f64]) -> Result<f64, String> {
    calc_one_d_nn_with_metric(points, CoordinateMetric::Linear)
}

pub fn calc_one_d_nn_with_metric(points: &[f64], metric: CoordinateMetric) -> Result<f64, String> {
    calc_joint_nn_with_metrics([points], [metric])
}

pub fn calc_one_d_nn_kdtree(points: Vec<f64>) -> Result<f64, String> {
    calc_one_d_nn(&points)
}

pub fn calc_two_d_nn(points_1: &[f64], points_2: &[f64]) -> Result<f64, String> {
    calc_two_d_nn_with_metrics(
        points_1,
        points_2,
        [CoordinateMetric::Linear, CoordinateMetric::Linear],
    )
}

pub fn calc_two_d_nn_with_metrics(
    points_1: &[f64],
    points_2: &[f64],
    metrics: [CoordinateMetric; 2],
) -> Result<f64, String> {
    calc_joint_nn_with_metrics([points_1, points_2], metrics)
}

pub fn calc_three_d_nn(
    points_1: &[f64],
    points_2: &[f64],
    points_3: &[f64],
) -> Result<f64, String> {
    calc_three_d_nn_with_metrics(points_1, points_2, points_3, [CoordinateMetric::Linear; 3])
}

pub fn calc_three_d_nn_with_metrics(
    points_1: &[f64],
    points_2: &[f64],
    points_3: &[f64],
    metrics: [CoordinateMetric; 3],
) -> Result<f64, String> {
    calc_joint_nn_with_metrics([points_1, points_2, points_3], metrics)
}

pub fn calc_four_d_nn(
    points_1: &[f64],
    points_2: &[f64],
    points_3: &[f64],
    points_4: &[f64],
) -> Result<f64, String> {
    calc_four_d_nn_with_metrics(
        points_1,
        points_2,
        points_3,
        points_4,
        [CoordinateMetric::Linear; 4],
    )
}

pub fn calc_four_d_nn_with_metrics(
    points_1: &[f64],
    points_2: &[f64],
    points_3: &[f64],
    points_4: &[f64],
    metrics: [CoordinateMetric; 4],
) -> Result<f64, String> {
    calc_joint_nn_with_metrics([points_1, points_2, points_3, points_4], metrics)
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

fn periodic_query_images<const K: usize>(
    point: [f64; K],
    metrics: [CoordinateMetric; K],
) -> Vec<[f64; K]> {
    let mut images = vec![point];
    for dimension in 0..K {
        if let CoordinateMetric::Periodic { period } = metrics[dimension] {
            let existing = images.clone();
            for image in existing {
                let mut below = image;
                below[dimension] -= period;
                images.push(below);
                let mut above = image;
                above[dimension] += period;
                images.push(above);
            }
        }
    }
    images
}

#[allow(clippy::needless_range_loop)]
fn calc_joint_nn_with_metrics<const K: usize>(
    coordinates: [&[f64]; K],
    metrics: [CoordinateMetric; K],
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

    let mut unique_points = points.clone();
    unique_points.sort_by(point_cmp);
    unique_points.dedup_by(|left, right| point_cmp(left, right).is_eq());
    if unique_points.len() < 2 {
        return Err(format!(
            "need at least two distinct points for {K}D nearest neighbor"
        ));
    }

    let kdtree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice(&unique_points);
    let query_count = NonZero::new(unique_points.len().min(2)).unwrap();
    let mut distance_total: f64 = 0.0;
    for point in points {
        let self_index = unique_points
            .binary_search_by(|candidate| point_cmp(candidate, &point))
            .map_err(|_| "canonical coordinate point was not found".to_string())?;
        let result = periodic_query_images(point, metrics)
            .into_iter()
            .flat_map(|image| {
                kdtree
                    .nearest_n::<SquaredEuclidean>(&image, query_count)
                    .into_iter()
            })
            .filter(|neighbor| neighbor.item as usize != self_index)
            .map(|neighbor| neighbor.distance)
            .min_by(f64::total_cmp)
            .ok_or_else(|| {
                format!("need at least two distinct points for {K}D nearest neighbor")
            })?;
        distance_total += result.sqrt().ln();
    }
    Ok(distance_total)
}
// Helper function to generate Gaussian data for testing
pub fn generate_normal(mean: f64, std_dev: f64, size: usize) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::thread_rng();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

pub fn estimate_entropy(
    nn_distance: f64,
    n_frames: usize,
    constant: f64,
    n_internal_coords: usize,
) -> f64 {
    // println!("{constant}");
    (nn_distance / (n_frames as f64)) + constant * (n_internal_coords as f64)
}

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

pub fn load_one_d_data(file_path: &str) -> Vec<Vec<f64>> {
    let mut all_data: Vec<Vec<f64>> = Vec::new();
    let file_result = File::open(file_path);
    let file = match file_result {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {e:?}"),
            },
            other_error => {
                panic!("Problem opening the file: {other_error:?}");
            }
        },
    };
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    for result in reader.records() {
        let mut internal_coordinate_data: Vec<f64> = Vec::new();
        match result {
            Ok(record) => {
                for entry in record.iter() {
                    let point: f64 = entry.parse().unwrap();
                    internal_coordinate_data.push(point);
                }
            }
            Err(e) => println!("Error reading record: {:?}", e),
        }
        all_data.push(internal_coordinate_data);
    }
    all_data
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
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

pub fn cross_product(b1: [f64; 3], b2: [f64; 3]) -> [f64; 3] {
    [
        // x-component: b2_y * b2_z - b2_z * b2_y
        b1[1] * b2[2] - b1[2] * b2[1],
        // y-component: b1_z * b2_x - b1_x * b2_z
        b1[2] * b2[0] - b1[0] * b2[2],
        // z-component: b1_x * b2_y - b1_y * b2_x
        b1[0] * b2[1] - b1[1] * b2[0],
    ]
}
use libm::atan2;

fn dot_product(array_1: [f64; 3], array_2: [f64; 3]) -> f64 {
    array_1.iter().zip(array_2.iter()).map(|(x, y)| x * y).sum()
}

pub fn calc_bond(atom_1: [f64; 3], atom_2: [f64; 3]) -> f64 {
    // sqSum = (a1[0]-a2[0])**2 + (a1[1]-a2[1])**2 + (a1[2]-a2[2])**2
    let square_sum = (atom_1[0] - atom_2[0]).powi(2)
        + (atom_1[1] - atom_2[1]).powi(2)
        + (atom_1[2] - atom_2[2]).powi(2);

    // dist = math.sqrt(sqSum)
    // return dist
    square_sum.sqrt()
}

pub fn calc_angle(a1: [f64; 3], a2: [f64; 3], a3: [f64; 3]) -> f64 {
    let v1 = [(a1[0] - a2[0]), (a1[1] - a2[1]), (a1[2] - a2[2])];
    let v2 = [(a3[0] - a2[0]), (a3[1] - a2[1]), (a3[2] - a2[2])];

    // v1Mag = math.sqrt((v1[0])**2 + (v1[1])**2 + (v1[2])**2)
    let vector_1_magnitude = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();

    // v2Mag = math.sqrt((v2[0])**2 + (v2[1])**2 + (v2[2])**2)
    let vector_2_magnitude = (v2[0].powi(2) + v2[1].powi(2) + v2[2].powi(2)).sqrt();

    // let dot: f64 = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
    let dot: f64 = dot_product(v1, v2);

    let v1v2_mag = vector_1_magnitude * vector_2_magnitude;

    // angle = math.acos(dot/v1v2Mag)
    // return angle
    (dot / v1v2_mag).acos()
}
pub fn calc_torsion(atom_1: [f64; 3], atom_2: [f64; 3], atom_3: [f64; 3], atom_4: [f64; 3]) -> f64 {
    let bond_1: [f64; 3] = [
        atom_1[0] - atom_2[0],
        atom_1[1] - atom_2[1],
        atom_1[2] - atom_2[2],
    ];
    let bond_2: [f64; 3] = [
        atom_2[0] - atom_3[0],
        atom_2[1] - atom_3[1],
        atom_2[2] - atom_3[2],
    ];
    let bond_3: [f64; 3] = [
        atom_3[0] - atom_4[0],
        atom_3[1] - atom_4[1],
        atom_3[2] - atom_4[2],
    ];

    let cross_1: [f64; 3] = cross_product(bond_2, bond_3);
    let cross_2: [f64; 3] = cross_product(bond_1, bond_2);

    let mut plane_1 = dot_product(bond_1, cross_1);
    plane_1 *= (dot_product(bond_2, bond_2)).sqrt();

    let plane_2 = dot_product(cross_1, cross_2);

    atan2(plane_1, plane_2)
}

// Equivalent to intC(batList, traj) in Joe Cruz's code
pub fn calc_internal_coords(bat_list: Vec<Vec<usize>>, traj: Vec<Vec<[f64; 3]>>) -> Vec<Vec<f64>> {
    let n_int_coords: usize = bat_list.len();

    let mut internal_coords: Vec<Vec<f64>> = Vec::new();

    for frame in &traj {
        let mut frame_coords: Vec<f64> = Vec::new();
        for j in 0..n_int_coords {
            if bat_list[j].len() == 2 {
                frame_coords.push(calc_bond(frame[bat_list[j][0]], frame[bat_list[j][1]]));
                // println!("{:?}", j);
            }
            if bat_list[j].len() == 3 {
                frame_coords.push(calc_angle(
                    frame[bat_list[j][0]],
                    frame[bat_list[j][1]],
                    frame[bat_list[j][2]],
                ));
            }
            if bat_list[j].len() == 4 {
                frame_coords.push(calc_torsion(
                    frame[bat_list[j][0]],
                    frame[bat_list[j][1]],
                    frame[bat_list[j][2]],
                    frame[bat_list[j][3]],
                ));
            }
        }
        internal_coords.push(frame_coords);
    }
    internal_coords
}
