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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FiniteSampleConstant {
    /// Historical Python-compatible large-sample approximation: ln(N) + gamma.
    #[default]
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

/// Optional settings shared by every entropy estimator.
///
/// `EntropyOptions::default()` treats every coordinate as linear and uses the
/// historical Python-compatible finite-sample constant.
#[derive(Debug, Clone, Copy, Default)]
pub struct EntropyOptions<'a> {
    /// Per-coordinate metrics; `None` treats every coordinate as linear.
    pub metrics: Option<&'a [CoordinateMetric]>,
    pub constant: FiniteSampleConstant,
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
            |acc, (ic, metric)| Ok::<f64, String>(acc + calc_joint_nn([ic], [*metric])?),
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
                    acc + calc_joint_nn(
                        [&one_d_data[i], &one_d_data[j]],
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
                    acc + calc_joint_nn(
                        [&one_d_data[i], &one_d_data[j], &one_d_data[k]],
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
                    acc + calc_joint_nn(
                        [
                            &one_d_data[i],
                            &one_d_data[j],
                            &one_d_data[k],
                            &one_d_data[l],
                        ],
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
    let coordinate_entropies = coordinate_entropy_impl(&one_d_data, &metrics, options.constant)?;
    let degrees_freedom = coordinate_entropies.len();
    if mie_order == 1 || degrees_freedom == 1 {
        return Ok(coordinate_entropies);
    }

    let pairwise_mutual_information = coordinate_mutual_information_impl(
        &one_d_data,
        &metrics,
        options.constant,
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
) -> Result<Vec<f64>, String> {
    let n_frames: usize = one_d_data[0].len();
    let one_d_constant = entropy_constant_with_convention(n_frames, 1, constant)?;

    one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .map(|(ic, metric)| {
            calc_joint_nn([ic], [*metric]).map(|distance| {
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
    let one_d_entropies = coordinate_entropy_impl(&one_d_data, &metrics, options.constant)?;
    coordinate_mutual_information_impl(&one_d_data, &metrics, options.constant, &one_d_entropies)
}

fn coordinate_mutual_information_impl(
    one_d_data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    constant: FiniteSampleConstant,
    one_d_entropies: &[f64],
) -> Result<Vec<f64>, String> {
    let n_frames: usize = one_d_data[0].len();
    let degrees_freedom: usize = one_d_data.len();
    let two_d_constant = entropy_constant_with_convention(n_frames, 2, constant)?;
    let two_d_degrees_freedom = binomial(degrees_freedom, 2);

    let mutual_information: Vec<f64> = (0..two_d_degrees_freedom)
        .into_par_iter()
        .map(|rank| {
            let [i, j] = combination_from_rank::<2>(rank, degrees_freedom);
            let joint_entropy = estimate_entropy_efficient(
                calc_joint_nn([&one_d_data[i], &one_d_data[j]], [metrics[i], metrics[j]])? * 2.0,
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

/// Sum over frames of ln(distance to the nearest other frame) in the K-dimensional
/// joint space of `coordinates`, using each coordinate's metric.
#[allow(clippy::needless_range_loop)]
pub fn calc_joint_nn<const K: usize>(
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
