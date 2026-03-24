use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::calculate_entropy_from_data;
use crate::estimate_tau_from_files;
use crate::load_internal_coordinate_data_from_files;
use crate::CoordinateSelection;
use crate::TauEstimate;
use crate::TauObservable;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationFitModel {
    InverseN,
    InverseSqrtN,
}

impl ExtrapolationFitModel {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "inverse-n" | "1/n" => Ok(Self::InverseN),
            "inverse-sqrt-n" | "1/sqrt(n)" | "sqrt(1/n)" => Ok(Self::InverseSqrtN),
            _ => Err(format!(
                "unknown fit model '{value}'; expected one of: inverse-n, inverse-sqrt-n"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::InverseN => "inverse-n",
            Self::InverseSqrtN => "inverse-sqrt-n",
        }
    }

    pub fn transform(self, effective_samples: f64) -> Result<f64, String> {
        if !effective_samples.is_finite() || effective_samples <= 0.0 {
            return Err(format!(
                "effective sample size must be positive and finite, got {effective_samples}"
            ));
        }

        Ok(match self {
            Self::InverseN => effective_samples.recip(),
            Self::InverseSqrtN => effective_samples.sqrt().recip(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExtrapolationConfig {
    pub subset_sizes: Vec<usize>,
    pub fit_model: ExtrapolationFitModel,
    pub tau: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ExtrapolationSubset {
    pub raw_samples: usize,
    pub effective_samples: f64,
    pub x: f64,
}

#[derive(Debug, Clone)]
pub struct ExtrapolationPlan {
    pub total_samples: usize,
    pub fit_model: ExtrapolationFitModel,
    pub tau: Option<f64>,
    pub subsets: Vec<ExtrapolationSubset>,
}

#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    pub replicates: usize,
    pub block_size: usize,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SubsetBootstrapStatistics {
    pub subset: ExtrapolationSubset,
    pub replicate_entropies: Vec<f64>,
    pub mean_entropy: f64,
    pub variance: f64,
}

#[derive(Debug, Clone)]
pub struct BootstrapExtrapolationData {
    pub plan: ExtrapolationPlan,
    pub bootstrap: BootstrapConfig,
    pub subsets: Vec<SubsetBootstrapStatistics>,
}

#[derive(Debug, Clone)]
pub struct ExtrapolationFitPoint {
    pub raw_samples: usize,
    pub effective_samples: f64,
    pub x: f64,
    pub y: f64,
    pub variance: f64,
    pub weight: f64,
    pub residual: f64,
}

#[derive(Debug, Clone)]
pub struct LinearFitResult {
    pub model: ExtrapolationFitModel,
    pub intercept: f64,
    pub slope: f64,
    pub intercept_std_err: f64,
    pub slope_std_err: f64,
    pub covariance_intercept_slope: f64,
    pub weighted_residual_sum_squares: f64,
    pub points: Vec<ExtrapolationFitPoint>,
}

#[derive(Debug, Clone)]
pub struct StabilityFitResult {
    pub subset_count: usize,
    pub fit: LinearFitResult,
}

#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    pub model: ExtrapolationFitModel,
    pub bootstrap_data: BootstrapExtrapolationData,
    pub fit: LinearFitResult,
    pub trailing_subset_fits: Vec<StabilityFitResult>,
}

#[derive(Debug, Clone)]
pub struct EntropyExtrapolationConfig {
    pub subset_sizes: Option<Vec<usize>>,
    pub subset_count: usize,
    pub fit_model: ExtrapolationFitModel,
    pub compare_models: bool,
    pub bootstrap_replicates: Option<usize>,
    pub block_size: Option<usize>,
    pub bootstrap_seed: Option<u64>,
    pub tau: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct EntropyExtrapolationReport {
    pub total_entropy: f64,
    pub tau_estimate: Option<TauEstimate>,
    pub primary: ModelDiagnostics,
    pub comparisons: Vec<ModelDiagnostics>,
}

fn write_model_csv_rows(output: &mut String, label: &str, diagnostics: &ModelDiagnostics) {
    for point in &diagnostics.fit.points {
        let fitted_entropy = diagnostics.fit.intercept + diagnostics.fit.slope * point.x;
        output.push_str(&format!(
            "fit_point,{label},{},{},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15},{:.15}\n",
            diagnostics.model.as_str(),
            point.raw_samples,
            point.effective_samples,
            point.x,
            point.y,
            point.variance,
            point.weight,
            fitted_entropy,
            point.residual,
            diagnostics.fit.intercept,
            diagnostics.fit.intercept_std_err,
            diagnostics.fit.slope,
            diagnostics.fit.slope_std_err,
            diagnostics.fit.weighted_residual_sum_squares,
        ));
    }

    for stability in &diagnostics.trailing_subset_fits {
        output.push_str(&format!(
            "stability,{label},{},{},,,,,,,,{:.15},{:.15},{:.15},{:.15},{:.15}\n",
            diagnostics.model.as_str(),
            stability.subset_count,
            stability.fit.intercept,
            stability.fit.intercept_std_err,
            stability.fit.slope,
            stability.fit.slope_std_err,
            stability.fit.weighted_residual_sum_squares,
        ));
    }
}

pub fn extrapolation_report_to_csv(report: &EntropyExtrapolationReport) -> String {
    let mut output = String::from(
        "row_type,series,model,n_or_subset_count,effective_samples,x,mean_entropy,variance,weight,fitted_entropy,residual,intercept,intercept_std_err,slope,slope_std_err,weighted_residual_sum_squares\n",
    );
    write_model_csv_rows(&mut output, "primary", &report.primary);
    for diagnostics in &report.comparisons {
        write_model_csv_rows(&mut output, "comparison", diagnostics);
    }
    output
}

pub fn effective_sample_size(raw_samples: usize, tau: Option<f64>) -> Result<f64, String> {
    if raw_samples < 2 {
        return Err(format!(
            "need at least two samples per subset, got {raw_samples}"
        ));
    }

    match tau {
        None => Ok(raw_samples as f64),
        Some(tau) => {
            if !tau.is_finite() || tau <= 0.0 {
                return Err(format!("tau must be positive and finite, got {tau}"));
            }
            Ok(((raw_samples as f64) / (2.0 * tau)).max(1.0))
        }
    }
}

pub fn validate_subset_sizes(total_samples: usize, subset_sizes: &[usize]) -> Result<(), String> {
    if total_samples < 2 {
        return Err(format!(
            "need at least two total samples, got {total_samples}"
        ));
    }
    if subset_sizes.len() < 2 {
        return Err("need at least two subset sizes for extrapolation".to_string());
    }

    let mut prev = 0usize;
    for (idx, &size) in subset_sizes.iter().enumerate() {
        if size < 2 {
            return Err(format!(
                "subset size at index {idx} must be at least 2, got {size}"
            ));
        }
        if size > total_samples {
            return Err(format!(
                "subset size at index {idx} exceeds total samples: {size} > {total_samples}"
            ));
        }
        if idx > 0 && size <= prev {
            return Err(format!(
                "subset sizes must be strictly increasing, got {} then {}",
                prev, size
            ));
        }
        prev = size;
    }

    Ok(())
}

pub fn build_default_subset_sizes(
    total_samples: usize,
    target_count: usize,
) -> Result<Vec<usize>, String> {
    if target_count < 2 {
        return Err(format!(
            "need at least two subset sizes, got target_count={target_count}"
        ));
    }
    if total_samples < target_count {
        return Err(format!(
            "total samples {total_samples} is smaller than target subset count {target_count}"
        ));
    }

    let mut subset_sizes = Vec::with_capacity(target_count);
    for idx in 1..=target_count {
        let numerator = idx * total_samples;
        let size = numerator.div_ceil(target_count);
        if size >= 2 && subset_sizes.last().copied() != Some(size) {
            subset_sizes.push(size);
        }
    }

    if subset_sizes.last().copied() != Some(total_samples) {
        subset_sizes.push(total_samples);
    }

    validate_subset_sizes(total_samples, &subset_sizes)?;
    Ok(subset_sizes)
}

pub fn build_extrapolation_plan(
    total_samples: usize,
    config: ExtrapolationConfig,
) -> Result<ExtrapolationPlan, String> {
    validate_subset_sizes(total_samples, &config.subset_sizes)?;

    let mut subsets = Vec::with_capacity(config.subset_sizes.len());
    for &raw_samples in &config.subset_sizes {
        let effective_samples = effective_sample_size(raw_samples, config.tau)?;
        let x = config.fit_model.transform(effective_samples)?;
        subsets.push(ExtrapolationSubset {
            raw_samples,
            effective_samples,
            x,
        });
    }

    Ok(ExtrapolationPlan {
        total_samples,
        fit_model: config.fit_model,
        tau: config.tau,
        subsets,
    })
}

pub fn validate_bootstrap_config(config: &BootstrapConfig) -> Result<(), String> {
    if config.replicates == 0 {
        return Err("bootstrap replicates must be at least 1".to_string());
    }
    if config.block_size == 0 {
        return Err("bootstrap block size must be at least 1".to_string());
    }
    Ok(())
}

fn validate_coordinate_matrix(one_d_data: &[Vec<f64>]) -> Result<usize, String> {
    if one_d_data.is_empty() {
        return Err("no coordinate data provided for bootstrap extrapolation".to_string());
    }
    let frame_count = one_d_data[0].len();
    if frame_count < 2 {
        return Err("need at least two frames for bootstrap extrapolation".to_string());
    }
    for (coord_idx, coord) in one_d_data.iter().enumerate() {
        if coord.len() != frame_count {
            return Err(format!(
                "coordinate {coord_idx} has {} samples, expected {frame_count}",
                coord.len()
            ));
        }
    }
    Ok(frame_count)
}

pub fn generate_block_bootstrap_indices<R: Rng + ?Sized>(
    sample_count: usize,
    block_size: usize,
    rng: &mut R,
) -> Result<Vec<usize>, String> {
    if sample_count < 2 {
        return Err(format!(
            "need at least two samples to bootstrap, got {sample_count}"
        ));
    }
    if block_size == 0 {
        return Err("bootstrap block size must be at least 1".to_string());
    }

    let mut indices = Vec::with_capacity(sample_count);
    while indices.len() < sample_count {
        let start = rng.gen_range(0..sample_count);
        for offset in 0..block_size {
            if indices.len() == sample_count {
                break;
            }
            indices.push((start + offset) % sample_count);
        }
    }
    Ok(indices)
}

fn sample_coordinate_prefix(
    one_d_data: &[Vec<f64>],
    raw_samples: usize,
    indices: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    let mut sampled = Vec::with_capacity(one_d_data.len());
    for (coord_idx, coord) in one_d_data.iter().enumerate() {
        if coord.len() < raw_samples {
            return Err(format!(
                "coordinate {coord_idx} has {} samples, expected at least {raw_samples}",
                coord.len()
            ));
        }
        let prefix = &coord[..raw_samples];
        let mut sampled_coord = Vec::with_capacity(indices.len());
        for &index in indices {
            sampled_coord.push(prefix[index]);
        }
        sampled.push(sampled_coord);
    }
    Ok(sampled)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64
}

pub fn bootstrap_extrapolation_data(
    one_d_data: &[Vec<f64>],
    plan: ExtrapolationPlan,
    bootstrap: BootstrapConfig,
) -> Result<BootstrapExtrapolationData, String> {
    validate_bootstrap_config(&bootstrap)?;
    let total_samples = validate_coordinate_matrix(one_d_data)?;
    if total_samples != plan.total_samples {
        return Err(format!(
            "plan total_samples={} does not match data sample count={total_samples}",
            plan.total_samples
        ));
    }

    let mut rng = match bootstrap.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut subset_stats = Vec::with_capacity(plan.subsets.len());
    for subset in &plan.subsets {
        let mut replicate_entropies = Vec::with_capacity(bootstrap.replicates);
        for _ in 0..bootstrap.replicates {
            let indices =
                generate_block_bootstrap_indices(subset.raw_samples, bootstrap.block_size, &mut rng)?;
            let sampled = sample_coordinate_prefix(one_d_data, subset.raw_samples, &indices)?;
            let entropy = calculate_entropy_from_data(sampled, subset.raw_samples).map_err(|err| {
                format!(
                    "entropy calculation failed for subset {} during bootstrap: {err}",
                    subset.raw_samples
                )
            })?;
            replicate_entropies.push(entropy);
        }
        let mean_entropy = mean(&replicate_entropies);
        let variance = sample_variance(&replicate_entropies, mean_entropy);
        subset_stats.push(SubsetBootstrapStatistics {
            subset: subset.clone(),
            replicate_entropies,
            mean_entropy,
            variance,
        });
    }

    Ok(BootstrapExtrapolationData {
        plan,
        bootstrap,
        subsets: subset_stats,
    })
}

pub fn fit_extrapolated_entropy(
    data: &BootstrapExtrapolationData,
) -> Result<LinearFitResult, String> {
    if data.subsets.len() < 2 {
        return Err("need at least two subset statistics to fit extrapolated entropy".to_string());
    }

    let mut s = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    let mut normalized_points = Vec::with_capacity(data.subsets.len());
    for subset in &data.subsets {
        if !subset.mean_entropy.is_finite() {
            return Err(format!(
                "subset {} has non-finite mean entropy",
                subset.subset.raw_samples
            ));
        }
        if !subset.subset.x.is_finite() {
            return Err(format!(
                "subset {} has non-finite fit coordinate",
                subset.subset.raw_samples
            ));
        }

        let variance = if subset.variance.is_finite() && subset.variance > 0.0 {
            subset.variance
        } else {
            1e-12
        };
        let weight = variance.recip();
        let x = subset.subset.x;
        let y = subset.mean_entropy;

        s += weight;
        sx += weight * x;
        sy += weight * y;
        sxx += weight * x * x;
        sxy += weight * x * y;

        normalized_points.push((subset, variance, weight));
    }

    let delta = s * sxx - sx * sx;
    if !delta.is_finite() || delta.abs() <= f64::EPSILON {
        return Err("weighted fit is singular; subset x values do not span a usable range".to_string());
    }

    let intercept = (sxx * sy - sx * sxy) / delta;
    let slope = (s * sxy - sx * sy) / delta;

    let covariance_00 = sxx / delta;
    let covariance_11 = s / delta;
    let covariance_01 = -sx / delta;

    let mut points = Vec::with_capacity(normalized_points.len());
    let mut weighted_residual_sum_squares = 0.0;
    for (subset, variance, weight) in normalized_points {
        let predicted = intercept + slope * subset.subset.x;
        let residual = subset.mean_entropy - predicted;
        weighted_residual_sum_squares += weight * residual * residual;
        points.push(ExtrapolationFitPoint {
            raw_samples: subset.subset.raw_samples,
            effective_samples: subset.subset.effective_samples,
            x: subset.subset.x,
            y: subset.mean_entropy,
            variance,
            weight,
            residual,
        });
    }

    Ok(LinearFitResult {
        model: data.plan.fit_model,
        intercept,
        slope,
        intercept_std_err: covariance_00.sqrt(),
        slope_std_err: covariance_11.sqrt(),
        covariance_intercept_slope: covariance_01,
        weighted_residual_sum_squares,
        points,
    })
}

fn bootstrap_data_with_model(
    one_d_data: &[Vec<f64>],
    subset_sizes: &[usize],
    fit_model: ExtrapolationFitModel,
    tau: Option<f64>,
    bootstrap: &BootstrapConfig,
) -> Result<BootstrapExtrapolationData, String> {
    let total_samples = validate_coordinate_matrix(one_d_data)?;
    let plan = build_extrapolation_plan(
        total_samples,
        ExtrapolationConfig {
            subset_sizes: subset_sizes.to_vec(),
            fit_model,
            tau,
        },
    )?;
    bootstrap_extrapolation_data(one_d_data, plan, bootstrap.clone())
}

fn trailing_subset_fit(
    stats: &[SubsetBootstrapStatistics],
    model: ExtrapolationFitModel,
    tau: Option<f64>,
    bootstrap: &BootstrapConfig,
) -> Result<LinearFitResult, String> {
    if stats.len() < 2 {
        return Err("need at least two subset statistics for stability fit".to_string());
    }
    let total_samples = stats
        .last()
        .map(|subset| subset.subset.raw_samples)
        .ok_or("missing subset statistics".to_string())?;
    let plan = build_extrapolation_plan(
        total_samples,
        ExtrapolationConfig {
            subset_sizes: stats.iter().map(|subset| subset.subset.raw_samples).collect(),
            fit_model: model,
            tau,
        },
    )?;
    let remapped_subsets = stats
        .iter()
        .zip(plan.subsets.iter())
        .map(|(original, subset)| SubsetBootstrapStatistics {
            subset: subset.clone(),
            replicate_entropies: original.replicate_entropies.clone(),
            mean_entropy: original.mean_entropy,
            variance: original.variance,
        })
        .collect();
    fit_extrapolated_entropy(&BootstrapExtrapolationData {
        plan,
        bootstrap: bootstrap.clone(),
        subsets: remapped_subsets,
    })
}

fn compute_trailing_subset_fits(
    data: &BootstrapExtrapolationData,
) -> Result<Vec<StabilityFitResult>, String> {
    let total = data.subsets.len();
    let mut results = Vec::new();
    for subset_count in 2..total {
        let start = total - subset_count;
        let fit = trailing_subset_fit(
            &data.subsets[start..],
            data.plan.fit_model,
            data.plan.tau,
            &data.bootstrap,
        )?;
        results.push(StabilityFitResult { subset_count, fit });
    }
    Ok(results)
}

fn build_model_diagnostics(
    one_d_data: &[Vec<f64>],
    subset_sizes: &[usize],
    fit_model: ExtrapolationFitModel,
    tau: Option<f64>,
    bootstrap: Option<&BootstrapConfig>,
) -> Result<ModelDiagnostics, String> {
    let bootstrap_data = match bootstrap {
        Some(bootstrap) => bootstrap_data_with_model(
            one_d_data,
            subset_sizes,
            fit_model,
            tau,
            bootstrap,
        )?,
        None => {
            let total_samples = validate_coordinate_matrix(one_d_data)?;
            let plan = build_extrapolation_plan(
                total_samples,
                ExtrapolationConfig {
                    subset_sizes: subset_sizes.to_vec(),
                    fit_model,
                    tau,
                },
            )?;
            let mut subsets = Vec::with_capacity(plan.subsets.len());
            for subset in &plan.subsets {
                let entropy = calculate_entropy_from_data(one_d_data.to_vec(), subset.raw_samples)
                    .map_err(|err| {
                        format!(
                            "entropy calculation failed for subset {}: {err}",
                            subset.raw_samples
                        )
                    })?;
                subsets.push(SubsetBootstrapStatistics {
                    subset: subset.clone(),
                    replicate_entropies: vec![entropy],
                    mean_entropy: entropy,
                    variance: 1.0,
                });
            }
            BootstrapExtrapolationData {
                plan,
                bootstrap: BootstrapConfig {
                    replicates: 0,
                    block_size: 0,
                    seed: None,
                },
                subsets,
            }
        }
    };
    let fit = fit_extrapolated_entropy(&bootstrap_data)?;
    let trailing_subset_fits = compute_trailing_subset_fits(&bootstrap_data)?;
    Ok(ModelDiagnostics {
        model: fit_model,
        bootstrap_data,
        fit,
        trailing_subset_fits,
    })
}

pub fn run_entropy_extrapolation(
    one_d_data: &[Vec<f64>],
    config: EntropyExtrapolationConfig,
    tau_estimate: Option<TauEstimate>,
) -> Result<EntropyExtrapolationReport, String> {
    let total_samples = validate_coordinate_matrix(one_d_data)?;
    let total_entropy = calculate_entropy_from_data(one_d_data.to_vec(), total_samples)?;

    let subset_sizes = match config.subset_sizes {
        Some(subset_sizes) => subset_sizes,
        None => build_default_subset_sizes(total_samples, config.subset_count)?,
    };
    let tau = config.tau.or_else(|| tau_estimate.as_ref().map(|estimate| estimate.tau));

    let block_size = match config.block_size {
        Some(block_size) => block_size,
        None => tau
            .map(|tau| tau.ceil().max(1.0) as usize)
            .unwrap_or(1),
    };

    let bootstrap = config.bootstrap_replicates.map(|replicates| BootstrapConfig {
        replicates,
        block_size,
        seed: config.bootstrap_seed,
    });
    let primary = build_model_diagnostics(
        one_d_data,
        &subset_sizes,
        config.fit_model,
        tau,
        bootstrap.as_ref(),
    )?;
    let comparisons = if config.compare_models {
        let models = [ExtrapolationFitModel::InverseN, ExtrapolationFitModel::InverseSqrtN];
        let mut diagnostics = Vec::new();
        for model in models {
            if model == config.fit_model {
                continue;
            }
            diagnostics.push(build_model_diagnostics(
                one_d_data,
                &subset_sizes,
                model,
                tau,
                bootstrap.as_ref(),
            )?);
        }
        diagnostics
    } else {
        Vec::new()
    };

    Ok(EntropyExtrapolationReport {
        total_entropy,
        tau_estimate,
        primary,
        comparisons,
    })
}

pub fn run_entropy_extrapolation_from_files(
    top_path: &std::path::Path,
    traj_path: &std::path::Path,
    start: usize,
    stop: Option<usize>,
    selection: CoordinateSelection,
    tau_observable: Option<TauObservable>,
    mdout_path: Option<&std::path::Path>,
    config: EntropyExtrapolationConfig,
) -> Result<EntropyExtrapolationReport, String> {
    let frames = stop.unwrap_or(usize::MAX);
    let mut one_d_data =
        load_internal_coordinate_data_from_files(top_path, traj_path, frames, selection)?;
    if one_d_data.is_empty() {
        return Err("no coordinate data loaded for extrapolation".to_string());
    }
    if start >= one_d_data[0].len() {
        return Err(format!(
            "start {start} is beyond available frames ({})",
            one_d_data[0].len()
        ));
    }
    for coord in &mut one_d_data {
        *coord = coord[start..].to_vec();
    }

    let tau_estimate = match tau_observable {
        Some(observable) => Some(estimate_tau_from_files(
            top_path,
            traj_path,
            start,
            stop,
            observable,
            mdout_path,
        )?),
        None => None,
    };

    run_entropy_extrapolation(&one_d_data, config, tau_estimate)
}
