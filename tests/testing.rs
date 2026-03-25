use nn_entropy::*;
use assert_approx_eq::assert_approx_eq;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use std::path::PathBuf;

fn test_data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn test_one_d_nn_real_data() {
    let coord = vec![1.50252827, 1.56517294, 1.53902767, 1.55957774, 1.50624973, 1.53406964, 1.55834527, 1.54724624, 1.57642681, 1.59808848];
    let expected_value: f64 = -53.50470139636346; // Value from rust
    
    let ln_distance = calc_one_d_nn(&coord).expect("calc_one_d_nn failed");
    assert_approx_eq!(ln_distance, expected_value, 5e-2);
}

#[test]
fn test_one_d_nn_real_data_with_repeats() {
    let coord: Vec<f64> = vec![1.50252827, 1.50252827, 1.50252827, 1.50252827, 1.50252827, 1.2];
    let expected_value: f64 = -7.173483307994341; // Value from rust

    let ln_distance: f64 = calc_one_d_nn(&coord).expect("calc_one_d_nn failed");

    assert_approx_eq!(ln_distance, expected_value);
}

#[test]
fn test_one_d_nn_constant_series_is_invalid() {
    let coord: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
    let err = calc_one_d_nn(&coord).expect_err("expected error for constant series");
    assert!(err.contains("unique"));
}

#[test]
fn test_two_d_nn_real_data() {
    let coord_1: Vec<f64> = vec![1.32254237, 1.34113319, 1.38538372, 1.37740432, 1.38594803, 1.32188178, 1.37434198, 1.35183515, 1.29332546, 1.3097266];
    let coord_2: Vec<f64> = vec![1.29885442, 1.33961716, 1.3461671, 1.35140196, 1.33317387, 1.32170911, 1.33895471, 1.3128584, 1.39146681, 1.31523388];
    let expected_value: f64 = -40.28617063678864; // Value from rust
    let ln_distance: f64 = calc_two_d_nn(&coord_1, &coord_2).expect("calc_two_d_nn failed");
    assert_approx_eq!(ln_distance, expected_value);
}

// Helper function to generate Guassian data
pub fn generate_normal(mean: f64, std_dev: f64, size: usize) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::thread_rng();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

// Helper function to compare floating-point arrays with tolerance
fn assert_approx_eq_array(a: [f64; 3], b: [f64; 3]) {
    let epsilon = 1e-10;
    assert!((a[0] - b[0]).abs() < epsilon, "x component mismatch: {} vs {}", a[0], b[0]);
    assert!((a[1] - b[1]).abs() < epsilon, "y component mismatch: {} vs {}", a[1], b[1]);
    assert!((a[2] - b[2]).abs() < epsilon, "z component mismatch: {} vs {}", a[2], b[2]);
}
#[test]
fn test_cross_product() {
    // Known value test
    let b1 = [2.0, 3.0, 4.0];
    let b2 = [5.0, 6.0, 7.0];
    let result = cross_product(b1, b2);
    let expected_result: [f64; 3] = [-3.0, 6.0, -3.0];
    assert_approx_eq_array(result, expected_result);
}

#[test]
fn test_calc_torsion() {
    let a1: [f64; 3] = [70.73, -91.32219, -87.903145];
    let a2: [f64; 3] = [71.543846, -91.90568, -87.45525];
    let a3: [f64; 3] = [70.995895, -92.99476, -86.545074];
    let a4: [f64; 3] = [71.78272, -93.75866, -86.517555];
    let result: f64 = calc_torsion(a1, a2, a3, a4);
    let expected_result: f64 = -2.71362;
    assert_approx_eq!(result, expected_result, 3e-6);
}

#[test]
fn test_calc_angle() {
    let a1: [f64; 3] = [71.78272, -93.75866, -86.517555];
    let a2: [f64; 3] = [70.995895, -92.99476, -86.545074];
    let a3: [f64; 3] = [71.543846, -91.90568, -87.45525];
    let result: f64 = calc_angle(a1, a2, a3);
    assert_approx_eq!(result, 1.828803312585639);
}
#[test]
fn test_calc_bond() {
    let a1: [f64; 3] = [71.543846, -91.90568, -87.45525];
    let a2: [f64; 3] = [70.995895, -92.99476, -86.545074];
    let result: f64 = calc_bond(a1, a2);
    let expected_result: f64 = 1.5214378340021502;
    assert_approx_eq!(result, expected_result, 3e-6);
}

#[test]
fn test_calc_internal_coords() {
    let bat_list: Vec<Vec<usize>> = vec![
    vec![0, 1],
    vec![7, 1, 0],
    vec![2, 0, 1],
    vec![3, 0, 1],
    vec![4, 0, 1],
    vec![5, 1, 0],
    vec![6, 1, 0],
    vec![2, 0, 1, 7],
    vec![3, 0, 1, 7],
    vec![4, 0, 1, 7],
    vec![5, 1, 0, 2],
    vec![6, 1, 0, 2]
];
    let trajectory: Vec<Vec<[f64; 3]>> = vec![
    vec![
    [71.543846, -91.90568, -87.45525],
    [70.995895, -92.99476, -86.545074],
    [70.73, -91.32219, -87.903145],
    [72.076584, -91.16638, -86.84448],
    [72.18585, -92.22224, -88.28653],
    [70.13876, -93.54273, -86.95551],
    [70.85134, -92.65517, -85.512024],
    [71.78272, -93.75866, -86.517555]
]
    ];
    let expected_result: Vec<Vec<f64>> = vec![
    vec![
        1.52143783,
        1.82880331,
        1.93671239,
        1.90100054,
        2.04602296,
        1.99889629,
        1.97058798,
        -2.71361995,
        1.59733725,
        -0.5814991,
        -0.76085043,
        1.55080378
    ]
];
    let result = calc_internal_coords(bat_list, trajectory);
    assert_eq!(result.len(), expected_result.len());
    assert_eq!(result[0].len(), expected_result[0].len());
    for idx in 0..expected_result[0].len() {
        assert_approx_eq!(result[0][idx], expected_result[0][idx], 1e-5)
    }
    
}

#[test]
fn test_load_potential_energy_from_mdout_deduplicates_nstep_blocks() {
    let path = test_data_path("tests/fixtures/sample.prod.out");
    let energies = load_potential_energy_from_mdout(&path).expect("mdout parsing failed");
    assert_eq!(energies, vec![-11.0, -10.5, -9.75]);
}

#[test]
fn test_integrated_autocorrelation_time_detects_correlated_series() {
    let mut series = Vec::with_capacity(256);
    let mut state = 0.0f64;
    for idx in 0..256 {
        let drive = ((idx % 7) as f64) - 3.0;
        state = 0.9 * state + drive;
        series.push(state);
    }

    let tau = estimate_integrated_autocorrelation_time(&series)
        .expect("tau estimation failed");
    assert!(tau > 1.0, "expected correlated series to have tau > 1, got {tau}");
}

#[test]
fn test_build_default_subset_sizes_returns_increasing_prefixes() {
    let subsets = build_default_subset_sizes(50_000, 5).expect("subset generation failed");
    assert_eq!(subsets, vec![10_000, 20_000, 30_000, 40_000, 50_000]);
}

#[test]
fn test_build_extrapolation_plan_uses_tau_in_effective_sample_size() {
    let plan = build_extrapolation_plan(
        100,
        ExtrapolationConfig {
            subset_sizes: vec![20, 40, 100],
            fit_model: ExtrapolationFitModel::InverseN,
            tau: Some(5.0),
        },
    )
    .expect("plan construction failed");

    assert_eq!(plan.subsets.len(), 3);
    assert_approx_eq!(plan.subsets[0].effective_samples, 2.0);
    assert_approx_eq!(plan.subsets[0].x, 0.5);
    assert_approx_eq!(plan.subsets[2].effective_samples, 10.0);
    assert_approx_eq!(plan.subsets[2].x, 0.1);
}

#[test]
fn test_inverse_sqrt_n_fit_model_transform() {
    let x = ExtrapolationFitModel::InverseSqrtN
        .transform(25.0)
        .expect("transform failed");
    assert_approx_eq!(x, 0.2);
}

#[test]
fn test_build_extrapolation_plan_rejects_non_monotonic_subsets() {
    let err = build_extrapolation_plan(
        100,
        ExtrapolationConfig {
            subset_sizes: vec![20, 20, 50],
            fit_model: ExtrapolationFitModel::InverseN,
            tau: None,
        },
    )
    .expect_err("expected invalid subset sizes to fail");
    assert!(err.contains("strictly increasing"));
}

#[test]
fn test_generate_block_bootstrap_indices_has_expected_length_and_bounds() {
    let mut rng = StdRng::seed_from_u64(7);
    let indices =
        generate_block_bootstrap_indices(11, 3, &mut rng).expect("index generation failed");
    assert_eq!(indices.len(), 11);
    assert!(indices.iter().all(|&idx| idx < 11));
}

#[test]
fn test_bootstrap_extrapolation_data_returns_subset_statistics() {
    let one_d_data = vec![
        (0..60).map(|idx| idx as f64 * 0.1 + 0.01).collect::<Vec<_>>(),
        (0..60)
            .map(|idx| (idx as f64 * 0.13).sin() + idx as f64 * 0.005)
            .collect::<Vec<_>>(),
        (0..60)
            .map(|idx| (idx as f64 * 0.17).cos() + idx as f64 * 0.004)
            .collect::<Vec<_>>(),
    ];

    let plan = build_extrapolation_plan(
        60,
        ExtrapolationConfig {
            subset_sizes: vec![20, 40, 60],
            fit_model: ExtrapolationFitModel::InverseSqrtN,
            tau: Some(2.0),
        },
    )
    .expect("plan construction failed");

    let stats = bootstrap_extrapolation_data(
        &one_d_data,
        plan,
        BootstrapConfig {
            replicates: 4,
            block_size: 5,
            seed: Some(1234),
        },
    )
    .expect("bootstrap extrapolation failed");

    assert_eq!(stats.subsets.len(), 3);
    for subset in &stats.subsets {
        assert_eq!(subset.replicate_entropies.len(), 4);
        assert!(subset.mean_entropy.is_finite());
        assert!(subset.variance.is_finite());
        assert!(subset.variance >= 0.0);
    }
}

#[test]
fn test_fit_extrapolated_entropy_recovers_inverse_n_intercept() {
    let plan = build_extrapolation_plan(
        100,
        ExtrapolationConfig {
            subset_sizes: vec![25, 50, 100],
            fit_model: ExtrapolationFitModel::InverseN,
            tau: None,
        },
    )
    .expect("plan construction failed");

    let stats = BootstrapExtrapolationData {
        plan,
        bootstrap: BootstrapConfig {
            replicates: 4,
            block_size: 1,
            seed: Some(1),
        },
        subsets: vec![
            SubsetBootstrapStatistics {
                subset: ExtrapolationSubset {
                    raw_samples: 25,
                    effective_samples: 25.0,
                    x: 1.0 / 25.0,
                },
                replicate_entropies: vec![1.38, 1.4, 1.41, 1.39],
                mean_entropy: 1.4,
                variance: 0.01,
            },
            SubsetBootstrapStatistics {
                subset: ExtrapolationSubset {
                    raw_samples: 50,
                    effective_samples: 50.0,
                    x: 1.0 / 50.0,
                },
                replicate_entropies: vec![1.19, 1.2, 1.21, 1.2],
                mean_entropy: 1.2,
                variance: 0.01,
            },
            SubsetBootstrapStatistics {
                subset: ExtrapolationSubset {
                    raw_samples: 100,
                    effective_samples: 100.0,
                    x: 1.0 / 100.0,
                },
                replicate_entropies: vec![1.09, 1.1, 1.11, 1.1],
                mean_entropy: 1.1,
                variance: 0.01,
            },
        ],
    };

    let fit = fit_extrapolated_entropy(&stats).expect("fit failed");
    assert_approx_eq!(fit.intercept, 1.0, 1e-10);
    assert_approx_eq!(fit.slope, 10.0, 1e-10);
    assert_approx_eq!(fit.r_squared, 1.0, 1e-10);
    assert!(fit.intercept_std_err.is_finite());
    assert_eq!(fit.points.len(), 3);
}

#[test]
fn test_fit_extrapolated_entropy_recovers_inverse_sqrt_n_intercept() {
    let stats = BootstrapExtrapolationData {
        plan: ExtrapolationPlan {
            total_samples: 100,
            fit_model: ExtrapolationFitModel::InverseSqrtN,
            tau: None,
            subsets: vec![
                ExtrapolationSubset {
                    raw_samples: 25,
                    effective_samples: 25.0,
                    x: 0.2,
                },
                ExtrapolationSubset {
                    raw_samples: 100,
                    effective_samples: 100.0,
                    x: 0.1,
                },
                ExtrapolationSubset {
                    raw_samples: 400,
                    effective_samples: 400.0,
                    x: 0.05,
                },
            ],
        },
        bootstrap: BootstrapConfig {
            replicates: 4,
            block_size: 1,
            seed: Some(2),
        },
        subsets: vec![
            SubsetBootstrapStatistics {
                subset: ExtrapolationSubset {
                    raw_samples: 25,
                    effective_samples: 25.0,
                    x: 0.2,
                },
                replicate_entropies: vec![2.38, 2.4, 2.42, 2.4],
                mean_entropy: 2.4,
                variance: 0.02,
            },
            SubsetBootstrapStatistics {
                subset: ExtrapolationSubset {
                    raw_samples: 100,
                    effective_samples: 100.0,
                    x: 0.1,
                },
                replicate_entropies: vec![2.18, 2.19, 2.21, 2.2],
                mean_entropy: 2.2,
                variance: 0.02,
            },
            SubsetBootstrapStatistics {
                subset: ExtrapolationSubset {
                    raw_samples: 400,
                    effective_samples: 400.0,
                    x: 0.05,
                },
                replicate_entropies: vec![2.08, 2.09, 2.11, 2.1],
                mean_entropy: 2.1,
                variance: 0.02,
            },
        ],
    };

    let fit = fit_extrapolated_entropy(&stats).expect("fit failed");
    assert_approx_eq!(fit.intercept, 2.0, 1e-10);
    assert_approx_eq!(fit.slope, 2.0, 1e-10);
    assert_approx_eq!(fit.r_squared, 1.0, 1e-10);
    assert!(fit.weighted_residual_sum_squares <= 1e-10);
}

#[test]
fn test_run_entropy_extrapolation_returns_full_report() {
    let one_d_data = vec![
        (0..40).map(|idx| idx as f64 * 0.07 + 0.01).collect::<Vec<_>>(),
        (0..40)
            .map(|idx| (idx as f64 * 0.19).sin() + idx as f64 * 0.004)
            .collect::<Vec<_>>(),
        (0..40)
            .map(|idx| (idx as f64 * 0.11).cos() + idx as f64 * 0.003)
            .collect::<Vec<_>>(),
    ];

    let report = run_entropy_extrapolation(
        &one_d_data,
        EntropyExtrapolationConfig {
            subset_sizes: Some(vec![20, 30, 40]),
            subset_count: 3,
            fit_model: ExtrapolationFitModel::InverseN,
            compare_models: true,
            bootstrap_replicates: Some(4),
            block_size: Some(4),
            bootstrap_seed: Some(99),
            tau: Some(2.0),
            show_progress: false,
        },
        None,
    )
    .expect("high-level extrapolation failed");

    assert!(report.total_entropy.is_finite());
    assert_eq!(report.primary.bootstrap_data.subsets.len(), 3);
    assert_eq!(report.primary.fit.points.len(), 3);
    assert!(report.primary.fit.intercept.is_finite());
    assert_eq!(report.comparisons.len(), 1);
    assert_eq!(report.primary.trailing_subset_fits.len(), 1);
}

#[test]
fn test_run_entropy_extrapolation_model_comparison_uses_both_models() {
    let one_d_data = vec![
        (0..50).map(|idx| idx as f64 * 0.05 + 0.01).collect::<Vec<_>>(),
        (0..50)
            .map(|idx| (idx as f64 * 0.07).sin() + idx as f64 * 0.002)
            .collect::<Vec<_>>(),
        (0..50)
            .map(|idx| (idx as f64 * 0.13).cos() + idx as f64 * 0.003)
            .collect::<Vec<_>>(),
    ];

    let report = run_entropy_extrapolation(
        &one_d_data,
        EntropyExtrapolationConfig {
            subset_sizes: Some(vec![20, 30, 40, 50]),
            subset_count: 4,
            fit_model: ExtrapolationFitModel::InverseSqrtN,
            compare_models: true,
            bootstrap_replicates: Some(3),
            block_size: Some(3),
            bootstrap_seed: Some(5),
            tau: Some(1.5),
            show_progress: false,
        },
        None,
    )
    .expect("extrapolation with model comparison failed");

    assert_eq!(report.primary.model, ExtrapolationFitModel::InverseSqrtN);
    assert_eq!(report.comparisons.len(), 1);
    assert_eq!(report.comparisons[0].model, ExtrapolationFitModel::InverseN);
    assert_eq!(report.primary.trailing_subset_fits.len(), 2);
    assert!(report
        .primary
        .trailing_subset_fits
        .iter()
        .all(|result| result.fit.intercept.is_finite()));
}

#[test]
fn test_extrapolation_report_to_csv_includes_models_and_stability_rows() {
    let one_d_data = vec![
        (0..40).map(|idx| idx as f64 * 0.07 + 0.01).collect::<Vec<_>>(),
        (0..40)
            .map(|idx| (idx as f64 * 0.19).sin() + idx as f64 * 0.004)
            .collect::<Vec<_>>(),
        (0..40)
            .map(|idx| (idx as f64 * 0.11).cos() + idx as f64 * 0.003)
            .collect::<Vec<_>>(),
    ];

    let report = run_entropy_extrapolation(
        &one_d_data,
        EntropyExtrapolationConfig {
            subset_sizes: Some(vec![20, 30, 40]),
            subset_count: 3,
            fit_model: ExtrapolationFitModel::InverseN,
            compare_models: true,
            bootstrap_replicates: Some(3),
            block_size: Some(3),
            bootstrap_seed: Some(8),
            tau: Some(2.0),
            show_progress: false,
        },
        None,
    )
    .expect("report generation failed");

    let csv = extrapolation_report_to_csv(&report);
    assert!(csv.contains("row_type,series,model"));
    assert!(csv.contains("r_squared"));
    assert!(csv.contains("fit_point,primary,inverse-n"));
    assert!(csv.contains("comparison,inverse-sqrt-n"));
    assert!(csv.contains("stability,primary,inverse-n"));
}

#[test]
fn test_run_entropy_extrapolation_without_bootstrap_uses_raw_subset_curve() {
    let one_d_data = vec![
        (0..30).map(|idx| idx as f64 * 0.09 + 0.01).collect::<Vec<_>>(),
        (0..30)
            .map(|idx| (idx as f64 * 0.17).sin() + idx as f64 * 0.002)
            .collect::<Vec<_>>(),
        (0..30)
            .map(|idx| (idx as f64 * 0.13).cos() + idx as f64 * 0.003)
            .collect::<Vec<_>>(),
    ];

    let report = run_entropy_extrapolation(
        &one_d_data,
        EntropyExtrapolationConfig {
            subset_sizes: Some(vec![10, 20, 30]),
            subset_count: 3,
            fit_model: ExtrapolationFitModel::InverseN,
            compare_models: false,
            bootstrap_replicates: None,
            block_size: None,
            bootstrap_seed: None,
            tau: None,
            show_progress: false,
        },
        None,
    )
    .expect("no-bootstrap extrapolation failed");

    assert_eq!(report.primary.bootstrap_data.bootstrap.replicates, 0);
    assert_eq!(report.primary.bootstrap_data.subsets.len(), 3);
    assert!(report.primary.fit.intercept.is_finite());
}
