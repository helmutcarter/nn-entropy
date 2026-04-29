use assert_approx_eq::assert_approx_eq;
use nn_entropy::*;
use rand_distr::{Distribution, Normal};

#[test]
fn test_one_d_nn_real_data() {
    let coord = vec![
        1.50252827, 1.56517294, 1.53902767, 1.55957774, 1.50624973, 1.53406964, 1.55834527,
        1.54724624, 1.57642681, 1.59808848,
    ];
    let expected_value: f64 = -53.50470139636346; // Value from rust

    let ln_distance = calc_one_d_nn(&coord).expect("calc_one_d_nn failed");
    assert_approx_eq!(ln_distance, expected_value, 5e-2);
}

#[test]
fn test_one_d_nn_real_data_with_repeats() {
    let coord: Vec<f64> = vec![
        1.50252827, 1.50252827, 1.50252827, 1.50252827, 1.50252827, 1.2,
    ];
    let expected_value: f64 = -7.173483307994341; // Value from rust

    let ln_distance: f64 = calc_one_d_nn(&coord).expect("calc_one_d_nn failed");

    assert_approx_eq!(ln_distance, expected_value);
}

#[test]
fn test_one_d_nn_non_adjacent_repeats_are_deduplicated() {
    let coord: Vec<f64> = vec![1.0, 2.0, 1.0, 3.0];
    let ln_distance = calc_one_d_nn(&coord).expect("calc_one_d_nn failed");

    assert!(ln_distance.is_finite(), "expected finite entropy distance");
    assert_approx_eq!(ln_distance, 0.0);
}

#[test]
fn test_one_d_nn_constant_series_is_invalid() {
    let coord: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
    let err = calc_one_d_nn(&coord).expect_err("expected error for constant series");
    assert!(err.contains("unique"));
}

#[test]
fn test_two_d_nn_real_data() {
    let coord_1: Vec<f64> = vec![
        1.32254237, 1.34113319, 1.38538372, 1.37740432, 1.38594803, 1.32188178, 1.37434198,
        1.35183515, 1.29332546, 1.3097266,
    ];
    let coord_2: Vec<f64> = vec![
        1.29885442, 1.33961716, 1.3461671, 1.35140196, 1.33317387, 1.32170911, 1.33895471,
        1.3128584, 1.39146681, 1.31523388,
    ];
    let expected_value: f64 = -40.28617063678864; // Value from rust
    let ln_distance: f64 = calc_two_d_nn(&coord_1, &coord_2).expect("calc_two_d_nn failed");
    assert_approx_eq!(ln_distance, expected_value);
}

#[test]
fn test_dual_tree_backend_matches_kdtree_for_two_d() {
    let coord_1 = vec![0.2, 1.1, 0.7, 2.4, 3.0, 1.8, 2.2, 0.4];
    let coord_2 = vec![1.3, 0.5, 2.1, 1.7, 0.9, 2.8, 3.3, 0.2];

    let kdtree =
        calc_two_d_nn_with_backend(&coord_1, &coord_2, JointNearestBackend::KdTree).unwrap();
    let dual_tree =
        calc_two_d_nn_with_backend(&coord_1, &coord_2, JointNearestBackend::DualTree).unwrap();

    assert_approx_eq!(dual_tree, kdtree);
}

#[test]
fn test_dual_tree_backend_matches_kdtree_for_three_d() {
    let coord_1 = vec![0.2, 1.1, 0.7, 2.4, 3.0, 1.8, 2.2, 0.4];
    let coord_2 = vec![1.3, 0.5, 2.1, 1.7, 0.9, 2.8, 3.3, 0.2];
    let coord_3 = vec![2.9, 2.1, 0.4, 3.6, 1.2, 0.8, 2.4, 1.6];

    let kdtree =
        calc_three_d_nn_with_backend(&coord_1, &coord_2, &coord_3, JointNearestBackend::KdTree)
            .unwrap();
    let dual_tree =
        calc_three_d_nn_with_backend(&coord_1, &coord_2, &coord_3, JointNearestBackend::DualTree)
            .unwrap();

    assert_approx_eq!(dual_tree, kdtree);
}

#[test]
fn test_dual_tree_backend_matches_kdtree_for_four_d() {
    let coord_1 = vec![0.2, 1.1, 0.7, 2.4, 3.0, 1.8, 2.2, 0.4];
    let coord_2 = vec![1.3, 0.5, 2.1, 1.7, 0.9, 2.8, 3.3, 0.2];
    let coord_3 = vec![2.9, 2.1, 0.4, 3.6, 1.2, 0.8, 2.4, 1.6];
    let coord_4 = vec![1.7, 3.1, 2.5, 0.6, 3.4, 2.2, 0.1, 1.4];

    let kdtree = calc_four_d_nn_with_backend(
        &coord_1,
        &coord_2,
        &coord_3,
        &coord_4,
        JointNearestBackend::KdTree,
    )
    .unwrap();
    let dual_tree = calc_four_d_nn_with_backend(
        &coord_1,
        &coord_2,
        &coord_3,
        &coord_4,
        JointNearestBackend::DualTree,
    )
    .unwrap();

    assert_approx_eq!(dual_tree, kdtree);
}

#[test]
fn test_dual_tree_backend_matches_kdtree_with_duplicate_frames() {
    let coord_1 = vec![0.2, 0.2, 1.0, 2.0, 2.5, 3.0];
    let coord_2 = vec![1.1, 1.1, 1.5, 2.4, 2.8, 3.2];

    let kdtree =
        calc_two_d_nn_with_backend(&coord_1, &coord_2, JointNearestBackend::KdTree).unwrap();
    let dual_tree =
        calc_two_d_nn_with_backend(&coord_1, &coord_2, JointNearestBackend::DualTree).unwrap();

    assert_approx_eq!(dual_tree, kdtree);
}

#[test]
fn test_dual_tree_backend_matches_kdtree_after_recursive_splits() {
    let coord_1 = (0..48)
        .map(|i| ((i * 17 % 41) as f64) * 0.13 + (i as f64).sin() * 0.01)
        .collect::<Vec<_>>();
    let coord_2 = (0..48)
        .map(|i| ((i * 11 % 37) as f64) * 0.17 + (i as f64).cos() * 0.01)
        .collect::<Vec<_>>();
    let coord_3 = (0..48)
        .map(|i| ((i * 7 % 43) as f64) * 0.19 + ((i * i) as f64).sin() * 0.01)
        .collect::<Vec<_>>();

    let kdtree =
        calc_three_d_nn_with_backend(&coord_1, &coord_2, &coord_3, JointNearestBackend::KdTree)
            .unwrap();
    let dual_tree =
        calc_three_d_nn_with_backend(&coord_1, &coord_2, &coord_3, JointNearestBackend::DualTree)
            .unwrap();

    assert_approx_eq!(dual_tree, kdtree);
}

#[test]
fn test_calculate_entropy_order_two_matches_default() {
    let data = vec![
        vec![0.1, 0.4, 0.8, 1.1, 1.7],
        vec![1.0, 1.3, 1.9, 2.2, 2.8],
        vec![2.0, 2.4, 2.7, 3.1, 3.5],
    ];
    let default_entropy =
        calculate_entropy_from_data(data.clone(), 5).expect("default entropy failed");
    let order_two_entropy =
        calculate_entropy_from_data_with_order(data, 5, 2).expect("order-2 entropy failed");

    assert_approx_eq!(default_entropy, order_two_entropy);
}

#[test]
fn test_calculate_entropy_dual_tree_backend_matches_kdtree_backend() {
    let data = vec![
        (0..24)
            .map(|i| ((i * 17 % 41) as f64) * 0.13 + (i as f64).sin() * 0.01)
            .collect::<Vec<_>>(),
        (0..24)
            .map(|i| ((i * 11 % 37) as f64) * 0.17 + (i as f64).cos() * 0.01)
            .collect::<Vec<_>>(),
        (0..24)
            .map(|i| ((i * 7 % 43) as f64) * 0.19 + ((i * i) as f64).sin() * 0.01)
            .collect::<Vec<_>>(),
        (0..24)
            .map(|i| ((i * 5 % 31) as f64) * 0.23 + ((i + 3) as f64).cos() * 0.01)
            .collect::<Vec<_>>(),
    ];

    let kdtree = calculate_entropy_from_data_with_order_and_backend(
        data.clone(),
        24,
        3,
        JointNearestBackend::KdTree,
    )
    .expect("kdtree entropy failed");
    let dual_tree = calculate_entropy_from_data_with_order_and_backend(
        data,
        24,
        3,
        JointNearestBackend::DualTree,
    )
    .expect("dual-tree entropy failed");

    assert_approx_eq!(dual_tree, kdtree);
}

#[test]
fn test_third_order_entropy_for_three_coordinates_matches_joint_entropy() {
    let coord_1 = vec![0.1, 0.4, 0.8, 1.1, 1.7];
    let coord_2 = vec![1.0, 1.3, 1.9, 2.2, 2.8];
    let coord_3 = vec![2.0, 2.4, 2.7, 3.1, 3.5];
    let data = vec![coord_1.clone(), coord_2.clone(), coord_3.clone()];
    let n_frames = data[0].len();
    const EULER: f64 = 0.57721566490153;
    let three_d_constant = ((n_frames as f64) * 4.0 * std::f64::consts::PI / 3.0).ln() + EULER;

    let entropy =
        calculate_entropy_from_data_with_order(data, n_frames, 3).expect("order-3 entropy failed");
    let direct_joint_entropy = estimate_entropy_efficient(
        calc_three_d_nn(&coord_1, &coord_2, &coord_3).expect("3D nearest neighbor failed") * 3.0,
        (n_frames as f64).recip(),
        three_d_constant,
    );

    assert_approx_eq!(entropy, direct_joint_entropy);
}

#[test]
fn test_fourth_order_entropy_for_four_coordinates_matches_joint_entropy() {
    let coord_1 = vec![0.1, 0.4, 0.8, 1.1, 1.7];
    let coord_2 = vec![1.0, 1.3, 1.9, 2.2, 2.8];
    let coord_3 = vec![2.0, 2.4, 2.7, 3.1, 3.5];
    let coord_4 = vec![3.0, 3.6, 4.0, 4.5, 5.2];
    let data = vec![
        coord_1.clone(),
        coord_2.clone(),
        coord_3.clone(),
        coord_4.clone(),
    ];
    let n_frames = data[0].len();
    const EULER: f64 = 0.57721566490153;
    let four_d_constant = ((n_frames as f64) * std::f64::consts::PI.powi(2) / 2.0).ln() + EULER;

    let entropy =
        calculate_entropy_from_data_with_order(data, n_frames, 4).expect("order-4 entropy failed");
    let direct_joint_entropy = estimate_entropy_efficient(
        calc_four_d_nn(&coord_1, &coord_2, &coord_3, &coord_4).expect("4D nearest neighbor failed")
            * 4.0,
        (n_frames as f64).recip(),
        four_d_constant,
    );

    assert_approx_eq!(entropy, direct_joint_entropy);
}

#[test]
fn test_unsupported_mie_order_is_invalid() {
    let data = vec![vec![0.1, 0.4, 0.8], vec![1.0, 1.3, 1.9]];
    let err = calculate_entropy_from_data_with_order(data, 3, 5)
        .expect_err("expected unsupported MIE order error");

    assert!(err.contains("unsupported MIE order"));
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
    assert!(
        (a[0] - b[0]).abs() < epsilon,
        "x component mismatch: {} vs {}",
        a[0],
        b[0]
    );
    assert!(
        (a[1] - b[1]).abs() < epsilon,
        "y component mismatch: {} vs {}",
        a[1],
        b[1]
    );
    assert!(
        (a[2] - b[2]).abs() < epsilon,
        "z component mismatch: {} vs {}",
        a[2],
        b[2]
    );
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
        vec![6, 1, 0, 2],
    ];
    let trajectory: Vec<Vec<[f64; 3]>> = vec![vec![
        [71.543846, -91.90568, -87.45525],
        [70.995895, -92.99476, -86.545074],
        [70.73, -91.32219, -87.903145],
        [72.076584, -91.16638, -86.84448],
        [72.18585, -92.22224, -88.28653],
        [70.13876, -93.54273, -86.95551],
        [70.85134, -92.65517, -85.512024],
        [71.78272, -93.75866, -86.517555],
    ]];
    let expected_result: Vec<Vec<f64>> = vec![vec![
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
        1.55080378,
    ]];
    let result = calc_internal_coords(bat_list, trajectory);
    assert_eq!(result.len(), expected_result.len());
    assert_eq!(result[0].len(), expected_result[0].len());
    for idx in 0..expected_result[0].len() {
        assert_approx_eq!(result[0][idx], expected_result[0][idx], 1e-5)
    }
}
