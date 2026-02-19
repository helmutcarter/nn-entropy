use nn_entropy::*;
use assert_approx_eq::assert_approx_eq;
use rand_distr::{Normal, Distribution};

// const _test_data: [f64; 6] = [1.2, 6.2, 1.2, 1.0, 1.33, 6.9];


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
