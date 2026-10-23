//! Basin detection on periodic coordinates.
//!
//! Each test draws frames from known wells, so the expected count and the
//! well each frame came from are known exactly.

mod common;

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use common::Rng;
use nn_entropy::CoordinateMetric;
use nn_entropy::basins::{BasinOptions, Basins, detect_basins};

const TORSION: CoordinateMetric = CoordinateMetric::Periodic { period: TAU };

/// Frames drawn from wrapped normal wells (center, standard deviation,
/// weight), with the index of the well each came from.
fn wells(rng: &mut Rng, n: usize, wells: &[(f64, f64, f64)]) -> (Vec<f64>, Vec<usize>) {
    let total: f64 = wells.iter().map(|w| w.2).sum();
    (0..n)
        .map(|_| {
            let mut pick = rng.uniform() * total;
            let index = wells
                .iter()
                .position(|w| {
                    pick -= w.2;
                    pick < 0.0
                })
                .unwrap_or(wells.len() - 1);
            let (center, sd, _) = wells[index];
            ((center + sd * rng.normal()).rem_euclid(TAU), index)
        })
        .unzip()
}

fn detect(values: &[f64]) -> Basins {
    detect_basins(values, TORSION, &BasinOptions::default()).unwrap()
}

/// Fraction of frames whose basin is the one most of their well's frames
/// share, with distinct wells mapped to distinct basins.
fn agreement(basins: &Basins, truth: &[usize]) -> f64 {
    let mut votes: HashMap<(usize, u32), usize> = HashMap::new();
    for (&well, &label) in truth.iter().zip(&basins.labels) {
        *votes.entry((well, label)).or_default() += 1;
    }
    let wells = truth.iter().max().unwrap() + 1;
    let mapping: Vec<u32> = (0..wells)
        .map(|well| {
            (0..basins.count as u32)
                .max_by_key(|&label| votes.get(&(well, label)).copied().unwrap_or(0))
                .unwrap()
        })
        .collect();
    let mut distinct = mapping.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        wells,
        "two wells share a basin: {mapping:?}"
    );
    let correct = truth
        .iter()
        .zip(&basins.labels)
        .filter(|&(&well, &label)| mapping[well] == label)
        .count();
    correct as f64 / truth.len() as f64
}

#[test]
fn one_well_is_one_hard_basin() {
    let mut rng = Rng::new(61);
    let (values, _) = wells(&mut rng, 20_000, &[(1.0, 0.3, 1.0)]);
    let basins = detect(&values);
    assert_eq!(basins.count, 1);
    assert!(basins.boundaries.is_empty());
    assert_eq!(basins.populations, vec![1.0]);
    assert!(basins.labels.iter().all(|&l| l == 0));
    assert!(basins.resultant > 0.9, "{}", basins.resultant);
    assert!(!basins.is_soft(&BasinOptions::default()));
}

#[test]
fn two_and_three_wells_are_separated() {
    let mut rng = Rng::new(62);
    let (values, truth) = wells(&mut rng, 20_000, &[(-1.5, 0.25, 1.0), (1.5, 0.25, 1.0)]);
    let basins = detect(&values);
    assert_eq!(basins.count, 2);
    assert!(agreement(&basins, &truth) > 0.99);
    assert!(basins.is_soft(&BasinOptions::default()));

    // Butane-like: trans at pi, gauche at +-pi/3, unequal populations.
    let (values, truth) = wells(
        &mut rng,
        30_000,
        &[(PI, 0.3, 0.6), (PI / 3.0, 0.3, 0.2), (-PI / 3.0, 0.3, 0.2)],
    );
    let basins = detect(&values);
    assert_eq!(basins.count, 3);
    assert!(agreement(&basins, &truth) > 0.99);
    let total: f64 = basins.populations.iter().sum();
    assert!((total - 1.0).abs() < 1e-12);
    assert!(basins.labels.iter().all(|&l| (l as usize) < basins.count));
}

#[test]
fn a_well_across_the_wrap_point_stays_whole() {
    let mut rng = Rng::new(63);
    let (values, truth) = wells(&mut rng, 20_000, &[(0.0, 0.3, 1.0), (PI, 0.3, 1.0)]);
    let basins = detect(&values);
    assert_eq!(basins.count, 2);
    assert!(agreement(&basins, &truth) > 0.99);

    // Rotating the data moves the boundaries but not the partition.
    for shift in [0.4, 1.7, 3.0] {
        let rotated: Vec<f64> = values.iter().map(|v| (v + shift).rem_euclid(TAU)).collect();
        let moved = detect(&rotated);
        assert_eq!(moved.count, 2, "shift {shift}");
        assert!(agreement(&moved, &truth) > 0.99, "shift {shift}");
    }
}

#[test]
fn a_uniform_distribution_is_one_soft_basin() {
    let mut rng = Rng::new(64);
    let values = rng.uniform_vec(20_000, 0.0, TAU);
    let basins = detect(&values);
    assert_eq!(basins.count, 1);
    assert!(basins.resultant < 0.05, "{}", basins.resultant);
    assert!(basins.is_soft(&BasinOptions::default()));
    let without_rotor = BasinOptions {
        free_rotor_resultant: None,
        ..Default::default()
    };
    assert!(!basins.is_soft(&without_rotor));
}

#[test]
fn a_rare_well_is_merged() {
    let mut rng = Rng::new(65);
    let (values, _) = wells(&mut rng, 20_000, &[(0.0, 0.3, 0.99), (PI, 0.2, 0.01)]);
    assert_eq!(detect(&values).count, 1);
    let keep = BasinOptions {
        min_population: 0.005,
        ..Default::default()
    };
    assert_eq!(detect_basins(&values, TORSION, &keep).unwrap().count, 2);
}

#[test]
fn a_small_noisy_sample_of_one_well_is_not_split() {
    let mut rng = Rng::new(66);
    let (values, _) = wells(&mut rng, 400, &[(2.0, 0.6, 1.0)]);
    assert_eq!(detect(&values).count, 1);
}

#[test]
fn detection_is_deterministic() {
    let mut rng = Rng::new(67);
    let (values, _) = wells(&mut rng, 10_000, &[(-2.0, 0.3, 1.0), (0.5, 0.4, 2.0)]);
    assert_eq!(detect(&values), detect(&values));
}

#[test]
fn invalid_input_is_refused() {
    let linear = detect_basins(
        &[0.0, 1.0],
        CoordinateMetric::Linear,
        &BasinOptions::default(),
    );
    assert!(linear.unwrap_err().contains("periodic"));
    assert!(detect_basins(&[], TORSION, &BasinOptions::default()).is_err());
    assert!(detect_basins(&[f64::NAN], TORSION, &BasinOptions::default()).is_err());
    for options in [
        BasinOptions {
            bins: 2,
            ..Default::default()
        },
        BasinOptions {
            saddle_fraction: 0.0,
            ..Default::default()
        },
        BasinOptions {
            min_population: 1.0,
            ..Default::default()
        },
        BasinOptions {
            smoothing: -1.0,
            ..Default::default()
        },
    ] {
        assert!(
            detect_basins(&[0.0, 1.0], TORSION, &options).is_err(),
            "{options:?}"
        );
    }
}
