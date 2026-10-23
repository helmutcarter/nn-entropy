//! The split (soft/hard) estimator.
//!
//! Expected values come from the estimators the split one is assembled from
//! (exact identities), from closed forms, or from a synthetic system whose
//! entropy is known: a torsion with well-separated wells, and a hard
//! coordinate whose Gaussian spread depends on the well.

mod common;

use std::f64::consts::{E, PI, TAU};

use common::Rng;
use nn_entropy::quasi_harmonic::{
    PeriodicCorrelation, QuasiHarmonicOptions, estimate_quasi_harmonic_entropy,
};
use nn_entropy::split::{
    Coupling, HardEstimator, SoftEstimator, SoftFeatures, SoftSelection, SplitOptions,
    StateDefinition, StateOptions, estimate_split_entropy,
};
use nn_entropy::{
    CoordinateMetric, EntropyOptions, MAX_JOINT_DIMENSION, calculate_entropy,
    estimate_joint_entropy,
};

const TORSION: CoordinateMetric = CoordinateMetric::Periodic { period: TAU };
const LINEAR: CoordinateMetric = CoordinateMetric::Linear;

/// Wells of the soft torsion: center, weight, and the standard deviation of
/// the hard coordinate in that well.
const WELLS: [(f64, f64, f64); 3] = [
    (0.0, 0.5, 0.1),
    (2.0 * PI / 3.0, 0.3, 0.2),
    (4.0 * PI / 3.0, 0.2, 0.4),
];
const WELL_WIDTH: f64 = 0.2;

/// A torsion drawn from `WELLS`, a hard coordinate whose spread depends on the
/// well, the well of each frame, and an independent narrow torsion.
fn well_system(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<u32>) {
    let (mut phi, mut x, mut wells, mut narrow) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let mut pick = rng.uniform();
        let well = WELLS
            .iter()
            .position(|w| {
                pick -= w.1;
                pick < 0.0
            })
            .unwrap_or(WELLS.len() - 1);
        let (center, _, sd) = WELLS[well];
        phi.push((center + WELL_WIDTH * rng.normal()).rem_euclid(TAU));
        x.push(sd * rng.normal());
        wells.push(well as u32);
        narrow.push((1.0 + 0.15 * rng.normal()).rem_euclid(TAU));
    }
    (vec![phi, x, narrow], wells)
}

/// The exact entropy of `well_system`'s torsion and hard coordinate: with
/// wells this far apart, H(phi) is the wells' own entropy plus the mixing
/// entropy, and x depends on phi only through the well.
fn well_system_entropy() -> f64 {
    WELLS
        .iter()
        .map(|&(_, p, sd)| {
            p * (0.5 * (TAU * E * WELL_WIDTH.powi(2)).ln() + 0.5 * (TAU * E * sd * sd).ln()
                - p.ln())
        })
        .sum()
}

fn options(metrics: &[CoordinateMetric]) -> EntropyOptions<'_> {
    EntropyOptions {
        metrics: Some(metrics),
        ..Default::default()
    }
}

fn split(
    data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    split: &SplitOptions,
) -> nn_entropy::split::SplitEntropy {
    estimate_split_entropy(data, data[0].len(), split, &options(metrics)).unwrap()
}

// ---------------------------------------------------------------------------
// Exact identities
// ---------------------------------------------------------------------------

#[test]
fn without_soft_coordinates_it_is_the_quasi_harmonic_entropy() {
    let mut rng = Rng::new(71);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    // The split estimator's default centers hard periodic coordinates.
    let centered = QuasiHarmonicOptions {
        periodic: PeriodicCorrelation::Centered,
        ..Default::default()
    };
    let qh = estimate_quasi_harmonic_entropy(&data, 3_000, &centered, &options(&metrics)).unwrap();
    for (hard_estimator, expected) in [
        (HardEstimator::QuasiHarmonic, qh.quasi_harmonic),
        (
            HardEstimator::BoltzmannQuasiHarmonic,
            qh.boltzmann_quasi_harmonic,
        ),
    ] {
        let mut settings = SplitOptions::new(SoftSelection::Indices(&[]));
        settings.hard_estimator = hard_estimator;
        let result = split(&data, &metrics, &settings);
        assert_eq!(result.total, expected, "{hard_estimator:?}");
        assert_eq!((result.soft, result.coupling), (0.0, 0.0));
        assert_eq!(result.hard_coordinates, vec![0, 1, 2]);
    }
}

#[test]
fn without_coupling_it_is_the_sum_of_its_parts() {
    let mut rng = Rng::new(72);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 2]));
    settings.coupling = Coupling::None;
    let result = split(&data, &metrics, &settings);

    let soft = calculate_entropy(
        &[data[0].clone(), data[2].clone()],
        3_000,
        2,
        &options(&[TORSION, TORSION]),
    )
    .unwrap();
    let hard = estimate_quasi_harmonic_entropy(
        &[data[1].clone()],
        3_000,
        &QuasiHarmonicOptions::default(),
        &options(&[LINEAR]),
    )
    .unwrap()
    .quasi_harmonic;
    assert_eq!(result.soft, soft);
    assert_eq!(result.hard, hard);
    assert_eq!(result.total, soft + hard);
    assert_eq!(result.soft_coordinates, vec![0, 2]);
    assert_eq!(result.hard_coordinates, vec![1]);
}

#[test]
fn without_hard_coordinates_it_is_the_soft_entropy() {
    let mut rng = Rng::new(73);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let result = split(
        &data,
        &metrics,
        &SplitOptions::new(SoftSelection::Indices(&[0, 1, 2])),
    );
    assert_eq!(
        result.total,
        calculate_entropy(&data, 3_000, 2, &options(&metrics)).unwrap()
    );
    assert_eq!((result.hard, result.coupling), (0.0, 0.0));

    let mut joint = SplitOptions::new(SoftSelection::Indices(&[0, 1, 2]));
    joint.soft_estimator = SoftEstimator::JointNearestNeighbor;
    assert_eq!(
        split(&data, &metrics, &joint).total,
        estimate_joint_entropy(&data, 3_000, &options(&metrics)).unwrap()
    );
}

#[test]
fn for_one_soft_coordinate_the_mie_and_joint_estimates_agree() {
    let mut rng = Rng::new(74);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let mut mie = SplitOptions::new(SoftSelection::Indices(&[0]));
    mie.coupling = Coupling::None;
    let mut joint = mie;
    joint.soft_estimator = SoftEstimator::JointNearestNeighbor;
    assert_eq!(
        split(&data, &metrics, &mie).soft,
        split(&data, &metrics, &joint).soft
    );
}

// ---------------------------------------------------------------------------
// Coupling
// ---------------------------------------------------------------------------

#[test]
fn gaussian_coupling_recovers_a_linear_dependence_on_cos_and_sin() {
    // x = a cos(phi) + noise with phi uniform: the Gaussian mutual information
    // between (cos phi, sin phi) and x is -1/2 ln(1 - rho^2), with
    // rho^2 = (a^2 / 2) / (a^2 / 2 + sigma^2).
    let mut rng = Rng::new(75);
    let n = 50_000;
    let (a, sigma) = (1.0, 0.5);
    let phi = rng.uniform_vec(n, 0.0, TAU);
    let x: Vec<f64> = phi
        .iter()
        .map(|p| a * p.cos() + sigma * rng.normal())
        .collect();
    let data = vec![phi, x];
    let metrics = [TORSION, LINEAR];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::CosSin,
    };
    let rho_squared = (a * a / 2.0) / (a * a / 2.0 + sigma * sigma);
    let coupling = split(&data, &metrics, &settings).coupling;
    assert!(
        (coupling - 0.5 * (1.0 - rho_squared).ln()).abs() < 0.01,
        "{coupling}"
    );

    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::Complex,
    };
    assert!(split(&data, &metrics, &settings).coupling < 0.0);
}

#[test]
fn independent_sets_have_almost_no_gaussian_coupling() {
    let mut rng = Rng::new(76);
    let n = 50_000;
    let data = vec![
        rng.uniform_vec(n, 0.0, TAU),
        rng.normal_vec(n, 0.0, 1.0),
        rng.normal_vec(n, 0.0, 1.0),
    ];
    let metrics = [TORSION, LINEAR, LINEAR];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::Complex,
    };
    let coupling = split(&data, &metrics, &settings).coupling;
    assert!(coupling <= 0.0 && coupling > -0.005, "{coupling}");
}

#[test]
fn per_state_coupling_recovers_a_well_dependent_spread() {
    let mut rng = Rng::new(77);
    let (data, _) = well_system(&mut rng, 60_000);
    let data = vec![data[0].clone(), data[1].clone()];
    let metrics = [TORSION, LINEAR];
    let truth = well_system_entropy();

    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::PerState(StateOptions::default());
    let per_state = split(&data, &metrics, &settings);
    assert!(
        (per_state.total - truth).abs() < 0.03,
        "{} vs {truth}",
        per_state.total
    );
    let states = per_state.states.unwrap();
    assert_eq!(states.defining, vec![0]);
    assert_eq!(states.states.len(), 3);
    assert_eq!(states.dropped_frames, 0);
    assert!(per_state.coupling < -0.1);

    // Without conditioning, the hard coordinate is a mixture of Gaussians of
    // different widths, so the entropy comes out too high.
    let mut independent = SplitOptions::new(SoftSelection::Indices(&[0]));
    independent.coupling = Coupling::None;
    assert!(split(&data, &metrics, &independent).total > truth + 0.1);
}

#[test]
fn a_sparse_state_is_dropped_and_the_rest_renormalized() {
    let mut rng = Rng::new(78);
    let (data, mut wells) = well_system(&mut rng, 20_000);
    let data = vec![data[0].clone(), data[1].clone()];
    // Make 50 frames a state of their own.
    for well in wells.iter_mut().take(50) {
        *well = 7;
    }
    let metrics = [TORSION, LINEAR];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Labels(&wells),
        ..Default::default()
    });
    let result = split(&data, &metrics, &settings);
    let states = result.states.unwrap();
    assert_eq!(states.dropped_frames, 50);
    assert!((states.dropped_population - 50.0 / 20_000.0).abs() < 1e-12);

    // The conditional entropy is the population-weighted mean of the kept
    // states' QH entropies, computed here from the frames directly.
    let (mut weighted, mut used) = (0.0, 0.0);
    for state in states.states.iter().filter(|s| s.hard_entropy.is_some()) {
        let frames: Vec<f64> = data[1]
            .iter()
            .zip(&wells)
            .filter(|&(_, &w)| w == state.basins[0])
            .map(|(&x, _)| x)
            .collect();
        let entropy = estimate_quasi_harmonic_entropy(
            std::slice::from_ref(&frames),
            frames.len(),
            &QuasiHarmonicOptions::default(),
            &options(&[LINEAR]),
        )
        .unwrap()
        .quasi_harmonic;
        assert_eq!(state.hard_entropy, Some(entropy));
        weighted += state.population * entropy;
        used += state.population;
    }
    assert!((result.coupling - (weighted / used - result.hard)).abs() < 1e-12);
}

#[test]
fn symmetric_or_single_basin_torsions_define_no_states() {
    let mut rng = Rng::new(79);
    let (data, _) = well_system(&mut rng, 20_000);
    let metrics = [TORSION, LINEAR, TORSION];
    // Excluding the multi-basin torsion leaves only the narrow one, which has
    // one basin: a single state, so no coupling.
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 2]));
    settings.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Detected {
            defining: None,
            exclude: &[0],
        },
        ..Default::default()
    });
    let result = split(&data, &metrics, &settings);
    assert_eq!(result.coupling, 0.0);
    let states = result.states.unwrap();
    assert!(states.defining.is_empty());
    assert_eq!(states.states.len(), 1);
}

// ---------------------------------------------------------------------------
// Detecting the soft set
// ---------------------------------------------------------------------------

#[test]
fn detection_picks_the_multi_basin_torsion() {
    let mut rng = Rng::new(80);
    let (data, _) = well_system(&mut rng, 20_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let result = split(
        &data,
        &metrics,
        &SplitOptions::new(SoftSelection::Detect { candidates: None }),
    );
    assert_eq!(result.soft_coordinates, vec![0]);
    assert_eq!(result.hard_coordinates, vec![1, 2]);
    assert_eq!(result.basin_counts, vec![Some(3), None, Some(1)]);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn invalid_settings_are_refused() {
    let mut rng = Rng::new(81);
    let (data, wells) = well_system(&mut rng, 2_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let error = |settings: &SplitOptions, theiler_window: usize| {
        let options = EntropyOptions {
            metrics: Some(&metrics),
            theiler_window,
            ..Default::default()
        };
        estimate_split_entropy(&data, 2_000, settings, &options).unwrap_err()
    };

    assert!(error(&SplitOptions::new(SoftSelection::Indices(&[3])), 0).contains("out of range"));
    assert!(error(&SplitOptions::new(SoftSelection::Indices(&[0, 0])), 0).contains("repeated"));
    let linear_candidate = SplitOptions::new(SoftSelection::Detect {
        candidates: Some(&[1]),
    });
    assert!(error(&linear_candidate, 0).contains("not periodic"));

    let mut short_labels = SplitOptions::new(SoftSelection::Indices(&[0]));
    short_labels.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Labels(&wells[..10]),
        ..Default::default()
    });
    assert!(error(&short_labels, 0).contains("state labels"));

    let mut windowed = SplitOptions::new(SoftSelection::Indices(&[0]));
    windowed.hard_estimator = HardEstimator::BoltzmannQuasiHarmonic;
    windowed.coupling = Coupling::PerState(StateOptions::default());
    assert!(error(&windowed, 1).contains("Theiler window"));

    let mut too_many = SplitOptions::new(SoftSelection::Indices(&[0]));
    too_many.coupling = Coupling::PerState(StateOptions {
        min_frames: 1,
        max_states: 2,
        ..Default::default()
    });
    assert!(error(&too_many, 0).contains("more than the maximum"));

    // The joint estimate has a dimension limit.
    let wide: Vec<Vec<f64>> = (0..=MAX_JOINT_DIMENSION)
        .map(|_| rng.uniform_vec(200, 0.0, TAU))
        .collect();
    let wide_metrics = vec![TORSION; wide.len()];
    let all: Vec<usize> = (0..wide.len()).collect();
    let mut joint = SplitOptions::new(SoftSelection::Indices(&all));
    joint.soft_estimator = SoftEstimator::JointNearestNeighbor;
    let err = estimate_split_entropy(&wide, 200, &joint, &options(&wide_metrics)).unwrap_err();
    assert!(
        err.contains("use the mutual information expansion"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

mod cli {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use super::common::Rng;
    use super::common::amber::{
        CHAIN_ATOMS, CHAIN_BONDS, NetcdfOptions, Topology, chain_frames, concat_frames,
        scratch_path, write_netcdf, write_prmtop,
    };
    use nn_entropy::EntropyOptions;
    use nn_entropy::bat_library::InternalCoordinates;
    use nn_entropy::split::{
        Coupling, SoftSelection, SplitOptions, StateDefinition, StateOptions,
        estimate_split_entropy,
    };

    /// A chain molecule (three rotatable bonds) with more frames than
    /// coordinates.
    fn chain(name: &str) -> (PathBuf, PathBuf) {
        let mut topology = Topology::default();
        topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
        let top = scratch_path(&format!("split_{name}.parm7"));
        let traj = scratch_path(&format!("split_{name}.nc"));
        write_prmtop(&top, &topology);
        let mut rng = Rng::new(0x5b17);
        let frames = chain_frames(&mut rng, 80, [0.0, 0.0, 0.0]);
        write_netcdf(&traj, &concat_frames(&[&frames]), NetcdfOptions::default());
        (top, traj)
    }

    fn run(top: &Path, traj: &Path, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_nn_entropy"))
            .args([top.to_str().unwrap(), traj.to_str().unwrap()])
            .args(args)
            .output()
            .expect("failed to run CLI")
    }

    fn lines(output: &Output) -> Vec<String> {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn value(lines: &[String], name: &str) -> String {
        lines
            .iter()
            .find_map(|line| line.strip_prefix(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("no {name} line in {lines:?}"))
            .to_string()
    }

    fn library(
        top: &Path,
        traj: &Path,
        configure: impl Fn(&mut SplitOptions, &InternalCoordinates),
    ) -> nn_entropy::split::SplitEntropy {
        let mut internal = InternalCoordinates::new(top).unwrap();
        internal
            .calculate_internal_coords(traj, usize::MAX, false)
            .unwrap();
        let data = internal.columns(0, 1).unwrap();
        let metrics = internal.coordinate_metrics();
        let soft = internal.soft_torsions();
        let mut settings = SplitOptions::new(SoftSelection::Indices(&soft));
        configure(&mut settings, &internal);
        let options = EntropyOptions {
            metrics: Some(&metrics),
            ..Default::default()
        };
        estimate_split_entropy(&data, data[0].len(), &settings, &options).unwrap()
    }

    #[test]
    fn default_split_output_matches_the_library() {
        let (top, traj) = chain("cli_default");
        let lines = lines(&run(&top, &traj, &["--estimator", "split"]));
        let expected = library(&top, &traj, |_, _| {});
        let numbers: Vec<String> = expected
            .soft_coordinates
            .iter()
            .map(|i| (i + 1).to_string())
            .collect();
        assert_eq!(value(&lines, "Soft coordinates"), numbers.join(","));
        assert_eq!(
            value(&lines, "Soft entropy").parse::<f64>().unwrap(),
            expected.soft
        );
        assert_eq!(
            value(&lines, "Hard entropy").parse::<f64>().unwrap(),
            expected.hard
        );
        assert_eq!(
            value(&lines, "Coupling term").parse::<f64>().unwrap(),
            expected.coupling
        );
        assert_eq!(
            lines.last().unwrap(),
            &format!("Total entropy = {}", expected.total)
        );
        assert_eq!(numbers.len(), 3);
    }

    #[test]
    fn per_state_output_lists_the_states() {
        let (top, traj) = chain("cli_states");
        let lines = lines(&run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--coupling",
                "states",
                "--min-state-frames",
                "10",
            ],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.coupling = Coupling::PerState(StateOptions {
                min_frames: 10,
                ..Default::default()
            });
        });
        let states = expected.states.unwrap();
        assert!(lines.contains(&"state\tbasins\tframes\tpopulation\thard_entropy".to_string()));
        assert_eq!(
            value(&lines, "Dropped frames"),
            states.dropped_frames.to_string()
        );
        assert_eq!(
            value(&lines, "Total entropy").parse::<f64>().unwrap(),
            expected.total
        );
    }

    #[test]
    fn soft_and_state_lists_count_from_one() {
        let (top, traj) = chain("cli_lists");
        let lines = lines(&run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--soft",
                "8,10",
                "--coupling",
                "states",
                "--state-torsions",
                "8",
                "--min-state-frames",
                "10",
            ],
        ));
        assert_eq!(value(&lines, "Soft coordinates"), "8,10");
        let expected = library(&top, &traj, |settings, _| {
            settings.soft = SoftSelection::Indices(&[7, 9]);
            settings.coupling = Coupling::PerState(StateOptions {
                definition: StateDefinition::Detected {
                    defining: Some(&[7]),
                    exclude: &[],
                },
                min_frames: 10,
                ..Default::default()
            });
        });
        assert_eq!(
            value(&lines, "Total entropy").parse::<f64>().unwrap(),
            expected.total
        );
        // Detection runs on every torsion.
        lines_ok(&run(
            &top,
            &traj,
            &["--estimator", "split", "--soft", "detect"],
        ));
    }

    fn lines_ok(output: &Output) {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn split_flags_are_checked() {
        let (top, traj) = chain("cli_rejected");
        for (args, message) in [
            (
                &["--coupling", "none"][..],
                "--coupling applies only to --estimator split",
            ),
            (
                &["--estimator", "split", "--soft", "99"][..],
                "--soft coordinate 99 is beyond",
            ),
            (
                &["--estimator", "split", "--soft", "0"][..],
                "invalid --soft value",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--coupling",
                    "states",
                    "--soft-features",
                    "cossin",
                ][..],
                "--soft-features applies only",
            ),
            (
                &["--estimator", "split", "--state-torsions", "auto"][..],
                "apply only to --coupling states",
            ),
            (
                &["--estimator", "split", "--per-coordinate"][..],
                "cannot be combined with --estimator split",
            ),
            (
                &["--estimator", "split", "--no-phase-angles"][..],
                "--soft topology needs phase angles",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--soft-estimator",
                    "joint",
                    "--mie-order",
                    "3",
                ][..],
                "--mie-order applies only",
            ),
            (
                &["--estimator", "split", "--histogram-bins", "20"][..],
                "--histogram-bins applies only",
            ),
            (
                &["--estimator", "split", "--coupling", "maybe"][..],
                "--coupling must be one of gaussian, states, none",
            ),
        ] {
            let output = run(&top, &traj, args);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(1), "{args:?}: {stderr}");
            assert!(stderr.contains(message), "{args:?}: {stderr}");
        }
    }
}
