//! The split estimator: soft coordinates non-parametrically, hard ones with a
//! Gaussian.
//!
//! Multi-basin torsions are poorly described by a Gaussian, while bonds,
//! angles and phase angles are close to harmonic, and a Gaussian captures
//! their correlations to every order. The split estimator divides the
//! coordinates into a soft set S and a hard set H and writes
//!
//! S = S_soft + S_hard + coupling,
//!
//! - S_soft: the nearest-neighbor MIE or a joint nearest-neighbor estimate
//!   over the soft coordinates;
//! - S_hard: the quasi-harmonic (QH) or Boltzmann quasi-harmonic (BQH)
//!   entropy of the hard coordinates over all frames;
//! - the coupling term, at most 0, for the dependence between the two sets:
//!   either the Gaussian mutual information between them, or the change in
//!   the hard entropy when it is computed within each conformational state of
//!   the soft torsions and averaged over the states.
//!
//! The soft set is given explicitly, or detected from the data with
//! [`detect_basins`]. For coordinates read from a topology,
//! [`InternalCoordinates::soft_torsions`](crate::bat_library::InternalCoordinates::soft_torsions)
//! gives the topological default: the primary torsion about each rotatable
//! bond.

use std::collections::{BTreeMap, HashMap};

use crate::basins::{BasinOptions, Basins, detect_basins};
use crate::quasi_harmonic::{
    MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions, Representation,
    correlation_log_det, quasi_harmonic_terms,
};
use crate::{
    CoordinateMetric, EntropyOptions, calculate_entropy, estimate_joint_entropy, resolve_input,
};

/// Which coordinates are soft.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoftSelection<'a> {
    /// Exactly these coordinates, by index.
    Indices(&'a [usize]),
    /// The periodic coordinates among `candidates` (all periodic coordinates
    /// when `None`) that [`Basins::is_soft`] calls soft: more than one basin,
    /// or a free rotor.
    Detect {
        /// Coordinates to consider; every one must be periodic.
        candidates: Option<&'a [usize]>,
    },
}

/// How the soft entropy is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftEstimator {
    /// The nearest-neighbor mutual information expansion at `order`, 1 to 4.
    NearestNeighborMie {
        /// Expansion order.
        order: usize,
    },
    /// One joint nearest-neighbor estimate over every soft coordinate, up to
    /// [`MAX_JOINT_DIMENSION`](crate::MAX_JOINT_DIMENSION) of them.
    JointNearestNeighbor,
}

impl Default for SoftEstimator {
    /// The MIE at order 2.
    fn default() -> Self {
        Self::NearestNeighborMie { order: 2 }
    }
}

/// How the hard entropy is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HardEstimator {
    /// A Gaussian fit to the hard coordinates (QH).
    #[default]
    QuasiHarmonic,
    /// Each hard coordinate's own entropy from its distribution, plus the
    /// Gaussian correlation term (BQH).
    BoltzmannQuasiHarmonic,
}

/// How soft torsions enter the Gaussian coupling term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoftFeatures {
    /// As e^(i phi): one complex entry each.
    #[default]
    Complex,
    /// As cos(phi) and sin(phi): two real entries each.
    CosSin,
}

/// Which conformational states the per-state coupling conditions on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateDefinition<'a> {
    /// The joint basin of the state-defining torsions: those in `defining`
    /// (the soft coordinates when `None`) not in `exclude`, keeping only those
    /// with at least two basins. Exclude symmetric rotors such as methyl
    /// groups, whose basins are equivalent.
    Detected {
        /// Candidate state-defining coordinates.
        defining: Option<&'a [usize]>,
        /// Coordinates that never define states.
        exclude: &'a [usize],
    },
    /// A state label for every frame, given by the caller.
    Labels(&'a [u32]),
}

/// Settings for the per-state coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateOptions<'a> {
    /// Which states to condition on.
    pub definition: StateDefinition<'a>,
    /// States with fewer frames than this, or than the number of hard
    /// coordinates plus 2, are dropped and the remaining populations
    /// renormalized.
    pub min_frames: usize,
    /// Fail when more states than this have at least `min_frames` frames.
    pub max_states: usize,
}

impl Default for StateOptions<'_> {
    /// Detected states from all soft coordinates, at least 300 frames per
    /// state, at most 256 states.
    fn default() -> Self {
        Self {
            definition: StateDefinition::Detected {
                defining: None,
                exclude: &[],
            },
            min_frames: 300,
            max_states: 256,
        }
    }
}

/// How the dependence between the soft and hard sets is accounted for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Coupling<'a> {
    /// Not at all: the soft and hard sets are treated as independent.
    None,
    /// The hard entropy within each conformational state, averaged over the
    /// states, in place of the hard entropy over all frames.
    PerState(StateOptions<'a>),
    /// The Gaussian mutual information between the two sets.
    Gaussian {
        /// How soft torsions enter the correlation matrix.
        features: SoftFeatures,
    },
}

/// Settings for [`estimate_split_entropy`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitOptions<'a> {
    /// Which coordinates are soft.
    pub soft: SoftSelection<'a>,
    /// How the soft entropy is estimated.
    pub soft_estimator: SoftEstimator,
    /// How the hard entropy is estimated.
    pub hard_estimator: HardEstimator,
    /// Quasi-harmonic settings for the hard set: BQH marginals, and how hard
    /// periodic coordinates (phase angles) enter the correlation matrix.
    pub hard: QuasiHarmonicOptions,
    /// How the dependence between the sets is accounted for.
    pub coupling: Coupling<'a>,
    /// Settings for basin detection, used by [`SoftSelection::Detect`] and
    /// [`StateDefinition::Detected`].
    pub basins: BasinOptions,
}

impl<'a> SplitOptions<'a> {
    /// The default settings for the soft set `soft`: NN MIE-2 for the soft
    /// coordinates; QH for the hard ones, with hard periodic coordinates
    /// centered on their circular mean; and Gaussian coupling with soft
    /// torsions as e^(i phi). These scored best against the butane TI/MBAR
    /// references; see the validation page of the user guide.
    pub fn new(soft: SoftSelection<'a>) -> Self {
        Self {
            soft,
            soft_estimator: SoftEstimator::default(),
            hard_estimator: HardEstimator::default(),
            hard: QuasiHarmonicOptions {
                periodic: PeriodicCorrelation::Centered,
                ..Default::default()
            },
            coupling: Coupling::Gaussian {
                features: SoftFeatures::Complex,
            },
            basins: BasinOptions::default(),
        }
    }
}

/// One conformational state of the per-state coupling.
#[derive(Debug, Clone, PartialEq)]
pub struct StateEntropy {
    /// Basin of each state-defining coordinate, or the caller's label.
    pub basins: Vec<u32>,
    /// Frames in the state.
    pub frames: usize,
    /// Fraction of all frames in the state.
    pub population: f64,
    /// Hard entropy within the state; `None` when the state was dropped.
    pub hard_entropy: Option<f64>,
}

/// The states of the per-state coupling.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSummary {
    /// The coordinates whose basins define the states; empty for labels given
    /// by the caller.
    pub defining: Vec<usize>,
    /// Every observed state, in order of its basins.
    pub states: Vec<StateEntropy>,
    /// Frames in dropped states.
    pub dropped_frames: usize,
    /// Fraction of the frames in dropped states.
    pub dropped_population: f64,
}

/// Split entropy and the terms it is built from, in units of k_B.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitEntropy {
    /// `soft + hard + coupling`.
    pub total: f64,
    /// Entropy of the soft coordinates.
    pub soft: f64,
    /// Entropy of the hard coordinates over all frames.
    pub hard: f64,
    /// Dependence between the sets, at most 0 up to sampling error.
    pub coupling: f64,
    /// Indices of the soft coordinates.
    pub soft_coordinates: Vec<usize>,
    /// Indices of the hard coordinates.
    pub hard_coordinates: Vec<usize>,
    /// Number of basins of each coordinate whose basins were detected.
    pub basin_counts: Vec<Option<usize>>,
    /// The states, for the per-state coupling.
    pub states: Option<StateSummary>,
}

/// Split entropy of the coordinates in `one_d_data`. `options` applies to every
/// part: the metrics mark the periodic coordinates, and the finite-sample
/// constant and Theiler window apply to the nearest-neighbor estimates.
pub fn estimate_split_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    split: &SplitOptions,
    options: &EntropyOptions,
) -> Result<SplitEntropy, String> {
    let (data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let dimensions = data.len();
    let mut basins: HashMap<usize, Basins> = HashMap::new();
    let mut detect = |index: usize| -> Result<Basins, String> {
        if let Some(found) = basins.get(&index) {
            return Ok(found.clone());
        }
        if !matches!(metrics[index], CoordinateMetric::Periodic { .. }) {
            return Err(format!(
                "coordinate {index} is not periodic, so it has no basins to detect"
            ));
        }
        let found = detect_basins(&data[index], metrics[index], &split.basins)
            .map_err(|err| format!("coordinate {index}: {err}"))?;
        basins.insert(index, found.clone());
        Ok(found)
    };

    let soft_coordinates = match split.soft {
        SoftSelection::Indices(indices) => checked_indices(indices, dimensions, "soft")?,
        SoftSelection::Detect { candidates } => {
            let candidates = match candidates {
                Some(candidates) => checked_indices(candidates, dimensions, "candidate")?,
                None => (0..dimensions)
                    .filter(|&i| matches!(metrics[i], CoordinateMetric::Periodic { .. }))
                    .collect(),
            };
            let mut soft = Vec::new();
            for index in candidates {
                if detect(index)?.is_soft(&split.basins) {
                    soft.push(index);
                }
            }
            soft
        }
    };
    let hard_coordinates: Vec<usize> = (0..dimensions)
        .filter(|i| !soft_coordinates.contains(i))
        .collect();

    let subset_options = |indices: &[usize]| -> Vec<CoordinateMetric> {
        indices.iter().map(|&i| metrics[i]).collect()
    };
    let columns =
        |indices: &[usize]| -> Vec<Vec<f64>> { indices.iter().map(|&i| data[i].clone()).collect() };

    // Soft entropy.
    let soft = if soft_coordinates.is_empty() {
        0.0
    } else {
        let soft_metrics = subset_options(&soft_coordinates);
        let soft_data = columns(&soft_coordinates);
        match split.soft_estimator {
            SoftEstimator::NearestNeighborMie { order } => calculate_entropy(
                &soft_data,
                frames_end,
                order,
                &with_metrics(options, &soft_metrics),
            ),
            SoftEstimator::JointNearestNeighbor => estimate_joint_entropy(
                &soft_data,
                frames_end,
                &with_metrics(options, &soft_metrics),
            ),
        }
        .map_err(|err| format!("soft set: {err}"))?
    };

    // Hard entropy over all frames.
    let hard_metrics = subset_options(&hard_coordinates);
    let hard_data = columns(&hard_coordinates);
    let hard_entropy = |data: &[Vec<f64>], frames: usize| -> Result<f64, String> {
        let boltzmann = split.hard_estimator == HardEstimator::BoltzmannQuasiHarmonic;
        let terms = quasi_harmonic_terms(
            data,
            frames,
            &split.hard,
            &with_metrics(options, &hard_metrics),
            boltzmann,
        )?;
        Ok(if boltzmann {
            terms.boltzmann_quasi_harmonic
        } else {
            terms.quasi_harmonic
        })
    };
    let hard = if hard_coordinates.is_empty() {
        0.0
    } else {
        hard_entropy(&hard_data, frames_end).map_err(|err| format!("hard set: {err}"))?
    };

    // Coupling.
    let mut states = None;
    let coupling = if soft_coordinates.is_empty() || hard_coordinates.is_empty() {
        0.0
    } else {
        match split.coupling {
            Coupling::None => 0.0,
            Coupling::Gaussian { features } => {
                let soft_representation = match features {
                    SoftFeatures::Complex => Representation::Complex,
                    SoftFeatures::CosSin => Representation::CosSin,
                };
                let representation = |i: usize| {
                    if soft_coordinates.contains(&i) {
                        soft_representation
                    } else {
                        Representation::of(metrics[i], split.hard.periodic)
                    }
                };
                let log_det = |indices: &[usize]| -> Result<f64, String> {
                    let columns: Vec<&[f64]> =
                        indices.iter().map(|&i| data[i].as_slice()).collect();
                    let representations: Vec<Representation> =
                        indices.iter().map(|&i| representation(i)).collect();
                    correlation_log_det(&columns, &subset_options(indices), &representations)
                };
                let all: Vec<usize> = (0..dimensions).collect();
                0.5 * (log_det(&all).map_err(|err| format!("soft and hard sets: {err}"))?
                    - log_det(&soft_coordinates).map_err(|err| format!("soft set: {err}"))?
                    - log_det(&hard_coordinates).map_err(|err| format!("hard set: {err}"))?)
            }
            Coupling::PerState(state_options) => {
                if options.theiler_window > 0
                    && split.hard_estimator == HardEstimator::BoltzmannQuasiHarmonic
                    && split.hard.marginals == MarginalEstimator::NearestNeighbor
                {
                    return Err("per-state coupling cannot apply a Theiler window to the hard set's nearest-neighbor marginals: a state's frames are not consecutive".to_string());
                }
                let (defining, labels) = match state_options.definition {
                    StateDefinition::Labels(labels) => {
                        if labels.len() < frames_end {
                            return Err(format!(
                                "{} state labels for {frames_end} frames",
                                labels.len()
                            ));
                        }
                        (
                            Vec::new(),
                            labels[..frames_end].iter().map(|&l| vec![l]).collect(),
                        )
                    }
                    StateDefinition::Detected { defining, exclude } => {
                        let candidates = match defining {
                            Some(defining) => {
                                checked_indices(defining, dimensions, "state-defining")?
                            }
                            None => soft_coordinates.clone(),
                        };
                        let mut defining = Vec::new();
                        for index in candidates.into_iter().filter(|i| !exclude.contains(i)) {
                            if detect(index)?.count >= 2 {
                                defining.push(index);
                            }
                        }
                        let labels: Vec<Vec<u32>> = (0..frames_end)
                            .map(|frame| defining.iter().map(|i| basins[i].labels[frame]).collect())
                            .collect();
                        (defining, labels)
                    }
                };
                let summary = per_state(
                    &hard_data,
                    &labels,
                    hard,
                    &state_options,
                    hard_coordinates.len(),
                    &hard_entropy,
                    defining,
                )?;
                let conditional = summary.0;
                states = Some(summary.1);
                conditional - hard
            }
        }
    };

    let basin_counts = (0..dimensions)
        .map(|i| basins.get(&i).map(|b| b.count))
        .collect();
    Ok(SplitEntropy {
        total: soft + hard + coupling,
        soft,
        hard,
        coupling,
        soft_coordinates,
        hard_coordinates,
        basin_counts,
        states,
    })
}

/// The hard entropy of some frames' hard columns: (columns, frame count).
type HardEntropy<'a> = dyn Fn(&[Vec<f64>], usize) -> Result<f64, String> + 'a;

/// The population-weighted hard entropy over the states, and the states.
fn per_state(
    hard_data: &[Vec<f64>],
    labels: &[Vec<u32>],
    hard_over_all_frames: f64,
    options: &StateOptions,
    hard_dimensions: usize,
    hard_entropy: &HardEntropy,
    defining: Vec<usize>,
) -> Result<(f64, StateSummary), String> {
    let frames_end = labels.len();
    let mut groups: BTreeMap<&[u32], Vec<usize>> = BTreeMap::new();
    for (frame, label) in labels.iter().enumerate() {
        groups.entry(label.as_slice()).or_default().push(frame);
    }
    if groups.len() == 1 {
        // One state: conditioning changes nothing.
        let state = StateEntropy {
            basins: groups.keys().next().map(|k| k.to_vec()).unwrap_or_default(),
            frames: frames_end,
            population: 1.0,
            hard_entropy: Some(hard_over_all_frames),
        };
        return Ok((
            hard_over_all_frames,
            StateSummary {
                defining,
                states: vec![state],
                dropped_frames: 0,
                dropped_population: 0.0,
            },
        ));
    }
    let large = groups
        .values()
        .filter(|frames| frames.len() >= options.min_frames)
        .count();
    if large > options.max_states {
        return Err(format!(
            "{large} conformational states have at least {} frames, more than the maximum of {}",
            options.min_frames, options.max_states
        ));
    }

    let threshold = options.min_frames.max(hard_dimensions + 2);
    let mut states = Vec::with_capacity(groups.len());
    let (mut weighted, mut used) = (0.0, 0.0);
    for (basins, frames) in groups {
        let population = frames.len() as f64 / frames_end as f64;
        let entropy = if frames.len() < threshold {
            None
        } else {
            let gathered: Vec<Vec<f64>> = hard_data
                .iter()
                .map(|column| frames.iter().map(|&f| column[f]).collect())
                .collect();
            hard_entropy(&gathered, frames.len())
                .ok()
                .filter(|e| e.is_finite())
        };
        if let Some(entropy) = entropy {
            weighted += population * entropy;
            used += population;
        }
        states.push(StateEntropy {
            basins: basins.to_vec(),
            frames: frames.len(),
            population,
            hard_entropy: entropy,
        });
    }
    if used == 0.0 {
        return Err(format!(
            "no conformational state has enough frames (at least {threshold}) for its hard entropy"
        ));
    }
    let dropped_frames = states
        .iter()
        .filter(|s| s.hard_entropy.is_none())
        .map(|s| s.frames)
        .sum();
    Ok((
        weighted / used,
        StateSummary {
            defining,
            states,
            dropped_frames,
            dropped_population: 1.0 - used,
        },
    ))
}

/// `options` with the metrics of a subset of the coordinates.
fn with_metrics<'a>(
    options: &EntropyOptions,
    metrics: &'a [CoordinateMetric],
) -> EntropyOptions<'a> {
    EntropyOptions {
        metrics: Some(metrics),
        constant: options.constant,
        theiler_window: options.theiler_window,
    }
}

/// `indices` checked to be in range and distinct, in ascending order.
fn checked_indices(indices: &[usize], dimensions: usize, what: &str) -> Result<Vec<usize>, String> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    if let Some(&bad) = sorted.iter().find(|&&i| i >= dimensions) {
        return Err(format!(
            "{what} coordinate {bad} is out of range for {dimensions} coordinates"
        ));
    }
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(format!("{what} coordinates are repeated"));
    }
    Ok(sorted)
}
