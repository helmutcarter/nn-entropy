//! Basins of a periodic coordinate, found from its sampled distribution.
//!
//! [`detect_basins`] counts the wells of a torsion's distribution and assigns
//! every frame to one. The split estimator uses it to decide which torsions are
//! soft (more than one basin, or a free rotor) and to define conformational
//! states.
//!
//! The method is deterministic and periodic throughout:
//!
//! 1. Histogram the values over one period and smooth the histogram with a
//!    wrapped Gaussian kernel.
//! 2. Take every circular local maximum as a peak; a plateau counts once.
//! 3. Merge neighboring peaks whose separating minimum (the saddle) is at least
//!    `saddle_fraction` of the lower peak, highest saddle first: such peaks are
//!    noise on one well, not separate wells.
//! 4. Put the basin boundaries at the remaining saddles.
//! 5. Merge any basin holding less than `min_population` of the frames into
//!    the neighbor across its higher saddle.

use crate::CoordinateMetric;

/// Settings for [`detect_basins`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasinOptions {
    /// Histogram bins per period. Fewer are used when there are fewer than 20
    /// frames per bin, but never fewer than 12.
    pub bins: usize,
    /// Standard deviation of the smoothing kernel, in bins; 0 disables
    /// smoothing.
    pub smoothing: f64,
    /// Two peaks separated by a saddle at least this fraction of the lower
    /// peak belong to one basin.
    pub saddle_fraction: f64,
    /// Smallest fraction of the frames a basin may hold.
    pub min_population: f64,
    /// A single-basin coordinate whose mean resultant length is below this is
    /// a free rotor and counts as soft; `None` counts only multiple basins.
    pub free_rotor_resultant: Option<f64>,
}

impl Default for BasinOptions {
    /// 5-degree bins, 10-degree smoothing, saddles at half the lower peak, 2%
    /// minimum population, and free rotors below a resultant of 0.3.
    fn default() -> Self {
        Self {
            bins: 72,
            smoothing: 2.0,
            saddle_fraction: 0.5,
            min_population: 0.02,
            free_rotor_resultant: Some(0.3),
        }
    }
}

/// The basins of one coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct Basins {
    /// Number of basins, at least 1.
    pub count: usize,
    /// Basin of each frame, from 0 to `count - 1`, numbered in order of their
    /// lower boundary within [0, P).
    pub labels: Vec<u32>,
    /// Boundaries between basins in [0, P), ascending; empty for one basin.
    /// Basin i covers `[boundaries[i], boundaries[i + 1])`, and the last basin
    /// wraps around through P.
    pub boundaries: Vec<f64>,
    /// Fraction of the frames in each basin.
    pub populations: Vec<f64>,
    /// Mean resultant length |<e^(i 2 pi q / P)>|: near 1 for a narrow
    /// distribution, near 0 for a uniform one.
    pub resultant: f64,
}

impl Basins {
    /// Whether the coordinate is soft: more than one basin, or a free rotor.
    pub fn is_soft(&self, options: &BasinOptions) -> bool {
        self.count >= 2
            || options
                .free_rotor_resultant
                .is_some_and(|threshold| self.resultant < threshold)
    }
}

/// The basins of the periodic coordinate `values`.
pub fn detect_basins(
    values: &[f64],
    metric: CoordinateMetric,
    options: &BasinOptions,
) -> Result<Basins, String> {
    let CoordinateMetric::Periodic { period } = metric else {
        return Err("basins can only be detected for a periodic coordinate".to_string());
    };
    validate(values, period, options)?;
    let n = values.len();
    let canonical: Vec<f64> = values.iter().map(|v| v.rem_euclid(period)).collect();
    let resultant = {
        let to_radians = std::f64::consts::TAU / period;
        let (s, c) = canonical.iter().fold((0.0, 0.0), |(s, c), v| {
            let angle = v * to_radians;
            (s + angle.sin(), c + angle.cos())
        });
        (s * s + c * c).sqrt() / n as f64
    };

    let bins = options.bins.min((n / 20).max(12)).max(1);
    let width = period / bins as f64;
    let bin_of = |v: f64| ((v / width) as usize).min(bins - 1);
    let mut counts = vec![0.0; bins];
    for &v in &canonical {
        counts[bin_of(v)] += 1.0;
    }
    let smoothed = smooth(&counts, options.smoothing);

    let mut peaks = circular_maxima(&smoothed);
    let mut saddles = saddles_between(&smoothed, &peaks);
    merge_shallow_saddles(&smoothed, &mut peaks, &mut saddles, options.saddle_fraction);

    // Boundaries at the saddle bins' lower edges keep every frame of a bin in
    // one basin.
    let mut boundary_bins: Vec<usize> = if peaks.len() >= 2 {
        saddles
    } else {
        Vec::new()
    };
    boundary_bins.sort_unstable();
    loop {
        let populations = populations(&canonical, &boundary_bins, bin_of);
        let Some((smallest, &population)) = populations
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))
        else {
            break;
        };
        if boundary_bins.len() < 2 || population >= options.min_population {
            if boundary_bins.len() == 1 {
                // One boundary on a circle does not separate anything.
                boundary_bins.clear();
            }
            break;
        }
        // Basin `smallest` lies between boundaries `smallest` and
        // `smallest + 1`; drop the higher one, merging it with that neighbor.
        let lower = smallest;
        let upper = (smallest + 1) % boundary_bins.len();
        let remove = if smoothed[boundary_bins[upper]] > smoothed[boundary_bins[lower]] {
            upper
        } else {
            lower
        };
        boundary_bins.remove(remove);
    }

    let boundaries: Vec<f64> = boundary_bins.iter().map(|&b| b as f64 * width).collect();
    let labels: Vec<u32> = canonical
        .iter()
        .map(|&v| label(bin_of(v), &boundary_bins))
        .collect();
    let populations = populations(&canonical, &boundary_bins, bin_of);
    Ok(Basins {
        count: populations.len(),
        labels,
        boundaries,
        populations,
        resultant,
    })
}

fn validate(values: &[f64], period: f64, options: &BasinOptions) -> Result<(), String> {
    if values.is_empty() {
        return Err("no values to find basins in".to_string());
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("values contain non-finite entries".to_string());
    }
    if !period.is_finite() || period <= 0.0 {
        return Err("period must be finite and positive".to_string());
    }
    if options.bins < 3 {
        return Err("basin detection needs at least 3 bins".to_string());
    }
    if options.smoothing.is_nan() || options.smoothing < 0.0 {
        return Err("smoothing must be at least 0".to_string());
    }
    if !(options.saddle_fraction > 0.0 && options.saddle_fraction <= 1.0) {
        return Err("saddle_fraction must be in (0, 1]".to_string());
    }
    if !(0.0..1.0).contains(&options.min_population) {
        return Err("min_population must be in [0, 1)".to_string());
    }
    Ok(())
}

/// Circular convolution with a Gaussian of standard deviation `sigma` bins.
fn smooth(counts: &[f64], sigma: f64) -> Vec<f64> {
    let n = counts.len();
    if sigma == 0.0 {
        return counts.to_vec();
    }
    let radius = ((4.0 * sigma).ceil() as usize).min(n / 2);
    let kernel: Vec<f64> = (0..=radius)
        .map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp())
        .collect();
    (0..n)
        .map(|i| {
            let mut total = kernel[0] * counts[i];
            for (k, weight) in kernel.iter().enumerate().skip(1) {
                total += weight * (counts[(i + k) % n] + counts[(i + n - k % n) % n]);
            }
            total
        })
        .collect()
}

/// Bins of the circular local maxima; a plateau counts once, at its first bin.
/// A flat histogram has a single maximum at bin 0.
fn circular_maxima(h: &[f64]) -> Vec<usize> {
    let n = h.len();
    // Runs of equal height, starting at a bin whose predecessor differs, so a
    // run never straddles the start.
    let Some(start) = (0..n).find(|&i| h[i] != h[(i + n - 1) % n]) else {
        return vec![0];
    };
    let mut runs: Vec<(usize, f64)> = Vec::new();
    for offset in 0..n {
        let i = (start + offset) % n;
        if runs.last().is_none_or(|&(_, height)| height != h[i]) {
            runs.push((i, h[i]));
        }
    }
    let r = runs.len();
    let mut maxima: Vec<usize> = (0..r)
        .filter(|&k| runs[k].1 > runs[(k + r - 1) % r].1 && runs[k].1 > runs[(k + 1) % r].1)
        .map(|k| runs[k].0)
        .collect();
    maxima.sort_unstable();
    maxima
}

/// For consecutive peaks (circularly), the bin of the lowest point between
/// them; the first such bin on a tie. `saddles[i]` lies between `peaks[i]` and
/// `peaks[i + 1]`.
fn saddles_between(h: &[f64], peaks: &[usize]) -> Vec<usize> {
    let n = h.len();
    let k = peaks.len();
    if k < 2 {
        return Vec::new();
    }
    (0..k)
        .map(|i| {
            let (from, to) = (peaks[i], peaks[(i + 1) % k]);
            let span = (to + n - from) % n;
            (1..span)
                .map(|offset| (from + offset) % n)
                .min_by(|&a, &b| h[a].total_cmp(&h[b]))
                .unwrap_or(from)
        })
        .collect()
}

/// Merges neighboring peaks whose saddle is at least `fraction` of the lower
/// peak, highest saddle-to-peak ratio first.
fn merge_shallow_saddles(
    h: &[f64],
    peaks: &mut Vec<usize>,
    saddles: &mut Vec<usize>,
    fraction: f64,
) {
    while peaks.len() >= 2 {
        let k = peaks.len();
        let ratio = |i: usize| h[saddles[i]] / h[peaks[i]].min(h[peaks[(i + 1) % k]]);
        let Some(i) = (0..k)
            .filter(|&i| ratio(i) >= fraction)
            .max_by(|&a, &b| ratio(a).total_cmp(&ratio(b)).then(b.cmp(&a)))
        else {
            break;
        };
        let (a, b) = (i, (i + 1) % k);
        let lower = if h[peaks[b]] > h[peaks[a]] { a } else { b };
        peaks.remove(lower);
        // The merged peak's saddles are the lowest points of the joined arcs.
        *saddles = saddles_between(h, peaks);
    }
}

/// Basin of a value in bin `bin`, given sorted boundary bins: basin i covers
/// bins [boundaries[i], boundaries[i + 1]), and the last basin wraps.
fn label(bin: usize, boundaries: &[usize]) -> u32 {
    if boundaries.len() < 2 {
        return 0;
    }
    match boundaries.iter().rposition(|&b| b <= bin) {
        Some(i) => i as u32,
        None => (boundaries.len() - 1) as u32,
    }
}

fn populations(canonical: &[f64], boundaries: &[usize], bin_of: impl Fn(f64) -> usize) -> Vec<f64> {
    let mut totals = vec![
        0.0;
        if boundaries.len() < 2 {
            1
        } else {
            boundaries.len()
        }
    ];
    for &v in canonical {
        totals[label(bin_of(v), boundaries) as usize] += 1.0;
    }
    let n = canonical.len() as f64;
    totals.iter().map(|t| t / n).collect()
}
