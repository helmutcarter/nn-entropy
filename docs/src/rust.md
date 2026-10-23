# Rust

The crate is not on crates.io; depend on it through git:

```toml
[dependencies]
nn_entropy = { git = "https://github.com/helmutcarter/nn-entropy" }
```

Every public item has API documentation, with examples that are compiled and tested:

```bash
cargo doc --no-deps --open
```

## Overview

```rust
use nn_entropy::{CoordinateMetric, EntropyOptions, FiniteSampleConstant, calculate_entropy};

// one_d_data is Vec<Vec<f64>> with shape [n_coords][n_frames]
let entropy = calculate_entropy(&one_d_data, frames_end, 2, &EntropyOptions::default())?;

// Periodic torsions and the exact finite-sample constant
let options = EntropyOptions {
    metrics: Some(&metrics), // one CoordinateMetric per coordinate
    constant: FiniteSampleConstant::Exact,
    ..Default::default()
};
let entropy = calculate_entropy(&one_d_data, frames_end, 2, &options)?;
```

The NN MIE functions all take an `EntropyOptions`, described under [Nearest-neighbor MIE](nn-mie.md#options):
- `calculate_entropy(data, frames_end, mie_order, &options)` returns the total entropy at MIE order 1, 2, 3 or 4.
- `estimate_coordinate_entropy(data, frames_end, mie_order, &options)` returns each coordinate's share of the entropy at MIE order 1 or 2, so the values sum to `calculate_entropy` at the same order and with the same options. Order 1 gives the marginal entropies. Order 2 also splits each pairwise mutual information term evenly between its two coordinates. Orders 3 and 4 are rejected.
- `estimate_coordinate_mutual_information(data, frames_end, &options)` returns the mutual information of each coordinate pair.
- `calc_joint_nn([&x, &y, ...], metrics)` is the underlying nearest-neighbor sum, `sum over frames of ln(nearest-neighbor distance)`, in any number of dimensions; the MIE uses 1 to 4. `calc_joint_nn_with_window(coordinates, metrics, w)` is the same sum with a Theiler window.

To build coordinates from files, use `bat_library::InternalCoordinates`:

```rust
use nn_entropy::bat_library::{BondSet, CoordinateOptions, InternalCoordinates};

let options = CoordinateOptions { bonds: BondSet::Tree, ..Default::default() }; // phase angles by default
let mut internal = InternalCoordinates::with_options(&top, options)?; // ::new(&top) for the defaults
internal.calculate_internal_coords(&traj, usize::MAX, false)?;
let metrics = internal.coordinate_metrics();   // one per coordinate
let atoms = internal.coordinate_atoms();       // topology atom indices of each coordinate
let references = internal.phase_references();  // Some(i) for a phase angle of torsion i
let log_jacobian = internal.log_jacobian()?;   // per frame; tree bonds only
```

`internal.int_coords` holds the values with shape `[n_frames][n_coords]`. The estimators take `[n_coords][n_frames]`, which `internal.columns(start, stride)` returns for the frames `start`, `start + stride`, ..., the same selection as the CLI's `--start` and `--stride`. `internal.coordinate_kinds()` gives each coordinate's kind (`Bond`, `Angle`, `Torsion`, or `PhaseAngle { reference }`).

The quasi-harmonic estimators take the same data and options, plus their own settings:

```rust
use nn_entropy::quasi_harmonic::{
    MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions, estimate_quasi_harmonic_entropy,
};

let quasi_harmonic = QuasiHarmonicOptions::default(); // NN marginals, complex torsions
// or QuasiHarmonicOptions { marginals: MarginalEstimator::Histogram { bins: 100 }, periodic: PeriodicCorrelation::Centered }
let entropy = estimate_quasi_harmonic_entropy(&one_d_data, frames_end, &quasi_harmonic, &options)?;
println!("QH {} BQH {}", entropy.quasi_harmonic, entropy.boltzmann_quasi_harmonic);
```

The result also carries the terms the totals are built from: `first_order_gaussian`, `first_order_boltzmann` and `correlation`. `estimate_coordinate_quasi_harmonic_entropy` takes the same arguments and returns each coordinate's share of every one of them.

The split estimator takes the soft set from the coordinates' topology:

```rust
use nn_entropy::split::{SoftSelection, SplitOptions, estimate_split_entropy};

let soft = internal.soft_torsions(); // primary torsions about rotatable bonds
let split = SplitOptions::new(SoftSelection::Indices(&soft)); // validated defaults
let result = estimate_split_entropy(&data, frames, &split, &options)?;
println!("{} = {} + {} + {}", result.total, result.soft, result.hard, result.coupling);
```

`SoftSelection::Detect` finds soft coordinates from their basins (`basins::detect_basins`), and `estimate_joint_entropy` gives a joint nearest-neighbor estimate over up to 8 coordinates.
