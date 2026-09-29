# nn-entropy

[![CI](https://github.com/helmutcarter/nn-entropy/actions/workflows/ci.yml/badge.svg)](https://github.com/helmutcarter/nn-entropy/actions/workflows/ci.yml)

Estimate the configurational entropy of a molecular system, with a Rust library, CLI, and Python bindings.

This crate provides non-parametric entropy estimation using the nearest-neighbor method with mutual information expansion on internal coordinates (bond lengths, bond angles, and torsion angles) with built-in conversion from MD trajectory files. File-based calculations use linear metrics for bonds and angles and a 2π-periodic metric for torsions.

## Features
- Rust library API for MIE entropy up to fourth order, per-coordinate entropy, and mutual information estimates.
- CLI that reads `.parm7` + `.nc` and converts to internal coordinates and prints total configurational entropy.
- Python bindings via `pyo3` for in-memory arrays or direct file-based calculation.
- Entropy calculations parallelized with `rayon`.

## Requirements
- Rust toolchain (edition 2024).
- For Python bindings: a Python environment with build tooling for `pyo3` (see below).

## Build

```bash
cargo build --release
```

## CLI usage

```bash
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--no-periodic] [--exact-constant] [--start N] [--stop N] [--stride N] [--mie-order 1|2|3|4]
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

Notes:
- `--stop` limits the number of frames read to N.
- `--start` skips the first N frames.
- `--stride` retains every Nth frame after `--start` (default: 1). Choose a stride based on a separate correlation-time analysis; the crate does not estimate an effective sample size automatically.
- `--mie-order` selects the expansion order. The default is 2, and this should be good for most applications.
- `--no-periodic` forces every coordinate to use an ordinary linear metric. This produces a significant speedup, but the torsion entropy contribution loses some correctness.
- `--exact-constant` uses the exact finite-sample term `psi(N) - psi(1) = H_(N-1)`. The default is the common approximation `ln(N) + EulerGamma`.

## Rust library usage

```rust
use nn_entropy::{CoordinateMetric, EntropyOptions, FiniteSampleConstant, calculate_entropy};

// one_d_data is Vec<Vec<f64>> with shape [n_coords][n_frames]
let entropy = calculate_entropy(&one_d_data, frames_end, 2, &EntropyOptions::default())?;

// Periodic torsions and the exact finite-sample constant
let options = EntropyOptions {
    metrics: Some(&metrics), // one CoordinateMetric per coordinate
    constant: FiniteSampleConstant::Exact,
};
let entropy = calculate_entropy(&one_d_data, frames_end, 2, &options)?;
```

`EntropyOptions` has two fields:
- `metrics`: per-coordinate `CoordinateMetric::Linear` or `CoordinateMetric::Periodic { period }`. The default, `None`, treats every coordinate as linear.
- `constant`: `FiniteSampleConstant::PythonCompatibleAsymptotic` (the default) or `FiniteSampleConstant::Exact`.

Every estimator takes the same options:
- `calculate_entropy(data, frames_end, mie_order, &options)` returns the total entropy at MIE order 1, 2, 3 or 4.
- `estimate_coordinate_entropy(data, frames_end, mie_order, &options)` returns each coordinate's share of the entropy at MIE order 1 or 2, so the values sum to `calculate_entropy` at the same order and with the same options. Order 1 gives the marginal entropies. Order 2 also splits each pairwise mutual information term evenly between its two coordinates. Orders 3 and 4 are rejected.
- `estimate_coordinate_mutual_information(data, frames_end, &options)` returns the mutual information of each coordinate pair.
- `calc_joint_nn([&x, &y, ...], metrics)` is the underlying nearest-neighbor sum, `sum over frames of ln(nearest-neighbor distance)`, in 1 to 4 dimensions.

## Python bindings

The crate exposes a `nn_entropy` Python module (built from `src/pyo3_api.rs`) with:
- `load_system(top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None)`
- `estimate_entropy(data_or_system, mie_order=None, periods=None, exact_constant=False)`. `mie_order` defaults to 2; `mie_order=1` gives the sum of the marginal entropies.
- `estimate_coordinate_entropy(data_or_system, mie_order=None, periods=None, exact_constant=False)`. Returns each coordinate's share of the entropy; `mie_order` defaults to 2 and must be 1 or 2. The values sum to `estimate_entropy` at the same order.
- `estimate_coordinate_mutual_information(data_or_system, periods=None, exact_constant=False)`
- `System` has the same three estimators as methods, each taking `exact_constant` (and `mie_order` for the two entropy estimators).
- To start from an Amber topology and trajectory, call `load_system(...)` once and use the `System` methods or pass the `System` to any estimator. The coordinate metrics come from the generated BAT coordinates automatically, and the trajectory is only read once no matter how many quantities you compute.

`exact_constant=True` uses the exact finite-sample term, like the CLI's `--exact-constant`.

For a raw array, `periods` is a sequence with one entry per coordinate. Use `None` for a linear dimension and a positive period for a periodic dimension. Omitting `periods` preserves the linear behavior of the original raw-array API.

Preferred usage:

```python
import nn_entropy

system = nn_entropy.load_system("system.parm7", "trajectory.nc")
entropy = nn_entropy.estimate_entropy(system, mie_order=2)

coordinate_entropy = system.estimate_coordinate_entropy()  # order 2, sums to `entropy`
marginal_entropy = system.estimate_coordinate_entropy(mie_order=1)
```

A typical build workflow uses `maturin`:

```bash
maturin develop --release
```

Threading note:
- The Python wrappers release the GIL while loading trajectories and running entropy calculations, so multiple Python threads can call into `nn_entropy` concurrently.
- Internal parallelism is handled in Rust with Rayon. Set `RAYON_NUM_THREADS=N` if you want to cap or tune the Rust worker pool.

## Interpretation and limitations

- File-based calculations select the solute automatically. Water is excluded when its atom type is an Amber water type (`OW`, `HW`, `EP`, `EPW`) or its residue is a water name (`WAT`, `HOH`, `SOL`, `TIP3`, `SPCE`, `OPC`, and similar), so four-point models and non-Amber water types are handled. Massless virtual sites are excluded because they carry no degrees of freedom. Monatomic species such as counter-ions contribute nothing, since they have no internal coordinates, and a heavy-atom diatomic contributes only its bond length. Solute atoms may appear anywhere in the topology.
- The trajectory must have exactly as many atoms as the topology. A stripped topology paired with a solvated trajectory is rejected rather than read on the assumption that atoms line up.
- Results are differential internal-coordinate entropies in natural-log units (equivalently, units of `k_B`), not kcal mol⁻¹ K⁻¹.
- To remain numerically compatible with the historical Python implementation, the nearest-neighbor constant uses the large-sample approximation `ln(N) + EulerGamma` in place of the exact finite-sample term `psi(N) - psi(1) = H_(N-1)`. Their per-entropy difference is approximately `1/(2N)` and is negligible for the intended 50,000-frame calculations, but it can accumulate across MIE terms or become important for very small samples. The convention is a property of the estimator, not of the total alone: pairwise mutual information shifts by exactly `H_(N-1) - (ln(N) + EulerGamma)` per pair when it changes, so per-coordinate results must be computed under the same convention as the total they are compared against.
- The default second-order mutual-information expansion is a truncation and can omit higher-order correlations.
- The reported value does not include a BAT-to-Cartesian Jacobian correction, momentum entropy, or a standard-state term, and must not be described as an absolute thermodynamic entropy.
- Exact duplicate samples use the nearest distinct coordinate point. Many ties usually indicate inadequate coordinate precision or sampling and should be investigated.
- MD frames are generally correlated. Use a defensible stride or otherwise decorrelate the samples before interpreting the estimate.
- Corrected file-based results differ from historical versions because torsions are now periodic; the historical asymptotic finite-sample constant is retained for Python compatibility.

## Tests

```bash
cargo test --release
```

The suite is split by concern:

| File | Covers |
|---|---|
| `tests/common/mod.rs` | Shared helpers: a self-contained RNG, closed-form entropies, an O(N^2) nearest-neighbor reference. Calls none of the code it checks. |
| `tests/analytic.rs` | Closed-form ground truth: uniform, normal, exponential, circular uniform, bivariate-normal joint entropy and mutual information, plus a convergence check. |
| `tests/reference.rs` | Brute-force cross-checks of the kd-tree search for 1-4 dimensions and every metric combination, and of the MIE expansion against an independent inclusion-exclusion evaluation. |
| `tests/invariance.rs` | Translation, reflection, relabeling, frame reordering, scaling equivariance, periodic rotation, and floating-point behavior. |
| `tests/geometry.rs` | Bonds, angles and torsions from constructed configurations; the torsion sign convention; the end-to-end BAT path on the fixture. |
| `tests/validation.rs` | The input-validation contract: which inputs are refused and what each error says. |
| `tests/testing.rs`, `tests/entropy.rs`, `tests/cli.rs` | Unit tests, fixture regressions, and CLI behavior. |

Expected values in `analytic.rs`, `reference.rs`, `invariance.rs` and `geometry.rs`
are derived from theory or from an independent reference, never recorded from a
previous run, per the requirements in `REVIEW.md`.

The Rust suite does not link the Python extension, so the bindings have a
separate smoke test that runs against a built wheel:

```bash
maturin build --out dist
pip install --no-index --find-links dist nn_entropy
python tests/python_smoke.py
```

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, `cargo clippy -D warnings`
and `cargo test` on every push and pull request, plus the wheel build and smoke
test on Python 3.10 and 3.12.

## Project layout
- `src/lib.rs`: core entropy estimation and internal coordinate utilities.
- `src/bat_library/`: NetCDF reader for `.parm7` + `.nc`, and internal coordinate (BAT) conversion.
- `src/main.rs`: CLI entry point.
- `src/pyo3_api.rs`: Python bindings.
- `tests/`: unit and regression tests.

## Further Reading
- [Grid inhomogeneous solvation theory: Hydration structure and thermodynamics of the miniature receptor cucurbit[7]uril](https://pmc.ncbi.nlm.nih.gov/articles/PMC3416872/) - Uses nearest neighbor method to calculate first-order estimate of solvent entropy
- [Extraction of configurational entropy from molecular simulations via an expansion approximation](https://pubmed.ncbi.nlm.nih.gov/17640119/) - Uses mutual information expansion to increase accuracy of entropy estimation of highly correlated systems like molecules
- [Sample Estimate of the Entropy of a Random Vector](https://dmitripavlov.org/scans/kozachenko-leonenko.pdf) - First introduction of nearest neighbors entropy estimation


## License
© 2026 Helmut Carter, Kurtzman Lab. All rights reserved.
