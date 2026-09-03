# nn-entropy

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
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--no-periodic] [--start N] [--stop N] [--stride N] [--mie-order 1|2|3|4]
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

Notes:
- `--stop` limits the number of frames read to N.
- `--start` skips the first N frames.
- `--stride` retains every Nth frame after `--start` (default: 1). Choose a stride based on a separate correlation-time analysis; the crate does not estimate an effective sample size automatically.
- `--mie-order` selects the expansion order. The default is 2, matching previous behavior.
- `--no-periodic` forces every coordinate to use an ordinary linear metric. This is intended for compatibility checks and diagnostics; file-based BAT calculations otherwise use the scientifically preferred periodic torsion metric.

## Rust library usage

```rust
use nn_entropy::calculate_entropy_from_data;

// one_d_data is Vec<Vec<f64>> with shape [n_coords][n_frames]
let entropy = calculate_entropy_from_data(one_d_data, frames_end)?;
```

Other helpers:
- `calculate_entropy_from_data_with_order` for explicit MIE order 1, 2, 3, or 4.
- `calculate_entropy_from_data_with_metrics` for explicit per-coordinate `CoordinateMetric::Linear` or `CoordinateMetric::Periodic { period }` metadata.
- `estimate_coordinate_entropy_rust` for per-coordinate entropy.
- `estimate_coordinate_mutual_information_rust` for pairwise mutual information.
- `estimate_coordinate_mie_entropy_rust` for per-coordinate second-order MIE entropy contributions. Each pairwise mutual information term is split evenly between the two coordinates, so the returned values sum to the order-2 total entropy.

## Python bindings

The crate exposes a `nn_entropy` Python module (built from `src/pyo3_api.rs`) with:
- `load_system(top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None)`
- `estimate_entropy(data_or_system, mie_order=None, periods=None)`
- `estimate_mie_entropy(data, mie_order=None, periods=None)`
- `estimate_coordinate_entropy(data, periods=None)`
- `estimate_coordinate_mutual_information(data, periods=None)`
- `estimate_coordinate_mie_entropy(data, periods=None)`
- File-based estimator functions accept the same selection arguments, including `stride`; their coordinate metrics come from the generated BAT coordinates automatically.

For a raw array, `periods` is a sequence with one entry per coordinate. Use `None` for a linear dimension and a positive period for a periodic dimension. Omitting `periods` preserves the linear behavior of the original raw-array API.

Preferred usage:

```python
import nn_entropy

system = nn_entropy.load_system("system.parm7", "trajectory.nc")
entropy = nn_entropy.estimate_entropy(system, mie_order=2)

coordinate_entropy = system.estimate_coordinate_entropy()
coordinate_mie_entropy = system.estimate_coordinate_mie_entropy()
```

A typical build workflow uses `maturin`:

```bash
maturin develop --release
```

Threading note:
- The Python wrappers release the GIL while loading trajectories and running entropy calculations, so multiple Python threads can call into `nn_entropy` concurrently.
- Internal parallelism is handled in Rust with Rayon. Set `RAYON_NUM_THREADS=N` if you want to cap or tune the Rust worker pool.

## Interpretation and limitations

- Results are differential internal-coordinate entropies in natural-log units (equivalently, units of `k_B`), not kcal mol⁻¹ K⁻¹.
- To remain numerically compatible with the historical Python implementation, the nearest-neighbor constant uses the large-sample approximation `ln(N) + EulerGamma` in place of the exact finite-sample term `psi(N) - psi(1) = H_(N-1)`. Their per-entropy difference is approximately `1/(2N)` and is negligible for the intended 50,000-frame calculations, but it can accumulate across MIE terms or become important for very small samples.
- The default second-order mutual-information expansion is a truncation and can omit higher-order correlations.
- The reported value does not include a BAT-to-Cartesian Jacobian correction, momentum entropy, or a standard-state term, and must not be described as an absolute thermodynamic entropy.
- Exact duplicate samples use the nearest distinct coordinate point. Many ties usually indicate inadequate coordinate precision or sampling and should be investigated.
- MD frames are generally correlated. Use a defensible stride or otherwise decorrelate the samples before interpreting the estimate.
- Corrected file-based results differ from historical versions because torsions are now periodic; the historical asymptotic finite-sample constant is retained for Python compatibility.

## Tests

```bash
cargo test --release
```

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
