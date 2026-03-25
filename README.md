# nn-entropy

Estimate the configurational entropy of a molecular system, with a Rust library, CLI, and Python bindings.

This crate provides non-parametric entropy estimation using the nearest neighbor method with first order mutual information expansion on internal coordinates (i.e., bonds lengths, bond angles, and torsion angles) with built-in conversion from MD trajectory files.

## Features
- Rust library API for first order entropy, per-coordinate entropy, and mutual information estimates.
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
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--start N] [--stop N] [--tau-observable NAME] [--mdout PATH] [--extrapolate] [--subset-sizes A,B,C] [--subset-count N] [--bootstrap-replicates N] [--block-size N] [--fit-model NAME] [--bootstrap-seed N] [--output-csv PATH]
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

Notes:
- `--torsions-only`: off by default. When omitted, the CLI uses all internal coordinates.
- `--start`: defaults to `0`.
- `--stop`: defaults to reading all available frames.
- `--tau-observable`: off by default. Supported values are `all-internal-coordinates`, `dihedral-angles`, and `potential-energy`.
- `--mdout`: no default. It is required with `--tau-observable potential-energy` and should point to an AMBER `mdout` file containing `EPtot` records.
- `--extrapolate`: off by default. When enabled, the CLI runs the finite-sample workflow: nested subsets, direct fitting on the raw subset curve by default, automatic comparison between `inverse-n` and `inverse-sqrt-n`, and trailing-subset stability checks.
- `--fit-model`: defaults to `inverse-n`. Supported values are `inverse-n` and `inverse-sqrt-n`.
- `--subset-sizes`: no default. When omitted, the CLI generates nested subset sizes automatically.
- `--subset-count`: defaults to `5` when `--subset-sizes` is omitted.
- `--bootstrap-replicates`: off by default. When provided, bootstrap becomes enabled with the requested replicate count.
- `--block-size`: only used when bootstrap is enabled. It defaults to `ceil(tau)` when `tau` is available, otherwise `1`.
- `--bootstrap-seed`: only used when bootstrap is enabled. When omitted, bootstrap sampling is nondeterministic.
- `--output-csv`: no default. When provided with `--extrapolate`, writes the convergence table to a CSV file.

Example extrapolation run on toy dataset:

```bash
cargo run --release tests/fixtures/test.parm7 tests/fixtures/test.nc --extrapolate --fit-model inverse-sqrt-n --subset-sizes 3,4,5 --output-csv convergence.csv
```

To enable bootstrap explicitly:

```bash
cargo run --release tests/fixtures/test.parm7 tests/fixtures/test.nc --extrapolate --fit-model inverse-sqrt-n --subset-sizes 3,4,5 --bootstrap-replicates 3 --block-size 5 --output-csv convergence.csv
```

The CSV contains one row per subset per model, plus stability rows for largest-subset refits. The main columns are:
- `row_type`: `fit_point` or `stability`
- `series`: `primary` or `comparison`
- `model`: `inverse-n` or `inverse-sqrt-n`
- `n_or_subset_count`: subset size `N` for `fit_point` rows, or the number of largest subsets used for a `stability` refit
- `effective_samples`, `x`, `mean_entropy`, `variance`, `weight`, `fitted_entropy`, `residual`
- `intercept`, `intercept_std_err`, `slope`, `slope_std_err`, `weighted_residual_sum_squares`

## Rust library usage

```rust
use nn_entropy::calculate_entropy_from_data;

// one_d_data is Vec<Vec<f64>> with shape [n_coords][n_frames]
let entropy = calculate_entropy_from_data(one_d_data, frames_end)?;
```

Other helpers:
- `estimate_coordinate_entropy_rust` for per-coordinate entropy.
- `estimate_coordinate_mutual_information_rust` for pairwise mutual information.
- `run_entropy_extrapolation` for the in-memory extrapolation workflow.
- `run_entropy_extrapolation_from_files` for the file-based extrapolation workflow.

## Python bindings

The crate exposes a `nn_entropy` Python module (built from `src/pyo3_api.rs`) with:
- `estimate_mie_entropy(data)`
- `estimate_coordinate_entropy(data)`
- `estimate_coordinate_mutual_information(data)`
- `estimate_mie_entropy_from_files(top_path, traj_path, start=None, stop=None, torsions_only=None)`

A typical build workflow uses `maturin`:

```bash
maturin develop --release
```

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
