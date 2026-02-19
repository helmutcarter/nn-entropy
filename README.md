# nn-entropy

Estimate the configurational entropy of a molecular system, with a Rust library, CLI, and Python bindings.

This crate provides non-parametric entropy estimation using the nearest neighbor method with first order mutual information expansion on internal coordinates (i.e., bonds lengths, bond angles, and torsion angles) and conversion from MD trajectory files.

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
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--start N] [--stop N] 
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

Notes:
- `--stop` limits the number of frames read to N.
- `--start` skips the first N frames

## Rust library usage

```rust
use nn_entropy::calculate_entropy_from_data;

// one_d_data is Vec<Vec<f64>> with shape [n_coords][n_frames]
let entropy = calculate_entropy_from_data(one_d_data, frames_end)?;
```

Other helpers:
- `estimate_coordinate_entropy_rust` for per-coordinate entropy.
- `estimate_coordinate_mutual_information_rust` for pairwise mutual information.

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