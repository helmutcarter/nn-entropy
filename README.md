# nn-entropy

Nearest-neighbor entropy estimation for internal coordinates, with a Rust library, CLI, and Python bindings.

This crate provides non-parametric entropy estimation using the nearest neighbor method with first order mutual information expansion on internal coordinate time series (e.g., molecular bonds, angles, and torsions converted from MD trajectory files).

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
cargo build
```

## CLI usage

```bash
cargo run -- <path_to_parm7> <path_to_nc> [--torsions-only] [--frames N] [--start N] [--stop N] [--python]
```

Example:

```bash
cargo run -- <path_to_parm7> <path_to_nc> --torsions-only
```

Notes:
- `--frames` and `--stop` are mutually exclusive in practice; both limit the number of frames read.
- `--start` skips initial frames after reading.

## Rust library usage

```rust
use nn_entropy::calculate_entropy_from_data;

// one_d_data is Vec<Vec<f64>> with shape [n_coords][n_frames]
let entropy = calculate_entropy_from_data(one_d_data, frames_end);
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
maturin develop
```

## Tests

```bash
cargo test
```

## Project layout
- `src/lib.rs`: core entropy estimation and internal coordinate utilities.
- `src/bat_library/`: BAT list builder and NetCDF reader for `.parm7` + `.nc`.
- `src/main.rs`: CLI entry point.
- `src/pyo3_api.rs`: Python bindings.
- `tests/`: unit and regression tests.

## License

No license specified yet.
