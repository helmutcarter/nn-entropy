# nn-entropy

[![CI](https://github.com/helmutcarter/nn-entropy/actions/workflows/ci.yml/badge.svg)](https://github.com/helmutcarter/nn-entropy/actions/workflows/ci.yml)

Estimate the configurational entropy of a molecular system, with a Rust library, CLI, and Python bindings.

The crate converts Amber trajectories to internal coordinates (bond lengths, bond angles, and torsions) and estimates their entropy. File-based calculations use linear metrics for bonds and angles and a 2π-periodic metric for torsions.

## Features
- Estimators: the nearest-neighbor mutual information expansion (NN MIE) up to fourth order, with pairwise mutual information; quasi-harmonic (QH) and Boltzmann quasi-harmonic (BQH) entropy; and a split estimator that treats soft torsions non-parametrically and stiff coordinates with a Gaussian. Each of the first three can also report every coordinate's share of the entropy.
- Validated against thermodynamic reference entropies; see [Validation](https://helmutcarter.github.io/nn-entropy/validation.html).
- A CLI that reads `.parm7` + `.nc`, a Rust library, and Python bindings via `pyo3` for in-memory arrays or file-based calculation.
- A [user guide](https://helmutcarter.github.io/nn-entropy/) covering the CLI, Python and Rust interfaces and the methods.
- Calculations parallelized with `rayon`.

## Requirements
- Rust 1.94 or newer.
- For Python bindings: a Python environment with build tooling for `pyo3` (see below).

## Build
```bash
cargo build --release
```

For the Python module, use [maturin](https://www.maturin.rs/): `maturin develop --release`.

## Quick start

From the command line:

```bash
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--no-periodic] [--exact-constant] [--start N] [--stop N] [--stride N] [--mie-order 1|2|3|4] [--theiler-window W] [--tree-bonds] [--no-phase-angles] [--jacobian] [--legacy] [--estimator nn|qh|bqh|split] [--histogram-bins N] [--periodic-correlation complex|centered] [--per-coordinate] [--mutual-information] [--soft topology|detect|LIST] [--soft-estimator mie|joint] [--hard-estimator qh|bqh] [--coupling gaussian|states|none] [--soft-features complex|cossin] [--state-torsions auto|LIST] [--min-state-frames N]
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

For example, Boltzmann quasi-harmonic entropy, the most accurate estimator here:

```bash
cargo run --release system.parm7 trajectory.nc --estimator bqh
```

From Python:

```python
import nn_entropy

system = nn_entropy.load_system("system.parm7", "trajectory.nc")
entropy = system.estimate_quasi_harmonic_entropy()["boltzmann_quasi_harmonic"]
```

The [user guide](https://helmutcarter.github.io/nn-entropy/) documents every [option](https://helmutcarter.github.io/nn-entropy/cli.html), the [Python](https://helmutcarter.github.io/nn-entropy/python.html) and [Rust](https://helmutcarter.github.io/nn-entropy/rust.html) interfaces, and the [methods](https://helmutcarter.github.io/nn-entropy/coordinates.html).

Results are differential entropies of the internal coordinates in units of k_B, meaningful as differences between ensembles of the same molecule, not absolute thermodynamic entropies; see [Interpretation and limitations](https://helmutcarter.github.io/nn-entropy/limitations.html).

## Validation

On the butane end states of an alchemical transformation, BQH with phase angles agrees with thermodynamic integration within the reference's uncertainty. The [validation page](https://helmutcarter.github.io/nn-entropy/validation.html) has the results for every estimator and an example program that reproduces them.

## Documentation

The user guide is an mdBook in `docs/`, published at <https://helmutcarter.github.io/nn-entropy/> from the main branch. To read it locally, run `mdbook serve docs`. The Rust API documentation is built with `cargo doc --no-deps --open`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test suite, CI, and the layout of the code.

## License
© 2026 Helmut Carter, Kurtzman Lab. All rights reserved.
