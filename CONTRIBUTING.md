# Contributing

## Tests

```bash
cargo test --release
```

The suite is split by concern:

| File | Covers |
|---|---|
| `tests/common/` | Shared helpers: a self-contained RNG, closed-form entropies, an O(N^2) nearest-neighbor reference, and writers for Amber topologies and NetCDF trajectories (`amber.rs`). Calls none of the code it checks. |
| `tests/analytic.rs` | Closed-form ground truth: uniform, normal, exponential, circular uniform, bivariate-normal joint entropy and mutual information, plus a convergence check. |
| `tests/reference.rs` | Brute-force cross-checks of the kd-tree search for 1-4 dimensions and every metric combination, and of the MIE expansion against an independent inclusion-exclusion evaluation. |
| `tests/invariance.rs` | Translation, reflection, relabeling, frame reordering, scaling equivariance, periodic rotation, and floating-point behavior. |
| `tests/geometry.rs` | Bonds, angles and torsions from constructed configurations; the torsion sign convention; the end-to-end BAT path on the fixture. |
| `tests/topology.rs` | Atom selection and file reading across layouts the fixture does not cover: solvent before or between solute molecules, counter-ions, four-point and non-Amber water, and NetCDF corner cases. |
| `tests/validation.rs` | The input-validation contract: which inputs are refused and what each error says. |
| `tests/coordinates.rs` | Coordinate sets: 3N - 6 tree bonds forming a spanning tree, phase angles as wrapped torsion offsets that are stiff for a rigid rotor, the BAT Jacobian against the constructed geometry, and the ring and rotor classification (soft, symmetric-rotor and ring torsions of ethane, cyclopropane, methylcyclopropane and a chain). |
| `tests/quasi_harmonic.rs` | QH against the Gaussian entropy of independently computed sample covariances and the bivariate-normal closed form, BQH correcting QH for a bimodal coordinate, histograms against hand-counted bins, the complex torsion correlations against hand-computed Hermitian coefficients and determinants, the per-coordinate shares (sums, the two-coordinate split, relabelling, independence), periodic centering, validation, and the CLI flags. |
| `tests/joint.rs` | Joint nearest-neighbor entropy at run-time dimension: the unit-ball volume against an independent recurrence, bit-equality with the const-generic search, brute-force agreement for 5-8 dimensions across the periodic wrap and with a Theiler window, and zero entropy on a uniform periodic box. |
| `tests/basins.rs` | Basin detection on samples from known wells: 1, 2 and 3 wells (including across the wrap), uniform distributions, rare wells, rotation invariance, determinism, and invalid input. |
| `tests/split.rs` | The split estimator: exact identities with the estimators it is built from, a closed-form Gaussian coupling, a well-dependent system of known entropy recovered by per-state coupling, sparse-state renormalization, soft-set detection, validation, and the CLI flags. |
| `tests/legacy.rs` | `--legacy` against totals produced by the original Python script on the fixture, compared bit for bit, and the legacy angle and torsion formulas. |
| `tests/theiler.rs` | Theiler window: bitwise equality with the plain estimator at `W = 0`, brute-force agreement for every window and dimension, the closed-form constant, validation, and AR(1) bias behavior. |
| `tests/testing.rs`, `tests/entropy.rs`, `tests/cli.rs` | Unit tests, fixture regressions, and CLI behavior. |

Expected values in `analytic.rs`, `reference.rs`, `invariance.rs`, `geometry.rs`,
`coordinates.rs`, `quasi_harmonic.rs`, `joint.rs`, `basins.rs`, `split.rs` and
`theiler.rs` are derived from theory or
from an independent reference, never recorded from a previous run; those in
`legacy.rs` were produced by the original Python script. The pinned totals in
`entropy.rs` are regression values.

The Rust suite does not link the Python extension, so the bindings have a
separate smoke test that runs against a built wheel:

```bash
maturin build --out dist
pip install --no-index --find-links dist nn_entropy
python tests/python_smoke.py
```

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, `cargo clippy -D warnings`
and `cargo test` (doctests included) on every push to main and every pull request,
plus the wheel build and smoke test on Python 3.10 and 3.12. Separate jobs build
the user guide and the API documentation and check that the crate still builds on
the minimum supported Rust version, 1.94 (`rust-version` in `Cargo.toml`).
A coverage job reports line and region coverage per file in the run's summary,
without a threshold. To measure it locally:

```bash
cargo llvm-cov --release --summary-only --ignore-filename-regex 'pyo3_api\.rs'
```

The Python bindings are excluded: the Rust tests cannot reach them, and the
smoke test that does runs outside `cargo`.

## Documentation

The user guide lives in `docs/` as an mdBook. Build it with `mdbook build docs` (CI uses mdBook 0.4.52) or preview it with `mdbook serve docs`. `.github/workflows/pages.yml` publishes it on every push to main. Public Rust items must have doc comments: `#![warn(missing_docs)]` together with clippy's `-D warnings` enforces it, and CI builds the API docs with `RUSTDOCFLAGS=-D warnings`.

## Project layout
- `src/lib.rs`: the nearest-neighbor estimators, the MIE, and the nearest-neighbor search; also the crate documentation.
- `src/quasi_harmonic.rs`: the QH and BQH estimators.
- `src/split.rs`: the split (soft/hard) estimator.
- `src/basins.rs`: basin detection for periodic coordinates.
- `src/legacy.rs`: the reproduction of the original Python implementation.
- `src/bat_library/`: `.parm7` parser, NetCDF trajectory reader, solute selection, BAT coordinate construction, and ring and rotor classification (`geometry.rs` holds the bond, angle and torsion functions).
- `src/main.rs`: CLI entry point.
- `src/pyo3_api.rs`: Python bindings.
- `docs/`: the user guide (mdBook).
- `tests/`: the test suite (see [Tests](#tests)).
- `examples/butane_validation.rs`: reproduces the validation results from the butane trajectories.
