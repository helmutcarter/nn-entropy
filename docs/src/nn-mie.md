# Nearest-neighbor MIE

The nearest-neighbor estimator of Kozachenko and Leonenko estimates the entropy of a set of coordinates from the distance from each frame to its nearest neighbor among the other frames. The mutual information expansion (MIE) combines such estimates for single coordinates, pairs, and optionally triples and quadruples into an estimate for the whole molecule: order 1 is the sum of the single-coordinate entropies, order 2 also subtracts the mutual information of every pair, and so on. Select the order with `--mie-order` (default 2), `mie_order=` in Python, or the `mie_order` argument in Rust.

Periodic coordinates use the distance around the circle. `--no-periodic` treats every coordinate as linear, which is faster but loses some correctness for torsions.

## Options

In Rust these are the fields of `EntropyOptions`. The CLI and Python have matching settings: `--exact-constant` or `exact_constant=True`, and `--theiler-window` or `theiler_window=`. For file-based calculations the metrics come from the coordinates; for a raw array in Python, pass `periods=`.

- `metrics`: per-coordinate `CoordinateMetric::Linear` or `CoordinateMetric::Periodic { period }`. The default, `None`, treats every coordinate as linear.
- `constant`: `FiniteSampleConstant::Asymptotic` (the default) or `FiniteSampleConstant::Exact`.
- `theiler_window`: frames j with |i - j| <= `theiler_window` are never frame i's nearest neighbor. The default, 0, excludes only frame i itself.

## Joint estimate

The MIE builds the entropy from estimates in one to four dimensions. A single nearest-neighbor estimate over a whole set of coordinates is also available, with no expansion: `estimate_joint_entropy` in Rust, for up to 8 coordinates (`MAX_JOINT_DIMENSION`), and the split estimator's `--soft-estimator joint`. It has no truncation error, but its bias grows quickly with the number of coordinates, so it suits small sets such as the soft torsions of a ligand. On a periodic box the nearest-neighbor search skips the periodic images that cannot hold a closer neighbor, which keeps it fast in higher dimensions.

## Notes

- The default second-order mutual-information expansion is a truncation and can omit higher-order correlations.
- Following the historical Python implementation, the default nearest-neighbor constant uses the large-sample approximation `ln(N) + EulerGamma` in place of the exact finite-sample term `psi(N) - psi(1) = H_(N-1)`. Their per-entropy difference is approximately `1/(2N)` and is negligible for the intended 50,000-frame calculations, but it can accumulate across MIE terms or become important for very small samples. The convention is a property of the estimator, not of the total alone: pairwise mutual information shifts by exactly `H_(N-1) - (ln(N) + EulerGamma)` per pair when it changes, so per-coordinate results must be computed under the same convention as the total they are compared against. The Python implementation truncated EulerGamma to `0.57721566490153`; this crate uses the full double-precision value, so results can differ from it by roughly 1e-14. `--legacy` uses the truncated value.
- Exact duplicate samples use the nearest distinct coordinate point. Many ties usually indicate inadequate coordinate precision or sampling and should be investigated.

For correlated frames, see [Correlated frames](correlated-frames.md).
