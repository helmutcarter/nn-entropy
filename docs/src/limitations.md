# Interpretation and limitations

- Results are differential internal-coordinate entropies in natural-log units (equivalently, units of `k_B`), not kcal mol⁻¹ K⁻¹.
- The reported value does not include a BAT-to-Cartesian Jacobian correction, momentum entropy, or a standard-state term, and must not be described as an absolute thermodynamic entropy. `--jacobian` reports the Jacobian term separately, for the tree bond set, but does not add it to the total.
- MD frames are generally correlated, which biases nearest-neighbor estimates; see [Correlated frames](correlated-frames.md).
- The quasi-harmonic correlation term is a Gaussian approximation: it captures correlations in the second moments of the coordinates (for torsions, of `e^{iφ}`), not nonlinear dependence. See [Quasi-harmonic estimators](quasi-harmonic.md).
- The MIE is truncated; see [Nearest-neighbor MIE](nn-mie.md#notes).
- The split estimator's topological soft set misses rotors that the bond graph cannot identify as symmetric, such as t-butyl (whose methyls are not terminal atoms), so their basins can define states; detection can call a rotatable bond stuck in one well hard. Its per-state coupling fragments the frames and fits a Gaussian within each state, which is biased for small states. See [Split estimator](split.md).
