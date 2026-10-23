# Quasi-harmonic estimators

Both quasi-harmonic estimators describe correlations with the correlation matrix C of the coordinates, `C_ij = σ_ij / √(σ_ii σ_jj)`, which contributes `½ ln|C| ≤ 0`. They differ in the first-order term:

- **QH** (`--estimator qh`, Karplus and Kushick 1981) fits a Gaussian to each coordinate: `S = Σ ½ ln(2πe σ_ii) + ½ ln|C|`, which is `½ ln((2πe)^n |σ|)` when no coordinate is periodic.
- **BQH** (`--estimator bqh`, Harpole and Sharp, *J. Phys. Chem. B* 2011) keeps the correlation term but takes each coordinate's first-order entropy, `-∫ p ln p`, from its actual distribution, so multimodal coordinates are not treated as one wide well.

The BQH first-order term uses the one-dimensional nearest-neighbor estimate by default, the same first-order term as the MIE, so BQH and the NN MIE differ only in how they treat correlations; `--exact-constant` and `--theiler-window` apply to it. `--histogram-bins N` uses N equal bins instead: linear coordinates are binned over their observed range, periodic ones over a full period. The histogram gives the differential entropy `-Σ p_k ln(p_k / Δ)`, Harpole and Sharp's eq 9, comparable with the other estimators. Some later BQH work reports the discrete `-Σ p_k ln p_k`, which is this value minus `ln Δ` per coordinate, a difference that cancels in entropy changes computed with the same bins. Harpole and Sharp choose bins by width (3-5° for angles, 0.1 Å for lengths) rather than by count. All values are differential entropies in k_B, like the nearest-neighbor results.

The covariance is the population covariance over the selected frames, so there must be more frames than coordinates, and no coordinate can be a linear combination of others.

**Torsions.** By default (`--periodic-correlation complex`), each periodic coordinate enters the correlation matrix as `z = e^{iφ}` (for period P, `e^{2πiq/P}`), following Harpole and Sharp, which makes C Hermitian; `C_ij = ⟨δa_i δa_j*⟩ / √(⟨|δa_i|²⟩⟨|δa_j|²⟩)` with `δa = a - ⟨a⟩`, and linear coordinates stay real. This captures correlations on the circle whatever the angles' spread, such as the fixed 120° offsets between the three hydrogen torsions of a rotating methyl group. The first-order Gaussian term of QH uses each torsion's variance after rotating its circular mean to 0 and wrapping to (-π, π].

`--periodic-correlation centered` puts the centered and wrapped torsions into C as real coordinates instead. That is adequate for torsions confined to one narrow well, where both treatments agree, but a correlation between torsions that sweep the circle looks like a weak sawtooth after wrapping and is largely missed.

## Per-coordinate entropies

`--per-coordinate`, `estimate_coordinate_quasi_harmonic_entropy` in Python and Rust, gives each coordinate's share of the QH and BQH entropy. The first-order terms belong to single coordinates already. The correlation term `½ ln|C|` belongs to the whole set and is shared out using `ln|C| = tr(log C)`: coordinate i receives `½ (log C)_ii`, computed from the eigendecomposition of C. The shares

- sum to the correlation term exactly,
- do not depend on the order of the coordinates,
- are each at most 0, because C has a unit diagonal, and are about 0 for a coordinate independent of the others,
- split the correlation of two coordinates evenly: each receives `¼ ln(1 - |ρ|²)`.

Like the NN MIE's even split of pairwise mutual information, this is one convention for dividing a quantity that belongs to the whole set; its advantage is that it involves no further choices, such as an ordering of the coordinates.

The [split estimator](split.md) combines both approaches: QH or BQH for the stiff coordinates, a nearest-neighbor estimate for the soft torsions, and the Gaussian correlation term of this page for the coupling between them.

See [Validation](validation.md) for how the estimators compare with thermodynamic reference values.
