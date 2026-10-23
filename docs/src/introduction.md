# nn-entropy

nn-entropy estimates the configurational entropy of a molecule from a molecular dynamics trajectory. It reads Amber topologies (`.parm7`) and NetCDF trajectories (`.nc`), converts every frame to internal coordinates (bond lengths, bond angles and torsions), and estimates the entropy of those coordinates. The same estimators are available from the [command line](cli.md), from [Python](python.md) and as a [Rust library](rust.md).

## Choosing an estimator

- **Boltzmann quasi-harmonic (BQH)** takes each coordinate's own entropy from its actual distribution and adds a Gaussian estimate of the correlations between coordinates. It is the most accurate estimator in the crate on the systems tested so far; see [Validation](validation.md). Use it unless you have a reason not to: `--estimator bqh`.
- **The nearest-neighbor mutual information expansion (NN MIE)** combines nearest-neighbor entropies of single coordinates and of pairs (up to sets of four). It makes no assumption about the shape of any distribution, but the sum of many pairwise terms is noisy. It is the CLI's default estimator. See [Nearest-neighbor MIE](nn-mie.md).
- **The split estimator** treats soft torsions (those about rotatable bonds) non-parametrically and the stiff coordinates with a Gaussian, plus a coupling term. It is meant for molecules with several coupled rotatable bonds: `--estimator split`. See [Split estimator](split.md).
- **Quasi-harmonic (QH)** fits a Gaussian to every coordinate. It is fast but overestimates the entropy of multiwell torsions. See [Quasi-harmonic estimators](quasi-harmonic.md).
- **Legacy mode** reproduces the original Python implementation bit for bit, for comparison with historical results. See [Legacy mode](legacy.md).

Every estimator returns a differential entropy in natural-log units (units of k_B) of the internal coordinates as sampled. It does not include the BAT Jacobian, momentum, or a standard state, so it is meaningful as a difference between ensembles of the same molecule rather than as an absolute thermodynamic entropy. See [Interpretation and limitations](limitations.md).
