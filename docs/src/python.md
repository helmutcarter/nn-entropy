# Python

## Functions

The `nn_entropy` module provides:
- `load_system(top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None, tree_bonds=False, phase_angles=True)`. See [Coordinates](coordinates.md).
- `estimate_entropy(data_or_system, mie_order=None, periods=None, periodic=True, exact_constant=False, theiler_window=0)`. `mie_order` defaults to 2; `mie_order=1` gives the sum of the marginal entropies.
- `estimate_coordinate_entropy(data_or_system, mie_order=None, periods=None, periodic=True, exact_constant=False, theiler_window=0)`. Returns each coordinate's share of the entropy; `mie_order` defaults to 2 and must be 1 or 2. The values sum to `estimate_entropy` at the same order.
- `estimate_coordinate_mutual_information(data_or_system, periods=None, periodic=True, exact_constant=False, theiler_window=0)`. Returns the mutual information of each coordinate pair.
- `estimate_quasi_harmonic_entropy(data_or_system, periods=None, periodic=True, histogram_bins=None, periodic_correlation="complex", exact_constant=False, theiler_window=0)`. Returns a dict with `quasi_harmonic`, `boltzmann_quasi_harmonic`, `first_order_gaussian`, `first_order_boltzmann` and `correlation`. See [Quasi-harmonic estimators](quasi-harmonic.md).
- `estimate_coordinate_quasi_harmonic_entropy(data_or_system, periods=None, periodic=True, histogram_bins=None, periodic_correlation="complex", exact_constant=False, theiler_window=0)`. Each coordinate's share of the QH and BQH entropy: a dict with the same keys, each a list with one value per coordinate that sums to the matching total. See [Quasi-harmonic estimators](quasi-harmonic.md#per-coordinate-entropies).
- `estimate_split_entropy(data_or_system, soft=None, soft_estimator="mie", mie_order=2, hard_estimator="qh", coupling="gaussian", soft_features="complex", state_coordinates=None, state_labels=None, min_state_frames=300, histogram_bins=None, periodic_correlation="centered", periods=None, periodic=True, exact_constant=False, theiler_window=0)`. Returns a dict with `total`, `soft`, `hard`, `coupling`, `soft_coordinates`, `hard_coordinates`, `basin_counts` and `states`. `soft` is `None`, `"topology"`, `"detect"` or a list of coordinate indices from 0; `None` means `"topology"` for a `System` and `"detect"` for an array. See [Split estimator](split.md).
- `detect_basins(values, period=2π, bins=72, smoothing=2.0, saddle_fraction=0.5, min_population=0.02, free_rotor_resultant=0.3)`. Returns a dict with `count`, `labels`, `boundaries`, `populations`, `resultant` and `soft`.
- `System` has the same six estimators as methods, each taking `periodic`, `exact_constant` and `theiler_window` (and `mie_order` for the two MIE entropy estimators, `histogram_bins` and `periodic_correlation` for the quasi-harmonic ones). Its `soft_torsions`, `symmetric_rotor_torsions` and `ring_torsions` properties give the split estimator's topological classes. Its `atoms`, `kinds` and `phase_references` properties describe the coordinates, in the order of every per-coordinate result: the topology atom indices of each (counted from 0), its kind (`"bond"`, `"angle"`, `"torsion"` or `"phase angle"`), and for a phase angle the index of its reference torsion, otherwise `None`. Its `mean_log_jacobian` property is the mean log BAT Jacobian over the selected frames. It raises `ValueError` unless the system was loaded with `tree_bonds=True` and without `torsions_only`.
- To start from an Amber topology and trajectory, call `load_system(...)` once and use the `System` methods or pass the `System` to any estimator. The coordinate metrics come from the generated BAT coordinates automatically, and the trajectory is only read once no matter how many quantities you compute.

`exact_constant=True` uses the exact finite-sample term, like the CLI's `--exact-constant`, and `theiler_window` matches the CLI's `--theiler-window`. `periodic=False` measures every coordinate linearly for that calculation, like the CLI's `--no-periodic`; a `System` keeps its own periods. For a raw array it cannot be combined with `periods`.

For a raw array, `periods` is a sequence with one entry per coordinate. Use `None` for a linear dimension and a positive period for a periodic dimension. Omitting `periods` preserves the linear behavior of the original raw-array API.

## Example


```python
import nn_entropy

system = nn_entropy.load_system("system.parm7", "trajectory.nc")
bqh = system.estimate_quasi_harmonic_entropy()["boltzmann_quasi_harmonic"]
entropy = nn_entropy.estimate_entropy(system, mie_order=2)  # NN MIE

coordinate_entropy = system.estimate_coordinate_entropy()  # order 2, sums to `entropy`
marginal_entropy = system.estimate_coordinate_entropy(mie_order=1)
```

See [Installation](installation.md) for building the module.

## Threads

- The Python wrappers release the GIL while loading trajectories and running entropy calculations, so multiple Python threads can call into `nn_entropy` concurrently.
- Internal parallelism is handled in Rust with Rayon. Set `RAYON_NUM_THREADS=N` if you want to cap or tune the Rust worker pool.
