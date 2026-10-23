"""Smoke test for the built `nn_entropy` Python extension.

`cargo test` exercises the Rust core but never links the cdylib against
libpython, so this checks the parts only the wheel can prove: that the module
imports, that every documented entry point is exported, and that the numpy
bridge returns the right numbers.

Expected values are derived analytically below, not copied from a previous run.

Run against an installed wheel:

    maturin build --out dist
    pip install --no-index --find-links dist nn_entropy
    python tests/python_smoke.py
"""

import math
import sys
from pathlib import Path

import numpy as np

import nn_entropy

EULER_MASCHERONI = float(np.euler_gamma)

EXPECTED_EXPORTS = {
    "System",
    "load_system",
    "estimate_entropy",
    "estimate_coordinate_entropy",
    "estimate_coordinate_mutual_information",
    "estimate_quasi_harmonic_entropy",
    "estimate_coordinate_quasi_harmonic_entropy",
    "estimate_split_entropy",
    "detect_basins",
}


REMOVED_EXPORTS = {
    "estimate_mie_entropy",
    "estimate_coordinate_mie_entropy",
    "estimate_mie_entropy_from_files",
    "estimate_coordinate_entropy_from_files",
    "estimate_coordinate_mutual_information_from_files",
    "estimate_coordinate_mie_entropy_from_files",
}


def check_exports():
    missing = EXPECTED_EXPORTS - set(dir(nn_entropy))
    if missing:
        raise AssertionError(f"module is missing exports: {sorted(missing)}")
    stale = REMOVED_EXPORTS & set(dir(nn_entropy))
    if stale:
        raise AssertionError(f"module still exports removed names: {sorted(stale)}")
    print(f"exports: all {len(EXPECTED_EXPORTS)} entry points present")


def check_closed_form_entropy():
    """One coordinate, two frames one unit apart.

    Both nearest-neighbor distances are 1, so the mean log distance vanishes and
    the order-1 estimate collapses to the constant alone:
        ln(V_1) + ln(N) + gamma = ln(2) + ln(2) + gamma = ln(4) + gamma.
    """
    data = np.array([[0.0, 1.0]])
    got = nn_entropy.estimate_entropy(data, mie_order=1)
    want = math.log(4.0) + EULER_MASCHERONI
    assert abs(got - want) < 1e-12, f"estimate_entropy: got {got!r}, want {want!r}"
    print(f"order-1 closed form: {got!r}")


def check_exact_constant_closed_form():
    """Same input as above with the exact finite-sample term psi(N) - psi(1).

    For N = 2 that term is H_1 = 1, so the estimate is ln(2) + 1.
    """
    data = np.array([[0.0, 1.0]])
    got = nn_entropy.estimate_entropy(data, mie_order=1, exact_constant=True)
    want = math.log(2.0) + 1.0
    assert abs(got - want) < 1e-12, f"exact constant: got {got!r}, want {want!r}"
    per_coordinate = nn_entropy.estimate_coordinate_entropy(
        data, mie_order=1, exact_constant=True
    )
    assert abs(per_coordinate[0] - want) < 1e-12, per_coordinate
    print(f"order-1 exact closed form: {got!r}")


def check_theiler_window_closed_form():
    """Four frames one unit apart with theiler_window=1.

    Every frame's nearest neighbor outside the window is two units away, and
    frames 0 and 3 keep n_i = 3 candidates while frames 1 and 2 keep n_i = 2,
    so the exact estimate is ln(2) + ln(V_1) + mean(H_(n_i - 1))
    = 2 ln(2) + (1.5 + 1 + 1 + 1.5) / 4.
    """
    data = np.array([[0.0, 1.0, 2.0, 3.0]])
    got = nn_entropy.estimate_entropy(
        data, mie_order=1, exact_constant=True, theiler_window=1
    )
    want = 2.0 * math.log(2.0) + 1.25
    assert abs(got - want) < 1e-12, f"theiler window: got {got!r}, want {want!r}"
    assert nn_entropy.estimate_entropy(data, mie_order=1) == nn_entropy.estimate_entropy(
        data, mie_order=1, theiler_window=0
    )
    try:
        nn_entropy.estimate_entropy(data, mie_order=1, theiler_window=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a window wider than the series should raise ValueError")
    print(f"theiler window closed form: {got!r}")


def check_coordinate_entropies_sum_to_total():
    """The order-1 total is by definition the sum of the per-coordinate values."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 64))
    per_coordinate = nn_entropy.estimate_coordinate_entropy(data, mie_order=1)
    total = nn_entropy.estimate_entropy(data, mie_order=1)
    assert len(per_coordinate) == 4, per_coordinate
    assert abs(sum(per_coordinate) - total) < 1e-9, (sum(per_coordinate), total)
    print(f"order-1 additivity: {total!r}")


def check_mie_entropy_sums_to_second_order_total():
    """Splitting each pairwise mutual information evenly is an algebraic
    identity, so the per-coordinate values at the default order (2) must sum to
    the default-order total."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4, 64))
    per_coordinate = nn_entropy.estimate_coordinate_entropy(data)
    total = nn_entropy.estimate_entropy(data)
    assert abs(total - nn_entropy.estimate_entropy(data, mie_order=2)) < 1e-12
    assert abs(sum(per_coordinate) - total) < 1e-9, (sum(per_coordinate), total)
    print(f"order-2 additivity: {total!r}")


def check_translation_invariance():
    """Differential entropy is invariant under a rigid shift of any coordinate."""
    rng = np.random.default_rng(2)
    data = rng.normal(size=(3, 64))
    shifted = data + np.array([[10.0], [-4.0], [0.5]])
    base = nn_entropy.estimate_entropy(data, mie_order=2)
    moved = nn_entropy.estimate_entropy(shifted, mie_order=2)
    assert abs(base - moved) < 1e-9, (base, moved)
    print(f"translation invariance: {base!r}")


def check_periodic_metric_accepted():
    """`periods` must be accepted per coordinate and must change the answer for
    samples that straddle the branch cut."""
    period = 2.0 * math.pi
    data = np.array([[0.05, period - 0.05, 3.0, 3.4, 1.2]])
    linear = nn_entropy.estimate_entropy(data, mie_order=1)
    periodic = nn_entropy.estimate_entropy(data, mie_order=1, periods=[period])
    assert periodic < linear, (periodic, linear)
    print(f"periodic metric: linear {linear!r} -> periodic {periodic!r}")


def check_invalid_period_is_rejected():
    data = np.array([[0.0, 1.0, 2.0]])
    try:
        nn_entropy.estimate_entropy(data, mie_order=1, periods=[-1.0])
    except ValueError:
        print("invalid period: rejected with ValueError")
    else:
        raise AssertionError("a negative period should raise ValueError")


def check_coordinate_options():
    """Phase angles rewrite only torsions; the Jacobian needs the tree bond set.

    The values themselves are checked in tests/coordinates.rs; this checks that
    the keyword arguments reach the Rust options and the getter behaves.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    top, traj = str(fixtures / "test.parm7"), str(fixtures / "test.nc")

    plain = nn_entropy.load_system(top, traj, phase_angles=False)
    phase = nn_entropy.load_system(top, traj)  # phase angles are the default
    assert plain.coordinates == phase.coordinates
    changed = set()
    for index, (period, a, b) in enumerate(zip(plain.periods, plain.data, phase.data)):
        if a != b:
            assert period is not None, f"phase angles changed non-torsion {index}"
            changed.add(index)
    assert changed, "phase_angles=True changed no torsion"

    try:
        plain.mean_log_jacobian
    except ValueError:
        pass
    else:
        raise AssertionError("mean_log_jacobian needs tree_bonds=True")

    # The coordinates are described in the order of every per-coordinate result.
    assert len(phase.atoms) == len(phase.kinds) == len(phase.phase_references) == phase.coordinates
    sizes = {"bond": 2, "angle": 3, "torsion": 4, "phase angle": 4}
    for atoms, kind, reference, period in zip(phase.atoms, phase.kinds, phase.phase_references, phase.periods):
        assert len(atoms) == sizes[kind], (atoms, kind)
        assert (reference is not None) == (kind == "phase angle"), (kind, reference)
        assert (period is not None) == (len(atoms) == 4), (kind, period)
    assert "phase angle" in phase.kinds and "phase angle" not in plain.kinds

    # periodic=False measures every coordinate linearly for one calculation;
    # the System's own periods are unchanged.
    linear = phase.estimate_entropy(periodic=False)
    assert linear != phase.estimate_entropy()
    assert nn_entropy.estimate_entropy(phase, periodic=False) == linear
    assert any(p is not None for p in phase.periods)
    array = np.array(phase.data)
    assert nn_entropy.estimate_entropy(array, periodic=False) == nn_entropy.estimate_entropy(array)
    try:
        nn_entropy.estimate_entropy(array, periods=phase.periods, periodic=False)
    except ValueError:
        pass
    else:
        raise AssertionError("periods and periodic=False together should raise ValueError")

    tree = nn_entropy.load_system(top, traj, tree_bonds=True, phase_angles=True)
    assert tree.coordinates > plain.coordinates, "tree bonds should add X-H bonds"
    assert math.isfinite(tree.mean_log_jacobian)
    print(
        f"coordinate options: {len(changed)} phase angles, "
        f"mean ln J {tree.mean_log_jacobian!r}"
    )


def check_quasi_harmonic():
    """QH is the Gaussian entropy of the sample covariance, computed here with numpy."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=2000)
    y = 0.6 * x + rng.normal(size=2000)
    data = np.array([x, y])
    got = nn_entropy.estimate_quasi_harmonic_entropy(data)
    sigma = np.cov(data, bias=True)
    want = math.log(2 * math.pi * math.e) + 0.5 * math.log(np.linalg.det(sigma))
    assert abs(got["quasi_harmonic"] - want) < 1e-10, (got, want)
    r = sigma[0, 1] / math.sqrt(sigma[0, 0] * sigma[1, 1])
    assert abs(got["correlation"] - 0.5 * math.log(1 - r * r)) < 1e-10, got
    assert got["boltzmann_quasi_harmonic"] == got["first_order_boltzmann"] + got["correlation"]

    shares = nn_entropy.estimate_coordinate_quasi_harmonic_entropy(data)
    assert set(shares) == set(got), shares.keys()
    for key, values in shares.items():
        assert len(values) == 2 and abs(sum(values) - got[key]) < 1e-9, (key, values, got[key])
    # Two coordinates split the correlation term evenly.
    assert abs(shares["correlation"][0] - shares["correlation"][1]) < 1e-12, shares

    histogram = nn_entropy.estimate_quasi_harmonic_entropy(data, histogram_bins=50)
    assert histogram["quasi_harmonic"] == got["quasi_harmonic"]
    assert histogram["first_order_boltzmann"] != got["first_order_boltzmann"]

    # A rotor pair: e^(i phi) captures the 120 degree offset, centering does not.
    phi = rng.uniform(-math.pi, math.pi, size=5000)
    psi = np.angle(np.exp(1j * (phi + 2 * math.pi / 3 + 0.2 * rng.normal(size=5000))))
    rotor = np.array([phi, psi])
    periods = [2 * math.pi, 2 * math.pi]
    complex_ = nn_entropy.estimate_quasi_harmonic_entropy(rotor, periods=periods)
    centered = nn_entropy.estimate_quasi_harmonic_entropy(
        rotor, periods=periods, periodic_correlation="centered"
    )
    dz = [np.exp(1j * a) - np.exp(1j * a).mean() for a in (phi, psi)]
    rho = (dz[0] * dz[1].conj()).mean() / math.sqrt(
        (abs(dz[0]) ** 2).mean() * (abs(dz[1]) ** 2).mean()
    )
    assert abs(complex_["correlation"] - 0.5 * math.log(1 - abs(rho) ** 2)) < 1e-10, complex_
    assert centered["correlation"] > complex_["correlation"] + 0.5, (centered, complex_)
    try:
        nn_entropy.estimate_quasi_harmonic_entropy(rotor, periods=periods, periodic_correlation="polar")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown periodic_correlation should raise ValueError")

    # The fixture has 5 frames and 29 coordinates: too few for a correlation matrix.
    fixtures = Path(__file__).resolve().parent / "fixtures"
    system = nn_entropy.load_system(str(fixtures / "test.parm7"), str(fixtures / "test.nc"))
    for method in (system.estimate_quasi_harmonic_entropy, system.estimate_coordinate_quasi_harmonic_entropy):
        try:
            method()
        except ValueError:
            pass
        else:
            raise AssertionError(f"{method.__name__}: 5 frames of 29 coordinates should be rejected")
    print(f"quasi-harmonic: QH {got['quasi_harmonic']!r}, BQH {got['boltzmann_quasi_harmonic']!r}")


def check_split():
    """Split entropy and basin detection on synthetic data; values are checked
    against the library in the Rust tests, so this checks the bindings."""
    rng = np.random.default_rng(11)
    n = 6000
    well = rng.integers(0, 2, size=n)
    phi = np.angle(np.exp(1j * (np.where(well == 0, -1.5, 1.5) + 0.25 * rng.normal(size=n))))
    x = np.where(well == 0, 0.1, 0.3) * rng.normal(size=n)
    narrow = np.angle(np.exp(1j * (1.0 + 0.1 * rng.normal(size=n))))
    data = np.array([phi, x, narrow])
    periods = [2 * math.pi, None, 2 * math.pi]

    basins = nn_entropy.detect_basins(phi)
    assert basins["count"] == 2 and basins["soft"], basins["count"]
    assert len(basins["labels"]) == n and abs(sum(basins["populations"]) - 1) < 1e-12
    assert not nn_entropy.detect_basins(narrow)["soft"]
    assert nn_entropy.detect_basins(rng.uniform(0, 2 * math.pi, size=n))["soft"]

    result = nn_entropy.estimate_split_entropy(data, periods=periods)
    assert result["soft_coordinates"] == [0] and result["hard_coordinates"] == [1, 2], result
    assert abs(result["total"] - (result["soft"] + result["hard"] + result["coupling"])) < 1e-12
    assert result["coupling"] <= 0 and result["states"] is None
    assert result["basin_counts"] == [2, None, 1], result["basin_counts"]

    states = nn_entropy.estimate_split_entropy(
        data, periods=periods, soft=[0], coupling="states", state_labels=list(well)
    )
    assert len(states["states"]["states"]) == 2 and states["states"]["dropped_frames"] == 0
    for bad in (dict(soft="topology"), dict(coupling="maybe"), dict(state_labels=[0] * n)):
        try:
            nn_entropy.estimate_split_entropy(data, periods=periods, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should raise ValueError")

    fixtures = Path(__file__).resolve().parent / "fixtures"
    system = nn_entropy.load_system(str(fixtures / "test.parm7"), str(fixtures / "test.nc"))
    kinds = system.kinds
    assert system.soft_torsions and all(kinds[i] == "torsion" for i in system.soft_torsions)
    assert set(system.symmetric_rotor_torsions) <= set(system.soft_torsions)
    assert all(kinds[i] in ("torsion", "phase angle") for i in system.ring_torsions)
    print(f"split: {result['total']!r}, soft torsions {system.soft_torsions}")


def main():
    for check in (
        check_exports,
        check_closed_form_entropy,
        check_exact_constant_closed_form,
        check_theiler_window_closed_form,
        check_coordinate_entropies_sum_to_total,
        check_mie_entropy_sums_to_second_order_total,
        check_translation_invariance,
        check_periodic_metric_accepted,
        check_invalid_period_is_rejected,
        check_coordinate_options,
        check_quasi_harmonic,
        check_split,
    ):
        check()
    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
