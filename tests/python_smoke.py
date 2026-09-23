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

import numpy as np

import nn_entropy

EULER_MASCHERONI = 0.57721566490153

EXPECTED_EXPORTS = {
    "System",
    "load_system",
    "estimate_entropy",
    "estimate_mie_entropy",
    "estimate_coordinate_entropy",
    "estimate_coordinate_mutual_information",
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


def check_coordinate_entropies_sum_to_total():
    """The order-1 total is by definition the sum of the per-coordinate values."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 64))
    per_coordinate = nn_entropy.estimate_coordinate_entropy(data)
    total = nn_entropy.estimate_entropy(data, mie_order=1)
    assert len(per_coordinate) == 4, per_coordinate
    assert abs(sum(per_coordinate) - total) < 1e-9, (sum(per_coordinate), total)
    print(f"order-1 additivity: {total!r}")


def check_mie_entropy_sums_to_second_order_total():
    """Splitting each pairwise mutual information evenly is an algebraic
    identity, so the per-coordinate MIE values must sum to the order-2 total."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4, 64))
    per_coordinate = nn_entropy.estimate_coordinate_mie_entropy(data)
    total = nn_entropy.estimate_entropy(data, mie_order=2)
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


def main():
    for check in (
        check_exports,
        check_closed_form_entropy,
        check_coordinate_entropies_sum_to_total,
        check_mie_entropy_sums_to_second_order_total,
        check_translation_invariance,
        check_periodic_metric_accepted,
        check_invalid_period_is_rejected,
    ):
        check()
    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
