# Validation

This page records tests of the estimators
against entropies computed without any density estimate, so their accuracy
can be judged on real molecular data rather than only on the closed-form
distributions of the test suite.

## Butane end states

### Reference

Two states of butane differ only in their bonded potential: state 0 has the
torsion terms switched off, state 1 is ordinary butane. Both were simulated
with OpenMM 8.5 in the gas phase (no cutoff) at 320 K with a Langevin
integrator (2 fs, friction 1/ps) and the X-H bonds constrained.

Thermodynamic integration and MBAR over the linear path
U(λ) = (1 - λ) U0 + λ U1, in 21 windows of 10 ns each, give the free energy
difference ΔA. The mean potential energies of the end states give ΔU, and
T ΔS = ΔU - ΔA. Because the masses do not change, the momentum entropy
cancels, and ΔS is the difference in configurational entropy. The metric
term of the constraints is -0.0003 nats, negligible. Three replicates
started from independent frames with independent random seeds:

| replicate | ΔA (kcal/mol) | ΔU (kcal/mol) | S0 - S1 (nats) |
|---|---|---|---|
| 1 | 10.592 ± 0.006 | 1.550 ± 0.051 | 14.219 ± 0.080 |
| 2 | 10.613 ± 0.006 | 1.438 ± 0.051 | 14.429 ± 0.081 |
| 3 | 10.619 ± 0.006 | 1.509 ± 0.059 | 14.325 ± 0.093 |
| mean ± sd | | | 14.32 ± 0.11 |

### Estimates

Each estimate uses the 10,000 frames (one per ps) of the λ = 0 and λ = 1
windows of the same replicate. With the X-H bonds constrained, the default
`CoordinateOptions` give exactly
the 26 unconstrained, nonredundant coordinates of butane: 3 C-C bonds, 12
angles and 11 torsions. The reference is a Cartesian entropy, so every
estimate below includes the change in the mean log BAT Jacobian,
Δ⟨Σ 2 ln b + Σ ln sin θ⟩ over the unconstrained bonds and the angles, about
-0.06 nats; the constrained bonds contribute a constant that cancels.

`S0 - S1` in nats. Mean error and RMSE are against each replicate's own
reference.

| coordinates | estimator | rep 1 | rep 2 | rep 3 | mean error | sd | RMSE |
|---|---|---|---|---|---|---|---|
| | **reference** | 14.219 | 14.429 | 14.325 | | | |
| phase angles | **BQH** | 14.238 | 14.480 | 14.348 | **+0.031** | 0.017 | **0.034** |
| phase angles | BQH, centered torsions | 14.214 | 14.459 | 14.327 | +0.009 | 0.019 | 0.018 |
| phase angles | BQH, 100-bin histogram | 14.165 | 14.340 | 14.226 | -0.081 | 0.024 | 0.083 |
| phase angles | NN MIE-2 | 14.541 | 13.950 | 14.160 | -0.107 | 0.403 | 0.346 |
| phase angles | QH | 13.388 | 13.505 | 13.408 | -0.891 | 0.051 | 0.892 |
| raw torsions | QH | 13.730 | 13.651 | 13.620 | -0.657 | 0.150 | 0.669 |
| raw torsions | NN MIE-2 | 16.315 | 15.446 | 17.879 | +2.222 | 1.273 | 2.453 |
| raw torsions | BQH, 100-bin histogram | 17.132 | 17.331 | 17.167 | +2.886 | 0.038 | 2.886 |
| raw torsions | BQH | 17.189 | 17.446 | 17.147 | +2.936 | 0.102 | 2.938 |
| raw torsions | BQH, centered torsions | 20.175 | 16.012 | 15.787 | +3.000 | 2.560 | 3.657 |

BQH is `estimate_quasi_harmonic_entropy`
with the default options (nearest-neighbor marginals, complex torsions);
"centered torsions" is `PeriodicCorrelation::Centered`
and the histogram is `MarginalEstimator::Histogram`.
NN MIE-2 is `calculate_entropy` at order 2.
"Raw torsions" sets `CoordinateOptions::phase_angles`
to false.

### Split estimator

The [split estimator](split.md) with the topological soft set, on the same trajectories (`S0 - S1` in nats). Its defaults are the best of 96 configurations scored here: soft estimator {MIE-2, joint with either finite-sample constant} × hard estimator {QH, BQH} × hard phase angles {centered, complex} × coupling {per-state, Gaussian e^{iφ}, Gaussian cos/sin, none} × soft set {topology, detected}.

| estimator | rep 1 | rep 2 | rep 3 | mean error | RMSE |
|---|---|---|---|---|---|
| **split** (MIE-2 soft, QH hard, Gaussian coupling) | 14.215 | 14.499 | 14.407 | +0.050 | 0.063 |
| split, per-state coupling | 14.246 | 14.529 | 14.440 | +0.081 | 0.089 |
| split, joint soft, BQH hard, no coupling | 14.220 | 14.557 | 14.368 | +0.057 | 0.078 |

On butane the coupling between soft and hard coordinates is small (Gaussian and no coupling differ by 0.001 nats), detection chooses the same soft set as topology, and the finite-sample constant cancels in the difference. Centered hard phase angles beat complex ones by about 0.02 nats, and per-state coupling is about 0.03 nats worse than Gaussian coupling, from the bias of a Gaussian fitted within each state. The split estimator improves on an earlier hand-tuned version of the same decomposition (soft torsions by a joint estimate or MIE-2, hard Gaussian within hand-binned states: +0.06 and +0.14 nats), but plain BQH is better on butane, whose single multi-basin torsion BQH's nearest-neighbor marginals already describe.

### What it shows

- **BQH with phase angles agrees with the reference within the reference's
  own uncertainty** (about 0.08 nats per replicate), and follows its
  variation from replicate to replicate. The centered and complex torsion
  treatments are indistinguishable here: with phase angles only four
  torsions remain free, and their correlations are weak.
- **QH misses by 0.9 nats.** It shares BQH's correlation term, so the
  difference is its Gaussian first-order term, which cannot represent the
  multiwell torsions of state 1.
- **NN MIE-2 is right on average but noisy**, ±0.4 nats between replicates:
  the pairwise mutual informations it sums each carry estimator error.
- **Phase angles matter for every estimator.** With the raw torsions, the
  three hydrogen torsions of each methyl group are redundant; every
  estimator is off by 0.7 to 3.7 nats.

These are estimates for one small molecule. They show which estimators work
on it, not an error bar for other systems.

### Reproducing

`examples/butane_validation.rs` computes every estimate in the table from
the end-state trajectories of each replicate:

```bash
cargo run --release --example butane_validation -- \
    rep1/endpoints 14.219 rep2/endpoints 14.429 rep3/endpoints 14.325
```

Each directory holds `0.0.top`, `0.0.nc`, `1.0.top` and `1.0.nc`; the number
after it is that replicate's reference `S0 - S1`.
