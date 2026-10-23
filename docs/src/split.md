# Split estimator

Multi-basin torsions are poorly described by a Gaussian, while bonds, angles and phase angles are close to harmonic, and a Gaussian captures all of their correlations at once. The split estimator (`--estimator split`, `estimate_split_entropy`) uses each where it fits. It divides the coordinates into a **soft** set S and a **hard** set H and writes

S = S<sub>soft</sub> + S<sub>hard</sub> + coupling

- **S<sub>soft</sub>**: the nearest-neighbor MIE over the soft coordinates (`--soft-estimator mie`, the default; `--mie-order` sets the order), or one joint nearest-neighbor estimate over all of them (`joint`, up to 8 soft coordinates).
- **S<sub>hard</sub>**: the quasi-harmonic entropy of the hard coordinates over all frames (`--hard-estimator qh`, the default), or BQH (`bqh`). Hard phase angles are centered on their circular mean (`--periodic-correlation centered`, the split default).
- **Coupling**: the dependence between the two sets, at most 0 up to sampling error. See [Coupling](#coupling).

The output lists each term before the total. Add ⟨ln J⟩ for a Cartesian entropy, as for every estimator.

## Which coordinates are soft

`--soft` (`soft=` in Python) chooses the soft set:

- **`topology`** (the default for files and for a Python `System`): the primary torsion about each rotatable bond, that is, about every bond not on a ring. With phase angles, the default, these are the coordinates of type `torsion`; methyl rotors count. Torsions about ring bonds, and every phase angle, bond and angle, are hard. This needs phase angles, because the raw secondary torsions of a rotor are as multimodal as the primary one.
- **`detect`** (the default for a raw array in Python): every periodic coordinate whose distribution has more than one basin, or is a free rotor (mean resultant length below 0.3), as found by [basin detection](#basin-detection).
- **a list** of coordinate numbers, counted from 1 on the command line (as printed by `--per-coordinate`) and from 0 in Python and Rust.

In Rust, `InternalCoordinates::soft_torsions()` gives the topological set; in Python, `System.soft_torsions`. `symmetric_rotor_torsions` lists the soft torsions about symmetric rotors (one end carries only two or more terminal atoms of one element, as in CH<sub>3</sub>, NH<sub>3</sub><sup>+</sup> and CF<sub>3</sub>), and `ring_torsions` those about ring bonds.

## Coupling

`--coupling` (`coupling=`) chooses how the dependence between the sets is accounted for:

- **`gaussian`** (the default): the Gaussian mutual information between the sets, `½ [ln|C(S∪H)| − ln|C(S)| − ln|C(H)|]`, with soft torsions entering the correlation matrix as `e^{iφ}` (`--soft-features complex`, the default) or as cos φ and sin φ (`cossin`).
- **`states`**: the hard entropy is computed within each conformational state of the soft torsions, the states' entropies are averaged with their populations, and the coupling is the difference from the hard entropy over all frames. A state is the combination of basins of the state-defining torsions: by default (`--state-torsions auto`) the soft torsions with more than one basin, excluding symmetric rotors, whose basins are equivalent; or a list. States with fewer than `--min-state-frames` frames (300 by default) or than the number of hard coordinates plus 2 are dropped and the others' populations renormalized; the output lists every state and the dropped frames, and warns when more than 5% are dropped. In Python, `state_labels` gives each frame's state directly.
- **`none`**: the sets are treated as independent.

No mixing term is added: the entropy of moving between states is already part of S<sub>soft</sub>.

## Basin detection

`detect_basins` (Rust `basins::detect_basins`, Python `nn_entropy.detect_basins`) finds the basins of a periodic coordinate:

1. Histogram the values over one period (72 bins, fewer for small samples) and smooth with a wrapped Gaussian (standard deviation 2 bins).
2. Take every circular local maximum as a peak.
3. Merge neighboring peaks whose separating minimum is at least half the lower peak.
4. Put the basin boundaries at the remaining minima.
5. Merge basins holding less than 2% of the frames into a neighbor.

The result is deterministic and does not depend on where the period starts.

## Choosing settings

On the butane end states, the defaults scored best of 96 configurations, with an RMSE of 0.063 nats against thermodynamic integration; see [Validation](validation.md#split-estimator). Butane has a single multi-basin torsion, which BQH also handles well, so plain BQH scores better there; the split estimator is meant for molecules with several coupled rotatable bonds, where the pairwise terms of the MIE and the single Gaussian of QH both fall short.

- **Per-state coupling** conditions on discrete states. It needs enough frames per state, and a Gaussian fitted to a small state is biased; on butane it was 0.03 nats worse than Gaussian coupling.
- **The joint soft estimator** has no truncation error but its bias grows with the number of soft torsions; keep it to a few.
- **Detection** can call a rotatable bond stuck in one well hard, and a stiff torsion that samples two wells soft; the topological rule cannot.

`--per-coordinate` and `--mutual-information` do not apply to the split estimator.
