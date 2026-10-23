//! Coordinate-set options: the tree bond set, phase angles, and the BAT
//! Jacobian.
//!
//! Each test builds a small molecule with the writers in `tests/common/amber.rs`.
//! Expected values come from the constructed geometry or from counting rules
//! (3N - 6, spanning trees), and phase angles are checked against torsions read
//! without the option, never against a recorded run.

mod common;

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use common::Rng;
use common::amber::{
    CHAIN_ATOMS, CHAIN_BONDS, NetcdfOptions, Topology, chain_frames, concat_frames, scratch_path,
    write_netcdf, write_prmtop,
};
use nn_entropy::bat_library::{BondSet, CoordinateKind, CoordinateOptions, InternalCoordinates};

const FRAMES: usize = 40;

const ETHANE_ATOMS: [(&str, &str, f64, i64); 8] = [
    ("C1", "c3", 12.01, 6),
    ("C2", "c3", 12.01, 6),
    ("H11", "hc", 1.008, 1),
    ("H12", "hc", 1.008, 1),
    ("H13", "hc", 1.008, 1),
    ("H21", "hc", 1.008, 1),
    ("H22", "hc", 1.008, 1),
    ("H23", "hc", 1.008, 1),
];
const ETHANE_BONDS: [(usize, usize); 7] = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)];

const CYCLOPROPANE_ATOMS: [(&str, &str, f64, i64); 9] = [
    ("C1", "cx", 12.01, 6),
    ("C2", "cx", 12.01, 6),
    ("C3", "cx", 12.01, 6),
    ("H11", "hc", 1.008, 1),
    ("H12", "hc", 1.008, 1),
    ("H21", "hc", 1.008, 1),
    ("H22", "hc", 1.008, 1),
    ("H31", "hc", 1.008, 1),
    ("H32", "hc", 1.008, 1),
];
const CYCLOPROPANE_BONDS: [(usize, usize); 9] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (0, 4),
    (1, 5),
    (1, 6),
    (2, 7),
    (2, 8),
];

const HCL_ATOMS: [(&str, &str, f64, i64); 2] = [("H", "h1", 1.008, 1), ("Cl", "cl", 35.45, 17)];
const HCL_BONDS: [(usize, usize); 1] = [(0, 1)];

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Wrap to (-pi, pi] through atan2, independently of the crate's wrapping.
fn wrap(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

/// Ethane along x with tetrahedral methyls. The second methyl is rotated about
/// the C-C axis by a random angle each frame, so the H-C-C-H torsions sweep the
/// whole circle while the offsets between hydrogens on one carbon stay fixed.
fn ethane_frames(rng: &mut Rng) -> Vec<Vec<[f64; 3]>> {
    let c1 = [0.0, 0.0, 0.0];
    let c2 = [1.53, 0.0, 0.0];
    let radial = (8.0_f64 / 9.0).sqrt() * 1.09;
    let axial = 1.09 / 3.0;
    let jitter = |rng: &mut Rng, p: [f64; 3]| {
        [
            p[0] + 0.01 * rng.normal(),
            p[1] + 0.01 * rng.normal(),
            p[2] + 0.01 * rng.normal(),
        ]
    };
    (0..FRAMES)
        .map(|_| {
            let rotation = rng.uniform_range(0.0, 2.0 * PI);
            let mut atoms = vec![jitter(rng, c1), jitter(rng, c2)];
            for k in 0..3 {
                let phi = 2.0 * PI * k as f64 / 3.0;
                let h = [-axial, radial * phi.cos(), radial * phi.sin()];
                atoms.push(jitter(rng, add(c1, h)));
            }
            for k in 0..3 {
                let phi = 2.0 * PI * k as f64 / 3.0 + rotation;
                let h = [axial, radial * phi.cos(), radial * phi.sin()];
                atoms.push(jitter(rng, add(c2, h)));
            }
            atoms
        })
        .collect()
}

fn cyclopropane_frames(rng: &mut Rng) -> Vec<Vec<[f64; 3]>> {
    let ideal: [[f64; 3]; 9] = [
        [0.0, 0.87, 0.0],
        [-0.75, -0.43, 0.0],
        [0.75, -0.43, 0.0],
        [0.0, 1.45, 0.9],
        [0.0, 1.45, -0.9],
        [-1.25, -0.72, 0.9],
        [-1.25, -0.72, -0.9],
        [1.25, -0.72, 0.9],
        [1.25, -0.72, -0.9],
    ];
    (0..FRAMES)
        .map(|_| {
            ideal
                .iter()
                .map(|p| {
                    [
                        p[0] + 0.03 * rng.normal(),
                        p[1] + 0.03 * rng.normal(),
                        p[2] + 0.03 * rng.normal(),
                    ]
                })
                .collect()
        })
        .collect()
}

fn hcl_frames(rng: &mut Rng) -> Vec<Vec<[f64; 3]>> {
    (0..FRAMES)
        .map(|_| vec![[0.0, 0.0, 0.0], [1.27 + 0.02 * rng.normal(), 0.0, 0.0]])
        .collect()
}

/// Write one molecule's topology and trajectory and return their paths, along
/// with the positions as stored (f32, like the trajectory).
fn write_system(
    name: &str,
    atoms: &[(&'static str, &'static str, f64, i64)],
    bonds: &[(usize, usize)],
    frames: &Vec<Vec<[f64; 3]>>,
) -> (PathBuf, PathBuf, Vec<Vec<[f64; 3]>>) {
    let mut topology = Topology::default();
    topology.push_molecule("MOL", atoms, bonds);
    let top = scratch_path(&format!("coordinates_{name}.parm7"));
    let traj = scratch_path(&format!("coordinates_{name}.nc"));
    write_prmtop(&top, &topology);
    let stored = concat_frames(&[frames]);
    write_netcdf(&traj, &stored, NetcdfOptions::default());
    let positions = stored
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect()
        })
        .collect();
    (top, traj, positions)
}

fn load(
    top: &Path,
    traj: &Path,
    bonds: BondSet,
    phase_angles: bool,
    torsions_only: bool,
) -> InternalCoordinates {
    let options = CoordinateOptions {
        bonds,
        phase_angles,
        ..Default::default()
    };
    let mut internal = InternalCoordinates::with_options(top, options).unwrap();
    internal
        .calculate_internal_coords(traj, usize::MAX, torsions_only)
        .unwrap();
    internal
}

fn ethane(name: &str) -> (PathBuf, PathBuf, Vec<Vec<[f64; 3]>>) {
    let mut rng = Rng::new(0x5eed_e7a2);
    write_system(name, &ETHANE_ATOMS, &ETHANE_BONDS, &ethane_frames(&mut rng))
}

fn cyclopropane(name: &str) -> (PathBuf, PathBuf, Vec<Vec<[f64; 3]>>) {
    let mut rng = Rng::new(0xc3c3_c3c3);
    write_system(
        name,
        &CYCLOPROPANE_ATOMS,
        &CYCLOPROPANE_BONDS,
        &cyclopropane_frames(&mut rng),
    )
}

/// Spread of an angle series on the circle: sqrt(-2 ln R), R the mean
/// resultant length. Near 0 for a stiff angle, large for a uniform one.
fn circular_spread(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let c = values.iter().map(|v| v.cos()).sum::<f64>() / n;
    let s = values.iter().map(|v| v.sin()).sum::<f64>() / n;
    (-2.0 * (c * c + s * s).sqrt().ln()).sqrt()
}

fn column(internal: &InternalCoordinates, index: usize) -> Vec<f64> {
    internal
        .int_coords
        .iter()
        .map(|frame| frame[index])
        .collect()
}

// ---------------------------------------------------------------------------
// Tree bond set
// ---------------------------------------------------------------------------

#[test]
fn tree_bonds_give_three_n_minus_six_coordinates() {
    for (name, (top, traj, _), n_atoms) in [
        ("tree_count_ethane", ethane("tree_count_ethane"), 8),
        (
            "tree_count_cyclopropane",
            cyclopropane("tree_count_cyclopropane"),
            9,
        ),
    ] {
        let internal = load(&top, &traj, BondSet::Tree, false, false);
        let atoms = internal.coordinate_atoms();
        let count = |k: usize| atoms.iter().filter(|a| a.len() == k).count();
        assert_eq!(atoms.len(), 3 * n_atoms - 6, "{name}: coordinate count");
        assert_eq!(count(2), n_atoms - 1, "{name}: bond count");
        assert_eq!(count(3), n_atoms - 2, "{name}: angle count");
        assert_eq!(count(4), n_atoms - 3, "{name}: torsion count");
        assert_eq!(internal.int_coords[0].len(), 3 * n_atoms - 6);
    }
}

#[test]
fn tree_bonds_form_a_spanning_tree_of_topology_bonds() {
    let (top, traj, _) = cyclopropane("tree_spanning");
    let internal = load(&top, &traj, BondSet::Tree, false, false);
    let topology_bonds: HashSet<(usize, usize)> = CYCLOPROPANE_BONDS
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    let tree: Vec<(usize, usize)> = internal
        .coordinate_atoms()
        .into_iter()
        .filter(|a| a.len() == 2)
        .map(|a| (a[0].min(a[1]), a[0].max(a[1])))
        .collect();

    for bond in &tree {
        assert!(topology_bonds.contains(bond), "{bond:?} is not a bond");
    }
    // N - 1 edges that connect all N atoms form a tree, so exactly one ring
    // bond is left out.
    let n_atoms = CYCLOPROPANE_ATOMS.len();
    assert_eq!(tree.len(), n_atoms - 1);
    let mut component: Vec<usize> = (0..n_atoms).collect();
    fn root(component: &mut [usize], mut a: usize) -> usize {
        while component[a] != a {
            a = component[a];
        }
        a
    }
    for &(a, b) in &tree {
        let (ra, rb) = (root(&mut component, a), root(&mut component, b));
        assert_ne!(ra, rb, "tree bond ({a}, {b}) closes a cycle");
        component[ra] = rb;
    }
}

#[test]
fn tree_bonds_include_bonds_to_hydrogen() {
    let (top, traj, _) = ethane("tree_hydrogen");
    let heavy = load(&top, &traj, BondSet::HeavyAtom, false, false);
    let tree = load(&top, &traj, BondSet::Tree, false, false);
    let bonds = |internal: &InternalCoordinates| {
        internal
            .coordinate_atoms()
            .into_iter()
            .filter(|a| a.len() == 2)
            .count()
    };
    assert_eq!(bonds(&heavy), 1, "the heavy-atom set keeps only C-C");
    assert_eq!(bonds(&tree), 7, "the tree set keeps every ethane bond");
}

#[test]
fn tree_bonds_leave_angles_and_torsions_unchanged() {
    let (top, traj, _) = cyclopropane("tree_same_angles");
    let heavy = load(&top, &traj, BondSet::HeavyAtom, false, false);
    let tree = load(&top, &traj, BondSet::Tree, false, false);
    let non_bonds = |internal: &InternalCoordinates| {
        let atoms = internal.coordinate_atoms();
        let keep: Vec<usize> = (0..atoms.len()).filter(|&i| atoms[i].len() > 2).collect();
        let values: Vec<Vec<f64>> = internal
            .int_coords
            .iter()
            .map(|frame| keep.iter().map(|&i| frame[i]).collect())
            .collect();
        let atoms: Vec<Vec<usize>> = keep.iter().map(|&i| atoms[i].clone()).collect();
        (atoms, values)
    };
    assert_eq!(non_bonds(&heavy), non_bonds(&tree));
}

#[test]
fn a_diatomic_with_hydrogen_keeps_its_bond_only_in_the_tree_set() {
    let mut rng = Rng::new(17);
    let (top, traj, positions) =
        write_system("tree_hcl", &HCL_ATOMS, &HCL_BONDS, &hcl_frames(&mut rng));
    let heavy = load(&top, &traj, BondSet::HeavyAtom, false, false);
    assert!(heavy.coordinate_atoms().is_empty());

    let tree = load(&top, &traj, BondSet::Tree, false, false);
    assert_eq!(tree.coordinate_atoms(), vec![vec![0, 1]]);
    for (frame, values) in positions.iter().zip(&tree.int_coords) {
        let expected = norm(sub(frame[1], frame[0]));
        assert!((values[0] - expected).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Phase angles
// ---------------------------------------------------------------------------

#[test]
fn phase_references_point_to_the_first_torsion_about_the_same_bond() {
    let (top, traj, _) = ethane("phase_references");
    let internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    let atoms = internal.coordinate_atoms();
    let references = internal.phase_references();
    assert_eq!(references.len(), atoms.len());

    let mut first_about: HashMap<(usize, usize), usize> = HashMap::new();
    let mut converted = 0;
    for (index, entry) in atoms.iter().enumerate() {
        if entry.len() != 4 {
            assert_eq!(
                references[index], None,
                "coordinate {index} is not a torsion"
            );
            continue;
        }
        let bond = (entry[1].min(entry[2]), entry[1].max(entry[2]));
        let first = *first_about.entry(bond).or_insert(index);
        let expected = (first != index).then_some(first);
        assert_eq!(references[index], expected, "torsion {index}");
        converted += usize::from(expected.is_some());
    }
    // Ethane's five BAT torsions are all about the C-C bond.
    assert_eq!(converted, 4);

    let plain = load(&top, &traj, BondSet::HeavyAtom, false, false);
    assert!(plain.phase_references().iter().all(Option::is_none));
}

#[test]
fn phase_angles_are_wrapped_offsets_of_the_raw_torsions() {
    let (top, traj, _) = ethane("phase_values");
    let raw = load(&top, &traj, BondSet::HeavyAtom, false, false);
    let phase = load(&top, &traj, BondSet::HeavyAtom, true, false);
    let references = phase.phase_references();

    for (raw_frame, phase_frame) in raw.int_coords.iter().zip(&phase.int_coords) {
        for (index, reference) in references.iter().enumerate() {
            match *reference {
                None => assert_eq!(phase_frame[index], raw_frame[index], "coordinate {index}"),
                Some(reference) => {
                    let value = phase_frame[index];
                    let expected = wrap(raw_frame[index] - raw_frame[reference]);
                    assert!((value - expected).abs() < 1e-12, "coordinate {index}");
                    assert!(value > -PI && value <= PI, "{value} outside (-pi, pi]");
                }
            }
        }
    }
}

#[test]
fn phase_angles_of_a_rigid_rotor_are_stiff() {
    let (top, traj, _) = ethane("phase_stiff");
    let raw = load(&top, &traj, BondSet::HeavyAtom, false, false);
    let phase = load(&top, &traj, BondSet::HeavyAtom, true, false);

    for (index, reference) in phase.phase_references().iter().enumerate() {
        let Some(reference) = *reference else {
            continue;
        };
        // The methyl rotation is uniform, so the raw torsions are spread out.
        assert!(circular_spread(&column(&raw, index)) > 1.0);
        assert!(circular_spread(&column(&raw, reference)) > 1.0);

        // Hydrogens on one tetrahedral carbon sit 120 degrees apart, so every
        // offset is a multiple of 2 pi / 3, up to the positional jitter.
        let offsets = column(&phase, index);
        assert!(
            circular_spread(&offsets) < 0.1,
            "phase angle {index} is not stiff"
        );
        for offset in offsets {
            let thirds = offset / (2.0 * PI / 3.0);
            assert!(
                (thirds - thirds.round()).abs() * 2.0 * PI / 3.0 < 0.1,
                "offset {offset} is not a multiple of 2 pi / 3"
            );
        }
    }
}

#[test]
fn phase_angles_with_torsions_only_match_the_full_set() {
    let (top, traj, _) = ethane("phase_torsions_only");
    let full = load(&top, &traj, BondSet::HeavyAtom, true, false);
    let torsions = load(&top, &traj, BondSet::HeavyAtom, true, true);

    let atoms = full.coordinate_atoms();
    let keep: Vec<usize> = (0..atoms.len()).filter(|&i| atoms[i].len() == 4).collect();
    assert_eq!(torsions.coordinate_atoms().len(), keep.len());
    assert_eq!(torsions.phase_references().iter().flatten().count(), 4);
    for (full_frame, torsion_frame) in full.int_coords.iter().zip(&torsions.int_coords) {
        let expected: Vec<f64> = keep.iter().map(|&i| full_frame[i]).collect();
        assert_eq!(torsion_frame, &expected);
    }
}

// ---------------------------------------------------------------------------
// BAT Jacobian
// ---------------------------------------------------------------------------

#[test]
fn log_jacobian_matches_the_constructed_geometry() {
    let (top, traj, positions) = cyclopropane("jacobian_geometry");
    let internal = load(&top, &traj, BondSet::Tree, false, false);
    let atoms = internal.coordinate_atoms();
    let log_jacobian = internal.log_jacobian().unwrap();
    assert_eq!(log_jacobian.len(), FRAMES);

    for (frame, value) in positions.iter().zip(log_jacobian) {
        let mut expected = 0.0;
        for entry in &atoms {
            match *entry.as_slice() {
                [a, b] => expected += 2.0 * norm(sub(frame[a], frame[b])).ln(),
                [a, b, c] => {
                    let (u, v) = (sub(frame[a], frame[b]), sub(frame[c], frame[b]));
                    let cos = dot(u, v) / (norm(u) * norm(v));
                    expected += (1.0 - cos * cos).sqrt().ln();
                }
                _ => {}
            }
        }
        // The reader subtracts f32 positions before widening, so agreement is
        // to f32 rounding of the bond vectors, not to f64 precision.
        assert!((value - expected).abs() < 1e-5, "{value} vs {expected}");
    }
}

#[test]
fn phase_angles_leave_the_jacobian_unchanged() {
    let (top, traj, _) = ethane("jacobian_phase");
    let plain = load(&top, &traj, BondSet::Tree, false, false);
    let phase = load(&top, &traj, BondSet::Tree, true, false);
    assert_eq!(plain.log_jacobian().unwrap(), phase.log_jacobian().unwrap());
}

#[test]
fn log_jacobian_needs_the_full_tree_set() {
    let (top, traj, _) = ethane("jacobian_refused");
    let heavy = load(&top, &traj, BondSet::HeavyAtom, false, false);
    let err = heavy.log_jacobian().unwrap_err().to_string();
    assert!(err.contains("tree bond set"), "{err}");

    let torsions = load(&top, &traj, BondSet::Tree, false, true);
    let err = torsions.log_jacobian().unwrap_err().to_string();
    assert!(err.contains("torsions only"), "{err}");
}

#[test]
fn coordinate_pairs_follow_the_mutual_information_order() {
    let (top, traj, _) = ethane("pairs");
    let mut internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    internal.coordinate_pairs();
    let n = internal.dim;
    let expected: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    assert_eq!(internal.pairs, expected);
    assert_eq!(internal.pairs.len(), n * (n - 1) / 2);
}

#[test]
fn columns_select_and_transpose_frames() {
    let (top, traj, _) = ethane("columns");
    let internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    for (start, stride) in [(0, 1), (3, 1), (2, 5), (FRAMES - 1, 7)] {
        let columns = internal.columns(start, stride).unwrap();
        assert_eq!(columns.len(), internal.dim);
        let frames: Vec<usize> = (start..FRAMES).step_by(stride).collect();
        for (coordinate, column) in columns.iter().enumerate() {
            let expected: Vec<f64> = frames
                .iter()
                .map(|&f| internal.int_coords[f][coordinate])
                .collect();
            assert_eq!(column, &expected, "start {start}, stride {stride}");
        }
    }
}

#[test]
fn columns_refuse_what_they_cannot_select() {
    let (top, traj, _) = ethane("columns_refused");
    let internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    let message = |start, stride| internal.columns(start, stride).unwrap_err().to_string();
    assert_eq!(message(0, 0), "stride must be at least 1");
    assert_eq!(
        message(FRAMES, 1),
        format!("start {FRAMES} is beyond the {FRAMES} frames read")
    );

    let mut none = InternalCoordinates::new(&top).unwrap();
    none.calculate_internal_coords(&traj, 0, false).unwrap();
    assert_eq!(
        none.columns(0, 1).unwrap_err().to_string(),
        "no frames were read from the trajectory"
    );

    // HCl has no coordinate under the heavy-atom bond set.
    let mut rng = Rng::new(17);
    let (top, traj, _) = write_system("columns_hcl", &HCL_ATOMS, &HCL_BONDS, &hcl_frames(&mut rng));
    let hcl = load(&top, &traj, BondSet::HeavyAtom, false, false);
    assert_eq!(
        hcl.columns(0, 1).unwrap_err().to_string(),
        "no internal coordinates were generated"
    );
}

#[test]
fn coordinate_kinds_follow_atoms_and_phase_references() {
    let (top, traj, _) = ethane("kinds");
    for phase_angles in [true, false] {
        let internal = load(&top, &traj, BondSet::Tree, phase_angles, false);
        let references = internal.phase_references();
        for ((kind, atoms), reference) in internal
            .coordinate_kinds()
            .into_iter()
            .zip(internal.coordinate_atoms())
            .zip(references)
        {
            let expected = match (atoms.len(), reference) {
                (2, _) => CoordinateKind::Bond,
                (3, _) => CoordinateKind::Angle,
                (_, Some(reference)) => CoordinateKind::PhaseAngle { reference },
                _ => CoordinateKind::Torsion,
            };
            assert_eq!(kind, expected);
        }
    }
    assert_eq!(
        CoordinateKind::PhaseAngle { reference: 3 }.name(),
        "phase angle"
    );
}

// ---------------------------------------------------------------------------
// Ring and rotor topology
// ---------------------------------------------------------------------------

/// Methylcyclopropane: a three-membered ring with a methyl group on C1.
const METHYLCYCLOPROPANE_ATOMS: [(&str, &str, f64, i64); 12] = [
    ("C1", "cx", 12.01, 6),
    ("C2", "cx", 12.01, 6),
    ("C3", "cx", 12.01, 6),
    ("C4", "c3", 12.01, 6),
    ("H1", "hc", 1.008, 1),
    ("H21", "hc", 1.008, 1),
    ("H22", "hc", 1.008, 1),
    ("H31", "hc", 1.008, 1),
    ("H32", "hc", 1.008, 1),
    ("H41", "hc", 1.008, 1),
    ("H42", "hc", 1.008, 1),
    ("H43", "hc", 1.008, 1),
];
const METHYLCYCLOPROPANE_BONDS: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (0, 4),
    (1, 5),
    (1, 6),
    (2, 7),
    (2, 8),
    (3, 9),
    (3, 10),
    (3, 11),
];

fn methylcyclopropane(name: &str) -> (PathBuf, PathBuf, Vec<Vec<[f64; 3]>>) {
    let ideal: [[f64; 3]; 12] = [
        [0.0, 0.87, 0.0],
        [-0.75, -0.43, 0.0],
        [0.75, -0.43, 0.0],
        [0.0, 2.3, 0.4],
        [0.0, 1.2, -0.95],
        [-1.25, -0.72, 0.9],
        [-1.25, -0.72, -0.9],
        [1.25, -0.72, 0.9],
        [1.25, -0.72, -0.9],
        [0.9, 2.7, 0.0],
        [-0.9, 2.7, 0.0],
        [0.0, 2.5, 1.45],
    ];
    let mut rng = Rng::new(0x3c3e);
    let frames = (0..FRAMES)
        .map(|_| {
            ideal
                .iter()
                .map(|p| {
                    [
                        p[0] + 0.03 * rng.normal(),
                        p[1] + 0.03 * rng.normal(),
                        p[2] + 0.03 * rng.normal(),
                    ]
                })
                .collect()
        })
        .collect();
    write_system(
        name,
        &METHYLCYCLOPROPANE_ATOMS,
        &METHYLCYCLOPROPANE_BONDS,
        &frames,
    )
}

fn central_bond(internal: &InternalCoordinates, index: usize) -> (usize, usize) {
    let atoms = &internal.coordinate_atoms()[index];
    (atoms[1].min(atoms[2]), atoms[1].max(atoms[2]))
}

#[test]
fn ethane_has_one_soft_symmetric_rotor() {
    let (top, traj, _) = ethane("topology_ethane");
    let internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    let soft = internal.soft_torsions();
    assert_eq!(soft.len(), 1);
    assert_eq!(central_bond(&internal, soft[0]), (0, 1));
    assert_eq!(
        internal.coordinate_kinds()[soft[0]],
        CoordinateKind::Torsion
    );
    assert_eq!(internal.symmetric_rotor_torsions(), soft);
    assert!(internal.ring_torsions().is_empty());
}

#[test]
fn ring_torsions_are_never_soft() {
    let (top, traj, _) = cyclopropane("topology_cyclopropane");
    let internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    assert!(internal.soft_torsions().is_empty());
    let torsions: Vec<usize> = (0..internal.dim)
        .filter(|&i| internal.coordinate_atoms()[i].len() == 4)
        .collect();
    assert_eq!(internal.ring_torsions(), torsions);
}

#[test]
fn an_exocyclic_methyl_is_soft_and_the_ring_is_not() {
    let (top, traj, _) = methylcyclopropane("topology_methylcyclopropane");
    let internal = load(&top, &traj, BondSet::HeavyAtom, true, false);
    let soft = internal.soft_torsions();
    assert_eq!(soft.len(), 1, "{:?}", internal.coordinate_atoms());
    assert_eq!(central_bond(&internal, soft[0]), (0, 3));
    assert_eq!(internal.symmetric_rotor_torsions(), soft);
    let ring = internal.ring_torsions();
    assert!(!ring.is_empty());
    for &i in &ring {
        let bond = central_bond(&internal, i);
        assert!(bond.0 <= 2 && bond.1 <= 2, "{bond:?} is not a ring bond");
    }
    // Torsions only keeps the classification.
    let torsions = load(&top, &traj, BondSet::HeavyAtom, true, true);
    assert_eq!(torsions.soft_torsions().len(), 1);
    assert_eq!(torsions.ring_torsions().len(), ring.len());
}

#[test]
fn a_chain_has_one_soft_torsion_per_rotatable_bond() {
    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let top = scratch_path("topology_chain.parm7");
    let traj = scratch_path("topology_chain.nc");
    write_prmtop(&top, &topology);
    let mut rng = Rng::new(0xc4a1);
    write_netcdf(
        &traj,
        &concat_frames(&[&chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0])]),
        NetcdfOptions::default(),
    );
    for phase_angles in [true, false] {
        let internal = load(&top, &traj, BondSet::HeavyAtom, phase_angles, false);
        let soft = internal.soft_torsions();
        let bonds: HashSet<(usize, usize)> =
            soft.iter().map(|&i| central_bond(&internal, i)).collect();
        assert_eq!(bonds, HashSet::from([(1, 2), (2, 3), (3, 4)]));
        assert_eq!(soft.len(), 3);
        assert!(internal.symmetric_rotor_torsions().is_empty());
    }
}
