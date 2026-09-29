//! Atom selection and file reading across topology layouts the checked-in
//! fixture does not cover: solvent before solute, solvent between solute
//! molecules, counter-ions, four-point water, non-Amber water types, and
//! NetCDF corner cases.
//!
//! Each test builds its system with the writers in `tests/common/amber.rs`.
//! Most compare a rearranged system against the same solute written alone,
//! so the expected values come from a layout the reader already handles
//! correctly rather than from a recorded run.

mod common;

use common::Rng;
use common::amber::{
    CHAIN_ATOMS, CHAIN_BONDS, CHARMM_WATER_ATOMS, CHLORIDE, NetcdfOptions, SODIUM, TIP3P_ATOMS,
    TIP4PEW_ATOMS, TIP4PEW_BONDS, Topology, WATER_BONDS, chain_frames, concat_frames, ion_frames,
    scratch_path, water_frames, write_netcdf, write_prmtop,
};
use nn_entropy::CoordinateMetric;
use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::bat_library::geometry::bond_length;

const FRAMES: usize = 12;

/// Internal coordinates and metrics for a topology/trajectory pair.
fn load(
    name: &str,
    topology: &Topology,
    frames: &[Vec<[f32; 3]>],
    options: NetcdfOptions,
) -> Result<(Vec<Vec<f64>>, Vec<CoordinateMetric>), String> {
    let top = scratch_path(&format!("{name}.parm7"));
    let traj = scratch_path(&format!("{name}.nc"));
    write_prmtop(&top, topology);
    write_netcdf(&traj, frames, options);

    let mut internal = InternalCoordinates::new(&top).map_err(|e| e.to_string())?;
    internal
        .calculate_internal_coords(&traj, usize::MAX, false)
        .map_err(|e| e.to_string())?;
    let metrics = internal.coordinate_metrics();
    Ok((internal.int_coords, metrics))
}

/// The reference: one chain molecule and nothing else.
/// `name` keeps the scratch files of concurrently running tests apart.
fn chain_alone(name: &str, chain: &Vec<Vec<[f64; 3]>>) -> (Vec<Vec<f64>>, Vec<CoordinateMetric>) {
    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    load(
        &format!("{name}_reference"),
        &topology,
        &concat_frames(&[chain]),
        NetcdfOptions::default(),
    )
    .expect("a lone solute molecule must load")
}

fn assert_same_coordinates(
    actual: &(Vec<Vec<f64>>, Vec<CoordinateMetric>),
    expected: &(Vec<Vec<f64>>, Vec<CoordinateMetric>),
    what: &str,
) {
    assert_eq!(actual.1, expected.1, "{what}: coordinate metrics differ");
    assert_eq!(
        actual.0.len(),
        expected.0.len(),
        "{what}: frame counts differ"
    );
    for (f, (a, e)) in actual.0.iter().zip(&expected.0).enumerate() {
        assert_eq!(a, e, "{what}: frame {f} internal coordinates differ");
    }
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

#[test]
fn a_lone_chain_reads_the_bonds_it_was_built_with() {
    // Control for the writers: the reader must see exactly the geometry that
    // was written. The three heavy-heavy bonds come first in the BAT list, so
    // their lengths can be checked against independently computed distances.
    let mut rng = Rng::new(1);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let (coordinates, metrics) = chain_alone("lone_chain", &chain);

    // H-C-C-C-C-H: 3 heavy bonds, a root angle, 3 angles, 3 torsions.
    assert_eq!(metrics.len(), 10, "unexpected BAT size {metrics:?}");
    assert_eq!(
        metrics
            .iter()
            .filter(|m| matches!(m, CoordinateMetric::Periodic { .. }))
            .count(),
        3
    );

    for (f, frame) in coordinates.iter().enumerate() {
        let written: Vec<[f64; 3]> = chain[f]
            .iter()
            .map(|p| [p[0] as f32 as f64, p[1] as f32 as f64, p[2] as f32 as f64])
            .collect();
        let mut expected: Vec<f64> = [(1, 2), (2, 3), (3, 4)]
            .iter()
            .map(|&(a, b)| bond_length(written[a], written[b]))
            .collect();
        let mut actual = frame[..3].to_vec();
        expected.sort_by(f64::total_cmp);
        actual.sort_by(f64::total_cmp);
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-6, "frame {f}: bond {a} vs written {e}");
        }
    }
}

#[test]
fn solvent_after_the_solute_is_ignored() {
    // The layout tleap produces, and the one the fixture uses.
    let mut rng = Rng::new(2);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 3);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    topology.push_molecule("WAT", &TIP3P_ATOMS, &WATER_BONDS);

    let actual = load(
        "solvent_after",
        &topology,
        &concat_frames(&[&chain, &water]),
        NetcdfOptions::default(),
    )
    .expect("solute-first layout must load");
    assert_same_coordinates(
        &actual,
        &chain_alone("solvent_after", &chain),
        "solvent after solute",
    );
}

// ---------------------------------------------------------------------------
// Atom ordering
// ---------------------------------------------------------------------------

#[test]
fn solvent_before_the_solute_is_ignored() {
    let mut rng = Rng::new(3);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water_a = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 3);
    let water_b = water_frames(&mut rng, FRAMES, [-8.0, 0.0, 0.0], 3);

    let mut topology = Topology::default();
    topology.push_molecule("WAT", &TIP3P_ATOMS, &WATER_BONDS);
    topology.push_molecule("WAT", &TIP3P_ATOMS, &WATER_BONDS);
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);

    let actual = load(
        "solvent_before",
        &topology,
        &concat_frames(&[&water_a, &water_b, &chain]),
        NetcdfOptions::default(),
    )
    .expect("solvent-first layout must load");
    assert_same_coordinates(
        &actual,
        &chain_alone("solvent_before", &chain),
        "solvent before solute",
    );
}

#[test]
fn solvent_between_two_solute_molecules_is_ignored() {
    let mut rng = Rng::new(4);
    let first = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 3);
    let second = chain_frames(&mut rng, FRAMES, [16.0, 0.0, 0.0]);

    // Reference: the two chains written back to back.
    let mut reference = Topology::default();
    reference.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    reference.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let expected = load(
        "two_chains",
        &reference,
        &concat_frames(&[&first, &second]),
        NetcdfOptions::default(),
    )
    .unwrap();

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    topology.push_molecule("WAT", &TIP3P_ATOMS, &WATER_BONDS);
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let actual = load(
        "solvent_between",
        &topology,
        &concat_frames(&[&first, &water, &second]),
        NetcdfOptions::default(),
    )
    .expect("solvent-between layout must load");

    assert_same_coordinates(&actual, &expected, "solvent between solutes");
}

// ---------------------------------------------------------------------------
// Solvent and ion recognition
// ---------------------------------------------------------------------------

#[test]
fn counter_ions_contribute_no_coordinates() {
    // A monatomic ion has no internal degrees of freedom, so neutralizing a
    // system must not change -- or break -- the solute's coordinates.
    let mut rng = Rng::new(5);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let na = ion_frames(&mut rng, FRAMES, [6.0, 0.0, 0.0]);
    let cl = ion_frames(&mut rng, FRAMES, [-6.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [0.0, 8.0, 0.0], 3);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    topology.push_molecule("Na+", &SODIUM, &[]);
    topology.push_molecule("Cl-", &CHLORIDE, &[]);
    topology.push_molecule("WAT", &TIP3P_ATOMS, &WATER_BONDS);

    let actual = load(
        "counter_ions",
        &topology,
        &concat_frames(&[&chain, &na, &cl, &water]),
        NetcdfOptions::default(),
    )
    .expect("a neutralized system must load");
    assert_same_coordinates(
        &actual,
        &chain_alone("counter_ions", &chain),
        "counter-ions",
    );
}

#[test]
fn four_point_water_extra_points_are_excluded() {
    // TIP4P-Ew and OPC carry a massless extra point typed EPW.
    let mut rng = Rng::new(6);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 4);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    topology.push_molecule("WAT", &TIP4PEW_ATOMS, &TIP4PEW_BONDS);

    let actual = load(
        "tip4pew",
        &topology,
        &concat_frames(&[&chain, &water]),
        NetcdfOptions::default(),
    )
    .expect("a four-point water system must load");
    assert_same_coordinates(&actual, &chain_alone("tip4pew", &chain), "four-point water");
}

#[test]
fn water_is_recognized_by_residue_name() {
    // Types OT/HT match no Amber water type, so only the residue name marks
    // these as solvent. Without it the water's own bonds, angles and torsions
    // would be silently added to the solute's entropy.
    let mut rng = Rng::new(7);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 3);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    topology.push_molecule("HOH", &CHARMM_WATER_ATOMS, &WATER_BONDS);

    let actual = load(
        "residue_water",
        &topology,
        &concat_frames(&[&chain, &water]),
        NetcdfOptions::default(),
    )
    .expect("a residue-named water system must load");
    assert_same_coordinates(
        &actual,
        &chain_alone("residue_water", &chain),
        "residue-named water",
    );
}

// ---------------------------------------------------------------------------
// Trajectory consistency
// ---------------------------------------------------------------------------

#[test]
fn a_trajectory_with_too_few_atoms_is_an_error_not_a_panic() {
    // A topology and trajectory from different systems must be diagnosed.
    let mut rng = Rng::new(8);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 3);

    let mut topology = Topology::default();
    topology.push_molecule("WAT", &TIP3P_ATOMS, &WATER_BONDS);
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);

    // Trajectory holds only the first four atoms: too short for the solute.
    let short: Vec<Vec<[f32; 3]>> = concat_frames(&[&water, &chain])
        .into_iter()
        .map(|frame| frame[..4].to_vec())
        .collect();
    let error = load("too_few_atoms", &topology, &short, NetcdfOptions::default())
        .expect_err("a mismatched trajectory must be rejected");
    assert!(
        error.contains("atom"),
        "the error should say the atom counts disagree: {error:?}"
    );
}

// ---------------------------------------------------------------------------
// NetCDF format corner cases
// ---------------------------------------------------------------------------

#[test]
fn a_trajectory_without_global_attributes_is_readable() {
    // An absent attribute list is encoded as ZERO ZERO -- eight bytes. A
    // reader that consumes only the tag misparses everything after it.
    let mut rng = Rng::new(9);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let actual = load(
        "no_global_attributes",
        &topology,
        &concat_frames(&[&chain]),
        NetcdfOptions {
            global_attributes: false,
            ..NetcdfOptions::default()
        },
    )
    .expect("a file without global attributes must load");
    assert_same_coordinates(
        &actual,
        &chain_alone("no_global_attributes", &chain),
        "no global attributes",
    );
}

#[test]
fn a_streaming_trajectory_reads_every_written_frame() {
    // numrecs = 0xFFFFFFFF marks an unfinalized (streaming) file; the frame
    // count must then come from the file length.
    let mut rng = Rng::new(10);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let actual = load(
        "streaming",
        &topology,
        &concat_frames(&[&chain]),
        NetcdfOptions {
            streaming: true,
            ..NetcdfOptions::default()
        },
    )
    .expect("a streaming file must load");
    assert_same_coordinates(
        &actual,
        &chain_alone("streaming", &chain),
        "streaming numrecs",
    );
}

#[test]
fn a_trajectory_with_extra_atoms_is_rejected() {
    // A stripped topology paired with a solvated trajectory lines up only by
    // coincidence of ordering, so any atom-count disagreement is refused.
    let mut rng = Rng::new(11);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let water = water_frames(&mut rng, FRAMES, [8.0, 0.0, 0.0], 3);

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let error = load(
        "extra_atoms",
        &topology,
        &concat_frames(&[&chain, &water]),
        NetcdfOptions::default(),
    )
    .expect_err("a trajectory with more atoms than the topology must be rejected");
    assert!(
        error.contains("9 atoms") && error.contains('6'),
        "{error:?}"
    );
}

#[test]
fn a_heavy_atom_diatomic_contributes_exactly_its_bond() {
    // A diatomic's only internal coordinate is its bond length.
    const DIATOMIC: [(&str, &str, f64, i64); 2] =
        [("CL1", "cl", 35.45, 17), ("CL2", "cl", 35.45, 17)];
    let mut rng = Rng::new(12);
    let chain = chain_frames(&mut rng, FRAMES, [0.0, 0.0, 0.0]);
    let diatomic: Vec<Vec<[f64; 3]>> = (0..FRAMES)
        .map(|_| vec![[8.0, 0.0, 0.0], [10.0 + 0.05 * rng.normal(), 0.0, 0.0]])
        .collect();

    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    topology.push_molecule("CL2", &DIATOMIC, &[(0, 1)]);
    let (coordinates, metrics) = load(
        "diatomic",
        &topology,
        &concat_frames(&[&chain, &diatomic]),
        NetcdfOptions::default(),
    )
    .expect("a system with a diatomic must load");

    let (reference, reference_metrics) = chain_alone("diatomic", &chain);
    assert_eq!(metrics.len(), reference_metrics.len() + 1);
    assert_eq!(metrics.last(), Some(&CoordinateMetric::Linear));
    for (f, frame) in coordinates.iter().enumerate() {
        assert_eq!(
            &frame[..frame.len() - 1],
            &reference[f][..],
            "frame {f} chain coordinates"
        );
        let written = diatomic[f][1][0] as f32 as f64 - 8.0;
        assert!(
            (frame[frame.len() - 1] - written).abs() < 1e-6,
            "frame {f} diatomic bond"
        );
    }
}
