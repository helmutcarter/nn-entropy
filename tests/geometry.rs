//! Internal-coordinate geometry, checked against configurations whose bond
//! lengths, angles and torsions are known by construction.
//!
//! Every expected value below is derived from the geometry that builds the
//! input, not read off a previous run. The torsion tests also pin the sign
//! convention, which is a property callers need to know and which no other test
//! in the suite records.

mod common;

use std::path::PathBuf;

use nn_entropy::bat_library::InternalCoordinates;
use nn_entropy::{
    CoordinateMetric, calc_angle, calc_bond, calc_internal_coords, calc_torsion, cross_product,
};

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let difference = (actual - expected).abs();
    // Inclusive so a tolerance of exactly 0.0 asserts bitwise equality.
    assert!(
        difference <= tolerance,
        "{what}: got {actual}, expected {expected}, difference {difference} exceeds {tolerance}"
    );
}

// ---------------------------------------------------------------------------
// Bonds
// ---------------------------------------------------------------------------

#[test]
fn bond_lengths_match_constructed_distances() {
    assert_close(
        calc_bond([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        1.0,
        1e-15,
        "unit separation along x",
    );
    // 3-4-5 right triangle, then the 3-4-12-13 Pythagorean quadruple.
    assert_close(
        calc_bond([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]),
        5.0,
        1e-15,
        "3-4-5",
    );
    assert_close(
        calc_bond([0.0, 0.0, 0.0], [3.0, 4.0, 12.0]),
        13.0,
        1e-15,
        "3-4-12-13",
    );
    // Translation invariance of a distance.
    assert_close(
        calc_bond([10.5, -3.25, 7.0], [13.5, 0.75, 19.0]),
        13.0,
        1e-13,
        "translated 3-4-12-13",
    );
    // Coincident atoms give exactly zero, not a NaN.
    assert_close(
        calc_bond([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        0.0,
        0.0,
        "coincident atoms",
    );
}

// ---------------------------------------------------------------------------
// Angles
// ---------------------------------------------------------------------------

#[test]
fn angles_match_constructed_geometry() {
    let origin = [0.0, 0.0, 0.0];

    // Right angle between the x and y axes.
    assert_close(
        calc_angle([1.0, 0.0, 0.0], origin, [0.0, 1.0, 0.0]),
        std::f64::consts::FRAC_PI_2,
        1e-15,
        "90 degrees",
    );

    // Equilateral triangle: the apex subtends 60 degrees.
    let equilateral_apex = [0.5, 3.0_f64.sqrt() / 2.0, 0.0];
    assert_close(
        calc_angle(origin, equilateral_apex, [1.0, 0.0, 0.0]),
        std::f64::consts::FRAC_PI_3,
        1e-15,
        "60 degrees",
    );

    // Ideal tetrahedral angle, acos(-1/3) ~ 109.47 degrees: two vertices of a
    // regular tetrahedron seen from its center.
    assert_close(
        calc_angle([1.0, 1.0, 1.0], origin, [1.0, -1.0, -1.0]),
        (-1.0_f64 / 3.0).acos(),
        1e-15,
        "tetrahedral",
    );

    // Angle magnitude does not depend on the arm lengths.
    assert_close(
        calc_angle([7.0, 0.0, 0.0], origin, [0.0, 0.125, 0.0]),
        std::f64::consts::FRAC_PI_2,
        1e-15,
        "90 degrees with unequal arms",
    );
}

#[test]
fn collinear_and_parallel_arms_stay_finite() {
    let origin = [0.0, 0.0, 0.0];

    // Exactly antiparallel arms: the cosine is -1 and the angle is pi. This is
    // the configuration where an unclamped acos would be at risk of NaN, so the
    // assertion is that the result is finite as well as correct.
    let straight = calc_angle([-2.0, 0.0, 0.0], origin, [5.0, 0.0, 0.0]);
    assert!(straight.is_finite(), "collinear angle was {straight}");
    assert_close(straight, std::f64::consts::PI, 1e-12, "180 degrees");

    // Exactly parallel arms: cosine +1, angle 0.
    let folded = calc_angle([2.0, 0.0, 0.0], origin, [5.0, 0.0, 0.0]);
    assert!(folded.is_finite(), "parallel angle was {folded}");
    assert_close(folded, 0.0, 1e-12, "0 degrees");

    // A near-linear arrangement, as an sp-hybridized centre would produce.
    let nearly = calc_angle([-3.0, 1e-7, 0.0], origin, [4.0, 0.0, 0.0]);
    assert!(nearly.is_finite(), "near-linear angle was {nearly}");
    assert!(
        (nearly - std::f64::consts::PI).abs() < 1e-6,
        "near-linear angle {nearly} should be just under pi"
    );
}

#[test]
fn angles_lie_in_the_closed_zero_pi_interval() {
    let mut rng = common::Rng::new(41);
    for _ in 0..500 {
        let a = [rng.normal(), rng.normal(), rng.normal()];
        let b = [rng.normal(), rng.normal(), rng.normal()];
        let c = [rng.normal(), rng.normal(), rng.normal()];
        let angle = calc_angle(a, b, c);
        assert!(
            angle.is_finite() && (0.0..=std::f64::consts::PI).contains(&angle),
            "angle {angle} outside [0, pi] for {a:?} {b:?} {c:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Torsions
// ---------------------------------------------------------------------------

/// Four atoms whose IUPAC dihedral about the a2->a3 axis is exactly `phi`.
///
/// a2 sits at the origin and a3 one unit along z, so the axis is z. a1 lies
/// along x, giving the reference half-plane; a4 is placed at angle `phi` from
/// it in the xy-plane.
fn torsion_configuration(phi: f64) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    (
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [phi.cos(), phi.sin(), 1.0],
    )
}

#[test]
fn torsion_returns_the_negated_iupac_dihedral() {
    // The crate builds its bond vectors as a1-a2, a2-a3, a3-a4, which are the
    // negatives of the conventional b1, b2, b3. The odd-order triple product
    // flips sign while the even-order term does not, so `calc_torsion` returns
    // -phi where phi is the IUPAC dihedral.
    //
    // Entropy is invariant under reflection, so this does not affect any
    // estimate -- but a torsion value exported for comparison against cpptraj
    // or MDAnalysis will carry the opposite sign. This test exists to pin that
    // convention so a future change to it cannot pass unnoticed.
    for phi in [0.0_f64, 0.5, 1.0, 2.0, 3.0, -0.5, -2.5] {
        let (a1, a2, a3, a4) = torsion_configuration(phi);
        assert_close(
            calc_torsion(a1, a2, a3, a4),
            -phi,
            1e-12,
            &format!("torsion at phi = {phi}"),
        );
    }
}

#[test]
fn canonical_conformers_give_canonical_torsions() {
    // cis (syn) is 0, trans (anti) is pi, and the two gauche wells sit at
    // +/- pi/3 from cis in the IUPAC convention.
    let (a1, a2, a3, a4) = torsion_configuration(0.0);
    assert_close(calc_torsion(a1, a2, a3, a4), 0.0, 1e-12, "cis");

    let (a1, a2, a3, a4) = torsion_configuration(std::f64::consts::PI);
    assert_close(
        calc_torsion(a1, a2, a3, a4).abs(),
        std::f64::consts::PI,
        1e-12,
        "trans",
    );

    let (a1, a2, a3, a4) = torsion_configuration(std::f64::consts::FRAC_PI_3);
    assert_close(
        calc_torsion(a1, a2, a3, a4),
        -std::f64::consts::FRAC_PI_3,
        1e-12,
        "gauche+",
    );
}

#[test]
fn torsion_is_antisymmetric_under_reflection() {
    // Mirroring the whole configuration reverses the chirality of the dihedral.
    for phi in [0.4_f64, 1.3, 2.9] {
        let (a1, a2, a3, a4) = torsion_configuration(phi);
        let mirror = |p: [f64; 3]| [p[0], -p[1], p[2]];
        assert_close(
            calc_torsion(mirror(a1), mirror(a2), mirror(a3), mirror(a4)),
            -calc_torsion(a1, a2, a3, a4),
            1e-12,
            &format!("reflected torsion at phi = {phi}"),
        );
    }
}

#[test]
fn torsions_lie_in_the_atan2_range() {
    let mut rng = common::Rng::new(42);
    for _ in 0..500 {
        let p = |rng: &mut common::Rng| [rng.normal(), rng.normal(), rng.normal()];
        let (a, b, c, d) = (p(&mut rng), p(&mut rng), p(&mut rng), p(&mut rng));
        let torsion = calc_torsion(a, b, c, d);
        assert!(
            torsion.is_finite() && torsion.abs() <= std::f64::consts::PI,
            "torsion {torsion} outside (-pi, pi]"
        );
    }
}

#[test]
fn cross_product_is_antisymmetric_and_orthogonal() {
    let mut rng = common::Rng::new(43);
    for _ in 0..200 {
        let a = [rng.normal(), rng.normal(), rng.normal()];
        let b = [rng.normal(), rng.normal(), rng.normal()];
        let axb = cross_product(a, b);
        let bxa = cross_product(b, a);
        for k in 0..3 {
            assert_close(axb[k], -bxa[k], 1e-12, "antisymmetry");
        }
        let dot_a: f64 = (0..3).map(|k| axb[k] * a[k]).sum();
        let dot_b: f64 = (0..3).map(|k| axb[k] * b[k]).sum();
        assert!(dot_a.abs() < 1e-12 && dot_b.abs() < 1e-12, "not orthogonal");
    }
}

// ---------------------------------------------------------------------------
// BAT list dispatch
// ---------------------------------------------------------------------------

#[test]
fn internal_coordinate_dispatch_follows_entry_length() {
    // A two-atom entry is a bond, three is an angle, four is a torsion.
    let frame = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    let bat_list = vec![vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]];
    let coordinates = calc_internal_coords(bat_list, vec![frame.clone()]);

    assert_eq!(coordinates.len(), 1);
    assert_eq!(coordinates[0].len(), 3);
    assert_close(coordinates[0][0], 1.0, 1e-15, "bond entry");
    assert_close(
        coordinates[0][1],
        std::f64::consts::FRAC_PI_2,
        1e-15,
        "angle entry",
    );
    assert_close(
        coordinates[0][2],
        calc_torsion(frame[0], frame[1], frame[2], frame[3]),
        1e-15,
        "torsion entry",
    );
}

// ---------------------------------------------------------------------------
// End-to-end: the production BAT path on the checked-in fixture
// ---------------------------------------------------------------------------

fn fixture_internal_coordinates() -> InternalCoordinates {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut internal = InternalCoordinates::new(&root.join("tests/fixtures/test.parm7"))
        .expect("failed to build BAT list from topology");
    internal
        .calculate_internal_coords(&root.join("tests/fixtures/test.nc"), usize::MAX, false)
        .expect("failed to read trajectory or compute BAT coordinates");
    internal
}

#[test]
fn fixture_coordinates_are_physically_plausible() {
    // This exercises the whole production path -- prmtop parsing, BAT list
    // construction, the NetCDF reader and the f32 geometry kernels -- none of
    // which the unit tests above reach.
    //
    // The checks are physical rather than numerical: a bond that came out at
    // 30 angstrom, or an angle outside [0, pi], would mean atom indices had
    // drifted out of step with the coordinate buffer. That is the failure mode
    // that produces plausible-looking but wrong entropies.
    let internal = fixture_internal_coordinates();
    let metrics = internal.coordinate_metrics();
    assert!(!internal.int_coords.is_empty(), "no frames were read");
    assert_eq!(internal.int_coords[0].len(), metrics.len());

    let mut periodic_seen = 0usize;
    let mut linear_seen = 0usize;

    for (frame_index, frame) in internal.int_coords.iter().enumerate() {
        assert_eq!(
            frame.len(),
            metrics.len(),
            "frame {frame_index} has wrong width"
        );
        for (coordinate, (&value, &metric)) in frame.iter().zip(&metrics).enumerate() {
            assert!(
                value.is_finite(),
                "frame {frame_index} coordinate {coordinate} is not finite: {value}"
            );
            match metric {
                CoordinateMetric::Periodic { period } => {
                    assert_close(period, std::f64::consts::TAU, 0.0, "torsion period");
                    assert!(
                        value.abs() <= std::f64::consts::PI,
                        "frame {frame_index} torsion {coordinate} = {value} outside (-pi, pi]"
                    );
                    periodic_seen += 1;
                }
                CoordinateMetric::Linear => {
                    // Bonds and angles share this branch. Bond lengths for the
                    // elements in this fixture stay well under 2 A, and angles
                    // cannot exceed pi, so a single upper bound of 3.2 catches
                    // a misaligned atom index without needing to tell the two
                    // kinds apart.
                    assert!(
                        value > 0.0 && value < 3.2,
                        "frame {frame_index} coordinate {coordinate} = {value} is not a \
                         plausible bond length or angle"
                    );
                    linear_seen += 1;
                }
            }
        }
    }

    assert!(periodic_seen > 0, "fixture produced no torsions");
    assert!(linear_seen > 0, "fixture produced no bonds or angles");
}

#[test]
fn torsions_only_selects_exactly_the_periodic_coordinates() {
    let full = fixture_internal_coordinates();
    let full_torsions = full
        .coordinate_metrics()
        .iter()
        .filter(|m| matches!(m, CoordinateMetric::Periodic { .. }))
        .count();

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut torsions_only =
        InternalCoordinates::new(&root.join("tests/fixtures/test.parm7")).unwrap();
    torsions_only
        .calculate_internal_coords(&root.join("tests/fixtures/test.nc"), usize::MAX, true)
        .unwrap();
    let metrics = torsions_only.coordinate_metrics();

    assert_eq!(
        metrics.len(),
        full_torsions,
        "torsions-only should keep exactly the torsion coordinates"
    );
    assert!(
        metrics
            .iter()
            .all(|m| matches!(m, CoordinateMetric::Periodic { .. })),
        "torsions-only left a non-periodic coordinate behind"
    );
    assert_eq!(
        torsions_only.int_coords.len(),
        full.int_coords.len(),
        "frame count should not change"
    );
}

#[test]
fn frame_limit_truncates_without_changing_earlier_frames() {
    // `--stop` is a count of frames read, so a smaller limit must return a
    // strict prefix of the full trajectory.
    let full = fixture_internal_coordinates();
    assert!(
        full.int_coords.len() >= 4,
        "fixture is too short to truncate"
    );
    let limit = full.int_coords.len() / 2;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut limited = InternalCoordinates::new(&root.join("tests/fixtures/test.parm7")).unwrap();
    limited
        .calculate_internal_coords(&root.join("tests/fixtures/test.nc"), limit, false)
        .unwrap();

    assert_eq!(limited.int_coords.len(), limit);
    for (frame_index, frame) in limited.int_coords.iter().enumerate() {
        assert_eq!(
            frame, &full.int_coords[frame_index],
            "frame {frame_index} differs between the limited and full reads"
        );
    }
}
