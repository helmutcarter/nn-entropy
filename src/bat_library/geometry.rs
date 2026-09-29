//! Bond, angle and torsion geometry used to turn Cartesian frames into BAT
//! internal coordinates.
//!
//! The f64 functions are public so they can be tested directly; the f32 variants
//! read NetCDF coordinates without an intermediate copy.

/// Distance between two atoms.
pub fn bond_length(a1: [f64; 3], a2: [f64; 3]) -> f64 {
    let dx = a1[0] - a2[0];
    let dy = a1[1] - a2[1];
    let dz = a1[2] - a2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Angle a1-a2-a3 in radians, in [0, pi].
///
/// Uses atan2(|v1 x v2|, v1 . v2) rather than acos of the normalized dot
/// product: rounding can push that ratio just outside [-1, 1] for a straight
/// angle, where acos returns NaN, and acos also loses precision near 0 and pi.
/// Coincident atoms still give NaN so input validation rejects them.
pub fn bond_angle(a1: [f64; 3], a2: [f64; 3], a3: [f64; 3]) -> f64 {
    let v1 = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]];
    let v2 = [a3[0] - a2[0], a3[1] - a2[1], a3[2] - a2[2]];
    let v1_squared = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let v2_squared = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
    if v1_squared == 0.0 || v2_squared == 0.0 {
        return f64::NAN;
    }
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let cross_mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    cross_mag.atan2(dot)
}

/// Torsion a1-a2-a3-a4 in radians, in (-pi, pi]. The sign is the negative of
/// the IUPAC dihedral convention.
pub fn torsion_angle(a1: [f64; 3], a2: [f64; 3], a3: [f64; 3], a4: [f64; 3]) -> f64 {
    let b1 = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]];
    let b2 = [a2[0] - a3[0], a2[1] - a3[1], a2[2] - a3[2]];
    let b3 = [a3[0] - a4[0], a3[1] - a4[1], a3[2] - a4[2]];

    let c1 = [
        b2[1] * b3[2] - b2[2] * b3[1],
        b2[2] * b3[0] - b2[0] * b3[2],
        b2[0] * b3[1] - b2[1] * b3[0],
    ];
    let c2 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ];

    let p1 = (b1[0] * c1[0] + b1[1] * c1[1] + b1[2] * c1[2])
        * (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
    let p2 = c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2];

    p1.atan2(p2)
}

fn bond_length_f32(a1: [f32; 3], a2: [f32; 3]) -> f64 {
    let dx = a1[0] - a2[0];
    let dy = a1[1] - a2[1];
    let dz = a1[2] - a2[2];
    let sum = (dx as f64) * (dx as f64) + (dy as f64) * (dy as f64) + (dz as f64) * (dz as f64);
    sum.sqrt()
}

fn bond_angle_f32(a1: [f32; 3], a2: [f32; 3], a3: [f32; 3]) -> f64 {
    bond_angle(a1.map(f64::from), a2.map(f64::from), a3.map(f64::from))
}

fn torsion_angle_f32(a1: [f32; 3], a2: [f32; 3], a3: [f32; 3], a4: [f32; 3]) -> f64 {
    let b1 = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]];
    let b2 = [a2[0] - a3[0], a2[1] - a3[1], a2[2] - a3[2]];
    let b3 = [a3[0] - a4[0], a3[1] - a4[1], a3[2] - a4[2]];

    let c1 = [
        b2[1] * b3[2] - b2[2] * b3[1],
        b2[2] * b3[0] - b2[0] * b3[2],
        b2[0] * b3[1] - b2[1] * b3[0],
    ];
    let c2 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ];

    let p1_f32 = b1[0] * c1[0] + b1[1] * c1[1] + b1[2] * c1[2];
    let b2_sum = b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2];
    let p1 = (p1_f32 as f64) * (b2_sum as f64).sqrt();
    let p2_f32 = c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2];

    p1.atan2(p2_f32 as f64)
}

/// Evaluates every BAT entry on every frame: a two-atom entry is a bond length,
/// three atoms a bond angle and four atoms a torsion.
pub fn internal_coordinates(bat_list: &[Vec<usize>], traj: &[Vec<[f64; 3]>]) -> Vec<Vec<f64>> {
    let frame_number = traj.len();
    let int_coord_number = bat_list.len();
    let mut int_coords = vec![vec![0.0f64; int_coord_number]; frame_number];

    for i in 0..frame_number {
        for j in 0..int_coord_number {
            let entry = &bat_list[j];
            if entry.len() == 2 {
                int_coords[i][j] = bond_length(traj[i][entry[0]], traj[i][entry[1]]);
            } else if entry.len() == 3 {
                int_coords[i][j] =
                    bond_angle(traj[i][entry[0]], traj[i][entry[1]], traj[i][entry[2]]);
            } else if entry.len() == 4 {
                int_coords[i][j] = torsion_angle(
                    traj[i][entry[0]],
                    traj[i][entry[1]],
                    traj[i][entry[2]],
                    traj[i][entry[3]],
                );
            }
        }
    }

    int_coords
}

pub(super) fn internal_coordinates_f32(
    bat_list: &[Vec<usize>],
    traj: &[Vec<[f32; 3]>],
) -> Vec<Vec<f64>> {
    let frame_number = traj.len();
    let int_coord_number = bat_list.len();
    let mut int_coords = vec![vec![0.0f64; int_coord_number]; frame_number];

    for i in 0..frame_number {
        for j in 0..int_coord_number {
            let entry = &bat_list[j];
            if entry.len() == 2 {
                int_coords[i][j] = bond_length_f32(traj[i][entry[0]], traj[i][entry[1]]);
            } else if entry.len() == 3 {
                int_coords[i][j] =
                    bond_angle_f32(traj[i][entry[0]], traj[i][entry[1]], traj[i][entry[2]]);
            } else if entry.len() == 4 {
                int_coords[i][j] = torsion_angle_f32(
                    traj[i][entry[0]],
                    traj[i][entry[1]],
                    traj[i][entry[2]],
                    traj[i][entry[3]],
                );
            }
        }
    }

    int_coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn a_straight_angle_is_pi_not_nan() {
        // Before the atan2 form, rounding pushed cos just below -1 for triples
        // like this one (taken from a random search) and acos returned NaN.
        let a1 = [9.532_804_5_f32, 17.210_114, 25.240_194];
        let a2 = [9.169_646_f32, 16.730_146, 24.347_971];
        let a3 = [8.564_333_f32, 15.930_14, 22.860_813];
        let angle = bond_angle_f32(a1, a2, a3);
        assert!((angle - PI).abs() < 1e-3, "got {angle}");

        let wide = |p: [f32; 3]| p.map(f64::from);
        let angle = bond_angle(wide(a1), wide(a2), wide(a3));
        assert!((angle - PI).abs() < 1e-3, "got {angle}");
    }

    #[test]
    fn straight_angles_along_many_directions_are_finite() {
        // A deterministic sweep of exactly collinear triples. With the old acos
        // form roughly a quarter of these came out NaN in f64 and half in f32.
        for i in 0..2000 {
            let t = i as f64 * 0.618_033_988_75;
            let direction = [
                t.cos() * (2.0 * t).sin(),
                t.sin() * (2.0 * t).sin(),
                (2.0 * t).cos(),
            ];
            let center = [10.0 + t % 7.0, 20.0 - t % 5.0, 30.0 + t % 3.0];
            let (l1, l2) = (1.0 + (t % 1.0), 1.5 - (t % 0.5));
            let a1 = [0, 1, 2].map(|k| center[k] - l1 * direction[k]);
            let a3 = [0, 1, 2].map(|k| center[k] + l2 * direction[k]);

            let angle = bond_angle(a1, center, a3);
            assert!((angle - PI).abs() < 1e-6, "f64 sample {i}: got {angle}");

            let narrow = |p: [f64; 3]| p.map(|x| x as f32);
            let angle = bond_angle_f32(narrow(a1), narrow(center), narrow(a3));
            assert!((angle - PI).abs() < 1e-3, "f32 sample {i}: got {angle}");
        }
    }

    #[test]
    fn angles_match_known_geometry() {
        let origin = [0.0, 0.0, 0.0];
        let x = [1.0, 0.0, 0.0];
        let cases = [
            ([2.0, 0.0, 0.0], 0.0),
            ([0.0, 3.0, 0.0], PI / 2.0),
            ([0.5, 3.0_f64.sqrt() / 2.0, 0.0], PI / 3.0),
            ([-1.0, 1.0, 0.0], 3.0 * PI / 4.0),
            ([-4.0, 0.0, 0.0], PI),
        ];
        for (other, expected) in cases {
            let angle = bond_angle(x, origin, other);
            assert!(
                (angle - expected).abs() < 1e-12,
                "{other:?}: {angle} vs {expected}"
            );
            let angle = bond_angle_f32(
                x.map(|v| v as f32),
                origin.map(|v| v as f32),
                other.map(|v| v as f32),
            );
            assert!(
                (angle - expected).abs() < 1e-6,
                "f32 {other:?}: {angle} vs {expected}"
            );
        }
    }

    #[test]
    fn a_nearly_straight_angle_keeps_its_precision() {
        // acos(cos(pi - d)) cannot resolve d much below 1e-8; atan2 can.
        let deviation = 1e-9_f64;
        let a1 = [1.0, 0.0, 0.0];
        let a3 = [-deviation.cos(), deviation.sin(), 0.0];
        let angle = bond_angle(a1, [0.0; 3], a3);
        assert!(
            ((PI - angle) - deviation).abs() < 1e-15,
            "got pi - {}",
            PI - angle
        );
    }

    #[test]
    fn coincident_atoms_give_nan_so_validation_rejects_them() {
        let p = [1.0, 2.0, 3.0];
        assert!(bond_angle(p, p, [4.0, 5.0, 6.0]).is_nan());
        assert!(bond_angle_f32([1.0; 3], [2.0; 3], [2.0; 3]).is_nan());
    }
}
