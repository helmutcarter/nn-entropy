//! Writers for minimal Amber topologies (`.parm7`) and NetCDF classic
//! trajectories, so tests can build systems with any atom ordering, solvent
//! model or file-format corner case instead of relying on the single fixture.
//!
//! The writers follow the published formats directly (Amber prmtop `%FLAG`
//! sections; NetCDF classic CDF-1), not the crate's reader.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Atom {
    pub name: &'static str,
    pub amber_type: &'static str,
    pub mass: f64,
    pub atomic_number: i64,
    /// Index into `Topology::residues`.
    pub residue: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Topology {
    pub atoms: Vec<Atom>,
    /// Residue labels, in order. Every residue's atoms must be contiguous.
    pub residues: Vec<&'static str>,
    /// Zero-based atom index pairs.
    pub bonds: Vec<(usize, usize)>,
}

impl Topology {
    /// Append a molecule as one new residue, returning its atoms' indices.
    pub fn push_molecule(
        &mut self,
        residue: &'static str,
        atoms: &[(&'static str, &'static str, f64, i64)],
        bonds: &[(usize, usize)],
    ) -> Vec<usize> {
        let residue_index = self.residues.len();
        self.residues.push(residue);
        let first = self.atoms.len();
        for &(name, amber_type, mass, atomic_number) in atoms {
            self.atoms.push(Atom {
                name,
                amber_type,
                mass,
                atomic_number,
                residue: residue_index,
            });
        }
        for &(a, b) in bonds {
            self.bonds.push((first + a, first + b));
        }
        (first..first + atoms.len()).collect()
    }
}

// Building blocks -----------------------------------------------------------

/// A six-atom chain H-C-C-C-C-H. Four heavy atoms give three heavy-heavy bonds,
/// and the chain has enough atoms for angles and torsions.
pub const CHAIN_ATOMS: [(&str, &str, f64, i64); 6] = [
    ("H1", "hc", 1.008, 1),
    ("C1", "c3", 12.01, 6),
    ("C2", "c3", 12.01, 6),
    ("C3", "c3", 12.01, 6),
    ("C4", "c3", 12.01, 6),
    ("H4", "hc", 1.008, 1),
];
pub const CHAIN_BONDS: [(usize, usize); 5] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];

/// TIP3P-style water: Amber types OW/HW with the SHAKE H-H bond.
pub const TIP3P_ATOMS: [(&str, &str, f64, i64); 3] = [
    ("O", "OW", 15.999, 8),
    ("H1", "HW", 1.008, 1),
    ("H2", "HW", 1.008, 1),
];
pub const WATER_BONDS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// TIP4P-Ew-style water: a massless extra point typed `EPW`.
pub const TIP4PEW_ATOMS: [(&str, &str, f64, i64); 4] = [
    ("O", "OW", 15.999, 8),
    ("H1", "HW", 1.008, 1),
    ("H2", "HW", 1.008, 1),
    ("EPW", "EPW", 0.0, 0),
];
pub const TIP4PEW_BONDS: [(usize, usize); 4] = [(0, 1), (0, 2), (1, 2), (0, 3)];

/// Water with CHARMM-style types that no Amber type-name check recognizes;
/// only its residue name identifies it as solvent.
pub const CHARMM_WATER_ATOMS: [(&str, &str, f64, i64); 3] = [
    ("OH2", "OT", 15.999, 8),
    ("H1", "HT", 1.008, 1),
    ("H2", "HT", 1.008, 1),
];

pub const SODIUM: [(&str, &str, f64, i64); 1] = [("Na+", "Na+", 22.99, 11)];
pub const CHLORIDE: [(&str, &str, f64, i64); 1] = [("Cl-", "Cl-", 35.45, 17)];

// Coordinates ---------------------------------------------------------------

/// Ideal zig-zag positions for `CHAIN_ATOMS`, perturbed per frame.
pub fn chain_frames(rng: &mut super::Rng, frames: usize, origin: [f64; 3]) -> Vec<Vec<[f64; 3]>> {
    let ideal = [
        [-0.6, 0.9, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.4, 0.2],
        [3.5, 1.5, 0.4],
        [4.1, 0.6, 0.9],
    ];
    jitter(rng, frames, origin, &ideal)
}

pub fn water_frames(
    rng: &mut super::Rng,
    frames: usize,
    origin: [f64; 3],
    atoms: usize,
) -> Vec<Vec<[f64; 3]>> {
    let ideal = [
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
        [0.1, 0.1, 0.0],
    ];
    jitter(rng, frames, origin, &ideal[..atoms])
}

pub fn ion_frames(rng: &mut super::Rng, frames: usize, origin: [f64; 3]) -> Vec<Vec<[f64; 3]>> {
    jitter(rng, frames, origin, &[[0.0, 0.0, 0.0]])
}

fn jitter(
    rng: &mut super::Rng,
    frames: usize,
    origin: [f64; 3],
    ideal: &[[f64; 3]],
) -> Vec<Vec<[f64; 3]>> {
    (0..frames)
        .map(|_| {
            ideal
                .iter()
                .map(|p| {
                    [
                        origin[0] + p[0] + 0.05 * rng.normal(),
                        origin[1] + p[1] + 0.05 * rng.normal(),
                        origin[2] + p[2] + 0.05 * rng.normal(),
                    ]
                })
                .collect()
        })
        .collect()
}

/// Concatenate per-molecule frame blocks, in the given molecule order.
pub fn concat_frames(blocks: &[&Vec<Vec<[f64; 3]>>]) -> Vec<Vec<[f32; 3]>> {
    let frames = blocks[0].len();
    (0..frames)
        .map(|f| {
            blocks
                .iter()
                .flat_map(|block| block[f].iter())
                .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                .collect()
        })
        .collect()
}

// Files ---------------------------------------------------------------------

/// A fresh path under cargo's per-target scratch directory.
pub fn scratch_path(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("amber-fixtures");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn write_section(
    out: &mut String,
    flag: &str,
    format: &str,
    width: usize,
    per_line: usize,
    tokens: &[String],
) {
    out.push_str(&format!("%FLAG {flag}\n%FORMAT({format})\n"));
    if tokens.is_empty() {
        out.push('\n');
        return;
    }
    for line in tokens.chunks(per_line) {
        for token in line {
            out.push_str(&format!("{token:>width$}"));
        }
        out.push('\n');
    }
}

fn int_tokens(values: &[i64]) -> Vec<String> {
    values.iter().map(|v| v.to_string()).collect()
}

fn text_tokens(values: &[&str]) -> Vec<String> {
    // a4 fields are left-justified, space padded.
    values.iter().map(|v| format!("{v:<4}")).collect()
}

fn write_text_section(out: &mut String, flag: &str, values: &[&str]) {
    out.push_str(&format!("%FLAG {flag}\n%FORMAT(20a4)\n"));
    for line in values.chunks(20) {
        for token in text_tokens(line) {
            out.push_str(&token);
        }
        out.push('\n');
    }
}

/// Write a minimal prmtop containing the sections the reader consumes.
pub fn write_prmtop(path: &Path, topology: &Topology) {
    let natom = topology.atoms.len();
    let is_hydrogen = |i: usize| topology.atoms[i].atomic_number == 1;
    let with_h: Vec<(usize, usize)> = topology
        .bonds
        .iter()
        .copied()
        .filter(|&(a, b)| is_hydrogen(a) || is_hydrogen(b))
        .collect();
    let without_h: Vec<(usize, usize)> = topology
        .bonds
        .iter()
        .copied()
        .filter(|&(a, b)| !(is_hydrogen(a) || is_hydrogen(b)))
        .collect();

    let mut pointers = vec![0i64; 31];
    pointers[0] = natom as i64; // NATOM
    pointers[2] = with_h.len() as i64; // NBONH
    pointers[3] = without_h.len() as i64; // MBONA
    pointers[11] = topology.residues.len() as i64; // NRES
    pointers[12] = without_h.len() as i64; // NBONA

    // Bond arrays hold 3 * zero-based index, then a one-based type index.
    let bond_tokens = |bonds: &[(usize, usize)]| -> Vec<i64> {
        bonds
            .iter()
            .flat_map(|&(a, b)| [3 * a as i64, 3 * b as i64, 1])
            .collect()
    };

    let mut residue_pointer = Vec::new();
    for residue in 0..topology.residues.len() {
        let first = topology
            .atoms
            .iter()
            .position(|a| a.residue == residue)
            .expect("empty residue");
        residue_pointer.push(first as i64 + 1);
    }

    let mut out = String::from("%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/26  00:00:00\n");
    out.push_str("%FLAG TITLE\n%FORMAT(20a4)\nsynthetic\n");
    write_section(&mut out, "POINTERS", "10I8", 8, 10, &int_tokens(&pointers));
    let names: Vec<&str> = topology.atoms.iter().map(|a| a.name).collect();
    write_text_section(&mut out, "ATOM_NAME", &names);
    let masses: Vec<String> = topology
        .atoms
        .iter()
        .map(|a| format!("{:16.8E}", a.mass))
        .collect();
    write_section(&mut out, "MASS", "5E16.8", 16, 5, &masses);
    let atomic_numbers: Vec<i64> = topology.atoms.iter().map(|a| a.atomic_number).collect();
    write_section(
        &mut out,
        "ATOMIC_NUMBER",
        "10I8",
        8,
        10,
        &int_tokens(&atomic_numbers),
    );
    write_text_section(&mut out, "RESIDUE_LABEL", &topology.residues);
    write_section(
        &mut out,
        "RESIDUE_POINTER",
        "10I8",
        8,
        10,
        &int_tokens(&residue_pointer),
    );
    write_section(
        &mut out,
        "BONDS_INC_HYDROGEN",
        "10I8",
        8,
        10,
        &int_tokens(&bond_tokens(&with_h)),
    );
    write_section(
        &mut out,
        "BONDS_WITHOUT_HYDROGEN",
        "10I8",
        8,
        10,
        &int_tokens(&bond_tokens(&without_h)),
    );
    let types: Vec<&str> = topology.atoms.iter().map(|a| a.amber_type).collect();
    write_text_section(&mut out, "AMBER_ATOM_TYPE", &types);

    File::create(path)
        .unwrap()
        .write_all(out.as_bytes())
        .unwrap();
}

// NetCDF classic ------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct NetcdfOptions {
    /// Write a global attribute list; `false` writes the 8-byte ABSENT marker.
    pub global_attributes: bool,
    /// Write numrecs as the STREAMING sentinel 0xFFFFFFFF instead of a count.
    pub streaming: bool,
}

impl Default for NetcdfOptions {
    fn default() -> Self {
        NetcdfOptions {
            global_attributes: true,
            streaming: false,
        }
    }
}

const NC_CHAR: u32 = 2;
const NC_FLOAT: u32 = 5;
const NC_DIMENSION: u32 = 10;
const NC_VARIABLE: u32 = 11;
const NC_ATTRIBUTE: u32 = 12;

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn put_name(buf: &mut Vec<u8>, name: &str) {
    put_u32(buf, name.len() as u32);
    buf.extend_from_slice(name.as_bytes());
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
}

fn put_absent(buf: &mut Vec<u8>) {
    // ABSENT is ZERO ZERO: a zero tag followed by a zero element count.
    put_u32(buf, 0);
    put_u32(buf, 0);
}

fn put_text_attributes(buf: &mut Vec<u8>, attributes: &[(&str, &str)]) {
    if attributes.is_empty() {
        put_absent(buf);
        return;
    }
    put_u32(buf, NC_ATTRIBUTE);
    put_u32(buf, attributes.len() as u32);
    for (name, value) in attributes {
        put_name(buf, name);
        put_u32(buf, NC_CHAR);
        put_u32(buf, value.len() as u32);
        buf.extend_from_slice(value.as_bytes());
        while !buf.len().is_multiple_of(4) {
            buf.push(0);
        }
    }
}

fn netcdf_header(atoms: usize, numrecs: u32, options: NetcdfOptions, begins: [u32; 2]) -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"CDF\x01");
    put_u32(&mut h, numrecs);

    // Dimensions: frame (unlimited), spatial, atom.
    put_u32(&mut h, NC_DIMENSION);
    put_u32(&mut h, 3);
    put_name(&mut h, "frame");
    put_u32(&mut h, 0);
    put_name(&mut h, "spatial");
    put_u32(&mut h, 3);
    put_name(&mut h, "atom");
    put_u32(&mut h, atoms as u32);

    if options.global_attributes {
        put_text_attributes(
            &mut h,
            &[("Conventions", "AMBER"), ("program", "synthetic")],
        );
    } else {
        put_absent(&mut h);
    }

    // Two record variables, as Amber writes them. `time` carries no
    // attributes, exercising an ABSENT variable attribute list.
    put_u32(&mut h, NC_VARIABLE);
    put_u32(&mut h, 2);

    put_name(&mut h, "time");
    put_u32(&mut h, 1);
    put_u32(&mut h, 0); // frame
    put_absent(&mut h);
    put_u32(&mut h, NC_FLOAT);
    put_u32(&mut h, 4);
    put_u32(&mut h, begins[0]);

    put_name(&mut h, "coordinates");
    put_u32(&mut h, 3);
    put_u32(&mut h, 0); // frame
    put_u32(&mut h, 2); // atom
    put_u32(&mut h, 1); // spatial
    put_text_attributes(&mut h, &[("units", "angstrom")]);
    put_u32(&mut h, NC_FLOAT);
    put_u32(&mut h, (atoms * 12) as u32);
    put_u32(&mut h, begins[1]);
    h
}

/// Write a CDF-1 Amber-style trajectory of f32 coordinates.
pub fn write_netcdf(path: &Path, frames: &[Vec<[f32; 3]>], options: NetcdfOptions) {
    let atoms = frames[0].len();
    let numrecs = if options.streaming {
        u32::MAX
    } else {
        frames.len() as u32
    };
    // The header length does not depend on the begin offsets (fixed-width
    // fields), so size it once with placeholders.
    let header_len = netcdf_header(atoms, numrecs, options, [0, 0]).len() as u32;
    let mut out = netcdf_header(atoms, numrecs, options, [header_len, header_len + 4]);

    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.len(), atoms);
        out.extend_from_slice(&(index as f32).to_be_bytes());
        for p in frame {
            for v in p {
                out.extend_from_slice(&v.to_be_bytes());
            }
        }
    }
    File::create(path).unwrap().write_all(&out).unwrap();
}
