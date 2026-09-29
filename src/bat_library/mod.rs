use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::CoordinateMetric;

pub mod geometry;

use geometry::{internal_coordinates, internal_coordinates_f32};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct Prmtop {
    natom: usize,
    masses: Vec<f64>,
    atom_types: Vec<String>,
    /// Per-atom atomic numbers; empty when the topology predates ATOMIC_NUMBER.
    atomic_numbers: Vec<i64>,
    /// Per-atom residue labels; empty when RESIDUE_LABEL/POINTER are absent.
    residue_labels: Vec<String>,
    bonds: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct FormatSpec {
    count: usize,
    width: usize,
    kind: char,
}

fn parse_format_spec(line: &str) -> Result<FormatSpec> {
    let start = line.find('(').ok_or("missing '(' in %FORMAT")? + 1;
    let end = line.find(')').ok_or("missing ')' in %FORMAT")?;
    let inner = line[start..end].trim();
    let mut digits = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else {
            let count = digits.parse::<usize>()?;
            let kind = c;
            let mut width_digits = String::new();
            for c2 in chars {
                if c2.is_ascii_digit() {
                    width_digits.push(c2);
                } else {
                    break;
                }
            }
            let width = width_digits.parse::<usize>()?;
            return Ok(FormatSpec { count, width, kind });
        }
    }
    Err("invalid %FORMAT line".into())
}

fn parse_fixed_width_tokens(line: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let end = (i + width).min(bytes.len());
        let chunk = &bytes[i..end];
        let token = String::from_utf8_lossy(chunk).trim().to_string();
        if !token.is_empty() {
            tokens.push(token);
        }
        i += width;
    }
    tokens
}

fn parse_prmtop(path: &Path) -> Result<Prmtop> {
    let text = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut sections: HashMap<String, (FormatSpec, Vec<String>)> = HashMap::new();

    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("%FLAG") {
            let flag = line
                .split_whitespace()
                .nth(1)
                .ok_or("missing flag name")?
                .to_string();
            i += 1;
            if i >= lines.len() {
                return Err("unexpected end after %FLAG".into());
            }
            let fmt_line = lines[i].trim();
            if !fmt_line.starts_with("%FORMAT") {
                return Err("expected %FORMAT after %FLAG".into());
            }
            let fmt = parse_format_spec(fmt_line)?;
            i += 1;
            let mut tokens = Vec::new();
            while i < lines.len() && !lines[i].trim().starts_with("%FLAG") {
                tokens.extend(parse_fixed_width_tokens(lines[i], fmt.width));
                i += 1;
            }
            sections.insert(flag, (fmt, tokens));
        } else {
            i += 1;
        }
    }

    let pointers = sections
        .get("POINTERS")
        .ok_or("missing POINTERS section")?
        .1
        .iter()
        .map(|s| s.parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if pointers.len() < 32 {
        let mut padded = pointers;
        padded.resize(32, 0);
        return parse_prmtop_from_pointers(padded, &sections);
    }
    parse_prmtop_from_pointers(pointers, &sections)
}

fn parse_prmtop_from_pointers(
    pointers: Vec<i64>,
    sections: &HashMap<String, (FormatSpec, Vec<String>)>,
) -> Result<Prmtop> {
    let natom = pointers[0] as usize;
    let nbonh = pointers[2] as usize;
    let mbona = pointers[3] as usize;
    let nbona = pointers[12] as usize;

    let masses = sections
        .get("MASS")
        .ok_or("missing MASS section")?
        .1
        .iter()
        .take(natom)
        .map(|s| s.parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let atom_types = if let Some(sec) = sections.get("AMBER_ATOM_TYPE") {
        sec.1
            .iter()
            .take(natom)
            .map(|s| s.trim().to_string())
            .collect()
    } else if let Some(sec) = sections.get("ATOM_NAME") {
        sec.1
            .iter()
            .take(natom)
            .map(|s| s.trim().to_string())
            .collect()
    } else {
        return Err("missing AMBER_ATOM_TYPE and ATOM_NAME sections".into());
    };

    let bond_h_tokens = sections
        .get("BONDS_INC_HYDROGEN")
        .map(|s| s.1.clone())
        .unwrap_or_default();
    let bond_no_h_tokens = sections
        .get("BONDS_WITHOUT_HYDROGEN")
        .map(|s| s.1.clone())
        .unwrap_or_default();

    let bond_h_count = if bond_h_tokens.len() >= nbonh * 3 {
        nbonh
    } else {
        bond_h_tokens.len() / 3
    };
    let bond_no_h_count = if bond_no_h_tokens.len() >= nbona * 3 {
        nbona
    } else if bond_no_h_tokens.len() >= mbona * 3 {
        mbona
    } else {
        bond_no_h_tokens.len() / 3
    };

    let atomic_numbers = match sections.get("ATOMIC_NUMBER") {
        Some(sec) => sec
            .1
            .iter()
            .take(natom)
            .map(|s| s.parse::<i64>())
            .collect::<std::result::Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };

    let residue_labels = match (
        sections.get("RESIDUE_LABEL"),
        sections.get("RESIDUE_POINTER"),
    ) {
        (Some(labels), Some(pointers)) => {
            let starts = pointers
                .1
                .iter()
                .map(|s| s.parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if starts.len() != labels.1.len() {
                return Err("RESIDUE_LABEL and RESIDUE_POINTER lengths differ".into());
            }
            let mut per_atom = vec![String::new(); natom];
            for (residue, &start) in starts.iter().enumerate() {
                // RESIDUE_POINTER is one-based; a residue runs to the next start.
                let first = start
                    .checked_sub(1)
                    .ok_or("RESIDUE_POINTER entry of zero")?;
                let end = starts.get(residue + 1).map_or(natom, |&next| next - 1);
                if first > end || end > natom {
                    return Err("RESIDUE_POINTER is out of order or out of range".into());
                }
                for label in &mut per_atom[first..end] {
                    label.clone_from(&labels.1[residue]);
                }
            }
            per_atom
        }
        _ => Vec::new(),
    };

    let mut bonds = Vec::new();
    for i in 0..bond_h_count {
        let a = bond_h_tokens[i * 3].parse::<i64>()?;
        let b = bond_h_tokens[i * 3 + 1].parse::<i64>()?;
        let ai = (a.unsigned_abs() as usize) / 3;
        let bi = (b.unsigned_abs() as usize) / 3;
        bonds.push((ai, bi));
    }
    for i in 0..bond_no_h_count {
        let a = bond_no_h_tokens[i * 3].parse::<i64>()?;
        let b = bond_no_h_tokens[i * 3 + 1].parse::<i64>()?;
        let ai = (a.unsigned_abs() as usize) / 3;
        let bi = (b.unsigned_abs() as usize) / 3;
        bonds.push((ai, bi));
    }

    Ok(Prmtop {
        natom,
        masses,
        atom_types,
        atomic_numbers,
        residue_labels,
        bonds,
    })
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NetcdfDim {
    name: String,
    len: u64,
}

#[derive(Debug, Clone)]
struct NetcdfVar {
    name: String,
    dim_ids: Vec<usize>,
    vartype: u32,
    vsize: u64,
    begin: u64,
    is_record: bool,
}

#[allow(dead_code)]
struct NetcdfReader {
    file: File,
    /// Atom count the topology declares; the trajectory must match it.
    expected_atoms: usize,
    version: u8,
    numrecs: u64,
    dims: Vec<NetcdfDim>,
    vars: Vec<NetcdfVar>,
    record_size: u64,
}

/// NetCDF `numrecs` value for a file still being written.
const STREAMING_NUMRECS: u32 = u32::MAX;

impl NetcdfReader {
    fn open(path: &Path, expected_atoms: usize) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic[0..3] != b"CDF" {
            return Err("not a NetCDF classic file".into());
        }
        let version = magic[3];
        if version != 1 && version != 2 {
            return Err("unsupported NetCDF version".into());
        }
        let declared_records = Self::read_u32(&mut file)?;

        let dims = Self::read_dim_list(&mut file)?;
        Self::skip_attr_list(&mut file)?;
        let vars = Self::read_var_list(&mut file, version, &dims)?;
        let record_size: u64 = vars.iter().filter(|v| v.is_record).map(|v| v.vsize).sum();

        // 0xFFFFFFFF marks a streaming (unfinalized) file whose record count
        // was never written back; count the complete records actually present.
        let numrecs = if declared_records == STREAMING_NUMRECS {
            let first_record = vars
                .iter()
                .filter(|v| v.is_record)
                .map(|v| v.begin)
                .min()
                .ok_or("streaming NetCDF file has no record variables")?;
            if record_size == 0 {
                return Err("streaming NetCDF file has zero-sized records".into());
            }
            let file_len = file.metadata()?.len();
            file_len.saturating_sub(first_record) / record_size
        } else {
            declared_records as u64
        };

        Ok(NetcdfReader {
            file,
            expected_atoms,
            version,
            numrecs,
            dims,
            vars,
            record_size,
        })
    }

    fn coordinates_var(&self) -> Result<NetcdfVar> {
        self.vars
            .iter()
            .find(|v| v.name == "coordinates")
            .cloned()
            .ok_or("coordinates variable not found".into())
    }

    fn coordinates_vartype(&self) -> Result<u32> {
        Ok(self.coordinates_var()?.vartype)
    }

    /// Read `atoms` (sorted, zero-based indices into the file's atom
    /// dimension) from the first `frames` frames, decoding each value with
    /// `decode`. Only the span from the first to the last requested atom is
    /// read, so a solute stored ahead of its solvent costs no more than before.
    fn read_coordinates<T: Copy + Default>(
        &mut self,
        frames: usize,
        atoms: &[usize],
        bytes_per_value: usize,
        decode: impl Fn(&[u8]) -> T,
    ) -> Result<Vec<Vec<[T; 3]>>> {
        let var = self.coordinates_var()?;
        let dims: Vec<u64> = var.dim_ids.iter().map(|&i| self.dims[i].len).collect();

        let (frame_count, atom_dim, spatial_dim) = if var.is_record {
            if dims.len() != 3 {
                return Err("coordinates variable must be (frame, atom, spatial)".into());
            }
            (frames.min(self.numrecs as usize), dims[1], dims[2])
        } else {
            if dims.len() != 2 {
                return Err("coordinates variable must be (atom, spatial)".into());
            }
            (1usize, dims[0], dims[1])
        };
        if spatial_dim != 3 {
            return Err("coordinates spatial dimension is not 3".into());
        }
        if atom_dim != self.expected_atoms as u64 {
            return Err(format!(
                "trajectory has {atom_dim} atoms but the topology has {}",
                self.expected_atoms
            )
            .into());
        }
        let (Some(&first), Some(&last)) = (atoms.first(), atoms.last()) else {
            return Ok(vec![Vec::new(); frame_count]);
        };
        debug_assert!(atoms.windows(2).all(|w| w[0] < w[1]));
        if last as u64 >= atom_dim {
            return Err(
                format!("topology selects atom {last} but the trajectory has {atom_dim}").into(),
            );
        }

        let atom_bytes = 3 * bytes_per_value;
        let mut span = vec![0u8; (last - first + 1) * atom_bytes];
        let mut frames_out = Vec::with_capacity(frame_count);
        for frame_idx in 0..frame_count {
            let frame_start = if var.is_record {
                var.begin + (frame_idx as u64) * self.record_size
            } else {
                var.begin
            };
            self.file
                .seek(SeekFrom::Start(frame_start + (first * atom_bytes) as u64))?;
            self.file.read_exact(&mut span)?;

            let frame = atoms
                .iter()
                .map(|&atom| {
                    let at = (atom - first) * atom_bytes;
                    let mut xyz = [T::default(); 3];
                    for (axis, value) in xyz.iter_mut().enumerate() {
                        let offset = at + axis * bytes_per_value;
                        *value = decode(&span[offset..offset + bytes_per_value]);
                    }
                    xyz
                })
                .collect();
            frames_out.push(frame);
        }
        Ok(frames_out)
    }

    fn read_coordinates_f64(
        &mut self,
        frames: usize,
        atoms: &[usize],
    ) -> Result<Vec<Vec<[f64; 3]>>> {
        match self.coordinates_vartype()? {
            5 => self.read_coordinates(frames, atoms, 4, |b| {
                f32::from_be_bytes(b.try_into().unwrap()) as f64
            }),
            6 => self.read_coordinates(frames, atoms, 8, |b| {
                f64::from_be_bytes(b.try_into().unwrap())
            }),
            _ => Err("unsupported coordinates data type".into()),
        }
    }

    fn read_coordinates_f32(
        &mut self,
        frames: usize,
        atoms: &[usize],
    ) -> Result<Vec<Vec<[f32; 3]>>> {
        if self.coordinates_vartype()? != 5 {
            return Err("coordinates variable is not float".into());
        }
        self.read_coordinates(frames, atoms, 4, |b| {
            f32::from_be_bytes(b.try_into().unwrap())
        })
    }

    fn read_dim_list(file: &mut File) -> Result<Vec<NetcdfDim>> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
            Self::read_absent_count(file)?;
            return Ok(Vec::new());
        }
        if tag != 10 {
            return Err("unexpected tag in dimension list".into());
        }
        let count = Self::read_u32(file)? as usize;
        let mut dims = Vec::with_capacity(count);
        for _ in 0..count {
            let name = Self::read_string(file)?;
            let len = Self::read_u32(file)? as u64;
            dims.push(NetcdfDim { name, len });
        }
        Ok(dims)
    }

    fn skip_attr_list(file: &mut File) -> Result<()> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
            Self::read_absent_count(file)?;
            return Ok(());
        }
        if tag != 12 {
            return Err("unexpected tag in attribute list".into());
        }
        let count = Self::read_u32(file)? as usize;
        for _ in 0..count {
            let _name = Self::read_string(file)?;
            let typ = Self::read_u32(file)?;
            let len = Self::read_u32(file)? as u64;
            let bytes = Self::type_size(typ)? as u64 * len;
            let padded = Self::pad4(bytes);
            file.seek(SeekFrom::Current(padded as i64))?;
        }
        Ok(())
    }

    fn read_var_list(file: &mut File, version: u8, dims: &[NetcdfDim]) -> Result<Vec<NetcdfVar>> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
            Self::read_absent_count(file)?;
            return Ok(Vec::new());
        }
        if tag != 11 {
            return Err("unexpected tag in variable list".into());
        }
        let count = Self::read_u32(file)? as usize;
        let mut vars = Vec::with_capacity(count);
        for _ in 0..count {
            let name = Self::read_string(file)?;
            let dim_count = Self::read_u32(file)? as usize;
            let mut dim_ids = Vec::with_capacity(dim_count);
            for _ in 0..dim_count {
                dim_ids.push(Self::read_u32(file)? as usize);
            }
            Self::skip_attr_list(file)?;
            let vartype = Self::read_u32(file)?;
            if !(1..=6).contains(&vartype) {
                return Err("unexpected NetCDF variable type".into());
            }
            let vsize = Self::read_u32(file)? as u64;
            let begin = if version == 1 {
                Self::read_u32(file)? as u64
            } else {
                Self::read_u64(file)?
            };
            let is_record = if !dim_ids.is_empty() {
                dims[dim_ids[0]].len == 0
            } else {
                false
            };
            vars.push(NetcdfVar {
                name,
                dim_ids,
                vartype,
                vsize,
                begin,
                is_record,
            });
        }
        Ok(vars)
    }

    /// An absent list is encoded as ZERO ZERO; consume and check the count.
    fn read_absent_count(file: &mut File) -> Result<()> {
        if Self::read_u32(file)? != 0 {
            return Err("malformed NetCDF header: absent list has a non-zero count".into());
        }
        Ok(())
    }

    fn read_u32(file: &mut File) -> Result<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    fn read_u64(file: &mut File) -> Result<u64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        Ok(u64::from_be_bytes(buf))
    }

    fn read_string(file: &mut File) -> Result<String> {
        let len = Self::read_u32(file)? as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        let padded = Self::pad4(len as u64) - len as u64;
        if padded > 0 {
            file.seek(SeekFrom::Current(padded as i64))?;
        }
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    fn type_size(typ: u32) -> Result<usize> {
        match typ {
            1 | 2 => Ok(1),
            3 => Ok(2),
            4 | 5 => Ok(4),
            6 => Ok(8),
            _ => Err("unknown NetCDF type".into()),
        }
    }

    fn pad4(n: u64) -> u64 {
        let rem = n % 4;
        if rem == 0 { n } else { n + (4 - rem) }
    }
}

#[derive(Debug, Clone)]
struct Bat {
    root: [usize; 3],
    torsions: Vec<[usize; 4]>,
    angles: Vec<[usize; 3]>,
}

fn sort_atoms_by_mass(atoms: &[usize], masses: &[f64], reverse: bool) -> Vec<usize> {
    let mut list = atoms.to_vec();
    list.sort_by(|&a, &b| {
        let ma = masses[a];
        let mb = masses[b];
        if ma == mb {
            if reverse { b.cmp(&a) } else { a.cmp(&b) }
        } else if reverse {
            mb.partial_cmp(&ma).unwrap()
        } else {
            ma.partial_cmp(&mb).unwrap()
        }
    });
    list
}

fn build_bat(fragment: &[usize], adjacency: &[Vec<usize>], masses: &[f64]) -> Result<Bat> {
    let fragment_set: HashSet<usize> = fragment.iter().copied().collect();
    if fragment.len() < 3 {
        return Err("fragment must have at least 3 atoms".into());
    }

    let mut terminal_atoms = Vec::new();
    for &a in fragment {
        let degree = adjacency[a]
            .iter()
            .filter(|n| fragment_set.contains(n))
            .count();
        if degree == 1 {
            terminal_atoms.push(a);
        }
    }
    if terminal_atoms.is_empty() {
        return Err("no terminal atoms found for BAT root".into());
    }

    let terminal_sorted = sort_atoms_by_mass(&terminal_atoms, masses, true);
    let initial_atom = terminal_sorted[0];

    let second_atom = adjacency[initial_atom]
        .iter()
        .find(|n| fragment_set.contains(n))
        .copied()
        .ok_or("initial atom has no bonded atom")?;

    let mut third_candidates: Vec<usize> = adjacency[second_atom]
        .iter()
        .filter(|n| **n != initial_atom && fragment_set.contains(n))
        .copied()
        .collect();
    if fragment.len() != 3 {
        let terminal_set: HashSet<usize> = terminal_atoms.into_iter().collect();
        third_candidates.retain(|a| !terminal_set.contains(a));
    }
    if third_candidates.is_empty() {
        return Err("no valid third atom for BAT root".into());
    }
    let third_sorted = sort_atoms_by_mass(&third_candidates, masses, true);
    let third_atom = third_sorted[0];

    let root = [initial_atom, second_atom, third_atom];
    let mut selected_atoms = vec![root[0], root[1], root[2]];
    let mut torsions = Vec::new();

    while selected_atoms.len() < fragment.len() {
        let mut torsion_added = false;
        let mut idx = 0usize;
        while idx < selected_atoms.len() {
            let a1 = selected_atoms[idx];
            let a0_list: Vec<usize> = adjacency[a1]
                .iter()
                .filter(|n| fragment_set.contains(n) && !selected_atoms.contains(n))
                .copied()
                .collect();
            let a0_sorted = sort_atoms_by_mass(&a0_list, masses, false);
            for &a0 in a0_sorted.iter() {
                let a2_list: Vec<usize> = adjacency[a1]
                    .iter()
                    .filter(|n| {
                        **n != a0
                            && fragment_set.contains(n)
                            && selected_atoms.contains(n)
                            && adjacency[**n]
                                .iter()
                                .filter(|m| fragment_set.contains(m))
                                .count()
                                > 1
                    })
                    .copied()
                    .collect();
                let a2_sorted = sort_atoms_by_mass(&a2_list, masses, false);
                if let Some(&a2) = a2_sorted.first() {
                    let a3_list: Vec<usize> = adjacency[a2]
                        .iter()
                        .filter(|n| {
                            **n != a1 && fragment_set.contains(n) && selected_atoms.contains(n)
                        })
                        .copied()
                        .collect();
                    let a3_sorted = sort_atoms_by_mass(&a3_list, masses, false);
                    if let Some(&a3) = a3_sorted.first() {
                        torsions.push([a0, a1, a2, a3]);
                        selected_atoms.push(a0);
                        torsion_added = true;
                    }
                }
            }
            idx += 1;
        }
        if !torsion_added {
            return Err("BAT torsion search failed".into());
        }
    }

    let mut angles = Vec::with_capacity(torsions.len());
    for t in torsions.iter() {
        angles.push([t[0], t[1], t[2]]);
    }

    Ok(Bat {
        root,
        torsions,
        angles,
    })
}

fn build_bat_list(
    fragment: &[usize],
    adjacency: &[Vec<usize>],
    hydrogens: &HashSet<usize>,
    masses: &[f64],
) -> Result<Vec<Vec<usize>>> {
    let mut bat_list = Vec::new();

    let fragment_set: HashSet<usize> = fragment.iter().copied().collect();
    for &a in fragment {
        for &b in adjacency[a].iter() {
            if a < b
                && fragment_set.contains(&b)
                && !hydrogens.contains(&a)
                && !hydrogens.contains(&b)
            {
                bat_list.push(vec![a, b]);
            }
        }
    }

    let bat = build_bat(fragment, adjacency, masses)?;
    bat_list.push(vec![bat.root[0], bat.root[1], bat.root[2]]);
    for angle in bat.angles.iter() {
        bat_list.push(vec![angle[0], angle[1], angle[2]]);
    }
    for torsion in bat.torsions.iter() {
        bat_list.push(vec![torsion[0], torsion[1], torsion[2], torsion[3]]);
    }

    Ok(bat_list)
}

pub struct InternalCoordinates {
    /// BAT entries, indexing into `atoms` (not into the topology).
    bat_list: Vec<Vec<usize>>,
    /// Topology indices of the atoms the BAT list uses, sorted ascending.
    atoms: Vec<usize>,
    /// Total atom count declared by the topology.
    natom: usize,
    pub dim: usize,
    pub int_coords: Vec<Vec<f64>>,
    pub pairs: Vec<(usize, usize)>,
}

/// Amber atom types that belong only to water models (including the extra
/// point of four-point models such as TIP4P-Ew and OPC).
const WATER_ATOM_TYPES: [&str; 4] = ["OW", "HW", "EP", "EPW"];

/// Residue names used for water by Amber, CHARMM and GROMACS topologies.
const WATER_RESIDUES: [&str; 16] = [
    "WAT", "HOH", "H2O", "SOL", "TIP3", "TP3", "T3P", "TIP4", "TP4", "T4P", "T4E", "TIP5", "TP5",
    "SPC", "SPCE", "OPC",
];

/// Whether atom `index` is excluded before molecules are assembled: water, by
/// type or residue name, and massless virtual sites, which carry no degrees of
/// freedom of their own.
fn is_excluded(prmtop: &Prmtop, index: usize) -> bool {
    let atom_type = prmtop.atom_types[index].trim();
    if WATER_ATOM_TYPES.contains(&atom_type) {
        return true;
    }
    if let Some(label) = prmtop.residue_labels.get(index) {
        let label = label.trim().to_ascii_uppercase();
        if WATER_RESIDUES.contains(&label.as_str()) {
            return true;
        }
    }
    let massless = prmtop.masses[index] <= 0.0;
    let no_element = prmtop.atomic_numbers.get(index) == Some(&0);
    massless || no_element
}

impl InternalCoordinates {
    pub fn new(top: &Path) -> Result<Self> {
        let prmtop = parse_prmtop(top)?;
        if prmtop.masses.len() != prmtop.natom || prmtop.atom_types.len() != prmtop.natom {
            return Err("topology MASS or atom type section is shorter than NATOM".into());
        }
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); prmtop.natom];
        for (a, b) in prmtop.bonds.iter().copied() {
            if a >= prmtop.natom || b >= prmtop.natom {
                return Err(format!("bond ({a}, {b}) references an atom beyond NATOM").into());
            }
            adjacency[a].push(b);
            adjacency[b].push(a);
        }

        let mut molecule_mask = vec![true; prmtop.natom];
        let mut hydrogens = HashSet::new();
        for (i, typ) in prmtop.atom_types.iter().enumerate() {
            let t = typ.trim();
            if is_excluded(&prmtop, i) {
                molecule_mask[i] = false;
            }
            if (t.starts_with('H') || t.starts_with('h')) && t != "HW" {
                hydrogens.insert(i);
            }
        }

        let mut visited = vec![false; prmtop.natom];
        let mut bat_list = Vec::new();
        let mut atoms = Vec::new();
        for i in 0..prmtop.natom {
            if !molecule_mask[i] || visited[i] {
                continue;
            }
            let mut queue = VecDeque::new();
            let mut fragment = Vec::new();
            queue.push_back(i);
            visited[i] = true;
            while let Some(a) = queue.pop_front() {
                fragment.push(a);
                for &b in adjacency[a].iter() {
                    if molecule_mask[b] && !visited[b] {
                        visited[b] = true;
                        queue.push_back(b);
                    }
                }
            }

            match fragment.len() {
                // A lone atom (a monatomic ion, say) has no internal
                // coordinates at all.
                1 => continue,
                // A diatomic has a single internal coordinate, its bond, which
                // is kept under the same heavy-atom rule as every other bond.
                2 => {
                    let (a, b) = (fragment[0].min(fragment[1]), fragment[0].max(fragment[1]));
                    if !hydrogens.contains(&a) && !hydrogens.contains(&b) {
                        bat_list.push(vec![a, b]);
                    }
                }
                _ => bat_list.extend(build_bat_list(
                    &fragment,
                    &adjacency,
                    &hydrogens,
                    &prmtop.masses,
                )?),
            }
            atoms.extend(fragment);
        }

        // The trajectory reader gathers exactly `atoms`, in ascending order, so
        // rewrite every BAT entry to index that gathered buffer.
        atoms.sort_unstable();
        let mut local = vec![usize::MAX; prmtop.natom];
        for (position, &atom) in atoms.iter().enumerate() {
            local[atom] = position;
        }
        for entry in bat_list.iter_mut() {
            for atom in entry.iter_mut() {
                *atom = local[*atom];
                debug_assert_ne!(*atom, usize::MAX, "BAT entry uses an unselected atom");
            }
        }

        let dim = bat_list.len();
        Ok(InternalCoordinates {
            bat_list,
            atoms,
            natom: prmtop.natom,
            dim,
            int_coords: Vec::new(),
            pairs: Vec::new(),
        })
    }

    pub fn calculate_internal_coords(
        &mut self,
        traj: &Path,
        frames: usize,
        torsions_only: bool,
    ) -> Result<()> {
        let mut reader = NetcdfReader::open(traj, self.natom)?;
        let vartype = reader.coordinates_vartype()?;
        if torsions_only {
            self.bat_list = self
                .bat_list
                .iter()
                .filter(|x| x.len() == 4)
                .cloned()
                .collect();
            self.dim = self.bat_list.len();
        }
        if vartype == 5 {
            let coords = reader.read_coordinates_f32(frames, &self.atoms)?;
            self.int_coords = internal_coordinates_f32(&self.bat_list, &coords);
        } else {
            let coords = reader.read_coordinates_f64(frames, &self.atoms)?;
            self.int_coords = internal_coordinates(&self.bat_list, &coords);
        }
        Ok(())
    }

    pub fn coordinate_pairs(&mut self) {
        let mut pairs = Vec::new();
        for i in 0..self.dim {
            for j in 0..self.dim {
                if i >= j {
                    continue;
                }
                pairs.push((i, j));
            }
        }
        self.pairs = pairs;
    }

    pub fn coordinate_metrics(&self) -> Vec<CoordinateMetric> {
        self.bat_list
            .iter()
            .map(|entry| {
                if entry.len() == 4 {
                    CoordinateMetric::Periodic {
                        period: 2.0 * std::f64::consts::PI,
                    }
                } else {
                    CoordinateMetric::Linear
                }
            })
            .collect()
    }
}
