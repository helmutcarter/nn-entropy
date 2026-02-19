use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct Prmtop {
    natom: usize,
    masses: Vec<f64>,
    atom_types: Vec<String>,
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
        sec.1.iter().take(natom).map(|s| s.trim().to_string()).collect()
    } else if let Some(sec) = sections.get("ATOM_NAME") {
        sec.1.iter().take(natom).map(|s| s.trim().to_string()).collect()
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

    let mut bonds = Vec::new();
    for i in 0..bond_h_count {
        let a = bond_h_tokens[i * 3].parse::<i64>()?;
        let b = bond_h_tokens[i * 3 + 1].parse::<i64>()?;
        let ai = (a.abs() as usize) / 3;
        let bi = (b.abs() as usize) / 3;
        bonds.push((ai, bi));
    }
    for i in 0..bond_no_h_count {
        let a = bond_no_h_tokens[i * 3].parse::<i64>()?;
        let b = bond_no_h_tokens[i * 3 + 1].parse::<i64>()?;
        let ai = (a.abs() as usize) / 3;
        let bi = (b.abs() as usize) / 3;
        bonds.push((ai, bi));
    }

    Ok(Prmtop {
        natom,
        masses,
        atom_types,
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
    version: u8,
    numrecs: u64,
    dims: Vec<NetcdfDim>,
    vars: Vec<NetcdfVar>,
    record_size: u64,
}

impl NetcdfReader {
    fn open(path: &Path) -> Result<Self> {
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
        let numrecs = Self::read_u32(&mut file)? as u64;

        let dims = Self::read_dim_list(&mut file)?;
        Self::skip_attr_list(&mut file)?;
        let vars = Self::read_var_list(&mut file, version, &dims)?;
        let record_size = vars
            .iter()
            .filter(|v| v.is_record)
            .map(|v| v.vsize)
            .sum();

        Ok(NetcdfReader {
            file,
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

    fn read_coordinates_f64(
        &mut self,
        frames: usize,
        atom_num: usize,
    ) -> Result<Vec<Vec<[f64; 3]>>> {
        let var = self.coordinates_var()?;

        let dims: Vec<u64> = var
            .dim_ids
            .iter()
            .map(|&i| self.dims[i].len)
            .collect();
        if dims.len() < 2 {
            return Err("coordinates variable has too few dimensions".into());
        }

        let (frame_count, atom_dim, spatial_dim) = if var.is_record {
            let total_frames = if self.numrecs == u64::MAX {
                frames as u64
            } else {
                self.numrecs
            };
            let frame_count = frames.min(total_frames as usize);
            let atom_dim = dims[1];
            let spatial_dim = dims[2];
            (frame_count, atom_dim, spatial_dim)
        } else {
            let atom_dim = dims[0];
            let spatial_dim = dims[1];
            (1usize, atom_dim, spatial_dim)
        };

        if spatial_dim != 3 {
            return Err("coordinates spatial dimension is not 3".into());
        }
        if atom_num as u64 > atom_dim {
            return Err("requested atom count exceeds file atom dimension".into());
        }

        let bytes_per = match var.vartype {
            5 => 4u64, // NC_FLOAT
            6 => 8u64, // NC_DOUBLE
            _ => return Err("unsupported coordinates data type".into()),
        };
        let _ = bytes_per;

        let mut frames_out = Vec::with_capacity(frame_count);
        for frame_idx in 0..frame_count {
            let offset = if var.is_record {
                var.begin + (frame_idx as u64) * self.record_size
            } else {
                var.begin
            };
            self.file.seek(SeekFrom::Start(offset))?;

            let mut atoms = Vec::with_capacity(atom_num);
            for _ in 0..atom_num {
                let x = Self::read_num(&mut self.file, var.vartype)?;
                let y = Self::read_num(&mut self.file, var.vartype)?;
                let z = Self::read_num(&mut self.file, var.vartype)?;
                atoms.push([x, y, z]);
            }
            frames_out.push(atoms);

            let var_end = offset + var.vsize;
            self.file.seek(SeekFrom::Start(var_end))?;
        }

        Ok(frames_out)
    }

    fn read_coordinates_f32(
        &mut self,
        frames: usize,
        atom_num: usize,
    ) -> Result<Vec<Vec<[f32; 3]>>> {
        let var = self.coordinates_var()?;
        if var.vartype != 5 {
            return Err("coordinates variable is not float".into());
        }

        let dims: Vec<u64> = var
            .dim_ids
            .iter()
            .map(|&i| self.dims[i].len)
            .collect();
        if dims.len() < 2 {
            return Err("coordinates variable has too few dimensions".into());
        }

        let (frame_count, atom_dim, spatial_dim) = if var.is_record {
            let total_frames = if self.numrecs == u64::MAX {
                frames as u64
            } else {
                self.numrecs
            };
            let frame_count = frames.min(total_frames as usize);
            let atom_dim = dims[1];
            let spatial_dim = dims[2];
            (frame_count, atom_dim, spatial_dim)
        } else {
            let atom_dim = dims[0];
            let spatial_dim = dims[1];
            (1usize, atom_dim, spatial_dim)
        };

        if spatial_dim != 3 {
            return Err("coordinates spatial dimension is not 3".into());
        }
        if atom_num as u64 > atom_dim {
            return Err("requested atom count exceeds file atom dimension".into());
        }

        let mut frames_out = Vec::with_capacity(frame_count);
        for frame_idx in 0..frame_count {
            let offset = if var.is_record {
                var.begin + (frame_idx as u64) * self.record_size
            } else {
                var.begin
            };
            self.file.seek(SeekFrom::Start(offset))?;

            let mut atoms = Vec::with_capacity(atom_num);
            for _ in 0..atom_num {
                let x = Self::read_f32(&mut self.file)?;
                let y = Self::read_f32(&mut self.file)?;
                let z = Self::read_f32(&mut self.file)?;
                atoms.push([x, y, z]);
            }
            frames_out.push(atoms);

            let var_end = offset + var.vsize;
            self.file.seek(SeekFrom::Start(var_end))?;
        }

        Ok(frames_out)
    }

    fn read_dim_list(file: &mut File) -> Result<Vec<NetcdfDim>> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
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
            let mut vartype = Self::read_u32(file)?;
            if vartype == 0 {
                let possible = Self::read_u32(file)?;
                if (1..=6).contains(&possible) {
                    vartype = possible;
                } else {
                    return Err("unexpected NetCDF variable type".into());
                }
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

    fn read_num(file: &mut File, vartype: u32) -> Result<f64> {
        match vartype {
            5 => {
                let v = Self::read_f32(file)?;
                Ok(v as f64)
            }
            6 => {
                let v = Self::read_f64(file)?;
                Ok(v)
            }
            _ => Err("unsupported numeric type".into()),
        }
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

    fn read_f32(file: &mut File) -> Result<f32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        Ok(f32::from_be_bytes(buf))
    }

    fn read_f64(file: &mut File) -> Result<f64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        Ok(f64::from_be_bytes(buf))
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
        if rem == 0 {
            n
        } else {
            n + (4 - rem)
        }
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
            if reverse {
                b.cmp(&a)
            } else {
                a.cmp(&b)
            }
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
                for &a2 in a2_sorted.iter() {
                    let a3_list: Vec<usize> = adjacency[a2]
                        .iter()
                        .filter(|n| **n != a1 && fragment_set.contains(n) && selected_atoms.contains(n))
                        .copied()
                        .collect();
                    let a3_sorted = sort_atoms_by_mass(&a3_list, masses, false);
                    for &a3 in a3_sorted.iter() {
                        torsions.push([a0, a1, a2, a3]);
                        selected_atoms.push(a0);
                        torsion_added = true;
                        break;
                    }
                    break;
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

    Ok(Bat { root, torsions, angles })
}

fn bond_calc(a1: [f64; 3], a2: [f64; 3]) -> f64 {
    let dx = a1[0] - a2[0];
    let dy = a1[1] - a2[1];
    let dz = a1[2] - a2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn angle_calc(a1: [f64; 3], a2: [f64; 3], a3: [f64; 3]) -> f64 {
    let v1 = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]];
    let v2 = [a3[0] - a2[0], a3[1] - a2[1], a3[2] - a2[2]];
    let v1_mag = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let v2_mag = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let denom = v1_mag * v2_mag;
    (dot / denom).acos()
}

fn torsion_calc(a1: [f64; 3], a2: [f64; 3], a3: [f64; 3], a4: [f64; 3]) -> f64 {
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

fn bond_calc_f32(a1: [f32; 3], a2: [f32; 3]) -> f64 {
    let dx = a1[0] - a2[0];
    let dy = a1[1] - a2[1];
    let dz = a1[2] - a2[2];
    let sum = (dx as f64) * (dx as f64)
        + (dy as f64) * (dy as f64)
        + (dz as f64) * (dz as f64);
    sum.sqrt()
}

fn angle_calc_f32(a1: [f32; 3], a2: [f32; 3], a3: [f32; 3]) -> f64 {
    let v1 = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]];
    let v2 = [a3[0] - a2[0], a3[1] - a2[1], a3[2] - a2[2]];
    let v1_sum = (v1[0] as f64) * (v1[0] as f64)
        + (v1[1] as f64) * (v1[1] as f64)
        + (v1[2] as f64) * (v1[2] as f64);
    let v2_sum = (v2[0] as f64) * (v2[0] as f64)
        + (v2[1] as f64) * (v2[1] as f64)
        + (v2[2] as f64) * (v2[2] as f64);
    let v1_mag = v1_sum.sqrt();
    let v2_mag = v2_sum.sqrt();
    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let denom = v1_mag * v2_mag;
    ((dot as f64) / denom).acos()
}

fn torsion_calc_f32(a1: [f32; 3], a2: [f32; 3], a3: [f32; 3], a4: [f32; 3]) -> f64 {
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
            if a < b && fragment_set.contains(&b) {
                if !hydrogens.contains(&a) && !hydrogens.contains(&b) {
                    bat_list.push(vec![a, b]);
                }
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

fn int_c(bat_list: &[Vec<usize>], traj: &[Vec<[f64; 3]>]) -> Vec<Vec<f64>> {
    let frame_number = traj.len();
    let int_coord_number = bat_list.len();
    let mut int_coords = vec![vec![0.0f64; int_coord_number]; frame_number];

    for i in 0..frame_number {
        for j in 0..int_coord_number {
            let entry = &bat_list[j];
            if entry.len() == 2 {
                int_coords[i][j] = bond_calc(traj[i][entry[0]], traj[i][entry[1]]);
            } else if entry.len() == 3 {
                int_coords[i][j] =
                    angle_calc(traj[i][entry[0]], traj[i][entry[1]], traj[i][entry[2]]);
            } else if entry.len() == 4 {
                int_coords[i][j] = torsion_calc(
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

fn int_c_f32(bat_list: &[Vec<usize>], traj: &[Vec<[f32; 3]>]) -> Vec<Vec<f64>> {
    let frame_number = traj.len();
    let int_coord_number = bat_list.len();
    let mut int_coords = vec![vec![0.0f64; int_coord_number]; frame_number];

    for i in 0..frame_number {
        for j in 0..int_coord_number {
            let entry = &bat_list[j];
            if entry.len() == 2 {
                int_coords[i][j] = bond_calc_f32(traj[i][entry[0]], traj[i][entry[1]]);
            } else if entry.len() == 3 {
                int_coords[i][j] = angle_calc_f32(
                    traj[i][entry[0]],
                    traj[i][entry[1]],
                    traj[i][entry[2]],
                );
            } else if entry.len() == 4 {
                int_coords[i][j] = torsion_calc_f32(
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

pub struct InternalCoordinates {
    bat_list: Vec<Vec<usize>>,
    atom_num: usize,
    pub dim: usize,
    pub int_coords: Vec<Vec<f64>>,
    pub pairs: Vec<(usize, usize)>,
}

impl InternalCoordinates {
    pub fn new(top: &Path) -> Result<Self> {
        let prmtop = parse_prmtop(top)?;
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); prmtop.natom];
        for (a, b) in prmtop.bonds.iter().copied() {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }

        let mut molecule_mask = vec![true; prmtop.natom];
        let mut hydrogens = HashSet::new();
        for (i, typ) in prmtop.atom_types.iter().enumerate() {
            let t = typ.trim();
            if t == "HW" || t == "OW" || t == "EP" {
                molecule_mask[i] = false;
            }
            if (t.starts_with('H') || t.starts_with('h')) && t != "HW" {
                hydrogens.insert(i);
            }
        }

        let mut visited = vec![false; prmtop.natom];
        let mut bat_list = Vec::new();
        let mut atom_num = 0usize;
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
            atom_num += fragment.len();
            bat_list.extend(build_bat_list(&fragment, &adjacency, &hydrogens, &prmtop.masses)?);
        }

        let dim = bat_list.len();
        Ok(InternalCoordinates {
            bat_list,
            atom_num,
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
        let mut reader = NetcdfReader::open(traj)?;
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
            let coords = reader.read_coordinates_f32(frames, self.atom_num)?;
            self.int_coords = int_c_f32(&self.bat_list, &coords);
        } else {
            let coords = reader.read_coordinates_f64(frames, self.atom_num)?;
            self.int_coords = int_c(&self.bat_list, &coords);
        }
        Ok(())
    }

    pub fn coordinate_pairs(&mut self) {
        let mut pairs = Vec::new();
        for i in 0..self.dim {
            for j in 0..self.dim {
                if i == j || i > j {
                    continue;
                }
                pairs.push((i, j));
            }
        }
        self.pairs = pairs;
    }
}
