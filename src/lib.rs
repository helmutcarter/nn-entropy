use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use std::error::Error;
use std::num::NonZero;
use std::fs::File;
use std::io::ErrorKind;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;

pub mod pyo3_api;
pub fn calculate_entropy_from_data(one_d_data: Vec<Vec<f64>>, frames_end: usize) -> f64 {
    let mut one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect::<Vec<_>>();

    let n_frames = one_d_data[0].len();
    let degrees_freedom = one_d_data.len();

    const EULER: f64 = 0.57721566490153;
    let one_d_constant = ((n_frames as f64) * 2.0).ln() + EULER;
    let two_d_constant = ((n_frames as f64) * std::f64::consts::PI).ln() + EULER;

    let one_d_distances_total: f64 = one_d_data
        .iter()
        .map(|ic| calc_one_d_nn(ic.clone()))
        .sum();

    let one_d_entropy =
        estimate_entropy(one_d_distances_total, n_frames, one_d_constant, degrees_freedom);

    let two_d_degrees_freedom = (0..degrees_freedom)
        .flat_map(|i| (i + 1)..degrees_freedom)
        .count();

    let two_d_distances_total: f64 = (0..degrees_freedom)
        .into_par_iter()
        .map(|i| {
            (i + 1..degrees_freedom)
                .map(|j| calc_two_d_nn(&one_d_data[i], &one_d_data[j]))
                .sum::<f64>()
        })
        .sum();

    let two_d_entropy = estimate_entropy(
        two_d_distances_total * 2.0,
        n_frames,
        two_d_constant,
        two_d_degrees_freedom,
    );

    two_d_entropy - ((degrees_freedom - 2) as f64) * one_d_entropy
}

pub fn estimate_coordinate_entropy_rust(one_d_data: Vec<Vec<f64>>, frames_end: usize) -> Vec<f64> {
    let mut one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect::<Vec<_>>();

    let n_frames: usize = one_d_data[0].len();
    let degrees_freedom: usize = one_d_data.len();

    const EULER: f64 = 0.57721566490153;
    let one_d_constant: f64 = ((n_frames as f64) * 2.0).ln() + EULER;

    let one_d_distances: Vec<f64> = one_d_data
        .iter()
        .map(|ic| calc_one_d_nn(ic.clone()))
        .collect();

    let one_d_entropies: Vec<f64> = one_d_distances
        .iter()
        .map(|&distance| estimate_entropy(distance, n_frames, one_d_constant, 1))
        .collect();

    one_d_entropies
}

pub fn estimate_coordinate_mutual_information_rust(one_d_data: Vec<Vec<f64>>, frames_end: usize) -> Vec<f64> {
    let one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect::<Vec<_>>();

    let n_frames: usize = one_d_data[0].len();
    let degrees_freedom: usize = one_d_data.len();

    const EULER: f64 = 0.57721566490153;
    let two_d_constant: f64 = ((n_frames as f64) * std::f64::consts::PI).ln() + EULER;

    let two_d_degrees_freedom = (0..degrees_freedom)
        .flat_map(|i| (i + 1)..degrees_freedom)
        .count();

    let two_d_distances: Vec<f64> = (0..degrees_freedom)
        .into_par_iter()
        .flat_map(|i| {
            (i + 1..degrees_freedom)
                // .into_par_iter()
                .map(|j| calc_two_d_nn(&one_d_data[i], &one_d_data[j]))
                .collect::<Vec<f64>>()
        })
        .collect();

    assert_eq!(two_d_distances.len(), two_d_degrees_freedom); // Just to be safe

    let two_d_entropies: Vec<f64> = two_d_distances
        .iter()
        .map(|&distance| estimate_entropy(distance * 2.0, n_frames, two_d_constant, 1))
        .collect();
    two_d_entropies
}

pub fn calc_one_d_nn(points: Vec<f64>) -> f64 {
    let mut unique_points = points.clone();
    unique_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_points.dedup();
    let total_unique_points = unique_points.len();  
    // println!("{:.2}% of values are unique.", (total_unique_points as f32)/(points.len() as f32)*100.0);
    let mut distance_total: f64 = 0.0;

    for point in points 
    {    
        let index = unique_points.binary_search_by(|probe| probe.total_cmp(&point)).unwrap();
        if index == 0 {
            distance_total += f64::min(distance(point, unique_points[total_unique_points-1]),
                            distance(point, unique_points[index+1])).ln();
        }
        else if index == total_unique_points-1 {
            distance_total += f64::min(distance(point, unique_points[index-1]),
                            distance(point, unique_points[0])).ln(); 
        }
        else {
            distance_total += f64::min(distance(point, unique_points[index-1]),
                            distance(point, unique_points[index+1])).ln();
                        }}
    distance_total
}

pub fn calc_one_d_nn_kdtree(points: Vec<f64>) -> f64 {
    let mut unique_points = points.clone();
    unique_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_points.dedup();
    let mut unique_point_vec_array: Vec<[f64; 1]> = Vec::new();
    for point in &unique_points {
        unique_point_vec_array.push([*point])
    }

    let kdtree: ImmutableKdTree<f64, 1> = ImmutableKdTree::new_from_slice(&unique_point_vec_array);

    // println!("{:?}", unique_points);
    // let total_unique_points = unique_points.len();  
    // println!("{:.2}% of values are unique.", (total_unique_points as f32)/(points.len() as f32)*100.0);
    let mut distance_total: f64 = 0.0;

    for point in points {
        let result: f64 = kdtree.nearest_n::<SquaredEuclidean>(&[point], NonZero::new(2).unwrap())[1].distance;
        distance_total += result.sqrt().ln();
    }
    distance_total
}

pub fn calc_two_d_nn(points_1: &Vec<f64>, points_2: &Vec<f64>) -> f64 {
    let mut points: Vec<[f64; 2]> = Vec::new();
    for (point_1, point_2) in points_1.into_iter().zip(points_2) {
        points.push([*point_1, *point_2]);
    }

    let kdtree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&points);
    let mut distance_total: f64 = 0.0;
    for point in points {
        let mut result: f64 = kdtree.nearest_n::<SquaredEuclidean>(&point, NonZero::new(2).unwrap())[1].distance;
        if result == 0.0 {
            result = kdtree.nearest_n::<SquaredEuclidean>(&point, NonZero::new(3).unwrap())[2].distance;
        }
        distance_total += result.sqrt().ln();
    }
    distance_total
}
// Helper function to generate Guassian data
pub fn generate_normal(mean: f64, std_dev: f64, size: usize) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::thread_rng();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

fn distance(first_point: f64, second_points: f64) -> f64 {
    (first_point - second_points).abs()
}
pub fn estimate_entropy(nn_distance: f64, n_frames: usize, constant: f64, n_internal_coords: usize) -> f64 {
    // println!("{constant}");
    (nn_distance/(n_frames as f64)) + constant*(n_internal_coords as f64)
}
pub fn load_one_d_data(file_path: &str) -> Vec<Vec<f64>> {
    let mut all_data: Vec<Vec<f64>> = Vec::new();
    let file_result = File::open(file_path);
    let file = match file_result {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {e:?}"),
            },
            other_error => {
                panic!("Problem opening the file: {other_error:?}");
            }
        },
    };
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(file);
    
    for result in reader.records() {
        let mut internal_coordinate_data: Vec<f64> = Vec::new();
        match result {
            Ok(record) => for entry in record.iter() {
                let point: f64 = entry.parse().unwrap();
                internal_coordinate_data.push(point);
                },
            Err(e) => println!("Error reading record: {:?}", e),
        }
        all_data.push(internal_coordinate_data);
    }
    all_data
}

pub fn cross_product(b1: [f64; 3], b2: [f64; 3]) -> [f64; 3] {
    [
        // x-component: b2_y * b2_z - b2_z * b2_y
        b1[1] * b2[2] - b1[2] * b2[1],
        // y-component: b1_z * b2_x - b1_x * b2_z
        b1[2] * b2[0] - b1[0] * b2[2],
        // z-component: b1_x * b2_y - b1_y * b2_x
        b1[0] * b2[1] - b1[1] * b2[0],
    ]
}
use libm::atan2;

fn dot_product(array_1: [f64; 3], array_2: [f64; 3]) -> f64 {
    array_1.iter().zip(array_2.iter()).map(|(x, y)| x * y).sum()
}

pub fn calc_bond(atom_1: [f64; 3], atom_2: [f64; 3]) -> f64 {
    // sqSum = (a1[0]-a2[0])**2 + (a1[1]-a2[1])**2 + (a1[2]-a2[2])**2
    let square_sum = (atom_1[0]-atom_2[0]).powi(2) + (atom_1[1]-atom_2[1]).powi(2) + (atom_1[2]-atom_2[2]).powi(2);
    
    // dist = math.sqrt(sqSum)
    // return dist
    square_sum.sqrt()
}

pub fn calc_angle(a1: [f64; 3], a2: [f64; 3], a3: [f64; 3]) -> f64 {
    let v1 = [(a1[0]-a2[0]),(a1[1]-a2[1]),(a1[2]-a2[2])];
    let v2 = [(a3[0]-a2[0]),(a3[1]-a2[1]),(a3[2]-a2[2])];

    // v1Mag = math.sqrt((v1[0])**2 + (v1[1])**2 + (v1[2])**2)
    let vector_1_magnitude = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();

    // v2Mag = math.sqrt((v2[0])**2 + (v2[1])**2 + (v2[2])**2)
    let vector_2_magnitude = (v2[0].powi(2) + v2[1].powi(2) + v2[2].powi(2)).sqrt();

    // let dot: f64 = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
    let dot: f64 = dot_product(v1, v2);

    let v1v2_mag = vector_1_magnitude*vector_2_magnitude;

    // angle = math.acos(dot/v1v2Mag)
    // return angle
    (dot/v1v2_mag).acos()
}
pub fn calc_torsion(atom_1: [f64; 3], atom_2: [f64; 3], atom_3: [f64; 3], atom_4: [f64; 3]) -> f64 {
    let bond_1: [f64; 3] = [atom_1[0] - atom_2[0], atom_1[1] - atom_2[1], atom_1[2] - atom_2[2]];
    let bond_2: [f64; 3] = [atom_2[0] - atom_3[0], atom_2[1] - atom_3[1], atom_2[2] - atom_3[2]];
    let bond_3: [f64; 3] = [atom_3[0] - atom_4[0], atom_3[1] - atom_4[1], atom_3[2] - atom_4[2]];

    let cross_1: [f64; 3] = cross_product(bond_2, bond_3);
    let cross_2: [f64; 3] = cross_product(bond_1, bond_2);

    let mut plane_1 = dot_product(bond_1, cross_1);
    plane_1 *= (dot_product(bond_2, bond_2)).sqrt();

    let plane_2 = dot_product(cross_1, cross_2);

    atan2(plane_1, plane_2)
}

// Equivalent to intC(batList, traj) in Joe Cruz's code
pub fn calc_internal_coords(bat_list: Vec<Vec<usize>>, traj: Vec<Vec<[f64; 3]>>) -> Vec<Vec<f64>> {
    let n_frames: usize = traj.len();
    let n_int_coords: usize = bat_list.len();

    // intCoords = np.empty((frameNumber,intCoordNumber))
    let mut internal_coords: Vec<Vec<f64>> = Vec::new();

    // for i in range(frameNumber):
    for i in 0..n_frames as usize {
        let mut frame_coords: Vec<f64> = Vec::new();
        // for j in range(intCoordNumber):
        for j in 0..n_int_coords as usize {
            if bat_list[j].len() == 2 {
                // intCoords[i,j] = bondCalc(traj[i,bat_list[j][0]],traj[i,bat_list[j][1]])
                frame_coords.push(calc_bond(traj[i][bat_list[j][0]],traj[i][bat_list[j][1]]));
                // println!("{:?}", j);
            }
            if bat_list[j].len() == 3 {
                // intCoords[i,j] = angleCalc(traj[i][bat_list[j][0]],traj[i][bat_list[j][1]],traj[i][bat_list[j][2]]);
                frame_coords.push(calc_angle(traj[i][bat_list[j][0]],traj[i][bat_list[j][1]],traj[i][bat_list[j][2]]));
            }
            if bat_list[j].len() == 4 {
                // intCoords[i,j] = torsionCalc(traj[i,bat_list[j][0]],traj[i,bat_list[j][1]],traj[i,bat_list[j][2]],traj[i,bat_list[j][3]])
                frame_coords.push(calc_torsion(traj[i][bat_list[j][0]],traj[i][bat_list[j][1]],traj[i][bat_list[j][2]], traj[i][bat_list[j][3]]));
            }
        }
        internal_coords.push(frame_coords);
    }
    // return intCoords
    // println!("{:?}", internal_coords);
    internal_coords
}

use std::io::{BufRead, BufReader};
#[allow(dead_code)]
fn get_atom_count(parm_path: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(parm_path)?;
    let reader = BufReader::new(file);

    let mut found_pointers = false;
    let mut in_pointers = false;

    for line in reader.lines() {
        let line = line?;

        if line.starts_with("%FLAG POINTERS") {
            found_pointers = true;
            continue;
        }

        if found_pointers {
            if line.starts_with("%FORMAT") {
                in_pointers = true;
                continue;
            }

            if in_pointers {
                // Stop if we hit the next section
                if line.starts_with("%FLAG") {
                    break;
                }

                // Process the first valid line of data
                let first_number = line.chars()
                    .take(8)  // First 8-character field
                    .collect::<String>()
                    .trim()
                    .parse::<usize>()?;
                
                return Ok(first_number);
            }
        }
    }

    Err("Could not find POINTERS section or valid first number".into())
}

#[allow(dead_code)]
fn parse_bond_data(parm_path: &str, include_hydrogens: bool) -> Result<Vec<[usize; 2]>, Box<dyn Error>> {
    let file = File::open(parm_path)?;
    let reader = BufReader::new(file);
    let query_section: &str = if include_hydrogens { 
        "%FLAG BONDS_INC_HYDROGEN" 
    } else { 
        "%FLAG BONDS_WITHOUT_HYDROGEN" 
    };
    let mut found_flag: bool = false;
    let mut in_flag:bool = false;
    let mut output: Vec<[usize; 2]> = Vec::new();

    for line in reader.lines() {
        let line = line?;
            if line.starts_with(query_section) {
            found_flag = true;
            continue;
        }

        if found_flag {
            if line.starts_with("%FORMAT") {
                in_flag = true;
                continue;
            }

            if in_flag {
                // Stop if we hit the next section
                if line.starts_with("%FLAG") {
                    break;
                }
                let data: Vec<usize> = line.split_ascii_whitespace().map(|x: &str| x.parse::<usize>().unwrap()).collect();
                for chunk in data.chunks(3) {
                    output.push([chunk[0], chunk[1]])
                }
            }
        }
    }
    Ok(output) 
}

#[allow(dead_code)]
fn parse_atom_data(parm_path: &str, include_hydrogens: bool) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(parm_path)?;
    let reader = BufReader::new(file);
    let query_section: &str = "ATOM_NAME";
    let mut found_flag: bool = false;
    let mut in_flag:bool = false;
    let mut output: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line?;
            if line.starts_with(query_section) {
            found_flag = true;
            continue;
        }

        if found_flag {
            if line.starts_with("%FORMAT") {
                in_flag = true;
                continue;
            }

            if in_flag {
                // Stop if we hit the next section
                if line.starts_with("%FLAG") {
                    break;
                }
                output = if include_hydrogens { 
                    line.split_ascii_whitespace()
                        .filter(|x| x.starts_with("H"))
                        .map(|x| x.to_string())
                        .collect()
                } else { 
                    line.split_ascii_whitespace()
                        .filter(|x| !x.starts_with("H"))
                        .map(|x| x.to_string())
                        .collect()
                };
                }
            }
        }
    
    Ok(output) 
}
// Doing things this way not because it makes sense but because it matches Joe Cruz's python code
// It may make sense though. I don't know enough about rust to say for sure.
#[allow(dead_code)]
struct Molecule {
    atoms: Vec<String>, // I think we don't actually strictly need this? Maybe later?
    atom_count: usize,
    hydrogens: Vec<String>,
    int_coord_list: Vec<Vec<usize>>,
    torsion_count: usize,
    int_coords_count: usize,
    bonds_with_hydrogens: Vec<[usize; 2]>,
    bonds_without_hydrogens: Vec<[usize; 2]>,
}

// impl Molecule {
//     fn new(topology_file: &str, trajectory_file: &str,) -> Result<Self, Box<dyn Error>> {
//         let molecule = Molecule{
//             atoms: parse_atom_data(&topology_file, false)?,
//             atom_count:  get_atom_count(&topology_file)?,
//             hydrogens: parse_atom_data(&topology_file, true)?,
//             int_coord_list: int_coord_list,
//             torsion_count: torsion_count,
//             int_coords_count: int_coords_count,
//             bonds_with_hydrogens: parse_bond_data(&topology_file, true)?,
//             bonds_without_hydrogens: parse_bond_data(&topology_file, false)?,
//         };
//         Ok(molecule)
//     }
// }

