use nn_entropy::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let frames_end: usize = args[1].parse().unwrap();
    let mut one_d_data: Vec<Vec<f64>> = load_one_d_data("/home/helmut/Downloads/JC_NN_Code_backup/2-methyl-hexane_unrestrained_1000000_ICs.csv");
    // one_d_data = one_d_data[(one_d_data.len()-66)..one_d_data.len()].to_vec(); // torsion angles only
    // one_d_data = one_d_data[..40].to_vec(); // bond lengths only
    // one_d_data = one_d_data[40..107].to_vec(); // bond angles only

    // Looking at only some frames to assess convergence
    one_d_data = one_d_data
        .into_iter()
        .map(|internal_coordinate| internal_coordinate[..frames_end].to_vec())
        .collect();

    let n_frames = one_d_data[0].len() as usize;
    
    let degrees_freedom = one_d_data.len() as usize;
    const EULER: f64 = 0.57721566490153;
    // println!("Number of frames: {n_frames}");
    print!("{:?},", n_frames);

    let one_d_constant: f64 = ((n_frames as f64) * 2.0).ln() + EULER;
    let two_d_constant: f64 = ((n_frames as f64)*std::f64::consts::PI).ln() + EULER;
    // println!("1D Constant: {one_d_constant}");
    // println!("2D Constant: {two_d_constant}");
    let mut one_d_distances_total: f64 = 0.0;
    for internal_coordinate in &one_d_data {
        one_d_distances_total += calc_one_d_nn(internal_coordinate.to_vec())
    }

    let one_d_entropy: f64 = estimate_entropy(one_d_distances_total, n_frames, one_d_constant, degrees_freedom);
    // println!("Total 1D entropy: {:?}", one_d_entropy);
    print!("{:?},", one_d_entropy);

    // 2D distance calculation
    let mut two_d_degrees_freedom: usize = 0;
    // let mut two_d_distances_total: f64 = 0.0;
        for i in 0..degrees_freedom {
            for j in 0..degrees_freedom {
                if i == j || i > j {}
                else {two_d_degrees_freedom += 1;}
            }
        }
    use rayon::prelude::*;

    let two_d_distances_total: f64 = (0..degrees_freedom)
        .into_par_iter()  // Convert to parallel iterator
        .map(|i| {
            let mut sum = 0.0;
            // Inner loop remains sequential but processes only j > i pairs
            for j in (i + 1)..degrees_freedom {
                sum += calc_two_d_nn(&one_d_data[i], &one_d_data[j]);
            }
            sum
        })
        .sum();  // Sum all partial sums from different threads
    let two_d_entropy: f64 = estimate_entropy(two_d_distances_total*2.0, n_frames, two_d_constant, two_d_degrees_freedom);

    // println!("Total 2D entropy: {:?}", two_d_entropy);
    print!("{:?},", two_d_entropy);
    let total_entropy = two_d_entropy - ((degrees_freedom-2) as f64)*one_d_entropy;
    // println!("Total entropy: {:?}", total_entropy);
    println!("{:?}", total_entropy);
}
