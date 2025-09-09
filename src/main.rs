use std::env;
use nn_entropy::{load_one_d_data, calculate_entropy_from_data};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_csv>", args[0]);
        std::process::exit(1);
    }
    let data_path = &args[1];

    let one_d_data: Vec<Vec<f64>> = load_one_d_data(data_path);
    let frames_end = one_d_data[0].len();

    let entropy = calculate_entropy_from_data(one_d_data, frames_end);
    println!("Total entropy = {}", entropy);
}
