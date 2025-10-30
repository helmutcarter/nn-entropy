use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use crate::calculate_entropy_from_data;
use crate::estimate_coordinate_entropy_rust;
use crate::estimate_coordinate_mutual_information_rust;

/// Python wrapper around the main entropy function
#[pyfunction]
fn estimate_mie_entropy(data: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let array = data.as_array();
    let one_d_data: Vec<Vec<f64>> = array.outer_iter().map(|row| row.to_vec()).collect();
    let frames_end = array.shape()[1]; // use all frames
    Ok(calculate_entropy_from_data(one_d_data, frames_end))
}

// Create python wrapper to take BAT coordinates and return a numpy array with an entropy for each coordinate
#[pyfunction]
fn estimate_coordinate_entropy(data: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    let array = data.as_array();
    let one_d_data: Vec<Vec<f64>> = array.outer_iter().map(|row| row.to_vec()).collect();
    let frames_end = array.shape()[1]; // use all frames

    Ok(estimate_coordinate_entropy_rust(one_d_data, frames_end))
}

// Create python wrapper to take BAT coordinates and return a numpy array with the mutual information for each coordinate pair
#[pyfunction]
fn estimate_coordinate_mutual_information(data: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    let array = data.as_array();
    let one_d_data: Vec<Vec<f64>> = array.outer_iter().map(|row| row.to_vec()).collect();
    let frames_end = array.shape()[1]; // use all frames
    Ok(estimate_coordinate_mutual_information_rust(one_d_data, frames_end))
}

#[pymodule]
fn nn_entropy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_mie_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_mutual_information, m)?)?;
    Ok(())
}