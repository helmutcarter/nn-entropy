#![allow(unsafe_op_in_unsafe_fn)]
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use std::path::Path;
use crate::calculate_entropy_from_data;
use crate::estimate_coordinate_entropy_rust;
use crate::estimate_coordinate_mutual_information_rust;
use crate::bat_library::InternalCoordinates;

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

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

/// Python wrapper to read .parm7 + .nc and compute entropy directly
#[pyfunction]
fn estimate_mie_entropy_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
) -> PyResult<f64> {
    let top = Path::new(top_path);
    let traj = Path::new(traj_path);
    let start = start.unwrap_or(0);
    let torsions_only = torsions_only.unwrap_or(false);

    let mut internal = InternalCoordinates::new(top).map_err(to_py_err)?;
    internal
        .calculate_internal_coords(traj, stop.unwrap_or(usize::MAX), torsions_only)
        .map_err(to_py_err)?;

    let frame_count = internal.int_coords.len();
    if frame_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "no frames read from trajectory",
        ));
    }
    if start >= frame_count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start is beyond available frames",
        ));
    }
    let dim = internal.int_coords[0].len();

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frame_count); dim];
    for frame in &internal.int_coords[start..] {
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    let used_frames = one_d_data[0].len();
    Ok(calculate_entropy_from_data(one_d_data, used_frames))
}

#[pymodule]
fn nn_entropy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_mie_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_mie_entropy_from_files, m)?)?;
    Ok(())
}
