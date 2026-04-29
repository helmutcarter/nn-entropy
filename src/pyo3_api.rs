#![allow(unsafe_op_in_unsafe_fn)]
use crate::bat_library::InternalCoordinates;
use crate::calculate_entropy_from_data_with_order;
use crate::estimate_coordinate_entropy_rust;
use crate::estimate_coordinate_mie_entropy_rust;
use crate::estimate_coordinate_mutual_information_rust;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use std::path::Path;

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

fn array_to_one_d_data(data: PyReadonlyArray2<f64>) -> (Vec<Vec<f64>>, usize) {
    let array = data.as_array();
    let one_d_data: Vec<Vec<f64>> = array.outer_iter().map(|row| row.to_vec()).collect();
    let frames_end = array.shape()[1];
    (one_d_data, frames_end)
}

fn one_d_data_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
) -> PyResult<(Vec<Vec<f64>>, usize)> {
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
    if dim == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "no internal coordinates were generated",
        ));
    }

    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(frame_count - start); dim];
    for frame in &internal.int_coords[start..] {
        if frame.len() != dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "inconsistent internal coordinate dimensions",
            ));
        }
        for (i, value) in frame.iter().enumerate() {
            one_d_data[i].push(*value);
        }
    }

    let used_frames = one_d_data[0].len();
    Ok((one_d_data, used_frames))
}

/// Python wrapper around the main entropy function
#[pyfunction(signature = (data, mie_order=None))]
fn estimate_mie_entropy(data: PyReadonlyArray2<f64>, mie_order: Option<usize>) -> PyResult<f64> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    calculate_entropy_from_data_with_order(one_d_data, frames_end, mie_order.unwrap_or(2))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

// Create python wrapper to take BAT coordinates and return a numpy array with an entropy for each coordinate
#[pyfunction]
fn estimate_coordinate_entropy(data: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    estimate_coordinate_entropy_rust(one_d_data, frames_end)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

// Create python wrapper to take BAT coordinates and return a numpy array with the mutual information for each coordinate pair
#[pyfunction]
fn estimate_coordinate_mutual_information(data: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    estimate_coordinate_mutual_information_rust(one_d_data, frames_end)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

// Create python wrapper to return each coordinate's second-order MIE entropy contribution.
#[pyfunction]
fn estimate_coordinate_mie_entropy(data: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    estimate_coordinate_mie_entropy_rust(one_d_data, frames_end)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Python wrapper to read .parm7 + .nc and compute entropy directly
#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, mie_order=None))]
fn estimate_mie_entropy_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    mie_order: Option<usize>,
) -> PyResult<f64> {
    let (one_d_data, used_frames) =
        one_d_data_from_files(top_path, traj_path, start, stop, torsions_only)?;
    calculate_entropy_from_data_with_order(one_d_data, used_frames, mie_order.unwrap_or(2))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None))]
fn estimate_coordinate_entropy_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
) -> PyResult<Vec<f64>> {
    let (one_d_data, used_frames) =
        one_d_data_from_files(top_path, traj_path, start, stop, torsions_only)?;
    estimate_coordinate_entropy_rust(one_d_data, used_frames)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None))]
fn estimate_coordinate_mutual_information_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
) -> PyResult<Vec<f64>> {
    let (one_d_data, used_frames) =
        one_d_data_from_files(top_path, traj_path, start, stop, torsions_only)?;
    estimate_coordinate_mutual_information_rust(one_d_data, used_frames)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None))]
fn estimate_coordinate_mie_entropy_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
) -> PyResult<Vec<f64>> {
    let (one_d_data, used_frames) =
        one_d_data_from_files(top_path, traj_path, start, stop, torsions_only)?;
    estimate_coordinate_mie_entropy_rust(one_d_data, used_frames)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pymodule]
fn nn_entropy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_mie_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_mie_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_mie_entropy_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_entropy_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimate_coordinate_mutual_information_from_files,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimate_coordinate_mie_entropy_from_files,
        m
    )?)?;
    Ok(())
}
