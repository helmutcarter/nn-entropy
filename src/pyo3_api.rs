#![allow(unsafe_op_in_unsafe_fn)]
use crate::CoordinateMetric;
use crate::bat_library::InternalCoordinates;
use crate::calculate_entropy_from_data_with_metrics;
use crate::estimate_coordinate_entropy_with_metrics;
use crate::estimate_coordinate_mie_entropy_with_metrics;
use crate::estimate_coordinate_mutual_information_with_metrics;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyAny;
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

fn metrics_from_periods(
    periods: Option<Vec<Option<f64>>>,
    dimensions: usize,
) -> PyResult<Vec<CoordinateMetric>> {
    let Some(periods) = periods else {
        return Ok(vec![CoordinateMetric::Linear; dimensions]);
    };
    if periods.len() != dimensions {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "received {} periods for {dimensions} coordinates",
            periods.len()
        )));
    }
    periods
        .into_iter()
        .enumerate()
        .map(|(index, period)| match period {
            None => Ok(CoordinateMetric::Linear),
            Some(period) if period.is_finite() && period > 0.0 => {
                Ok(CoordinateMetric::Periodic { period })
            }
            Some(_) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "period for coordinate {index} must be finite and positive"
            ))),
        })
        .collect()
}

fn one_d_data_from_files(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, usize, Vec<CoordinateMetric>)> {
    let top = Path::new(top_path);
    let traj = Path::new(traj_path);
    let start = start.unwrap_or(0);
    let torsions_only = torsions_only.unwrap_or(false);
    let stride = stride.unwrap_or(1);
    if stride == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stride must be at least 1",
        ));
    }

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

    let metrics = internal.coordinate_metrics();
    let selected_frames = (frame_count - start).div_ceil(stride);
    let mut one_d_data: Vec<Vec<f64>> = vec![Vec::with_capacity(selected_frames); dim];
    for frame in internal.int_coords.iter().skip(start).step_by(stride) {
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
    Ok((one_d_data, used_frames, metrics))
}

fn one_d_data_from_py(
    data: &Bound<'_, PyAny>,
    periods: Option<Vec<Option<f64>>>,
) -> PyResult<(Vec<Vec<f64>>, usize, Vec<CoordinateMetric>)> {
    if let Ok(system) = data.extract::<PyRef<'_, System>>() {
        if periods.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "periods cannot override metrics stored in a System",
            ));
        }
        return Ok((
            system.one_d_data.clone(),
            system.frames,
            system.metrics.clone(),
        ));
    }

    let array = data.extract::<PyReadonlyArray2<f64>>()?;
    let (one_d_data, frames) = array_to_one_d_data(array);
    let metrics = metrics_from_periods(periods, one_d_data.len())?;
    Ok((one_d_data, frames, metrics))
}

#[pyclass]
struct System {
    one_d_data: Vec<Vec<f64>>,
    frames: usize,
    metrics: Vec<CoordinateMetric>,
}

#[pymethods]
impl System {
    #[getter]
    fn frames(&self) -> usize {
        self.frames
    }

    #[getter]
    fn coordinates(&self) -> usize {
        self.one_d_data.len()
    }

    #[getter]
    fn data(&self) -> Vec<Vec<f64>> {
        self.one_d_data.clone()
    }

    #[getter]
    fn periods(&self) -> Vec<Option<f64>> {
        self.metrics
            .iter()
            .map(|metric| match metric {
                CoordinateMetric::Linear => None,
                CoordinateMetric::Periodic { period } => Some(*period),
            })
            .collect()
    }

    #[pyo3(signature = (mie_order=None))]
    fn estimate_entropy(&self, py: Python<'_>, mie_order: Option<usize>) -> PyResult<f64> {
        let one_d_data = self.one_d_data.clone();
        let frames = self.frames;
        let metrics = self.metrics.clone();
        py.allow_threads(move || {
            calculate_entropy_from_data_with_metrics(
                one_d_data,
                frames,
                mie_order.unwrap_or(2),
                &metrics,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn estimate_coordinate_entropy(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let one_d_data = self.one_d_data.clone();
        let frames = self.frames;
        let metrics = self.metrics.clone();
        py.allow_threads(move || {
            estimate_coordinate_entropy_with_metrics(one_d_data, frames, &metrics)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn estimate_coordinate_mutual_information(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let one_d_data = self.one_d_data.clone();
        let frames = self.frames;
        let metrics = self.metrics.clone();
        py.allow_threads(move || {
            estimate_coordinate_mutual_information_with_metrics(one_d_data, frames, &metrics)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn estimate_coordinate_mie_entropy(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let one_d_data = self.one_d_data.clone();
        let frames = self.frames;
        let metrics = self.metrics.clone();
        py.allow_threads(move || {
            estimate_coordinate_mie_entropy_with_metrics(one_d_data, frames, &metrics)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None))]
fn load_system(
    py: Python<'_>,
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
) -> PyResult<System> {
    let (one_d_data, frames, metrics) = py.allow_threads(move || {
        one_d_data_from_files(top_path, traj_path, start, stop, torsions_only, stride)
    })?;
    Ok(System {
        one_d_data,
        frames,
        metrics,
    })
}

#[pyfunction(signature = (data, mie_order=None, periods=None))]
fn estimate_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    mie_order: Option<usize>,
    periods: Option<Vec<Option<f64>>>,
) -> PyResult<f64> {
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods)?;
    py.allow_threads(move || {
        calculate_entropy_from_data_with_metrics(
            one_d_data,
            frames_end,
            mie_order.unwrap_or(2),
            &metrics,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

/// Python wrapper around the main entropy function
#[pyfunction(signature = (data, mie_order=None, periods=None))]
fn estimate_mie_entropy(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    mie_order: Option<usize>,
    periods: Option<Vec<Option<f64>>>,
) -> PyResult<f64> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    let metrics = metrics_from_periods(periods, one_d_data.len())?;
    py.allow_threads(move || {
        calculate_entropy_from_data_with_metrics(
            one_d_data,
            frames_end,
            mie_order.unwrap_or(2),
            &metrics,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

// Create python wrapper to take BAT coordinates and return a numpy array with an entropy for each coordinate
#[pyfunction(signature = (data, periods=None))]
fn estimate_coordinate_entropy(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    periods: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    let metrics = metrics_from_periods(periods, one_d_data.len())?;
    py.allow_threads(move || {
        estimate_coordinate_entropy_with_metrics(one_d_data, frames_end, &metrics)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

// Create python wrapper to take BAT coordinates and return a numpy array with the mutual information for each coordinate pair
#[pyfunction(signature = (data, periods=None))]
fn estimate_coordinate_mutual_information(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    periods: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    let metrics = metrics_from_periods(periods, one_d_data.len())?;
    py.allow_threads(move || {
        estimate_coordinate_mutual_information_with_metrics(one_d_data, frames_end, &metrics)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

// Create python wrapper to return each coordinate's second-order MIE entropy contribution.
#[pyfunction(signature = (data, periods=None))]
fn estimate_coordinate_mie_entropy(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    periods: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end) = array_to_one_d_data(data);
    let metrics = metrics_from_periods(periods, one_d_data.len())?;
    py.allow_threads(move || {
        estimate_coordinate_mie_entropy_with_metrics(one_d_data, frames_end, &metrics)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

/// Python wrapper to read .parm7 + .nc and compute entropy directly
#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, mie_order=None, stride=None))]
#[allow(clippy::too_many_arguments)]
fn estimate_mie_entropy_from_files(
    py: Python<'_>,
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    mie_order: Option<usize>,
    stride: Option<usize>,
) -> PyResult<f64> {
    py.allow_threads(move || {
        let (one_d_data, used_frames, metrics) =
            one_d_data_from_files(top_path, traj_path, start, stop, torsions_only, stride)?;
        calculate_entropy_from_data_with_metrics(
            one_d_data,
            used_frames,
            mie_order.unwrap_or(2),
            &metrics,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None))]
fn estimate_coordinate_entropy_from_files(
    py: Python<'_>,
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
) -> PyResult<Vec<f64>> {
    py.allow_threads(move || {
        let (one_d_data, used_frames, metrics) =
            one_d_data_from_files(top_path, traj_path, start, stop, torsions_only, stride)?;
        estimate_coordinate_entropy_with_metrics(one_d_data, used_frames, &metrics)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None))]
fn estimate_coordinate_mutual_information_from_files(
    py: Python<'_>,
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
) -> PyResult<Vec<f64>> {
    py.allow_threads(move || {
        let (one_d_data, used_frames, metrics) =
            one_d_data_from_files(top_path, traj_path, start, stop, torsions_only, stride)?;
        estimate_coordinate_mutual_information_with_metrics(one_d_data, used_frames, &metrics)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None))]
fn estimate_coordinate_mie_entropy_from_files(
    py: Python<'_>,
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
) -> PyResult<Vec<f64>> {
    py.allow_threads(move || {
        let (one_d_data, used_frames, metrics) =
            one_d_data_from_files(top_path, traj_path, start, stop, torsions_only, stride)?;
        estimate_coordinate_mie_entropy_with_metrics(one_d_data, used_frames, &metrics)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

#[pymodule]
fn nn_entropy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<System>()?;
    m.add_function(wrap_pyfunction!(load_system, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_entropy, m)?)?;
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
