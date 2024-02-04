use numpy::PyReadwriteArray1;
use phastft::{fft as fft_rs, planner::Planner};
use pyo3::prelude::*;

#[pyfunction]
fn fft(mut reals: PyReadwriteArray1<f64>, mut imags: PyReadwriteArray1<f64>) {
    let mut planner = Planner::new(reals.len());
    fft_rs(
        reals.as_slice_mut().unwrap(),
        imags.as_slice_mut().unwrap(),
        &mut planner,
    );
}

/// A Python module implemented in Rust.
#[pymodule]
fn pybindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    Ok(())
}
