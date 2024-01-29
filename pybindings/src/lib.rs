use numpy::PyReadwriteArray1;
use phastft::fft_dif;
use pyo3::prelude::*;

#[pyfunction]
fn fft(mut reals: PyReadwriteArray1<f64>, mut imags: PyReadwriteArray1<f64>) {
    fft_dif(reals.as_slice_mut().unwrap(), imags.as_slice_mut().unwrap());
}

/// A Python module implemented in Rust.
#[pymodule]
fn pybindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    Ok(())
}
