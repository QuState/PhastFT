use numpy::PyReadwriteArray1;
use phastft::{fft_64 as fft_64_rs, planner::Direction};
use pyo3::prelude::*;

#[pyfunction]
fn fft(mut reals: PyReadwriteArray1<f64>, mut imags: PyReadwriteArray1<f64>, direction: char) {
    assert!(direction == 'f' || direction == 'r');
    let dir = if direction == 'f' {
        Direction::Forward
    } else {
        Direction::Reverse
    };

    fft_64_rs(
        reals.as_slice_mut().unwrap(),
        imags.as_slice_mut().unwrap(),
        dir,
    );
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyphastft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    Ok(())
}
