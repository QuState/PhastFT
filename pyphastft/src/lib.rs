use numpy::PyReadwriteArray1;
use phastft::{fft_64 as fft_64_rs, planner::Direction};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Compute the Fast Fourier Transform of complex data.
///
/// Args:
///     reals: Real components of the input signal (modified in-place)
///     imags: Imaginary components of the input signal (modified in-place)
///     direction: 'f' for forward FFT, 'r' for reverse (inverse) FFT
///
/// Raises:
///     ValueError: If arrays have different lengths, are not power-of-2 sized,
///                 or direction is not 'f' or 'r'
#[pyfunction]
#[pyo3(text_signature = "(reals, imags, direction)")]
fn fft(mut reals: PyReadwriteArray1<f64>, mut imags: PyReadwriteArray1<f64>, direction: char) -> PyResult<()> {
    if direction != 'f' && direction != 'r' {
        return Err(PyValueError::new_err("direction must be 'f' (forward) or 'r' (reverse)"));
    }

    let reals_slice = reals.as_slice_mut()
        .map_err(|_| PyValueError::new_err("Failed to access real array as contiguous slice"))?;
    let imags_slice = imags.as_slice_mut()
        .map_err(|_| PyValueError::new_err("Failed to access imaginary array as contiguous slice"))?;

    if reals_slice.len() != imags_slice.len() {
        return Err(PyValueError::new_err("Real and imaginary arrays must have the same length"));
    }

    if !reals_slice.len().is_power_of_two() {
        return Err(PyValueError::new_err("Array length must be a power of 2"));
    }

    let dir = if direction == 'f' {
        Direction::Forward
    } else {
        Direction::Reverse
    };

    fft_64_rs(reals_slice, imags_slice, dir);
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyphastft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    Ok(())
}
