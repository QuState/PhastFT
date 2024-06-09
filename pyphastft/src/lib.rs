use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use phastft::{fft_64 as fft_64_rs, fft::r2c_fft_f64, planner::Direction};
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

#[pyfunction]
fn rfft(reals: PyReadonlyArray1<f64>, direction: char) -> (Vec<f64>, Vec<f64>) {
    assert!(direction == 'f' || direction == 'r');
    let _dir = if direction == 'f' {
        Direction::Forward
    } else {
        Direction::Reverse
    };
    
    let big_n = reals.as_slice().unwrap().len();

    let mut output_re = vec![0.0; big_n];
    let mut output_im = vec![0.0; big_n];
    r2c_fft_f64(reals.as_slice().unwrap(), &mut output_re, &mut output_im);
    (output_re, output_im)
} 


/// A Python module implemented in Rust.
#[pymodule]
fn pyphastft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    Ok(())
}
