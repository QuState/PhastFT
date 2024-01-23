use num_complex::Complex;
use phastft::fft_dif;
use pyo3::prelude::*;
use spinoza::core::State;
use spinoza::math::Float;

#[pyfunction]
fn fft(reals: Vec<Float>, imags: Vec<Float>) -> (Vec<Float>, Vec<Float>) {
    let n = reals.len().ilog2() as u8;

    // println!("reals: {reals:?}");
    // println!("imags: {imags:?}");

    let mut state = State { reals, imags, n };

    let now = std::time::Instant::now();
    fft_dif(&mut state);
    let elapsed = now.elapsed().as_micros();
    println!("pure rust PhastFT call: {elapsed} us");

    // println!("{state}");

    (state.reals, state.imags)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pybindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    Ok(())
}
