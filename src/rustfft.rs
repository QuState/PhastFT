use rustfft::{num_complex::Complex64, FftPlanner};
use spinoza::utils::pretty_print_int;

fn main() {
    let N = 25;
    println!("run RustFFT with {N} qubits");

    let n = 1 << N;
    let now = std::time::Instant::now();
    let mut buffer: Vec<Complex64> = (1..n + 1)
        .map(|i| Complex64::new(i as f64, i as f64))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(buffer.len());
    fft.process(&mut buffer);
    let elapsed = pretty_print_int(now.elapsed().as_micros());
    println!("time elapsed: {elapsed} us");
    //println!("{:?}", buffer);
}
