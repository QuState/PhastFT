use rustfft::{num_complex::Complex64, FftPlanner};

fn main() {
    let n = 25;

    let mut buffer: Vec<Complex64> = (1..n + 1)
        .map(|i| Complex64::new(i as f64, i as f64))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(buffer.len());
    fft.process(&mut buffer);
}
