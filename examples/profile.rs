use std::env;
use std::str::FromStr;

use phastft::fft::r2c_fft_f64;
use phastft::fft_64;
use phastft::planner::Direction;

#[allow(dead_code)]
fn benchmark_fft(num_qubits: usize) {
    let n = 1 << num_qubits;
    let mut reals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut imags: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    fft_64(&mut reals, &mut imags, Direction::Forward);
}

fn benchmark_r2c_fft(n: usize) {
    let big_n = 1 << n;
    let reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
    let mut output_re = vec![0.0; big_n];
    let mut output_im = vec![0.0; big_n];
    r2c_fft_f64(&reals, &mut output_re, &mut output_im);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    // benchmark_fft(n);
    benchmark_r2c_fft(n);
}
