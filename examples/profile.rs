use std::env;
use std::str::FromStr;

use phastft::fft_64;
use phastft::planner::Direction;

fn benchmark_fft(num_qubits: usize) {
    let n = 1 << num_qubits;
    let mut reals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut imags: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    fft_64(&mut reals, &mut imags, Direction::Forward);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    benchmark_fft(n);
}
