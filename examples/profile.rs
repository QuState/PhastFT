use std::env;
use std::str::FromStr;

use phastft::fft;
use phastft::planner::Planner;

fn benchmark_fft(num_qubits: usize) {
    let n = 1 << num_qubits;
    let mut reals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut imags: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut planner = Planner::new(n);
    fft(&mut reals, &mut imags, &mut planner);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    benchmark_fft(n);
}
