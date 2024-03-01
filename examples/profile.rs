use std::env;
use std::str::FromStr;

use phastft::planner::Direction;
use phastft::{fft, Float};

fn benchmark_fft(num_qubits: usize) {
    let n = 1 << num_qubits;
    let mut reals: Vec<_> = (1..=n).map(|i| i as Float).collect();
    let mut imags: Vec<_> = (1..=n).map(|i| i as Float).collect();
    fft(&mut reals, &mut imags, Direction::Forward);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    benchmark_fft(n);
}
