use std::env;
use std::str::FromStr;

use phastft::fft_64_dit;
use phastft::planner::Direction;
use utilities::gen_random_signal_f64;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    let big_n = 1 << n; // 2.pow(n)
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal_f64(&mut reals, &mut imags);

    fft_64_dit(&mut reals, &mut imags, Direction::Forward);
}
