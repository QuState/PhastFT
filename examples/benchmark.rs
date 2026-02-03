use std::env;
use std::str::FromStr;

use phastft::fft_64_dit_with_planner;
use phastft::planner::{Direction, PlannerDit64};
use utilities::gen_random_signal;

fn benchmark_fft_64_dit(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal(&mut reals, &mut imags);

    // Pre-create planner for DIT
    let planner = PlannerDit64::new(reals.len(), Direction::Forward);

    let now = std::time::Instant::now();
    for _ in 0..iterations {
        fft_64_dit_with_planner(&mut reals, &mut imags, &planner);
    }
    let elapsed = now.elapsed().as_nanos();
    println!("took {elapsed} for {iterations} iterations");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 3, "Usage {} <n> <iterations>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    let iterations = usize::from_str(&args[2]).unwrap();

    benchmark_fft_64_dit(n, iterations);
}
