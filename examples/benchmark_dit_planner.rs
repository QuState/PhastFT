use std::env;
use std::str::FromStr;

use phastft::planner::{Direction, PlannerDit64};
use phastft::{fft_64_dit, fft_64_dit_with_planner};
use utilities::gen_random_signal;

fn benchmark_dit_with_planner(n: usize, iterations: usize) {
    let big_n = 1 << n;

    // Create planner once
    let planner = PlannerDit64::new(big_n, Direction::Forward);

    let mut total_time = 0u128;

    for _ in 0..iterations {
        let mut reals = vec![0.0; big_n];
        let mut imags = vec![0.0; big_n];
        gen_random_signal(&mut reals, &mut imags);

        let now = std::time::Instant::now();
        fft_64_dit_with_planner(&mut reals, &mut imags, &planner);
        total_time += now.elapsed().as_nanos();
    }

    println!(
        "With planner (avg over {} runs): {} ns",
        iterations,
        total_time / iterations as u128
    );
}

fn benchmark_dit_without_planner(n: usize, iterations: usize) {
    let big_n = 1 << n;
    let mut total_time = 0u128;

    for _ in 0..iterations {
        let mut reals = vec![0.0; big_n];
        let mut imags = vec![0.0; big_n];
        gen_random_signal(&mut reals, &mut imags);

        let now = std::time::Instant::now();
        fft_64_dit(&mut reals, &mut imags, Direction::Forward);
        total_time += now.elapsed().as_nanos();
    }

    println!(
        "Without planner (avg over {} runs): {} ns",
        iterations,
        total_time / iterations as u128
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n = if args.len() > 1 {
        usize::from_str(&args[1]).unwrap_or(20)
    } else {
        20
    };

    let iterations = if args.len() > 2 {
        usize::from_str(&args[2]).unwrap_or(10)
    } else {
        10
    };

    println!("Benchmarking DIT FFT for size 2^{} = {}", n, 1 << n);
    println!("Running {} iterations each\n", iterations);

    benchmark_dit_with_planner(n, iterations);
    benchmark_dit_without_planner(n, iterations);
}
