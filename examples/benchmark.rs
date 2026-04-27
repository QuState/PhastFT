use std::env;
use std::str::FromStr;

use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_32_dit_with_planner, fft_64_dit_with_planner};
use utilities::{gen_random_signal_f32, gen_random_signal_f64};

// macOS monotonic clock ticks at ~41 ns. At small n a single FFT is
// sub-tick, so we amortize by timing a batch of FFTs under one pair of
// Instant::now() calls and reporting the average.
fn batch_size(n: usize) -> usize {
    match n {
        0..=7 => 256,
        8..=9 => 32,
        _ => 1,
    }
}

fn bench32(n: usize, iterations: usize) {
    let big_n = 1usize << n;
    let mut reals = vec![0.0f32; big_n];
    let mut imags = vec![0.0f32; big_n];

    let planner = PlannerDit32::new(big_n);

    let batch = batch_size(n).min(iterations.max(1));
    let batches = (iterations / batch).max(1);
    for _ in 0..batches {
        gen_random_signal_f32(&mut reals, &mut imags);

        let now = std::time::Instant::now();
        for _ in 0..batch {
            fft_32_dit_with_planner(&mut reals, &mut imags, Direction::Forward, &planner);
            std::hint::black_box(&mut reals);
            std::hint::black_box(&mut imags);
        }
        let elapsed = now.elapsed().as_nanos();
        println!("{}", elapsed / batch as u128);
    }
}

fn bench64(n: usize, iterations: usize) {
    let big_n = 1usize << n;
    let mut reals = vec![0.0f64; big_n];
    let mut imags = vec![0.0f64; big_n];

    let planner = PlannerDit64::new(big_n);

    let batch = batch_size(n).min(iterations.max(1));
    let batches = (iterations / batch).max(1);
    for _ in 0..batches {
        gen_random_signal_f64(&mut reals, &mut imags);

        let now = std::time::Instant::now();
        for _ in 0..batch {
            fft_64_dit_with_planner(&mut reals, &mut imags, Direction::Forward, &planner);
            std::hint::black_box(&mut reals);
            std::hint::black_box(&mut imags);
        }
        let elapsed = now.elapsed().as_nanos();
        println!("{}", elapsed / batch as u128);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() >= 4,
        "Usage: {} <32|64> <n> <iterations>",
        args[0]
    );

    let n = usize::from_str(&args[2]).unwrap();
    let iterations = usize::from_str(&args[3]).unwrap();

    match args[1].as_str() {
        "32" => bench32(n, iterations),
        "64" => bench64(n, iterations),
        other => panic!("Invalid precision: {other}. Use 32 or 64"),
    }
}
