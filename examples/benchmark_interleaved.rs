use std::env;
use std::str::FromStr;

use num_complex::{Complex32, Complex64};
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_32_interleaved_with_planner, fft_64_interleaved_with_planner};
use utilities::{gen_random_signal_f32, gen_random_signal_f64};

fn benchmark_fft_32_dit(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal_f32(&mut reals, &mut imags);
    let mut signal = vec![Complex32::default(); big_n];
    reals
        .drain(..)
        .zip(imags.drain(..))
        .zip(signal.iter_mut())
        .for_each(|((re, im), z)| {
            z.re = re;
            z.im = im;
        });

    // Pre-create planner for DIT
    let planner = PlannerDit32::new(signal.len(), Direction::Forward);

    let now = std::time::Instant::now();
    for _ in 0..iterations {
        fft_32_interleaved_with_planner(&mut signal, &planner);
    }
    let elapsed = now.elapsed().as_nanos();
    let elapsed_per_iteration = elapsed / iterations as u128;
    println!("{elapsed_per_iteration}");
}

fn benchmark_fft_64_dit(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal_f64(&mut reals, &mut imags);
    let mut signal = vec![Complex64::default(); big_n];
    reals
        .drain(..)
        .zip(imags.drain(..))
        .zip(signal.iter_mut())
        .for_each(|((re, im), z)| {
            z.re = re;
            z.im = im;
        });

    // Pre-create planner for DIT
    let planner = PlannerDit64::new(signal.len(), Direction::Forward);

    let now = std::time::Instant::now();
    for _ in 0..iterations {
        fft_64_interleaved_with_planner(&mut signal, &planner);
    }
    let elapsed = now.elapsed().as_nanos();
    let elapsed_per_iteration = elapsed / iterations as u128;
    println!("{elapsed_per_iteration}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() >= 4,
        "Usage {} <32|64> <n> <iterations>",
        args[0]
    );

    let n = usize::from_str(&args[2]).unwrap();
    let iterations = usize::from_str(&args[3]).unwrap();

    match args[1].as_str() {
        "32" => benchmark_fft_32_dit(n, iterations),
        "64" => benchmark_fft_64_dit(n, iterations),
        other => panic!("Invalid precision: {other}. Please pass 32 or 64"),
    }
}
