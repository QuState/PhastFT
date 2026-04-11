use std::env;
use std::str::FromStr;

use utilities::rustfft::num_complex::{Complex32, Complex64};
use utilities::rustfft::num_traits::Zero;
use utilities::rustfft::FftPlanner;
use utilities::{gen_random_signal_f32, gen_random_signal_f64};

fn benchmark_rustfft_32(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)

    let mut reals = vec![0.0f32; big_n];
    let mut imags = vec![0.0f32; big_n];

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

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut scratch = vec![Complex32::zero(); fft.get_inplace_scratch_len()];

    let now = std::time::Instant::now();
    for _ in 0..iterations {
        fft.process_with_scratch(&mut signal, scratch.as_mut_slice());
        // mark the result as used so that the compiler doesn't optimize out parts of FFT
        std::hint::black_box(&mut signal);
    }
    let elapsed = now.elapsed().as_nanos();
    let elapsed_per_iteration = elapsed / iterations as u128;
    println!("{elapsed_per_iteration}");
}

fn benchmark_rustfft_64(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)

    let mut reals = vec![0.0f64; big_n];
    let mut imags = vec![0.0f64; big_n];

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

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut scratch = vec![Complex64::zero(); fft.get_inplace_scratch_len()];

    let now = std::time::Instant::now();
    for _ in 0..iterations {
        fft.process_with_scratch(&mut signal, scratch.as_mut_slice());
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
        "32" => benchmark_rustfft_32(n, iterations),
        "64" => benchmark_rustfft_64(n, iterations),
        other => panic!("Invalid precision: {other}. Please pass 32 or 64"),
    }
}
