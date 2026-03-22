use std::env;
use std::str::FromStr;

use utilities::gen_random_signal;
use utilities::rustfft::num_complex::Complex64;
use utilities::rustfft::num_traits::Zero;
use utilities::rustfft::FftPlanner;

fn benchmark_rustfft(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)

    let mut reals = vec![0.0f64; big_n];
    let mut imags = vec![0.0f64; big_n];

    gen_random_signal(&mut reals, &mut imags);
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
    println!("took {elapsed} for {iterations} iterations");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 3, "Usage {} <n> <iterations>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    let iterations = usize::from_str(&args[2]).unwrap();

    benchmark_rustfft(n, iterations);
}
