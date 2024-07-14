use std::env;
use std::str::FromStr;

use utilities::{
    gen_random_signal,
    rustfft::{num_complex::Complex64, FftPlanner},
};

fn benchmark_rustfft(n: usize) {
    let big_n = 1 << n;

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

    let now = std::time::Instant::now();
    fft.process(&mut signal);
    let elapsed = now.elapsed().as_nanos();
    println!("{elapsed}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    benchmark_rustfft(n);
}
