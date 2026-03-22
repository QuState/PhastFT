use std::env;
use std::ptr::slice_from_raw_parts_mut;
use std::str::FromStr;

use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan64};
use fftw::types::{Flag, Sign};
use utilities::gen_random_signal_f64;
use utilities::rustfft::num_complex::Complex;

fn benchmark_fftw(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)

    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];

    gen_random_signal_f64(&mut reals, &mut imags);
    let mut nums = AlignedVec::new(big_n);
    reals
        .drain(..)
        .zip(imags.drain(..))
        .zip(nums.iter_mut())
        .for_each(|((re, im), z)| *z = Complex::new(re, im));

    let mut plan = C2CPlan64::aligned(
        &[big_n],
        Sign::Backward,
        Flag::DESTROYINPUT | Flag::ESTIMATE,
    )
    .unwrap();

    let now = std::time::Instant::now();
    for _ in 0..iterations {
        plan.c2c(
            // SAFETY: See above comment.
            unsafe { &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), big_n) },
            &mut nums,
        )
        .unwrap();
    }
    let elapsed = now.elapsed().as_nanos();
    let elapsed_per_iteration = elapsed / iterations as u128;
    println!("{elapsed_per_iteration}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 3, "Usage {} <n> <iterations>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    let iterations = usize::from_str(&args[2]).unwrap();

    benchmark_fftw(n, iterations);
}
