use std::env;
use std::ptr::slice_from_raw_parts_mut;
use std::str::FromStr;

use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan32, C2CPlan64};
use fftw::types::{Flag, Sign};
use utilities::rustfft::num_complex::Complex;
use utilities::{gen_random_signal_f32, gen_random_signal_f64};

fn benchmark_fftw_32(n: usize, iterations: usize) {
    let big_n = 1 << n; // 2.pow(n)

    let mut reals = vec![0.0f32; big_n];
    let mut imags = vec![0.0f32; big_n];

    gen_random_signal_f32(&mut reals, &mut imags);
    let mut nums = AlignedVec::new(big_n);
    reals
        .drain(..)
        .zip(imags.drain(..))
        .zip(nums.iter_mut())
        .for_each(|((re, im), z)| *z = Complex::new(re, im));

    let mut plan = C2CPlan32::aligned(
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
        // mark the result as used so that the compiler doesn't optimize out parts of FFT
        std::hint::black_box(&mut nums);
    }
    let elapsed = now.elapsed().as_nanos();
    let elapsed_per_iteration = elapsed / iterations as u128;
    println!("{elapsed_per_iteration}");
}

fn benchmark_fftw_64(n: usize, iterations: usize) {
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
    assert!(
        args.len() >= 4,
        "Usage {} <32|64> <n> <iterations>",
        args[0]
    );

    let n = usize::from_str(&args[2]).unwrap();
    let iterations = usize::from_str(&args[3]).unwrap();

    match args[1].as_str() {
        "32" => benchmark_fftw_32(n, iterations),
        "64" => benchmark_fftw_64(n, iterations),
        other => panic!("Invalid precision: {other}. Please pass 32 or 64"),
    }
}
