use std::env;
use std::ptr::slice_from_raw_parts_mut;
use std::str::FromStr;

use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan32, C2CPlan64};
use fftw::types::{Flag, Sign};
use utilities::rustfft::num_complex::Complex;
use utilities::{gen_random_signal_f32, gen_random_signal_f64};

// See examples/benchmark.rs for the rationale behind batch-timing.
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
    let mut nums: AlignedVec<Complex<f32>> = AlignedVec::new(big_n);

    let mut plan =
        C2CPlan32::aligned(&[big_n], Sign::Forward, Flag::DESTROYINPUT | Flag::PATIENT).unwrap();

    let batch = batch_size(n).min(iterations.max(1));
    let batches = (iterations / batch).max(1);
    for _ in 0..batches {
        gen_random_signal_f32(&mut reals, &mut imags);
        nums.iter_mut()
            .zip(reals.iter())
            .zip(imags.iter())
            .for_each(|((z, &re), &im)| *z = Complex::new(re, im));

        let now = std::time::Instant::now();
        for _ in 0..batch {
            plan.c2c(
                // SAFETY: `nums` is alive for the whole call, big_n matches the
                // allocation size, and FFTW permits in-place execution via
                // DESTROYINPUT. The raw pointer is only used to form a disjoint
                // mutable view so the borrow checker accepts passing `&mut nums`
                // twice (input and output).
                unsafe { &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), big_n) },
                &mut nums,
            )
            .unwrap();
            std::hint::black_box(&mut nums);
        }
        let elapsed = now.elapsed().as_nanos();
        println!("{}", elapsed / batch as u128);
    }
}

fn bench64(n: usize, iterations: usize) {
    let big_n = 1usize << n;

    let mut reals = vec![0.0f64; big_n];
    let mut imags = vec![0.0f64; big_n];
    let mut nums: AlignedVec<Complex<f64>> = AlignedVec::new(big_n);

    let mut plan =
        C2CPlan64::aligned(&[big_n], Sign::Forward, Flag::DESTROYINPUT | Flag::PATIENT).unwrap();

    let batch = batch_size(n).min(iterations.max(1));
    let batches = (iterations / batch).max(1);
    for _ in 0..batches {
        gen_random_signal_f64(&mut reals, &mut imags);
        nums.iter_mut()
            .zip(reals.iter())
            .zip(imags.iter())
            .for_each(|((z, &re), &im)| *z = Complex::new(re, im));

        let now = std::time::Instant::now();
        for _ in 0..batch {
            plan.c2c(
                // SAFETY: see bench32 comment above.
                unsafe { &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), big_n) },
                &mut nums,
            )
            .unwrap();
            std::hint::black_box(&mut nums);
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
