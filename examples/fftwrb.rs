use std::{env, ptr::slice_from_raw_parts_mut, str::FromStr};

use fftw::{
    array::AlignedVec,
    plan::{C2CPlan, C2CPlan64},
    types::{Flag, Sign},
};
use utilities::{gen_random_signal, rustfft::num_complex::Complex};

fn benchmark_fftw(n: usize) {
    let big_n = 1 << n;

    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];

    gen_random_signal(&mut reals, &mut imags);
    let mut nums = AlignedVec::new(big_n);
    reals
        .drain(..)
        .zip(imags.drain(..))
        .zip(nums.iter_mut())
        .for_each(|((re, im), z)| *z = Complex::new(re, im));

    let now = std::time::Instant::now();
    C2CPlan64::aligned(
        &[big_n],
        Sign::Backward,
        Flag::DESTROYINPUT | Flag::ESTIMATE,
    )
    .unwrap()
    .c2c(
        // SAFETY: See above comment.
        unsafe { &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), big_n) },
        &mut nums,
    )
    .unwrap();
    let elapsed = now.elapsed().as_nanos();
    println!("{elapsed}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    benchmark_fftw(n);
}
