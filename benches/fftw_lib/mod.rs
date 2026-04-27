//! Shared FFTW C2C bench logic. Three `[[bench]]` binaries
//! (`fftw_estimate`, `fftw_measure`, `fftw_conserve`) each call `run_all`
//! with their own series ID and `Flag` set; the per-binary split exists
//! only so FFTW's per-process wisdom cache cannot leak between planning
//! modes (every run starts with a fresh process and empty wisdom).
//!
//! Like `common/mod.rs`, this is a plain shared module under `benches/`,
//! not a `[[bench]]` target — each consumer pulls it in with `mod
//! fftw_lib;` and compiles its own copy.

#![allow(dead_code)]

use std::ptr::slice_from_raw_parts_mut;

use criterion::{BatchSize, BenchmarkId, Criterion};
use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan32, C2CPlan64};
use fftw::types::{Flag, Sign};
use utilities::rustfft::num_complex::Complex;

use crate::common::{bench_at_sizes, groups, split_complex, throughput_complex, LENGTHS};

macro_rules! fftw_sweep {
    ($name:ident, $float:ty, $plan:ty, $sign:expr, $group:expr) => {
        fn $name(c: &mut Criterion, id: &str, flags: Flag) {
            // `Flag` (bitflags 2 in the fftw crate) does not derive `Copy`,
            // so we can't capture it by value across iterations of an FnMut
            // closure. Round-trip through the underlying `u32` bits, which
            // are Copy, and reconstruct the flag set per `n`.
            let flag_bits = flags.bits();
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                move |g, len| {
                    let mut plan =
                        <$plan>::aligned(&[len], $sign, Flag::from_bits_retain(flag_bits)).unwrap();
                    g.bench_function(BenchmarkId::new(id, len), |b| {
                        b.iter_batched(
                            || {
                                let (reals, imags) = split_complex::<$float>(len);
                                let mut nums: AlignedVec<Complex<$float>> = AlignedVec::new(len);
                                for (z, (&re, &im)) in
                                    nums.iter_mut().zip(reals.iter().zip(imags.iter()))
                                {
                                    *z = Complex::new(re, im);
                                }
                                nums
                            },
                            |mut nums| {
                                plan.c2c(
                                    // SAFETY: identical shape to examples/fftwrb.rs:42-48.
                                    // DESTROYINPUT permits in-place c2c; the raw slice
                                    // aliases `nums` for the duration of the call and
                                    // `len` matches the AlignedVec allocation.
                                    unsafe {
                                        &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), len)
                                    },
                                    &mut nums,
                                )
                                .unwrap();
                                std::hint::black_box(&mut nums);
                            },
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

fftw_sweep!(
    fwd_f32,
    f32,
    C2CPlan32,
    Sign::Forward,
    groups::C2C_FORWARD_F32
);
fftw_sweep!(
    inv_f32,
    f32,
    C2CPlan32,
    Sign::Backward,
    groups::C2C_INVERSE_F32
);
fftw_sweep!(
    fwd_f64,
    f64,
    C2CPlan64,
    Sign::Forward,
    groups::C2C_FORWARD_F64
);
fftw_sweep!(
    inv_f64,
    f64,
    C2CPlan64,
    Sign::Backward,
    groups::C2C_INVERSE_F64
);

/// Run all four c2c_* groups (forward/inverse × f32/f64) with the given
/// FFTW `flags` and series `id`. The three per-mode bench binaries each
/// call this once with their own `Flag` set.
pub fn run_all(c: &mut Criterion, id: &str, flags: Flag) {
    // `Flag` isn't `Copy`, so reconstruct from its u32 bits for each call.
    let bits = flags.bits();
    let mk = || Flag::from_bits_retain(bits);
    fwd_f32(c, id, mk());
    inv_f32(c, id, mk());
    fwd_f64(c, id, mk());
    inv_f64(c, id, mk());
}
