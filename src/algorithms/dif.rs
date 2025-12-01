//! Decimation-in-Frequency (DIF) FFT Implementation
//!
//! The DIF algorithm applies butterflies that progressively increase from size 2^{0} up to size
//! `2^{log(N)}`, where `N` is the size of the input, and `log(N)` is the # of stages required to
//! process the input.
//!
//! Input is processed in natural order and output can be bit-reversed.
//!
//! ## Algorithm Overview
//!
//! 1. Start with butterflies with strides of 2^{log(N)}, where N is the size of the input.
//! 2. Works up to the last stage, with log(N) stages in total.
//! 3. Optionally apply bit-reversal at the end
//!
use crate::algorithms::cobra::cobra_apply;
use crate::kernels::common::{fft_chunk_2, fft_chunk_4};
use crate::kernels::dif::{fft_32_chunk_n_simd, fft_64_chunk_n_simd, fft_chunk_n};
use crate::options::Options;
use crate::parallel::parallel_join;
use crate::planner::{Direction, Planner32, Planner64};
use crate::twiddles::filter_twiddles;

/// DIF FFT for f64 with options and pre-computed planner
///
/// This is the core DIF implementation that processes data from large butterflies
/// to small, optionally applying bit-reversal at the end.
///
/// # Arguments
///
/// * `reals` - Real components of the signal (modified in-place)
/// * `imags` - Imaginary components of the signal (modified in-place)
/// * `opts` - Options controlling optimization strategies
/// * `planner` - Pre-computed planner with twiddle factors
///
/// # Panics
///
/// Panics if input length is not a power of 2 or if real and imaginary arrays have different lengths
#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
            "x86_64+avx2+fma", // x86_64-v3
            "x86_64+sse4.2", // x86_64-v2
            "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
            "x86+avx2+fma",
            "x86+sse4.2",
            "x86+sse2",
))]
pub fn fft_64_with_opts_and_plan(
    reals: &mut [f64],
    imags: &mut [f64],
    opts: &Options,
    planner: &Planner64,
) {
    assert!(reals.len() == imags.len() && reals.len().is_power_of_two());
    let n: usize = reals.len().ilog2() as usize;

    let twiddles_re = &planner.twiddles_re;
    let twiddles_im = &planner.twiddles_im;

    assert!(twiddles_re.len() == reals.len() / 2 && twiddles_im.len() == imags.len() / 2);

    // Handle inverse transform
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    };

    // First stage - no twiddle filtering needed
    let dist = 1 << (n - 1);
    let chunk_size = dist << 1;

    if chunk_size > 4 {
        if chunk_size >= 8 * 2 {
            fft_64_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
        } else {
            fft_chunk_n(reals, imags, twiddles_re, twiddles_im, dist);
        }
    } else if chunk_size == 4 {
        fft_chunk_4(reals, imags);
    } else if chunk_size == 2 {
        fft_chunk_2(reals, imags);
    }

    let (mut filtered_twiddles_re, mut filtered_twiddles_im) =
        filter_twiddles(twiddles_re, twiddles_im);

    // Subsequent stages with filtered twiddles
    for t in (0..n - 1).rev() {
        let dist = 1 << t;
        let chunk_size = dist << 1;

        if chunk_size > 4 {
            if chunk_size >= 8 * 2 {
                fft_64_chunk_n_simd(
                    reals,
                    imags,
                    &filtered_twiddles_re,
                    &filtered_twiddles_im,
                    dist,
                );
            } else {
                fft_chunk_n(
                    reals,
                    imags,
                    &filtered_twiddles_re,
                    &filtered_twiddles_im,
                    dist,
                );
            }
        } else if chunk_size == 4 {
            fft_chunk_4(reals, imags);
        } else if chunk_size == 2 {
            fft_chunk_2(reals, imags);
        }

        (filtered_twiddles_re, filtered_twiddles_im) =
            filter_twiddles(&filtered_twiddles_re, &filtered_twiddles_im);
    }

    // Optional bit reversal (controlled by options)
    if opts.dif_perform_bit_reversal {
        parallel_join(
            opts.multithreaded_bit_reversal,
            || cobra_apply(reals, n),
            || cobra_apply(imags, n),
        );
    }

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = (reals.len() as f64).recip();
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    }
}

/// DIF FFT for f32 with options and pre-computed planner
///
/// This is the core DIF implementation for single-precision floating point.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_32_with_opts_and_plan(
    reals: &mut [f32],
    imags: &mut [f32],
    opts: &Options,
    planner: &Planner32,
) {
    assert!(reals.len() == imags.len() && reals.len().is_power_of_two());
    let n: usize = reals.len().ilog2() as usize;

    let twiddles_re = &planner.twiddles_re;
    let twiddles_im = &planner.twiddles_im;

    assert!(twiddles_re.len() == reals.len() / 2 && twiddles_im.len() == imags.len() / 2);

    // Handle inverse transform
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    }

    // First stage - no twiddle filtering needed
    let dist = 1 << (n - 1);
    let chunk_size = dist << 1;

    if chunk_size > 4 {
        if chunk_size >= 16 * 2 {
            fft_32_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
        } else {
            fft_chunk_n(reals, imags, twiddles_re, twiddles_im, dist);
        }
    } else if chunk_size == 4 {
        fft_chunk_4(reals, imags);
    } else if chunk_size == 2 {
        fft_chunk_2(reals, imags);
    }

    let (mut filtered_twiddles_re, mut filtered_twiddles_im) =
        filter_twiddles(twiddles_re, twiddles_im);

    // Subsequent stages with filtered twiddles
    for t in (0..n - 1).rev() {
        let dist = 1 << t;
        let chunk_size = dist << 1;

        if chunk_size > 4 {
            if chunk_size >= 16 * 2 {
                fft_32_chunk_n_simd(
                    reals,
                    imags,
                    &filtered_twiddles_re,
                    &filtered_twiddles_im,
                    dist,
                );
            } else {
                fft_chunk_n(
                    reals,
                    imags,
                    &filtered_twiddles_re,
                    &filtered_twiddles_im,
                    dist,
                );
            }
        } else if chunk_size == 4 {
            fft_chunk_4(reals, imags);
        } else if chunk_size == 2 {
            fft_chunk_2(reals, imags);
        }

        (filtered_twiddles_re, filtered_twiddles_im) =
            filter_twiddles(&filtered_twiddles_re, &filtered_twiddles_im);
    }

    if opts.dif_perform_bit_reversal {
        parallel_join(
            opts.multithreaded_bit_reversal,
            || cobra_apply(reals, n),
            || cobra_apply(imags, n),
        );
    }

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = (reals.len() as f32).recip();
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    };
}
