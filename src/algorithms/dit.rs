//! Decimation-in-Time (DIT) FFT Implementation
//!
//! The DIT algorithm decomposes the DFT from small to large sub-problems. Input is processed in
//! bit-reversed order, and output is produced in natural order.
//!
//! ## Algorithm Overview
//!
//! 1. Apply bit-reversal to input data
//! 2. Start with small butterflies (size 2)
//! 3. Work up to stage `log(N)`, where `N` is the size of the input.
//!
//! ## Memory Access Pattern
//!
//! DIT starts with fine-grained memory access and progressively works with
//! larger contiguous chunks.
//!
use crate::cobra::cobra_apply;
use crate::kernels::common::{fft_dit_chunk_2, fft_dit_chunk_4};
use crate::kernels::dit::{
    fft_dit_32_chunk_n_simd, fft_dit_64_chunk_n_simd, fft_dit_chunk_16_simd_f32,
    fft_dit_chunk_16_simd_f64, fft_dit_chunk_32_simd_f32, fft_dit_chunk_32_simd_f64,
    fft_dit_chunk_64_simd_f32, fft_dit_chunk_64_simd_f64, fft_dit_chunk_8_simd_f32,
    fft_dit_chunk_8_simd_f64,
};
use crate::options::Options;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};

/// DIT FFT for f64 with pre-computed planner and options
///
/// This implementation uses the Decimation-in-Time algorithm which:
/// - Requires bit-reversed input (performed automatically)
/// - Produces output in natural order
/// - Processes from small butterflies to large
///
/// # Arguments
///
/// * `reals` - Real components of the signal (modified in-place)
/// * `imags` - Imaginary components of the signal (modified in-place)
/// * `planner` - Pre-computed planner with twiddle factors
/// * `opts` - Options controlling optimization strategies
///
/// # Panics
///
/// Panics if input length is not a power of 2 or if real and imaginary arrays have different lengths
///
pub fn fft_64_dit_with_planner_and_opts(
    reals: &mut [f64],
    imags: &mut [f64],
    planner: &PlannerDit64,
    opts: &Options,
) {
    assert_eq!(reals.len(), imags.len());
    assert!(reals.len().is_power_of_two());

    let n = reals.len();
    let log_n = n.ilog2() as usize;
    assert_eq!(log_n, planner.log_n);

    // DIT requires bit-reversed input
    if opts.multithreaded_bit_reversal {
        std::thread::scope(|s| {
            s.spawn(|| cobra_apply(reals, log_n));
            s.spawn(|| cobra_apply(imags, log_n));
        });
    } else {
        cobra_apply(reals, log_n);
        cobra_apply(imags, log_n);
    }

    // Handle inverse FFT
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    }

    // DIT processes from small to large butterflies
    let mut stage_twiddle_idx = 0;
    for stage in 0..log_n {
        let dist = 1 << stage;
        let chunk_size = dist << 1;

        if chunk_size == 2 {
            fft_dit_chunk_2(reals, imags);
        } else if chunk_size == 4 {
            fft_dit_chunk_4(reals, imags);
        } else if chunk_size == 8 {
            fft_dit_chunk_8_simd_f64(reals, imags);
        } else if chunk_size == 16 {
            fft_dit_chunk_16_simd_f64(reals, imags);
        } else if chunk_size == 32 {
            fft_dit_chunk_32_simd_f64(reals, imags);
        } else if chunk_size == 64 {
            fft_dit_chunk_64_simd_f64(reals, imags);
        } else {
            // For larger chunks, use general kernel with twiddles from planner
            let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
            fft_dit_64_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
            stage_twiddle_idx += 1;
        }
    }

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = 1.0 / n as f64;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    }
}

/// DIT FFT for f32 with pre-computed planner and options
///
/// Single-precision version of the DIT FFT algorithm.
/// See [`fft_64_dit_with_planner_and_opts`] for `f64` version.
pub fn fft_32_dit_with_planner_and_opts(
    reals: &mut [f32],
    imags: &mut [f32],
    planner: &PlannerDit32,
    opts: &Options,
) {
    assert_eq!(reals.len(), imags.len());
    assert!(reals.len().is_power_of_two());

    let n = reals.len();
    let log_n = n.ilog2() as usize;
    assert_eq!(log_n, planner.log_n);

    // DIT requires bit-reversed input
    if opts.multithreaded_bit_reversal {
        std::thread::scope(|s| {
            s.spawn(|| cobra_apply(reals, log_n));
            s.spawn(|| cobra_apply(imags, log_n));
        });
    } else {
        cobra_apply(reals, log_n);
        cobra_apply(imags, log_n);
    }

    // Handle inverse FFT
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    }

    // DIT processes from small to large butterflies
    let mut stage_twiddle_idx = 0;
    for stage in 0..log_n {
        let dist = 1 << stage;
        let chunk_size = dist << 1;

        if chunk_size == 2 {
            fft_dit_chunk_2(reals, imags);
        } else if chunk_size == 4 {
            fft_dit_chunk_4(reals, imags);
        } else if chunk_size == 8 {
            fft_dit_chunk_8_simd_f32(reals, imags);
        } else if chunk_size == 16 {
            fft_dit_chunk_16_simd_f32(reals, imags);
        } else if chunk_size == 32 {
            fft_dit_chunk_32_simd_f32(reals, imags);
        } else if chunk_size == 64 {
            fft_dit_chunk_64_simd_f32(reals, imags);
        } else {
            // For larger chunks, use general kernel with twiddles from planner
            let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
            fft_dit_32_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
            stage_twiddle_idx += 1;
        }
    }

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = 1.0 / n as f32;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    }
}
