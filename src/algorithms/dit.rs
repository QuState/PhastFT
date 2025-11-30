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
use crate::algorithms::cobra::cobra_apply;
use crate::kernels::dit::{
    fft_dit_32_chunk_n_simd, fft_dit_64_chunk_n_simd, fft_dit_chunk_16_simd_f32,
    fft_dit_chunk_16_simd_f64, fft_dit_chunk_2, fft_dit_chunk_32_simd_f32,
    fft_dit_chunk_32_simd_f64, fft_dit_chunk_4_simd_f32, fft_dit_chunk_4_simd_f64,
    fft_dit_chunk_64_simd_f32, fft_dit_chunk_64_simd_f64, fft_dit_chunk_8_simd_f32,
    fft_dit_chunk_8_simd_f64,
};
use crate::options::Options;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};

/// L1 cache block size in complex elements (8KB for f32, 16KB for f64)
const L1_BLOCK_SIZE: usize = 1024;

/// Recursive cache-blocked DIT FFT for f64 using post-order traversal.
///
/// Recursively divides by 2 until reaching L1 cache size, processes stages within
/// each block, then processes cross-block stages on return.
fn recursive_dit_fft_f64(
    reals: &mut [f64],
    imags: &mut [f64],
    size: usize,
    planner: &PlannerDit64,
    mut stage_twiddle_idx: usize,
) -> usize {
    let log_size = size.ilog2() as usize;

    if size <= L1_BLOCK_SIZE {
        for stage in 0..log_size {
            stage_twiddle_idx = execute_dit_stage_f64(
                &mut reals[..size],
                &mut imags[..size],
                stage,
                planner,
                stage_twiddle_idx,
            );
        }
        stage_twiddle_idx
    } else {
        let half = size / 2;
        let log_half = half.ilog2() as usize;

        let (re_first_half, re_second_half) = reals.split_at_mut(half);
        let (im_first_half, im_second_half) = imags.split_at_mut(half);
        // Recursively process both halves
        rayon::join(
            || recursive_dit_fft_f64(re_first_half, im_first_half, half, planner, 0),
            || recursive_dit_fft_f64(re_second_half, im_second_half, half, planner, 0),
        );

        // Both halves completed stages 0..log_half-1
        // Stages 0-5 use hardcoded twiddles, 6+ use planner
        stage_twiddle_idx = log_half.saturating_sub(6);

        // Process remaining stages that span both halves
        for stage in log_half..log_size {
            stage_twiddle_idx = execute_dit_stage_f64(
                &mut reals[..size],
                &mut imags[..size],
                stage,
                planner,
                stage_twiddle_idx,
            );
        }

        stage_twiddle_idx
    }
}

/// Recursive cache-blocked DIT FFT for f32 using post-order traversal.
fn recursive_dit_fft_f32(
    reals: &mut [f32],
    imags: &mut [f32],
    offset: usize,
    size: usize,
    planner: &PlannerDit32,
    mut stage_twiddle_idx: usize,
) -> usize {
    let log_size = size.ilog2() as usize;

    if size <= L1_BLOCK_SIZE {
        for stage in 0..log_size {
            stage_twiddle_idx = execute_dit_stage_f32(
                &mut reals[offset..offset + size],
                &mut imags[offset..offset + size],
                stage,
                planner,
                stage_twiddle_idx,
            );
        }
        stage_twiddle_idx
    } else {
        let half = size / 2;
        let log_half = half.ilog2() as usize;

        recursive_dit_fft_f32(reals, imags, offset, half, planner, 0);
        recursive_dit_fft_f32(reals, imags, offset + half, half, planner, 0);

        stage_twiddle_idx = log_half.saturating_sub(6);

        for stage in log_half..log_size {
            stage_twiddle_idx = execute_dit_stage_f32(
                &mut reals[offset..offset + size],
                &mut imags[offset..offset + size],
                stage,
                planner,
                stage_twiddle_idx,
            );
        }

        stage_twiddle_idx
    }
}

/// Execute a single DIT stage, dispatching to appropriate kernel based on chunk size.
/// Returns updated stage_twiddle_idx.
#[inline]
fn execute_dit_stage_f64(
    reals: &mut [f64],
    imags: &mut [f64],
    stage: usize,
    planner: &PlannerDit64,
    stage_twiddle_idx: usize,
) -> usize {
    let dist = 1 << stage;
    let chunk_size = dist << 1;

    if chunk_size == 2 {
        fft_dit_chunk_2(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 4 {
        fft_dit_chunk_4_simd_f64(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 8 {
        fft_dit_chunk_8_simd_f64(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 16 {
        fft_dit_chunk_16_simd_f64(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 32 {
        fft_dit_chunk_32_simd_f64(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 64 {
        fft_dit_chunk_64_simd_f64(reals, imags);
        stage_twiddle_idx
    } else {
        // For larger chunks, use general kernel with twiddles from planner
        let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
        fft_dit_64_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
        stage_twiddle_idx + 1
    }
}

/// Execute a single DIT stage, dispatching to appropriate kernel based on chunk size.
/// Returns updated stage_twiddle_idx.
#[inline]
fn execute_dit_stage_f32(
    reals: &mut [f32],
    imags: &mut [f32],
    stage: usize,
    planner: &PlannerDit32,
    stage_twiddle_idx: usize,
) -> usize {
    let dist = 1 << stage;
    let chunk_size = dist << 1;

    if chunk_size == 2 {
        fft_dit_chunk_2(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 4 {
        fft_dit_chunk_4_simd_f32(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 8 {
        fft_dit_chunk_8_simd_f32(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 16 {
        fft_dit_chunk_16_simd_f32(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 32 {
        fft_dit_chunk_32_simd_f32(reals, imags);
        stage_twiddle_idx
    } else if chunk_size == 64 {
        fft_dit_chunk_64_simd_f32(reals, imags);
        stage_twiddle_idx
    } else {
        // For larger chunks, use general kernel with twiddles from planner
        let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
        fft_dit_32_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
        stage_twiddle_idx + 1
    }
}

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
        rayon::join(|| cobra_apply(reals, log_n), || cobra_apply(imags, log_n));
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

    recursive_dit_fft_f64(reals, imags, n, planner, 0);

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
        rayon::join(|| cobra_apply(reals, log_n), || cobra_apply(imags, log_n));
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

    recursive_dit_fft_f32(reals, imags, 0, n, planner, 0);

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = 1.0 / n as f32;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    }
}
