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
use fearless_simd::{dispatch, Simd};

use crate::algorithms::bravo::{bit_rev_bravo_f32, bit_rev_bravo_f64};
use crate::kernels::dit::*;
use crate::options::Options;
use crate::parallel::run_maybe_in_parallel;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};

/// L1 cache block size in complex elements (8KB for f32, 16KB for f64)
const L1_BLOCK_SIZE: usize = 1024;

/// Recursive cache-blocked DIT FFT for f64 using post-order traversal.
///
/// Recursively divides by 2 until reaching L1 cache size, processes stages within
/// each block, then processes cross-block stages on return.
fn recursive_dit_fft_f64<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    size: usize,
    planner: &PlannerDit64,
    opts: &Options,
    mut stage_twiddle_idx: usize,
) -> usize {
    let log_size = size.ilog2() as usize;

    if size <= L1_BLOCK_SIZE {
        let mut stage = 0;
        while stage < log_size {
            let (new_idx, consumed) = execute_dit_stages_f64(
                simd,
                &mut reals[..size],
                &mut imags[..size],
                stage,
                log_size - stage,
                planner,
                stage_twiddle_idx,
            );
            stage_twiddle_idx = new_idx;
            stage += consumed;
        }
        stage_twiddle_idx
    } else {
        let quarter = size / 4;
        let log_quarter = quarter.ilog2() as usize;

        let (re_lo, re_hi) = reals.split_at_mut(size / 2);
        let (im_lo, im_hi) = imags.split_at_mut(size / 2);
        let (re_q0, re_q1) = re_lo.split_at_mut(quarter);
        let (im_q0, im_q1) = im_lo.split_at_mut(quarter);
        let (re_q2, re_q3) = re_hi.split_at_mut(quarter);
        let (im_q2, im_q3) = im_hi.split_at_mut(quarter);

        // Recursively process all 4 quarters (parallel across pairs)
        run_maybe_in_parallel(
            size > opts.smallest_parallel_chunk_size,
            || {
                run_maybe_in_parallel(
                    size / 2 > opts.smallest_parallel_chunk_size,
                    || recursive_dit_fft_f64(simd, re_q0, im_q0, quarter, planner, opts, 0),
                    || recursive_dit_fft_f64(simd, re_q1, im_q1, quarter, planner, opts, 0),
                )
            },
            || {
                run_maybe_in_parallel(
                    size / 2 > opts.smallest_parallel_chunk_size,
                    || recursive_dit_fft_f64(simd, re_q2, im_q2, quarter, planner, opts, 0),
                    || recursive_dit_fft_f64(simd, re_q3, im_q3, quarter, planner, opts, 0),
                )
            },
        );

        // All 4 quarters completed stages 0..log_quarter-1.
        // Now process the 2 remaining cross-block stages (log_quarter and log_quarter+1)
        // using the fused kernel.
        // Stages 0-5 use hardcoded twiddles, 6+ use planner
        stage_twiddle_idx = log_quarter.saturating_sub(6);

        let mut stage = log_quarter;
        while stage < log_size {
            let (new_idx, consumed) = execute_dit_stages_f64(
                simd,
                &mut reals[..size],
                &mut imags[..size],
                stage,
                log_size - stage,
                planner,
                stage_twiddle_idx,
            );
            stage_twiddle_idx = new_idx;
            stage += consumed;
        }

        stage_twiddle_idx
    }
}

/// Recursive cache-blocked DIT FFT for f32 using post-order traversal.
fn recursive_dit_fft_f32<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    size: usize,
    planner: &PlannerDit32,
    opts: &Options,
    mut stage_twiddle_idx: usize,
) -> usize {
    let log_size = size.ilog2() as usize;

    if size <= L1_BLOCK_SIZE {
        let mut stage = 0;
        while stage < log_size {
            let (new_idx, consumed) = execute_dit_stages_f32(
                simd,
                &mut reals[..size],
                &mut imags[..size],
                stage,
                log_size - stage,
                planner,
                stage_twiddle_idx,
            );
            stage_twiddle_idx = new_idx;
            stage += consumed;
        }
        stage_twiddle_idx
    } else {
        let quarter = size / 4;
        let log_quarter = quarter.ilog2() as usize;

        let (re_lo, re_hi) = reals.split_at_mut(size / 2);
        let (im_lo, im_hi) = imags.split_at_mut(size / 2);
        let (re_q0, re_q1) = re_lo.split_at_mut(quarter);
        let (im_q0, im_q1) = im_lo.split_at_mut(quarter);
        let (re_q2, re_q3) = re_hi.split_at_mut(quarter);
        let (im_q2, im_q3) = im_hi.split_at_mut(quarter);

        // Recursively process all 4 quarters (parallel across pairs)
        run_maybe_in_parallel(
            size > opts.smallest_parallel_chunk_size,
            || {
                run_maybe_in_parallel(
                    size / 2 > opts.smallest_parallel_chunk_size,
                    || recursive_dit_fft_f32(simd, re_q0, im_q0, quarter, planner, opts, 0),
                    || recursive_dit_fft_f32(simd, re_q1, im_q1, quarter, planner, opts, 0),
                )
            },
            || {
                run_maybe_in_parallel(
                    size / 2 > opts.smallest_parallel_chunk_size,
                    || recursive_dit_fft_f32(simd, re_q2, im_q2, quarter, planner, opts, 0),
                    || recursive_dit_fft_f32(simd, re_q3, im_q3, quarter, planner, opts, 0),
                )
            },
        );

        // All 4 quarters completed stages 0..log_quarter-1.
        // Now process the 2 remaining cross-block stages (log_quarter and log_quarter+1)
        // using the fused kernel.
        // Stages 0-5 use hardcoded twiddles, 6+ use planner
        stage_twiddle_idx = log_quarter.saturating_sub(6);

        let mut stage = log_quarter;
        while stage < log_size {
            let (new_idx, consumed) = execute_dit_stages_f32(
                simd,
                &mut reals[..size],
                &mut imags[..size],
                stage,
                log_size - stage,
                planner,
                stage_twiddle_idx,
            );
            stage_twiddle_idx = new_idx;
            stage += consumed;
        }

        stage_twiddle_idx
    }
}

/// Execute one or two DIT stages, dispatching to appropriate kernel based on chunk size.
/// Returns (updated stage_twiddle_idx, number of stages consumed).
fn execute_dit_stages_f64<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    stage: usize,
    stages_remaining: usize,
    planner: &PlannerDit64,
    stage_twiddle_idx: usize,
) -> (usize, usize) {
    let dist = 1 << stage; // 2.pow(stage)
    let chunk_size = dist * 2;

    if chunk_size == 2 {
        simd.vectorize(|| fft_dit_chunk_2(simd, reals, imags));
        (stage_twiddle_idx, 1)
    } else if chunk_size == 4 {
        fft_dit_chunk_4_f64(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 8 {
        fft_dit_chunk_8_f64(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 16 {
        fft_dit_chunk_16_f64(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 32 {
        fft_dit_chunk_32_f64(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 64 {
        fft_dit_chunk_64_f64(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else {
        #[cfg(feature = "parallel")]
        {
            // When nearing the end, use the parallelized kernel because recursion doesn't parallelize enough
            // TODO: make this dynamic based on std::thread::available_parallelism(), maybe?
            // The jump from 16 threads to all threads is probably not very large so this is entering diminishing returns
            if stages_remaining > 4 {
                // Fuse two stages into a single pass over memory
                let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f64_narrow(
                    simd,
                    reals,
                    imags,
                    twiddles_re,
                    twiddles_im,
                    twiddles_re2,
                    twiddles_im2,
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            } else if stages_remaining >= 2 {
                let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f64_narrow_parallel(
                    simd,
                    reals,
                    imags,
                    twiddles_re,
                    twiddles_im,
                    twiddles_re2,
                    twiddles_im2,
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            } else {
                // Last stage (odd number of stages remaining), use single-stage kernel
                let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
                fft_dit_chunk_n_f64(simd, reals, imags, twiddles_re, twiddles_im, dist);
                (stage_twiddle_idx + 1, 1)
            }
        }
        #[cfg(not(feature = "parallel"))]
        if stages_remaining >= 2 {
            // Fuse two stages into a single pass over memory
            let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
            let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
            fft_dit_fused_2stage_f64_narrow(
                simd,
                reals,
                imags,
                twiddles_re,
                twiddles_im,
                twiddles_re2,
                twiddles_im2,
                dist,
            );
            (stage_twiddle_idx + 2, 2)
        } else {
            // Last stage (odd number of stages remaining), use single-stage kernel
            let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
            fft_dit_chunk_n_f64(simd, reals, imags, twiddles_re, twiddles_im, dist);
            (stage_twiddle_idx + 1, 1)
        }
    }
}

/// Execute one or two DIT stages, dispatching to appropriate kernel based on chunk size.
/// Returns (updated stage_twiddle_idx, number of stages consumed).
fn execute_dit_stages_f32<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    stage: usize,
    stages_remaining: usize,
    planner: &PlannerDit32,
    stage_twiddle_idx: usize,
) -> (usize, usize) {
    let dist = 1 << stage; // 2.pow(stage)
    let chunk_size = dist * 2;

    if chunk_size == 2 {
        simd.vectorize(|| fft_dit_chunk_2(simd, reals, imags));
        (stage_twiddle_idx, 1)
    } else if chunk_size == 4 {
        fft_dit_chunk_4_f32(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 8 {
        fft_dit_chunk_8_f32(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 16 {
        fft_dit_chunk_16_f32(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 32 {
        fft_dit_chunk_32_f32(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if chunk_size == 64 {
        fft_dit_chunk_64_f32(simd, reals, imags);
        (stage_twiddle_idx, 1)
    } else if cfg!(feature = "parallel") {
        // When nearing the end, use the parallelized kernel because recursion doesn't parallelize enough
        #[cfg(feature = "parallel")]
        {
            // TODO: make this dynamic based on std::thread::available_parallelism(), maybe?
            // The jump from 16 threads to all threads is probably not very large so this is entering diminishing returns
            if stages_remaining > 4 {
                // Fuse two stages into a single pass over memory
                let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f32_narrow(
                    simd,
                    reals,
                    imags,
                    twiddles_re,
                    twiddles_im,
                    twiddles_re2,
                    twiddles_im2,
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            } else if stages_remaining >= 2 {
                let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f32_narrow_parallel(
                    simd,
                    reals,
                    imags,
                    twiddles_re,
                    twiddles_im,
                    twiddles_re2,
                    twiddles_im2,
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            } else {
                // Last stage (odd number of stages remaining), use single-stage kernel
                let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
                fft_dit_chunk_n_f32(simd, reals, imags, twiddles_re, twiddles_im, dist);
                (stage_twiddle_idx + 1, 1)
            }
            #[cfg(not(feature = "parallel"))]
            {
                unreachable!()
            }
        }
    } else if stages_remaining >= 2 {
        // Fuse two stages into a single pass over memory
        let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
        let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
        fft_dit_fused_2stage_f32_narrow(
            simd,
            reals,
            imags,
            twiddles_re,
            twiddles_im,
            twiddles_re2,
            twiddles_im2,
            dist,
        );
        (stage_twiddle_idx + 2, 2)
    } else {
        // Last stage (odd number of stages remaining), use single-stage kernel
        let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
        fft_dit_chunk_n_f32(simd, reals, imags, twiddles_re, twiddles_im, dist);
        (stage_twiddle_idx + 1, 1)
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
    // Dynamic dispatch overhead becomes really noticeable at small FFT sizes.
    // Dispatch only once at the top of the program to
    dispatch!(planner.simd_level, simd => fft_64_dit_with_planner_and_opts_impl(simd, reals, imags, planner, opts))
}

#[inline(always)] // required by fearless_simd
fn fft_64_dit_with_planner_and_opts_impl<S: Simd>(
    simd: S,
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
    run_maybe_in_parallel(
        opts.multithreaded_bit_reversal,
        || {
            simd.vectorize(
                #[inline(always)]
                || bit_rev_bravo_f64(simd, reals, log_n),
            )
        },
        || {
            simd.vectorize(
                #[inline(always)]
                || bit_rev_bravo_f64(simd, imags, log_n),
            )
        },
    );

    // Handle inverse FFT
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    }

    simd.vectorize(
        #[inline(always)]
        || recursive_dit_fft_f64(simd, reals, imags, n, planner, opts, 0),
    );

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
    // Dynamic dispatch overhead becomes really noticeable at small FFT sizes.
    // Dispatch only once at the top of the program to
    dispatch!(planner.simd_level, simd => fft_32_dit_with_planner_and_opts_impl(simd, reals, imags, planner, opts))
}

fn fft_32_dit_with_planner_and_opts_impl<S: Simd>(
    simd: S,
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
    run_maybe_in_parallel(
        opts.multithreaded_bit_reversal,
        || {
            simd.vectorize(
                #[inline(always)]
                || bit_rev_bravo_f32(simd, reals, log_n),
            )
        },
        || {
            simd.vectorize(
                #[inline(always)]
                || bit_rev_bravo_f32(simd, imags, log_n),
            )
        },
    );

    // Handle inverse FFT
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    }

    simd.vectorize(
        #[inline(always)]
        || recursive_dit_fft_f32(simd, reals, imags, n, planner, opts, 0),
    );

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = 1.0 / n as f32;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    }
}
