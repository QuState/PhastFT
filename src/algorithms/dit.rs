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
use std::ops::Range;

use fearless_simd::{dispatch, f32x16, f32x8, f64x4, f64x8, Simd};

use crate::algorithms::bravo::{bit_rev_bravo_f32, bit_rev_bravo_f64};
use crate::kernels::codelets::{fft_dit_codelet_16_f64, fft_dit_codelet_32_f32};
use crate::kernels::dit::*;
use crate::options::Options;
use crate::parallel::run_maybe_in_parallel;
use crate::planner::{Direction, PlanTwiddles, PlannerDit32, PlannerDit64};
use crate::twiddles::TwiddleSource;

/// Maximum leaf size in complex elements (8KB for f32, 16KB for f64).
/// Four-way recursion intentionally gives 1024-point leaves for even exponents
/// and 512-point leaves for odd exponents. Forcing equal leaves introduces an
/// unfused merge stage without a consistent benefit in the parity benchmarks.
const L1_BLOCK_SIZE: usize = 1024;

/// Two fused passes, each consuming two radix-2 stages.
#[cfg(feature = "parallel")]
const PARALLEL_STAGES: usize = 4;

/// Require enough work for 16 workers before distributing a fused pass.
/// This is fixed until the scheduling thresholds account for the active pool.
#[cfg(feature = "parallel")]
const PARALLEL_KERNEL_CHUNKS: usize = 16;

enum GeneralStage {
    Single,
    Fused,
    #[cfg(feature = "parallel")]
    FusedParallel,
}

/// Shared scheduling policy for both precisions. Keep an unmatched stage at
/// the beginning so the larger stages remain fused. Only distribute the last
/// two fused passes across workers, and require 16 times the caller's minimum
/// chunk size so each worker has enough work.
#[inline]
#[allow(unused_variables)] // Parallel policy inputs are unused without the feature.
fn general_stage(
    stages: Range<usize>,
    total_stages: usize,
    size: usize,
    opts: &Options,
) -> GeneralStage {
    if stages.len() % 2 == 1 {
        return GeneralStage::Single;
    }
    #[cfg(feature = "parallel")]
    if total_stages - stages.start <= PARALLEL_STAGES
        && size
            > opts
                .smallest_parallel_chunk_size
                .saturating_mul(PARALLEL_KERNEL_CHUNKS)
    {
        return GeneralStage::FusedParallel;
    }
    GeneralStage::Fused
}

/// Recursive cache-blocked DIT FFT for f64 using post-order traversal.
///
/// Recursively divides by 4 until reaching L1 cache size, processes stages within
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
        // Use FFT-16 codelet to fuse stages 0-3 into a single pass per 16-element chunk
        let codelet_stages = 4;
        let start_stage = if stage_twiddle_idx == 0 && size >= power_of_two(codelet_stages) {
            fft_dit_codelet_16_f64(simd, &mut reals[..size], &mut imags[..size]);
            codelet_stages
        } else {
            0
        };

        let mut stage = start_stage;
        while stage < log_size {
            let (new_idx, consumed) = execute_dit_stages_f64(
                simd,
                &mut reals[..size],
                &mut imags[..size],
                stage..log_size,
                planner,
                opts,
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
                stage..log_size,
                planner,
                opts,
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
        // Use FFT-32 codelet to fuse stages 0-4 into a single pass per 32-element chunk
        let codelet_stages = 5;
        let start_stage = if stage_twiddle_idx == 0 && size >= power_of_two(codelet_stages) {
            fft_dit_codelet_32_f32(simd, &mut reals[..size], &mut imags[..size]);
            codelet_stages
        } else {
            0
        };

        let mut stage = start_stage;
        while stage < log_size {
            let (new_idx, consumed) = execute_dit_stages_f32(
                simd,
                &mut reals[..size],
                &mut imags[..size],
                stage..log_size,
                planner,
                opts,
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
                stage..log_size,
                planner,
                opts,
                stage_twiddle_idx,
            );
            stage_twiddle_idx = new_idx;
            stage += consumed;
        }

        stage_twiddle_idx
    }
}

/// Execute one or two stages within the current recursive subproblem.
/// Returns (updated stage_twiddle_idx, number of stages consumed).
fn execute_dit_stages_f64<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    stages: Range<usize>,
    planner: &PlannerDit64,
    opts: &Options,
    stage_twiddle_idx: usize,
) -> (usize, usize) {
    let dist = 1 << stages.start;
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
        let stage_kind = general_stage(stages, planner.log_n, reals.len(), opts);
        let consumed = match &planner.stage_twiddles {
            PlanTwiddles::Precomputed(twiddles) => execute_general_stages_f64(
                simd,
                reals,
                imags,
                &twiddles[stage_twiddle_idx..],
                stage_kind,
                dist,
            ),
            PlanTwiddles::LowMemory(twiddles) => execute_general_stages_f64(
                simd,
                reals,
                imags,
                &twiddles[stage_twiddle_idx..],
                stage_kind,
                dist,
            ),
        };
        (stage_twiddle_idx + consumed, consumed)
    }
}

/// Execute one or two stages within the current recursive subproblem.
/// Returns (updated stage_twiddle_idx, number of stages consumed).
fn execute_dit_stages_f32<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    stages: Range<usize>,
    planner: &PlannerDit32,
    opts: &Options,
    stage_twiddle_idx: usize,
) -> (usize, usize) {
    let dist = 1 << stages.start;
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
    } else {
        let stage_kind = general_stage(stages, planner.log_n, reals.len(), opts);
        let consumed = match &planner.stage_twiddles {
            PlanTwiddles::Precomputed(twiddles) => execute_general_stages_f32(
                simd,
                reals,
                imags,
                &twiddles[stage_twiddle_idx..],
                stage_kind,
                dist,
            ),
            PlanTwiddles::LowMemory(twiddles) => execute_general_stages_f32(
                simd,
                reals,
                imags,
                &twiddles[stage_twiddle_idx..],
                stage_kind,
                dist,
            ),
        };
        (stage_twiddle_idx + consumed, consumed)
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
/// * `direction` - Forward or inverse transform
/// * `planner` - Pre-computed planner with twiddle factors
/// * `opts` - Options controlling optimization strategies
///
/// # Panics
///
/// Panics if input length is not a power of 2 or if real and imaginary arrays have different lengths
///
pub fn fft_f64_dit_with_planner_and_opts(
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerDit64,
    opts: &Options,
) {
    // Dynamic dispatch overhead becomes really noticeable at small FFT sizes.
    // Dispatch only once at the top of the program to amortize the cost.
    dispatch!(planner.simd_level, simd => fft_f64_dit_with_planner_and_opts_impl(simd, reals, imags, direction, planner, opts))
}

#[inline(always)] // required by fearless_simd
fn fft_f64_dit_with_planner_and_opts_impl<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerDit64,
    opts: &Options,
) {
    assert_eq!(reals.len(), imags.len());
    assert!(reals.len().is_power_of_two());

    let n = reals.len();
    let log_n = n.ilog2() as usize;
    assert_eq!(log_n, planner.log_n);

    // IFFT via swap trick: swap(IDFT(z)) = (1/N) * DFT(swap(z)), where
    // swap(z) = swap(a + bi) = b + ai.
    // Hence, we hand the caller's imaginaries slice to the forward
    // FFT as its "reals" arg (and vice versa).
    let (reals, imags) = match direction {
        Direction::Forward => (reals, imags),
        Direction::Inverse => (imags, reals),
    };

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

    simd.vectorize(
        #[inline(always)]
        || recursive_dit_fft_f64(simd, reals, imags, n, planner, opts, 0),
    );

    // Scaling for inverse transform
    if let Direction::Inverse = direction {
        let scaling_factor = 1.0 / n as f64;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= scaling_factor;
        }
    }
}

/// DIT FFT for f32 with pre-computed planner and options
///
/// Single-precision version of the DIT FFT algorithm.
/// See [`fft_f64_dit_with_planner_and_opts`] for `f64` version.
///
/// # Panics
///
/// Panics if input length is not a power of 2 or if real and imaginary arrays have different lengths
pub fn fft_f32_dit_with_planner_and_opts(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerDit32,
    opts: &Options,
) {
    // Dynamic dispatch overhead becomes really noticeable at small FFT sizes.
    // Dispatch only once at the top of the program to
    dispatch!(planner.simd_level, simd => fft_f32_dit_with_planner_and_opts_impl(simd, reals, imags, direction, planner, opts))
}

#[inline(always)] // required by fearless_simd
fn fft_f32_dit_with_planner_and_opts_impl<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerDit32,
    opts: &Options,
) {
    assert_eq!(reals.len(), imags.len());
    assert!(reals.len().is_power_of_two());

    let n = reals.len();
    let log_n = n.ilog2() as usize;
    assert_eq!(log_n, planner.log_n);

    // See `fft_f64_dit_with_planner_and_opts_impl` for the swap-trick rationale.
    let (reals, imags) = match direction {
        Direction::Forward => (reals, imags),
        Direction::Inverse => (imags, reals),
    };

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

    simd.vectorize(
        #[inline(always)]
        || recursive_dit_fft_f32(simd, reals, imags, n, planner, opts, 0),
    );

    // Scaling for inverse transform
    if let Direction::Inverse = direction {
        let scaling_factor = 1.0 / n as f32;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= scaling_factor;
        }
    }
}

#[inline]
const fn power_of_two(power: usize) -> usize {
    // 2.pow() requires a lot of ugly type annotations so here's a helper function
    debug_assert!(power < usize::BITS as usize);
    1 << power
}

// The storage enum is resolved above; all twiddle access below is static.
fn execute_general_stages_f64<S: Simd, T>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    stage_twiddles: &[T],
    stage_kind: GeneralStage,
    dist: usize,
) -> usize
where
    T: TwiddleSource<S, f64x8<S>> + TwiddleSource<S, f64x4<S>>,
{
    let twiddles = &stage_twiddles[0];
    match stage_kind {
        GeneralStage::Single => {
            fft_dit_chunk_n_f64(simd, reals, imags, twiddles, dist);
            1
        }
        GeneralStage::Fused => {
            let next_twiddles = &stage_twiddles[1];
            fft_dit_fused_2stage_f64_narrow(
                simd,
                reals,
                imags,
                FusedTwiddles {
                    stage: twiddles,
                    next_stage: next_twiddles,
                },
                dist,
            );
            2
        }
        #[cfg(feature = "parallel")]
        GeneralStage::FusedParallel => {
            let next_twiddles = &stage_twiddles[1];
            fft_dit_fused_2stage_f64_narrow_parallel(
                simd,
                reals,
                imags,
                FusedTwiddles {
                    stage: twiddles,
                    next_stage: next_twiddles,
                },
                dist,
            );
            2
        }
    }
}

// The storage enum is resolved above; all twiddle access below is static.
fn execute_general_stages_f32<S: Simd, T>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    stage_twiddles: &[T],
    stage_kind: GeneralStage,
    dist: usize,
) -> usize
where
    T: TwiddleSource<S, f32x16<S>> + TwiddleSource<S, f32x8<S>>,
{
    let twiddles = &stage_twiddles[0];
    match stage_kind {
        GeneralStage::Single => {
            fft_dit_chunk_n_f32(simd, reals, imags, twiddles, dist);
            1
        }
        GeneralStage::Fused => {
            let next_twiddles = &stage_twiddles[1];
            fft_dit_fused_2stage_f32_narrow(
                simd,
                reals,
                imags,
                FusedTwiddles {
                    stage: twiddles,
                    next_stage: next_twiddles,
                },
                dist,
            );
            2
        }
        #[cfg(feature = "parallel")]
        GeneralStage::FusedParallel => {
            let next_twiddles = &stage_twiddles[1];
            fft_dit_fused_2stage_f32_narrow_parallel(
                simd,
                reals,
                imags,
                FusedTwiddles {
                    stage: twiddles,
                    next_stage: next_twiddles,
                },
                dist,
            );
            2
        }
    }
}
