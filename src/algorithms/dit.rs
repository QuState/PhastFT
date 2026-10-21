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

use fearless_simd::{dispatch, Simd};

use crate::algorithms::bravo::{bit_rev_bravo_f32, bit_rev_bravo_f64};
use crate::kernels::codelets::{fft_dit_codelet_16_f64, fft_dit_codelet_32_f32};
use crate::kernels::dit::*;
use crate::options::Options;
use crate::parallel::run_maybe_in_parallel;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};

/// Maximum leaf size in complex elements (8KB for f32, 16KB for f64).
/// Grouped four-way recursion gives 1024-point leaves for even exponents and
/// 512-point leaves for odd exponents. Forcing equal leaves introduces an
/// unfused merge stage without a consistent benefit in the parity benchmarks.
const L1_BLOCK_SIZE: usize = 1024;

/// Distribute at most the last four radix-2 stages across workers.
#[cfg(feature = "parallel")]
const PARALLEL_STAGES: usize = 4;

/// Require enough work for 16 workers before distributing a fused pass.
/// This is fixed until the scheduling thresholds account for the active pool.
#[cfg(feature = "parallel")]
const PARALLEL_KERNEL_CHUNKS: usize = 16;

#[derive(Debug, PartialEq, Eq)]
enum GeneralStage {
    Single,
    Fused2,
    Fused4,
    #[cfg(feature = "parallel")]
    Fused2Parallel,
    #[cfg(feature = "parallel")]
    Fused4Parallel,
}

/// Shared scheduling policy for both precisions. Keep an unmatched stage at
/// the beginning so the larger stages remain fused. Only distribute the last
/// four stages across workers, and require 16 times the caller's minimum
/// chunk size so each worker has enough work.
#[inline]
#[allow(unused_variables)] // Parallel policy inputs are unused without the feature.
fn general_stage(
    stages: Range<usize>,
    total_stages: usize,
    size: usize,
    opts: &Options,
) -> GeneralStage {
    debug_assert!(!stages.is_empty());
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
        return if stages.len().is_multiple_of(4) {
            GeneralStage::Fused4Parallel
        } else {
            GeneralStage::Fused2Parallel
        };
    }
    if stages.len().is_multiple_of(4) {
        GeneralStage::Fused4
    } else {
        GeneralStage::Fused2
    }
}

/// Group two four-way recursion levels when both are above the leaf cutoff.
/// Keeping a two-stage merge immediately above leaves preserves leaf sizes.
fn merge_stages(size: usize) -> usize {
    if size / 4 > L1_BLOCK_SIZE {
        4
    } else {
        2
    }
}

/// Visit equal-sized recursive children using balanced binary splits. This
/// preserves the existing parallel threshold at every split, including the
/// intermediate levels of a sixteen-child merge.
fn for_each_dit_child<T: Send, F: Fn(&mut [T], &mut [T]) + Sync>(
    reals: &mut [T],
    imags: &mut [T],
    child_size: usize,
    opts: &Options,
    process: &F,
) {
    let size = reals.len();
    if size == child_size {
        process(reals, imags);
    } else {
        let (re_lo, re_hi) = reals.split_at_mut(size / 2);
        let (im_lo, im_hi) = imags.split_at_mut(size / 2);
        run_maybe_in_parallel(
            size > opts.smallest_parallel_chunk_size,
            || for_each_dit_child(re_lo, im_lo, child_size, opts, process),
            || for_each_dit_child(re_hi, im_hi, child_size, opts, process),
        );
    }
}

/// Recursive cache-blocked DIT FFT for f64 using post-order traversal.
///
/// Recursively divides by 16 (or 4 near leaves), processes stages within
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
        let child_size = size >> merge_stages(size);
        let log_child = child_size.ilog2() as usize;
        for_each_dit_child(reals, imags, child_size, opts, &|re, im| {
            recursive_dit_fft_f64(simd, re, im, child_size, planner, opts, 0);
        });

        // Children completed stages below log_child. Merge the remaining two
        // or four stages; planner tables begin at stage 6.
        stage_twiddle_idx = log_child.saturating_sub(6);

        let mut stage = log_child;
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
        let child_size = size >> merge_stages(size);
        let log_child = child_size.ilog2() as usize;
        for_each_dit_child(reals, imags, child_size, opts, &|re, im| {
            recursive_dit_fft_f32(simd, re, im, child_size, planner, opts, 0);
        });

        // Children completed stages below log_child. Merge the remaining two
        // or four stages; planner tables begin at stage 6.
        stage_twiddle_idx = log_child.saturating_sub(6);

        let mut stage = log_child;
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

/// Execute one, two, or four stages within the current recursive subproblem.
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
        let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
        match general_stage(stages, planner.log_n, reals.len(), opts) {
            GeneralStage::Single => {
                fft_dit_chunk_n_f64(simd, reals, imags, twiddles_re, twiddles_im, dist);
                (stage_twiddle_idx + 1, 1)
            }
            GeneralStage::Fused4 => {
                let (base_re, base_im) = &planner.stage_twiddles[stage_twiddle_idx + 3];
                fft_dit_fused_4stage_f64_narrow(
                    simd,
                    reals,
                    imags,
                    Fused4Twiddles {
                        re: base_re,
                        im: base_im,
                    },
                    dist,
                );
                (stage_twiddle_idx + 4, 4)
            }
            #[cfg(feature = "parallel")]
            GeneralStage::Fused4Parallel => {
                let (base_re, base_im) = &planner.stage_twiddles[stage_twiddle_idx + 3];
                fft_dit_fused_4stage_f64_narrow_parallel(
                    simd,
                    reals,
                    imags,
                    Fused4Twiddles {
                        re: base_re,
                        im: base_im,
                    },
                    dist,
                );
                (stage_twiddle_idx + 4, 4)
            }
            GeneralStage::Fused2 => {
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f64_narrow(
                    simd,
                    reals,
                    imags,
                    FusedTwiddles {
                        stage_re: twiddles_re,
                        stage_im: twiddles_im,
                        next_stage_re: twiddles_re2,
                        next_stage_im: twiddles_im2,
                    },
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            }
            #[cfg(feature = "parallel")]
            GeneralStage::Fused2Parallel => {
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f64_narrow_parallel(
                    simd,
                    reals,
                    imags,
                    FusedTwiddles {
                        stage_re: twiddles_re,
                        stage_im: twiddles_im,
                        next_stage_re: twiddles_re2,
                        next_stage_im: twiddles_im2,
                    },
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            }
        }
    }
}

/// Execute one, two, or four stages within the current recursive subproblem.
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
        let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
        match general_stage(stages, planner.log_n, reals.len(), opts) {
            GeneralStage::Single => {
                fft_dit_chunk_n_f32(simd, reals, imags, twiddles_re, twiddles_im, dist);
                (stage_twiddle_idx + 1, 1)
            }
            GeneralStage::Fused4 => {
                let (base_re, base_im) = &planner.stage_twiddles[stage_twiddle_idx + 3];
                fft_dit_fused_4stage_f32_narrow(
                    simd,
                    reals,
                    imags,
                    Fused4Twiddles {
                        re: base_re,
                        im: base_im,
                    },
                    dist,
                );
                (stage_twiddle_idx + 4, 4)
            }
            #[cfg(feature = "parallel")]
            GeneralStage::Fused4Parallel => {
                let (base_re, base_im) = &planner.stage_twiddles[stage_twiddle_idx + 3];
                fft_dit_fused_4stage_f32_narrow_parallel(
                    simd,
                    reals,
                    imags,
                    Fused4Twiddles {
                        re: base_re,
                        im: base_im,
                    },
                    dist,
                );
                (stage_twiddle_idx + 4, 4)
            }
            GeneralStage::Fused2 => {
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f32_narrow(
                    simd,
                    reals,
                    imags,
                    FusedTwiddles {
                        stage_re: twiddles_re,
                        stage_im: twiddles_im,
                        next_stage_re: twiddles_re2,
                        next_stage_im: twiddles_im2,
                    },
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            }
            #[cfg(feature = "parallel")]
            GeneralStage::Fused2Parallel => {
                let (twiddles_re2, twiddles_im2) = &planner.stage_twiddles[stage_twiddle_idx + 1];
                fft_dit_fused_2stage_f32_narrow_parallel(
                    simd,
                    reals,
                    imags,
                    FusedTwiddles {
                        stage_re: twiddles_re,
                        stage_im: twiddles_im,
                        next_stage_re: twiddles_re2,
                        next_stage_im: twiddles_im2,
                    },
                    dist,
                );
                (stage_twiddle_idx + 2, 2)
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_stage_remainders() {
        let opts = Options {
            smallest_parallel_chunk_size: usize::MAX,
            ..Options::default()
        };
        let expected: &[&[usize]] = &[
            &[1],
            &[2],
            &[1, 2],
            &[4],
            &[1, 4],
            &[2, 4],
            &[1, 2, 4],
            &[4, 4],
            &[1, 4, 4],
            &[2, 4, 4],
        ];
        for (index, expected) in expected.iter().enumerate() {
            let end = 6 + index + 1;
            let mut stage = 6;
            let mut actual = Vec::new();
            while stage < end {
                let consumed = match general_stage(stage..end, end, 1 << end, &opts) {
                    GeneralStage::Single => 1,
                    GeneralStage::Fused2 => 2,
                    GeneralStage::Fused4 => 4,
                    #[cfg(feature = "parallel")]
                    _ => panic!("parallelism was disabled"),
                };
                actual.push(consumed);
                stage += consumed;
            }
            assert_eq!(&actual, expected);
        }
    }

    #[test]
    fn recursion_preserves_leaf_sizes() {
        assert_eq!(merge_stages(2048), 2);
        assert_eq!(merge_stages(4096), 2);
        assert_eq!(merge_stages(8192), 4);
        assert_eq!(merge_stages(16384), 4);
        for exponent in 11..=24 {
            let mut size = 1 << exponent;
            while size > L1_BLOCK_SIZE {
                size >>= merge_stages(size);
            }
            assert_eq!(size, if exponent % 2 == 0 { 1024 } else { 512 });
        }
    }

    #[test]
    fn fused_parallel_thresholds() {
        let mut opts = Options::default();
        let cutoff = opts.smallest_parallel_chunk_size * 16;
        for (stages, serial) in [
            (15..19, GeneralStage::Fused4),
            (17..19, GeneralStage::Fused2),
        ] {
            assert_eq!(general_stage(stages.clone(), 19, cutoff, &opts), serial);
            let selected = general_stage(stages.clone(), 19, cutoff + 1, &opts);
            #[cfg(feature = "parallel")]
            assert_eq!(
                selected,
                if stages.len() == 4 {
                    GeneralStage::Fused4Parallel
                } else {
                    GeneralStage::Fused2Parallel
                }
            );
            #[cfg(not(feature = "parallel"))]
            assert_eq!(selected, serial);
            opts.smallest_parallel_chunk_size = 0;
            assert_eq!(general_stage(stages.clone(), 24, cutoff + 1, &opts), serial);
            opts.smallest_parallel_chunk_size = usize::MAX;
            assert_eq!(general_stage(stages, 19, usize::MAX, &opts), serial);
            opts = Options::default();
        }
        opts.smallest_parallel_chunk_size = 0;
        assert_eq!(
            general_stage(18..19, 19, 1 << 19, &opts),
            GeneralStage::Single
        );
        #[cfg(feature = "parallel")]
        assert_eq!(
            general_stage(6..10, 10, 1024, &opts),
            GeneralStage::Fused4Parallel
        );
    }

    macro_rules! test_twiddle_indices {
        ($name:ident, $float:ty, $planner:ty, $execute:ident) => {
            #[test]
            fn $name() {
                let planner = <$planner>::new(1 << 13);
                let mut reals = vec![0.0 as $float; 1 << 13];
                let mut imags = reals.clone();
                for threshold in [0, usize::MAX] {
                    let opts = Options {
                        smallest_parallel_chunk_size: threshold,
                        ..Options::default()
                    };
                    // Seven general stages must execute as 1 + 2 + 4, including
                    // a four-stage kernel starting at a nonzero planner index.
                    for (stage, index, expected) in [
                        (6, 0, (1, 1)), (7, 1, (3, 2)), (9, 3, (7, 4)),
                    ] {
                        let result = dispatch!(planner.simd_level, simd => $execute(
                            simd, &mut reals, &mut imags, stage..13,
                            &planner, &opts, index,
                        ));
                        assert_eq!(result, expected);
                    }
                }
            }
        };
    }

    test_twiddle_indices!(
        twiddle_indices_f32,
        f32,
        PlannerDit32,
        execute_dit_stages_f32
    );
    test_twiddle_indices!(
        twiddle_indices_f64,
        f64,
        PlannerDit64,
        execute_dit_stages_f64
    );
}
