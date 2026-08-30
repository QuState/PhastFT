#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::correctness,
    clippy::suspicious
)]
#![forbid(unsafe_code)]

#[cfg(feature = "complex-nums")]
use num_complex::Complex;

#[cfg(feature = "complex-nums")]
use crate::complex_nums::{deinterleave_complex32, deinterleave_complex64, interleave_complex};
use crate::options::Options;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};
#[cfg(feature = "complex-nums")]
use crate::planner::{PlannerBluestein32, PlannerBluestein64};

#[cfg(not(feature = "bench-internals"))]
mod algorithms;
#[cfg(feature = "bench-internals")]
pub mod algorithms;
#[cfg(all(feature = "complex-nums", not(feature = "bench-internals")))]
mod complex_nums;
#[cfg(feature = "bench-internals")]
pub mod complex_nums;
mod kernels;
pub mod options;
mod parallel;
pub mod planner;

pub use algorithms::bluestein::{
    fft_f32_bluestein, fft_f32_bluestein_with_planner, fft_f32_bluestein_with_planner_and_opts,
    fft_f64_bluestein, fft_f64_bluestein_with_planner, fft_f64_bluestein_with_planner_and_opts,
};
pub use algorithms::dit::{fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts};
pub use algorithms::r2c::{
    c2r_fft_f32, c2r_fft_f32_with_planner, c2r_fft_f32_with_planner_and_opts, c2r_fft_f64,
    c2r_fft_f64_with_planner, c2r_fft_f64_with_planner_and_opts, r2c_fft_f32,
    r2c_fft_f32_with_planner, r2c_fft_f32_with_planner_and_opts, r2c_fft_f64,
    r2c_fft_f64_with_planner, r2c_fft_f64_with_planner_and_opts,
};

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_for {
    ($func_name:ident, $precision:ty, $fft_func:ident, $deinterleaving_func: ident, $planner:ty) => {
        /// FFT Interleaved — alternative entry point when the input data is a
        /// slice of [`Complex`]. Analogous to the
        /// `fft_*_dit_with_planner_and_opts` family except for the input format.
        ///
        /// **Note**: This function has to make a deinterleaved copy of the data.
        /// For maximum performance with minimal memory usage, use the split-array
        /// `fft_*_dit_with_planner_and_opts` API directly.
        pub fn $func_name(
            signal: &mut [Complex<$precision>],
            direction: Direction,
            planner: &$planner,
            opts: &Options,
        ) {
            let (mut reals, mut imags) = $deinterleaving_func(signal);
            $fft_func(&mut reals, &mut imags, direction, planner, opts);
            interleave_complex(&reals, &imags, signal);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(
    fft_f32_dit_interleaved_with_planner_and_opts,
    f32,
    fft_f32_dit_with_planner_and_opts,
    deinterleave_complex32,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(
    fft_f64_dit_interleaved_with_planner_and_opts,
    f64,
    fft_f64_dit_with_planner_and_opts,
    deinterleave_complex64,
    PlannerDit64
);

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_with_planner {
    ($func_name:ident, $precision:ty, $fft_with_opts_func:ident, $planner:ty) => {
        /// FFT Interleaved with pre-computed planner -- convenience wrapper around
        /// the `_with_planner_and_opts` variant that automatically guesses options.
        ///
        /// For better control over options, use the `_with_planner_and_opts` variant.
        pub fn $func_name(
            signal: &mut [Complex<$precision>],
            direction: Direction,
            planner: &$planner,
        ) {
            let opts = Options::guess_options(signal.len());
            $fft_with_opts_func(signal, direction, planner, &opts);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_with_planner!(
    fft_f32_dit_interleaved_with_planner,
    f32,
    fft_f32_dit_interleaved_with_planner_and_opts,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_with_planner!(
    fft_f64_dit_interleaved_with_planner,
    f64,
    fft_f64_dit_interleaved_with_planner_and_opts,
    PlannerDit64
);

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved {
    ($func_name:ident, $precision:ty, $fft_with_planner_func:ident, $planner:ty) => {
        /// FFT Interleaved -- convenience wrapper that creates a planner automatically.
        ///
        /// For better performance when running multiple FFTs of the same size,
        /// consider using the `_with_planner` variant.
        pub fn $func_name(signal: &mut [Complex<$precision>], direction: Direction) {
            let planner = <$planner>::new(signal.len());
            $fft_with_planner_func(signal, direction, &planner);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved!(
    fft_f32_dit_interleaved,
    f32,
    fft_f32_dit_interleaved_with_planner,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved!(
    fft_f64_dit_interleaved,
    f64,
    fft_f64_dit_interleaved_with_planner,
    PlannerDit64
);

#[cfg(feature = "complex-nums")]
macro_rules! impl_bluestein_interleaved_with_opts {
    ($func_name:ident, $precision:ty, $planar_func:ident, $deinterleaving_func:ident, $planner:ty) => {
        /// Interleaved [`Complex`] Bluestein FFT with reusable scratch.
        ///
        /// This wrapper allocates the two length-`N` arrays needed to
        /// deinterleave the input. The caller supplies the larger convolution
        /// scratch buffers. For a zero-allocation hot path, use
        #[doc = concat!("`", stringify!($planar_func), "` directly.")]
        ///
        /// # Panics
        ///
        /// Panics if `signal.len()` does not match `planner.fft_len()`, or if
        /// either scratch buffer is shorter than `planner.scratch_len()`.
        pub fn $func_name(
            signal: &mut [Complex<$precision>],
            direction: Direction,
            planner: &$planner,
            opts: &Options,
            scratch_re: &mut [$precision],
            scratch_im: &mut [$precision],
        ) {
            assert_eq!(
                signal.len(),
                planner.fft_len(),
                "signal length must match planner.fft_len()"
            );
            assert!(
                scratch_re.len() >= planner.scratch_len(),
                "scratch_re is too short: need at least {} elements, got {}",
                planner.scratch_len(),
                scratch_re.len()
            );
            assert!(
                scratch_im.len() >= planner.scratch_len(),
                "scratch_im is too short: need at least {} elements, got {}",
                planner.scratch_len(),
                scratch_im.len()
            );
            let (mut reals, mut imags) = $deinterleaving_func(signal);
            $planar_func(
                &mut reals, &mut imags, direction, planner, opts, scratch_re, scratch_im,
            );
            interleave_complex(&reals, &imags, signal);
        }
    };
}

#[cfg(feature = "complex-nums")]
macro_rules! impl_bluestein_interleaved_with_planner {
    ($func_name:ident, $precision:ty, $with_opts_func:ident, $planner:ty) => {
        /// Computes an interleaved [`Complex`] Bluestein FFT with a reusable plan.
        ///
        /// Scratch is allocated and `Options` are chosen for the inner FFT.
        ///
        /// # Panics
        ///
        /// Panics if `signal.len()` does not match `planner.fft_len()`.
        pub fn $func_name(
            signal: &mut [Complex<$precision>],
            direction: Direction,
            planner: &$planner,
        ) {
            let scratch_len = planner.scratch_len();
            let opts = Options::guess_options(scratch_len);
            let mut scratch_re = vec![0.0; scratch_len];
            let mut scratch_im = vec![0.0; scratch_len];
            $with_opts_func(
                signal,
                direction,
                planner,
                &opts,
                &mut scratch_re,
                &mut scratch_im,
            );
        }
    };
}

#[cfg(feature = "complex-nums")]
macro_rules! impl_bluestein_interleaved {
    ($func_name:ident, $precision:ty, $with_planner_func:ident, $planner:ty) => {
        /// Computes an interleaved [`Complex`] Bluestein FFT, building a plan
        /// for this call.
        ///
        /// # Panics
        ///
        /// Panics if `signal` is empty.
        pub fn $func_name(signal: &mut [Complex<$precision>], direction: Direction) {
            let planner = <$planner>::new(signal.len());
            $with_planner_func(signal, direction, &planner);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_bluestein_interleaved_with_opts!(
    fft_f64_bluestein_interleaved_with_planner_and_opts,
    f64,
    fft_f64_bluestein_with_planner_and_opts,
    deinterleave_complex64,
    PlannerBluestein64
);
#[cfg(feature = "complex-nums")]
impl_bluestein_interleaved_with_opts!(
    fft_f32_bluestein_interleaved_with_planner_and_opts,
    f32,
    fft_f32_bluestein_with_planner_and_opts,
    deinterleave_complex32,
    PlannerBluestein32
);

#[cfg(feature = "complex-nums")]
impl_bluestein_interleaved_with_planner!(
    fft_f64_bluestein_interleaved_with_planner,
    f64,
    fft_f64_bluestein_interleaved_with_planner_and_opts,
    PlannerBluestein64
);
#[cfg(feature = "complex-nums")]
impl_bluestein_interleaved_with_planner!(
    fft_f32_bluestein_interleaved_with_planner,
    f32,
    fft_f32_bluestein_interleaved_with_planner_and_opts,
    PlannerBluestein32
);

#[cfg(feature = "complex-nums")]
impl_bluestein_interleaved!(
    fft_f64_bluestein_interleaved,
    f64,
    fft_f64_bluestein_interleaved_with_planner,
    PlannerBluestein64
);
#[cfg(feature = "complex-nums")]
impl_bluestein_interleaved!(
    fft_f32_bluestein_interleaved,
    f32,
    fft_f32_bluestein_interleaved_with_planner,
    PlannerBluestein32
);

/// FFT using the Decimation-In-Time (DIT) algorithm for `f64`, reusing a
/// pre-computed planner. Options are guessed from the input size.
///
/// For full control over [`Options`], use [`fft_f64_dit_with_planner_and_opts`].
///
/// # Panics
///
/// Panics if the input length is not a power of two, or does not match the planner size.
pub fn fft_f64_dit_with_planner(
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerDit64,
) {
    let opts = Options::guess_options(reals.len());
    algorithms::dit::fft_f64_dit_with_planner_and_opts(reals, imags, direction, planner, &opts);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f64.
///
/// This is a convenient wrapper that creates a planner automatically.
/// For better performance when running multiple FFTs of the same size,
/// consider using [`fft_f64_dit_with_planner`].
///
/// # Arguments
///
/// * `reals` - Real parts of the complex numbers (modified in-place)
/// * `imags` - Imaginary parts of the complex numbers (modified in-place)
/// * `direction` - Forward or inverse transform
///
/// # Panics
///
/// Panics if the input length is not a power of 2.
///
/// # Example
///
/// ```
/// use phastft::{fft_f64_dit, planner::Direction};
///
/// let mut reals = vec![1.0, 0.0, 0.0, 0.0];
/// let mut imags = vec![0.0; 4];
/// fft_f64_dit(&mut reals, &mut imags, Direction::Forward);
/// // Output is in normal order
/// ```
///
pub fn fft_f64_dit(reals: &mut [f64], imags: &mut [f64], direction: Direction) {
    let planner = PlannerDit64::new(reals.len());
    fft_f64_dit_with_planner(reals, imags, direction, &planner);
}

/// FFT using the Decimation-In-Time (DIT) algorithm for `f32`, reusing a
/// pre-computed planner. Options are guessed from the input size.
///
/// For full control over [`Options`], use [`fft_f32_dit_with_planner_and_opts`].
///
/// # Panics
///
/// Panics if the input length is not a power of two, or does not match the planner size.
pub fn fft_f32_dit_with_planner(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerDit32,
) {
    let opts = Options::guess_options(reals.len());
    fft_f32_dit_with_planner_and_opts(reals, imags, direction, planner, &opts);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f32.
///
/// This is a convenient wrapper that creates a planner automatically.
/// For better performance when running multiple FFTs of the same size,
/// consider using [`fft_f32_dit_with_planner`].
///
/// # Arguments
///
/// * `reals` - Real parts of the complex numbers (modified in-place)
/// * `imags` - Imaginary parts of the complex numbers (modified in-place)
/// * `direction` - Forward or inverse transform
///
/// # Panics
///
/// Panics if the input length is not a power of 2.
///
/// # Example
///
/// ```
/// use phastft::{fft_f32_dit, planner::Direction};
///
/// let mut reals = vec![1.0, 0.0, 0.0, 0.0];
/// let mut imags = vec![0.0; 4];
/// fft_f32_dit(&mut reals, &mut imags, Direction::Forward);
/// // Output is in normal order
/// ```
///
pub fn fft_f32_dit(reals: &mut [f32], imags: &mut [f32], direction: Direction) {
    let planner = PlannerDit32::new(reals.len());
    fft_f32_dit_with_planner(reals, imags, direction, &planner);
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use utilities::rustfft::num_complex::Complex;
    use utilities::rustfft::FftPlanner;
    use utilities::{assert_float_closeness, gen_random_signal_f32, gen_random_signal_f64};

    use super::*;

    macro_rules! non_power_of_2_planner {
        ($test_name:ident, $planner:ty) => {
            #[should_panic]
            #[test]
            fn $test_name() {
                let num_points = 5;

                // this test _should_ always fail at this stage
                let _ = <$planner>::new(num_points);
            }
        };
    }

    non_power_of_2_planner!(non_power_of_2_planner_32, PlannerDit32);
    non_power_of_2_planner!(non_power_of_2_planner_64, PlannerDit64);

    macro_rules! wrong_num_points_in_planner {
        ($test_name:ident, $planner:ty, $fft_with_opts_and_plan:ident) => {
            // A regression test to make sure the `Planner` is compatible with fft execution.
            #[should_panic]
            #[test]
            fn $test_name() {
                let n = 16;
                let num_points = 1 << n; // 2.pow(n)

                // We purposely set n = 16 and pass it to the planner.
                // n == 16 == 2^{4} is clearly a power of two, so the planner won't throw it out.
                // However, the call to `fft_with_opts_and_plan` should panic since it tests that the
                // size of the generated twiddle factors is half the size of the input.
                // In this case, we have an input of size 1024 (used for mp3), but we tell the planner the
                // input size is 16.
                let mut planner = <$planner>::new(n);

                let mut reals = vec![0.0; num_points];
                let mut imags = vec![0.0; num_points];
                let opts = Options::guess_options(reals.len());

                // this call should panic
                $fft_with_opts_and_plan(
                    &mut reals,
                    &mut imags,
                    Direction::Forward,
                    &mut planner,
                    &opts,
                );
            }
        };
    }

    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_32,
        PlannerDit32,
        fft_f32_dit_with_planner_and_opts
    );
    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_64,
        PlannerDit64,
        fft_f64_dit_with_planner_and_opts
    );

    macro_rules! test_fft_correctness {
        ($test_name:ident, $precision:ty, $fft_type:ident, $range_start:literal, $range_end:literal) => {
            #[test]
            fn $test_name() {
                let range = Range {
                    start: $range_start,
                    end: $range_end,
                };

                for k in range {
                    let n: usize = 1 << k; // 2.pow(k)

                    let mut reals: Vec<$precision> = (1..=n).map(|i| i as $precision).collect();
                    let mut imags: Vec<$precision> = (1..=n).map(|i| i as $precision).collect();
                    $fft_type(&mut reals, &mut imags, Direction::Forward);

                    let mut buffer: Vec<Complex<$precision>> = (1..=n)
                        .map(|i| Complex::new(i as $precision, i as $precision))
                        .collect();

                    let mut planner = FftPlanner::new();
                    let fft = planner.plan_fft_forward(buffer.len());
                    fft.process(&mut buffer);

                    reals
                        .iter()
                        .zip(imags.iter())
                        .enumerate()
                        .for_each(|(i, (z_re, z_im))| {
                            let expect_re = buffer[i].re;
                            let expect_im = buffer[i].im;
                            assert_float_closeness(*z_re, expect_re, 0.01);
                            assert_float_closeness(*z_im, expect_im, 0.01);
                        });
                }
            }
        };
    }

    test_fft_correctness!(fft_correctness_32, f32, fft_f32_dit, 4, 9);
    test_fft_correctness!(fft_correctness_64, f64, fft_f64_dit, 4, 17);

    #[cfg(feature = "complex-nums")]
    #[test]
    fn fft_interleaved_correctness() {
        let n = 10;
        let big_n = 1 << n; // 2.pow(n)
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f64).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_f64_dit_interleaved(&mut actual_signal, Direction::Forward);
        fft_f64_dit(&mut expected_reals, &mut expected_imags, Direction::Forward);

        actual_signal
            .iter()
            .zip(expected_reals)
            .zip(expected_imags)
            .for_each(|((z, z_re), z_im)| {
                assert_float_closeness(z.re, z_re, 1e-10);
                assert_float_closeness(z.im, z_im, 1e-10);
            });

        let n = 10;
        let big_n = 1 << n; // 2.pow(n)
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f32, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f32).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_f32_dit_interleaved(&mut actual_signal, Direction::Forward);
        fft_f32_dit(&mut expected_reals, &mut expected_imags, Direction::Forward);

        actual_signal
            .iter()
            .zip(expected_reals)
            .zip(expected_imags)
            .for_each(|((z, z_re), z_im)| {
                assert_float_closeness(z.re, z_re, 1e-10);
                assert_float_closeness(z.im, z_im, 1e-10);
            });
    }

    #[cfg(feature = "complex-nums")]
    #[test]
    fn bluestein_interleaved_tiers_match_planar() {
        use utilities::assert_float_closeness;

        for direction in [Direction::Forward, Direction::Inverse] {
            for &len in &[3usize, 5, 17, 100, 127] {
                let input: Vec<Complex<f64>> = (1..=len)
                    .map(|i| Complex::new(i as f64, i as f64 * 0.5))
                    .collect();
                let mut expected_re: Vec<_> = input.iter().map(|value| value.re).collect();
                let mut expected_im: Vec<_> = input.iter().map(|value| value.im).collect();
                fft_f64_bluestein(&mut expected_re, &mut expected_im, direction);

                let mut simple = input.clone();
                fft_f64_bluestein_interleaved(&mut simple, direction);

                let planner = PlannerBluestein64::new(len);
                let mut planned = input.clone();
                fft_f64_bluestein_interleaved_with_planner(&mut planned, direction, &planner);

                let opts = Options::guess_options(planner.scratch_len());
                let mut scratch_re = vec![0.0; planner.scratch_len() + 3];
                let mut scratch_im = vec![0.0; planner.scratch_len() + 3];
                let mut with_scratch = input;
                fft_f64_bluestein_interleaved_with_planner_and_opts(
                    &mut with_scratch,
                    direction,
                    &planner,
                    &opts,
                    &mut scratch_re,
                    &mut scratch_im,
                );

                for (k, ((simple, planned), with_scratch)) in
                    simple.iter().zip(&planned).zip(&with_scratch).enumerate()
                {
                    for actual in [simple, planned, with_scratch] {
                        assert_float_closeness(actual.re, expected_re[k], 1e-9);
                        assert_float_closeness(actual.im, expected_im[k], 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn test_dit_fft_64_followed_by_ifft_correctness() {
        for n in 4..12 {
            let size = 1 << n; // 2.pow(n)
            let mut reals_original = vec![0.0f64; size];
            let mut imags_original = vec![0.0f64; size];
            let mut reals = vec![0.0f64; size];
            let mut imags = vec![0.0f64; size];

            gen_random_signal_f64(&mut reals_original, &mut imags_original);
            reals.copy_from_slice(&reals_original);
            imags.copy_from_slice(&imags_original);

            fft_f64_dit(&mut reals, &mut imags, Direction::Forward);

            fft_f64_dit(&mut reals, &mut imags, Direction::Inverse);

            for i in 0..size {
                assert_float_closeness(reals[i], reals_original[i], 1e-10);
                assert_float_closeness(imags[i], imags_original[i], 1e-10);
            }
        }
    }

    #[test]
    fn test_dit_fft_32_followed_by_ifft_correctness() {
        for n in 4..12 {
            let size = 1 << n; // 2.pow(n)
            let mut reals_original = vec![0.0f32; size];
            let mut imags_original = vec![0.0f32; size];
            let mut reals = vec![0.0f32; size];
            let mut imags = vec![0.0f32; size];

            gen_random_signal_f32(&mut reals_original, &mut imags_original);
            reals.copy_from_slice(&reals_original);
            imags.copy_from_slice(&imags_original);

            fft_f32_dit(&mut reals, &mut imags, Direction::Forward);
            fft_f32_dit(&mut reals, &mut imags, Direction::Inverse);

            for i in 0..size {
                assert_float_closeness(reals[i], reals_original[i], 1e-7);
                assert_float_closeness(imags[i], imags_original[i], 1e-7);
            }
        }
    }

    #[test]
    fn public_types_impl_standard_traits() {
        assert_eq!(Direction::Forward, Direction::Forward);
        assert_ne!(Direction::Forward, Direction::Inverse);
        assert_eq!(format!("{:?}", Direction::Forward), "Forward");

        let a = Options::guess_options(1 << 10);
        assert_eq!(a, a.clone());

        let planner = PlannerDit64::new(1 << 10);
        let _ = format!("{planner:?}"); // terse Debug must not panic
        let _ = planner.clone();

        let planner = planner::PlannerBluestein64::new(101);
        assert_eq!(planner.fft_len(), 101);
        let _ = format!("{planner:?}");
        let _ = planner.clone();
    }
}
