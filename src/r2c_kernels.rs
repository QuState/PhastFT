use crate::cobra::cobra_apply;
use crate::Direction;
use crate::filter_twiddles;
use crate::options::Options;
use crate::planner::{Planner32, Planner64};

use crate::planner::{Planner32, Planner64};
macro_rules! impl_r2c_fft_for {
    ($func_name:ident, $precision:ty, $planner:ty, $opts_and_plan:ident) => {
        /// FFT -- Decimation in Frequency. This is just the decimation-in-time algorithm, reversed.
        /// This call to FFT is run, in-place.
        /// The input should be provided in normal order, and then the modified input is bit-reversed.
        ///
        /// # Panics
        ///
        /// Panics if `reals.len() != imags.len()`, or if the input length is _not_ a power of 2.
        ///
        /// ## References
        /// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
        pub fn $func_name(reals: &mut [$precision], direction: Direction) -> Vec<$precision> {
            assert_eq!(
                reals.len(),
                imags.len(),
                "real and imaginary inputs must be of equal size, but got: {} {}",
                reals.len(),
                imags.len()
            );

            let mut planner = <$planner>::new(reals.len(), direction);
            assert!(
                planner.num_twiddles().is_power_of_two()
                    && planner.num_twiddles() == reals.len() / 2
            );

            let opts = Options::guess_options(reals.len());
            $opts_and_plan(reals, imags, &opts, &mut planner);
        }
    };
}

macro_rules! impl_r2c_fft_with_opts_and_plan_for {
    ($func_name:ident, $precision:ty, $planner:ty, $simd_butterfly_kernel:ident, $lanes:literal) => {
        /// Same as [fft], but also accepts [`Options`] that control optimization strategies, as well as
        /// a [`Planner`] in the case that this FFT will need to be run multiple times.
        ///
        /// `fft` automatically guesses the best strategy for a given input,
        /// so you only need to call this if you are tuning performance for a specific hardware platform.
        ///
        /// In addition, `fft` automatically creates a planner to be used. In the case that you plan
        /// on running an FFT many times on inputs of the same size, use this function with the pre-built
        /// [`Planner`].
        ///
        /// # Panics
        ///
        /// Panics if `reals.len() != imags.len()`, or if the input length is _not_ a power of 2.
        #[multiversion::multiversion(
                                    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                            "x86_64+avx2+fma", // x86_64-v3
                                            "x86_64+sse4.2", // x86_64-v2
                                            "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                            "x86+avx2+fma",
                                            "x86+sse4.2",
                                            "x86+sse2",
        ))]
        pub fn $func_name(
            reals: &mut [$precision],
            imags: &mut [$precision],
            opts: &Options,
            planner: &mut $planner,
        ) {
            assert!(reals.len() == imags.len() && reals.len().is_power_of_two());
            let n: usize = reals.len().ilog2() as usize;

            let twiddles_re = &mut planner.twiddles_re;
            let twiddles_im = &mut planner.twiddles_im;

            // We shouldn't be able to execute FFT if the # of twiddles isn't equal to the distance
            // between pairs
            assert!(twiddles_re.len() == reals.len() / 2 && twiddles_im.len() == imags.len() / 2);

            for t in (0..n).rev() {
                let dist = 1 << t;
                let chunk_size = dist << 1;

                if chunk_size > 4 {
                    if t < n - 1 {
                        filter_twiddles(twiddles_re, twiddles_im);
                    }
                    if chunk_size >= $lanes * 2 {
                        $simd_butterfly_kernel(reals, imags, twiddles_re, twiddles_im, dist);
                    } else {
                        fft_r2c_chunk_n(reals, imags, twiddles_re, twiddles_im, dist);
                    }
                } else if chunk_size == 2 {
                    fft_r2c_chunk_2(reals, imags);
                } else if chunk_size == 4 {
                    fft_r2c_chunk_4(reals, imags);
                }
            }

            if opts.multithreaded_bit_reversal {
                std::thread::scope(|s| {
                    s.spawn(|| cobra_apply(reals, n));
                    s.spawn(|| cobra_apply(imags, n));
                });
            } else {
                cobra_apply(reals, n);
                cobra_apply(imags, n);
            }
        }
    };
}

impl_r2c_fft_with_opts_and_plan_for!(
    fft_64_r2c_with_opts_and_plan,
    f64,
    Planner64,
    fft_64_r2c_chunk_n_simd,
    8
);

impl_r2c_fft_with_opts_and_plan_for!(
    fft_32_r2c_with_opts_and_plan,
    f32,
    Planner32,
    fft_32_r2c_chunk_n_simd,
    16
);

impl_r2c_fft_for!(fft_64_r2c, f64, Planner64, fft_64_r2c_with_opts_and_plan);
impl_r2c_fft_for!(fft_32_r2c, f32, Planner32, fft_32_r2c_with_opts_and_plan);

macro_rules! fft_r2c_butterfly_n_simd {
    ($func_name:ident, $precision:ty, $lanes:literal, $simd_vector:ty) => {
        #[inline]
        pub fn $func_name(
            reals: &mut [$precision],
            imags: &mut [$precision],
            twiddles_re: &[$precision],
            twiddles_im: &[$precision],
            dist: usize,
        ) {
            let chunk_size = dist << 1;
            assert!(chunk_size >= $lanes * 2);
            reals
                .chunks_exact_mut(chunk_size)
                .zip(imags.chunks_exact_mut(chunk_size))
                .for_each(|(reals_chunk, imags_chunk)| {
                    let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
                    let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

                    reals_s0
                        .chunks_exact_mut($lanes)
                        .zip(reals_s1.chunks_exact_mut($lanes))
                        .zip(imags_s0.chunks_exact_mut($lanes))
                        .zip(imags_s1.chunks_exact_mut($lanes))
                        .zip(twiddles_re.chunks_exact($lanes))
                        .zip(twiddles_im.chunks_exact($lanes))
                        .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                            let real_c0 = <$simd_vector>::from_slice(re_s0);
                            let real_c1 = <$simd_vector>::from_slice(re_s1);
                            let tw_re = <$simd_vector>::from_slice(tw_re);
                            let tw_im = <$simd_vector>::from_slice(tw_im);

                            re_s0.copy_from_slice((real_c0 + real_c1).as_array());

                            let temp_real = real_c0 - real_c1;
                            let twiddled_real = temp_real * tw_re;
                            let twiddled_imag = temp_real * tw_im;

                            re_s1.copy_from_slice(twiddled_real.as_array());
                            im_s0.copy_from_slice(twiddled_imag.as_array());
                            im_s1.copy_from_slice((-twiddled_imag).as_array()); // Negative due to symmetry
                        });
                });
        }
    };
}
