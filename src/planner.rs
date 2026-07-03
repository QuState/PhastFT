//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Currently, the planner is responsible for
//! pre-computing twiddle factors based on the input signal length, as well as the
//! direction of the FFT.

use crate::options::Options;

/// Inverse is for running the Inverse Fast Fourier Transform (IFFT)
/// Forward is for running the regular FFT
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Leave the exponent term in the twiddle factor alone
    Forward = 1,
    /// Multiply the exponent term in the twiddle factor by -1
    Inverse = -1,
}

macro_rules! impl_planner_dit_for {
    ($struct_name:ident, $precision:ident, $fft_func:path) => {
        /// DIT-specific planner that pre-computes twiddles for all stages.
        ///
        /// The planner is direction-agnostic. Namely, the same instance can drive both forward and
        /// inverse transforms. Direction is supplied per-call to the `fft_*_dit*` functions.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Twiddles for each stage that needs them (stages with chunk_size > 64)
            /// Each element contains (twiddles_re, twiddles_im) for that stage
            pub(crate) stage_twiddles: Vec<(Vec<$precision>, Vec<$precision>)>,
            /// The log2 of the FFT size
            pub(crate) log_n: usize,
            /// The level of SIMD instruction support, detected at runtime on x86 and hardcoded elsewhere
            pub(crate) simd_level: fearless_simd::Level,
        }

        impl $struct_name {
            /// Create a DIT planner for an FFT of size `num_points`.
            ///
            /// Pre-computes the per-stage twiddle factors and detects the SIMD
            /// support level once, so the planner can be reused across many
            /// FFTs of the same size.
            ///
            /// # Panics
            ///
            /// Panics if `num_points` is not a power of two.
            pub fn new(num_points: usize) -> Self {
                assert!(num_points > 0 && num_points.is_power_of_two());

                let simd_level = fearless_simd::Level::new();

                let log_n = num_points.ilog2() as usize;
                let mut stage_twiddles = Vec::new();

                // Pre-compute twiddles for each stage that needs them
                for stage in 0..log_n {
                    let dist = 1 << stage; // 2.pow(stage)
                    let chunk_size = dist * 2;

                    // Only stages with chunk_size > 64 need twiddles (we have SIMD kernels up to 64)
                    if chunk_size > 64 {
                        let mut twiddles_re = vec![0.0 as $precision; dist];
                        let mut twiddles_im = vec![0.0 as $precision; dist];

                        let angle_mult =
                            -2.0 * std::$precision::consts::PI / chunk_size as $precision;
                        for k in 0..dist {
                            let angle = angle_mult * k as $precision;
                            twiddles_re[k] = angle.cos();
                            twiddles_im[k] = angle.sin();
                        }

                        stage_twiddles.push((twiddles_re, twiddles_im));
                    }
                }

                Self {
                    stage_twiddles,
                    log_n,
                    simd_level,
                }
            }
        }

        impl core::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct_name))
                    .field("fft_size", &(1usize << self.log_n))
                    .finish_non_exhaustive()
            }
        }
    };
}

impl_planner_dit_for!(
    PlannerDit64,
    f64,
    crate::algorithms::dit::fft_f64_dit_with_planner_and_opts
);
impl_planner_dit_for!(
    PlannerDit32,
    f32,
    crate::algorithms::dit::fft_f32_dit_with_planner_and_opts
);

// ---------------------------------------------------------------------------
// R2C / C2R planners
// ---------------------------------------------------------------------------

fn compute_r2c_twiddles_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let half = n / 2;
    let mut w_re = vec![0.0f64; half];
    let mut w_im = vec![0.0f64; half];

    // Forward R2C twiddles 0.5 * W_N^k = 0.5 * exp(-2 * pi * i * k / N).
    // The 0.5 factor is folded in here so the untangle / c2r-preprocess hot
    // loops avoid one multiply per bin. C2R conjugates at use time.
    let angle_step = -std::f64::consts::PI / half as f64;
    let (st, ct) = angle_step.sin_cos();
    let (mut wr, mut wi) = (1.0f64, 0.0f64);

    for k in 0..half {
        w_re[k] = 0.5 * wr;
        w_im[k] = 0.5 * wi;
        let tmp = wr;
        wr = tmp * ct - wi * st;
        wi = tmp * st + wi * ct;
    }

    (w_re, w_im)
}

fn compute_r2c_twiddles_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let half = n / 2;
    let mut w_re = vec![0.0f32; half];
    let mut w_im = vec![0.0f32; half];

    // 0.5 folded in (see f64 variant). Compute in f64 to avoid recurrence drift, then cast.
    let angle_step = -std::f64::consts::PI / half as f64;
    let (st, ct) = angle_step.sin_cos();
    let (mut wr, mut wi) = (1.0f64, 0.0f64);

    for k in 0..half {
        w_re[k] = (0.5 * wr) as f32;
        w_im[k] = (0.5 * wi) as f32;
        let tmp = wr;
        wr = tmp * ct - wi * st;
        wi = tmp * st + wi * ct;
    }

    (w_re, w_im)
}

macro_rules! impl_planner_r2c_for {
    ($struct_name:ident, $precision:ident, $dit_planner:ident, $twiddle_fn:ident) => {
        /// Planner for real-to-complex (R2C) and complex-to-real (C2R) FFTs.
        ///
        /// Pre-computes the inner DIT planner for the half-length complex FFT
        /// and the untangle twiddle factors for the post-processing step.
        ///
        /// The planner is direction-agnostic. Namely, the same instance can drive both
        /// R2C and C2R transforms.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Inner DIT planner for the N/2 complex FFT
            pub(crate) dit_planner: $dit_planner,
            /// Pre-computed untangle twiddle factors (real parts).
            /// 0.5 is pre-folded in so the hot loops avoid a per-bin multiply.
            pub(crate) w_re: Vec<$precision>,
            /// Pre-computed untangle twiddle factors (imaginary parts), 0.5 folded in.
            pub(crate) w_im: Vec<$precision>,
            /// Full real signal length N
            pub(crate) n: usize,
        }

        impl $struct_name {
            /// Create a planner for real FFTs of length `n`.
            ///
            /// # Panics
            ///
            /// Panics if `n` is not a power of 2 or `n < 4`.
            pub fn new(n: usize) -> Self {
                assert!(n >= 4 && n.is_power_of_two(), "n must be a power of 2 >= 4");
                let (w_re, w_im) = $twiddle_fn(n);

                Self {
                    dit_planner: $dit_planner::new(n / 2),
                    w_re,
                    w_im,
                    n,
                }
            }
        }

        impl core::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct_name))
                    .field("n", &self.n)
                    .finish_non_exhaustive()
            }
        }
    };
}

impl_planner_r2c_for!(PlannerR2c64, f64, PlannerDit64, compute_r2c_twiddles_f64);
impl_planner_r2c_for!(PlannerR2c32, f32, PlannerDit32, compute_r2c_twiddles_f32);

// ---------------------------------------------------------------------------
// Bluestein planners
// ---------------------------------------------------------------------------

fn bluestein_convolution_len(n: usize) -> usize {
    assert!(n > 0, "Bluestein FFT size must be greater than 0");

    let min_len = n
        .checked_sub(1)
        .and_then(|n_minus_one| n_minus_one.checked_mul(2))
        .and_then(|twice_n_minus_two| twice_n_minus_two.checked_add(1))
        .expect("Bluestein inner FFT size overflow");

    min_len
        .checked_next_power_of_two()
        .expect("Bluestein inner FFT size overflow")
}

/// Chirp table for `c[k] = exp(-i*pi*k^2/n)`.
fn compute_bluestein_chirp_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut c_re = vec![0.0f64; n];
    let mut c_im = vec![0.0f64; n];

    let two_n = 2 * n;
    let mut sq = 0usize;
    for k in 0..n {
        // Reduce k^2 modulo 2n before evaluating the phase.
        let angle = -core::f64::consts::PI * (sq as f64) / (n as f64);
        let (sin, cos) = angle.sin_cos();
        c_re[k] = cos;
        c_im[k] = sin;

        sq += 2 * k + 1;
        if sq >= two_n {
            sq -= two_n;
        }
    }

    (c_re, c_im)
}

/// `f32` chirps are computed in `f64` and cast to reduce phase error.
fn compute_bluestein_chirp_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let (c_re, c_im) = compute_bluestein_chirp_f64(n);
    (
        c_re.iter().map(|&x| x as f32).collect(),
        c_im.iter().map(|&x| x as f32).collect(),
    )
}

macro_rules! impl_planner_bluestein_for {
    ($struct_name:ident, $precision:ident, $dit_planner:ident, $dit_fft:path, $chirp_fn:ident) => {
        /// Planner for an arbitrary-length Bluestein (chirp-z) FFT.
        ///
        /// Holds the chirp table, the filter spectrum, and the inner DIT planner.
        ///
        /// The same planner can be used for forward and inverse transforms.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Inner DIT planner for the size-`M` convolution FFT.
            pub(crate) dit_planner: $dit_planner,
            /// Chirp table `c[k] = exp(-i*pi*k^2/N)` (real parts), length `N`.
            pub(crate) c_re: Vec<$precision>,
            /// Chirp table (imaginary parts), length `N`.
            pub(crate) c_im: Vec<$precision>,
            /// Precomputed filter spectrum `B = FFT(b)` (real parts), length `M`.
            pub(crate) b_re: Vec<$precision>,
            /// Precomputed filter spectrum (imaginary parts), length `M`.
            pub(crate) b_im: Vec<$precision>,
            /// Transform length `N`.
            pub(crate) n: usize,
            /// Inner convolution length `M = next_power_of_two(2N - 1)`.
            pub(crate) m: usize,
        }

        impl $struct_name {
            /// Create a Bluestein planner for transforms of length
            /// `num_points`, which may be any integer `>= 1`.
            ///
            /// # Panics
            ///
            /// Panics if `num_points` is 0, or if the required inner FFT length
            /// does not fit in `usize`.
            #[must_use]
            pub fn new(num_points: usize) -> Self {
                let n = num_points;
                let m = bluestein_convolution_len(n);
                let (c_re, c_im) = $chirp_fn(n);

                let mut b_re = vec![0.0; m];
                let mut b_im = vec![0.0; m];
                for j in 0..n {
                    b_re[j] = c_re[j];
                    b_im[j] = -c_im[j];
                }
                for j in 1..n {
                    b_re[m - j] = c_re[j];
                    b_im[m - j] = -c_im[j];
                }

                let dit_planner = $dit_planner::new(m);
                let opts = Options::guess_options(m);
                $dit_fft(
                    &mut b_re,
                    &mut b_im,
                    Direction::Forward,
                    &dit_planner,
                    &opts,
                );

                Self {
                    dit_planner,
                    c_re,
                    c_im,
                    b_re,
                    b_im,
                    n,
                    m,
                }
            }

            /// The inner power-of-2 convolution length `M = next_power_of_two(2N - 1)`.
            ///
            /// The `_with_planner_and_opts` tier needs `scratch_re` / `scratch_im`
            /// of this length, and its `Options` should be tuned for `M`.
            #[must_use]
            pub fn inner_fft_len(&self) -> usize {
                self.m
            }
        }

        impl core::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct_name))
                    .field("n", &self.n)
                    .field("m", &self.m)
                    .finish_non_exhaustive()
            }
        }
    };
}

impl_planner_bluestein_for!(
    PlannerBluestein64,
    f64,
    PlannerDit64,
    crate::algorithms::dit::fft_f64_dit_with_planner_and_opts,
    compute_bluestein_chirp_f64
);
impl_planner_bluestein_for!(
    PlannerBluestein32,
    f32,
    PlannerDit32,
    crate::algorithms::dit::fft_f32_dit_with_planner_and_opts,
    compute_bluestein_chirp_f32
);

#[cfg(test)]
mod bluestein_planner_tests {
    use super::*;

    #[test]
    fn inner_fft_len_is_next_pow2_of_2n_minus_1() {
        // M = next_pow2(2N - 1)
        let cases = [
            (1usize, 1usize),
            (2, 4),
            (3, 8),
            (5, 16),
            (7, 16),
            (17, 64),
            (1000, 2048),
        ];
        for (n, expected_m) in cases {
            assert_eq!(
                PlannerBluestein64::new(n).inner_fft_len(),
                expected_m,
                "n={n}"
            );
            assert_eq!(
                PlannerBluestein32::new(n).inner_fft_len(),
                expected_m,
                "n={n}"
            );
        }
    }

    #[test]
    fn accepts_non_power_of_two_sizes() {
        // Arbitrary (non-power-of-2) N must be accepted.
        for n in [3usize, 5, 6, 7, 100, 101] {
            let _ = PlannerBluestein64::new(n);
            let _ = PlannerBluestein32::new(n);
        }
    }

    #[test]
    #[should_panic]
    fn rejects_zero_64() {
        let _ = PlannerBluestein64::new(0);
    }

    #[test]
    #[should_panic]
    fn rejects_zero_32() {
        let _ = PlannerBluestein32::new(0);
    }

    #[test]
    #[should_panic(expected = "Bluestein inner FFT size overflow")]
    fn rejects_inner_fft_len_overflow() {
        let _ = PlannerBluestein64::new(usize::MAX / 2 + 1);
    }
}
