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

fn bluestein_convolution_len(len: usize) -> usize {
    assert!(len > 0, "Bluestein FFT size must be greater than 0");

    let min_inner_len = len
        .checked_mul(2)
        .and_then(|twice_len| twice_len.checked_sub(1))
        .expect("Bluestein inner FFT size overflow");

    min_inner_len
        .checked_next_power_of_two()
        .expect("Bluestein inner FFT size overflow")
}

/// Returns `exp(-i * pi * k^2 / len)` for `k = 0..len`.
fn bluestein_chirp_f64(len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut re = vec![0.0; len];
    let mut im = vec![0.0; len];

    // The phase is periodic in k^2 modulo 2N. Tracking that residue avoids
    // overflowing k^2 and keeps the trigonometric argument small.
    let period = 2 * len;
    let mut square = 0usize;
    for k in 0..len {
        let angle = -core::f64::consts::PI * square as f64 / len as f64;
        let (sin, cos) = angle.sin_cos();
        re[k] = cos;
        im[k] = sin;

        square += 2 * k + 1;
        if square >= period {
            square -= period;
        }
    }

    (re, im)
}

fn bluestein_chirp_f32(len: usize) -> (Vec<f32>, Vec<f32>) {
    // Computing the phase in f64 noticeably reduces error for larger f32
    // transforms, while planning remains outside the hot path.
    let (re, im) = bluestein_chirp_f64(len);
    (
        re.into_iter().map(|value| value as f32).collect(),
        im.into_iter().map(|value| value as f32).collect(),
    )
}

macro_rules! impl_planner_bluestein_for {
    ($struct_name:ident, $precision:ident, $dit_planner:ident, $dit_fft:path, $chirp_fn:ident) => {
        /// Reusable plan for an arbitrary-length FFT using Bluestein's algorithm.
        ///
        /// The plan owns the chirp table, convolution-kernel spectrum, and
        /// inner DIT plan. It can execute both forward and inverse transforms.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Inner DIT planner for the convolution FFT.
            pub(crate) dit_planner: $dit_planner,
            /// `exp(-i * pi * k^2 / N)`, split into real and imaginary parts.
            pub(crate) chirp_re: Vec<$precision>,
            pub(crate) chirp_im: Vec<$precision>,
            /// Fourier transform of the convolution kernel, scaled by `1 / M`.
            pub(crate) kernel_fft_re: Vec<$precision>,
            pub(crate) kernel_fft_im: Vec<$precision>,
            /// Number of points transformed by this plan.
            pub(crate) len: usize,
            /// Power-of-two length of the inner convolution FFT.
            pub(crate) inner_len: usize,
        }

        impl $struct_name {
            /// Creates a plan for transforms of `num_points` complex values.
            ///
            /// # Panics
            ///
            /// Panics if `num_points` is 0, or if the required inner FFT length
            /// does not fit in `usize`.
            #[must_use]
            pub fn new(num_points: usize) -> Self {
                let len = num_points;
                let inner_len = bluestein_convolution_len(len);
                let (chirp_re, chirp_im) = $chirp_fn(len);

                // Scaling the kernel here lets execution use a second forward
                // FFT (with swapped components) as an unscaled inverse. That
                // avoids a separate normalization pass over the M-point buffer.
                let scale = (1.0 / inner_len as f64) as $precision;
                let mut kernel_fft_re = vec![0.0; inner_len];
                let mut kernel_fft_im = vec![0.0; inner_len];
                for j in 0..len {
                    kernel_fft_re[j] = chirp_re[j] * scale;
                    kernel_fft_im[j] = -chirp_im[j] * scale;
                }
                for j in 1..len {
                    kernel_fft_re[inner_len - j] = chirp_re[j] * scale;
                    kernel_fft_im[inner_len - j] = -chirp_im[j] * scale;
                }

                let dit_planner = $dit_planner::new(inner_len);
                let opts = Options::guess_options(inner_len);
                $dit_fft(
                    &mut kernel_fft_re,
                    &mut kernel_fft_im,
                    Direction::Forward,
                    &dit_planner,
                    &opts,
                );

                Self {
                    dit_planner,
                    chirp_re,
                    chirp_im,
                    kernel_fft_re,
                    kernel_fft_im,
                    len,
                    inner_len,
                }
            }

            /// Returns the number of points transformed by this plan.
            #[must_use]
            pub fn fft_len(&self) -> usize {
                self.len
            }

            /// Returns the minimum length of each scratch buffer.
            ///
            /// This is the power-of-two convolution length
            /// `next_power_of_two(2 * self.fft_len() - 1)`. Options for this
            /// plan should be tuned using this length.
            #[must_use]
            pub fn scratch_len(&self) -> usize {
                self.inner_len
            }
        }

        impl core::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct_name))
                    .field("fft_len", &self.len)
                    .field("scratch_len", &self.inner_len)
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
    bluestein_chirp_f64
);
impl_planner_bluestein_for!(
    PlannerBluestein32,
    f32,
    PlannerDit32,
    crate::algorithms::dit::fft_f32_dit_with_planner_and_opts,
    bluestein_chirp_f32
);

#[cfg(test)]
mod bluestein_planner_tests {
    use super::*;

    #[test]
    fn reports_transform_and_scratch_lengths() {
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
            let plan_f64 = PlannerBluestein64::new(n);
            assert_eq!(plan_f64.fft_len(), n);
            assert_eq!(plan_f64.scratch_len(), expected_m, "n={n}");

            let plan_f32 = PlannerBluestein32::new(n);
            assert_eq!(plan_f32.fft_len(), n);
            assert_eq!(plan_f32.scratch_len(), expected_m, "n={n}");
        }
    }

    #[test]
    fn accepts_non_power_of_two_sizes() {
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
    fn rejects_convolution_length_overflow() {
        let _ = PlannerBluestein64::new(usize::MAX / 2 + 1);
    }
}
