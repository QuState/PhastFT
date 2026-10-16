//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Plans cache compact DIT twiddle factor
//! tables and SIMD detection, and can be reused in either transform direction.

use crate::twiddles::{TwiddleTable32, TwiddleTable64};

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
    ($struct_name:ident, $twiddles:ident) => {
        /// DIT-specific planner with compact twiddle tables for each stage.
        ///
        /// Table storage grows logarithmically with the transform size. Twiddles
        /// are generated from independent anchors and these small tables, avoiding
        /// the rounding drift of a long recurrence.
        ///
        /// The planner is direction-agnostic. Namely, the same instance can drive both forward and
        /// inverse transforms. Direction is supplied per-call to the `fft_*_dit*` functions.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Fixed-size factor tables for stages with chunk_size > 64.
            pub(crate) stage_twiddles: Vec<$twiddles>,
            /// The log2 of the FFT size
            pub(crate) log_n: usize,
            /// The level of SIMD instruction support, detected at runtime on x86 and hardcoded elsewhere
            pub(crate) simd_level: fearless_simd::Level,
        }

        impl $struct_name {
            /// Create a DIT planner for an FFT of size `num_points`.
            ///
            /// Pre-computes the compact per-stage factor tables and detects the SIMD
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
                let stage_twiddles = (6..log_n)
                    .map(|stage| $twiddles::new(1 << (stage + 1)))
                    .collect();

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

impl_planner_dit_for!(PlannerDit64, TwiddleTable64);
impl_planner_dit_for!(PlannerDit32, TwiddleTable32);

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
