//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Currently, the planner is responsible for
//! pre-computing twiddle factors based on the input signal length, as well as the
//! direction of the FFT.

/// Reverse is for running the Inverse Fast Fourier Transform (IFFT)
/// Forward is for running the regular FFT
#[derive(Copy, Clone)]
pub enum Direction {
    /// Leave the exponent term in the twiddle factor alone
    Forward = 1,
    /// Multiply the exponent term in the twiddle factor by -1
    Reverse = -1,
}

/// Controls how the planner selects algorithm variants.
///
/// The planner can choose between different internal strategies (e.g., whether to
/// use the fused 32-point codelet). `Heuristic` uses a conservative static rule
/// with zero planning overhead. `Tune` benchmarks both paths at plan time and
/// picks whichever is faster, at the cost of additional planning time.
#[derive(Copy, Clone, Debug, Default)]
pub enum PlannerMode {
    /// Use a conservative static heuristic. Zero overhead at plan time.
    #[default]
    Heuristic,
    /// Benchmark both paths at plan time, pick whichever is faster.
    /// Adds planning overhead proportional to FFT size.
    Tune,
}

macro_rules! impl_planner_dit_for {
    ($struct_name:ident, $precision:ident, $fft_func:path) => {
        /// DIT-specific planner that pre-computes twiddles for all stages.
        ///
        /// The planner is direction-agnostic. Namely, the same instance can drive both forward and
        /// inverse transforms. Direction is supplied per-call to the `fft_*_dit*` functions.
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
            /// Uses [`PlannerMode::Heuristic`] to decide algorithm variants.
            /// For explicit control, use [`Self::with_mode`].
            pub fn new(num_points: usize) -> Self {
                Self::with_mode(num_points, PlannerMode::Heuristic)
            }

            /// Create a DIT planner with explicit control over algorithm selection.
            ///
            /// - [`PlannerMode::Heuristic`]: Zero-cost static rule. Conservative — may
            ///   leave performance on the table on platforms with large L1i caches.
            /// - [`PlannerMode::Tune`]: Benchmarks both paths at plan time. Use this
            ///   when you can afford extra planning time (e.g., planner is reused).
            pub fn with_mode(num_points: usize, _mode: PlannerMode) -> Self {
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
    };
}

impl_planner_dit_for!(
    PlannerDit64,
    f64,
    crate::algorithms::dit::fft_64_dit_with_planner_and_opts
);
impl_planner_dit_for!(
    PlannerDit32,
    f32,
    crate::algorithms::dit::fft_32_dit_with_planner_and_opts
);

// ---------------------------------------------------------------------------
// R2C / C2R planners
// ---------------------------------------------------------------------------

fn compute_r2c_twiddles_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let half = n / 2;
    let mut w_re = vec![0.0f64; half];
    let mut w_im = vec![0.0f64; half];

    // Forward R2C twiddles W_N^k = exp(-2 * pi * i * k / N); C2R conjugates at use time.
    let angle_step = -std::f64::consts::PI / half as f64;
    let (st, ct) = angle_step.sin_cos();
    let (mut wr, mut wi) = (1.0f64, 0.0f64);

    for k in 0..half {
        w_re[k] = wr;
        w_im[k] = wi;
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

    // Forward R2C twiddles W_N^k = exp(-2 * pi * i * k / N); C2R conjugates at use time.
    // Compute in f64 to avoid recurrence drift, then cast to f32.
    let angle_step = -std::f64::consts::PI / half as f64;
    let (st, ct) = angle_step.sin_cos();
    let (mut wr, mut wi) = (1.0f64, 0.0f64);

    for k in 0..half {
        w_re[k] = wr as f32;
        w_im[k] = wi as f32;
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
        pub struct $struct_name {
            /// Inner DIT planner for the N/2 complex FFT
            pub(crate) dit_planner: $dit_planner,
            /// Pre-computed untangle twiddle factors (real parts)
            pub(crate) w_re: Vec<$precision>,
            /// Pre-computed untangle twiddle factors (imaginary parts)
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
                let half = n / 2;
                let (w_re, w_im) = $twiddle_fn(n);

                Self {
                    dit_planner: $dit_planner::new(half),
                    w_re,
                    w_im,
                    n,
                }
            }
        }
    };
}

impl_planner_r2c_for!(PlannerR2c64, f64, PlannerDit64, compute_r2c_twiddles_f64);
impl_planner_r2c_for!(PlannerR2c32, f32, PlannerDit32, compute_r2c_twiddles_f32);
