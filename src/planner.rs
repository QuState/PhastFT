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
        /// DIT-specific planner that pre-computes twiddles for all stages
        pub struct $struct_name {
            /// Twiddles for each stage that needs them (stages with chunk_size > 64)
            /// Each element contains (twiddles_re, twiddles_im) for that stage
            pub(crate) stage_twiddles: Vec<(Vec<$precision>, Vec<$precision>)>,
            /// The direction of the FFT
            pub(crate) direction: Direction,
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
            pub fn new(num_points: usize, direction: Direction) -> Self {
                Self::with_mode(num_points, direction, PlannerMode::Heuristic)
            }

            /// Create a DIT planner with explicit control over algorithm selection.
            ///
            /// - [`PlannerMode::Heuristic`]: Zero-cost static rule. Conservative — may
            ///   leave performance on the table on platforms with large L1i caches.
            /// - [`PlannerMode::Tune`]: Benchmarks both paths at plan time. Use this
            ///   when you can afford extra planning time (e.g., planner is reused).
            pub fn with_mode(num_points: usize, direction: Direction, mode: PlannerMode) -> Self {
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

                let planner = Self {
                    stage_twiddles,
                    direction,
                    log_n,
                    simd_level,
                };

                planner
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
