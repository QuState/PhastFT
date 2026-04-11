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
            /// Whether to use the fused 32-point codelet for stages 0-4 in L1 blocks
            pub(crate) use_codelet_32: bool,
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

                let use_codelet_32 = Self::estimate_use_codelet_32(log_n);

                let mut planner = Self {
                    stage_twiddles,
                    direction,
                    log_n,
                    simd_level,
                    use_codelet_32,
                };

                if matches!(mode, PlannerMode::Tune) {
                    planner.tune_codelet_32(num_points);
                }

                planner
            }

            /// Conservative, arch-independent heuristic for whether the 32-point
            /// codelet is beneficial.
            ///
            /// At small sizes (N ≤ 8192) the codelet dominates runtime and
            /// cross-block kernel eviction from the µop cache doesn't matter.
            /// At large sizes the codelet's code footprint can evict the
            /// cross-block kernel on platforms with small L1i caches (e.g., 32KB
            /// on x86). Use [`PlannerMode::Tune`] to discover the real threshold
            /// on your hardware.
            fn estimate_use_codelet_32(log_n: usize) -> bool {
                // Codelet needs at least 32 elements (5 stages)
                if log_n < 5 {
                    return false;
                }

                // Conservative threshold: enable only for N ≤ 8192 where the
                // codelet dominates runtime. On platforms with large L1i (e.g.,
                // Apple Silicon at 192KB), Tune mode will discover that the
                // codelet wins at larger sizes too.
                // log_n <= 13
                true
            }

            /// Benchmark both paths and set `use_codelet_32` to whichever is faster.
            fn tune_codelet_32(&mut self, num_points: usize) {
                if self.log_n < 5 {
                    self.use_codelet_32 = false;
                    return;
                }

                let opts = crate::options::Options {
                    multithreaded_bit_reversal: false,
                    smallest_parallel_chunk_size: usize::MAX,
                };

                // Generate random complex signal via xorshift64 (no rand dependency)
                let mut rng_state: u64 = 0x517C_C1B7_2722_0A95;
                let mut next_f = || -> $precision {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    (rng_state as $precision) / (u64::MAX as $precision) * 2.0 - 1.0
                };
                let reals_orig: Vec<$precision> = (0..num_points).map(|_| next_f()).collect();
                let imags_orig: Vec<$precision> = (0..num_points).map(|_| next_f()).collect();
                let mut reals = reals_orig.clone();
                let mut imags = imags_orig.clone();

                const WARMUP: usize = 3;
                const ITERS: usize = 5;

                // Time WITHOUT codelet
                self.use_codelet_32 = false;
                for _ in 0..WARMUP {
                    reals.copy_from_slice(&reals_orig);
                    imags.copy_from_slice(&imags_orig);
                    $fft_func(&mut reals, &mut imags, &*self, &opts);
                }

                let mut best_without = std::time::Duration::MAX;
                for _ in 0..ITERS {
                    reals.copy_from_slice(&reals_orig);
                    imags.copy_from_slice(&imags_orig);
                    let start = std::time::Instant::now();
                    $fft_func(&mut reals, &mut imags, &*self, &opts);
                    best_without = best_without.min(start.elapsed());
                }

                // Time WITH codelet
                self.use_codelet_32 = true;
                for _ in 0..WARMUP {
                    reals.copy_from_slice(&reals_orig);
                    imags.copy_from_slice(&imags_orig);
                    $fft_func(&mut reals, &mut imags, &*self, &opts);
                }

                let mut best_with = std::time::Duration::MAX;
                for _ in 0..ITERS {
                    reals.copy_from_slice(&reals_orig);
                    imags.copy_from_slice(&imags_orig);
                    let start = std::time::Instant::now();
                    $fft_func(&mut reals, &mut imags, &*self, &opts);
                    best_with = best_with.min(start.elapsed());
                }

                self.use_codelet_32 = best_with < best_without;
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
