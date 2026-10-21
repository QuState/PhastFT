//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Plans cache twiddle factors and SIMD
//! detection, and can be reused in either transform direction.

use crate::twiddles::{StoredTwiddles, TwiddleTable32, TwiddleTable64};

/// Inverse is for running the Inverse Fast Fourier Transform (IFFT)
/// Forward is for running the regular FFT
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Leave the exponent term in the twiddle factor alone
    Forward = 1,
    /// Multiply the exponent term in the twiddle factor by -1
    Inverse = -1,
}

/// How a plan stores and produces twiddle factors.
///
/// This is a planning choice, independent of execution/threading options.
/// Both explicit modes support all transform sizes and directions.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TwiddleMode {
    /// Store all general-stage twiddles: O(N) plan storage and vector loads
    /// during execution.
    Precomputed,
    /// Store compact factor tables: O(log N) plan storage, generating twiddle
    /// vectors from independent anchors during execution. No long recurrence
    /// is used, so rounding error does not accumulate along a stage.
    LowMemory,
    /// Choose a mode once at plan construction using a size heuristic.
    /// The heuristic may change in future releases; select an explicit mode
    /// to override it. This is the default used by `PlannerDit*::new`.
    #[default]
    Auto,
}

// Initial cache-footprint heuristic; benchmarked with benches/twiddle_modes.rs.
// Express in bytes so f32 and f64 switch at comparable working-set sizes.
const AUTO_LOW_MEMORY_BYTES: usize = 1 << 20;

impl TwiddleMode {
    fn uses_low_memory<T>(self, num_points: usize) -> bool {
        match self {
            Self::Precomputed => false,
            Self::LowMemory => true,
            Self::Auto => num_points >= AUTO_LOW_MEMORY_BYTES / (2 * size_of::<T>()),
        }
    }
}

/// Store the mode once for the whole plan, including fused stage pairs.
#[derive(Clone)]
pub(crate) enum PlanTwiddles<T, Compact> {
    Precomputed(Vec<StoredTwiddles<T>>),
    LowMemory(Vec<Compact>),
}

impl<T, Compact> PlanTwiddles<T, Compact> {
    fn mode(&self) -> TwiddleMode {
        match self {
            Self::Precomputed(_) => TwiddleMode::Precomputed,
            Self::LowMemory(_) => TwiddleMode::LowMemory,
        }
    }
}

macro_rules! impl_planner_dit_for {
    ($struct_name:ident, $precision:ty, $twiddles:ident) => {
        /// DIT-specific planner with selectable twiddle storage.
        ///
        /// The planner is direction-agnostic. Namely, the same instance can drive both forward and
        /// inverse transforms. Direction is supplied per-call to the `fft_*_dit*` functions.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Twiddle sources for stages with chunk_size > 64.
            pub(crate) stage_twiddles: PlanTwiddles<$precision, $twiddles>,
            /// The log2 of the FFT size
            pub(crate) log_n: usize,
            /// The level of SIMD instruction support, detected at runtime on x86 and hardcoded elsewhere
            pub(crate) simd_level: fearless_simd::Level,
        }

        impl $struct_name {
            /// Create a DIT planner for an FFT of size `num_points`.
            ///
            /// Uses [`TwiddleMode::Auto`] and detects SIMD support once, so the
            /// planner can be reused across many FFTs of the same size.
            ///
            /// # Panics
            ///
            /// Panics if `num_points` is not a power of two.
            pub fn new(num_points: usize) -> Self {
                Self::with_twiddle_mode(num_points, TwiddleMode::Auto)
            }

            /// Create a DIT planner with explicit control over twiddle storage.
            ///
            /// `Auto` resolves once; [`Self::twiddle_mode`] reports the chosen mode.
            ///
            /// # Panics
            ///
            /// Panics if `num_points` is not a power of two.
            pub fn with_twiddle_mode(num_points: usize, mode: TwiddleMode) -> Self {
                assert!(num_points > 0 && num_points.is_power_of_two());

                let simd_level = fearless_simd::Level::new();

                let log_n = num_points.ilog2() as usize;
                let stage_twiddles = if mode.uses_low_memory::<$precision>(num_points) {
                    PlanTwiddles::LowMemory(
                        (6..log_n)
                            .map(|stage| $twiddles::new(1 << (stage + 1)))
                            .collect(),
                    )
                } else {
                    PlanTwiddles::Precomputed(
                        (6..log_n)
                            .map(|stage| StoredTwiddles::<$precision>::new(1 << (stage + 1)))
                            .collect(),
                    )
                };

                Self {
                    stage_twiddles,
                    log_n,
                    simd_level,
                }
            }

            /// The resolved storage mode, always `Precomputed` or `LowMemory`.
            pub fn twiddle_mode(&self) -> TwiddleMode {
                self.stage_twiddles.mode()
            }
        }

        impl core::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct_name))
                    .field("fft_size", &(1usize << self.log_n))
                    .field("twiddle_mode", &self.twiddle_mode())
                    .finish_non_exhaustive()
            }
        }
    };
}

impl_planner_dit_for!(PlannerDit64, f64, TwiddleTable64);
impl_planner_dit_for!(PlannerDit32, f32, TwiddleTable32);

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

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_modes {
        ($name:ident, $planner:ident, $float:ty) => {
            #[test]
            fn $name() {
                let threshold = AUTO_LOW_MEMORY_BYTES / (2 * size_of::<$float>());
                for (n, expected) in [
                    (1, TwiddleMode::Precomputed),
                    (threshold / 2, TwiddleMode::Precomputed),
                    (threshold, TwiddleMode::LowMemory),
                    (threshold * 2, TwiddleMode::LowMemory),
                ] {
                    assert_eq!($planner::new(n).twiddle_mode(), expected);
                    for mode in [TwiddleMode::Precomputed, TwiddleMode::LowMemory] {
                        let plan = $planner::with_twiddle_mode(n, mode);
                        assert_eq!(plan.twiddle_mode(), mode);
                        assert_eq!(plan.clone().twiddle_mode(), mode);
                    }
                }
                // A billion-point low-memory plan must stay small. Constructing
                // this also catches accidentally allocating the full tables.
                let plan = $planner::with_twiddle_mode(1 << 30, TwiddleMode::LowMemory);
                let PlanTwiddles::LowMemory(stages) = &plan.stage_twiddles else {
                    panic!("wrong storage mode");
                };
                assert_eq!(stages.len(), 30 - 6);
                assert!(std::mem::size_of_val(stages.as_slice()) < 32 * 1024);
            }
        };
    }

    test_modes!(dit_modes_f64, PlannerDit64, f64);
    test_modes!(dit_modes_f32, PlannerDit32, f32);
}
