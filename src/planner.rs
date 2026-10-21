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

/// Real-transform post/pre-processing uses the same sources, with 0.5 folded in.
#[derive(Clone)]
pub(crate) enum RealTwiddles<T, Compact> {
    Precomputed(StoredTwiddles<T>),
    LowMemory(Box<Compact>),
}

macro_rules! impl_planner_r2c_for {
    ($struct_name:ident, $precision:ident, $dit_planner:ident, $compact:ident) => {
        /// Planner for real-to-complex (R2C) and complex-to-real (C2R) FFTs.
        ///
        /// Uses the selected twiddle mode for both the inner half-length complex
        /// FFT and the real-transform post/pre-processing. In low-memory mode,
        /// total twiddle storage is O(log N). Input, output, and C2R scratch
        /// requirements are unchanged.
        ///
        /// The planner is direction-agnostic. Namely, the same instance can drive both
        /// R2C and C2R transforms.
        #[derive(Clone)]
        pub struct $struct_name {
            /// Inner DIT planner for the N/2 complex FFT
            pub(crate) dit_planner: $dit_planner,
            /// Untangle/preprocess twiddles, including the 0.5 scale factor.
            pub(crate) twiddles: RealTwiddles<$precision, $compact>,
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
                Self::with_twiddle_mode(n, TwiddleMode::Auto)
            }

            /// Create a real-transform planner with selectable twiddle storage.
            ///
            /// `Auto` resolves using the half-length complex FFT's working-set
            /// size. The resolved mode applies to both parts of the real plan.
            ///
            /// # Panics
            ///
            /// Panics if `n` is not a power of 2 or `n < 4`.
            pub fn with_twiddle_mode(n: usize, mode: TwiddleMode) -> Self {
                assert!(n >= 4 && n.is_power_of_two(), "n must be a power of 2 >= 4");
                let dit_planner = $dit_planner::with_twiddle_mode(n / 2, mode);
                let twiddles = match &dit_planner.stage_twiddles {
                    PlanTwiddles::Precomputed(_) => {
                        RealTwiddles::Precomputed(StoredTwiddles::<$precision>::new_scaled(n, 0.5))
                    }
                    PlanTwiddles::LowMemory(_) => {
                        RealTwiddles::LowMemory(Box::new($compact::new_scaled(n, 0.5)))
                    }
                };
                Self {
                    dit_planner,
                    twiddles,
                    n,
                }
            }

            /// The resolved mode used by both the inner FFT and real twiddles.
            pub fn twiddle_mode(&self) -> TwiddleMode {
                self.dit_planner.twiddle_mode()
            }
        }

        impl core::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct_name))
                    .field("n", &self.n)
                    .field("twiddle_mode", &self.twiddle_mode())
                    .finish_non_exhaustive()
            }
        }
    };
}

impl_planner_r2c_for!(PlannerR2c64, f64, PlannerDit64, TwiddleTable64);
impl_planner_r2c_for!(PlannerR2c32, f32, PlannerDit32, TwiddleTable32);

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

    macro_rules! test_real_modes {
        ($name:ident, $planner:ident, $float:ty) => {
            #[test]
            fn $name() {
                let threshold = AUTO_LOW_MEMORY_BYTES / size_of::<$float>();
                for (n, expected) in [
                    (4, TwiddleMode::Precomputed),
                    (threshold / 2, TwiddleMode::Precomputed),
                    (threshold, TwiddleMode::LowMemory),
                ] {
                    assert_eq!($planner::new(n).twiddle_mode(), expected);
                    for mode in [TwiddleMode::Precomputed, TwiddleMode::LowMemory] {
                        let plan = $planner::with_twiddle_mode(n, mode);
                        assert_eq!(plan.twiddle_mode(), mode);
                        assert_eq!(plan.clone().twiddle_mode(), mode);
                        assert_eq!(plan.dit_planner.twiddle_mode(), mode);
                        assert_eq!(
                            matches!(plan.twiddles, RealTwiddles::LowMemory(_)),
                            mode == TwiddleMode::LowMemory
                        );
                    }
                }
                // Also covers the real post-processing table: no O(N) allocation.
                let plan = $planner::with_twiddle_mode(1 << 30, TwiddleMode::LowMemory);
                assert!(matches!(plan.twiddles, RealTwiddles::LowMemory(_)));
            }
        };
    }

    test_real_modes!(real_modes_f64, PlannerR2c64, f64);
    test_real_modes!(real_modes_f32, PlannerR2c32, f32);
}
