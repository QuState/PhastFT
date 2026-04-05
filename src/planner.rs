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

macro_rules! impl_planner_dit_for {
    ($struct_name:ident, $precision:ident) => {
        /// DIT-specific planner
        ///
        /// Twiddle factors for stages with chunk_size > 64 are generated on the fly
        /// by the kernels using the arbitrary-offset approach, so no pre-computation
        /// is needed.
        pub struct $struct_name {
            /// The direction of the FFT
            pub(crate) direction: Direction,
            /// The log2 of the FFT size
            pub(crate) log_n: usize,
            /// The level of SIMD instruction support, detected at runtime on x86 and hardcoded elsewhere
            pub(crate) simd_level: fearless_simd::Level,
        }

        impl $struct_name {
            /// Create a DIT planner for an FFT of size `num_points`
            pub fn new(num_points: usize, direction: Direction) -> Self {
                assert!(num_points > 0 && num_points.is_power_of_two());

                let simd_level = fearless_simd::Level::new();
                let log_n = num_points.ilog2() as usize;

                Self {
                    direction,
                    log_n,
                    simd_level,
                }
            }
        }
    };
}

impl_planner_dit_for!(PlannerDit64, f64);
impl_planner_dit_for!(PlannerDit32, f32);
