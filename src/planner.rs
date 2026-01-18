//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Currently, the planner is responsible for
//! pre-computing twiddle factors based on the input signal length, as well as the
//! direction of the FFT.
use crate::bencher::{
    guess_fastest_bit_reversal_impl, measure_fastest_bit_reversal_impl, BitRevFunc,
};
use crate::twiddles::{generate_twiddles, generate_twiddles_simd_32, generate_twiddles_simd_64};

/// Reverse is for running the Inverse Fast Fourier Transform (IFFT)
/// Forward is for running the regular FFT
#[derive(Copy, Clone)]
pub enum Direction {
    /// Leave the exponent term in the twiddle factor alone
    Forward = 1,
    /// Multiply the exponent term in the twiddle factor by -1
    Reverse = -1,
}

macro_rules! impl_planner_for {
    ($struct_name:ident, $precision:ident, $generate_twiddles_simd_fn:ident) => {
        /// The planner is responsible for pre-computing and storing twiddle factors for all the
        /// `log_2(N)` stages of the FFT.
        /// The amount of twiddle factors should always be a power of 2. In addition,
        /// the amount of twiddle factors should always be `(1/2) * N`
        pub struct $struct_name {
            /// The real components of the twiddle factors
            pub twiddles_re: Vec<$precision>,
            /// The imaginary components of the twiddle factors
            pub twiddles_im: Vec<$precision>,
            /// The direction of the FFT associated with this `Planner`
            pub direction: Direction,
        }
        impl $struct_name {
            /// Create a `Planner` for an FFT of size `num_points`.
            /// The twiddle factors are pre-computed based on the provided [`Direction`].
            /// For `Forward`, use [`Direction::Forward`].
            /// For `Reverse`, use [`Direction::Reverse`].
            ///
            /// # Panics
            ///
            /// Panics if `num_points < 1` or if `num_points` is __not__ a power of 2.
            pub fn new(num_points: usize, direction: Direction) -> Self {
                assert!(num_points > 0 && num_points.is_power_of_two());

                if num_points <= 4 {
                    return Self {
                        twiddles_re: vec![],
                        twiddles_im: vec![],
                        direction,
                    };
                }

                let dist = num_points >> 1;

                let (twiddles_re, twiddles_im) = if dist >= 8 * 2 {
                    $generate_twiddles_simd_fn(dist, Direction::Forward)
                } else {
                    generate_twiddles(dist, Direction::Forward)
                };

                assert_eq!(twiddles_re.len(), twiddles_im.len());

                Self {
                    twiddles_re,
                    twiddles_im,
                    direction,
                }
            }

            pub(crate) fn num_twiddles(&self) -> usize {
                assert_eq!(self.twiddles_re.len(), self.twiddles_im.len());
                self.twiddles_re.len()
            }
        }
    };
}

impl_planner_for!(Planner64, f64, generate_twiddles_simd_64);
impl_planner_for!(Planner32, f32, generate_twiddles_simd_32);

macro_rules! impl_planner_dit_for {
    ($struct_name:ident, $precision:ident) => {
        /// DIT-specific planner that pre-computes twiddles for all stages
        pub struct $struct_name {
            /// Twiddles for each stage that needs them (stages with chunk_size > 64)
            /// Each element contains (twiddles_re, twiddles_im) for that stage
            pub stage_twiddles: Vec<(Vec<$precision>, Vec<$precision>)>,
            /// The direction of the FFT
            pub direction: Direction,
            /// The log2 of the FFT size
            pub log_n: usize,
            /// The chosen bit reversal implementation
            pub bit_rev_impl: BitRevFunc,
        }

        impl $struct_name {
            /// Create a DIT planner for an FFT of size `num_points`
            ///
            /// Measures several different implementations to pick the fastest one on your hardware.
            /// If you'd like to skip that step, use [Self::new_oneshot].
            pub fn new(num_points: usize, direction: Direction) -> Self {
                Self::new_internal(num_points, direction, true)
            }

            /// Create a DIT planner for an FFT of size `num_points` and skip selection of the fastest algorithm
            ///
            /// This is faster if you only need to perform FFT once, so you don't get to reuse the planner for multiple executions.
            pub fn new_oneshot(num_points: usize, direction: Direction) -> Self {
                Self::new_internal(num_points, direction, false)
            }

            fn new_internal(
                num_points: usize,
                direction: Direction,
                measure_bit_rev: bool,
            ) -> Self {
                assert!(num_points > 0 && num_points.is_power_of_two());

                let log_n = num_points.ilog2() as usize;
                let mut stage_twiddles = Vec::new();

                // Pre-compute twiddles for each stage that needs them
                for stage in 0..log_n {
                    let dist = 1 << stage;
                    let chunk_size = dist << 1;

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

                let bit_rev_impl = if measure_bit_rev {
                    measure_fastest_bit_reversal_impl(log_n)
                } else {
                    guess_fastest_bit_reversal_impl(log_n)
                };

                Self {
                    stage_twiddles,
                    direction,
                    log_n,
                    bit_rev_impl,
                }
            }
        }
    };
}

impl_planner_dit_for!(PlannerDit64, f64);
impl_planner_dit_for!(PlannerDit32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_no_twiddles {
        ($test_name:ident, $planner:ty) => {
            #[test]
            fn $test_name() {
                for num_points in [2, 4] {
                    let planner = <$planner>::new(num_points, Direction::Forward);
                    assert!(planner.twiddles_im.is_empty() && planner.twiddles_re.is_empty());
                }
            }
        };
    }

    test_no_twiddles!(no_twiddles_64, Planner64);
    test_no_twiddles!(no_twiddles_32, Planner32);
}
