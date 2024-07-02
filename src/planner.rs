//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Currently, the planner is responsible for
//! pre-computing twiddle factors based on the input signal length, as well as the
//! direction of the FFT.
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
                let dir = match direction {
                    Direction::Forward => Direction::Forward,
                    Direction::Reverse => Direction::Reverse,
                };

                if num_points <= 4 {
                    return Self {
                        twiddles_re: vec![],
                        twiddles_im: vec![],
                        direction: dir,
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
                    direction: dir,
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
