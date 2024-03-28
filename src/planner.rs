//! The planner module provides a convenient interface for planning and executing
//! a Fast Fourier Transform (FFT). Currently, the planner is responsible for
//! pre-computing twiddle factors based on the input signal length, as well as the
//! direction of the FFT.
use crate::twiddles::{generate_twiddles, generate_twiddles_simd_32, generate_twiddles_simd_64};

/// Reverse is for running the Inverse Fast Fourier Transform (IFFT)
/// Forward is for running the regular FFT
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
                    };
                }

                let dist = num_points >> 1;

                let (twiddles_re, twiddles_im) = if dist >= 8 * 2 {
                    $generate_twiddles_simd_fn(dist, direction)
                } else {
                    generate_twiddles(dist, direction)
                };

                assert_eq!(twiddles_re.len(), twiddles_im.len());

                Self {
                    twiddles_re,
                    twiddles_im,
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
    use utilities::assert_float_closeness;

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

    macro_rules! forward_mul_inverse_eq_identity {
        ($test_name:ident, $planner:ty) => {
            #[test]
            fn $test_name() {
                for i in 3..25 {
                    let num_points = 1 << i;
                    let planner_forward = <$planner>::new(num_points, Direction::Forward);
                    let planner_reverse = <$planner>::new(num_points, Direction::Reverse);

                    assert_eq!(
                        planner_reverse.num_twiddles(),
                        planner_forward.num_twiddles()
                    );

                    // (a + ib) (c + id) = ac + iad + ibc - bd
                    //                   = ac - bd + i(ad + bc)
                    planner_forward
                        .twiddles_re
                        .iter()
                        .zip(planner_forward.twiddles_im.iter())
                        .zip(planner_reverse.twiddles_re.iter())
                        .zip(planner_reverse.twiddles_im)
                        .for_each(|(((a, b), c), d)| {
                            let temp_re = a * c - b * d;
                            let temp_im = a * d + b * c;
                            assert_float_closeness(temp_re, 1.0, 1e-2);
                            assert_float_closeness(temp_im, 0.0, 1e-2);
                        });
                }
            }
        };
    }

    forward_mul_inverse_eq_identity!(forward_reverse_eq_identity_64, Planner64);
    forward_mul_inverse_eq_identity!(forward_reverse_eq_identity_32, Planner32);
}
