use std::simd::{f32x8, f64x8};

use num_traits::{Float, FloatConst, One, Zero};

use crate::planner::Direction;

/// This isn't really used except for testing.
/// It may be better to use this in the case where the input size is very large,
/// as to free up the cache.
pub(crate) struct Twiddles<T: Float> {
    st: T,
    ct: T,
    w_re_prev: T,
    w_im_prev: T,
}

impl<T: Float + FloatConst> Twiddles<T> {
    /// `cache_size` is the amount of roots of unity kept pre-built at any point in time.
    /// `num_roots` is the total number of roots of unity that will need to be computed.
    /// `cache_size` can be thought of as the length of a chunk of roots of unity from
    /// out of the total amount (i.e., `num_roots`)
    #[allow(dead_code)]
    pub fn new(num_roots: usize) -> Self {
        let theta = -T::PI() / (T::from(num_roots).unwrap());
        let (st, ct) = theta.sin_cos();
        Self {
            st,
            ct,
            w_re_prev: T::one(),
            w_im_prev: T::zero(),
        }
    }
}

// TODO: generate twiddles using the first quarter chunk of twiddle factors
// 1st chunk: old fashioned multiplication of complex nums
// 2nd chunk: reverse the 1st chunk, swap components, and negate both components
// 3rd chunk: No reversal. Swap the components and negate the *new* imaginary components
// 4th chunk: reverse the 1st chunk, and negate the real component
impl<T: Float> Iterator for Twiddles<T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<(T, T)> {
        let w_re = self.w_re_prev;
        let w_im = self.w_im_prev;

        let temp = self.w_re_prev;
        self.w_re_prev = temp * self.ct - self.w_im_prev * self.st;
        self.w_im_prev = temp * self.st + self.w_im_prev * self.ct;

        Some((w_re, w_im))
    }
}

pub fn generate_twiddles<T: Float + FloatConst>(
    dist: usize,
    direction: Direction,
) -> (Vec<T>, Vec<T>) {
    let mut twiddles_re = vec![T::zero(); dist];
    let mut twiddles_im = vec![T::zero(); dist];
    twiddles_re[0] = T::one();

    let sign = match direction {
        Direction::Forward => T::one(),
        Direction::Reverse => -T::one(),
    };

    let angle = sign * -T::PI() / T::from(dist).unwrap();
    let (st, ct) = angle.sin_cos();
    let (mut w_re, mut w_im) = (T::one(), T::zero());

    let mut i = 1;
    while i < (dist / 2) + 1 {
        let temp = w_re;
        w_re = w_re * ct - w_im * st;
        w_im = temp * st + w_im * ct;
        twiddles_re[i] = w_re;
        twiddles_im[i] = w_im;
        i += 1;
    }

    while i < dist {
        twiddles_re[i] = -twiddles_re[dist - i];
        twiddles_im[i] = twiddles_im[dist - i];
        i += 1;
    }

    (twiddles_re, twiddles_im)
}

macro_rules! generate_twiddles_simd {
    ($func_name:ident, $precision:ty, $lanes:literal, $simd_vector:ty) => {
        pub(crate) fn $func_name(
            dist: usize,
            direction: Direction,
        ) -> (Vec<$precision>, Vec<$precision>) {
            const CHUNK_SIZE: usize = 8; // TODO: make this a const generic?
            assert!(dist >= CHUNK_SIZE * 2);
            assert_eq!(dist % CHUNK_SIZE, 0);
            let mut twiddles_re = vec![0.0; dist];
            let mut twiddles_im = vec![0.0; dist];
            twiddles_re[0] = 1.0;

            let sign = match direction {
                Direction::Forward => 1.0,
                Direction::Reverse => -1.0,
            };

            let angle = sign * -<$precision>::PI() / dist as $precision;
            let (st, ct) = angle.sin_cos();
            let (mut w_re, mut w_im) = (<$precision>::one(), <$precision>::zero());

            let mut next_twiddle = || {
                let temp = w_re;
                w_re = w_re * ct - w_im * st;
                w_im = temp * st + w_im * ct;
                (w_re, w_im)
            };

            let apply_symmetry_re = |input: &[$precision], output: &mut [$precision]| {
                let first_re = <$simd_vector>::from_slice(input);
                let minus_one = <$simd_vector>::splat(-1.0);
                let negated = (first_re * minus_one).reverse();
                output.copy_from_slice(negated.as_array());
            };

            let apply_symmetry_im = |input: &[$precision], output: &mut [$precision]| {
                let mut buf: [$precision; CHUNK_SIZE] = [0.0; CHUNK_SIZE];
                buf.copy_from_slice(input);
                buf.reverse();
                output.copy_from_slice(&buf);
            };

            // Split the twiddles into two halves. There is a cheaper way to calculate the second half
            let (first_half_re, second_half_re) = twiddles_re[1..].split_at_mut(dist / 2);
            assert_eq!(first_half_re.len(), second_half_re.len() + 1);
            let (first_half_im, second_half_im) = twiddles_im[1..].split_at_mut(dist / 2);
            assert_eq!(first_half_im.len(), second_half_im.len() + 1);

            first_half_re
                .chunks_exact_mut(CHUNK_SIZE)
                .zip(first_half_im.chunks_exact_mut(CHUNK_SIZE))
                .zip(
                    second_half_re[CHUNK_SIZE - 1..]
                        .chunks_exact_mut(CHUNK_SIZE)
                        .rev(),
                )
                .zip(
                    second_half_im[CHUNK_SIZE - 1..]
                        .chunks_exact_mut(CHUNK_SIZE)
                        .rev(),
                )
                .for_each(
                    |(((first_ch_re, first_ch_im), second_ch_re), second_ch_im)| {
                        // Calculate a chunk of the first half in a plain old scalar way
                        first_ch_re
                            .iter_mut()
                            .zip(first_ch_im.iter_mut())
                            .for_each(|(re, im)| {
                                (*re, *im) = next_twiddle();
                            });
                        // Calculate a chunk of the second half in a clever way by copying the first chunk
                        // This avoids data dependencies of the regular calculation and gets vectorized.
                        // We do it up front while the values we just calculated are still in the cache,
                        // so we don't have to re-load them from memory later, which would be slow.
                        apply_symmetry_re(first_ch_re, second_ch_re);
                        apply_symmetry_im(first_ch_im, second_ch_im);
                    },
                );

            // Fill in the middle that the SIMD loop didn't
            twiddles_re[dist / 2 - CHUNK_SIZE + 1..][..(CHUNK_SIZE * 2) - 1]
                .iter_mut()
                .zip(twiddles_im[dist / 2 - CHUNK_SIZE + 1..][..(CHUNK_SIZE * 2) - 1].iter_mut())
                .for_each(|(re, im)| {
                    (*re, *im) = next_twiddle();
                });

            (twiddles_re, twiddles_im)
        }
    };
}

generate_twiddles_simd!(generate_twiddles_simd_64, f64, 8, f64x8);
generate_twiddles_simd!(generate_twiddles_simd_32, f32, 8, f32x8);

#[multiversion::multiversion(
                            targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                    "x86_64+avx2+fma", // x86_64-v3
                                    "x86_64+sse4.2", // x86_64-v2
                                    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                    "x86+avx2+fma",
                                    "x86+sse4.2",
                                    "x86+sse2",
))]
#[inline]
pub(crate) fn filter_twiddles<T: Float>(twiddles_re: &[T], twiddles_im: &[T]) -> (Vec<T>, Vec<T>) {
    assert_eq!(twiddles_re.len(), twiddles_im.len());
    let dist = twiddles_re.len();

    let filtered_twiddles_re: Vec<T> = twiddles_re.chunks_exact(2).map(|chunk| chunk[0]).collect();
    let filtered_twiddles_im: Vec<T> = twiddles_im.chunks_exact(2).map(|chunk| chunk[0]).collect();

    assert!(
        filtered_twiddles_re.len() == filtered_twiddles_im.len()
            && filtered_twiddles_re.len() == dist / 2
    );

    (filtered_twiddles_re, filtered_twiddles_im)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use utilities::assert_float_closeness;

    use super::*;

    // TODO(saveliy): try to use only real twiddle factors since sin is just a phase shift of cos
    #[test]
    fn twiddles_cos_only() {
        let n = 4;
        let big_n = 1 << n;

        let dist = big_n >> 1;

        let (fwd_twiddles_re, fwd_twiddles_im) = if dist >= 8 * 2 {
            generate_twiddles_simd_64(dist, Direction::Forward)
        } else {
            generate_twiddles(dist, Direction::Forward)
        };

        assert!(fwd_twiddles_re.len() == dist && fwd_twiddles_im.len() == dist);

        for i in 0..dist {
            let _w_re = fwd_twiddles_re[i];
            let expected_w_im = fwd_twiddles_im[i];

            let actual_w_im = -fwd_twiddles_re[(i + dist / 2) % dist];
            //assert_float_closeness(actual_w_im, expected_w_im, 1e-6);
            println!("actual: {actual_w_im} expected: {expected_w_im}");
        }
        println!("{:?}", fwd_twiddles_re);
        println!("{:?}", fwd_twiddles_im);
    }

    #[test]
    fn twiddles_4() {
        const N: usize = 4;
        let mut twiddle_iter = Twiddles::new(N);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_float_closeness(w_re, 1.0, 1e-10);
        assert_float_closeness(w_im, 0.0, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_float_closeness(w_re, FRAC_1_SQRT_2, 1e-10);
        assert_float_closeness(w_im, -FRAC_1_SQRT_2, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_float_closeness(w_re, 0.0, 1e-10);
        assert_float_closeness(w_im, -1.0, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_float_closeness(w_re, -FRAC_1_SQRT_2, 1e-10);
        assert_float_closeness(w_im, -FRAC_1_SQRT_2, 1e-10);
    }

    macro_rules! test_twiddles_simd {
        ($test_name:ident, $generate_twiddles_simd:ident, $epsilon:literal) => {
            #[test]
            fn $test_name() {
                for n in 4..25 {
                    let dist = 1 << n;

                    let (twiddles_re_ref, twiddles_im_ref) =
                        generate_twiddles(dist, Direction::Forward);
                    let (twiddles_re, twiddles_im) =
                        $generate_twiddles_simd(dist, Direction::Forward);

                    twiddles_re
                        .iter()
                        .zip(twiddles_re_ref.iter())
                        .for_each(|(simd, reference)| {
                            assert_float_closeness(*simd, *reference, $epsilon);
                        });

                    twiddles_im
                        .iter()
                        .zip(twiddles_im_ref.iter())
                        .for_each(|(simd, reference)| {
                            assert_float_closeness(*simd, *reference, $epsilon);
                        });
                }
            }
        };
    }

    test_twiddles_simd!(twiddles_simd_32, generate_twiddles_simd_32, 1e-1);
    test_twiddles_simd!(twiddles_simd_64, generate_twiddles_simd_64, 1e-10);

    #[test]
    fn twiddles_filter() {
        // Assume n = 28
        let n = 28;

        // distance := 2^{n} / 2 == 2^{n-1}
        let dist = 1 << (n - 1);

        let mut twiddles_iter = Twiddles::new(dist);

        let (twiddles_re, twiddles_im) = generate_twiddles(dist, Direction::Forward);

        for i in 0..dist {
            let (w_re, w_im) = twiddles_iter.next().unwrap();
            assert_float_closeness(twiddles_re[i], w_re, 1e-6);
            assert_float_closeness(twiddles_im[i], w_im, 1e-6);
        }

        let (mut tw_re, mut tw_im) = (twiddles_re.clone(), twiddles_im.clone());

        for t in (0..n - 1).rev() {
            let dist = 1 << t;
            let mut twiddles_iter = Twiddles::new(dist);

            // Don't re-compute all the twiddles.
            // Just filter them out by taking every other twiddle factor
            (tw_re, tw_im) = filter_twiddles(&tw_re, &tw_im);

            assert!(tw_re.len() == dist && tw_im.len() == dist);

            for i in 0..dist {
                let (w_re, w_im) = twiddles_iter.next().unwrap();
                assert_float_closeness(tw_re[i], w_re, 1e-6);
                assert_float_closeness(tw_im[i], w_im, 1e-6);
            }
        }
    }

    macro_rules! forward_mul_inverse_eq_identity {
        ($test_name:ident, $generate_twiddles_simd_fn:ident) => {
            #[test]
            fn $test_name() {
                for i in 3..25 {
                    let num_points = 1 << i;
                    let dist = num_points >> 1;

                    let (fwd_twiddles_re, fwd_twiddles_im) = if dist >= 8 * 2 {
                        $generate_twiddles_simd_fn(dist, Direction::Forward)
                    } else {
                        generate_twiddles(dist, Direction::Forward)
                    };

                    assert_eq!(fwd_twiddles_re.len(), fwd_twiddles_im.len());

                    let (rev_twiddles_re, rev_twiddles_im) = if dist >= 8 * 2 {
                        $generate_twiddles_simd_fn(dist, Direction::Reverse)
                    } else {
                        generate_twiddles(dist, Direction::Reverse)
                    };

                    assert_eq!(rev_twiddles_re.len(), rev_twiddles_im.len());

                    // (a + ib) (c + id) = ac + iad + ibc - bd
                    //                   = ac - bd + i(ad + bc)
                    fwd_twiddles_re
                        .iter()
                        .zip(fwd_twiddles_im.iter())
                        .zip(rev_twiddles_re.iter())
                        .zip(rev_twiddles_im.iter())
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

    forward_mul_inverse_eq_identity!(forward_reverse_eq_identity_64, generate_twiddles_simd_64);
    forward_mul_inverse_eq_identity!(forward_reverse_eq_identity_32, generate_twiddles_simd_32);
}
