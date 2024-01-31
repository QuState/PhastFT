use std::{f64::consts::PI, simd::f64x8};

use crate::kernels::Float;

pub(crate) struct Twiddles {
    st: Float,
    ct: Float,
    w_re_prev: Float,
    w_im_prev: Float,
}

impl Twiddles {
    /// `cache_size` is the amount of roots of unity kept pre-built at any point in time.
    /// `num_roots` is the total number of roots of unity that will need to be computed.
    /// `cache_size` can be thought of as the length of a chunk of roots of unity from
    /// out of the total amount (i.e., `num_roots`)
    #[allow(dead_code)]
    pub fn new(num_roots: usize) -> Self {
        let theta = -PI / (num_roots as Float);
        let (st, ct) = theta.sin_cos();
        Self {
            st,
            ct,
            w_re_prev: 1.0,
            w_im_prev: 0.0,
        }
    }
}

impl Iterator for Twiddles {
    type Item = (Float, Float);

    fn next(&mut self) -> Option<(f64, f64)> {
        let w_re = self.w_re_prev;
        let w_im = self.w_im_prev;

        let temp = self.w_re_prev;
        self.w_re_prev = temp * self.ct - self.w_im_prev * self.st;
        self.w_im_prev = temp * self.st + self.w_im_prev * self.ct;

        Some((w_re, w_im))
    }
}

pub fn generate_twiddles(dist: usize) -> (Vec<f64>, Vec<f64>) {
    let mut twiddles_re = vec![0.0; dist];
    let mut twiddles_im = vec![0.0; dist];
    twiddles_re[0] = 1.0;

    let angle = -PI / (dist as f64);
    let (st, ct) = angle.sin_cos();
    let (mut w_re, mut w_im) = (1.0, 0.0);

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

pub(crate) fn generate_twiddles_simd(dist: usize) -> (Vec<f64>, Vec<f64>) {
    const SIMD_WIDTH: usize = 8; // TODO: make this a const generic?
    assert!(dist >= SIMD_WIDTH * 2);
    assert_eq!(dist % SIMD_WIDTH, 0);
    let mut twiddles_re = vec![0.0; dist];
    let mut twiddles_im = vec![0.0; dist];
    twiddles_re[0] = 1.0;

    let angle = -PI / (dist as f64);
    let (st, ct) = angle.sin_cos();
    let (mut w_re, mut w_im) = (1.0, 0.0);

    let mut next_twiddle = || {
        let temp = w_re;
        w_re = w_re * ct - w_im * st;
        w_im = temp * st + w_im * ct;
        (w_re, w_im)
    };

    let apply_symmetry_re = |input: &[Float], output: &mut [Float]| {
        let first_re = f64x8::from_slice(input);
        let minus_one = f64x8::splat(-1.0);
        let negated = (first_re * minus_one).reverse();
        output.copy_from_slice(negated.as_array());
    };

    let apply_symmetry_im = |input: &[Float], output: &mut [Float]| {
        let mut buf: [Float; SIMD_WIDTH] = [Float::default(); SIMD_WIDTH];
        buf.copy_from_slice(input);
        buf.reverse();
        output.copy_from_slice(&buf);
    };

    // Split the twiddles into two halves. There is a cheaper way to calculate the second half
    let (first_half_re, second_half_re) = twiddles_re[1..].split_at_mut(dist / 2);
    assert!(first_half_re.len() == second_half_re.len() + 1);
    let (first_half_im, second_half_im) = twiddles_im[1..].split_at_mut(dist / 2);
    assert!(first_half_im.len() == second_half_im.len() + 1);

    first_half_re
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(first_half_im.chunks_exact_mut(SIMD_WIDTH))
        .zip(
            second_half_re[SIMD_WIDTH - 1..]
                .chunks_exact_mut(SIMD_WIDTH)
                .rev(),
        )
        .zip(
            second_half_im[SIMD_WIDTH - 1..]
                .chunks_exact_mut(SIMD_WIDTH)
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
                // We do it up front while the values we just calculated are still in the cache
                // so we don't have to re-load them from memory later, which would be slow.
                apply_symmetry_re(&first_ch_re, second_ch_re);
                apply_symmetry_im(&first_ch_im, second_ch_im);
            },
        );

    // Fill in the middle that the SIMD loop didn't
    twiddles_re[dist / 2 - SIMD_WIDTH + 1..][..(SIMD_WIDTH * 2) - 1]
        .iter_mut()
        .zip(twiddles_im[dist / 2 - SIMD_WIDTH + 1..][..(SIMD_WIDTH * 2) - 1].iter_mut())
        .for_each(|(re, im)| {
            (*re, *im) = next_twiddle();
        });

    (twiddles_re, twiddles_im)
}

pub(crate) fn filter_twiddles(twiddles_re: &mut Vec<f64>, twiddles_im: &mut Vec<f64>) {
    assert_eq!(twiddles_re.len(), twiddles_im.len());
    let dist = twiddles_re.len();

    let filtered_twiddles_re: Vec<f64> =
        twiddles_re.chunks_exact(2).map(|chunk| chunk[0]).collect();
    let filtered_twiddles_im: Vec<f64> =
        twiddles_im.chunks_exact(2).map(|chunk| chunk[0]).collect();

    assert!(
        filtered_twiddles_re.len() == filtered_twiddles_im.len()
            && filtered_twiddles_re.len() == dist / 2
    );

    let _ = std::mem::replace(twiddles_re, filtered_twiddles_re);
    let _ = std::mem::replace(twiddles_im, filtered_twiddles_im);
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use crate::utils::assert_f64_closeness;

    use super::*;

    #[test]
    fn twiddles_4() {
        const N: usize = 4;
        let mut twiddle_iter = Twiddles::new(N);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_f64_closeness(w_re, 1.0, 1e-10);
        assert_f64_closeness(w_im, 0.0, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_f64_closeness(w_re, FRAC_1_SQRT_2, 1e-10);
        assert_f64_closeness(w_im, -FRAC_1_SQRT_2, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_f64_closeness(w_re, 0.0, 1e-10);
        assert_f64_closeness(w_im, -1.0, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{w_re} {w_im}");
        assert_f64_closeness(w_re, -FRAC_1_SQRT_2, 1e-10);
        assert_f64_closeness(w_im, -FRAC_1_SQRT_2, 1e-10);
    }

    #[test]
    fn twiddles_simd() {
        for n in 4..28 {
            let dist = 1 << n;

            let (mut twiddles_re_ref, mut twiddles_im_ref) = generate_twiddles(dist);
            let (mut twiddles_re, mut twiddles_im) = generate_twiddles_simd(dist);

            twiddles_re
                .iter()
                .zip(twiddles_re_ref.iter())
                .for_each(|(simd, reference)| {
                    assert_f64_closeness(*simd, *reference, 1e-10);
                });

            twiddles_im
                .iter()
                .zip(twiddles_im_ref.iter())
                .for_each(|(simd, reference)| {
                    assert_f64_closeness(*simd, *reference, 1e-10);
                });
        }
    }

    #[test]
    fn twiddles_filter() {
        let n = 28;

        let dist = 1 << (n - 1);
        let mut twiddles_iter = Twiddles::new(dist);

        let (mut twiddles_re, mut twiddles_im) = generate_twiddles(dist);

        for i in 0..dist {
            let (tw_re, tw_im) = twiddles_iter.next().unwrap();
            assert_f64_closeness(twiddles_re[i], tw_re, 1e-10);
            assert_f64_closeness(twiddles_im[i], tw_im, 1e-10);
        }

        for t in (0..n - 1).rev() {
            let dist = 1 << t;
            let mut twiddles_iter = Twiddles::new(dist);

            // Don't re-compute all the twiddles.
            // Just filter them out by taking every other twiddle factor
            filter_twiddles(&mut twiddles_re, &mut twiddles_im);

            assert!(twiddles_re.len() == dist && twiddles_im.len() == dist);

            for i in 0..dist {
                let (tw_re, tw_im) = twiddles_iter.next().unwrap();
                // eprintln!("actual: {} expected: {}", tw_re, twiddles_re[i]);
                assert_f64_closeness(twiddles_re[i], tw_re, 1e-6);
                assert_f64_closeness(twiddles_im[i], tw_im, 1e-6);
            }
        }
    }
}
