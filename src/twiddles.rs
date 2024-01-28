use crate::kernels::Float;
use std::{f64::consts::PI, simd::f64x8};

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

pub(crate) fn generate_twiddles(dist: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(dist.is_power_of_two());
    const CHUNK_SIZE: usize = 8; // TODO: make this a const generic?
    let mut twiddles_re = vec![0.0; dist];
    let mut twiddles_im = vec![0.0; dist];
    twiddles_re[0] = 1.0;

    if dist == 1 {
        // To simplify logic later on
        return (twiddles_re, twiddles_im);
    }

    let angle = -PI / (dist as f64);
    let (st, ct) = angle.sin_cos();
    let (mut w_re, mut w_im) = (1.0, 0.0);
    // Split the twiddles into two halves. There is a cheaper way to calculate the second half
    let (first_half_re, second_half_re) = twiddles_re[1..].split_at_mut(dist / 2);
    assert!(first_half_re.len() == second_half_re.len() + 1);
    let (first_half_im, second_half_im) = twiddles_im[1..].split_at_mut(dist / 2);
    assert!(first_half_im.len() == second_half_im.len() + 1);

    first_half_re
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(first_half_im.chunks_exact_mut(CHUNK_SIZE))
        .zip(second_half_re.chunks_exact_mut(CHUNK_SIZE))
        .zip(second_half_im.chunks_exact_mut(CHUNK_SIZE))
        .for_each(
            |(((first_ch_re, first_ch_im), second_ch_re), second_ch_im)| {
                // Calculate a chunk of the first half in a plain old scalar way
                first_ch_re
                    .iter_mut()
                    .zip(first_ch_im.iter_mut())
                    .for_each(|(re, im)| {
                        let temp = w_re;
                        w_re = w_re * ct - w_im * st;
                        w_im = temp * st + w_im * ct;
                        *re = w_re;
                        *im = w_im;
                    });
                // Calculate a chunk of the second half in a clever way by copying the first chunk
                // This avoids data dependencies of the regular calculation and gets vectorized.
                // We do it up front while the values we just calculated are still in the cache
                // so we don't have to re-load them from memory later, which would be slow.
                let first_re = f64x8::from_slice(first_ch_re);
                let minus_one = f64x8::splat(-1.0);
                second_ch_re.copy_from_slice((first_re * minus_one).as_array());
                second_ch_im.copy_from_slice(&first_ch_im);
            },
        );

    // Handle remainder that the chunks did not process.
    //
    // Interesting edge cases:
    // We can have `first_half.len = 0 && second_half.len = 0` for `dist = 1`.
    // We can have `first_half.len = 1 && second_half.len = 0` for `dist = 2`.
    // We do not handle them here because the rest of the library never invokes those.
    assert!(dist >= 4);
    // We also cannot blindly rely on `remainder()` on the chunked iterators because
    // the first half is 1 longer than the second one and may indeed fit a chunk
    // while the second one does not, e.g. for `dist = 8` we will have
    // `first_half.len() == 4 && second_half.len == 3`
    // Therefore we rely on the remainder on the *shorter* chunk.
    let second_remainder_re = second_half_re.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let first_remainder_len = second_remainder_re.len() + 1;
    let first_rem_skip = first_half_re.len() - first_remainder_len;
    // get the last `first_remainder_len` elements of first_half_re
    let first_remainder_re = &mut first_half_re[first_rem_skip..];
    // repeat the same for the imaginary part
    let second_remainder_im = second_half_im.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let first_remainder_im = &mut first_half_im[first_rem_skip..];

    // Do the rest of the scalar calculation on the remainder of the first half.
    // We do NOT zip the halves together in this loop because the lengths are different,
    // and a loop that processes both would leave the last element of the first half unprocessed.
    // Also no point in doing that performance-wise.
    first_remainder_re
        .iter_mut()
        .zip(first_remainder_im.iter_mut())
        .for_each(|(re, im)| {
            let temp = w_re;
            w_re = w_re * ct - w_im * st;
            w_im = temp * st + w_im * ct;
            *re = w_re;
            *im = w_im;
        });
    // ...and the clever calculation on the remainder of the second half.
    first_remainder_re
        .iter()
        .zip(second_remainder_re.iter_mut())
        .for_each(|(first, second)| {
            *second = -first;
        });
    second_remainder_im.copy_from_slice(&first_remainder_im[..second_remainder_im.len()]);

    (twiddles_re, twiddles_im)
}

#[cfg(test)]
mod tests {
    use crate::utils::assert_f64_closeness;
    use std::f64::consts::FRAC_1_SQRT_2;

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
    fn test_twiddles() {
        let dist = 8;
        let (twiddles_im, twiddles_re) = generate_twiddles(dist);
        let mut tw_iter = Twiddles::new(8);

        for (w_re, w_im) in twiddles_re.iter().zip(twiddles_im.iter()) {
            let (z_re, z_im) = tw_iter.next().unwrap();
            eprintln!("actual re: {w_re} im: {w_im} --- expected: re: {z_re} im: {z_im}");
            assert_f64_closeness(*w_re, z_re, 1e-10);
            assert_f64_closeness(*w_im, z_im, 1e-10);
        }
    }
}
