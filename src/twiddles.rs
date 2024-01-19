use spinoza::math::{Float, PI};

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

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use spinoza::utils::assert_float_closeness;

    use super::*;

    #[test]
    fn twiddles_4() {
        const N: usize = 4;
        let mut twiddle_iter = Twiddles::new(N);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{} {}", w_re, w_im);
        assert_float_closeness(w_re, 1.0, 1e-10);
        assert_float_closeness(w_im, 0.0, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{} {}", w_re, w_im);
        assert_float_closeness(w_re, FRAC_1_SQRT_2, 1e-10);
        assert_float_closeness(w_im, -FRAC_1_SQRT_2, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{} {}", w_re, w_im);
        assert_float_closeness(w_re, 0.0, 1e-10);
        assert_float_closeness(w_im, -1.0, 1e-10);

        let (w_re, w_im) = twiddle_iter.next().unwrap();
        println!("{} {}", w_re, w_im);
        assert_float_closeness(w_re, -FRAC_1_SQRT_2, 1e-10);
        assert_float_closeness(w_im, -FRAC_1_SQRT_2, 1e-10);
    }
}