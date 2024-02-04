use crate::twiddles::{generate_twiddles, generate_twiddles_simd};

pub struct Planner {
    pub twiddles_re: Vec<f64>,
    pub twiddles_im: Vec<f64>,
}

impl Planner {
    /// Create a `Planner` for an FFT of size `num_points`
    ///
    /// # Panics
    ///
    /// Panics if `num_points` is less than 1
    pub fn new(num_points: usize) -> Self {
        assert!(num_points > 0);
        if num_points <= 4 {
            return Self {
                twiddles_re: vec![],
                twiddles_im: vec![],
            };
        }

        let dist = num_points >> 1;
        let (twiddles_re, twiddles_im) = if dist >= 8 * 2 {
            generate_twiddles_simd(dist)
        } else {
            generate_twiddles(dist)
        };

        assert_eq!(twiddles_re.len(), twiddles_im.len());

        Self {
            twiddles_re,
            twiddles_im,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::planner::Planner;

    #[test]
    fn no_twiddles() {
        for num_points in 2..=4 {
            let planner = Planner::new(num_points);
            assert!(planner.twiddles_im.is_empty() && planner.twiddles_re.is_empty());
        }
    }
}
