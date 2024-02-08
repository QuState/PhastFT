use crate::twiddles::{generate_twiddles, generate_twiddles_simd};

pub enum Direction {
    Forward = 1,
    Reverse = -1,
}

pub struct Planner {
    pub twiddles_re: Vec<f64>,
    pub twiddles_im: Vec<f64>,
}

impl Planner {
    /// Create a `Planner` for an FFT of size `num_points`.
    /// The twiddle factors are pre-computed based on the provided [`Direction`].
    /// For `Forward`, use [`Direction::Forward`].
    /// For `Reverse`, use [`Direction::Reverse`].
    ///
    /// # Panics
    ///
    /// Panics if `num_points` is less than 1
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
            generate_twiddles_simd(dist, direction)
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

#[cfg(test)]
mod tests {
    use utilities::assert_f64_closeness;

    use crate::planner::{Direction, Planner};

    #[test]
    fn no_twiddles() {
        for num_points in [2, 4] {
            let planner = Planner::new(num_points, Direction::Forward);
            assert!(planner.twiddles_im.is_empty() && planner.twiddles_re.is_empty());
        }
    }

    #[test]
    fn forward_mul_inverse_eq_identity() {
        for i in 3..25 {
            let num_points = 1 << i;
            let planner_forward = Planner::new(num_points, Direction::Forward);
            let planner_reverse = Planner::new(num_points, Direction::Reverse);

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
                    assert_f64_closeness(temp_re, 1.0, 1e-6);
                    assert_f64_closeness(temp_im, 0.0, 1e-6);
                });
        }
    }
}
