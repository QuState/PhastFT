use crate::twiddles::{generate_twiddles, generate_twiddles_simd};

pub enum Direction {
    Forward = 1,
    Reverse = -1,
}

pub struct Planner {
    pub twiddles_re: Vec<f64>,
    pub twiddles_im: Vec<f64>,
}

// TODO(saveliy yusufov): Add a parameter to `new` that will take into consideration whether we do inverse FFT (IFFT)
// In this case, the twiddle factors should be pre-computed as follows:
//
//                  FFT Twiddle Factor: e^{i2π*k/N}
//                  IFFT Twiddle Factor: e^{-i2π*k/N}
//
// source: https://dsp.stackexchange.com/q/73367
impl Planner {
    /// Create a `Planner` for an FFT of size `num_points`
    ///
    /// # Panics
    ///
    /// Panics if `num_points` is less than 1
    pub fn new(num_points: usize, direction: Direction) -> Self {
        assert!(num_points > 0);
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
}

#[cfg(test)]
mod tests {
    use crate::planner::{Direction, Planner};

    #[test]
    fn no_twiddles() {
        for num_points in 2..=4 {
            let planner = Planner::new(num_points, Direction::Forward);
            assert!(planner.twiddles_im.is_empty() && planner.twiddles_re.is_empty());
        }
    }
}
