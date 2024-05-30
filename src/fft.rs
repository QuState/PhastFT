//! Implementation of Real valued FFT
use num_traits::Float;

use crate::Direction;
use crate::fft_64;

fn compute_twiddle_factors<T: Float>(big_n: usize) -> (Vec<T>, Vec<T>) {
    let half_n = big_n / 2;
    let mut real_parts = Vec::with_capacity(half_n);
    let mut imag_parts = Vec::with_capacity(half_n);
    let two: T = T::from(2.0).unwrap();
    let pi: T = T::from(std::f64::consts::PI).unwrap();

    for k in 0..half_n {
        let angle = -two * pi * T::from(k).unwrap() / T::from(big_n).unwrap();
        real_parts.push(angle.cos());
        imag_parts.push(angle.sin());
    }

    (real_parts, imag_parts)
}

/// Implementation of Real-Valued FFT
///
/// # Panics
///
/// Panics if `output_re.len() != output_im.len()` and `input_re.len()` == `output_re.len()`
pub fn r2c_fft_f64(input_re: &[f64], output_re: &mut [f64], output_im: &mut [f64]) {
    assert!(output_re.len() == output_im.len() && input_re.len() == output_re.len());
    let big_n = input_re.len();

    // Splitting odd and even
    let (mut z_even, mut z_odd): (Vec<_>, Vec<_>) =
        input_re.chunks_exact(2).map(|c| (c[0], c[1])).unzip();

    // Z = np.fft.fft(z)
    fft_64(&mut z_even, &mut z_odd, Direction::Forward);

    // take care of the np.flip()
    let mut z_even_min_conj: Vec<_> = z_even.iter().copied().rev().collect();
    let mut z_odd_min_conj: Vec<_> = z_odd.iter().copied().rev().collect();

    // Zminconj = np.roll(np.flip(Z), 1).conj()
    // now roll both by 1
    z_even_min_conj.rotate_right(1);
    z_odd_min_conj.rotate_right(1);

    // the conj() call can be resolved by negating every imaginary component
    for zo in z_odd_min_conj.iter_mut() {
        *zo = -(*zo);
    }

    // Zx = 0.5  * (Z + Zminconj)
    let (z_x_re, z_x_im): (Vec<_>, Vec<_>) = z_even
        .iter()
        .zip(z_odd.iter())
        .zip(z_even_min_conj.iter())
        .zip(z_odd_min_conj.iter())
        .map(|(((ze, zo), ze_mc), zo_mc)| {
            let a = 0.5 * (ze + ze_mc);
            let b = 0.5 * (zo + zo_mc);
            (a, b)
        })
        .unzip();

    // Zy = -0.5j * (Z - Zminconj)
    let (z_y_re, z_y_im): (Vec<_>, Vec<_>) = z_even
        .iter()
        .zip(z_odd.iter())
        .zip(z_even_min_conj.iter())
        .zip(z_odd_min_conj.iter())
        .map(|(((ze, zo), ze_mc), zo_mc)| {
            let a = ze - ze_mc;
            let b = zo - zo_mc;

            // -0.5i (a + ib) = -0.5i * a - 0.5i * ib
            //                = -0.5i * a + 0.5 * b
            //                = 0.5 * b - 0.5 * ia
            (0.5 * b, -0.5 * a)
        })
        .unzip();

    let (twiddle_re, twiddle_im): (Vec<f64>, Vec<f64>) = compute_twiddle_factors(big_n);

    // Zall = np.concatenate([Zx + W*Zy, Zx - W*Zy])

    for i in 0..big_n / 2 {
        let zx_re = z_x_re[i];
        let zx_im = z_x_im[i];
        let zy_re = z_y_re[i];
        let zy_im = z_y_im[i];
        let w_re = twiddle_re[i];
        let w_im = twiddle_im[i];

        let wz_re = w_re * zy_re - w_im * zy_im;
        let wz_im = w_re * zy_im + w_im * zy_re;

        // Zx + W * Zy
        output_re[i] = zx_re + wz_re;
        output_im[i] = zx_im + wz_im;
    }

    for i in 0..big_n / 2 {
        let zx_re = z_x_re[i];
        let zx_im = z_x_im[i];
        let zy_re = z_y_re[i];
        let zy_im = z_y_im[i];
        let w_re = twiddle_re[i];
        let w_im = twiddle_im[i];

        let wz_re = w_re * zy_re - w_im * zy_im;
        let wz_im = w_re * zy_im + w_im * zy_re;

        // Zx - W * Zy
        output_re[i + big_n / 2] = zx_re - wz_re;
        output_im[i + big_n / 2] = zx_im - wz_im;
    }
}

#[cfg(test)]
mod tests {
    use utilities::assert_float_closeness;

    use super::*;

    macro_rules! impl_r2c_vs_c2c_test {
        ($func:ident, $precision:ty, $fft_precision:ident, $rftt_funct:ident) => {
            #[test]
            fn $func() {
                for n in 4..=11 {
                    let big_n = 1 << n;
                    let input_re: Vec<$precision> = (1..=big_n).map(|i| i as $precision).collect(); // Length is 7, which is a prime number
                    let (mut output_re, mut output_im) = (vec![0.0; big_n], vec![0.0; big_n]);

                    $rftt_funct(&input_re, &mut output_re, &mut output_im);

                    let mut input_re: Vec<_> = (1..=big_n).map(|i| i as $precision).collect();
                    let mut input_im = vec![0.0; input_re.len()]; // Assume the imaginary part is zero for this example
                    $fft_precision(&mut input_re, &mut input_im, Direction::Forward);

                    output_re
                        .iter()
                        .zip(output_im.iter())
                        .zip(input_re.iter())
                        .zip(input_im.iter())
                        .for_each(|(((a_re, a_im), e_re), e_im)| {
                            assert_float_closeness(*a_re, *e_re, 1e-6);
                            assert_float_closeness(*a_im, *e_im, 1e-6);
                        });
                }
            }
        };
    }

    impl_r2c_vs_c2c_test!(r2c_vs_c2c_f64, f64, fft_64, r2c_fft_f64);
    // impl_r2c_vs_c2c_test!(r2c_vs_c2c_f32, fft_32, r2c_fft_f32);
}
