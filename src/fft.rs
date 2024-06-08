//! Implementation of Real valued FFT
use crate::{fft_32, fft_64, twiddles::generate_twiddles, Direction};

#[macro_export]
macro_rules! impl_r2c_fft {
    ($func_name:ident, $precision:ty, $fft_func:ident) => {
        /// Implementation of Real-Valued FFT
        ///
        /// # Panics
        ///
        /// Panics if `output_re.len() != output_im.len()` and `input_re.len()` == `output_re.len()`
        pub fn $func_name(
            input_re: &[$precision],
            output_re: &mut [$precision],
            output_im: &mut [$precision],
        ) {
            assert!(output_re.len() == output_im.len() && input_re.len() == output_re.len());
            let big_n = input_re.len();

            // Splitting odd and even
            let (mut z_even, mut z_odd): (Vec<_>, Vec<_>) =
                input_re.chunks_exact(2).map(|c| (c[0], c[1])).unzip();

            // Z = np.fft.fft(z)
            $fft_func(&mut z_even, &mut z_odd, Direction::Forward);

            let mut z_x_re = Vec::with_capacity(z_even.len());
            let mut z_x_im = Vec::with_capacity(z_even.len());
            let mut z_y_re = Vec::with_capacity(z_even.len());
            let mut z_y_im = Vec::with_capacity(z_even.len());

            z_x_re.push(0.0);
            z_x_im.push(0.0);
            z_y_re.push(0.0);
            z_y_im.push(0.0);

            // Zminconj = np.roll(np.flip(Z), 1).conj()
            // Zx =  0.5  * (Z + Zminconj)
            // Zy = -0.5j * (Z - Zminconj)
            z_even
                .iter()
                .skip(1)
                .zip(z_odd.iter().skip(1))
                .zip(z_even.iter().skip(1).rev())
                .zip(z_odd.iter().skip(1).rev())
                .for_each(|(((z_e, z_o), z_e_mc), z_o_mc)| {
                    let a = *z_e;
                    let b = *z_o;
                    let c = *z_e_mc;
                    let d = -(*z_o_mc);

                    let t = 0.5 * (a + c);
                    let u = 0.5 * (b + d);
                    let v = -0.5 * (a - c);
                    let w = 0.5 * (b - d);

                    z_x_re.push(t);
                    z_x_im.push(u);
                    z_y_re.push(w);
                    z_y_im.push(v);
                });

            let a = z_even[0];
            let b = z_odd[0];
            let c = z_even[0];
            let d = -z_odd[0];

            let t = 0.5 * (a + c);
            let u = 0.5 * (b + d);
            let v = -0.5 * (a - c);
            let w = 0.5 * (b - d);

            z_x_re[0] = t;
            z_x_im[0] = u;
            z_y_re[0] = w;
            z_y_im[0] = v;

            let (twiddle_re, twiddle_im): (Vec<$precision>, Vec<$precision>) =
                generate_twiddles(big_n / 2, Direction::Forward);

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
    };
}

impl_r2c_fft!(r2c_fft_f32, f32, fft_32);
impl_r2c_fft!(r2c_fft_f64, f64, fft_64);

#[cfg(test)]
mod tests {
    use utilities::assert_float_closeness;

    use super::*;

    macro_rules! impl_r2c_vs_c2c_test {
        ($func:ident, $precision:ty, $fft_precision:ident, $rfft_func:ident, $epsilon:literal) => {
            #[test]
            fn $func() {
                for n in 4..=11 {
                    let big_n = 1 << n;
                    let input_re: Vec<$precision> = (1..=big_n).map(|i| i as $precision).collect(); // Length is 7, which is a prime number
                    let (mut output_re, mut output_im) = (vec![0.0; big_n], vec![0.0; big_n]);

                    $rfft_func(&input_re, &mut output_re, &mut output_im);

                    let mut input_re: Vec<$precision> =
                        (1..=big_n).map(|i| i as $precision).collect();
                    let mut input_im = vec![0.0; input_re.len()]; // Assume the imaginary part is zero for this example
                    $fft_precision(&mut input_re, &mut input_im, Direction::Forward);

                    output_re
                        .iter()
                        .zip(output_im.iter())
                        .zip(input_re.iter())
                        .zip(input_im.iter())
                        .for_each(|(((a_re, a_im), e_re), e_im)| {
                            assert_float_closeness(*a_re, *e_re, $epsilon);
                            assert_float_closeness(*a_im, *e_im, $epsilon);
                        });
                }
            }
        };
    }

    impl_r2c_vs_c2c_test!(r2c_vs_c2c_f64, f64, fft_64, r2c_fft_f64, 1e-6);
    impl_r2c_vs_c2c_test!(r2c_vs_c2c_f32, f32, fft_32, r2c_fft_f32, 3.5);
}
