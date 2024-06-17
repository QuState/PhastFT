//! Implementation of Real valued FFT
use std::simd::prelude::f64x8;

use crate::{fft_32, fft_64, twiddles::generate_twiddles, Direction};

macro_rules! impl_r2c_fft {
    ($func_name:ident, $precision:ty, $fft_func:ident) => {
        /// Performs a Real-Valued Fast Fourier Transform (FFT)
        ///
        /// This function computes the FFT of a real-valued input signal and produces
        /// complex-valued output. The implementation follows the principles of splitting
        /// the input into even and odd components and then performing the FFT on these
        /// components.
        ///
        /// # Arguments
        ///
        /// `input_re` - A slice containing the real-valued input signal.
        ///
        /// `output_re` - A mutable slice to store the real parts of the FFT output.
        ///
        /// `output_im` - A mutable slice to store the imaginary parts of the FFT output.
        ///
        /// # Panics
        ///
        /// Panics if `output_re.len() != output_im.len()` and `input_re.len()` == `output_re.len()`
        ///
        /// # Examples
        ///
        /// ```
        /// use phastft::fft::{r2c_fft_f32, r2c_fft_f64};
        ///
        /// let big_n = 16;
        /// let input: Vec<f64> = (1..=big_n).map(|x| x as f64).collect();
        /// let mut output_re = vec![0.0; big_n];
        /// let mut output_im = vec![0.0; big_n];
        /// r2c_fft_f64(&input, &mut output_re, &mut output_im);
        ///
        /// let input: Vec<f32> = (1..=big_n).map(|x| x as f32).collect();
        /// let mut output_re: Vec<f32> = vec![0.0; big_n];
        /// let mut output_im: Vec<f32> = vec![0.0; big_n];
        /// r2c_fft_f32(&input, &mut output_re, &mut output_im);
        /// ```
        /// # References
        ///
        /// This implementation is based on the concepts discussed in
        /// [Levente Kovács' post](https://kovleventer.com/blog/fft_real/).
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

            // let mut z_x_re = vec![0.0; big_n / 2];
            // let mut z_x_im = vec![0.0; big_n / 2];
            // let mut z_y_re = vec![0.0; big_n / 2];
            // let mut z_y_im = vec![0.0; big_n / 2];

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

            z_x_re
                .iter()
                .zip(z_x_im.iter())
                .zip(z_y_re.iter())
                .zip(z_y_im.iter())
                .zip(twiddle_re.iter())
                .zip(twiddle_im.iter())
                .zip(output_re[..big_n / 2].iter_mut())
                .zip(output_im[..big_n / 2].iter_mut())
                .for_each(
                    |(((((((zx_re, zx_im), zy_re), zy_im), w_re), w_im), o_re), o_im)| {
                        let wz_re = w_re * zy_re - w_im * zy_im;
                        let wz_im = w_re * zy_im + w_im * zy_re;

                        // Zx + W * Zy
                        *o_re = zx_re + wz_re;
                        *o_im = zx_im + wz_im;
                    },
                );

            z_x_re
                .iter()
                .zip(z_x_im.iter())
                .zip(z_y_re.iter())
                .zip(z_y_im.iter())
                .zip(twiddle_re.iter())
                .zip(twiddle_im.iter())
                .zip(output_re[big_n / 2..].iter_mut())
                .zip(output_im[big_n / 2..].iter_mut())
                .for_each(
                    |(((((((zx_re, zx_im), zy_re), zy_im), w_re), w_im), o_re), o_im)| {
                        let wz_re = w_re * zy_re - w_im * zy_im;
                        let wz_im = w_re * zy_im + w_im * zy_re;

                        // Zx + W * Zy
                        *o_re = zx_re - wz_re;
                        *o_im = zx_im - wz_im;
                    },
                );
        }
    };
}

impl_r2c_fft!(r2c_fft_f32, f32, fft_32);
impl_r2c_fft!(r2c_fft_f64, f64, fft_64);

/// Performs a Real-Valued, Inverse, Fast Fourier Transform (FFT)
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()` *and* `reals.len() != output.len()`
///
pub fn r2c_ifft_f64(reals: &mut [f64], imags: &mut [f64], output: &mut [f64]) {
    assert!(reals.len() == imags.len() && reals.len() == output.len());

    let big_n = reals.len();

    let mut z_x_re = vec![0.0; big_n / 2];
    let mut z_x_im = vec![0.0; big_n / 2];

    let (reals_first_half, reals_second_half) = reals.split_at(big_n / 2);
    let (imags_first_half, imags_second_half) = imags.split_at(big_n / 2);

    // Compute Zx
    for i in 0..big_n / 2 {
        let re = 0.5 * (reals_first_half[i] + reals_second_half[i]);
        let im = 0.5 * (imags_first_half[i] + imags_second_half[i]);

        z_x_re[i] = re;
        z_x_im[i] = im;
    }

    let mut z_y_re = vec![0.0; big_n / 2];
    let mut z_y_im = vec![0.0; big_n / 2];
    let (twiddles_re, twiddles_im) = generate_twiddles::<f64>(big_n / 2, Direction::Reverse);

    // Compute W * Zy
    for i in 0..big_n / 2 {
        let a = reals_first_half[i];
        let b = imags_first_half[i];
        let c = reals_second_half[i];
        let d = imags_second_half[i];

        let e = a - c;
        let f = b - d;
        let k = -0.5 * f;
        let l = 0.5 * e;

        let m = twiddles_re[i];
        let n = twiddles_im[i];

        z_y_re[i] = k * m - l * n;
        z_y_im[i] = k * n + l * m;
    }

    const CHUNK_SIZE: usize = 8;

    // Compute Zx + Zy
    z_x_re
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(z_x_im.chunks_exact_mut(CHUNK_SIZE))
        .zip(z_y_re.chunks_exact(CHUNK_SIZE))
        .zip(z_y_im.chunks_exact(CHUNK_SIZE))
        .for_each(|(((zxr, zxi), zyr), zyi)| {
            let mut z_x_r = f64x8::from_slice(zxr);
            let mut z_x_i = f64x8::from_slice(zxi);
            let z_y_r = f64x8::from_slice(zyr);
            let z_y_i = f64x8::from_slice(zyi);

            z_x_r += z_y_r;
            z_x_i += z_y_i;
            zxr.copy_from_slice(z_x_r.as_array());
            zxi.copy_from_slice(z_x_i.as_array());
        });

    fft_64(&mut z_x_re, &mut z_x_im, Direction::Reverse);

    // Store reals in the even indices, and imaginaries in the odd indices
    output
        .chunks_exact_mut(2)
        .zip(z_x_re)
        .zip(z_x_im)
        .for_each(|((out, z_r), z_i)| {
            out[0] = z_r;
            out[1] = z_i;
        });
}

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

    #[test]
    fn fw_inv_eq_identity() {
        let n = 4;
        let big_n = 1 << n;
        let expected_signal: Vec<f64> = (1..=big_n).map(|s| s as f64).collect();

        let mut output_re = vec![0.0; big_n];
        let mut output_im = vec![0.0; big_n];
        r2c_fft_f64(&expected_signal, &mut output_re, &mut output_im);

        let mut actual_signal = vec![0.0; big_n];
        r2c_ifft_f64(&mut output_re, &mut output_im, &mut actual_signal);

        for (z_a, z_e) in actual_signal.iter().zip(expected_signal.iter()) {
            assert_float_closeness(*z_a, *z_e, 1e-10);
        }

        println!(
            "expected signal: {:?}\nactual signal: {:?}",
            expected_signal, actual_signal
        );
    }
}
