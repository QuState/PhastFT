//! Implementation of Real valued FFT
use std::simd::prelude::f64x8;

use crate::planner::{Planner32, Planner64};
use crate::{
    fft_32_with_opts_and_plan, fft_64, fft_64_with_opts_and_plan,
    twiddles::{generate_twiddles, Twiddles},
    Direction, Options,
};

macro_rules! impl_r2c_fft {
    ($func_name:ident, $precision:ty, $planner:ident, $fft_w_opts_and_plan:ident) => {
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

            // let mut planner = <$planner>::new(big_n, Direction::Forward);

            // save these for the untanngling step
            // let twiddle_re = planner.twiddles_re;
            // let twiddle_im = planner.twiddles_im;
            let stride = big_n / 2;
            let twiddles_iter = Twiddles::<$precision>::new(stride);

            let planner = <$planner>::new(big_n / 2, Direction::Forward);

            // We only need (N / 2) / 2 twiddle factors for the actual FFT call, so we filter
            // filter_twiddles(&mut planner.twiddles_re, &mut planner.twiddles_im);

            let opts = Options::guess_options(z_even.len());
            $fft_w_opts_and_plan(&mut z_even, &mut z_odd, &opts, &planner);

            // Z = np.fft.fft(z)
            let mut z_x_re = vec![0.0; big_n / 2];
            let mut z_x_im = vec![0.0; big_n / 2];
            let mut z_y_re = vec![0.0; big_n / 2];
            let mut z_y_im = vec![0.0; big_n / 2];

            // Zminconj = np.roll(np.flip(Z), 1).conj()
            // Zx =  0.5  * (Z + Zminconj)
            // Zy = -0.5j * (Z - Zminconj)
            z_even[1..]
                .iter()
                .zip(z_odd[1..].iter())
                .zip(z_even[1..].iter().rev())
                .zip(z_odd[1..].iter().rev())
                .zip(z_x_re[1..].iter_mut())
                .zip(z_x_im[1..].iter_mut())
                .zip(z_y_re[1..].iter_mut())
                .zip(z_y_im[1..].iter_mut())
                .for_each(
                    |(((((((z_e, z_o), z_e_mc), z_o_mc), zx_re), zx_im), zy_re), zy_im)| {
                        let a = *z_e;
                        let b = *z_o;
                        let c = *z_e_mc;
                        let d = -(*z_o_mc);

                        let t = 0.5 * (a + c);
                        let u = 0.5 * (b + d);
                        let v = -0.5 * (a - c);
                        let w = 0.5 * (b - d);

                        *zx_re = t;
                        *zx_im = u;
                        *zy_re = w;
                        *zy_im = v;
                    },
                );

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

            // Zall = np.concatenate([Zx + W*Zy, Zx - W*Zy])
            let (output_re_first_half, output_re_second_half) = output_re.split_at_mut(big_n / 2);
            let (output_im_first_half, output_im_second_half) = output_im.split_at_mut(big_n / 2);

            z_x_re
                .iter()
                .zip(z_x_im.iter())
                .zip(z_y_re.iter())
                .zip(z_y_im.iter())
                .zip(output_re_first_half)
                .zip(output_im_first_half)
                .zip(output_re_second_half)
                .zip(output_im_second_half)
                .zip(twiddles_iter)
                .for_each(
                    |(
                        (
                            ((((((zx_re, zx_im), zy_re), zy_im), o_re_fh), o_im_fh), o_re_sh),
                            o_im_sh,
                        ),
                        (w_re, w_im),
                    )| {
                        let wz_re = w_re * zy_re - w_im * zy_im;
                        let wz_im = w_re * zy_im + w_im * zy_re;

                        // Zx + W * Zy
                        *o_re_fh = zx_re + wz_re;
                        *o_im_fh = zx_im + wz_im;

                        // Zx - W * Zy
                        *o_re_sh = zx_re - wz_re;
                        *o_im_sh = zx_im - wz_im;
                    },
                );
        }
    };
}

impl_r2c_fft!(r2c_fft_f32, f32, Planner32, fft_32_with_opts_and_plan);
impl_r2c_fft!(r2c_fft_f64, f64, Planner64, fft_64_with_opts_and_plan);

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

    use crate::fft_32;

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
