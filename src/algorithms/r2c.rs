//! Real-to-Complex FFT (R2C) and Complex-to-Real IFFT (C2R)
//!
//! Computes the FFT of a purely real-valued input signal using the
//! "pack-into-half-length-complex" trick:
//!
//! 1. Deinterleave N real values into even/odd halves (N/2 each)
//! 2. Treat (even, odd) as a complex signal and run a complex FFT of size N/2
//! 3. "Untangle" the result with a twiddle-factor post-processing step
//!    to recover the full N-point spectrum
//!
//! This halves the work compared to calling the complex FFT with a zeroed
//! imaginary array.
//!
//! # References
//!
//! Based on the approach described by Levente Kovács:
//! <https://kovleventer.com/blog/fft_real/>

use crate::algorithms::dit::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};
use crate::options::Options;
use crate::planner::{Direction, PlannerR2c32, PlannerR2c64};

/// Stride-2 deinterleave: `[a, b, c, d, ...]` → `([a, c, ...], [b, d, ...])`
#[inline]
fn deinterleave<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}

// Core untangle step
macro_rules! impl_untangle {
    ($func_name:ident, $precision:ty) => {
        fn $func_name(
            z_re: &[$precision],
            z_im: &[$precision],
            w_re: &[$precision],
            w_im: &[$precision],
            output_re: &mut [$precision],
            output_im: &mut [$precision],
        ) {
            let half = z_re.len();

            let (out_re_first, out_re_second) = output_re.split_at_mut(half);
            let (out_im_first, out_im_second) = output_im.split_at_mut(half);

            // DC component (k=0): mirror wraps to 0
            out_re_first[0] = z_re[0] + z_im[0];
            out_im_first[0] = 0.0;
            out_re_second[0] = z_re[0] - z_im[0];
            out_im_second[0] = 0.0;

            // k = 1 .. N/2 - 1
            for k in 1..half {
                let mirror = half - k;

                let a = z_re[k];
                let b = z_im[k];
                let c = z_re[mirror];
                let d_neg = -z_im[mirror]; // conjugate flips sign

                // Zx[k] = 0.5 * (Z[k] + Z*[mirror])
                let zx_re = 0.5 * (a + c);
                let zx_im = 0.5 * (b + d_neg);

                // Zy[k] = -0.5j * (Z[k] - Z*[mirror])
                //       = 0.5 * (Im_diff, -Re_diff)
                let zy_re = 0.5 * (b - d_neg);
                let zy_im = -0.5 * (a - c);

                // W^k · Zy[k]
                let wz_re = w_re[k] * zy_re - w_im[k] * zy_im;
                let wz_im = w_re[k] * zy_im + w_im[k] * zy_re;

                out_re_first[k] = zx_re + wz_re;
                out_im_first[k] = zx_im + wz_im;
                out_re_second[k] = zx_re - wz_re;
                out_im_second[k] = zx_im - wz_im;
            }
        }
    };
}

impl_untangle!(untangle_f64, f64);
impl_untangle!(untangle_f32, f32);

/// Performs a real-valued FFT on `f64` input data.
///
/// Computes the FFT of `input_re` (a real-valued signal of length N) and writes
/// the full complex spectrum into `output_re` and `output_im` (each of length N).
///
/// Uses approximately half the FLOPs of a full complex FFT by exploiting the
/// conjugate symmetry of real-valued signals.
///
/// # Panics
///
/// Panics if lengths don't match or N is not a power of 2 ≥ 4.
///
/// # Example
///
/// ```
/// use phastft::r2c_fft_f64;
///
/// let input: Vec<f64> = (1..=16).map(|x| x as f64).collect();
/// let mut out_re = vec![0.0; 16];
/// let mut out_im = vec![0.0; 16];
/// r2c_fft_f64(&input, &mut out_re, &mut out_im);
/// ```
pub fn r2c_fft_f64(input_re: &[f64], output_re: &mut [f64], output_im: &mut [f64]) {
    let n = input_re.len();
    let planner = PlannerR2c64::new(n);
    r2c_fft_f64_with_planner(input_re, output_re, output_im, &planner);
}

/// Performs a real-valued FFT of f64 data using a pre-computed planner.
///
/// This avoids recomputing twiddle factors on each call, which is beneficial
/// for repeated FFTs of the same size.
///
/// # Panics
///
/// Panics if input length doesn't match the planner size.
pub fn r2c_fft_f64_with_planner(
    input_re: &[f64],
    output_re: &mut [f64],
    output_im: &mut [f64],
    planner: &PlannerR2c64,
) {
    let n = input_re.len();
    assert_eq!(n, planner.n, "input length must match planner size");
    assert!(output_re.len() == n && output_im.len() == n);
    let half = n / 2;

    let (mut z_re, mut z_im) = deinterleave(input_re);

    let opts = Options::guess_options(half);
    fft_64_dit_with_planner_and_opts(
        &mut z_re,
        &mut z_im,
        Direction::Forward,
        &planner.dit_planner,
        &opts,
    );

    untangle_f64(
        &z_re,
        &z_im,
        &planner.w_re,
        &planner.w_im,
        output_re,
        output_im,
    );
}

/// Performs a real-valued FFT of f32 data.
///
/// See [`r2c_fft_f64`] for details. This is the single-precision variant.
pub fn r2c_fft_f32(input_re: &[f32], output_re: &mut [f32], output_im: &mut [f32]) {
    let n = input_re.len();
    let planner = PlannerR2c32::new(n);
    r2c_fft_f32_with_planner(input_re, output_re, output_im, &planner);
}

/// Performs a real-valued FFT of f32 data using a pre-computed planner.
///
/// See [`r2c_fft_f64_with_planner`] for details. This is the single-precision variant.
pub fn r2c_fft_f32_with_planner(
    input_re: &[f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
    planner: &PlannerR2c32,
) {
    let n = input_re.len();
    assert_eq!(n, planner.n, "input length must match planner size");
    assert!(output_re.len() == n && output_im.len() == n);
    let half = n / 2;

    let (mut z_re, mut z_im) = deinterleave(input_re);

    let opts = Options::guess_options(half);
    fft_32_dit_with_planner_and_opts(
        &mut z_re,
        &mut z_im,
        Direction::Forward,
        &planner.dit_planner,
        &opts,
    );

    untangle_f32(
        &z_re,
        &z_im,
        &planner.w_re,
        &planner.w_im,
        output_re,
        output_im,
    );
}

/// Performs the inverse real-valued FFT on `f64` data.
///
/// Given the full N-point complex spectrum (as produced by [`r2c_fft_f64`]),
/// recovers the original N real-valued samples.
///
/// # Panics
///
/// Panics if lengths don't match or N is not a power of 2 ≥ 4.
pub fn c2r_fft_f64(input_re: &[f64], input_im: &[f64], output: &mut [f64]) {
    let n = input_re.len();
    let planner = PlannerR2c64::new(n);
    c2r_fft_f64_with_planner(input_re, input_im, output, &planner);
}

/// Performs the inverse real-valued FFT of f64 data using a pre-computed planner.
///
/// # Panics
///
/// Panics if input length doesn't match the planner size.
pub fn c2r_fft_f64_with_planner(
    input_re: &[f64],
    input_im: &[f64],
    output: &mut [f64],
    planner: &PlannerR2c64,
) {
    let n = input_re.len();
    assert_eq!(n, planner.n, "input length must match planner size");
    assert!(input_im.len() == n && output.len() == n);

    let half = n / 2;

    let (re_first, re_second) = input_re.split_at(half);
    let (im_first, im_second) = input_im.split_at(half);

    let mut z_re = vec![0.0f64; half];
    let mut z_im = vec![0.0f64; half];

    for k in 0..half {
        let zx_re = 0.5 * (re_first[k] + re_second[k]);
        let zx_im = 0.5 * (im_first[k] + im_second[k]);

        let wzy_re = 0.5 * (re_first[k] - re_second[k]);
        let wzy_im = 0.5 * (im_first[k] - im_second[k]);

        // Planner stores forward W^k; C2R needs W^{-k}, so flip the sign on
        // every w_im term to consume the conjugate without a separate table.
        let c = planner.w_re[k];
        let s = planner.w_im[k];
        let zy_re = c * wzy_re + s * wzy_im;
        let zy_im = c * wzy_im - s * wzy_re;

        // Z[k] = Zx[k] + j · Zy[k]
        z_re[k] = zx_re - zy_im;
        z_im[k] = zx_im + zy_re;
    }

    let opts = Options::guess_options(half);
    fft_64_dit_with_planner_and_opts(
        &mut z_re,
        &mut z_im,
        Direction::Reverse,
        &planner.dit_planner,
        &opts,
    );

    for k in 0..half {
        output[2 * k] = z_re[k];
        output[2 * k + 1] = z_im[k];
    }
}

#[cfg(test)]
mod tests {
    use utilities::{assert_float_closeness, gen_random_signal_f64};

    use super::*;
    use crate::planner::Direction;
    use crate::{fft_32_dit, fft_64_dit};

    fn assert_f32_relative_closeness(actual: f32, expected: f32, rel_eps: f32) {
        let denom = expected.abs().max(f32::EPSILON);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err < rel_eps,
            "relative error {rel_err} >= {rel_eps} (actual={actual}, expected={expected})"
        );
    }

    #[test]
    fn r2c_vs_c2c_f64() {
        for n_log in 2..=14 {
            let n = 1 << n_log;
            let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

            let mut r2c_re = vec![0.0; n];
            let mut r2c_im = vec![0.0; n];
            r2c_fft_f64(&input, &mut r2c_re, &mut r2c_im);

            let mut ref_re = input.clone();
            let mut ref_im = vec![0.0; n];
            fft_64_dit(&mut ref_re, &mut ref_im, Direction::Forward);

            for k in 0..n {
                assert_float_closeness(r2c_re[k], ref_re[k], 1e-4);
                assert_float_closeness(r2c_im[k], ref_im[k], 1e-4);
            }
        }
    }

    #[test]
    fn r2c_vs_c2c_f32() {
        for n_log in 2..=10 {
            let n = 1 << n_log;
            let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

            let mut r2c_re = vec![0.0f32; n];
            let mut r2c_im = vec![0.0f32; n];
            r2c_fft_f32(&input, &mut r2c_re, &mut r2c_im);

            let mut ref_re: Vec<f32> = input.clone();
            let mut ref_im = vec![0.0f32; n];
            fft_32_dit(&mut ref_re, &mut ref_im, Direction::Forward);

            for k in 0..n {
                assert_f32_relative_closeness(r2c_re[k], ref_re[k], 1e-2);
                assert_f32_relative_closeness(r2c_im[k], ref_im[k], 1e-2);
            }
        }
    }

    #[test]
    fn roundtrip_f64() {
        for n_log in 2..=14 {
            let n = 1 << n_log;
            let original: Vec<f64> = (1..=n).map(|i| i as f64).collect();

            let mut spec_re = vec![0.0; n];
            let mut spec_im = vec![0.0; n];
            r2c_fft_f64(&original, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0; n];
            c2r_fft_f64(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original[k], 1e-6);
            }
        }
    }

    #[test]
    fn planner_matches_convenience() {
        let n = 1024;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let mut out_re_1 = vec![0.0; n];
        let mut out_im_1 = vec![0.0; n];
        r2c_fft_f64(&input, &mut out_re_1, &mut out_im_1);

        let planner = PlannerR2c64::new(n);
        let mut out_re_2 = vec![0.0; n];
        let mut out_im_2 = vec![0.0; n];
        r2c_fft_f64_with_planner(&input, &mut out_re_2, &mut out_im_2, &planner);

        for k in 0..n {
            assert_float_closeness(out_re_1[k], out_re_2[k], 1e-12);
            assert_float_closeness(out_im_1[k], out_im_2[k], 1e-12);
        }
    }

    #[test]
    fn c2r_planner_matches_convenience() {
        let n = 1024;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let mut spec_re = vec![0.0; n];
        let mut spec_im = vec![0.0; n];
        r2c_fft_f64(&input, &mut spec_re, &mut spec_im);

        let mut recovered_1 = vec![0.0; n];
        c2r_fft_f64(&spec_re, &spec_im, &mut recovered_1);

        let planner = PlannerR2c64::new(n);
        let mut recovered_2 = vec![0.0; n];
        c2r_fft_f64_with_planner(&spec_re, &spec_im, &mut recovered_2, &planner);

        for k in 0..n {
            assert_float_closeness(recovered_1[k], recovered_2[k], 1e-12);
            assert_float_closeness(recovered_2[k], input[k], 1e-6);
        }
    }

    #[test]
    fn roundtrip_random_f64() {
        for n_log in 4..=14 {
            let n = 1 << n_log;
            let mut original_re = vec![0.0f64; n];
            let mut dummy_im = vec![0.0f64; n];
            gen_random_signal_f64(&mut original_re, &mut dummy_im);

            let mut spec_re = vec![0.0; n];
            let mut spec_im = vec![0.0; n];
            r2c_fft_f64(&original_re, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0; n];
            c2r_fft_f64(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original_re[k], 1e-6);
            }
        }
    }
}
