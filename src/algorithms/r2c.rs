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

/// Stride-2 deinterleave.
/// `[a, b, c, d, ... ]` becomes `([a, c, ... ], [b, d, ... ])`
#[inline]
fn deinterleave<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}

/// Stride-2 interleave.
/// Writes `[z_re[0], z_im[0], z_re[1], z_im[1], ...]` into `output`.
#[inline]
fn interleave<T: Copy>(z_re: &[T], z_im: &[T], output: &mut [T]) {
    debug_assert!(z_re.len() == z_im.len() && output.len() == 2 * z_re.len());
    for k in 0..z_re.len() {
        output[2 * k] = z_re[k];
        output[2 * k + 1] = z_im[k];
    }
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

// Turns the full N-point spectrum into the `N/2` point complex spectrum that an inverse complex FFT
// can consume.
macro_rules! impl_c2r_preprocess {
    ($func_name:ident, $precision:ty) => {
        fn $func_name(
            input_re: &[$precision],
            input_im: &[$precision],
            w_re: &[$precision],
            w_im: &[$precision],
            z_re: &mut [$precision],
            z_im: &mut [$precision],
        ) {
            let half = z_re.len();

            let (re_first, re_second) = input_re.split_at(half);
            let (im_first, im_second) = input_im.split_at(half);

            for k in 0..half {
                let zx_re = 0.5 * (re_first[k] + re_second[k]);
                let zx_im = 0.5 * (im_first[k] + im_second[k]);

                let wzy_re = 0.5 * (re_first[k] - re_second[k]);
                let wzy_im = 0.5 * (im_first[k] - im_second[k]);

                // Planner stores forward W^k; C2R needs W^{-k}, so flip the sign on
                // every w_im term to consume the conjugate without a separate table.
                let c = w_re[k];
                let s = w_im[k];
                let zy_re = c * wzy_re + s * wzy_im;
                let zy_im = c * wzy_im - s * wzy_re;

                // Z[k] = Zx[k] + j · Zy[k]
                z_re[k] = zx_re - zy_im;
                z_im[k] = zx_im + zy_re;
            }
        }
    };
}

impl_c2r_preprocess!(c2r_preprocess_f64, f64);
impl_c2r_preprocess!(c2r_preprocess_f32, f32);

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
/// See [`r2c_fft_f64_with_planner`] for details. This is the single-precision (i.e., `f32`)
/// variant.
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
///
/// # Example
///
/// ```
/// use phastft::{c2r_fft_f64, r2c_fft_f64};
///
/// let signal: Vec<f64> = (1..=16).map(|x| x as f64).collect();
/// let mut spec_re = vec![0.0; 16];
/// let mut spec_im = vec![0.0; 16];
/// r2c_fft_f64(&signal, &mut spec_re, &mut spec_im);
///
/// let mut recovered = vec![0.0; 16];
/// c2r_fft_f64(&spec_re, &spec_im, &mut recovered);
/// ```
///
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
    let mut z_re = vec![0.0f64; half];
    let mut z_im = vec![0.0f64; half];

    c2r_preprocess_f64(
        input_re,
        input_im,
        &planner.w_re,
        &planner.w_im,
        &mut z_re,
        &mut z_im,
    );

    let opts = Options::guess_options(half);
    fft_64_dit_with_planner_and_opts(
        &mut z_re,
        &mut z_im,
        Direction::Reverse,
        &planner.dit_planner,
        &opts,
    );

    interleave(&z_re, &z_im, output);
}

/// Performs the inverse real-valued FFT on `f32` data.
///
/// See [`c2r_fft_f64`] for details. This is the single-precision variant.
///
/// # Panics
///
/// Panics if lengths don't match or N is not a power of 2 ≥ 4.
pub fn c2r_fft_f32(input_re: &[f32], input_im: &[f32], output: &mut [f32]) {
    let n = input_re.len();
    let planner = PlannerR2c32::new(n);
    c2r_fft_f32_with_planner(input_re, input_im, output, &planner);
}

/// Performs the inverse real-valued FFT of f32 data using a pre-computed planner.
///
/// See [`c2r_fft_f64_with_planner`] for details. This is the single-precision variant.
pub fn c2r_fft_f32_with_planner(
    input_re: &[f32],
    input_im: &[f32],
    output: &mut [f32],
    planner: &PlannerR2c32,
) {
    let n = input_re.len();
    assert_eq!(n, planner.n, "input length must match planner size");
    assert!(input_im.len() == n && output.len() == n);

    let half = n / 2;
    let mut z_re = vec![0.0f32; half];
    let mut z_im = vec![0.0f32; half];

    c2r_preprocess_f32(
        input_re,
        input_im,
        &planner.w_re,
        &planner.w_im,
        &mut z_re,
        &mut z_im,
    );

    let opts = Options::guess_options(half);
    fft_32_dit_with_planner_and_opts(
        &mut z_re,
        &mut z_im,
        Direction::Reverse,
        &planner.dit_planner,
        &opts,
    );

    interleave(&z_re, &z_im, output);
}

#[cfg(test)]
mod tests {
    use utilities::{assert_float_closeness, gen_random_signal_f32, gen_random_signal_f64};

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

    #[test]
    fn roundtrip_f32() {
        for n_log in 2..=10 {
            let n = 1 << n_log;
            let original: Vec<f32> = (1..=n).map(|i| i as f32).collect();

            let mut spec_re = vec![0.0f32; n];
            let mut spec_im = vec![0.0f32; n];
            r2c_fft_f32(&original, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0f32; n];
            c2r_fft_f32(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_f32_relative_closeness(recovered[k], original[k], 1e-2);
            }
        }
    }

    #[test]
    fn roundtrip_random_f32() {
        for n_log in 4..=12 {
            let n = 1 << n_log;
            let mut original_re = vec![0.0f32; n];
            let mut dummy_im = vec![0.0f32; n];
            gen_random_signal_f32(&mut original_re, &mut dummy_im);

            let mut spec_re = vec![0.0f32; n];
            let mut spec_im = vec![0.0f32; n];
            r2c_fft_f32(&original_re, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0f32; n];
            c2r_fft_f32(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original_re[k], 1e-5);
            }
        }
    }

    #[test]
    fn c2r_planner_matches_convenience_f32() {
        let n = 1024;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

        let mut spec_re = vec![0.0f32; n];
        let mut spec_im = vec![0.0f32; n];
        r2c_fft_f32(&input, &mut spec_re, &mut spec_im);

        let mut recovered_1 = vec![0.0f32; n];
        c2r_fft_f32(&spec_re, &spec_im, &mut recovered_1);

        let planner = PlannerR2c32::new(n);
        let mut recovered_2 = vec![0.0f32; n];
        c2r_fft_f32_with_planner(&spec_re, &spec_im, &mut recovered_2, &planner);

        for k in 0..n {
            // Same input, same planner contents -> bitwise-equivalent f32 output
            assert_eq!(recovered_1[k], recovered_2[k]);
        }
    }

    // ---------------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------------

    #[test]
    fn dc_only_f64() {
        // DFT of a constant 1: X[0] = N, all other bins zero.
        let n = 16;
        let input = vec![1.0f64; n];
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_re[0], n as f64, 1e-10);
        assert_float_closeness(out_im[0], 0.0, 1e-10);
        for k in 1..n {
            assert_float_closeness(out_re[k], 0.0, 1e-10);
            assert_float_closeness(out_im[k], 0.0, 1e-10);
        }
    }

    #[test]
    fn nyquist_only_f64() {
        // Alternating ±1 -> all energy at bin N/2 (Nyquist).
        let n = 16;
        let input: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

        for k in 0..n {
            let expected_re = if k == n / 2 { n as f64 } else { 0.0 };
            assert_float_closeness(out_re[k], expected_re, 1e-10);
            assert_float_closeness(out_im[k], 0.0, 1e-10);
        }
    }

    #[test]
    fn single_tone_f64() {
        // cos(2π·j/N) -> X[1] = X[N-1] = N/2, all other bins zero.
        let n = 32;
        let input: Vec<f64> = (0..n)
            .map(|j| (2.0 * std::f64::consts::PI * j as f64 / n as f64).cos())
            .collect();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

        for k in 0..n {
            let expected_re = if k == 1 || k == n - 1 {
                n as f64 / 2.0
            } else {
                0.0
            };
            assert_float_closeness(out_re[k], expected_re, 1e-9);
            assert_float_closeness(out_im[k], 0.0, 1e-9);
        }
    }

    #[test]
    fn all_zeros_f64() {
        // Output buffers are pre-filled to verify they get overwritten.
        let n = 16;
        let input = vec![0.0f64; n];
        let mut out_re = vec![1.0; n];
        let mut out_im = vec![1.0; n];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

        for k in 0..n {
            assert_float_closeness(out_re[k], 0.0, 1e-12);
            assert_float_closeness(out_im[k], 0.0, 1e-12);
        }
    }

    #[test]
    fn r2c_output_conjugate_symmetric_f64() {
        // For real input, the spectrum satisfies X[N-k] = conj(X[k]).
        // Bins 0 and N/2 are purely real.
        let n = 64;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_im[0], 0.0, 1e-10);
        assert_float_closeness(out_im[n / 2], 0.0, 1e-10);

        for k in 1..n / 2 {
            assert_float_closeness(out_re[n - k], out_re[k], 1e-10);
            assert_float_closeness(out_im[n - k], -out_im[k], 1e-10);
        }
    }

    // ---------------------------------------------------------------------
    // Panic tests — invariants enforced via PlannerR2c::new and length asserts
    // ---------------------------------------------------------------------

    macro_rules! r2c_panics_on_invalid_n {
        ($test_name:ident, $func:ident, $precision:ty, $n:expr) => {
            #[test]
            #[should_panic(expected = "n must be a power of 2 >= 4")]
            fn $test_name() {
                let n: usize = $n;
                let input = vec![<$precision>::default(); n];
                let mut out_re = vec![<$precision>::default(); n];
                let mut out_im = vec![<$precision>::default(); n];
                $func(&input, &mut out_re, &mut out_im);
            }
        };
    }

    r2c_panics_on_invalid_n!(r2c_fft_f64_panics_on_n_less_than_4, r2c_fft_f64, f64, 2);
    r2c_panics_on_invalid_n!(r2c_fft_f64_panics_on_non_power_of_two, r2c_fft_f64, f64, 5);
    r2c_panics_on_invalid_n!(r2c_fft_f32_panics_on_n_less_than_4, r2c_fft_f32, f32, 2);
    r2c_panics_on_invalid_n!(r2c_fft_f32_panics_on_non_power_of_two, r2c_fft_f32, f32, 5);

    macro_rules! c2r_panics_on_invalid_n {
        ($test_name:ident, $func:ident, $precision:ty, $n:expr) => {
            #[test]
            #[should_panic(expected = "n must be a power of 2 >= 4")]
            fn $test_name() {
                let n: usize = $n;
                let in_re = vec![<$precision>::default(); n];
                let in_im = vec![<$precision>::default(); n];
                let mut out = vec![<$precision>::default(); n];
                $func(&in_re, &in_im, &mut out);
            }
        };
    }

    c2r_panics_on_invalid_n!(c2r_fft_f64_panics_on_n_less_than_4, c2r_fft_f64, f64, 2);
    c2r_panics_on_invalid_n!(c2r_fft_f64_panics_on_non_power_of_two, c2r_fft_f64, f64, 5);
    c2r_panics_on_invalid_n!(c2r_fft_f32_panics_on_n_less_than_4, c2r_fft_f32, f32, 2);
    c2r_panics_on_invalid_n!(c2r_fft_f32_panics_on_non_power_of_two, c2r_fft_f32, f32, 5);

    #[test]
    #[should_panic]
    fn r2c_fft_f64_panics_on_output_length_mismatch() {
        let input = vec![0.0f64; 16];
        let mut out_re = vec![0.0f64; 8];
        let mut out_im = vec![0.0f64; 16];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);
    }

    #[test]
    #[should_panic]
    fn c2r_fft_f64_panics_on_input_length_mismatch() {
        let in_re = vec![0.0f64; 16];
        let in_im = vec![0.0f64; 8];
        let mut out = vec![0.0f64; 16];
        c2r_fft_f64(&in_re, &in_im, &mut out);
    }

    #[test]
    #[should_panic(expected = "input length must match planner size")]
    fn r2c_fft_f64_with_planner_panics_on_planner_size_mismatch() {
        let planner = PlannerR2c64::new(16);
        let input = vec![0.0f64; 32];
        let mut out_re = vec![0.0f64; 32];
        let mut out_im = vec![0.0f64; 32];
        r2c_fft_f64_with_planner(&input, &mut out_re, &mut out_im, &planner);
    }

    #[test]
    #[should_panic(expected = "input length must match planner size")]
    fn c2r_fft_f64_with_planner_panics_on_planner_size_mismatch() {
        let planner = PlannerR2c64::new(16);
        let in_re = vec![0.0f64; 32];
        let in_im = vec![0.0f64; 32];
        let mut out = vec![0.0f64; 32];
        c2r_fft_f64_with_planner(&in_re, &in_im, &mut out, &planner);
    }
}
