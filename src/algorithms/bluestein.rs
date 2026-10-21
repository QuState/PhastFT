//! Complex FFTs of arbitrary length using Bluestein's algorithm.
//!
//! Bluestein's identity rewrites an `N`-point DFT as a convolution. The
//! convolution is zero-padded to `M = (2N - 1).next_power_of_two()` and
//! evaluated with PhastFT's power-of-two DIT transform.
//!
//! Forward transforms are unnormalized. Inverse transforms include the usual
//! `1 / N` normalization, matching the rest of this crate.

use fearless_simd::{dispatch, f32x8, f64x4, Simd, SimdBase};

use crate::algorithms::dit::{
    fft_f32_dit_with_planner_and_opts_impl, fft_f64_dit_with_planner_and_opts_impl,
};
use crate::options::Options;
use crate::planner::{Direction, PlannerBluestein32, PlannerBluestein64};

// Multiply by the chirp and zero-pad to the convolution length.
macro_rules! impl_simd_bluestein_premul {
    ($name:ident, $T:ty, $V:ident, $lanes:expr) => {
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        fn $name<S: Simd>(
            simd: S,
            signal_re: &[$T],
            signal_im: &[$T],
            c_re: &[$T],
            c_im: &[$T],
            out_re: &mut [$T],
            out_im: &mut [$T],
            imag_sign: $T,
        ) {
            const LANES: usize = $lanes;
            let n = signal_re.len();
            let m = out_re.len();
            let imag_sign_v = $V::splat(simd, imag_sign);

            let n_blocks = n / LANES;
            for blk in 0..n_blocks {
                let off = blk * LANES;
                let xr = $V::from_slice(simd, &signal_re[off..off + LANES]);
                let xi = imag_sign_v * $V::from_slice(simd, &signal_im[off..off + LANES]);
                let cr = $V::from_slice(simd, &c_re[off..off + LANES]);
                let ci = $V::from_slice(simd, &c_im[off..off + LANES]);
                (xr * cr - xi * ci).store_slice(&mut out_re[off..off + LANES]);
                (xr * ci + xi * cr).store_slice(&mut out_im[off..off + LANES]);
            }
            for k in (n_blocks * LANES)..n {
                let xr = signal_re[k];
                let xi = imag_sign * signal_im[k];
                let cr = c_re[k];
                let ci = c_im[k];
                out_re[k] = xr * cr - xi * ci;
                out_im[k] = xr * ci + xi * cr;
            }
            out_re[n..m].fill(0.0);
            out_im[n..m].fill(0.0);
        }
    };
}

// Pointwise multiply by the precomputed convolution-kernel spectrum.
macro_rules! impl_simd_bluestein_pointwise {
    ($name:ident, $T:ty, $V:ident, $lanes:expr) => {
        #[inline(always)]
        fn $name<S: Simd>(simd: S, a_re: &mut [$T], a_im: &mut [$T], b_re: &[$T], b_im: &[$T]) {
            const LANES: usize = $lanes;
            let m = a_re.len();

            let n_blocks = m / LANES;
            for blk in 0..n_blocks {
                let off = blk * LANES;
                let ar = $V::from_slice(simd, &a_re[off..off + LANES]);
                let ai = $V::from_slice(simd, &a_im[off..off + LANES]);
                let br = $V::from_slice(simd, &b_re[off..off + LANES]);
                let bi = $V::from_slice(simd, &b_im[off..off + LANES]);
                (ar * br - ai * bi).store_slice(&mut a_re[off..off + LANES]);
                (ar * bi + ai * br).store_slice(&mut a_im[off..off + LANES]);
            }
            for k in (n_blocks * LANES)..m {
                let ar = a_re[k];
                let ai = a_im[k];
                let br = b_re[k];
                let bi = b_im[k];
                a_re[k] = ar * br - ai * bi;
                a_im[k] = ar * bi + ai * br;
            }
        }
    };
}

// Apply the second chirp and copy the first N samples back to the signal.
macro_rules! impl_simd_bluestein_postmul {
    ($name:ident, $T:ty, $V:ident, $lanes:expr) => {
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        fn $name<S: Simd>(
            simd: S,
            conv_re: &[$T],
            conv_im: &[$T],
            c_re: &[$T],
            c_im: &[$T],
            out_re: &mut [$T],
            out_im: &mut [$T],
            imag_sign: $T,
            scale: $T,
        ) {
            const LANES: usize = $lanes;
            let n = out_re.len();
            let scale_v = $V::splat(simd, scale);
            let imag_scale_v = $V::splat(simd, imag_sign) * scale_v;

            let n_blocks = n / LANES;
            for blk in 0..n_blocks {
                let off = blk * LANES;
                let vr = $V::from_slice(simd, &conv_re[off..off + LANES]);
                let vi = $V::from_slice(simd, &conv_im[off..off + LANES]);
                let cr = $V::from_slice(simd, &c_re[off..off + LANES]);
                let ci = $V::from_slice(simd, &c_im[off..off + LANES]);
                let pr = cr * vr - ci * vi;
                let pi = cr * vi + ci * vr;
                (scale_v * pr).store_slice(&mut out_re[off..off + LANES]);
                (imag_scale_v * pi).store_slice(&mut out_im[off..off + LANES]);
            }
            for k in (n_blocks * LANES)..n {
                let vr = conv_re[k];
                let vi = conv_im[k];
                let cr = c_re[k];
                let ci = c_im[k];
                let pr = cr * vr - ci * vi;
                let pi = cr * vi + ci * vr;
                out_re[k] = scale * pr;
                out_im[k] = imag_sign * scale * pi;
            }
        }
    };
}

impl_simd_bluestein_premul!(simd_bluestein_premul_f64, f64, f64x4, 4);
impl_simd_bluestein_pointwise!(simd_bluestein_pointwise_f64, f64, f64x4, 4);
impl_simd_bluestein_postmul!(simd_bluestein_postmul_f64, f64, f64x4, 4);

impl_simd_bluestein_premul!(simd_bluestein_premul_f32, f32, f32x8, 8);
impl_simd_bluestein_pointwise!(simd_bluestein_pointwise_f32, f32, f32x8, 8);
impl_simd_bluestein_postmul!(simd_bluestein_postmul_f32, f32, f32x8, 8);

#[inline]
fn direction_params_f64(direction: Direction, len: usize) -> (f64, f64) {
    match direction {
        Direction::Forward => (1.0, 1.0),
        Direction::Inverse => (-1.0, 1.0 / len as f64),
    }
}

#[inline]
fn direction_params_f32(direction: Direction, len: usize) -> (f32, f32) {
    match direction {
        Direction::Forward => (1.0, 1.0),
        Direction::Inverse => (-1.0, 1.0 / len as f32),
    }
}

/// Computes an arbitrary-length complex FFT with a reusable plan and scratch.
///
/// The transform is performed in place over separate real and imaginary
/// arrays. `scratch_re` and `scratch_im` must each contain at least
/// [`PlannerBluestein64::scratch_len`] elements. Their contents are ignored on
/// entry and unspecified on return.
///
/// `opts` controls the inner power-of-two FFT. When using
/// [`Options::guess_options`], pass `planner.scratch_len()` rather than the
/// input length.
///
/// # Panics
///
/// Panics if the input lengths differ from `planner.fft_len()`, or if either
/// scratch slice is shorter than `planner.scratch_len()`.
///
/// # Example
///
/// ```
/// use phastft::{fft_f64_bluestein_with_planner_and_opts, options::Options};
/// use phastft::planner::{Direction, PlannerBluestein64};
///
/// let n = 6; // not a power of two
/// let planner = PlannerBluestein64::new(n);
/// let scratch_len = planner.scratch_len();
/// let opts = Options::guess_options(scratch_len);
///
/// let mut reals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
/// let mut imags = vec![0.0; n];
/// let mut scratch_re = vec![0.0; scratch_len];
/// let mut scratch_im = vec![0.0; scratch_len];
/// fft_f64_bluestein_with_planner_and_opts(
///     &mut reals, &mut imags, Direction::Forward, &planner, &opts,
///     &mut scratch_re, &mut scratch_im,
/// );
/// ```
pub fn fft_f64_bluestein_with_planner_and_opts(
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerBluestein64,
    opts: &Options,
    scratch_re: &mut [f64],
    scratch_im: &mut [f64],
) {
    let len = planner.len;
    let inner_len = planner.inner_len;
    assert_eq!(
        reals.len(),
        len,
        "reals length must match planner.fft_len()"
    );
    assert_eq!(
        imags.len(),
        len,
        "imags length must match planner.fft_len()"
    );
    assert!(
        scratch_re.len() >= inner_len,
        "scratch_re is too short: need at least {inner_len} elements, got {}",
        scratch_re.len()
    );
    assert!(
        scratch_im.len() >= inner_len,
        "scratch_im is too short: need at least {inner_len} elements, got {}",
        scratch_im.len()
    );
    let scratch_re = &mut scratch_re[..inner_len];
    let scratch_im = &mut scratch_im[..inner_len];

    // The inverse is (1/N) * conj(DFT(conj(x))), so both directions can share
    // the same chirp and convolution-kernel spectrum.
    let (imag_sign, scale) = direction_params_f64(direction, len);

    dispatch!(planner.dit_planner.simd_level, simd => {
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_premul_f64(
                simd, reals, imags, &planner.chirp_re, &planner.chirp_im,
                scratch_re, scratch_im, imag_sign,
            ),
        );
        fft_f64_dit_with_planner_and_opts_impl(
            simd, scratch_re, scratch_im, Direction::Forward, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_pointwise_f64(
                simd, scratch_re, scratch_im,
                &planner.kernel_fft_re, &planner.kernel_fft_im,
            ),
        );
        // Swapping real and imaginary components turns the forward transform
        // into an unscaled inverse. The kernel's 1/M factor, applied while
        // planning, supplies the convolution normalization.
        fft_f64_dit_with_planner_and_opts_impl(
            simd, scratch_im, scratch_re, Direction::Forward, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_postmul_f64(
                simd, scratch_re, scratch_im, &planner.chirp_re, &planner.chirp_im,
                reals, imags, imag_sign, scale,
            ),
        );
    });
}

/// Computes a complex FFT with a reusable plan.
///
/// This convenience function allocates scratch and chooses options for the
/// inner FFT. Use [`fft_f64_bluestein_with_planner_and_opts`] to reuse scratch.
///
/// # Panics
///
/// Panics if either input length differs from `planner.fft_len()`.
pub fn fft_f64_bluestein_with_planner(
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerBluestein64,
) {
    let opts = Options::guess_options(planner.scratch_len());
    let mut scratch_re = vec![0.0; planner.scratch_len()];
    let mut scratch_im = vec![0.0; planner.scratch_len()];
    fft_f64_bluestein_with_planner_and_opts(
        reals,
        imags,
        direction,
        planner,
        &opts,
        &mut scratch_re,
        &mut scratch_im,
    );
}

/// Computes an arbitrary-length complex FFT, building a plan for this call.
///
/// For repeated transforms of the same length, reuse a
/// [`PlannerBluestein64`] via [`fft_f64_bluestein_with_planner`].
///
/// # Panics
///
/// Panics if `reals` and `imags` have different lengths, or the length is 0.
///
/// # Example
///
/// ```
/// use phastft::{fft_f64_bluestein, planner::Direction};
///
/// let mut reals = vec![1.0, 2.0, 3.0]; // N = 3, not a power of two
/// let mut imags = vec![0.0; 3];
/// fft_f64_bluestein(&mut reals, &mut imags, Direction::Forward);
/// ```
pub fn fft_f64_bluestein(reals: &mut [f64], imags: &mut [f64], direction: Direction) {
    assert_eq!(
        reals.len(),
        imags.len(),
        "reals and imags must have equal length"
    );
    let planner = PlannerBluestein64::new(reals.len());
    fft_f64_bluestein_with_planner(reals, imags, direction, &planner);
}

/// Computes an arbitrary-length `f32` complex FFT with a reusable plan and scratch.
///
/// Single-precision variant of [`fft_f64_bluestein_with_planner_and_opts`]. See
/// that function for the scratch and `Options` contract.
///
/// # Panics
///
/// Panics if the input lengths differ from `planner.fft_len()`, or if either
/// scratch slice is shorter than `planner.scratch_len()`.
pub fn fft_f32_bluestein_with_planner_and_opts(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerBluestein32,
    opts: &Options,
    scratch_re: &mut [f32],
    scratch_im: &mut [f32],
) {
    let len = planner.len;
    let inner_len = planner.inner_len;
    assert_eq!(
        reals.len(),
        len,
        "reals length must match planner.fft_len()"
    );
    assert_eq!(
        imags.len(),
        len,
        "imags length must match planner.fft_len()"
    );
    assert!(
        scratch_re.len() >= inner_len,
        "scratch_re is too short: need at least {inner_len} elements, got {}",
        scratch_re.len()
    );
    assert!(
        scratch_im.len() >= inner_len,
        "scratch_im is too short: need at least {inner_len} elements, got {}",
        scratch_im.len()
    );
    let scratch_re = &mut scratch_re[..inner_len];
    let scratch_im = &mut scratch_im[..inner_len];

    let (imag_sign, scale) = direction_params_f32(direction, len);

    dispatch!(planner.dit_planner.simd_level, simd => {
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_premul_f32(
                simd, reals, imags, &planner.chirp_re, &planner.chirp_im,
                scratch_re, scratch_im, imag_sign,
            ),
        );
        fft_f32_dit_with_planner_and_opts_impl(
            simd, scratch_re, scratch_im, Direction::Forward, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_pointwise_f32(
                simd, scratch_re, scratch_im,
                &planner.kernel_fft_re, &planner.kernel_fft_im,
            ),
        );
        fft_f32_dit_with_planner_and_opts_impl(
            simd, scratch_im, scratch_re, Direction::Forward, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_postmul_f32(
                simd, scratch_re, scratch_im, &planner.chirp_re, &planner.chirp_im,
                reals, imags, imag_sign, scale,
            ),
        );
    });
}

/// Computes an `f32` complex FFT with a reusable plan.
///
/// This convenience function allocates scratch and chooses options for the
/// inner FFT. Use [`fft_f32_bluestein_with_planner_and_opts`] to reuse scratch.
///
/// # Panics
///
/// Panics if either input length differs from `planner.fft_len()`.
pub fn fft_f32_bluestein_with_planner(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerBluestein32,
) {
    let opts = Options::guess_options(planner.scratch_len());
    let mut scratch_re = vec![0.0; planner.scratch_len()];
    let mut scratch_im = vec![0.0; planner.scratch_len()];
    fft_f32_bluestein_with_planner_and_opts(
        reals,
        imags,
        direction,
        planner,
        &opts,
        &mut scratch_re,
        &mut scratch_im,
    );
}

/// Computes an arbitrary-length `f32` complex FFT, building a plan for this call.
///
/// Single-precision variant of [`fft_f64_bluestein`].
///
/// # Panics
///
/// Panics if `reals` and `imags` have different lengths, or the length is 0.
pub fn fft_f32_bluestein(reals: &mut [f32], imags: &mut [f32], direction: Direction) {
    assert_eq!(
        reals.len(),
        imags.len(),
        "reals and imags must have equal length"
    );
    let planner = PlannerBluestein32::new(reals.len());
    fft_f32_bluestein_with_planner(reals, imags, direction, &planner);
}

#[cfg(test)]
mod tests {
    use utilities::{assert_float_closeness, gen_random_signal_f64};

    use super::*;

    fn assert_close_f64(actual: f64, expected: f64, rel: f64) {
        let denom = expected.abs().max(1.0);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err < rel,
            "rel_err {rel_err} >= {rel} (actual={actual}, expected={expected})"
        );
    }

    fn rustfft_reference_f64(re: &[f64], im: &[f64], direction: Direction) -> (Vec<f64>, Vec<f64>) {
        use utilities::rustfft::num_complex::Complex;
        use utilities::rustfft::FftPlanner;

        let n = re.len();
        let mut buf: Vec<Complex<f64>> = re
            .iter()
            .zip(im)
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut planner = FftPlanner::new();
        let (fft, scale) = match direction {
            Direction::Forward => (planner.plan_fft_forward(n), 1.0),
            Direction::Inverse => (planner.plan_fft_inverse(n), 1.0 / n as f64),
        };
        fft.process(&mut buf);
        (
            buf.iter().map(|c| c.re * scale).collect(),
            buf.iter().map(|c| c.im * scale).collect(),
        )
    }

    // Cover scalar tails and convolution-length transitions as well as prime,
    // composite, and power-of-two transform lengths.
    const SIZES: &[usize] = &[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 31, 32, 63, 64, 65, 100, 127, 128,
        129, 257, 1000, 1024,
    ];

    #[test]
    fn forward_vs_rustfft_f64() {
        for &n in SIZES {
            let re: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let im: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.5 - 1.0).collect();
            let (exp_re, exp_im) = rustfft_reference_f64(&re, &im, Direction::Forward);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f64_bluestein(&mut got_re, &mut got_im, Direction::Forward);

            for k in 0..n {
                assert_close_f64(got_re[k], exp_re[k], 1e-8);
                assert_close_f64(got_im[k], exp_im[k], 1e-8);
            }
        }
    }

    #[test]
    fn inverse_vs_rustfft_f64() {
        for &n in SIZES {
            let re: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.25).collect();
            let im: Vec<f64> = (1..=n).map(|i| 2.0 - i as f64).collect();
            let (exp_re, exp_im) = rustfft_reference_f64(&re, &im, Direction::Inverse);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f64_bluestein(&mut got_re, &mut got_im, Direction::Inverse);

            for k in 0..n {
                assert_close_f64(got_re[k], exp_re[k], 1e-8);
                assert_close_f64(got_im[k], exp_im[k], 1e-8);
            }
        }
    }

    #[test]
    fn roundtrip_f64() {
        for &n in SIZES {
            let mut re = vec![0.0f64; n];
            let mut im = vec![0.0f64; n];
            gen_random_signal_f64(&mut re, &mut im);
            let (orig_re, orig_im) = (re.clone(), im.clone());

            fft_f64_bluestein(&mut re, &mut im, Direction::Forward);
            fft_f64_bluestein(&mut re, &mut im, Direction::Inverse);

            for k in 0..n {
                assert_close_f64(re[k], orig_re[k], 1e-8);
                assert_close_f64(im[k], orig_im[k], 1e-8);
            }
        }
    }

    #[test]
    fn n_equals_one_is_identity_f64() {
        let mut re = vec![42.0f64];
        let mut im = vec![-7.0f64];
        fft_f64_bluestein(&mut re, &mut im, Direction::Forward);
        assert_float_closeness(re[0], 42.0, 1e-12);
        assert_float_closeness(im[0], -7.0, 1e-12);
    }

    #[test]
    #[should_panic]
    fn signal_length_mismatch_panics_f64() {
        let planner = PlannerBluestein64::new(7);
        let opts = crate::options::Options::guess_options(planner.scratch_len());
        let m = planner.scratch_len();
        let mut re = vec![0.0f64; 8];
        let mut im = vec![0.0f64; 8];
        let mut scratch_re = vec![0.0f64; m];
        let mut scratch_im = vec![0.0f64; m];
        fft_f64_bluestein_with_planner_and_opts(
            &mut re,
            &mut im,
            Direction::Forward,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "scratch_re is too short")]
    fn short_scratch_panics_f64() {
        let planner = PlannerBluestein64::new(7);
        let opts = crate::options::Options::guess_options(planner.scratch_len());
        let m = planner.scratch_len();
        let mut re = vec![0.0f64; 7];
        let mut im = vec![0.0f64; 7];
        let mut scratch_re = vec![0.0f64; m - 1];
        let mut scratch_im = vec![0.0f64; m];
        fft_f64_bluestein_with_planner_and_opts(
            &mut re,
            &mut im,
            Direction::Forward,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    fn oversized_scratch_is_accepted_f64() {
        let len = 17;
        let planner = PlannerBluestein64::new(len);
        let opts = Options::guess_options(planner.scratch_len());
        let mut re: Vec<f64> = (0..len).map(|i| (i as f64).sin()).collect();
        let mut im: Vec<f64> = (0..len).map(|i| (i as f64).cos()).collect();
        let (expected_re, expected_im) = rustfft_reference_f64(&re, &im, Direction::Forward);

        let extra = 7;
        let mut scratch_re = vec![f64::NAN; planner.scratch_len() + extra];
        let mut scratch_im = vec![f64::NAN; planner.scratch_len() + extra];
        fft_f64_bluestein_with_planner_and_opts(
            &mut re,
            &mut im,
            Direction::Forward,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );

        for k in 0..len {
            assert_close_f64(re[k], expected_re[k], 1e-8);
            assert_close_f64(im[k], expected_im[k], 1e-8);
        }
        assert!(scratch_re[planner.scratch_len()..]
            .iter()
            .all(|value| value.is_nan()));
        assert!(scratch_im[planner.scratch_len()..]
            .iter()
            .all(|value| value.is_nan()));
    }

    fn assert_close_f32(actual: f32, expected: f32, rel: f32) {
        let denom = expected.abs().max(1.0);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err < rel,
            "rel_err {rel_err} >= {rel} (actual={actual}, expected={expected})"
        );
    }

    fn rustfft_reference_f32(re: &[f32], im: &[f32], direction: Direction) -> (Vec<f32>, Vec<f32>) {
        use utilities::rustfft::num_complex::Complex;
        use utilities::rustfft::FftPlanner;

        let n = re.len();
        let mut buf: Vec<Complex<f32>> = re
            .iter()
            .zip(im)
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut planner = FftPlanner::new();
        let (fft, scale) = match direction {
            Direction::Forward => (planner.plan_fft_forward(n), 1.0),
            Direction::Inverse => (planner.plan_fft_inverse(n), 1.0 / n as f32),
        };
        fft.process(&mut buf);
        (
            buf.iter().map(|c| c.re * scale).collect(),
            buf.iter().map(|c| c.im * scale).collect(),
        )
    }

    const SIZES_F32: &[usize] = &[
        1, 2, 3, 4, 5, 7, 8, 9, 13, 15, 16, 17, 31, 32, 63, 64, 65, 100, 127, 128, 129, 256, 257,
        1000,
    ];

    #[test]
    fn forward_vs_rustfft_f32() {
        for &n in SIZES_F32 {
            let re: Vec<f32> = (1..=n).map(|i| (i as f32 * 0.1).sin()).collect();
            let im: Vec<f32> = (1..=n).map(|i| (i as f32 * 0.05).cos()).collect();
            let (exp_re, exp_im) = rustfft_reference_f32(&re, &im, Direction::Forward);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f32_bluestein(&mut got_re, &mut got_im, Direction::Forward);

            for k in 0..n {
                assert_close_f32(got_re[k], exp_re[k], 2e-4);
                assert_close_f32(got_im[k], exp_im[k], 2e-4);
            }
        }
    }

    #[test]
    fn inverse_vs_rustfft_f32() {
        for &n in SIZES_F32 {
            let re: Vec<f32> = (1..=n).map(|i| (i as f32) * 0.25).collect();
            let im: Vec<f32> = (1..=n).map(|i| 1.0 - i as f32 * 0.125).collect();
            let (exp_re, exp_im) = rustfft_reference_f32(&re, &im, Direction::Inverse);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f32_bluestein(&mut got_re, &mut got_im, Direction::Inverse);

            for k in 0..n {
                assert_close_f32(got_re[k], exp_re[k], 2e-4);
                assert_close_f32(got_im[k], exp_im[k], 2e-4);
            }
        }
    }

    #[test]
    fn roundtrip_f32() {
        for &n in SIZES_F32 {
            let re: Vec<f32> = (1..=n).map(|i| (i as f32).sin()).collect();
            let im: Vec<f32> = (1..=n).map(|i| (i as f32).cos()).collect();
            let (orig_re, orig_im) = (re.clone(), im.clone());

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f32_bluestein(&mut got_re, &mut got_im, Direction::Forward);
            fft_f32_bluestein(&mut got_re, &mut got_im, Direction::Inverse);

            for k in 0..n {
                assert_close_f32(got_re[k], orig_re[k], 2e-4);
                assert_close_f32(got_im[k], orig_im[k], 2e-4);
            }
        }
    }

    #[test]
    fn tiers_agree_f64() {
        let n = 100;
        let re: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let im: Vec<f64> = (1..=n).map(|i| -(i as f64)).collect();

        let mut bare_re = re.clone();
        let mut bare_im = im.clone();
        fft_f64_bluestein(&mut bare_re, &mut bare_im, Direction::Forward);

        let planner = PlannerBluestein64::new(n);
        let mut planned_re = re.clone();
        let mut planned_im = im.clone();
        fft_f64_bluestein_with_planner(
            &mut planned_re,
            &mut planned_im,
            Direction::Forward,
            &planner,
        );

        let opts = Options::guess_options(planner.scratch_len());
        let m = planner.scratch_len();
        let mut full_re = re.clone();
        let mut full_im = im.clone();
        let mut scratch_re = vec![0.0; m];
        let mut scratch_im = vec![0.0; m];
        fft_f64_bluestein_with_planner_and_opts(
            &mut full_re,
            &mut full_im,
            Direction::Forward,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );

        assert_eq!((bare_re, bare_im), (planned_re.clone(), planned_im.clone()));
        assert_eq!((planned_re, planned_im), (full_re, full_im));
    }

    #[test]
    fn scratch_reuse_across_calls_f64() {
        let n = 127;
        let planner = PlannerBluestein64::new(n);
        let opts = Options::guess_options(planner.scratch_len());
        let m = planner.scratch_len();
        let mut scratch_re = vec![0.0; m];
        let mut scratch_im = vec![0.0; m];

        for seed in 0..4u64 {
            let re: Vec<f64> = (0..n).map(|i| ((i as u64 + seed) as f64).sin()).collect();
            let im: Vec<f64> = (0..n).map(|i| ((i as u64 + seed) as f64).cos()).collect();
            let (exp_re, exp_im) = rustfft_reference_f64(&re, &im, Direction::Forward);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f64_bluestein_with_planner_and_opts(
                &mut got_re,
                &mut got_im,
                Direction::Forward,
                &planner,
                &opts,
                &mut scratch_re,
                &mut scratch_im,
            );

            for k in 0..n {
                assert_close_f64(got_re[k], exp_re[k], 1e-8);
                assert_close_f64(got_im[k], exp_im[k], 1e-8);
            }
        }
    }
}
