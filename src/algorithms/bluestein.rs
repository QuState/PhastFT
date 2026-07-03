//! Bluestein FFT for arbitrary-length complex signals.
//!
//! The transform is reduced to a length-`M` convolution, where
//! `M = next_power_of_two(2N - 1)`, and the convolution is evaluated with the
//! existing power-of-two DIT FFT.

use fearless_simd::{dispatch, f32x8, f64x4, Simd, SimdBase};

use crate::algorithms::dit::{
    fft_f32_dit_with_planner_and_opts_impl, fft_f64_dit_with_planner_and_opts_impl,
};
use crate::options::Options;
use crate::planner::{Direction, PlannerBluestein32, PlannerBluestein64};

// Pre-multiply by the chirp and zero-pad to the convolution length.
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
            conj_sign: $T,
        ) {
            const LANES: usize = $lanes;
            let n = signal_re.len();
            let m = out_re.len();
            let sign_v = $V::splat(simd, conj_sign);

            let n_blocks = n / LANES;
            for blk in 0..n_blocks {
                let off = blk * LANES;
                let xr = $V::from_slice(simd, &signal_re[off..off + LANES]);
                let xi = sign_v * $V::from_slice(simd, &signal_im[off..off + LANES]);
                let cr = $V::from_slice(simd, &c_re[off..off + LANES]);
                let ci = $V::from_slice(simd, &c_im[off..off + LANES]);
                (xr * cr - xi * ci).store_slice(&mut out_re[off..off + LANES]);
                (xr * ci + xi * cr).store_slice(&mut out_im[off..off + LANES]);
            }
            for k in (n_blocks * LANES)..n {
                let xr = signal_re[k];
                let xi = conj_sign * signal_im[k];
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

// Pointwise multiply in place by the filter spectrum.
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

// Post-multiply by the chirp and copy the first N samples back to the signal.
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
            scale_re: $T,
            scale_im: $T,
        ) {
            const LANES: usize = $lanes;
            let n = out_re.len();
            let sre = $V::splat(simd, scale_re);
            let sim = $V::splat(simd, scale_im);

            let n_blocks = n / LANES;
            for blk in 0..n_blocks {
                let off = blk * LANES;
                let vr = $V::from_slice(simd, &conv_re[off..off + LANES]);
                let vi = $V::from_slice(simd, &conv_im[off..off + LANES]);
                let cr = $V::from_slice(simd, &c_re[off..off + LANES]);
                let ci = $V::from_slice(simd, &c_im[off..off + LANES]);
                let pr = cr * vr - ci * vi;
                let pi = cr * vi + ci * vr;
                (sre * pr).store_slice(&mut out_re[off..off + LANES]);
                (sim * pi).store_slice(&mut out_im[off..off + LANES]);
            }
            for k in (n_blocks * LANES)..n {
                let vr = conv_re[k];
                let vi = conv_im[k];
                let cr = c_re[k];
                let ci = c_im[k];
                let pr = cr * vr - ci * vi;
                let pi = cr * vi + ci * vr;
                out_re[k] = scale_re * pr;
                out_im[k] = scale_im * pi;
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
fn direction_params_f64(direction: Direction, n: usize) -> (f64, f64, f64) {
    match direction {
        Direction::Forward => (1.0, 1.0, 1.0),
        Direction::Inverse => {
            let inv = 1.0 / n as f64;
            (-1.0, inv, -inv)
        }
    }
}

#[inline]
fn direction_params_f32(direction: Direction, n: usize) -> (f32, f32, f32) {
    match direction {
        Direction::Forward => (1.0, 1.0, 1.0),
        Direction::Inverse => {
            let inv = 1.0 / n as f32;
            (-1.0, inv, -inv)
        }
    }
}

/// Bluestein FFT for `f64`, arbitrary length `N`, reusing a precomputed planner
/// and caller-owned scratch.
///
/// In-place over the length-`N` `reals` / `imags`. `scratch_re` and `scratch_im`
/// must each be length `M = planner.inner_fft_len()`. Their contents on entry are
/// ignored and on exit unspecified, so reuse them across calls. `opts` govern the
/// inner size-`M` FFT, so a hand-built `Options` must be sized for `M`, not `N`.
///
/// # Panics
///
/// Panics if `reals`/`imags` are not length `N`, or if either scratch slice is
/// not length `M`.
///
/// # Example
///
/// ```
/// use phastft::{fft_f64_bluestein_with_planner_and_opts, options::Options};
/// use phastft::planner::{Direction, PlannerBluestein64};
///
/// let n = 6; // not a power of two
/// let planner = PlannerBluestein64::new(n);
/// let m = planner.inner_fft_len();
/// let opts = Options::guess_options(m);
///
/// let mut reals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
/// let mut imags = vec![0.0; n];
/// let mut scratch_re = vec![0.0; m];
/// let mut scratch_im = vec![0.0; m];
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
    let n = planner.n;
    let m = planner.m;
    assert_eq!(reals.len(), n, "reals length must match planner size N");
    assert_eq!(imags.len(), n, "imags length must match planner size N");
    assert_eq!(
        scratch_re.len(),
        m,
        "scratch_re must have length inner_fft_len()"
    );
    assert_eq!(
        scratch_im.len(),
        m,
        "scratch_im must have length inner_fft_len()"
    );

    // Inverse transforms use (1/N) * conj(DFT(conj(x))) so the planner's
    // forward filter spectrum can be reused.
    let (conj_sign, scale_re, scale_im) = direction_params_f64(direction, n);

    dispatch!(planner.dit_planner.simd_level, simd => {
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_premul_f64(
                simd, reals, imags, &planner.c_re, &planner.c_im,
                scratch_re, scratch_im, conj_sign,
            ),
        );
        fft_f64_dit_with_planner_and_opts_impl(
            simd, scratch_re, scratch_im, Direction::Forward, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_pointwise_f64(
                simd, scratch_re, scratch_im, &planner.b_re, &planner.b_im,
            ),
        );
        fft_f64_dit_with_planner_and_opts_impl(
            simd, scratch_re, scratch_im, Direction::Inverse, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_postmul_f64(
                simd, scratch_re, scratch_im, &planner.c_re, &planner.c_im,
                reals, imags, scale_re, scale_im,
            ),
        );
    });
}

/// Bluestein FFT for `f64` reusing a planner. Allocates scratch and guesses
/// `Options` for the inner FFT.
///
/// # Panics
///
/// Panics if `reals`/`imags` are not length `N` (the planner size).
pub fn fft_f64_bluestein_with_planner(
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerBluestein64,
) {
    let opts = Options::guess_options(planner.m);
    let mut scratch_re = vec![0.0; planner.m];
    let mut scratch_im = vec![0.0; planner.m];
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

/// Bluestein FFT for `f64`, arbitrary length, building a planner per call. For
/// repeated transforms of the same size, reuse a
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

/// Bluestein FFT for `f32`, arbitrary length `N`, reusing a precomputed planner
/// and caller-owned scratch.
///
/// Single-precision variant of [`fft_f64_bluestein_with_planner_and_opts`]. See
/// that function for the scratch/`Options` contract. `scratch_re` / `scratch_im`
/// must each be length `M = planner.inner_fft_len()`.
///
/// # Panics
///
/// Panics if `reals`/`imags` are not length `N`, or if either scratch slice is
/// not length `M`.
pub fn fft_f32_bluestein_with_planner_and_opts(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerBluestein32,
    opts: &Options,
    scratch_re: &mut [f32],
    scratch_im: &mut [f32],
) {
    let n = planner.n;
    let m = planner.m;
    assert_eq!(reals.len(), n, "reals length must match planner size N");
    assert_eq!(imags.len(), n, "imags length must match planner size N");
    assert_eq!(
        scratch_re.len(),
        m,
        "scratch_re must have length inner_fft_len()"
    );
    assert_eq!(
        scratch_im.len(),
        m,
        "scratch_im must have length inner_fft_len()"
    );

    let (conj_sign, scale_re, scale_im) = direction_params_f32(direction, n);

    dispatch!(planner.dit_planner.simd_level, simd => {
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_premul_f32(
                simd, reals, imags, &planner.c_re, &planner.c_im,
                scratch_re, scratch_im, conj_sign,
            ),
        );
        fft_f32_dit_with_planner_and_opts_impl(
            simd, scratch_re, scratch_im, Direction::Forward, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_pointwise_f32(
                simd, scratch_re, scratch_im, &planner.b_re, &planner.b_im,
            ),
        );
        fft_f32_dit_with_planner_and_opts_impl(
            simd, scratch_re, scratch_im, Direction::Inverse, &planner.dit_planner, opts,
        );
        simd.vectorize(
            #[inline(always)]
            || simd_bluestein_postmul_f32(
                simd, scratch_re, scratch_im, &planner.c_re, &planner.c_im,
                reals, imags, scale_re, scale_im,
            ),
        );
    });
}

/// Bluestein FFT for `f32` reusing a planner. Allocates scratch and guesses
/// `Options` for the inner FFT.
///
/// # Panics
///
/// Panics if `reals`/`imags` are not length `N` (the planner size).
pub fn fft_f32_bluestein_with_planner(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerBluestein32,
) {
    let opts = Options::guess_options(planner.m);
    let mut scratch_re = vec![0.0; planner.m];
    let mut scratch_im = vec![0.0; planner.m];
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

/// Bluestein FFT for `f32`, arbitrary length, building a planner per call.
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
    use crate::planner::{Direction, PlannerBluestein64};

    fn assert_close_f64(actual: f64, expected: f64, rel: f64) {
        let denom = expected.abs().max(1.0);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err < rel,
            "rel_err {rel_err} >= {rel} (actual={actual}, expected={expected})"
        );
    }

    /// Reference DFT via rustfft. `inverse` selects the normalized (1/N) inverse,
    /// matching PhastFT's convention. Forward is unnormalized.
    fn rustfft_reference_f64(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
        use utilities::rustfft::num_complex::Complex;
        use utilities::rustfft::FftPlanner;

        let n = re.len();
        let mut buf: Vec<Complex<f64>> = re
            .iter()
            .zip(im)
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut planner = FftPlanner::new();
        let fft = if inverse {
            planner.plan_fft_inverse(n)
        } else {
            planner.plan_fft_forward(n)
        };
        fft.process(&mut buf);
        let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
        (
            buf.iter().map(|c| c.re * scale).collect(),
            buf.iter().map(|c| c.im * scale).collect(),
        )
    }

    // Sizes: primes (worst case), composites, powers of two, and small edge
    // sizes that exercise the scalar SIMD tails.
    const SIZES: &[usize] = &[
        1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 17, 31, 64, 100, 127, 1000, 1024,
    ];

    #[test]
    fn forward_vs_rustfft_f64() {
        for &n in SIZES {
            let re: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let im: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.5 - 1.0).collect();
            let (exp_re, exp_im) = rustfft_reference_f64(&re, &im, false);

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
            let (exp_re, exp_im) = rustfft_reference_f64(&re, &im, true);

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
        let opts = crate::options::Options::guess_options(planner.inner_fft_len());
        let m = planner.inner_fft_len();
        let mut re = vec![0.0f64; 8];
        let mut im = vec![0.0f64; 8];
        let mut sr = vec![0.0f64; m];
        let mut si = vec![0.0f64; m];
        fft_f64_bluestein_with_planner_and_opts(
            &mut re,
            &mut im,
            Direction::Forward,
            &planner,
            &opts,
            &mut sr,
            &mut si,
        );
    }

    #[test]
    #[should_panic(expected = "scratch_re must have length inner_fft_len()")]
    fn scratch_length_mismatch_panics_f64() {
        let planner = PlannerBluestein64::new(7);
        let opts = crate::options::Options::guess_options(planner.inner_fft_len());
        let m = planner.inner_fft_len();
        let mut re = vec![0.0f64; 7];
        let mut im = vec![0.0f64; 7];
        let mut sr = vec![0.0f64; m - 1];
        let mut si = vec![0.0f64; m];
        fft_f64_bluestein_with_planner_and_opts(
            &mut re,
            &mut im,
            Direction::Forward,
            &planner,
            &opts,
            &mut sr,
            &mut si,
        );
    }

    fn assert_close_f32(actual: f32, expected: f32, rel: f32) {
        let denom = expected.abs().max(1.0);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err < rel,
            "rel_err {rel_err} >= {rel} (actual={actual}, expected={expected})"
        );
    }

    fn rustfft_reference_f32(re: &[f32], im: &[f32], inverse: bool) -> (Vec<f32>, Vec<f32>) {
        use utilities::rustfft::num_complex::Complex;
        use utilities::rustfft::FftPlanner;

        let n = re.len();
        let mut buf: Vec<Complex<f32>> = re
            .iter()
            .zip(im)
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut planner = FftPlanner::new();
        let fft = if inverse {
            planner.plan_fft_inverse(n)
        } else {
            planner.plan_fft_forward(n)
        };
        fft.process(&mut buf);
        let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
        (
            buf.iter().map(|c| c.re * scale).collect(),
            buf.iter().map(|c| c.im * scale).collect(),
        )
    }

    const SIZES_F32: &[usize] = &[1, 2, 3, 5, 7, 9, 11, 13, 17, 31, 64, 127, 256];

    #[test]
    fn forward_vs_rustfft_f32() {
        for &n in SIZES_F32 {
            let re: Vec<f32> = (1..=n).map(|i| i as f32 * 0.1).collect();
            let im: Vec<f32> = (1..=n).map(|i| 0.5 - i as f32 * 0.05).collect();
            let (exp_re, exp_im) = rustfft_reference_f32(&re, &im, false);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f32_bluestein(&mut got_re, &mut got_im, Direction::Forward);

            for k in 0..n {
                assert_close_f32(got_re[k], exp_re[k], 1e-3);
                assert_close_f32(got_im[k], exp_im[k], 1e-3);
            }
        }
    }

    #[test]
    fn inverse_vs_rustfft_f32() {
        for &n in SIZES_F32 {
            let re: Vec<f32> = (1..=n).map(|i| (i as f32) * 0.25).collect();
            let im: Vec<f32> = (1..=n).map(|i| 1.0 - i as f32 * 0.125).collect();
            let (exp_re, exp_im) = rustfft_reference_f32(&re, &im, true);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f32_bluestein(&mut got_re, &mut got_im, Direction::Inverse);

            for k in 0..n {
                assert_close_f32(got_re[k], exp_re[k], 1e-3);
                assert_close_f32(got_im[k], exp_im[k], 1e-3);
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
                assert_close_f32(got_re[k], orig_re[k], 1e-3);
                assert_close_f32(got_im[k], orig_im[k], 1e-3);
            }
        }
    }

    #[test]
    fn tiers_agree_f64() {
        use crate::options::Options;
        use crate::planner::{Direction, PlannerBluestein64};

        let n = 100;
        let re: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let im: Vec<f64> = (1..=n).map(|i| -(i as f64)).collect();

        let mut bare_re = re.clone();
        let mut bare_im = im.clone();
        fft_f64_bluestein(&mut bare_re, &mut bare_im, Direction::Forward);

        let planner = PlannerBluestein64::new(n);
        let mut wp_re = re.clone();
        let mut wp_im = im.clone();
        fft_f64_bluestein_with_planner(&mut wp_re, &mut wp_im, Direction::Forward, &planner);

        let opts = Options::guess_options(planner.inner_fft_len());
        let m = planner.inner_fft_len();
        let mut full_re = re.clone();
        let mut full_im = im.clone();
        let mut sr = vec![0.0; m];
        let mut si = vec![0.0; m];
        fft_f64_bluestein_with_planner_and_opts(
            &mut full_re,
            &mut full_im,
            Direction::Forward,
            &planner,
            &opts,
            &mut sr,
            &mut si,
        );

        assert_eq!((bare_re, bare_im), (wp_re.clone(), wp_im.clone()));
        assert_eq!((wp_re, wp_im), (full_re, full_im));
    }

    #[test]
    fn scratch_reuse_across_calls_f64() {
        use crate::options::Options;
        use crate::planner::{Direction, PlannerBluestein64};

        let n = 127;
        let planner = PlannerBluestein64::new(n);
        let opts = Options::guess_options(planner.inner_fft_len());
        let m = planner.inner_fft_len();
        let mut sr = vec![0.0; m];
        let mut si = vec![0.0; m];

        for seed in 0..4u64 {
            let re: Vec<f64> = (0..n).map(|i| ((i as u64 + seed) as f64).sin()).collect();
            let im: Vec<f64> = (0..n).map(|i| ((i as u64 + seed) as f64).cos()).collect();
            let (exp_re, exp_im) = rustfft_reference_f64(&re, &im, false);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            fft_f64_bluestein_with_planner_and_opts(
                &mut got_re,
                &mut got_im,
                Direction::Forward,
                &planner,
                &opts,
                &mut sr,
                &mut si,
            );

            for k in 0..n {
                assert_close_f64(got_re[k], exp_re[k], 1e-8);
                assert_close_f64(got_im[k], exp_im[k], 1e-8);
            }
        }
    }
}
