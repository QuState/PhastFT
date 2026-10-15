//! Real-to-Complex FFT (R2C) and Complex-to-Real IFFT (C2R)
//!
//! Computes the FFT of a purely real-valued input signal using the
//! "pack-into-half-length-complex" trick:
//!
//! 1. Deinterleave N real values into even/odd halves (N/2 each)
//! 2. Treat (even, odd) as a complex signal and run a complex FFT of size N/2
//! 3. "Untangle" the result with a twiddle-factor post-processing step to
//!    recover the N/2+1 unique bins of the spectrum
//!
//! The C2R inverse mirrors the forward path: a per-bin pre-processing step,
//! a half-length complex IFFT in scratch, and a SIMD re-interleave back into
//! the caller's real output buffer.
//!
//! This halves the work compared to calling the complex FFT with a zeroed
//! imaginary array, and halves the output memory compared to materializing
//! the full N-point spectrum (which is conjugate-symmetric for real input).
//!
//! # Output layout (N/2+1 "compact")
//!
//! For real input the DFT satisfies `X[N-k] = conj(X[k])`, so only bins
//! `0..=N/2` are independent. R2C writes `output_re` and `output_im` of length
//! `N/2 + 1`. C2R consumes the same layout. This matches the FFTW R2C and
//! `realfft` conventions.
//!
//! # Allocation
//!
//! The zero-allocation guarantees below describe the top
//! [`r2c_fft_f64_with_planner_and_opts`] / [`c2r_fft_f64_with_planner_and_opts`]
//! tier, where the caller supplies the planner, options, and (for C2R) scratch.
//! The convenience tiers ([`r2c_fft_f64`] / [`c2r_fft_f64`] and the
//! `_with_planner` forms) build those automatically and therefore allocate.
//!
//! - **R2C** is in-place: the output buffers double as scratch for the inner
//!   half-length complex FFT, so the hot path performs zero allocations.
//! - **C2R** requires `N/2` reals of scratch per array (re + im). At the top
//!   tier the caller provides reusable scratch buffers, so the hot path
//!   allocates nothing.
//!
//! # References
//!
//! Based on the approach described by Levente Kovács:
//! <https://kovleventer.com/blog/fft_real/>

use fearless_simd::{dispatch, f32x8, f64x4, Simd, SimdBase, SimdFloat};

use crate::algorithms::dit::{
    fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts,
};
use crate::options::Options;
use crate::planner::{Direction, PlannerR2c32, PlannerR2c64};

// ---------------------------------------------------------------------------
// SIMD helpers
// ---------------------------------------------------------------------------

// Reverse the lane order of an f64x4. Compiles to a single permute/shuffle on
// AVX2 / NEON / SSE4.2 once `#[inline(always)]` lets LLVM see the compile-time
// indices.
#[inline(always)]
fn rev_f64x4<S: Simd>(simd: S, v: f64x4<S>) -> f64x4<S> {
    let arr = simd.as_array_f64x4(v);
    let rev = [arr[3], arr[2], arr[1], arr[0]];
    f64x4::from_slice(simd, &rev)
}

#[inline(always)]
fn rev_f32x8<S: Simd>(simd: S, v: f32x8<S>) -> f32x8<S> {
    let arr = simd.as_array_f32x8(v);
    let rev = [
        arr[7], arr[6], arr[5], arr[4], arr[3], arr[2], arr[1], arr[0],
    ];
    f32x8::from_slice(simd, &rev)
}

// ---------------------------------------------------------------------------
// SIMD deinterleave: input[2k] -> output_re[k], input[2k+1] -> output_im[k]
// ---------------------------------------------------------------------------

#[inline(always)] // required by fearless_simd
fn simd_deinterleave_f64<S: Simd>(
    simd: S,
    input: &[f64],
    output_re: &mut [f64],
    output_im: &mut [f64],
) {
    const LANES: usize = 4;
    let half = output_re.len();
    debug_assert_eq!(input.len(), 2 * half);
    debug_assert_eq!(output_im.len(), half);

    let n_blocks = half / LANES;
    for blk in 0..n_blocks {
        let in_off = blk * (2 * LANES);
        let out_off = blk * LANES;
        let a = f64x4::from_slice(simd, &input[in_off..in_off + LANES]);
        let b = f64x4::from_slice(simd, &input[in_off + LANES..in_off + 2 * LANES]);
        let (re, im) = a.deinterleave(b);
        re.store_slice(&mut output_re[out_off..out_off + LANES]);
        im.store_slice(&mut output_im[out_off..out_off + LANES]);
    }
    for k in (n_blocks * LANES)..half {
        output_re[k] = input[2 * k];
        output_im[k] = input[2 * k + 1];
    }
}

#[inline(always)] // required by fearless_simd
fn simd_deinterleave_f32<S: Simd>(
    simd: S,
    input: &[f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
) {
    const LANES: usize = 8;
    let half = output_re.len();
    debug_assert_eq!(input.len(), 2 * half);
    debug_assert_eq!(output_im.len(), half);

    let n_blocks = half / LANES;
    for blk in 0..n_blocks {
        let in_off = blk * (2 * LANES);
        let out_off = blk * LANES;
        let a = f32x8::from_slice(simd, &input[in_off..in_off + LANES]);
        let b = f32x8::from_slice(simd, &input[in_off + LANES..in_off + 2 * LANES]);
        let (re, im) = a.deinterleave(b);
        re.store_slice(&mut output_re[out_off..out_off + LANES]);
        im.store_slice(&mut output_im[out_off..out_off + LANES]);
    }
    for k in (n_blocks * LANES)..half {
        output_re[k] = input[2 * k];
        output_im[k] = input[2 * k + 1];
    }
}

// ---------------------------------------------------------------------------
// SIMD untangle: in-place R2C post-processing
// ---------------------------------------------------------------------------
//
// Per-bin math (planner's W^k already pre-multiplied by 0.5):
//   s_re = 0.5*(a + c),  s_im = 0.5*(b - d)
//   t_re = b + d,        t_im = c - a
//   wzr  = wkr_h*t_re - wki_h*t_im
//   wzi  = wkr_h*t_im + wki_h*t_re
//   X[k]      = (s_re + wzr,  s_im + wzi)
//   X[mirror] = (s_re - wzr, -s_im + wzi)
// where (a, b) = Z[k], (c, d) = Z[mirror], mirror = half - k.
//
// SIMD scheme: process LANES front k's at k_block. The corresponding mirror
// indices are descending from `half - k_block` down. Loading the contiguous
// memory window `[mirror_low, mirror_low + LANES)` (mirror_low = half - k_block
// - LANES + 1) gives a vector whose lane i corresponds to k_block + (LANES-1-i)
// - the reverse of the front lanes. We reverse once on load, do all math in
// the front-lane convention, then reverse-store the mirror writes.

macro_rules! impl_simd_untangle_inplace {
    ($name:ident, $T:ty, $V:ident, $rev:ident, $lanes:expr) => {
        #[inline(always)] // required by fearless_simd
        fn $name<S: Simd>(
            simd: S,
            output_re: &mut [$T],
            output_im: &mut [$T],
            w_re: &[$T],
            w_im: &[$T],
            half: usize,
        ) {
            let a0 = output_re[0];
            let b0 = output_im[0];
            output_re[0] = a0 + b0;
            output_im[0] = 0.0;
            output_re[half] = a0 - b0;
            output_im[half] = 0.0;

            let q = half / 2;
            const LANES: usize = $lanes;
            let half_v = $V::splat(simd, 0.5);

            let mut k = 1;
            // Process while a full LANES-block stays strictly below the self-pair at q.
            while k + LANES <= q {
                let mirror_low = half - k - (LANES - 1);

                let a = $V::from_slice(simd, &output_re[k..k + LANES]);
                let b = $V::from_slice(simd, &output_im[k..k + LANES]);
                let c_loaded = $V::from_slice(simd, &output_re[mirror_low..mirror_low + LANES]);
                let d_loaded = $V::from_slice(simd, &output_im[mirror_low..mirror_low + LANES]);
                let c = $rev(simd, c_loaded);
                let d = $rev(simd, d_loaded);

                let s_re = half_v * (a + c);
                let s_im = half_v * (b - d);
                let t_re = b + d;
                let t_im = c - a;

                let wkr_h = $V::from_slice(simd, &w_re[k..k + LANES]);
                let wki_h = $V::from_slice(simd, &w_im[k..k + LANES]);
                let wzr = wkr_h * t_re - wki_h * t_im;
                let wzi = wkr_h * t_im + wki_h * t_re;

                let out_re_k = s_re + wzr;
                let out_im_k = s_im + wzi;
                let out_re_m = s_re - wzr;
                let out_im_m = wzi - s_im;

                out_re_k.store_slice(&mut output_re[k..k + LANES]);
                out_im_k.store_slice(&mut output_im[k..k + LANES]);
                $rev(simd, out_re_m).store_slice(&mut output_re[mirror_low..mirror_low + LANES]);
                $rev(simd, out_im_m).store_slice(&mut output_im[mirror_low..mirror_low + LANES]);

                k += LANES;
            }

            // Scalar tail (k in [k, q)) — typically <LANES iterations.
            while k < q {
                let mirror = half - k;
                let a = output_re[k];
                let b = output_im[k];
                let c = output_re[mirror];
                let d = output_im[mirror];

                let s_re = 0.5 * (a + c);
                let s_im = 0.5 * (b - d);
                let t_re = b + d;
                let t_im = c - a;

                let wkr_h = w_re[k];
                let wki_h = w_im[k];
                let wzr = wkr_h * t_re - wki_h * t_im;
                let wzi = wkr_h * t_im + wki_h * t_re;

                output_re[k] = s_re + wzr;
                output_im[k] = s_im + wzi;
                output_re[mirror] = s_re - wzr;
                output_im[mirror] = wzi - s_im;
                k += 1;
            }

            // Self-pair at k = q (always exists since N >= 4 ⇒ q >= 1).
            let a = output_re[q];
            let b = output_im[q];
            output_re[q] = a + 2.0 * w_re[q] * b;
            output_im[q] = 2.0 * w_im[q] * b;
        }
    };
}

impl_simd_untangle_inplace!(simd_untangle_inplace_f64, f64, f64x4, rev_f64x4, 4);
impl_simd_untangle_inplace!(simd_untangle_inplace_f32, f32, f32x8, rev_f32x8, 8);

// ---------------------------------------------------------------------------
// SIMD c2r preprocess
// ---------------------------------------------------------------------------
//
// Per-bin math (planner's 0.5*W^k):
//   re_first = input_re[k],          im_first = input_im[k]
//   re_second = input_re[mirror],    im_second = -input_im[mirror]   (conjugate)
//   zx_re = 0.5*(re_first + re_second),  zx_im = 0.5*(im_first + im_second)
//   dr = re_first - re_second,           di = im_first - im_second
//   zy_re =  c_h*dr + s_h*di
//   zy_im =  c_h*di - s_h*dr
//   z_re[k] = zx_re - zy_im
//   z_im[k] = zx_im + zy_re
//
// SIMD scheme: process LANES front k's at k_block. Mirror values at descending
// indices; load contiguous mirror window then reverse. k=0 (mirror = half,
// which is the Nyquist slot) is handled scalar-first.

#[inline(always)] // required by fearless_simd
fn simd_c2r_preprocess_f64<S: Simd>(
    simd: S,
    input_re: &[f64],
    input_im: &[f64],
    w_re: &[f64],
    w_im: &[f64],
    z_re: &mut [f64],
    z_im: &mut [f64],
) {
    let half = z_re.len();
    debug_assert_eq!(z_im.len(), half);
    debug_assert_eq!(input_re.len(), half + 1);
    debug_assert_eq!(input_im.len(), half + 1);

    // k = 0 special case: mirror = half (Nyquist bin), handled scalar.
    {
        let re_first = input_re[0];
        let im_first = input_im[0];
        let re_second = input_re[half];
        let im_second = -input_im[half];
        let zx_re = 0.5 * (re_first + re_second);
        let zx_im = 0.5 * (im_first + im_second);
        let dr = re_first - re_second;
        let di = im_first - im_second;
        let c_h = w_re[0];
        let s_h = w_im[0];
        let zy_re = c_h * dr + s_h * di;
        let zy_im = c_h * di - s_h * dr;
        z_re[0] = zx_re - zy_im;
        z_im[0] = zx_im + zy_re;
    }

    const LANES: usize = 4;
    let half_v = f64x4::splat(simd, 0.5);

    let mut k = 1;
    while k + LANES <= half {
        let mirror_low = half - k - (LANES - 1);

        let re_f = f64x4::from_slice(simd, &input_re[k..k + LANES]);
        let im_f = f64x4::from_slice(simd, &input_im[k..k + LANES]);
        let re_s_loaded = f64x4::from_slice(simd, &input_re[mirror_low..mirror_low + LANES]);
        let im_s_loaded = f64x4::from_slice(simd, &input_im[mirror_low..mirror_low + LANES]);
        // Reverse to align with front-lane convention; im is also negated for conjugate.
        let re_s = rev_f64x4(simd, re_s_loaded);
        let im_s = -rev_f64x4(simd, im_s_loaded);

        let zx_re = half_v * (re_f + re_s);
        let zx_im = half_v * (im_f + im_s);
        let dr = re_f - re_s;
        let di = im_f - im_s;

        let c_h = f64x4::from_slice(simd, &w_re[k..k + LANES]);
        let s_h = f64x4::from_slice(simd, &w_im[k..k + LANES]);
        let zy_re = c_h * dr + s_h * di;
        let zy_im = c_h * di - s_h * dr;

        (zx_re - zy_im).store_slice(&mut z_re[k..k + LANES]);
        (zx_im + zy_re).store_slice(&mut z_im[k..k + LANES]);

        k += LANES;
    }

    while k < half {
        let mirror = half - k;
        let re_first = input_re[k];
        let im_first = input_im[k];
        let re_second = input_re[mirror];
        let im_second = -input_im[mirror];

        let zx_re = 0.5 * (re_first + re_second);
        let zx_im = 0.5 * (im_first + im_second);
        let dr = re_first - re_second;
        let di = im_first - im_second;

        let c_h = w_re[k];
        let s_h = w_im[k];
        let zy_re = c_h * dr + s_h * di;
        let zy_im = c_h * di - s_h * dr;

        z_re[k] = zx_re - zy_im;
        z_im[k] = zx_im + zy_re;
        k += 1;
    }
}

#[inline(always)] // required by fearless_simd
fn simd_c2r_preprocess_f32<S: Simd>(
    simd: S,
    input_re: &[f32],
    input_im: &[f32],
    w_re: &[f32],
    w_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
) {
    let half = z_re.len();
    debug_assert_eq!(z_im.len(), half);
    debug_assert_eq!(input_re.len(), half + 1);
    debug_assert_eq!(input_im.len(), half + 1);

    {
        let re_first = input_re[0];
        let im_first = input_im[0];
        let re_second = input_re[half];
        let im_second = -input_im[half];
        let zx_re = 0.5 * (re_first + re_second);
        let zx_im = 0.5 * (im_first + im_second);
        let dr = re_first - re_second;
        let di = im_first - im_second;
        let c_h = w_re[0];
        let s_h = w_im[0];
        let zy_re = c_h * dr + s_h * di;
        let zy_im = c_h * di - s_h * dr;
        z_re[0] = zx_re - zy_im;
        z_im[0] = zx_im + zy_re;
    }

    const LANES: usize = 8;
    let half_v = f32x8::splat(simd, 0.5);

    let mut k = 1;
    while k + LANES <= half {
        let mirror_low = half - k - (LANES - 1);

        let re_f = f32x8::from_slice(simd, &input_re[k..k + LANES]);
        let im_f = f32x8::from_slice(simd, &input_im[k..k + LANES]);
        let re_s_loaded = f32x8::from_slice(simd, &input_re[mirror_low..mirror_low + LANES]);
        let im_s_loaded = f32x8::from_slice(simd, &input_im[mirror_low..mirror_low + LANES]);
        let re_s = rev_f32x8(simd, re_s_loaded);
        let im_s = -rev_f32x8(simd, im_s_loaded);

        let zx_re = half_v * (re_f + re_s);
        let zx_im = half_v * (im_f + im_s);
        let dr = re_f - re_s;
        let di = im_f - im_s;

        let c_h = f32x8::from_slice(simd, &w_re[k..k + LANES]);
        let s_h = f32x8::from_slice(simd, &w_im[k..k + LANES]);
        let zy_re = c_h * dr + s_h * di;
        let zy_im = c_h * di - s_h * dr;

        (zx_re - zy_im).store_slice(&mut z_re[k..k + LANES]);
        (zx_im + zy_re).store_slice(&mut z_im[k..k + LANES]);

        k += LANES;
    }

    while k < half {
        let mirror = half - k;
        let re_first = input_re[k];
        let im_first = input_im[k];
        let re_second = input_re[mirror];
        let im_second = -input_im[mirror];

        let zx_re = 0.5 * (re_first + re_second);
        let zx_im = 0.5 * (im_first + im_second);
        let dr = re_first - re_second;
        let di = im_first - im_second;

        let c_h = w_re[k];
        let s_h = w_im[k];
        let zy_re = c_h * dr + s_h * di;
        let zy_im = c_h * di - s_h * dr;

        z_re[k] = zx_re - zy_im;
        z_im[k] = zx_im + zy_re;
        k += 1;
    }
}

// ---------------------------------------------------------------------------
// SIMD interleave: z_re[k] -> output[2k], z_im[k] -> output[2k+1]
// ---------------------------------------------------------------------------
//
// Inverse of `simd_deinterleave_*`. Reads LANES contiguous reals from each of
// z_re / z_im and stores 2*LANES interleaved reals to output.

#[inline(always)] // required by fearless_simd
fn simd_interleave_f64<S: Simd>(simd: S, z_re: &[f64], z_im: &[f64], output: &mut [f64]) {
    const LANES: usize = 4;
    let half = z_re.len();
    debug_assert_eq!(z_im.len(), half);
    debug_assert_eq!(output.len(), 2 * half);

    let n_blocks = half / LANES;
    for blk in 0..n_blocks {
        let in_off = blk * LANES;
        let out_off = blk * (2 * LANES);
        let re = f64x4::from_slice(simd, &z_re[in_off..in_off + LANES]);
        let im = f64x4::from_slice(simd, &z_im[in_off..in_off + LANES]);
        let (lo, hi) = re.interleave(im);
        lo.store_slice(&mut output[out_off..out_off + LANES]);
        hi.store_slice(&mut output[out_off + LANES..out_off + 2 * LANES]);
    }
    for k in (n_blocks * LANES)..half {
        output[2 * k] = z_re[k];
        output[2 * k + 1] = z_im[k];
    }
}

#[inline(always)] // required by fearless_simd
fn simd_interleave_f32<S: Simd>(simd: S, z_re: &[f32], z_im: &[f32], output: &mut [f32]) {
    const LANES: usize = 8;
    let half = z_re.len();
    debug_assert_eq!(z_im.len(), half);
    debug_assert_eq!(output.len(), 2 * half);

    let n_blocks = half / LANES;
    for blk in 0..n_blocks {
        let in_off = blk * LANES;
        let out_off = blk * (2 * LANES);
        let re = f32x8::from_slice(simd, &z_re[in_off..in_off + LANES]);
        let im = f32x8::from_slice(simd, &z_im[in_off..in_off + LANES]);
        let (lo, hi) = re.interleave(im);
        lo.store_slice(&mut output[out_off..out_off + LANES]);
        hi.store_slice(&mut output[out_off + LANES..out_off + 2 * LANES]);
    }
    for k in (n_blocks * LANES)..half {
        output[2 * k] = z_re[k];
        output[2 * k + 1] = z_im[k];
    }
}

// ===========================================================================
// R2C: real (length N) → complex (length N/2 + 1)
// ===========================================================================

/// Performs a real-valued FFT on `f64` input data.
///
/// Computes the forward FFT of `input_re` (length `N`, real) and writes the
/// `N/2 + 1` independent complex bins into `output_re` / `output_im`. The
/// remaining `N/2 - 1` bins are determined by conjugate symmetry:
/// `X[N - k] = conj(X[k])`.
///
/// In-place: the output buffers double as scratch for the inner half-length
/// complex FFT, so this function performs zero allocations.
///
/// The `opts` argument governs the internal half-length (`N/2`) complex FFT, so
/// a hand-built `Options` must be sized for `N / 2`, not `N`.
///
/// # Panics
///
/// Panics if `input_re.len()` does not match the planner size, or if
/// `output_re` / `output_im` are not length `N / 2 + 1`.
///
/// # Example
///
/// ```
/// use phastft::{options::Options, planner::PlannerR2c64, r2c_fft_f64_with_planner_and_opts};
///
/// let n = 16;
/// let input: Vec<f64> = (1..=n).map(|x| x as f64).collect();
/// let mut out_re = vec![0.0; n / 2 + 1];
/// let mut out_im = vec![0.0; n / 2 + 1];
///
/// let planner = PlannerR2c64::new(n);
/// let opts = Options::guess_options(n / 2);
/// r2c_fft_f64_with_planner_and_opts(&input, &mut out_re, &mut out_im, &planner, &opts);
/// ```
pub fn r2c_fft_f64_with_planner_and_opts(
    input_re: &[f64],
    output_re: &mut [f64],
    output_im: &mut [f64],
    planner: &PlannerR2c64,
    opts: &Options,
) {
    let n = planner.n;
    let half = n / 2;
    assert_eq!(input_re.len(), n, "input length must match planner size");
    assert_eq!(
        output_re.len(),
        half + 1,
        "output_re must have length N/2 + 1"
    );
    assert_eq!(
        output_im.len(),
        half + 1,
        "output_im must have length N/2 + 1"
    );

    // Deinterleave input into the first `half` slots of output_re/output_im.
    // The Nyquist slot at index `half` stays untouched until the untangle.
    {
        let (z_re, _) = output_re.split_at_mut(half);
        let (z_im, _) = output_im.split_at_mut(half);
        dispatch!(
            planner.dit_planner.simd_level,
            simd => {
                simd.vectorize(
                    #[inline(always)]
                    || simd_deinterleave_f64(simd, input_re, z_re, z_im),
                );
            }
        );
    }

    // Inner half-length complex FFT in place on the deinterleaved data.
    {
        let z_re = &mut output_re[..half];
        let z_im = &mut output_im[..half];
        fft_f64_dit_with_planner_and_opts(
            z_re,
            z_im,
            Direction::Forward,
            &planner.dit_planner,
            opts,
        );
    }

    dispatch!(
        planner.dit_planner.simd_level,
        simd => {
            simd.vectorize(
                #[inline(always)]
                || simd_untangle_inplace_f64(simd, output_re, output_im, &planner.w_re, &planner.w_im, half),
            );
        }
    );
}

/// Performs a real-valued FFT on `f32` input data.
///
/// Single-precision variant of [`r2c_fft_f64_with_planner_and_opts`]; see that function for the
/// layout contract and example.
///
/// The `opts` argument governs the internal half-length (`N/2`) complex FFT, so
/// a hand-built `Options` must be sized for `N / 2`, not `N`.
pub fn r2c_fft_f32_with_planner_and_opts(
    input_re: &[f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
    planner: &PlannerR2c32,
    opts: &Options,
) {
    let n = planner.n;
    let half = n / 2;
    assert_eq!(input_re.len(), n, "input length must match planner size");
    assert_eq!(
        output_re.len(),
        half + 1,
        "output_re must have length N/2 + 1"
    );
    assert_eq!(
        output_im.len(),
        half + 1,
        "output_im must have length N/2 + 1"
    );

    {
        let (z_re, _) = output_re.split_at_mut(half);
        let (z_im, _) = output_im.split_at_mut(half);
        dispatch!(
            planner.dit_planner.simd_level,
            simd => {
                simd.vectorize(
                    #[inline(always)]
                    || simd_deinterleave_f32(simd, input_re, z_re, z_im),
                );
            }
        );
    }

    {
        let z_re = &mut output_re[..half];
        let z_im = &mut output_im[..half];
        fft_f32_dit_with_planner_and_opts(
            z_re,
            z_im,
            Direction::Forward,
            &planner.dit_planner,
            opts,
        );
    }

    dispatch!(
        planner.dit_planner.simd_level,
        simd => {
            simd.vectorize(
                #[inline(always)]
                || simd_untangle_inplace_f32(simd, output_re, output_im, &planner.w_re, &planner.w_im, half),
            );
        }
    );
}

// ===========================================================================
// C2R: complex (length N/2 + 1) → real (length N)
// ===========================================================================

/// Performs the inverse real-valued FFT on `f64` data.
///
/// Given the `N/2 + 1` independent complex bins produced by [`r2c_fft_f64_with_planner_and_opts`]
/// (conjugate-symmetric redundancy stripped), recovers the original `N` real
/// samples. Performs zero allocations: `scratch_re` and `scratch_im` must
/// each be length `N / 2`; their contents on entry are ignored and on exit
/// are unspecified (callers may reuse the buffers across calls).
///
/// The `opts` argument governs the internal half-length (`N/2`) complex FFT, so
/// a hand-built `Options` must be sized for `N / 2`, not `N`.
///
/// # Panics
///
/// Panics if `output.len()` does not match the planner size, if `input_re` /
/// `input_im` are not length `N / 2 + 1`, or if either scratch slice is not
/// length `N / 2`.
///
/// # Example
///
/// ```
/// use phastft::{
///     c2r_fft_f64_with_planner_and_opts, options::Options, planner::PlannerR2c64,
///     r2c_fft_f64_with_planner_and_opts,
/// };
///
/// let n = 16;
/// let signal: Vec<f64> = (1..=n).map(|x| x as f64).collect();
///
/// let planner = PlannerR2c64::new(n);
/// let opts = Options::guess_options(n / 2);
///
/// let mut spec_re = vec![0.0; n / 2 + 1];
/// let mut spec_im = vec![0.0; n / 2 + 1];
/// r2c_fft_f64_with_planner_and_opts(&signal, &mut spec_re, &mut spec_im, &planner, &opts);
///
/// let mut scratch_re = vec![0.0; n / 2];
/// let mut scratch_im = vec![0.0; n / 2];
/// let mut recovered = vec![0.0; n];
/// c2r_fft_f64_with_planner_and_opts(
///     &spec_re,
///     &spec_im,
///     &mut recovered,
///     &planner,
///     &opts,
///     &mut scratch_re,
///     &mut scratch_im,
/// );
/// ```
pub fn c2r_fft_f64_with_planner_and_opts(
    input_re: &[f64],
    input_im: &[f64],
    output: &mut [f64],
    planner: &PlannerR2c64,
    opts: &Options,
    scratch_re: &mut [f64],
    scratch_im: &mut [f64],
) {
    let n = planner.n;
    let half = n / 2;
    assert_eq!(output.len(), n, "output length must match planner size");
    assert_eq!(
        input_re.len(),
        half + 1,
        "input_re must have length N/2 + 1"
    );
    assert_eq!(
        input_im.len(),
        half + 1,
        "input_im must have length N/2 + 1"
    );
    assert_eq!(scratch_re.len(), half, "scratch_re must have length N/2");
    assert_eq!(scratch_im.len(), half, "scratch_im must have length N/2");

    dispatch!(
        planner.dit_planner.simd_level,
        simd => {
            simd.vectorize(
                #[inline(always)]
                || simd_c2r_preprocess_f64(
                    simd,
                    input_re,
                    input_im,
                    &planner.w_re,
                    &planner.w_im,
                    scratch_re,
                    scratch_im,
                ),
            );
        }
    );

    fft_f64_dit_with_planner_and_opts(
        scratch_re,
        scratch_im,
        Direction::Inverse,
        &planner.dit_planner,
        opts,
    );

    dispatch!(
        planner.dit_planner.simd_level,
        simd => {
            simd.vectorize(
                #[inline(always)]
                || simd_interleave_f64(simd, scratch_re, scratch_im, output),
            );
        }
    );
}

/// Performs the inverse real-valued FFT on `f32` data.
///
/// Single-precision variant of [`c2r_fft_f64_with_planner_and_opts`]; see that function for details.
///
/// The `opts` argument governs the internal half-length (`N/2`) complex FFT, so
/// a hand-built `Options` must be sized for `N / 2`, not `N`.
pub fn c2r_fft_f32_with_planner_and_opts(
    input_re: &[f32],
    input_im: &[f32],
    output: &mut [f32],
    planner: &PlannerR2c32,
    opts: &Options,
    scratch_re: &mut [f32],
    scratch_im: &mut [f32],
) {
    let n = planner.n;
    let half = n / 2;
    assert_eq!(output.len(), n, "output length must match planner size");
    assert_eq!(
        input_re.len(),
        half + 1,
        "input_re must have length N/2 + 1"
    );
    assert_eq!(
        input_im.len(),
        half + 1,
        "input_im must have length N/2 + 1"
    );
    assert_eq!(scratch_re.len(), half, "scratch_re must have length N/2");
    assert_eq!(scratch_im.len(), half, "scratch_im must have length N/2");

    dispatch!(
        planner.dit_planner.simd_level,
        simd => {
            simd.vectorize(
                #[inline(always)]
                || simd_c2r_preprocess_f32(
                    simd,
                    input_re,
                    input_im,
                    &planner.w_re,
                    &planner.w_im,
                    scratch_re,
                    scratch_im,
                ),
            );
        }
    );

    fft_f32_dit_with_planner_and_opts(
        scratch_re,
        scratch_im,
        Direction::Inverse,
        &planner.dit_planner,
        opts,
    );

    dispatch!(
        planner.dit_planner.simd_level,
        simd => {
            simd.vectorize(
                #[inline(always)]
                || simd_interleave_f32(simd, scratch_re, scratch_im, output),
            );
        }
    );
}

// ---------------------------------------------------------------------------
// Convenience tiers: bare (auto planner) and `_with_planner` (auto options),
// mirroring the three-tier shape of the complex FFT API.
// ---------------------------------------------------------------------------

/// Real-valued FFT on `f64` input — convenience wrapper that builds a planner
/// automatically.
///
/// For repeated transforms of the same size, reuse a planner via
/// [`r2c_fft_f64_with_planner`]. See [`r2c_fft_f64_with_planner_and_opts`] for
/// the output-layout contract.
///
/// # Panics
///
/// Panics if `input_re.len()` is not a power of two `>= 4`, or if `output_re` /
/// `output_im` are not length `input_re.len() / 2 + 1`.
///
/// # Example
///
/// ```
/// use phastft::r2c_fft_f64;
///
/// let n = 16;
/// let input: Vec<f64> = (1..=n).map(|x| x as f64).collect();
/// let mut out_re = vec![0.0; n / 2 + 1];
/// let mut out_im = vec![0.0; n / 2 + 1];
/// r2c_fft_f64(&input, &mut out_re, &mut out_im);
/// ```
pub fn r2c_fft_f64(input_re: &[f64], output_re: &mut [f64], output_im: &mut [f64]) {
    let planner = PlannerR2c64::new(input_re.len());
    r2c_fft_f64_with_planner(input_re, output_re, output_im, &planner);
}

/// Real-valued FFT on `f64` input using a pre-built planner; options are
/// guessed automatically.
///
/// See [`r2c_fft_f64_with_planner_and_opts`] for the output-layout contract.
///
/// # Panics
///
/// Panics if the buffer lengths do not match the planner size.
pub fn r2c_fft_f64_with_planner(
    input_re: &[f64],
    output_re: &mut [f64],
    output_im: &mut [f64],
    planner: &PlannerR2c64,
) {
    let opts = Options::guess_options(planner.n / 2);
    r2c_fft_f64_with_planner_and_opts(input_re, output_re, output_im, planner, &opts);
}

/// Inverse real-valued FFT on `f64` data — convenience wrapper that builds a
/// planner and allocates scratch automatically.
///
/// For repeated transforms, reuse a planner and scratch via
/// [`c2r_fft_f64_with_planner_and_opts`]. See it for the input-layout contract.
///
/// # Panics
///
/// Panics if `output.len()` is not a power of two `>= 4`, or if `input_re` /
/// `input_im` are not length `output.len() / 2 + 1`.
///
/// # Example
///
/// ```
/// use phastft::{c2r_fft_f64, r2c_fft_f64};
///
/// let n = 16;
/// let signal: Vec<f64> = (1..=n).map(|x| x as f64).collect();
///
/// let mut spec_re = vec![0.0; n / 2 + 1];
/// let mut spec_im = vec![0.0; n / 2 + 1];
/// r2c_fft_f64(&signal, &mut spec_re, &mut spec_im);
///
/// let mut recovered = vec![0.0; n];
/// c2r_fft_f64(&spec_re, &spec_im, &mut recovered);
/// ```
pub fn c2r_fft_f64(input_re: &[f64], input_im: &[f64], output: &mut [f64]) {
    let planner = PlannerR2c64::new(output.len());
    c2r_fft_f64_with_planner(input_re, input_im, output, &planner);
}

/// Inverse real-valued FFT on `f64` data using a pre-built planner; options are
/// guessed and scratch is allocated automatically.
///
/// # Panics
///
/// Panics if the buffer lengths do not match the planner size.
pub fn c2r_fft_f64_with_planner(
    input_re: &[f64],
    input_im: &[f64],
    output: &mut [f64],
    planner: &PlannerR2c64,
) {
    let opts = Options::guess_options(planner.n / 2);
    let mut scratch_re = vec![0.0; planner.n / 2];
    let mut scratch_im = vec![0.0; planner.n / 2];
    c2r_fft_f64_with_planner_and_opts(
        input_re,
        input_im,
        output,
        planner,
        &opts,
        &mut scratch_re,
        &mut scratch_im,
    );
}

/// Real-valued FFT on `f32` input — convenience wrapper that builds a planner
/// automatically.
///
/// Single-precision variant of [`r2c_fft_f64`]. See
/// [`r2c_fft_f32_with_planner_and_opts`] for the output-layout contract.
///
/// # Panics
///
/// Panics if `input_re.len()` is not a power of two `>= 4`, or if `output_re` /
/// `output_im` are not length `input_re.len() / 2 + 1`.
pub fn r2c_fft_f32(input_re: &[f32], output_re: &mut [f32], output_im: &mut [f32]) {
    let planner = PlannerR2c32::new(input_re.len());
    r2c_fft_f32_with_planner(input_re, output_re, output_im, &planner);
}

/// Real-valued FFT on `f32` input using a pre-built planner; options are
/// guessed automatically.
///
/// See [`r2c_fft_f32_with_planner_and_opts`] for the output-layout contract.
///
/// # Panics
///
/// Panics if the buffer lengths do not match the planner size.
pub fn r2c_fft_f32_with_planner(
    input_re: &[f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
    planner: &PlannerR2c32,
) {
    let opts = Options::guess_options(planner.n / 2);
    r2c_fft_f32_with_planner_and_opts(input_re, output_re, output_im, planner, &opts);
}

/// Inverse real-valued FFT on `f32` data — convenience wrapper that builds a
/// planner and allocates scratch automatically.
///
/// Single-precision variant of [`c2r_fft_f64`]. See
/// [`c2r_fft_f32_with_planner_and_opts`] for the input-layout contract.
///
/// # Panics
///
/// Panics if `output.len()` is not a power of two `>= 4`, or if `input_re` /
/// `input_im` are not length `output.len() / 2 + 1`.
pub fn c2r_fft_f32(input_re: &[f32], input_im: &[f32], output: &mut [f32]) {
    let planner = PlannerR2c32::new(output.len());
    c2r_fft_f32_with_planner(input_re, input_im, output, &planner);
}

/// Inverse real-valued FFT on `f32` data using a pre-built planner; options are
/// guessed and scratch is allocated automatically.
///
/// # Panics
///
/// Panics if the buffer lengths do not match the planner size.
pub fn c2r_fft_f32_with_planner(
    input_re: &[f32],
    input_im: &[f32],
    output: &mut [f32],
    planner: &PlannerR2c32,
) {
    let opts = Options::guess_options(planner.n / 2);
    let mut scratch_re = vec![0.0f32; planner.n / 2];
    let mut scratch_im = vec![0.0f32; planner.n / 2];
    c2r_fft_f32_with_planner_and_opts(
        input_re,
        input_im,
        output,
        planner,
        &opts,
        &mut scratch_re,
        &mut scratch_im,
    );
}

#[cfg(test)]
mod tests {
    use utilities::{assert_float_closeness, gen_random_signal_f32, gen_random_signal_f64};

    use super::*;
    use crate::planner::Direction;
    use crate::{fft_f32_dit, fft_f64_dit};

    fn assert_f32_relative_closeness(actual: f32, expected: f32, rel_eps: f32) {
        let denom = expected.abs().max(f32::EPSILON);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err < rel_eps,
            "relative error {rel_err} >= {rel_eps} (actual={actual}, expected={expected})"
        );
    }

    // Drive the public R2C/C2R API end-to-end (planner + opts + scratch) so
    // each test stays focused on its assertion rather than plumbing.
    fn run_r2c_f64(input: &[f64], out_re: &mut [f64], out_im: &mut [f64]) {
        let planner = PlannerR2c64::new(input.len());
        let opts = Options::guess_options(input.len() / 2);
        r2c_fft_f64_with_planner_and_opts(input, out_re, out_im, &planner, &opts);
    }

    fn run_r2c_f32(input: &[f32], out_re: &mut [f32], out_im: &mut [f32]) {
        let planner = PlannerR2c32::new(input.len());
        let opts = Options::guess_options(input.len() / 2);
        r2c_fft_f32_with_planner_and_opts(input, out_re, out_im, &planner, &opts);
    }

    fn run_c2r_f64(input_re: &[f64], input_im: &[f64], output: &mut [f64]) {
        let n = output.len();
        let planner = PlannerR2c64::new(n);
        let opts = Options::guess_options(n / 2);
        let mut scratch_re = vec![0.0; n / 2];
        let mut scratch_im = vec![0.0; n / 2];
        c2r_fft_f64_with_planner_and_opts(
            input_re,
            input_im,
            output,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    fn run_c2r_f32(input_re: &[f32], input_im: &[f32], output: &mut [f32]) {
        let n = output.len();
        let planner = PlannerR2c32::new(n);
        let opts = Options::guess_options(n / 2);
        let mut scratch_re = vec![0.0f32; n / 2];
        let mut scratch_im = vec![0.0f32; n / 2];
        c2r_fft_f32_with_planner_and_opts(
            input_re,
            input_im,
            output,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    // The convenience tiers are pure delegations to identical inner calls, so
    // their results are bit-identical to the full tier — exact assert_eq! holds.
    #[test]
    fn r2c_tiers_agree_f64() {
        let n = 256;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let planner = PlannerR2c64::new(n);
        let opts = Options::guess_options(n / 2);
        let mut ref_re = vec![0.0; n / 2 + 1];
        let mut ref_im = vec![0.0; n / 2 + 1];
        r2c_fft_f64_with_planner_and_opts(&input, &mut ref_re, &mut ref_im, &planner, &opts);

        let mut bare_re = vec![0.0; n / 2 + 1];
        let mut bare_im = vec![0.0; n / 2 + 1];
        r2c_fft_f64(&input, &mut bare_re, &mut bare_im);
        assert_eq!((bare_re, bare_im), (ref_re.clone(), ref_im.clone()));

        let mut wp_re = vec![0.0; n / 2 + 1];
        let mut wp_im = vec![0.0; n / 2 + 1];
        r2c_fft_f64_with_planner(&input, &mut wp_re, &mut wp_im, &planner);
        assert_eq!((wp_re, wp_im), (ref_re, ref_im));
    }

    #[test]
    fn c2r_tiers_agree_f64() {
        let n = 256;
        let half = n / 2;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mut spec_re = vec![0.0; half + 1];
        let mut spec_im = vec![0.0; half + 1];
        r2c_fft_f64(&input, &mut spec_re, &mut spec_im);

        let planner = PlannerR2c64::new(n);
        let opts = Options::guess_options(half);
        let mut scratch_re = vec![0.0; half];
        let mut scratch_im = vec![0.0; half];
        let mut ref_out = vec![0.0; n];
        c2r_fft_f64_with_planner_and_opts(
            &spec_re,
            &spec_im,
            &mut ref_out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );

        let mut bare_out = vec![0.0; n];
        c2r_fft_f64(&spec_re, &spec_im, &mut bare_out);
        assert_eq!(bare_out, ref_out);

        let mut wp_out = vec![0.0; n];
        c2r_fft_f64_with_planner(&spec_re, &spec_im, &mut wp_out, &planner);
        assert_eq!(wp_out, ref_out);
    }

    #[test]
    fn r2c_tiers_agree_f32() {
        let n = 256;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

        let planner = PlannerR2c32::new(n);
        let opts = Options::guess_options(n / 2);
        let mut ref_re = vec![0.0f32; n / 2 + 1];
        let mut ref_im = vec![0.0f32; n / 2 + 1];
        r2c_fft_f32_with_planner_and_opts(&input, &mut ref_re, &mut ref_im, &planner, &opts);

        let mut bare_re = vec![0.0f32; n / 2 + 1];
        let mut bare_im = vec![0.0f32; n / 2 + 1];
        r2c_fft_f32(&input, &mut bare_re, &mut bare_im);
        assert_eq!((bare_re, bare_im), (ref_re.clone(), ref_im.clone()));

        let mut wp_re = vec![0.0f32; n / 2 + 1];
        let mut wp_im = vec![0.0f32; n / 2 + 1];
        r2c_fft_f32_with_planner(&input, &mut wp_re, &mut wp_im, &planner);
        assert_eq!((wp_re, wp_im), (ref_re, ref_im));
    }

    #[test]
    fn c2r_tiers_agree_f32() {
        let n = 256;
        let half = n / 2;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let mut spec_re = vec![0.0f32; half + 1];
        let mut spec_im = vec![0.0f32; half + 1];
        r2c_fft_f32(&input, &mut spec_re, &mut spec_im);

        let planner = PlannerR2c32::new(n);
        let opts = Options::guess_options(half);
        let mut scratch_re = vec![0.0f32; half];
        let mut scratch_im = vec![0.0f32; half];
        let mut ref_out = vec![0.0f32; n];
        c2r_fft_f32_with_planner_and_opts(
            &spec_re,
            &spec_im,
            &mut ref_out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );

        let mut bare_out = vec![0.0f32; n];
        c2r_fft_f32(&spec_re, &spec_im, &mut bare_out);
        assert_eq!(bare_out, ref_out);

        let mut wp_out = vec![0.0f32; n];
        c2r_fft_f32_with_planner(&spec_re, &spec_im, &mut wp_out, &planner);
        assert_eq!(wp_out, ref_out);
    }

    #[test]
    fn r2c_vs_c2c_f64() {
        for n_log in 2..=14 {
            let n = 1 << n_log;
            let half = n / 2;
            let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

            let mut r2c_re = vec![0.0; half + 1];
            let mut r2c_im = vec![0.0; half + 1];
            run_r2c_f64(&input, &mut r2c_re, &mut r2c_im);

            let mut ref_re = input.clone();
            let mut ref_im = vec![0.0; n];
            fft_f64_dit(&mut ref_re, &mut ref_im, Direction::Forward);

            for k in 0..=half {
                assert_float_closeness(r2c_re[k], ref_re[k], 1e-4);
                assert_float_closeness(r2c_im[k], ref_im[k], 1e-4);
            }
        }
    }

    #[test]
    fn r2c_vs_c2c_f32() {
        for n_log in 2..=10 {
            let n = 1 << n_log;
            let half = n / 2;
            let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

            let mut r2c_re = vec![0.0f32; half + 1];
            let mut r2c_im = vec![0.0f32; half + 1];
            run_r2c_f32(&input, &mut r2c_re, &mut r2c_im);

            let mut ref_re: Vec<f32> = input.clone();
            let mut ref_im = vec![0.0f32; n];
            fft_f32_dit(&mut ref_re, &mut ref_im, Direction::Forward);

            for k in 0..=half {
                assert_f32_relative_closeness(r2c_re[k], ref_re[k], 1e-2);
                assert_f32_relative_closeness(r2c_im[k], ref_im[k], 1e-2);
            }
        }
    }

    #[test]
    fn roundtrip_f64() {
        for n_log in 2..=14 {
            let n = 1 << n_log;
            let half = n / 2;
            let original: Vec<f64> = (1..=n).map(|i| i as f64).collect();

            let mut spec_re = vec![0.0; half + 1];
            let mut spec_im = vec![0.0; half + 1];
            run_r2c_f64(&original, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0; n];
            run_c2r_f64(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original[k], 1e-6);
            }
        }
    }

    #[test]
    fn c2r_scratch_reuse_across_calls_f64() {
        // Same scratch buffer drives several c2r calls; each result must
        // match an independently-computed reference.
        let n = 256;
        let half = n / 2;
        let planner = PlannerR2c64::new(n);
        let opts = Options::guess_options(half);

        let mut scratch_re = vec![0.0; half];
        let mut scratch_im = vec![0.0; half];

        for seed in 0..4 {
            let input: Vec<f64> = (0..n).map(|i| ((i + seed) as f64).sin()).collect();

            let mut spec_re = vec![0.0; half + 1];
            let mut spec_im = vec![0.0; half + 1];
            r2c_fft_f64_with_planner_and_opts(&input, &mut spec_re, &mut spec_im, &planner, &opts);

            let mut reused = vec![0.0; n];
            c2r_fft_f64_with_planner_and_opts(
                &spec_re,
                &spec_im,
                &mut reused,
                &planner,
                &opts,
                &mut scratch_re,
                &mut scratch_im,
            );

            for k in 0..n {
                assert_float_closeness(reused[k], input[k], 1e-6);
            }
        }
    }

    #[test]
    fn roundtrip_random_f64() {
        for n_log in 4..=14 {
            let n = 1 << n_log;
            let half = n / 2;
            let mut original_re = vec![0.0f64; n];
            let mut dummy_im = vec![0.0f64; n];
            gen_random_signal_f64(&mut original_re, &mut dummy_im);

            let mut spec_re = vec![0.0; half + 1];
            let mut spec_im = vec![0.0; half + 1];
            run_r2c_f64(&original_re, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0; n];
            run_c2r_f64(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original_re[k], 1e-6);
            }
        }
    }

    #[test]
    fn roundtrip_f32() {
        for n_log in 2..=10 {
            let n = 1 << n_log;
            let half = n / 2;
            let original: Vec<f32> = (1..=n).map(|i| i as f32).collect();

            let mut spec_re = vec![0.0f32; half + 1];
            let mut spec_im = vec![0.0f32; half + 1];
            run_r2c_f32(&original, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0f32; n];
            run_c2r_f32(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_f32_relative_closeness(recovered[k], original[k], 1e-2);
            }
        }
    }

    #[test]
    fn roundtrip_random_f32() {
        for n_log in 4..=12 {
            let n = 1 << n_log;
            let half = n / 2;
            let mut original_re = vec![0.0f32; n];
            let mut dummy_im = vec![0.0f32; n];
            gen_random_signal_f32(&mut original_re, &mut dummy_im);

            let mut spec_re = vec![0.0f32; half + 1];
            let mut spec_im = vec![0.0f32; half + 1];
            run_r2c_f32(&original_re, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0f32; n];
            run_c2r_f32(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original_re[k], 1e-5);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn dc_only_f64() {
        // DFT of a constant 1: X[0] = N, all other bins zero.
        let n = 16;
        let half = n / 2;
        let input = vec![1.0f64; n];
        let mut out_re = vec![0.0; half + 1];
        let mut out_im = vec![0.0; half + 1];
        run_r2c_f64(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_re[0], n as f64, 1e-10);
        assert_float_closeness(out_im[0], 0.0, 1e-10);
        for k in 1..=half {
            assert_float_closeness(out_re[k], 0.0, 1e-10);
            assert_float_closeness(out_im[k], 0.0, 1e-10);
        }
    }

    #[test]
    fn nyquist_only_f64() {
        // Alternating ±1 -> all energy at bin N/2 (Nyquist).
        let n = 16;
        let half = n / 2;
        let input: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut out_re = vec![0.0; half + 1];
        let mut out_im = vec![0.0; half + 1];
        run_r2c_f64(&input, &mut out_re, &mut out_im);

        for k in 0..=half {
            let expected_re = if k == half { n as f64 } else { 0.0 };
            assert_float_closeness(out_re[k], expected_re, 1e-10);
            assert_float_closeness(out_im[k], 0.0, 1e-10);
        }
    }

    #[test]
    fn single_tone_f64() {
        // cos(2π·j/N) -> X[1] = X[N-1] = N/2; in N/2+1 layout only X[1] appears.
        let n = 32;
        let half = n / 2;
        let input: Vec<f64> = (0..n)
            .map(|j| (2.0 * std::f64::consts::PI * j as f64 / n as f64).cos())
            .collect();
        let mut out_re = vec![0.0; half + 1];
        let mut out_im = vec![0.0; half + 1];
        run_r2c_f64(&input, &mut out_re, &mut out_im);

        for k in 0..=half {
            let expected_re = if k == 1 { n as f64 / 2.0 } else { 0.0 };
            assert_float_closeness(out_re[k], expected_re, 1e-9);
            assert_float_closeness(out_im[k], 0.0, 1e-9);
        }
    }

    #[test]
    fn all_zeros_f64() {
        // Output buffers are pre-filled to verify they get overwritten.
        let n = 16;
        let half = n / 2;
        let input = vec![0.0f64; n];
        let mut out_re = vec![1.0; half + 1];
        let mut out_im = vec![1.0; half + 1];
        run_r2c_f64(&input, &mut out_re, &mut out_im);

        for k in 0..=half {
            assert_float_closeness(out_re[k], 0.0, 1e-12);
            assert_float_closeness(out_im[k], 0.0, 1e-12);
        }
    }

    #[test]
    fn dc_and_nyquist_real_f64() {
        // For real input, bins 0 and N/2 of the spectrum are purely real.
        let n = 64;
        let half = n / 2;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mut out_re = vec![0.0; half + 1];
        let mut out_im = vec![0.0; half + 1];
        run_r2c_f64(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_im[0], 0.0, 1e-10);
        assert_float_closeness(out_im[half], 0.0, 1e-10);
    }

    // -----------------------------------------------------------------------
    // f32 mirrors of the edge-case suite
    // -----------------------------------------------------------------------

    #[test]
    fn dc_only_f32() {
        let n = 16;
        let half = n / 2;
        let input = vec![1.0f32; n];
        let mut out_re = vec![0.0f32; half + 1];
        let mut out_im = vec![0.0f32; half + 1];
        run_r2c_f32(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_re[0], n as f32, 1e-4);
        assert_float_closeness(out_im[0], 0.0, 1e-4);
        for k in 1..=half {
            assert_float_closeness(out_re[k], 0.0, 1e-4);
            assert_float_closeness(out_im[k], 0.0, 1e-4);
        }
    }

    #[test]
    fn nyquist_only_f32() {
        let n = 16;
        let half = n / 2;
        let input: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut out_re = vec![0.0f32; half + 1];
        let mut out_im = vec![0.0f32; half + 1];
        run_r2c_f32(&input, &mut out_re, &mut out_im);

        for k in 0..=half {
            let expected_re = if k == half { n as f32 } else { 0.0 };
            assert_float_closeness(out_re[k], expected_re, 1e-4);
            assert_float_closeness(out_im[k], 0.0, 1e-4);
        }
    }

    #[test]
    fn all_zeros_f32() {
        let n = 16;
        let half = n / 2;
        let input = vec![0.0f32; n];
        let mut out_re = vec![1.0f32; half + 1];
        let mut out_im = vec![1.0f32; half + 1];
        run_r2c_f32(&input, &mut out_re, &mut out_im);

        for k in 0..=half {
            assert_float_closeness(out_re[k], 0.0, 1e-6);
            assert_float_closeness(out_im[k], 0.0, 1e-6);
        }
    }

    #[test]
    fn dc_and_nyquist_real_f32() {
        let n = 64;
        let half = n / 2;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let mut out_re = vec![0.0f32; half + 1];
        let mut out_im = vec![0.0f32; half + 1];
        run_r2c_f32(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_im[0], 0.0, 1e-3);
        assert_float_closeness(out_im[half], 0.0, 1e-3);
    }

    // -----------------------------------------------------------------------
    // Panic tests — invariant enforcement at PlannerR2c::new and per-call
    // length asserts.
    // -----------------------------------------------------------------------

    macro_rules! planner_panics_on_invalid_n {
        ($test_name:ident, $planner:ty, $n:expr) => {
            #[test]
            #[should_panic(expected = "n must be a power of 2 >= 4")]
            fn $test_name() {
                let _ = <$planner>::new($n);
            }
        };
    }

    planner_panics_on_invalid_n!(planner_r2c_64_panics_on_n_less_than_4, PlannerR2c64, 2);
    planner_panics_on_invalid_n!(planner_r2c_64_panics_on_non_power_of_two, PlannerR2c64, 5);
    planner_panics_on_invalid_n!(planner_r2c_32_panics_on_n_less_than_4, PlannerR2c32, 2);
    planner_panics_on_invalid_n!(planner_r2c_32_panics_on_non_power_of_two, PlannerR2c32, 5);

    #[test]
    #[should_panic(expected = "input length must match planner size")]
    fn r2c_fft_f64_panics_on_input_length_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let input = vec![0.0f64; 32];
        let mut out_re = vec![0.0f64; 17];
        let mut out_im = vec![0.0f64; 17];
        r2c_fft_f64_with_planner_and_opts(&input, &mut out_re, &mut out_im, &planner, &opts);
    }

    #[test]
    #[should_panic(expected = "output_re must have length N/2 + 1")]
    fn r2c_fft_f64_panics_on_output_re_length_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let input = vec![0.0f64; 16];
        let mut out_re = vec![0.0f64; 16];
        let mut out_im = vec![0.0f64; 9];
        r2c_fft_f64_with_planner_and_opts(&input, &mut out_re, &mut out_im, &planner, &opts);
    }

    #[test]
    #[should_panic(expected = "output_im must have length N/2 + 1")]
    fn r2c_fft_f64_panics_on_output_im_length_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let input = vec![0.0f64; 16];
        let mut out_re = vec![0.0f64; 9];
        let mut out_im = vec![0.0f64; 8];
        r2c_fft_f64_with_planner_and_opts(&input, &mut out_re, &mut out_im, &planner, &opts);
    }

    #[test]
    #[should_panic(expected = "output_re must have length N/2 + 1")]
    fn r2c_fft_f32_panics_on_output_re_length_mismatch() {
        let planner = PlannerR2c32::new(16);
        let opts = Options::guess_options(8);
        let input = vec![0.0f32; 16];
        let mut out_re = vec![0.0f32; 16];
        let mut out_im = vec![0.0f32; 9];
        r2c_fft_f32_with_planner_and_opts(&input, &mut out_re, &mut out_im, &planner, &opts);
    }

    #[test]
    #[should_panic(expected = "output length must match planner size")]
    fn c2r_fft_f64_panics_on_output_length_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let in_re = vec![0.0f64; 17];
        let in_im = vec![0.0f64; 17];
        let mut out = vec![0.0f64; 32];
        let mut scratch_re = vec![0.0f64; 8];
        let mut scratch_im = vec![0.0f64; 8];
        c2r_fft_f64_with_planner_and_opts(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "input_re must have length N/2 + 1")]
    fn c2r_fft_f64_panics_on_input_re_length_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let in_re = vec![0.0f64; 16];
        let in_im = vec![0.0f64; 9];
        let mut out = vec![0.0f64; 16];
        let mut scratch_re = vec![0.0f64; 8];
        let mut scratch_im = vec![0.0f64; 8];
        c2r_fft_f64_with_planner_and_opts(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "input_im must have length N/2 + 1")]
    fn c2r_fft_f64_panics_on_input_im_length_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let in_re = vec![0.0f64; 9];
        let in_im = vec![0.0f64; 16];
        let mut out = vec![0.0f64; 16];
        let mut scratch_re = vec![0.0f64; 8];
        let mut scratch_im = vec![0.0f64; 8];
        c2r_fft_f64_with_planner_and_opts(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "input_im must have length N/2 + 1")]
    fn c2r_fft_f32_panics_on_input_im_length_mismatch() {
        let planner = PlannerR2c32::new(16);
        let opts = Options::guess_options(8);
        let in_re = vec![0.0f32; 9];
        let in_im = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 16];
        let mut scratch_re = vec![0.0f32; 8];
        let mut scratch_im = vec![0.0f32; 8];
        c2r_fft_f32_with_planner_and_opts(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "scratch_re must have length N/2")]
    fn c2r_fft_f64_panics_on_scratch_re_size_mismatch() {
        let planner = PlannerR2c64::new(16);
        let opts = Options::guess_options(8);
        let in_re = vec![0.0f64; 9];
        let in_im = vec![0.0f64; 9];
        let mut out = vec![0.0f64; 16];
        let mut scratch_re = vec![0.0f64; 4];
        let mut scratch_im = vec![0.0f64; 8];
        c2r_fft_f64_with_planner_and_opts(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "scratch_im must have length N/2")]
    fn c2r_fft_f32_panics_on_scratch_im_size_mismatch() {
        let planner = PlannerR2c32::new(16);
        let opts = Options::guess_options(8);
        let in_re = vec![0.0f32; 9];
        let in_im = vec![0.0f32; 9];
        let mut out = vec![0.0f32; 16];
        let mut scratch_re = vec![0.0f32; 8];
        let mut scratch_im = vec![0.0f32; 4];
        c2r_fft_f32_with_planner_and_opts(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &opts,
            &mut scratch_re,
            &mut scratch_im,
        );
    }
}
