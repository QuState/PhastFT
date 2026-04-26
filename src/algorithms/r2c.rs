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
//! - **R2C** is in-place: the output buffers double as scratch for the inner
//!   half-length complex FFT, so the hot path performs zero allocations.
//! - **C2R** requires `N/2` reals of scratch per array (re + im). The
//!   `_with_planner_and_scratch` variants take caller-provided scratch and
//!   allocate nothing; the convenience wrappers (`c2r_fft_*`,
//!   `c2r_fft_*_with_planner`) allocate the scratch internally on each call.
//!
//! # References
//!
//! Based on the approach described by Levente Kovács:
//! <https://kovleventer.com/blog/fft_real/>

use fearless_simd::{dispatch, f32x8, f64x4, Simd, SimdBase, SimdFloat};

use crate::algorithms::dit::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};
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
        a.unzip_low(b)
            .store_slice(&mut output_re[out_off..out_off + LANES]);
        a.unzip_high(b)
            .store_slice(&mut output_im[out_off..out_off + LANES]);
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
        a.unzip_low(b)
            .store_slice(&mut output_re[out_off..out_off + LANES]);
        a.unzip_high(b)
            .store_slice(&mut output_im[out_off..out_off + LANES]);
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
// — the reverse of the front lanes. We reverse once on load, do all math in
// the front-lane convention, then reverse-store the mirror writes.

#[inline(always)] // required by fearless_simd
fn simd_untangle_inplace_f64<S: Simd>(
    simd: S,
    output_re: &mut [f64],
    output_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    half: usize,
) {
    let a0 = output_re[0];
    let b0 = output_im[0];
    output_re[0] = a0 + b0;
    output_im[0] = 0.0;
    output_re[half] = a0 - b0;
    output_im[half] = 0.0;

    let q = half / 2;
    const LANES: usize = 4;
    let half_v = f64x4::splat(simd, 0.5);

    let mut k = 1;
    // Process while a full LANES-block stays strictly below the self-pair at q.
    while k + LANES <= q {
        let mirror_low = half - k - (LANES - 1);

        let a = f64x4::from_slice(simd, &output_re[k..k + LANES]);
        let b = f64x4::from_slice(simd, &output_im[k..k + LANES]);
        let c_loaded = f64x4::from_slice(simd, &output_re[mirror_low..mirror_low + LANES]);
        let d_loaded = f64x4::from_slice(simd, &output_im[mirror_low..mirror_low + LANES]);
        let c = rev_f64x4(simd, c_loaded);
        let d = rev_f64x4(simd, d_loaded);

        let s_re = half_v * (a + c);
        let s_im = half_v * (b - d);
        let t_re = b + d;
        let t_im = c - a;

        let wkr_h = f64x4::from_slice(simd, &w_re[k..k + LANES]);
        let wki_h = f64x4::from_slice(simd, &w_im[k..k + LANES]);
        let wzr = wkr_h * t_re - wki_h * t_im;
        let wzi = wkr_h * t_im + wki_h * t_re;

        let out_re_k = s_re + wzr;
        let out_im_k = s_im + wzi;
        let out_re_m = s_re - wzr;
        let out_im_m = wzi - s_im;

        out_re_k.store_slice(&mut output_re[k..k + LANES]);
        out_im_k.store_slice(&mut output_im[k..k + LANES]);
        rev_f64x4(simd, out_re_m).store_slice(&mut output_re[mirror_low..mirror_low + LANES]);
        rev_f64x4(simd, out_im_m).store_slice(&mut output_im[mirror_low..mirror_low + LANES]);

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

#[inline(always)] // required by fearless_simd
fn simd_untangle_inplace_f32<S: Simd>(
    simd: S,
    output_re: &mut [f32],
    output_im: &mut [f32],
    w_re: &[f32],
    w_im: &[f32],
    half: usize,
) {
    let a0 = output_re[0];
    let b0 = output_im[0];
    output_re[0] = a0 + b0;
    output_im[0] = 0.0;
    output_re[half] = a0 - b0;
    output_im[half] = 0.0;

    let q = half / 2;
    const LANES: usize = 8;
    let half_v = f32x8::splat(simd, 0.5);

    let mut k = 1;
    while k + LANES <= q {
        let mirror_low = half - k - (LANES - 1);

        let a = f32x8::from_slice(simd, &output_re[k..k + LANES]);
        let b = f32x8::from_slice(simd, &output_im[k..k + LANES]);
        let c_loaded = f32x8::from_slice(simd, &output_re[mirror_low..mirror_low + LANES]);
        let d_loaded = f32x8::from_slice(simd, &output_im[mirror_low..mirror_low + LANES]);
        let c = rev_f32x8(simd, c_loaded);
        let d = rev_f32x8(simd, d_loaded);

        let s_re = half_v * (a + c);
        let s_im = half_v * (b - d);
        let t_re = b + d;
        let t_im = c - a;

        let wkr_h = f32x8::from_slice(simd, &w_re[k..k + LANES]);
        let wki_h = f32x8::from_slice(simd, &w_im[k..k + LANES]);
        let wzr = wkr_h * t_re - wki_h * t_im;
        let wzi = wkr_h * t_im + wki_h * t_re;

        let out_re_k = s_re + wzr;
        let out_im_k = s_im + wzi;
        let out_re_m = s_re - wzr;
        let out_im_m = wzi - s_im;

        out_re_k.store_slice(&mut output_re[k..k + LANES]);
        out_im_k.store_slice(&mut output_im[k..k + LANES]);
        rev_f32x8(simd, out_re_m).store_slice(&mut output_re[mirror_low..mirror_low + LANES]);
        rev_f32x8(simd, out_im_m).store_slice(&mut output_im[mirror_low..mirror_low + LANES]);

        k += LANES;
    }

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

    let a = output_re[q];
    let b = output_im[q];
    output_re[q] = a + 2.0 * w_re[q] * b;
    output_im[q] = 2.0 * w_im[q] * b;
}

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
// z_re / z_im and stores 2*LANES interleaved reals to output, using
// zip_low/zip_high (the inverse of unzip_low/unzip_high). On NEON the pair
// lowers to two register-level interleave ops; on AVX2 it's two `vunpck*`s.

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
        re.zip_low(im)
            .store_slice(&mut output[out_off..out_off + LANES]);
        re.zip_high(im)
            .store_slice(&mut output[out_off + LANES..out_off + 2 * LANES]);
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
        re.zip_low(im)
            .store_slice(&mut output[out_off..out_off + LANES]);
        re.zip_high(im)
            .store_slice(&mut output[out_off + LANES..out_off + 2 * LANES]);
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
/// complex FFT, so this entry point allocates only the per-call planner.
///
/// # Panics
///
/// Panics if `input_re.len()` is not a power of 2 ≥ 4, or if `output_re` /
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

/// Real-valued FFT of `f64` data with a pre-computed planner.
///
/// See [`r2c_fft_f64`] for the layout contract. This entry point allocates
/// nothing — the output buffers serve as the inner FFT's working memory.
///
/// # Panics
///
/// Panics if `input_re.len()` does not match the planner size, or if
/// `output_re` / `output_im` are not length `N / 2 + 1`.
pub fn r2c_fft_f64_with_planner(
    input_re: &[f64],
    output_re: &mut [f64],
    output_im: &mut [f64],
    planner: &PlannerR2c64,
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
        fft_64_dit_with_planner_and_opts(
            z_re,
            z_im,
            Direction::Forward,
            &planner.dit_planner,
            &planner.inner_opts,
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
/// See [`r2c_fft_f64`] for details. This is the single-precision variant.
pub fn r2c_fft_f32(input_re: &[f32], output_re: &mut [f32], output_im: &mut [f32]) {
    let planner = PlannerR2c32::new(input_re.len());
    r2c_fft_f32_with_planner(input_re, output_re, output_im, &planner);
}

/// Real-valued FFT of `f32` data with a pre-computed planner.
///
/// See [`r2c_fft_f64_with_planner`] for details. This is the single-precision
/// variant.
pub fn r2c_fft_f32_with_planner(
    input_re: &[f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
    planner: &PlannerR2c32,
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
        fft_32_dit_with_planner_and_opts(
            z_re,
            z_im,
            Direction::Forward,
            &planner.dit_planner,
            &planner.inner_opts,
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
/// Given the `N/2 + 1` independent complex bins produced by [`r2c_fft_f64`]
/// (conjugate-symmetric redundancy stripped), recovers the original `N` real
/// samples. Allocates `N` reals of scratch per call. To eliminate the
/// allocation, hand a reusable scratch buffer to
/// [`c2r_fft_f64_with_planner_and_scratch`].
///
/// # Panics
///
/// Panics if `output.len()` is not a power of 2 ≥ 4, or if `input_re` /
/// `input_im` are not length `output.len() / 2 + 1`.
///
/// # Example
///
/// ```
/// use phastft::{c2r_fft_f64, r2c_fft_f64};
///
/// let n = 16;
/// let signal: Vec<f64> = (1..=n).map(|x| x as f64).collect();
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

/// Inverse real-valued FFT of `f64` data with a pre-computed planner.
///
/// Allocates two `Vec<f64>` of length `N/2` as scratch on each call. For
/// zero allocation, hand reusable scratch buffers to
/// [`c2r_fft_f64_with_planner_and_scratch`].
///
/// # Panics
///
/// Panics if `output.len()` does not match the planner size, or if `input_re`
/// / `input_im` are not length `N / 2 + 1`.
pub fn c2r_fft_f64_with_planner(
    input_re: &[f64],
    input_im: &[f64],
    output: &mut [f64],
    planner: &PlannerR2c64,
) {
    let half = planner.n / 2;
    let mut scratch_re = vec![0.0f64; half];
    let mut scratch_im = vec![0.0f64; half];
    c2r_fft_f64_with_planner_and_scratch(
        input_re,
        input_im,
        output,
        planner,
        &mut scratch_re,
        &mut scratch_im,
    );
}

/// Inverse real-valued FFT of `f64` data with caller-provided scratch.
///
/// Performs no allocation. `scratch_re` and `scratch_im` must each be length
/// `N / 2`; their contents on entry are ignored and on exit are unspecified
/// (callers may reuse the buffers across calls).
///
/// # Panics
///
/// Panics if `output.len()` does not match the planner size, if `input_re` /
/// `input_im` are not length `N / 2 + 1`, or if either scratch slice is not
/// length `N / 2`.
pub fn c2r_fft_f64_with_planner_and_scratch(
    input_re: &[f64],
    input_im: &[f64],
    output: &mut [f64],
    planner: &PlannerR2c64,
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

    fft_64_dit_with_planner_and_opts(
        scratch_re,
        scratch_im,
        Direction::Reverse,
        &planner.dit_planner,
        &planner.inner_opts,
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
/// See [`c2r_fft_f64`] for details. This is the single-precision variant.
pub fn c2r_fft_f32(input_re: &[f32], input_im: &[f32], output: &mut [f32]) {
    let planner = PlannerR2c32::new(output.len());
    c2r_fft_f32_with_planner(input_re, input_im, output, &planner);
}

/// Inverse real-valued FFT of `f32` data with a pre-computed planner.
///
/// See [`c2r_fft_f64_with_planner`] for details. This is the single-precision
/// variant.
pub fn c2r_fft_f32_with_planner(
    input_re: &[f32],
    input_im: &[f32],
    output: &mut [f32],
    planner: &PlannerR2c32,
) {
    let half = planner.n / 2;
    let mut scratch_re = vec![0.0f32; half];
    let mut scratch_im = vec![0.0f32; half];
    c2r_fft_f32_with_planner_and_scratch(
        input_re,
        input_im,
        output,
        planner,
        &mut scratch_re,
        &mut scratch_im,
    );
}

/// Inverse real-valued FFT of `f32` data with caller-provided scratch.
///
/// See [`c2r_fft_f64_with_planner_and_scratch`] for details. This is the
/// single-precision variant.
pub fn c2r_fft_f32_with_planner_and_scratch(
    input_re: &[f32],
    input_im: &[f32],
    output: &mut [f32],
    planner: &PlannerR2c32,
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

    fft_32_dit_with_planner_and_opts(
        scratch_re,
        scratch_im,
        Direction::Reverse,
        &planner.dit_planner,
        &planner.inner_opts,
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
            let half = n / 2;
            let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

            let mut r2c_re = vec![0.0; half + 1];
            let mut r2c_im = vec![0.0; half + 1];
            r2c_fft_f64(&input, &mut r2c_re, &mut r2c_im);

            let mut ref_re = input.clone();
            let mut ref_im = vec![0.0; n];
            fft_64_dit(&mut ref_re, &mut ref_im, Direction::Forward);

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
            r2c_fft_f32(&input, &mut r2c_re, &mut r2c_im);

            let mut ref_re: Vec<f32> = input.clone();
            let mut ref_im = vec![0.0f32; n];
            fft_32_dit(&mut ref_re, &mut ref_im, Direction::Forward);

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
            r2c_fft_f64(&original, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0; n];
            c2r_fft_f64(&spec_re, &spec_im, &mut recovered);

            for k in 0..n {
                assert_float_closeness(recovered[k], original[k], 1e-6);
            }
        }
    }

    #[test]
    fn planner_matches_convenience_f64() {
        let n = 1024;
        let half = n / 2;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let mut out_re_1 = vec![0.0; half + 1];
        let mut out_im_1 = vec![0.0; half + 1];
        r2c_fft_f64(&input, &mut out_re_1, &mut out_im_1);

        let planner = PlannerR2c64::new(n);
        let mut out_re_2 = vec![0.0; half + 1];
        let mut out_im_2 = vec![0.0; half + 1];
        r2c_fft_f64_with_planner(&input, &mut out_re_2, &mut out_im_2, &planner);

        for k in 0..=half {
            assert_eq!(out_re_1[k], out_re_2[k]);
            assert_eq!(out_im_1[k], out_im_2[k]);
        }
    }

    #[test]
    fn planner_matches_convenience_f32() {
        let n = 1024;
        let half = n / 2;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

        let mut out_re_1 = vec![0.0f32; half + 1];
        let mut out_im_1 = vec![0.0f32; half + 1];
        r2c_fft_f32(&input, &mut out_re_1, &mut out_im_1);

        let planner = PlannerR2c32::new(n);
        let mut out_re_2 = vec![0.0f32; half + 1];
        let mut out_im_2 = vec![0.0f32; half + 1];
        r2c_fft_f32_with_planner(&input, &mut out_re_2, &mut out_im_2, &planner);

        for k in 0..=half {
            assert_eq!(out_re_1[k], out_re_2[k]);
            assert_eq!(out_im_1[k], out_im_2[k]);
        }
    }

    #[test]
    fn c2r_planner_matches_convenience_f64() {
        let n = 1024;
        let half = n / 2;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let mut spec_re = vec![0.0; half + 1];
        let mut spec_im = vec![0.0; half + 1];
        r2c_fft_f64(&input, &mut spec_re, &mut spec_im);

        let mut recovered_1 = vec![0.0; n];
        c2r_fft_f64(&spec_re, &spec_im, &mut recovered_1);

        let planner = PlannerR2c64::new(n);
        let mut recovered_2 = vec![0.0; n];
        c2r_fft_f64_with_planner(&spec_re, &spec_im, &mut recovered_2, &planner);

        for k in 0..n {
            assert_eq!(recovered_1[k], recovered_2[k]);
            assert_float_closeness(recovered_2[k], input[k], 1e-6);
        }
    }

    #[test]
    fn c2r_planner_matches_convenience_f32() {
        let n = 1024;
        let half = n / 2;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

        let mut spec_re = vec![0.0f32; half + 1];
        let mut spec_im = vec![0.0f32; half + 1];
        r2c_fft_f32(&input, &mut spec_re, &mut spec_im);

        let mut recovered_1 = vec![0.0f32; n];
        c2r_fft_f32(&spec_re, &spec_im, &mut recovered_1);

        let planner = PlannerR2c32::new(n);
        let mut recovered_2 = vec![0.0f32; n];
        c2r_fft_f32_with_planner(&spec_re, &spec_im, &mut recovered_2, &planner);

        for k in 0..n {
            assert_eq!(recovered_1[k], recovered_2[k]);
        }
    }

    // -----------------------------------------------------------------------
    // Scratch API: parity with allocating variants + cross-call reuse
    // -----------------------------------------------------------------------

    #[test]
    fn c2r_scratch_matches_allocating_f64() {
        let n = 1024;
        let half = n / 2;
        let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let mut spec_re = vec![0.0; half + 1];
        let mut spec_im = vec![0.0; half + 1];
        r2c_fft_f64(&input, &mut spec_re, &mut spec_im);

        let planner = PlannerR2c64::new(n);

        let mut allocating = vec![0.0; n];
        c2r_fft_f64_with_planner(&spec_re, &spec_im, &mut allocating, &planner);

        let mut scratch_re = vec![0.0; half];
        let mut scratch_im = vec![0.0; half];
        let mut scratched = vec![0.0; n];
        c2r_fft_f64_with_planner_and_scratch(
            &spec_re,
            &spec_im,
            &mut scratched,
            &planner,
            &mut scratch_re,
            &mut scratch_im,
        );

        for k in 0..n {
            assert_eq!(allocating[k], scratched[k]);
        }
    }

    #[test]
    fn c2r_scratch_matches_allocating_f32() {
        let n = 1024;
        let half = n / 2;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

        let mut spec_re = vec![0.0f32; half + 1];
        let mut spec_im = vec![0.0f32; half + 1];
        r2c_fft_f32(&input, &mut spec_re, &mut spec_im);

        let planner = PlannerR2c32::new(n);

        let mut allocating = vec![0.0f32; n];
        c2r_fft_f32_with_planner(&spec_re, &spec_im, &mut allocating, &planner);

        let mut scratch_re = vec![0.0f32; half];
        let mut scratch_im = vec![0.0f32; half];
        let mut scratched = vec![0.0f32; n];
        c2r_fft_f32_with_planner_and_scratch(
            &spec_re,
            &spec_im,
            &mut scratched,
            &planner,
            &mut scratch_re,
            &mut scratch_im,
        );

        for k in 0..n {
            assert_eq!(allocating[k], scratched[k]);
        }
    }

    #[test]
    fn c2r_scratch_reuse_across_calls_f64() {
        // Same scratch buffer drives several c2r calls; each result must
        // match an independently-computed reference.
        let n = 256;
        let half = n / 2;
        let planner = PlannerR2c64::new(n);

        let mut scratch_re = vec![0.0; half];
        let mut scratch_im = vec![0.0; half];

        for seed in 0..4 {
            let input: Vec<f64> = (0..n).map(|i| ((i + seed) as f64).sin()).collect();

            let mut spec_re = vec![0.0; half + 1];
            let mut spec_im = vec![0.0; half + 1];
            r2c_fft_f64_with_planner(&input, &mut spec_re, &mut spec_im, &planner);

            let mut reused = vec![0.0; n];
            c2r_fft_f64_with_planner_and_scratch(
                &spec_re,
                &spec_im,
                &mut reused,
                &planner,
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
            let half = n / 2;
            let original: Vec<f32> = (1..=n).map(|i| i as f32).collect();

            let mut spec_re = vec![0.0f32; half + 1];
            let mut spec_im = vec![0.0f32; half + 1];
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
            let half = n / 2;
            let mut original_re = vec![0.0f32; n];
            let mut dummy_im = vec![0.0f32; n];
            gen_random_signal_f32(&mut original_re, &mut dummy_im);

            let mut spec_re = vec![0.0f32; half + 1];
            let mut spec_im = vec![0.0f32; half + 1];
            r2c_fft_f32(&original_re, &mut spec_re, &mut spec_im);

            let mut recovered = vec![0.0f32; n];
            c2r_fft_f32(&spec_re, &spec_im, &mut recovered);

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
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f64(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f32(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f32(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f32(&input, &mut out_re, &mut out_im);

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
        r2c_fft_f32(&input, &mut out_re, &mut out_im);

        assert_float_closeness(out_im[0], 0.0, 1e-3);
        assert_float_closeness(out_im[half], 0.0, 1e-3);
    }

    // -----------------------------------------------------------------------
    // Panic tests — invariants enforced via PlannerR2c::new and length asserts
    // -----------------------------------------------------------------------

    macro_rules! r2c_panics_on_invalid_n {
        ($test_name:ident, $func:ident, $precision:ty, $n:expr) => {
            #[test]
            #[should_panic(expected = "n must be a power of 2 >= 4")]
            fn $test_name() {
                let n: usize = $n;
                let input = vec![<$precision>::default(); n];
                let mut out_re = vec![<$precision>::default(); n / 2 + 1];
                let mut out_im = vec![<$precision>::default(); n / 2 + 1];
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
                let in_re = vec![<$precision>::default(); n / 2 + 1];
                let in_im = vec![<$precision>::default(); n / 2 + 1];
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
    #[should_panic(expected = "output_re must have length N/2 + 1")]
    fn r2c_fft_f64_panics_on_output_re_length_mismatch() {
        let input = vec![0.0f64; 16];
        let mut out_re = vec![0.0f64; 16];
        let mut out_im = vec![0.0f64; 9];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);
    }

    #[test]
    #[should_panic(expected = "output_im must have length N/2 + 1")]
    fn r2c_fft_f64_panics_on_output_im_length_mismatch() {
        let input = vec![0.0f64; 16];
        let mut out_re = vec![0.0f64; 9];
        let mut out_im = vec![0.0f64; 8];
        r2c_fft_f64(&input, &mut out_re, &mut out_im);
    }

    #[test]
    #[should_panic(expected = "output_re must have length N/2 + 1")]
    fn r2c_fft_f32_panics_on_output_re_length_mismatch() {
        let input = vec![0.0f32; 16];
        let mut out_re = vec![0.0f32; 16];
        let mut out_im = vec![0.0f32; 9];
        r2c_fft_f32(&input, &mut out_re, &mut out_im);
    }

    #[test]
    #[should_panic(expected = "input_re must have length N/2 + 1")]
    fn c2r_fft_f64_panics_on_input_re_length_mismatch() {
        let in_re = vec![0.0f64; 16];
        let in_im = vec![0.0f64; 9];
        let mut out = vec![0.0f64; 16];
        c2r_fft_f64(&in_re, &in_im, &mut out);
    }

    #[test]
    #[should_panic(expected = "input_im must have length N/2 + 1")]
    fn c2r_fft_f64_panics_on_input_im_length_mismatch() {
        let in_re = vec![0.0f64; 9];
        let in_im = vec![0.0f64; 16];
        let mut out = vec![0.0f64; 16];
        c2r_fft_f64(&in_re, &in_im, &mut out);
    }

    #[test]
    #[should_panic(expected = "input_im must have length N/2 + 1")]
    fn c2r_fft_f32_panics_on_input_im_length_mismatch() {
        let in_re = vec![0.0f32; 9];
        let in_im = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 16];
        c2r_fft_f32(&in_re, &in_im, &mut out);
    }

    #[test]
    #[should_panic(expected = "input length must match planner size")]
    fn r2c_fft_f64_with_planner_panics_on_planner_size_mismatch() {
        let planner = PlannerR2c64::new(16);
        let input = vec![0.0f64; 32];
        let mut out_re = vec![0.0f64; 17];
        let mut out_im = vec![0.0f64; 17];
        r2c_fft_f64_with_planner(&input, &mut out_re, &mut out_im, &planner);
    }

    #[test]
    #[should_panic(expected = "output length must match planner size")]
    fn c2r_fft_f64_with_planner_panics_on_planner_size_mismatch() {
        let planner = PlannerR2c64::new(16);
        let in_re = vec![0.0f64; 17];
        let in_im = vec![0.0f64; 17];
        let mut out = vec![0.0f64; 32];
        c2r_fft_f64_with_planner(&in_re, &in_im, &mut out, &planner);
    }

    #[test]
    #[should_panic(expected = "scratch_re must have length N/2")]
    fn c2r_fft_f64_with_planner_and_scratch_panics_on_scratch_re_size_mismatch() {
        let planner = PlannerR2c64::new(16);
        let in_re = vec![0.0f64; 9];
        let in_im = vec![0.0f64; 9];
        let mut out = vec![0.0f64; 16];
        let mut scratch_re = vec![0.0f64; 4];
        let mut scratch_im = vec![0.0f64; 8];
        c2r_fft_f64_with_planner_and_scratch(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &mut scratch_re,
            &mut scratch_im,
        );
    }

    #[test]
    #[should_panic(expected = "scratch_im must have length N/2")]
    fn c2r_fft_f32_with_planner_and_scratch_panics_on_scratch_im_size_mismatch() {
        let planner = PlannerR2c32::new(16);
        let in_re = vec![0.0f32; 9];
        let in_im = vec![0.0f32; 9];
        let mut out = vec![0.0f32; 16];
        let mut scratch_re = vec![0.0f32; 8];
        let mut scratch_im = vec![0.0f32; 4];
        c2r_fft_f32_with_planner_and_scratch(
            &in_re,
            &in_im,
            &mut out,
            &planner,
            &mut scratch_re,
            &mut scratch_im,
        );
    }
}
