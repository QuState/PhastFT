//! DIT-specific FFT Kernels
//!
//! FFT kernels for the Decimation-in-Time algorithm.
//!
use core::f32;

use fearless_simd::{f32x16, f32x4, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};
use num_traits::Float;

/// DIT butterfly for chunk_size == 2
/// Identical to DIF version (no twiddles at size 2)
#[inline(always)] // required by fearless_simd
pub fn fft_dit_chunk_2<S: Simd, T: Float>(_simd: S, reals: &mut [T], imags: &mut [T]) {
    reals
        .chunks_exact_mut(2)
        .zip(imags.chunks_exact_mut(2))
        .for_each(|(reals_chunk, imags_chunk)| {
            let z0_re = reals_chunk[0];
            let z0_im = imags_chunk[0];
            let z1_re = reals_chunk[1];
            let z1_im = imags_chunk[1];

            reals_chunk[0] = z0_re + z1_re;
            imags_chunk[0] = z0_im + z1_im;
            reals_chunk[1] = z0_re - z1_re;
            imags_chunk[1] = z0_im - z1_im;
        });
}

/// DIT butterfly for chunk_size == 4 (f64)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_4_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_4_simd_f64(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 4 (f64)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_4_simd_f64<S: Simd>(_simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = 2.0_f64;

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // First pair (W_4^0 = 1)
            let in0_re = reals_s0[0];
            let in1_re = reals_s1[0];
            let in0_im = imags_s0[0];
            let in1_im = imags_s1[0];

            reals_s0[0] = in0_re + in1_re;
            imags_s0[0] = in0_im + in1_im;
            // out1 = 2*in0 - out0
            reals_s1[0] = in0_re.mul_add(two, -reals_s0[0]);
            imags_s1[0] = in0_im.mul_add(two, -imags_s0[0]);

            // Second pair (W_4^1 = -i)
            let in0_re = reals_s0[1];
            let in1_re = reals_s1[1];
            let in0_im = imags_s0[1];
            let in1_im = imags_s1[1];

            // W_4^1 = -i
            reals_s0[1] = in0_re + in1_im;
            imags_s0[1] = in0_im - in1_re;
            // out1 = 2*in0 - out0
            reals_s1[1] = in0_re.mul_add(two, -reals_s0[1]);
            imags_s1[1] = in0_im.mul_add(two, -imags_s0[1]);
        });
}

/// DIT butterfly for chunk_size == 4 (f32)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_4_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_4_simd_f32(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 4 (f32)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_4_simd_f32<S: Simd>(_simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = 2.0_f32;

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // First pair (W_4^0 = 1)
            let in0_re = reals_s0[0];
            let in1_re = reals_s1[0];
            let in0_im = imags_s0[0];
            let in1_im = imags_s1[0];

            reals_s0[0] = in0_re + in1_re;
            imags_s0[0] = in0_im + in1_im;
            // out1 = 2*in0 - out0
            reals_s1[0] = in0_re.mul_add(two, -reals_s0[0]);
            imags_s1[0] = in0_im.mul_add(two, -imags_s0[0]);

            // Second pair (W_4^1 = -i)
            let in0_re = reals_s0[1];
            let in1_re = reals_s1[1];
            let in0_im = imags_s0[1];
            let in1_im = imags_s1[1];

            // W_4^1 = -i
            reals_s0[1] = in0_re + in1_im;
            imags_s0[1] = in0_im - in1_re;
            // out1 = 2*in0 - out0
            reals_s1[1] = in0_re.mul_add(two, -reals_s0[1]);
            imags_s1[1] = in0_im.mul_add(two, -imags_s0[1]);
        });
}

/// DIT butterfly for chunk_size == 8 (f64) with SIMD
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_8_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_8_simd_f64(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 8 (f64) with SIMD
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_8_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 4;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f64x4::splat(simd, 2.0);
    let sqrt2_2 = f64x4::simd_from(
        simd,
        [
            1.0,                              // W_8^0 real
            std::f64::consts::FRAC_1_SQRT_2,  // W_8^1 real (sqrt(2)/2)
            0.0,                              // W_8^2 real
            -std::f64::consts::FRAC_1_SQRT_2, // W_8^3 real (-sqrt(2)/2)
        ],
    );
    let sqrt2_2_im = f64x4::simd_from(
        simd,
        [
            0.0,                              // W_8^0 imag
            -std::f64::consts::FRAC_1_SQRT_2, // W_8^1 imag (-sqrt(2)/2)
            -1.0,                             // W_8^2 imag
            -std::f64::consts::FRAC_1_SQRT_2, // W_8^3 imag (-sqrt(2)/2)
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            let in0_re = f64x4::from_slice(simd, &reals_s0[0..4]);
            let in1_re = f64x4::from_slice(simd, &reals_s1[0..4]);
            let in0_im = f64x4::from_slice(simd, &imags_s0[0..4]);
            let in1_im = f64x4::from_slice(simd, &imags_s1[0..4]);

            // out0.re = (in0.re + w.re * in1.re) - w.im * in1.im
            let out0_re = sqrt2_2_im.mul_add(-in1_im, sqrt2_2.mul_add(in1_re, in0_re));
            // out0.im = (in0.im + w.re*in1.im) + w.im*in1.re
            let out0_im = sqrt2_2_im.mul_add(in1_re, sqrt2_2.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(reals_s0);
            out0_im.store_slice(imags_s0);
            out1_re.store_slice(reals_s1);
            out1_im.store_slice(imags_s1);
        });
}

/// DIT butterfly for chunk_size == 8 (f32) with SIMD
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_8_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_8_simd_f32(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 8 (f32) with SIMD
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_8_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 4;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f32x4::splat(simd, 2.0);
    let sqrt2_2 = f32x4::simd_from(
        simd,
        [
            1.0_f32,                          // W_8^0 real
            std::f32::consts::FRAC_1_SQRT_2,  // W_8^1 real (sqrt(2)/2)
            0.0_f32,                          // W_8^2 real
            -std::f32::consts::FRAC_1_SQRT_2, // W_8^3 real (-sqrt(2)/2)
        ],
    );
    let sqrt2_2_im = f32x4::simd_from(
        simd,
        [
            0.0_f32,                          // W_8^0 imag
            -std::f32::consts::FRAC_1_SQRT_2, // W_8^1 imag (-sqrt(2)/2)
            -1.0_f32,                         // W_8^2 imag
            -std::f32::consts::FRAC_1_SQRT_2, // W_8^3 imag (-sqrt(2)/2)
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            let in0_re = f32x4::from_slice(simd, &reals_s0[0..4]);
            let in1_re = f32x4::from_slice(simd, &reals_s1[0..4]);
            let in0_im = f32x4::from_slice(simd, &imags_s0[0..4]);
            let in1_im = f32x4::from_slice(simd, &imags_s1[0..4]);

            // out0.re = (in0.re + w.re * in1.re) - w.im * in1.im
            let out0_re = sqrt2_2_im.mul_add(-in1_im, sqrt2_2.mul_add(in1_re, in0_re));
            // out0.im = (in0.im + w.re * in1.im) + w.im * in1.re
            let out0_im = sqrt2_2_im.mul_add(in1_re, sqrt2_2.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(reals_s0);
            out0_im.store_slice(imags_s0);
            out1_re.store_slice(reals_s1);
            out1_im.store_slice(imags_s1);
        });
}

/// DIT butterfly for chunk_size == 16 (f64)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_16_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_16_simd_f64(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 16 (f64)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_16_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 8;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f64x8::splat(simd, 2.0);

    // Twiddle factors for W_16^k where k = 0..7
    let twiddle_re = f64x8::simd_from(
        simd,
        [
            1.0,                              // W_16^0
            0.9238795325112867,               // W_16^1 = cos(pi/8)
            std::f64::consts::FRAC_1_SQRT_2,  // W_16^2 = sqrt(2)/2
            0.38268343236508984,              // W_16^3 = cos(3*pi/8)
            0.0,                              // W_16^4
            -0.38268343236508984,             // W_16^5 = -cos(3*pi/8)
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
            -0.9238795325112867,              // W_16^7 = -cos(pi/8)
        ],
    );

    let twiddle_im = f64x8::simd_from(
        simd,
        [
            0.0,                              // W_16^0
            -0.38268343236508984,             // W_16^1 = -sin(pi/8)
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^2 = -sqrt(2)/2
            -0.9238795325112867,              // W_16^3 = -sin(3*pi/8)
            -1.0,                             // W_16^4
            -0.9238795325112867,              // W_16^5 = -sin(3*pi/8)
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
            -0.38268343236508984,             // W_16^7 = -sin(pi/8)
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Load all 8 elements at once
            let in0_re = f64x8::from_slice(simd, &reals_s0[0..8]);
            let in1_re = f64x8::from_slice(simd, &reals_s1[0..8]);
            let in0_im = f64x8::from_slice(simd, &imags_s0[0..8]);
            let in1_im = f64x8::from_slice(simd, &imags_s1[0..8]);

            let out0_re = twiddle_im.mul_add(-in1_im, twiddle_re.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im.mul_add(in1_re, twiddle_re.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(reals_s0);
            out0_im.store_slice(imags_s0);
            out1_re.store_slice(reals_s1);
            out1_im.store_slice(imags_s1);
        });
}

/// DIT butterfly for chunk_size == 16 (f32)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_16_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_16_simd_f32(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 16 (f32)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_16_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 8;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f32x8::splat(simd, 2.0);

    // Twiddle factors for W_16^k where k = 0..7
    let twiddle_re = f32x8::simd_from(
        simd,
        [
            1.0_f32,                          // W_16^0
            0.923_879_5_f32,                  // W_16^1 = cos(pi/8)
            std::f32::consts::FRAC_1_SQRT_2,  // W_16^2 = sqrt(2)/2
            0.382_683_43_f32,                 // W_16^3 = cos(3*pi/8)
            0.0_f32,                          // W_16^4
            -0.382_683_43_f32,                // W_16^5 = -cos(3*pi/8)
            -std::f32::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
            -0.923_879_5_f32,                 // W_16^7 = -cos(pi/8)
        ],
    );

    let twiddle_im = f32x8::simd_from(
        simd,
        [
            0.0_f32,                          // W_16^0
            -0.382_683_43_f32,                // W_16^1 = -sin(pi/8)
            -std::f32::consts::FRAC_1_SQRT_2, // W_16^2 = -sqrt(2)/2
            -0.923_879_5_f32,                 // W_16^3 = -sin(3*pi/8)
            -1.0_f32,                         // W_16^4
            -0.923_879_5_f32,                 // W_16^5 = -sin(3*pi/8)
            -std::f32::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
            -0.382_683_43_f32,                // W_16^7 = -sin(pi/8)
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Load all 8 elements at once
            let in0_re = f32x8::from_slice(simd, &reals_s0[0..8]);
            let in1_re = f32x8::from_slice(simd, &reals_s1[0..8]);
            let in0_im = f32x8::from_slice(simd, &imags_s0[0..8]);
            let in1_im = f32x8::from_slice(simd, &imags_s1[0..8]);

            let out0_re = twiddle_im.mul_add(-in1_im, twiddle_re.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im.mul_add(in1_re, twiddle_re.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(reals_s0);
            out0_im.store_slice(imags_s0);
            out1_re.store_slice(reals_s1);
            out1_im.store_slice(imags_s1);
        });
}
/// DIT butterfly for chunk_size == 32 (f64)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_32_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_32_simd_f64(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 32 (f64)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_32_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 16;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f64x8::splat(simd, 2.0);

    // First 8 twiddle factors for W_32^k where k = 0..7
    let twiddle_re_0_7 = f64x8::simd_from(
        simd,
        [
            1.0,                             // W_32^0 = 1
            0.9807852804032304,              // W_32^1 = cos(π/16)
            0.9238795325112867,              // W_32^2 = cos(π/8)
            0.8314696123025452,              // W_32^3 = cos(3π/16)
            std::f64::consts::FRAC_1_SQRT_2, // W_32^4 = sqrt(2)/2
            0.5555702330196022,              // W_32^5 = cos(5π/16)
            0.3826834323650898,              // W_32^6 = cos(3π/8)
            0.19509032201612825,             // W_32^7 = cos(7π/16)
        ],
    );

    let twiddle_im_0_7 = f64x8::simd_from(
        simd,
        [
            0.0,                              // W_32^0
            -0.19509032201612825,             // W_32^1 = -sin(π/16)
            -0.3826834323650898,              // W_32^2 = -sin(π/8)
            -0.5555702330196022,              // W_32^3 = -sin(3π/16)
            -std::f64::consts::FRAC_1_SQRT_2, // W_32^4 = -sqrt(2)/2
            -0.8314696123025452,              // W_32^5 = -sin(5π/16)
            -0.9238795325112867,              // W_32^6 = -sin(3π/8)
            -0.9807852804032304,              // W_32^7 = -sin(7π/16)
        ],
    );

    // Second 8 twiddle factors for W_32^k where k = 8..15
    let twiddle_re_8_15 = f64x8::simd_from(
        simd,
        [
            0.0,                              // W_32^8 = 0 - i
            -0.19509032201612825,             // W_32^9
            -0.3826834323650898,              // W_32^10
            -0.5555702330196022,              // W_32^11
            -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
            -0.8314696123025452,              // W_32^13
            -0.9238795325112867,              // W_32^14
            -0.9807852804032304,              // W_32^15
        ],
    );

    let twiddle_im_8_15 = f64x8::simd_from(
        simd,
        [
            -1.0,                             // W_32^8
            -0.9807852804032304,              // W_32^9
            -0.9238795325112867,              // W_32^10
            -0.8314696123025452,              // W_32^11
            -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
            -0.5555702330196022,              // W_32^13
            -0.3826834323650898,              // W_32^14
            -0.19509032201612825,             // W_32^15
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process first 8 butterflies
            let in0_re_0_7 = f64x8::from_slice(simd, &reals_s0[0..8]);
            let in1_re_0_7 = f64x8::from_slice(simd, &reals_s1[0..8]);
            let in0_im_0_7 = f64x8::from_slice(simd, &imags_s0[0..8]);
            let in1_im_0_7 = f64x8::from_slice(simd, &imags_s1[0..8]);

            let out0_re_0_7 =
                twiddle_im_0_7.mul_add(-in1_im_0_7, twiddle_re_0_7.mul_add(in1_re_0_7, in0_re_0_7));
            let out0_im_0_7 =
                twiddle_im_0_7.mul_add(in1_re_0_7, twiddle_re_0_7.mul_add(in1_im_0_7, in0_im_0_7));

            let out1_re_0_7 = two.mul_sub(in0_re_0_7, out0_re_0_7);
            let out1_im_0_7 = two.mul_sub(in0_im_0_7, out0_im_0_7);

            out0_re_0_7.store_slice(&mut reals_s0[0..8]);
            out0_im_0_7.store_slice(&mut imags_s0[0..8]);
            out1_re_0_7.store_slice(&mut reals_s1[0..8]);
            out1_im_0_7.store_slice(&mut imags_s1[0..8]);

            // Process second 8 butterflies
            let in0_re_8_15 = f64x8::from_slice(simd, &reals_s0[8..16]);
            let in1_re_8_15 = f64x8::from_slice(simd, &reals_s1[8..16]);
            let in0_im_8_15 = f64x8::from_slice(simd, &imags_s0[8..16]);
            let in1_im_8_15 = f64x8::from_slice(simd, &imags_s1[8..16]);

            let out0_re_8_15 = twiddle_im_8_15.mul_add(
                -in1_im_8_15,
                twiddle_re_8_15.mul_add(in1_re_8_15, in0_re_8_15),
            );
            let out0_im_8_15 = twiddle_im_8_15.mul_add(
                in1_re_8_15,
                twiddle_re_8_15.mul_add(in1_im_8_15, in0_im_8_15),
            );

            let out1_re_8_15 = two.mul_sub(in0_re_8_15, out0_re_8_15);
            let out1_im_8_15 = two.mul_sub(in0_im_8_15, out0_im_8_15);

            out0_re_8_15.store_slice(&mut reals_s0[8..16]);
            out0_im_8_15.store_slice(&mut imags_s0[8..16]);
            out1_re_8_15.store_slice(&mut reals_s1[8..16]);
            out1_im_8_15.store_slice(&mut imags_s1[8..16]);
        });
}

/// DIT butterfly for chunk_size == 32 (f32)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_32_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_32_simd_f32(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 32 (f32)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_32_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 16;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f32x16::splat(simd, 2.0);

    // All 16 twiddle factors for W_32^k where k = 0..15
    let twiddle_re = f32x16::simd_from(
        simd,
        [
            1.0_f32,                         // W_32^0 = 1
            0.980_785_25_f32,                // W_32^1 = cos(π/16)
            0.923_879_5_f32,                 // W_32^2 = cos(π/8)
            0.831_469_6_f32,                 // W_32^3 = cos(3π/16)
            std::f32::consts::FRAC_1_SQRT_2, // W_32^4 = sqrt(2)/2
            0.555_570_24_f32,                // W_32^5 = cos(5π/16)
            0.382_683_43_f32,                // W_32^6 = cos(3π/8)
            0.195_090_32_f32,                // W_32^7 = cos(7π/16)
            0.0_f32,                         // W_32^8 = 0 - i
            -0.195_090_32_f32,               // W_32^9
            -0.382_683_43_f32,               // W_32^10
            -0.555_570_24_f32,               // W_32^11
            -f32::consts::FRAC_1_SQRT_2,     // W_32^12
            -0.831_469_6_f32,                // W_32^13
            -0.923_879_5_f32,                // W_32^14
            -0.980_785_25_f32,               // W_32^15
        ],
    );

    let twiddle_im = f32x16::simd_from(
        simd,
        [
            0.0_f32,                          // W_32^0
            -0.195_090_32_f32,                // W_32^1 = -sin(π/16)
            -0.382_683_43_f32,                // W_32^2 = -sin(π/8)
            -0.555_570_24_f32,                // W_32^3 = -sin(3π/16)
            -std::f32::consts::FRAC_1_SQRT_2, // W_32^4 = -sqrt(2)/2
            -0.831_469_6_f32,                 // W_32^5 = -sin(5π/16)
            -0.923_879_5_f32,                 // W_32^6 = -sin(3π/8)
            -0.980_785_25_f32,                // W_32^7 = -sin(7π/16)
            -1.0_f32,                         // W_32^8
            -0.980_785_25_f32,                // W_32^9
            -0.923_879_5_f32,                 // W_32^10
            -0.831_469_6_f32,                 // W_32^11
            -std::f32::consts::FRAC_1_SQRT_2, // W_32^12
            -0.555_570_24_f32,                // W_32^13
            -0.382_683_43_f32,                // W_32^14
            -0.195_090_32_f32,                // W_32^15
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process all 16 butterflies at once with f32x16
            let in0_re = f32x16::from_slice(simd, &reals_s0[0..16]);
            let in1_re = f32x16::from_slice(simd, &reals_s1[0..16]);
            let in0_im = f32x16::from_slice(simd, &imags_s0[0..16]);
            let in1_im = f32x16::from_slice(simd, &imags_s1[0..16]);

            let out0_re = twiddle_im.mul_add(-in1_im, twiddle_re.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im.mul_add(in1_re, twiddle_re.mul_add(in1_im, in0_im));

            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(reals_s0);
            out0_im.store_slice(imags_s0);
            out1_re.store_slice(reals_s1);
            out1_im.store_slice(imags_s1);
        });
}

/// DIT butterfly for chunk_size == 64 (f64)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_64_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_64_simd_f64(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 64 (f64)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_64_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 32;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f64x8::splat(simd, 2.0);

    // Process in 4 iterations of 8 butterflies each
    // Twiddles for W_64^k where k = 0..7
    let twiddle_re_0_7 = f64x8::simd_from(
        simd,
        [
            1.0,                // W_64^0 = 1
            0.9951847266721969, // W_64^1 = cos(π/32)
            0.9807852804032304, // W_64^2 = cos(π/16)
            0.9569403357322089, // W_64^3 = cos(3π/32)
            0.9238795325112867, // W_64^4 = cos(π/8)
            0.8819212643483549, // W_64^5 = cos(5π/32)
            0.8314696123025452, // W_64^6 = cos(3π/16)
            0.773010453362737,  // W_64^7 = cos(7π/32)
        ],
    );

    let twiddle_im_0_7 = f64x8::simd_from(
        simd,
        [
            0.0,                  // W_64^0
            -0.0980171403295606,  // W_64^1 = -sin(π/32)
            -0.19509032201612825, // W_64^2 = -sin(π/16)
            -0.29028467725446233, // W_64^3 = -sin(3π/32)
            -0.3826834323650898,  // W_64^4 = -sin(π/8)
            -0.47139673682599764, // W_64^5 = -sin(5π/32)
            -0.5555702330196022,  // W_64^6 = -sin(3π/16)
            -0.6343932841636455,  // W_64^7 = -sin(7π/32)
        ],
    );

    // Twiddles for k = 8..15
    let twiddle_re_8_15 = f64x8::simd_from(
        simd,
        [
            std::f64::consts::FRAC_1_SQRT_2, // W_64^8 = sqrt(2)/2
            0.6343932841636455,              // W_64^9
            0.5555702330196022,              // W_64^10
            0.47139673682599764,             // W_64^11
            0.3826834323650898,              // W_64^12
            0.29028467725446233,             // W_64^13
            0.19509032201612825,             // W_64^14
            0.0980171403295606,              // W_64^15
        ],
    );

    let twiddle_im_8_15 = f64x8::simd_from(
        simd,
        [
            -std::f64::consts::FRAC_1_SQRT_2, // W_64^8
            -0.773010453362737,               // W_64^9
            -0.8314696123025452,              // W_64^10
            -0.8819212643483549,              // W_64^11
            -0.9238795325112867,              // W_64^12
            -0.9569403357322089,              // W_64^13
            -0.9807852804032304,              // W_64^14
            -0.9951847266721969,              // W_64^15
        ],
    );

    // Twiddles for k = 16..23
    let twiddle_re_16_23 = f64x8::simd_from(
        simd,
        [
            0.0,                  // W_64^16 = -i
            -0.0980171403295606,  // W_64^17
            -0.19509032201612825, // W_64^18
            -0.29028467725446233, // W_64^19
            -0.3826834323650898,  // W_64^20
            -0.47139673682599764, // W_64^21
            -0.5555702330196022,  // W_64^22
            -0.6343932841636455,  // W_64^23
        ],
    );

    let twiddle_im_16_23 = f64x8::simd_from(
        simd,
        [
            -1.0,                // W_64^16
            -0.9951847266721969, // W_64^17
            -0.9807852804032304, // W_64^18
            -0.9569403357322089, // W_64^19
            -0.9238795325112867, // W_64^20
            -0.8819212643483549, // W_64^21
            -0.8314696123025452, // W_64^22
            -0.773010453362737,  // W_64^23
        ],
    );

    // Twiddles for k = 24..31
    let twiddle_re_24_31 = f64x8::simd_from(
        simd,
        [
            -std::f64::consts::FRAC_1_SQRT_2, // W_64^24
            -0.773010453362737,               // W_64^25
            -0.8314696123025452,              // W_64^26
            -0.8819212643483549,              // W_64^27
            -0.9238795325112867,              // W_64^28
            -0.9569403357322089,              // W_64^29
            -0.9807852804032304,              // W_64^30
            -0.9951847266721969,              // W_64^31
        ],
    );

    let twiddle_im_24_31 = f64x8::simd_from(
        simd,
        [
            -std::f64::consts::FRAC_1_SQRT_2, // W_64^24
            -0.6343932841636455,              // W_64^25
            -0.5555702330196022,              // W_64^26
            -0.47139673682599764,             // W_64^27
            -0.3826834323650898,              // W_64^28
            -0.29028467725446233,             // W_64^29
            -0.19509032201612825,             // W_64^30
            -0.0980171403295606,              // W_64^31
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process butterflies 0..7
            let in0_re = f64x8::from_slice(simd, &reals_s0[0..8]);
            let in1_re = f64x8::from_slice(simd, &reals_s1[0..8]);
            let in0_im = f64x8::from_slice(simd, &imags_s0[0..8]);
            let in1_im = f64x8::from_slice(simd, &imags_s1[0..8]);

            let out0_re = twiddle_im_0_7.mul_add(-in1_im, twiddle_re_0_7.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im_0_7.mul_add(in1_re, twiddle_re_0_7.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut reals_s0[0..8]);
            out0_im.store_slice(&mut imags_s0[0..8]);
            out1_re.store_slice(&mut reals_s1[0..8]);
            out1_im.store_slice(&mut imags_s1[0..8]);

            // Process butterflies 8..15
            let in0_re = f64x8::from_slice(simd, &reals_s0[8..16]);
            let in1_re = f64x8::from_slice(simd, &reals_s1[8..16]);
            let in0_im = f64x8::from_slice(simd, &imags_s0[8..16]);
            let in1_im = f64x8::from_slice(simd, &imags_s1[8..16]);

            let out0_re = twiddle_im_8_15.mul_add(-in1_im, twiddle_re_8_15.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im_8_15.mul_add(in1_re, twiddle_re_8_15.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut reals_s0[8..16]);
            out0_im.store_slice(&mut imags_s0[8..16]);
            out1_re.store_slice(&mut reals_s1[8..16]);
            out1_im.store_slice(&mut imags_s1[8..16]);

            // Process butterflies 16..23
            let in0_re = f64x8::from_slice(simd, &reals_s0[16..24]);
            let in1_re = f64x8::from_slice(simd, &reals_s1[16..24]);
            let in0_im = f64x8::from_slice(simd, &imags_s0[16..24]);
            let in1_im = f64x8::from_slice(simd, &imags_s1[16..24]);

            let out0_re =
                twiddle_im_16_23.mul_add(-in1_im, twiddle_re_16_23.mul_add(in1_re, in0_re));
            let out0_im =
                twiddle_im_16_23.mul_add(in1_re, twiddle_re_16_23.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut reals_s0[16..24]);
            out0_im.store_slice(&mut imags_s0[16..24]);
            out1_re.store_slice(&mut reals_s1[16..24]);
            out1_im.store_slice(&mut imags_s1[16..24]);

            // Process butterflies 24..31
            let in0_re = f64x8::from_slice(simd, &reals_s0[24..32]);
            let in1_re = f64x8::from_slice(simd, &reals_s1[24..32]);
            let in0_im = f64x8::from_slice(simd, &imags_s0[24..32]);
            let in1_im = f64x8::from_slice(simd, &imags_s1[24..32]);

            let out0_re =
                twiddle_im_24_31.mul_add(-in1_im, twiddle_re_24_31.mul_add(in1_re, in0_re));
            let out0_im =
                twiddle_im_24_31.mul_add(in1_re, twiddle_re_24_31.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut reals_s0[24..32]);
            out0_im.store_slice(&mut imags_s0[24..32]);
            out1_re.store_slice(&mut reals_s1[24..32]);
            out1_im.store_slice(&mut imags_s1[24..32]);
        });
}

/// DIT butterfly for chunk_size == 64 (f32)
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_64_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_64_simd_f32(simd, reals, imags),
    )
}

/// DIT butterfly for chunk_size == 64 (f32)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_64_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 32;
    const CHUNK_SIZE: usize = DIST * 2;

    let two = f32x16::splat(simd, 2.0);

    // Process in 2 iterations of 16 butterflies each
    // Twiddles for W_64^k where k = 0..15
    let twiddle_re_0_15 = f32x16::simd_from(
        simd,
        [
            1.0_f32,                         // W_64^0 = 1
            0.995_184_7_f32,                 // W_64^1 = cos(π/32)
            0.980_785_25_f32,                // W_64^2 = cos(π/16)
            0.956_940_35_f32,                // W_64^3 = cos(3π/32)
            0.923_879_5_f32,                 // W_64^4 = cos(π/8)
            0.881_921_3_f32,                 // W_64^5 = cos(5π/32)
            0.831_469_6_f32,                 // W_64^6 = cos(3π/16)
            0.773_010_43_f32,                // W_64^7 = cos(7π/32)
            std::f32::consts::FRAC_1_SQRT_2, // W_64^8 = sqrt(2)/2
            0.634_393_3_f32,                 // W_64^9
            0.555_570_24_f32,                // W_64^10
            0.471_396_74_f32,                // W_64^11
            0.382_683_43_f32,                // W_64^12
            0.290_284_66_f32,                // W_64^13
            0.195_090_32_f32,                // W_64^14
            0.098_017_14_f32,                // W_64^15
        ],
    );

    let twiddle_im_0_15 = f32x16::simd_from(
        simd,
        [
            0.0_f32,                          // W_64^0
            -0.098_017_14_f32,                // W_64^1 = -sin(π/32)
            -0.195_090_32_f32,                // W_64^2 = -sin(π/16)
            -0.290_284_66_f32,                // W_64^3 = -sin(3π/32)
            -0.382_683_43_f32,                // W_64^4 = -sin(π/8)
            -0.471_396_74_f32,                // W_64^5 = -sin(5π/32)
            -0.555_570_24_f32,                // W_64^6 = -sin(3π/16)
            -0.634_393_3_f32,                 // W_64^7 = -sin(7π/32)
            -std::f32::consts::FRAC_1_SQRT_2, // W_64^8
            -0.773_010_43_f32,                // W_64^9
            -0.831_469_6_f32,                 // W_64^10
            -0.881_921_3_f32,                 // W_64^11
            -0.923_879_5_f32,                 // W_64^12
            -0.956_940_35_f32,                // W_64^13
            -0.980_785_25_f32,                // W_64^14
            -0.995_184_7_f32,                 // W_64^15
        ],
    );

    // Twiddles for k = 16..31
    let twiddle_re_16_31 = f32x16::simd_from(
        simd,
        [
            0.0_f32,                          // W_64^16 = -i
            -0.098_017_14_f32,                // W_64^17
            -0.195_090_32_f32,                // W_64^18
            -0.290_284_66_f32,                // W_64^19
            -0.382_683_43_f32,                // W_64^20
            -0.471_396_74_f32,                // W_64^21
            -0.555_570_24_f32,                // W_64^22
            -0.634_393_3_f32,                 // W_64^23
            -std::f32::consts::FRAC_1_SQRT_2, // W_64^24
            -0.773_010_43_f32,                // W_64^25
            -0.831_469_6_f32,                 // W_64^26
            -0.881_921_3_f32,                 // W_64^27
            -0.923_879_5_f32,                 // W_64^28
            -0.956_940_35_f32,                // W_64^29
            -0.980_785_25_f32,                // W_64^30
            -0.995_184_7_f32,                 // W_64^31
        ],
    );

    let twiddle_im_16_31 = f32x16::simd_from(
        simd,
        [
            -1.0_f32,                         // W_64^16
            -0.995_184_7_f32,                 // W_64^17
            -0.980_785_25_f32,                // W_64^18
            -0.956_940_35_f32,                // W_64^19
            -0.923_879_5_f32,                 // W_64^20
            -0.881_921_3_f32,                 // W_64^21
            -0.831_469_6_f32,                 // W_64^22
            -0.773_010_43_f32,                // W_64^23
            -std::f32::consts::FRAC_1_SQRT_2, // W_64^24
            -0.634_393_3_f32,                 // W_64^25
            -0.555_570_24_f32,                // W_64^26
            -0.471_396_74_f32,                // W_64^27
            -0.382_683_43_f32,                // W_64^28
            -0.290_284_66_f32,                // W_64^29
            -0.195_090_32_f32,                // W_64^30
            -0.098_017_14_f32,                // W_64^31
        ],
    );

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process butterflies 0..15
            let in0_re = f32x16::from_slice(simd, &reals_s0[0..16]);
            let in1_re = f32x16::from_slice(simd, &reals_s1[0..16]);
            let in0_im = f32x16::from_slice(simd, &imags_s0[0..16]);
            let in1_im = f32x16::from_slice(simd, &imags_s1[0..16]);

            let out0_re = twiddle_im_0_15.mul_add(-in1_im, twiddle_re_0_15.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im_0_15.mul_add(in1_re, twiddle_re_0_15.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut reals_s0[0..16]);
            out0_im.store_slice(&mut imags_s0[0..16]);
            out1_re.store_slice(&mut reals_s1[0..16]);
            out1_im.store_slice(&mut imags_s1[0..16]);

            // Process butterflies 16..31
            let in0_re = f32x16::from_slice(simd, &reals_s0[16..32]);
            let in1_re = f32x16::from_slice(simd, &reals_s1[16..32]);
            let in0_im = f32x16::from_slice(simd, &imags_s0[16..32]);
            let in1_im = f32x16::from_slice(simd, &imags_s1[16..32]);

            let out0_re =
                twiddle_im_16_31.mul_add(-in1_im, twiddle_re_16_31.mul_add(in1_re, in0_re));
            let out0_im =
                twiddle_im_16_31.mul_add(in1_re, twiddle_re_16_31.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut reals_s0[16..32]);
            out0_im.store_slice(&mut imags_s0[16..32]);
            out1_re.store_slice(&mut reals_s1[16..32]);
            out1_im.store_slice(&mut imags_s1[16..32]);
        });
}

/// General DIT butterfly for f64
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_n_f64<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_re: &[f64],
    twiddles_im: &[f64],
    dist: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_n_simd_f64(simd, reals, imags, twiddles_re, twiddles_im, dist),
    )
}

/// General DIT butterfly for f64
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_n_simd_f64<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_re: &[f64],
    twiddles_im: &[f64],
    dist: usize,
) {
    const LANES: usize = 8;
    let chunk_size = dist * 2;
    assert!(chunk_size >= LANES * 2);

    // The structure of outer for loop with inner for_each is intentional: on x86
    // fearless_simd needs inlining all the way down to intrinsics to work properly,
    // and using for_each in the outer scope would require a call to for_each function
    // that we can't force to be inlined. This brings over the LLVM inlining thresholds
    // and then we end up with a `call` instruction for every FMA, destroying performance.
    // However, the inner loop being for_each has better codegen.
    // I haven't investigated why. Might have something to do with exterior vs interior iteration.
    for (reals_chunk, imags_chunk) in reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
    {
        let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
        let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

        (reals_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(reals_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(twiddles_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_im.as_chunks::<LANES>().0.iter())
            .for_each(|(((((re_s0, re_s1), im_s0), im_s1), tw_re), tw_im)| {
                let two = f64x8::splat(simd, 2.0);
                let in0_re = f64x8::simd_from(simd, *re_s0);
                let in1_re = f64x8::simd_from(simd, *re_s1);
                let in0_im = f64x8::simd_from(simd, *im_s0);
                let in1_im = f64x8::simd_from(simd, *im_s1);

                let tw_re = f64x8::simd_from(simd, *tw_re);
                let tw_im = f64x8::simd_from(simd, *tw_im);

                // out0.re = (in0.re + tw_re * in1.re) - tw_im * in1.im
                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                // out0.im = (in0.im + tw_re * in1.im) + tw_im * in1.re
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));

                // Use FMA for out1 = 2*in0 - out0
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_s0);
                out0_im.store_slice(im_s0);
                out1_re.store_slice(re_s1);
                out1_im.store_slice(im_s1);
            });
    }
}

/// General DIT butterfly for f64 (narrow SIMD: f64x4)
///
/// Uses 4-lane SIMD instead of 8-lane to reduce register pressure.
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_n_f64_narrow<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_re: &[f64],
    twiddles_im: &[f64],
    dist: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_n_simd_f64_narrow(simd, reals, imags, twiddles_re, twiddles_im, dist),
    )
}

/// General DIT butterfly for f64 (narrow SIMD: f64x4)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_n_simd_f64_narrow<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_re: &[f64],
    twiddles_im: &[f64],
    dist: usize,
) {
    const LANES: usize = 4;
    let chunk_size = dist * 2;
    assert!(chunk_size >= LANES * 2);

    // see fft_dit_chunk_n_simd_f64 for an explanation of this structure
    for (reals_chunk, imags_chunk) in reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
    {
        let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
        let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

        (reals_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(reals_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(twiddles_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_im.as_chunks::<LANES>().0.iter())
            .for_each(|(((((re_s0, re_s1), im_s0), im_s1), tw_re), tw_im)| {
                let two = f64x4::splat(simd, 2.0);
                let in0_re = f64x4::simd_from(simd, *re_s0);
                let in1_re = f64x4::simd_from(simd, *re_s1);
                let in0_im = f64x4::simd_from(simd, *im_s0);
                let in1_im = f64x4::simd_from(simd, *im_s1);

                let tw_re = f64x4::simd_from(simd, *tw_re);
                let tw_im = f64x4::simd_from(simd, *tw_im);

                // out0.re = (in0.re + tw_re * in1.re) - tw_im * in1.im
                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                // out0.im = (in0.im + tw_re * in1.im) + tw_im * in1.re
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));

                // Use FMA for out1 = 2*in0 - out0
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_s0);
                out0_im.store_slice(im_s0);
                out1_re.store_slice(re_s1);
                out1_im.store_slice(im_s1);
            });
    }
}

/// General DIT butterfly for f32
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_n_f32<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    dist: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_n_simd_f32(simd, reals, imags, twiddles_re, twiddles_im, dist),
    )
}

/// General DIT butterfly for f32
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_n_simd_f32<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    dist: usize,
) {
    const LANES: usize = 16;
    let chunk_size = dist * 2;
    assert!(chunk_size >= LANES * 2);

    // see fft_dit_chunk_n_simd_f64 for an explanation of this structure
    for (reals_chunk, imags_chunk) in reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
    {
        let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
        let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

        (reals_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(reals_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(twiddles_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_im.as_chunks::<LANES>().0.iter())
            .for_each(|(((((re_s0, re_s1), im_s0), im_s1), tw_re), tw_im)| {
                let two = f32x16::splat(simd, 2.0);
                let in0_re = f32x16::simd_from(simd, *re_s0);
                let in1_re = f32x16::simd_from(simd, *re_s1);
                let in0_im = f32x16::simd_from(simd, *im_s0);
                let in1_im = f32x16::simd_from(simd, *im_s1);

                let tw_re = f32x16::simd_from(simd, *tw_re);
                let tw_im = f32x16::simd_from(simd, *tw_im);

                // out0.re = (in0.re + tw_re * in1.re) - tw_im * in1.im
                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                // out0.im = (in0.im + tw_re * in1.im) + tw_im * in1.re
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));

                // Use FMA for out1 = 2*in0 - out0
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_s0);
                out0_im.store_slice(im_s0);
                out1_re.store_slice(re_s1);
                out1_im.store_slice(im_s1);
            });
    }
}

/// General DIT butterfly for f32 (narrow SIMD: f32x8)
///
/// Uses 8-lane SIMD instead of 16-lane to reduce register pressure.
#[inline(never)] // otherwise every kernel gets inlined into the parent
pub fn fft_dit_chunk_n_f32_narrow<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    dist: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_chunk_n_simd_f32_narrow(simd, reals, imags, twiddles_re, twiddles_im, dist),
    )
}

/// General DIT butterfly for f32 (narrow SIMD: f32x8)
#[inline(always)] // required by fearless_simd
fn fft_dit_chunk_n_simd_f32_narrow<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    dist: usize,
) {
    const LANES: usize = 8;
    let chunk_size = dist * 2;
    assert!(chunk_size >= LANES * 2);

    // see fft_dit_chunk_n_simd_f64 for an explanation of this structure
    for (reals_chunk, imags_chunk) in reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
    {
        let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
        let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

        (reals_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(reals_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s0.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(imags_s1.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(twiddles_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_im.as_chunks::<LANES>().0.iter())
            .for_each(|(((((re_s0, re_s1), im_s0), im_s1), tw_re), tw_im)| {
                let two = f32x8::splat(simd, 2.0);
                let in0_re = f32x8::simd_from(simd, *re_s0);
                let in1_re = f32x8::simd_from(simd, *re_s1);
                let in0_im = f32x8::simd_from(simd, *im_s0);
                let in1_im = f32x8::simd_from(simd, *im_s1);

                let tw_re = f32x8::simd_from(simd, *tw_re);
                let tw_im = f32x8::simd_from(simd, *tw_im);

                // out0.re = (in0.re + tw_re * in1.re) - tw_im * in1.im
                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                // out0.im = (in0.im + tw_re * in1.im) + tw_im * in1.re
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));

                // Use FMA for out1 = 2*in0 - out0
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_s0);
                out0_im.store_slice(im_s0);
                out1_re.store_slice(re_s1);
                out1_im.store_slice(im_s1);
            });
    }
}

/// Fused two-stage DIT butterfly for f64 (radix-2², narrow SIMD: f64x4)
///
/// Processes two consecutive DIT stages in a single pass over memory,
/// halving memory traffic for the fused stages. Uses the radix-2² identity
/// `W_{4D}^{k+D} = -j * W_{4D}^k` to derive the second-half twiddle factors
/// trivially.
///
/// `twiddles_s` are for stage s (size dist_s), `twiddles_s1` are for stage s+1
/// (only first dist_s elements needed; second half derived via -j multiply).
#[inline(never)]
pub fn fft_dit_fused_2stage_f64_narrow<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_s_re: &[f64],
    twiddles_s_im: &[f64],
    twiddles_s1_re: &[f64],
    twiddles_s1_im: &[f64],
    dist_s: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            fft_dit_fused_2stage_simd_f64_narrow(
                simd,
                reals,
                imags,
                twiddles_s_re,
                twiddles_s_im,
                twiddles_s1_re,
                twiddles_s1_im,
                dist_s,
            )
        },
    )
}

/// Parallel fused two-stage DIT butterfly for f64 (radix-2², narrow SIMD: f64x4)
///
/// Same as [`fft_dit_fused_2stage_f64_narrow`] but parallelizes the inner loop
/// using Rayon. Intended for the last stage of the FFT which spans the entire array
/// and cannot be parallelized through recursion.
#[cfg(feature = "parallel")]
#[inline(never)]
pub fn fft_dit_fused_2stage_f64_narrow_parallel<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_s_re: &[f64],
    twiddles_s_im: &[f64],
    twiddles_s1_re: &[f64],
    twiddles_s1_im: &[f64],
    dist_s: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            fft_dit_fused_2stage_simd_f64_narrow_parallel(
                simd,
                reals,
                imags,
                twiddles_s_re,
                twiddles_s_im,
                twiddles_s1_re,
                twiddles_s1_im,
                dist_s,
            )
        },
    )
}

/// Fused two-stage DIT butterfly for f64 (radix-2², narrow SIMD: f64x4)
#[inline(always)]
fn fft_dit_fused_2stage_simd_f64_narrow<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_s_re: &[f64],
    twiddles_s_im: &[f64],
    twiddles_s1_re: &[f64],
    twiddles_s1_im: &[f64],
    dist_s: usize,
) {
    const LANES: usize = 4;
    let block_size = dist_s * 4; // two stages: 4 sub-blocks of dist_s
    assert!(dist_s >= LANES);

    // Outer loop over blocks of 4*dist_s elements.
    // Using `for` (not `for_each`) for the same inlining reasons
    // as in fft_dit_chunk_n_simd_f64.
    for (reals_block, imags_block) in reals
        .chunks_exact_mut(block_size)
        .zip(imags.chunks_exact_mut(block_size))
    {
        // Split block into 4 sub-blocks of dist_s elements:
        // A = [0..D), B = [D..2D), C = [2D..3D), D_blk = [3D..4D)
        let (re_ab, re_cd) = reals_block.split_at_mut(dist_s * 2);
        let (im_ab, im_cd) = imags_block.split_at_mut(dist_s * 2);
        let (re_a, re_b) = re_ab.split_at_mut(dist_s);
        let (im_a, im_b) = im_ab.split_at_mut(dist_s);
        let (re_c, re_d) = re_cd.split_at_mut(dist_s);
        let (im_c, im_d) = im_cd.split_at_mut(dist_s);

        (re_a.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(re_b.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(re_c.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(re_d.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_a.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_b.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_c.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_d.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(twiddles_s_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_s_im.as_chunks::<LANES>().0.iter())
            .zip(twiddles_s1_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_s1_im.as_chunks::<LANES>().0.iter())
            .for_each(
                |(
                    ((((((((((ra, rb), rc), rd), ia), ib), ic), id), tw_s_re), tw_s_im), tw1_re),
                    tw1_im,
                )| {
                    let two = f64x4::splat(simd, 2.0);

                    // Load 4 data sub-blocks
                    let a_re = f64x4::simd_from(simd, *ra);
                    let a_im = f64x4::simd_from(simd, *ia);
                    let b_re = f64x4::simd_from(simd, *rb);
                    let b_im = f64x4::simd_from(simd, *ib);
                    let c_re = f64x4::simd_from(simd, *rc);
                    let c_im = f64x4::simd_from(simd, *ic);
                    let d_re = f64x4::simd_from(simd, *rd);
                    let d_im = f64x4::simd_from(simd, *id);

                    // Load stage-s twiddles
                    let tws_re = f64x4::simd_from(simd, *tw_s_re);
                    let tws_im = f64x4::simd_from(simd, *tw_s_im);

                    // Stage s butterfly: (A,B) pair
                    // A' = A + tw_s * B,  B' = A - tw_s * B
                    let ap_re = tws_im.mul_add(-b_im, tws_re.mul_add(b_re, a_re));
                    let ap_im = tws_im.mul_add(b_re, tws_re.mul_add(b_im, a_im));
                    let bp_re = two.mul_sub(a_re, ap_re);
                    let bp_im = two.mul_sub(a_im, ap_im);

                    // Stage s butterfly: (C,D) pair
                    // C' = C + tw_s * D,  D' = C - tw_s * D
                    let cp_re = tws_im.mul_add(-d_im, tws_re.mul_add(d_re, c_re));
                    let cp_im = tws_im.mul_add(d_re, tws_re.mul_add(d_im, c_im));
                    let dp_re = two.mul_sub(c_re, cp_re);
                    let dp_im = two.mul_sub(c_im, cp_im);

                    // Load stage-(s+1) twiddles (first half only)
                    let tw1r = f64x4::simd_from(simd, *tw1_re);
                    let tw1i = f64x4::simd_from(simd, *tw1_im);

                    // Stage s+1 butterfly: (A', C') pair with tw_{s+1}[k]
                    // out_A = A' + tw1 * C',  out_C = A' - tw1 * C'
                    let out_a_re = tw1i.mul_add(-cp_im, tw1r.mul_add(cp_re, ap_re));
                    let out_a_im = tw1i.mul_add(cp_re, tw1r.mul_add(cp_im, ap_im));
                    let out_c_re = two.mul_sub(ap_re, out_a_re);
                    let out_c_im = two.mul_sub(ap_im, out_a_im);

                    // Derive -j * tw1: (-j)(w_re + i*w_im) = w_im - i*w_re
                    let nj_tw1r = tw1i;
                    let nj_tw1i = -tw1r;

                    // Stage s+1 butterfly: (B', D') pair with -j * tw_{s+1}[k]
                    // out_B = B' + (-j*tw1) * D',  out_D = B' - (-j*tw1) * D'
                    let out_b_re = nj_tw1i.mul_add(-dp_im, nj_tw1r.mul_add(dp_re, bp_re));
                    let out_b_im = nj_tw1i.mul_add(dp_re, nj_tw1r.mul_add(dp_im, bp_im));
                    let out_d_re = two.mul_sub(bp_re, out_b_re);
                    let out_d_im = two.mul_sub(bp_im, out_b_im);

                    // Store results
                    out_a_re.store_slice(ra);
                    out_a_im.store_slice(ia);
                    out_b_re.store_slice(rb);
                    out_b_im.store_slice(ib);
                    out_c_re.store_slice(rc);
                    out_c_im.store_slice(ic);
                    out_d_re.store_slice(rd);
                    out_d_im.store_slice(id);
                },
            );
    }
}

/// Fused two-stage DIT butterfly for f64 (radix-2², narrow SIMD: f64x4) that runs multi-threaded
#[cfg(feature = "parallel")]
#[inline(always)]
fn fft_dit_fused_2stage_simd_f64_narrow_parallel<S: Simd>(
    simd: S,
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_s_re: &[f64],
    twiddles_s_im: &[f64],
    twiddles_s1_re: &[f64],
    twiddles_s1_im: &[f64],
    dist_s: usize,
) {
    use rayon::prelude::*;
    const LANES: usize = 4;
    let block_size = dist_s * 4; // two stages: 4 sub-blocks of dist_s
    assert!(dist_s >= LANES);

    // Outer loop over blocks of 4*dist_s elements.
    // Using `for` (not `for_each`) for the same inlining reasons
    // as in fft_dit_chunk_n_simd_f64.
    for (reals_block, imags_block) in reals
        .chunks_exact_mut(block_size)
        .zip(imags.chunks_exact_mut(block_size))
    {
        // Split block into 4 sub-blocks of dist_s elements:
        // A = [0..D), B = [D..2D), C = [2D..3D), D_blk = [3D..4D)
        let (re_ab, re_cd) = reals_block.split_at_mut(dist_s * 2);
        let (im_ab, im_cd) = imags_block.split_at_mut(dist_s * 2);
        let (re_a, re_b) = re_ab.split_at_mut(dist_s);
        let (im_a, im_b) = im_ab.split_at_mut(dist_s);
        let (re_c, re_d) = re_cd.split_at_mut(dist_s);
        let (im_c, im_d) = im_cd.split_at_mut(dist_s);

        (re_a.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(re_b.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(re_c.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(re_d.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(im_a.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(im_b.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(im_c.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(im_d.as_chunks_mut::<LANES>().0.par_iter_mut())
            .zip(twiddles_s_re.as_chunks::<LANES>().0.par_iter())
            .zip(twiddles_s_im.as_chunks::<LANES>().0.par_iter())
            .zip(twiddles_s1_re.as_chunks::<LANES>().0.par_iter())
            .zip(twiddles_s1_im.as_chunks::<LANES>().0.par_iter())
            .for_each(
                |(
                    ((((((((((ra, rb), rc), rd), ia), ib), ic), id), tw_s_re), tw_s_im), tw1_re),
                    tw1_im,
                )| {
                    simd.vectorize(
                        #[inline(always)]
                        || {
                            let two = f64x4::splat(simd, 2.0);

                            // Load 4 data sub-blocks
                            let a_re = f64x4::simd_from(simd, *ra);
                            let a_im = f64x4::simd_from(simd, *ia);
                            let b_re = f64x4::simd_from(simd, *rb);
                            let b_im = f64x4::simd_from(simd, *ib);
                            let c_re = f64x4::simd_from(simd, *rc);
                            let c_im = f64x4::simd_from(simd, *ic);
                            let d_re = f64x4::simd_from(simd, *rd);
                            let d_im = f64x4::simd_from(simd, *id);

                            // Load stage-s twiddles
                            let tws_re = f64x4::simd_from(simd, *tw_s_re);
                            let tws_im = f64x4::simd_from(simd, *tw_s_im);

                            // Stage s butterfly: (A,B) pair
                            // A' = A + tw_s * B,  B' = A - tw_s * B
                            let ap_re = tws_im.mul_add(-b_im, tws_re.mul_add(b_re, a_re));
                            let ap_im = tws_im.mul_add(b_re, tws_re.mul_add(b_im, a_im));
                            let bp_re = two.mul_sub(a_re, ap_re);
                            let bp_im = two.mul_sub(a_im, ap_im);

                            // Stage s butterfly: (C,D) pair
                            // C' = C + tw_s * D,  D' = C - tw_s * D
                            let cp_re = tws_im.mul_add(-d_im, tws_re.mul_add(d_re, c_re));
                            let cp_im = tws_im.mul_add(d_re, tws_re.mul_add(d_im, c_im));
                            let dp_re = two.mul_sub(c_re, cp_re);
                            let dp_im = two.mul_sub(c_im, cp_im);

                            // Load stage-(s+1) twiddles (first half only)
                            let tw1r = f64x4::simd_from(simd, *tw1_re);
                            let tw1i = f64x4::simd_from(simd, *tw1_im);

                            // Stage s+1 butterfly: (A', C') pair with tw_{s+1}[k]
                            // out_A = A' + tw1 * C',  out_C = A' - tw1 * C'
                            let out_a_re = tw1i.mul_add(-cp_im, tw1r.mul_add(cp_re, ap_re));
                            let out_a_im = tw1i.mul_add(cp_re, tw1r.mul_add(cp_im, ap_im));
                            let out_c_re = two.mul_sub(ap_re, out_a_re);
                            let out_c_im = two.mul_sub(ap_im, out_a_im);

                            // Derive -j * tw1: (-j)(w_re + i*w_im) = w_im - i*w_re
                            let nj_tw1r = tw1i;
                            let nj_tw1i = -tw1r;

                            // Stage s+1 butterfly: (B', D') pair with -j * tw_{s+1}[k]
                            // out_B = B' + (-j*tw1) * D',  out_D = B' - (-j*tw1) * D'
                            let out_b_re = nj_tw1i.mul_add(-dp_im, nj_tw1r.mul_add(dp_re, bp_re));
                            let out_b_im = nj_tw1i.mul_add(dp_re, nj_tw1r.mul_add(dp_im, bp_im));
                            let out_d_re = two.mul_sub(bp_re, out_b_re);
                            let out_d_im = two.mul_sub(bp_im, out_b_im);

                            // Store results
                            out_a_re.store_slice(ra);
                            out_a_im.store_slice(ia);
                            out_b_re.store_slice(rb);
                            out_b_im.store_slice(ib);
                            out_c_re.store_slice(rc);
                            out_c_im.store_slice(ic);
                            out_d_re.store_slice(rd);
                            out_d_im.store_slice(id);
                        },
                    );
                },
            );
    }
}

/// Fused two-stage DIT butterfly for f32 (radix-2², narrow SIMD: f32x8)
///
/// Processes two consecutive DIT stages in a single pass over memory,
/// halving memory traffic for the fused stages. Uses the radix-2² identity
/// `W_{4D}^{k+D} = -j * W_{4D}^k` to derive the second-half twiddle factors
/// trivially.
///
/// `twiddles_s` are for stage s (size dist_s), `twiddles_s1` are for stage s+1
/// (only first dist_s elements needed; second half derived via -j multiply).
#[inline(never)]
pub fn fft_dit_fused_2stage_f32_narrow<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_s_re: &[f32],
    twiddles_s_im: &[f32],
    twiddles_s1_re: &[f32],
    twiddles_s1_im: &[f32],
    dist_s: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            fft_dit_fused_2stage_simd_f32_narrow(
                simd,
                reals,
                imags,
                twiddles_s_re,
                twiddles_s_im,
                twiddles_s1_re,
                twiddles_s1_im,
                dist_s,
            )
        },
    )
}

/// Fused two-stage DIT butterfly for f32 (radix-2², narrow SIMD: f32x8)
#[inline(always)]
fn fft_dit_fused_2stage_simd_f32_narrow<S: Simd>(
    simd: S,
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_s_re: &[f32],
    twiddles_s_im: &[f32],
    twiddles_s1_re: &[f32],
    twiddles_s1_im: &[f32],
    dist_s: usize,
) {
    const LANES: usize = 8;
    let block_size = dist_s * 4;
    assert!(dist_s >= LANES);

    for (reals_block, imags_block) in reals
        .chunks_exact_mut(block_size)
        .zip(imags.chunks_exact_mut(block_size))
    {
        let (re_ab, re_cd) = reals_block.split_at_mut(dist_s * 2);
        let (im_ab, im_cd) = imags_block.split_at_mut(dist_s * 2);
        let (re_a, re_b) = re_ab.split_at_mut(dist_s);
        let (im_a, im_b) = im_ab.split_at_mut(dist_s);
        let (re_c, re_d) = re_cd.split_at_mut(dist_s);
        let (im_c, im_d) = im_cd.split_at_mut(dist_s);

        (re_a.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(re_b.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(re_c.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(re_d.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_a.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_b.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_c.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(im_d.as_chunks_mut::<LANES>().0.iter_mut())
            .zip(twiddles_s_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_s_im.as_chunks::<LANES>().0.iter())
            .zip(twiddles_s1_re.as_chunks::<LANES>().0.iter())
            .zip(twiddles_s1_im.as_chunks::<LANES>().0.iter())
            .for_each(
                |(
                    ((((((((((ra, rb), rc), rd), ia), ib), ic), id), tw_s_re), tw_s_im), tw1_re),
                    tw1_im,
                )| {
                    let two = f32x8::splat(simd, 2.0);

                    let a_re = f32x8::simd_from(simd, *ra);
                    let a_im = f32x8::simd_from(simd, *ia);
                    let b_re = f32x8::simd_from(simd, *rb);
                    let b_im = f32x8::simd_from(simd, *ib);
                    let c_re = f32x8::simd_from(simd, *rc);
                    let c_im = f32x8::simd_from(simd, *ic);
                    let d_re = f32x8::simd_from(simd, *rd);
                    let d_im = f32x8::simd_from(simd, *id);

                    let tws_re = f32x8::simd_from(simd, *tw_s_re);
                    let tws_im = f32x8::simd_from(simd, *tw_s_im);

                    // Stage s butterfly: (A,B) pair
                    let ap_re = tws_im.mul_add(-b_im, tws_re.mul_add(b_re, a_re));
                    let ap_im = tws_im.mul_add(b_re, tws_re.mul_add(b_im, a_im));
                    let bp_re = two.mul_sub(a_re, ap_re);
                    let bp_im = two.mul_sub(a_im, ap_im);

                    // Stage s butterfly: (C,D) pair
                    let cp_re = tws_im.mul_add(-d_im, tws_re.mul_add(d_re, c_re));
                    let cp_im = tws_im.mul_add(d_re, tws_re.mul_add(d_im, c_im));
                    let dp_re = two.mul_sub(c_re, cp_re);
                    let dp_im = two.mul_sub(c_im, cp_im);

                    let tw1r = f32x8::simd_from(simd, *tw1_re);
                    let tw1i = f32x8::simd_from(simd, *tw1_im);

                    // Stage s+1 butterfly: (A', C') pair with tw_{s+1}[k]
                    let out_a_re = tw1i.mul_add(-cp_im, tw1r.mul_add(cp_re, ap_re));
                    let out_a_im = tw1i.mul_add(cp_re, tw1r.mul_add(cp_im, ap_im));
                    let out_c_re = two.mul_sub(ap_re, out_a_re);
                    let out_c_im = two.mul_sub(ap_im, out_a_im);

                    // Derive -j * tw1
                    let nj_tw1r = tw1i;
                    let nj_tw1i = -tw1r;

                    // Stage s+1 butterfly: (B', D') pair with -j * tw_{s+1}[k]
                    let out_b_re = nj_tw1i.mul_add(-dp_im, nj_tw1r.mul_add(dp_re, bp_re));
                    let out_b_im = nj_tw1i.mul_add(dp_re, nj_tw1r.mul_add(dp_im, bp_im));
                    let out_d_re = two.mul_sub(bp_re, out_b_re);
                    let out_d_im = two.mul_sub(bp_im, out_b_im);

                    out_a_re.store_slice(ra);
                    out_a_im.store_slice(ia);
                    out_b_re.store_slice(rb);
                    out_b_im.store_slice(ib);
                    out_c_re.store_slice(rc);
                    out_c_im.store_slice(ic);
                    out_d_re.store_slice(rd);
                    out_d_im.store_slice(id);
                },
            );
    }
}
