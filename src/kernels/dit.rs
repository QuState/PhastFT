//! DIT-specific FFT Kernels
//!
//! FFT kernels for the Decimation-in-Time algorithm.
//!
use core::f32;

use num_traits::Float;
use wide::{f32x16, f32x4, f32x8, f64x4, f64x8};

use crate::kernels::common::fft_chunk_2;

/// DIT butterfly for chunk_size == 2
/// Identical to DIF version (no twiddles at size 2)
pub fn fft_dit_chunk_2<T: Float>(reals: &mut [T], imags: &mut [T]) {
    fft_chunk_2(reals, imags);
}

/// DIT butterfly for chunk_size == 4 (f64)
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_4_simd_f64(reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST << 1;

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
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_4_simd_f32(reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST << 1;

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
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_8_simd_f64(reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 4;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f64x4::splat(2.0);
    let sqrt2_2 = f64x4::new([
        1.0,                              // W_8^0 real
        std::f64::consts::FRAC_1_SQRT_2,  // W_8^1 real (sqrt(2)/2)
        0.0,                              // W_8^2 real
        -std::f64::consts::FRAC_1_SQRT_2, // W_8^3 real (-sqrt(2)/2)
    ]);
    let sqrt2_2_im = f64x4::new([
        0.0,                              // W_8^0 imag
        -std::f64::consts::FRAC_1_SQRT_2, // W_8^1 imag (-sqrt(2)/2)
        -1.0,                             // W_8^2 imag
        -std::f64::consts::FRAC_1_SQRT_2, // W_8^3 imag (-sqrt(2)/2)
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            let in0_re = f64x4::new(reals_s0[0..4].try_into().unwrap());
            let in1_re = f64x4::new(reals_s1[0..4].try_into().unwrap());
            let in0_im = f64x4::new(imags_s0[0..4].try_into().unwrap());
            let in1_im = f64x4::new(imags_s1[0..4].try_into().unwrap());

            // out0.re = (in0.re + w.re * in1.re) - w.im * in1.im
            let out0_re = sqrt2_2_im.mul_add(-in1_im, sqrt2_2.mul_add(in1_re, in0_re));
            // out0.im = (in0.im + w.re*in1.im) + w.im*in1.re
            let out0_im = sqrt2_2_im.mul_add(in1_re, sqrt2_2.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0.copy_from_slice(out0_re.as_array());
            imags_s0.copy_from_slice(out0_im.as_array());
            reals_s1.copy_from_slice(out1_re.as_array());
            imags_s1.copy_from_slice(out1_im.as_array());
        });
}

/// DIT butterfly for chunk_size == 8 (f32) with SIMD
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_8_simd_f32(reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 4;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f32x4::splat(2.0);
    let sqrt2_2 = f32x4::new([
        1.0_f32,                          // W_8^0 real
        std::f32::consts::FRAC_1_SQRT_2,  // W_8^1 real (sqrt(2)/2)
        0.0_f32,                          // W_8^2 real
        -std::f32::consts::FRAC_1_SQRT_2, // W_8^3 real (-sqrt(2)/2)
    ]);
    let sqrt2_2_im = f32x4::new([
        0.0_f32,                          // W_8^0 imag
        -std::f32::consts::FRAC_1_SQRT_2, // W_8^1 imag (-sqrt(2)/2)
        -1.0_f32,                         // W_8^2 imag
        -std::f32::consts::FRAC_1_SQRT_2, // W_8^3 imag (-sqrt(2)/2)
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            let in0_re = f32x4::new(reals_s0[0..4].try_into().unwrap());
            let in1_re = f32x4::new(reals_s1[0..4].try_into().unwrap());
            let in0_im = f32x4::new(imags_s0[0..4].try_into().unwrap());
            let in1_im = f32x4::new(imags_s1[0..4].try_into().unwrap());

            // out0.re = (in0.re + w.re * in1.re) - w.im * in1.im
            let out0_re = sqrt2_2_im.mul_add(-in1_im, sqrt2_2.mul_add(in1_re, in0_re));
            // out0.im = (in0.im + w.re * in1.im) + w.im * in1.re
            let out0_im = sqrt2_2_im.mul_add(in1_re, sqrt2_2.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0.copy_from_slice(out0_re.as_array());
            imags_s0.copy_from_slice(out0_im.as_array());
            reals_s1.copy_from_slice(out1_re.as_array());
            imags_s1.copy_from_slice(out1_im.as_array());
        });
}

/// DIT butterfly for chunk_size == 16 (f64)  
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_16_simd_f64(reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 8;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f64x8::splat(2.0);

    // Twiddle factors for W_16^k where k = 0..7
    let twiddle_re = f64x8::new([
        1.0,                              // W_16^0
        0.9238795325112867,               // W_16^1 = cos(pi/8)
        std::f64::consts::FRAC_1_SQRT_2,  // W_16^2 = sqrt(2)/2
        0.38268343236508984,              // W_16^3 = cos(3*pi/8)
        0.0,                              // W_16^4
        -0.38268343236508984,             // W_16^5 = -cos(3*pi/8)
        -std::f64::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
        -0.9238795325112867,              // W_16^7 = -cos(pi/8)
    ]);

    let twiddle_im = f64x8::new([
        0.0,                              // W_16^0
        -0.38268343236508984,             // W_16^1 = -sin(pi/8)
        -std::f64::consts::FRAC_1_SQRT_2, // W_16^2 = -sqrt(2)/2
        -0.9238795325112867,              // W_16^3 = -sin(3*pi/8)
        -1.0,                             // W_16^4
        -0.9238795325112867,              // W_16^5 = -sin(3*pi/8)
        -std::f64::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
        -0.38268343236508984,             // W_16^7 = -sin(pi/8)
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Load all 8 elements at once
            let in0_re = f64x8::new(reals_s0[0..8].try_into().unwrap());
            let in1_re = f64x8::new(reals_s1[0..8].try_into().unwrap());
            let in0_im = f64x8::new(imags_s0[0..8].try_into().unwrap());
            let in1_im = f64x8::new(imags_s1[0..8].try_into().unwrap());

            let out0_re = twiddle_im.mul_add(-in1_im, twiddle_re.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im.mul_add(in1_re, twiddle_re.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0.copy_from_slice(out0_re.as_array());
            imags_s0.copy_from_slice(out0_im.as_array());
            reals_s1.copy_from_slice(out1_re.as_array());
            imags_s1.copy_from_slice(out1_im.as_array());
        });
}

/// DIT butterfly for chunk_size == 16 (f32)
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_16_simd_f32(reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 8;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f32x8::splat(2.0);

    // Twiddle factors for W_16^k where k = 0..7
    let twiddle_re = f32x8::new([
        1.0_f32,                          // W_16^0
        0.923_879_5_f32,                  // W_16^1 = cos(pi/8)
        std::f32::consts::FRAC_1_SQRT_2,  // W_16^2 = sqrt(2)/2
        0.382_683_43_f32,                 // W_16^3 = cos(3*pi/8)
        0.0_f32,                          // W_16^4
        -0.382_683_43_f32,                // W_16^5 = -cos(3*pi/8)
        -std::f32::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
        -0.923_879_5_f32,                 // W_16^7 = -cos(pi/8)
    ]);

    let twiddle_im = f32x8::new([
        0.0_f32,                          // W_16^0
        -0.382_683_43_f32,                // W_16^1 = -sin(pi/8)
        -std::f32::consts::FRAC_1_SQRT_2, // W_16^2 = -sqrt(2)/2
        -0.923_879_5_f32,                 // W_16^3 = -sin(3*pi/8)
        -1.0_f32,                         // W_16^4
        -0.923_879_5_f32,                 // W_16^5 = -sin(3*pi/8)
        -std::f32::consts::FRAC_1_SQRT_2, // W_16^6 = -sqrt(2)/2
        -0.382_683_43_f32,                // W_16^7 = -sin(pi/8)
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Load all 8 elements at once
            let in0_re = f32x8::new(reals_s0[0..8].try_into().unwrap());
            let in1_re = f32x8::new(reals_s1[0..8].try_into().unwrap());
            let in0_im = f32x8::new(imags_s0[0..8].try_into().unwrap());
            let in1_im = f32x8::new(imags_s1[0..8].try_into().unwrap());

            let out0_re = twiddle_im.mul_add(-in1_im, twiddle_re.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im.mul_add(in1_re, twiddle_re.mul_add(in1_im, in0_im));

            // out1 = 2*in0 - out0
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0.copy_from_slice(out0_re.as_array());
            imags_s0.copy_from_slice(out0_im.as_array());
            reals_s1.copy_from_slice(out1_re.as_array());
            imags_s1.copy_from_slice(out1_im.as_array());
        });
}
/// DIT butterfly for chunk_size == 32 (f64)
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_32_simd_f64(reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 16;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f64x8::splat(2.0);

    // First 8 twiddle factors for W_32^k where k = 0..7
    let twiddle_re_0_7 = f64x8::new([
        1.0,                             // W_32^0 = 1
        0.9807852804032304,              // W_32^1 = cos(π/16)
        0.9238795325112867,              // W_32^2 = cos(π/8)
        0.8314696123025452,              // W_32^3 = cos(3π/16)
        std::f64::consts::FRAC_1_SQRT_2, // W_32^4 = sqrt(2)/2
        0.5555702330196022,              // W_32^5 = cos(5π/16)
        0.3826834323650898,              // W_32^6 = cos(3π/8)
        0.19509032201612825,             // W_32^7 = cos(7π/16)
    ]);

    let twiddle_im_0_7 = f64x8::new([
        0.0,                              // W_32^0
        -0.19509032201612825,             // W_32^1 = -sin(π/16)
        -0.3826834323650898,              // W_32^2 = -sin(π/8)
        -0.5555702330196022,              // W_32^3 = -sin(3π/16)
        -std::f64::consts::FRAC_1_SQRT_2, // W_32^4 = -sqrt(2)/2
        -0.8314696123025452,              // W_32^5 = -sin(5π/16)
        -0.9238795325112867,              // W_32^6 = -sin(3π/8)
        -0.9807852804032304,              // W_32^7 = -sin(7π/16)
    ]);

    // Second 8 twiddle factors for W_32^k where k = 8..15
    let twiddle_re_8_15 = f64x8::new([
        0.0,                              // W_32^8 = 0 - i
        -0.19509032201612825,             // W_32^9
        -0.3826834323650898,              // W_32^10
        -0.5555702330196022,              // W_32^11
        -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
        -0.8314696123025452,              // W_32^13
        -0.9238795325112867,              // W_32^14
        -0.9807852804032304,              // W_32^15
    ]);

    let twiddle_im_8_15 = f64x8::new([
        -1.0,                             // W_32^8
        -0.9807852804032304,              // W_32^9
        -0.9238795325112867,              // W_32^10
        -0.8314696123025452,              // W_32^11
        -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
        -0.5555702330196022,              // W_32^13
        -0.3826834323650898,              // W_32^14
        -0.19509032201612825,             // W_32^15
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process first 8 butterflies
            let in0_re_0_7 = f64x8::new(reals_s0[0..8].try_into().unwrap());
            let in1_re_0_7 = f64x8::new(reals_s1[0..8].try_into().unwrap());
            let in0_im_0_7 = f64x8::new(imags_s0[0..8].try_into().unwrap());
            let in1_im_0_7 = f64x8::new(imags_s1[0..8].try_into().unwrap());

            let out0_re_0_7 = twiddle_im_0_7
                .mul_add(-in1_im_0_7, twiddle_re_0_7.mul_add(in1_re_0_7, in0_re_0_7));
            let out0_im_0_7 =
                twiddle_im_0_7.mul_add(in1_re_0_7, twiddle_re_0_7.mul_add(in1_im_0_7, in0_im_0_7));

            let out1_re_0_7 = two.mul_sub(in0_re_0_7, out0_re_0_7);
            let out1_im_0_7 = two.mul_sub(in0_im_0_7, out0_im_0_7);

            reals_s0[0..8].copy_from_slice(out0_re_0_7.as_array());
            imags_s0[0..8].copy_from_slice(out0_im_0_7.as_array());
            reals_s1[0..8].copy_from_slice(out1_re_0_7.as_array());
            imags_s1[0..8].copy_from_slice(out1_im_0_7.as_array());

            // Process second 8 butterflies
            let in0_re_8_15 = f64x8::new(reals_s0[8..16].try_into().unwrap());
            let in1_re_8_15 = f64x8::new(reals_s1[8..16].try_into().unwrap());
            let in0_im_8_15 = f64x8::new(imags_s0[8..16].try_into().unwrap());
            let in1_im_8_15 = f64x8::new(imags_s1[8..16].try_into().unwrap());

            let out0_re_8_15 = twiddle_im_8_15.mul_add(-
                in1_im_8_15,
                twiddle_re_8_15.mul_add(in1_re_8_15, in0_re_8_15),
            );
            let out0_im_8_15 = twiddle_im_8_15.mul_add(
                in1_re_8_15,
                twiddle_re_8_15.mul_add(in1_im_8_15, in0_im_8_15),
            );

            let out1_re_8_15 = two.mul_sub(in0_re_8_15, out0_re_8_15);
            let out1_im_8_15 = two.mul_sub(in0_im_8_15, out0_im_8_15);

            reals_s0[8..16].copy_from_slice(out0_re_8_15.as_array());
            imags_s0[8..16].copy_from_slice(out0_im_8_15.as_array());
            reals_s1[8..16].copy_from_slice(out1_re_8_15.as_array());
            imags_s1[8..16].copy_from_slice(out1_im_8_15.as_array());
        });
}

/// DIT butterfly for chunk_size == 32 (f32)
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_32_simd_f32(reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 16;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f32x16::splat(2.0);

    // All 16 twiddle factors for W_32^k where k = 0..15
    let twiddle_re = f32x16::new([
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
    ]);

    let twiddle_im = f32x16::new([
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
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process all 16 butterflies at once with f32x16
            let in0_re = f32x16::new(reals_s0[0..16].try_into().unwrap());
            let in1_re = f32x16::new(reals_s1[0..16].try_into().unwrap());
            let in0_im = f32x16::new(imags_s0[0..16].try_into().unwrap());
            let in1_im = f32x16::new(imags_s1[0..16].try_into().unwrap());

            let out0_re = twiddle_im.mul_add(-in1_im, twiddle_re.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im.mul_add(in1_re, twiddle_re.mul_add(in1_im, in0_im));

            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0.copy_from_slice(out0_re.as_array());
            imags_s0.copy_from_slice(out0_im.as_array());
            reals_s1.copy_from_slice(out1_re.as_array());
            imags_s1.copy_from_slice(out1_im.as_array());
        });
}

/// DIT butterfly for chunk_size == 64 (f64)
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_64_simd_f64(reals: &mut [f64], imags: &mut [f64]) {
    const DIST: usize = 32;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f64x8::splat(2.0);

    // Process in 4 iterations of 8 butterflies each
    // Twiddles for W_64^k where k = 0..7
    let twiddle_re_0_7 = f64x8::new([
        1.0,                // W_64^0 = 1
        0.9951847266721969, // W_64^1 = cos(π/32)
        0.9807852804032304, // W_64^2 = cos(π/16)
        0.9569403357322089, // W_64^3 = cos(3π/32)
        0.9238795325112867, // W_64^4 = cos(π/8)
        0.8819212643483549, // W_64^5 = cos(5π/32)
        0.8314696123025452, // W_64^6 = cos(3π/16)
        0.773010453362737,  // W_64^7 = cos(7π/32)
    ]);

    let twiddle_im_0_7 = f64x8::new([
        0.0,                  // W_64^0
        -0.0980171403295606,  // W_64^1 = -sin(π/32)
        -0.19509032201612825, // W_64^2 = -sin(π/16)
        -0.29028467725446233, // W_64^3 = -sin(3π/32)
        -0.3826834323650898,  // W_64^4 = -sin(π/8)
        -0.47139673682599764, // W_64^5 = -sin(5π/32)
        -0.5555702330196022,  // W_64^6 = -sin(3π/16)
        -0.6343932841636455,  // W_64^7 = -sin(7π/32)
    ]);

    // Twiddles for k = 8..15
    let twiddle_re_8_15 = f64x8::new([
        std::f64::consts::FRAC_1_SQRT_2, // W_64^8 = sqrt(2)/2
        0.6343932841636455,              // W_64^9
        0.5555702330196022,              // W_64^10
        0.47139673682599764,             // W_64^11
        0.3826834323650898,              // W_64^12
        0.29028467725446233,             // W_64^13
        0.19509032201612825,             // W_64^14
        0.0980171403295606,              // W_64^15
    ]);

    let twiddle_im_8_15 = f64x8::new([
        -std::f64::consts::FRAC_1_SQRT_2, // W_64^8
        -0.773010453362737,               // W_64^9
        -0.8314696123025452,              // W_64^10
        -0.8819212643483549,              // W_64^11
        -0.9238795325112867,              // W_64^12
        -0.9569403357322089,              // W_64^13
        -0.9807852804032304,              // W_64^14
        -0.9951847266721969,              // W_64^15
    ]);

    // Twiddles for k = 16..23
    let twiddle_re_16_23 = f64x8::new([
        0.0,                  // W_64^16 = -i
        -0.0980171403295606,  // W_64^17
        -0.19509032201612825, // W_64^18
        -0.29028467725446233, // W_64^19
        -0.3826834323650898,  // W_64^20
        -0.47139673682599764, // W_64^21
        -0.5555702330196022,  // W_64^22
        -0.6343932841636455,  // W_64^23
    ]);

    let twiddle_im_16_23 = f64x8::new([
        -1.0,                // W_64^16
        -0.9951847266721969, // W_64^17
        -0.9807852804032304, // W_64^18
        -0.9569403357322089, // W_64^19
        -0.9238795325112867, // W_64^20
        -0.8819212643483549, // W_64^21
        -0.8314696123025452, // W_64^22
        -0.773010453362737,  // W_64^23
    ]);

    // Twiddles for k = 24..31
    let twiddle_re_24_31 = f64x8::new([
        -std::f64::consts::FRAC_1_SQRT_2, // W_64^24
        -0.773010453362737,               // W_64^25
        -0.8314696123025452,              // W_64^26
        -0.8819212643483549,              // W_64^27
        -0.9238795325112867,              // W_64^28
        -0.9569403357322089,              // W_64^29
        -0.9807852804032304,              // W_64^30
        -0.9951847266721969,              // W_64^31
    ]);

    let twiddle_im_24_31 = f64x8::new([
        -std::f64::consts::FRAC_1_SQRT_2, // W_64^24
        -0.6343932841636455,              // W_64^25
        -0.5555702330196022,              // W_64^26
        -0.47139673682599764,             // W_64^27
        -0.3826834323650898,              // W_64^28
        -0.29028467725446233,             // W_64^29
        -0.19509032201612825,             // W_64^30
        -0.0980171403295606,              // W_64^31
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process butterflies 0..7
            let in0_re = f64x8::new(reals_s0[0..8].try_into().unwrap());
            let in1_re = f64x8::new(reals_s1[0..8].try_into().unwrap());
            let in0_im = f64x8::new(imags_s0[0..8].try_into().unwrap());
            let in1_im = f64x8::new(imags_s1[0..8].try_into().unwrap());

            let out0_re =
                twiddle_im_0_7.mul_add(-in1_im, twiddle_re_0_7.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im_0_7.mul_add(in1_re, twiddle_re_0_7.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0[0..8].copy_from_slice(out0_re.as_array());
            imags_s0[0..8].copy_from_slice(out0_im.as_array());
            reals_s1[0..8].copy_from_slice(out1_re.as_array());
            imags_s1[0..8].copy_from_slice(out1_im.as_array());

            // Process butterflies 8..15
            let in0_re = f64x8::new(reals_s0[8..16].try_into().unwrap());
            let in1_re = f64x8::new(reals_s1[8..16].try_into().unwrap());
            let in0_im = f64x8::new(imags_s0[8..16].try_into().unwrap());
            let in1_im = f64x8::new(imags_s1[8..16].try_into().unwrap());

            let out0_re =
                twiddle_im_8_15.mul_add(-in1_im, twiddle_re_8_15.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im_8_15.mul_add(in1_re, twiddle_re_8_15.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0[8..16].copy_from_slice(out0_re.as_array());
            imags_s0[8..16].copy_from_slice(out0_im.as_array());
            reals_s1[8..16].copy_from_slice(out1_re.as_array());
            imags_s1[8..16].copy_from_slice(out1_im.as_array());

            // Process butterflies 16..23
            let in0_re = f64x8::new(reals_s0[16..24].try_into().unwrap());
            let in1_re = f64x8::new(reals_s1[16..24].try_into().unwrap());
            let in0_im = f64x8::new(imags_s0[16..24].try_into().unwrap());
            let in1_im = f64x8::new(imags_s1[16..24].try_into().unwrap());

            let out0_re =
                twiddle_im_16_23.mul_add(-in1_im, twiddle_re_16_23.mul_add(in1_re, in0_re));
            let out0_im =
                twiddle_im_16_23.mul_add(in1_re, twiddle_re_16_23.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0[16..24].copy_from_slice(out0_re.as_array());
            imags_s0[16..24].copy_from_slice(out0_im.as_array());
            reals_s1[16..24].copy_from_slice(out1_re.as_array());
            imags_s1[16..24].copy_from_slice(out1_im.as_array());

            // Process butterflies 24..31
            let in0_re = f64x8::new(reals_s0[24..32].try_into().unwrap());
            let in1_re = f64x8::new(reals_s1[24..32].try_into().unwrap());
            let in0_im = f64x8::new(imags_s0[24..32].try_into().unwrap());
            let in1_im = f64x8::new(imags_s1[24..32].try_into().unwrap());

            let out0_re =
                twiddle_im_24_31.mul_add(-in1_im, twiddle_re_24_31.mul_add(in1_re, in0_re));
            let out0_im =
                twiddle_im_24_31.mul_add(in1_re, twiddle_re_24_31.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0[24..32].copy_from_slice(out0_re.as_array());
            imags_s0[24..32].copy_from_slice(out0_im.as_array());
            reals_s1[24..32].copy_from_slice(out1_re.as_array());
            imags_s1[24..32].copy_from_slice(out1_im.as_array());
        });
}

/// DIT butterfly for chunk_size == 64 (f32)
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_chunk_64_simd_f32(reals: &mut [f32], imags: &mut [f32]) {
    const DIST: usize = 32;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = f32x16::splat(2.0);

    // Process in 2 iterations of 16 butterflies each
    // Twiddles for W_64^k where k = 0..15
    let twiddle_re_0_15 = f32x16::new([
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
    ]);

    let twiddle_im_0_15 = f32x16::new([
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
    ]);

    // Twiddles for k = 16..31
    let twiddle_re_16_31 = f32x16::new([
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
    ]);

    let twiddle_im_16_31 = f32x16::new([
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
    ]);

    (reals.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .zip(imags.as_chunks_mut::<CHUNK_SIZE>().0.iter_mut())
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Process butterflies 0..15
            let in0_re = f32x16::new(reals_s0[0..16].try_into().unwrap());
            let in1_re = f32x16::new(reals_s1[0..16].try_into().unwrap());
            let in0_im = f32x16::new(imags_s0[0..16].try_into().unwrap());
            let in1_im = f32x16::new(imags_s1[0..16].try_into().unwrap());

            let out0_re =
                twiddle_im_0_15.mul_add(-in1_im, twiddle_re_0_15.mul_add(in1_re, in0_re));
            let out0_im = twiddle_im_0_15.mul_add(in1_re, twiddle_re_0_15.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0[0..16].copy_from_slice(out0_re.as_array());
            imags_s0[0..16].copy_from_slice(out0_im.as_array());
            reals_s1[0..16].copy_from_slice(out1_re.as_array());
            imags_s1[0..16].copy_from_slice(out1_im.as_array());

            // Process butterflies 16..31
            let in0_re = f32x16::new(reals_s0[16..32].try_into().unwrap());
            let in1_re = f32x16::new(reals_s1[16..32].try_into().unwrap());
            let in0_im = f32x16::new(imags_s0[16..32].try_into().unwrap());
            let in1_im = f32x16::new(imags_s1[16..32].try_into().unwrap());

            let out0_re =
                twiddle_im_16_31.mul_add(-in1_im, twiddle_re_16_31.mul_add(in1_re, in0_re));
            let out0_im =
                twiddle_im_16_31.mul_add(in1_re, twiddle_re_16_31.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            reals_s0[16..32].copy_from_slice(out0_re.as_array());
            imags_s0[16..32].copy_from_slice(out0_im.as_array());
            reals_s1[16..32].copy_from_slice(out1_re.as_array());
            imags_s1[16..32].copy_from_slice(out1_im.as_array());
        });
}

/// General DIT butterfly for f64
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_64_chunk_n_simd(
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_re: &[f64],
    twiddles_im: &[f64],
    dist: usize,
) {
    const LANES: usize = 8;
    let chunk_size = dist << 1;
    assert!(chunk_size >= LANES * 2);

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            (reals_s0.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(reals_s1.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(imags_s0.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(imags_s1.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(twiddles_re.as_chunks::<LANES>().0.iter())
                .zip(twiddles_im.as_chunks::<LANES>().0.iter())
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), tw_re), tw_im)| {
                    let two = f64x8::splat(2.0);
                    let in0_re = f64x8::new(*re_s0);
                    let in1_re = f64x8::new(*re_s1);
                    let in0_im = f64x8::new(*im_s0);
                    let in1_im = f64x8::new(*im_s1);

                    let tw_re = f64x8::new(*tw_re);
                    let tw_im = f64x8::new(*tw_im);

                    // out0.re = (in0.re + tw_re * in1.re) - tw_im * in1.im
                    let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                    // out0.im = (in0.im + tw_re * in1.im) + tw_im * in1.re
                    let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));

                    // Use FMA for out1 = 2*in0 - out0
                    let out1_re = two.mul_sub(in0_re, out0_re);
                    let out1_im = two.mul_sub(in0_im, out0_im);

                    re_s0.copy_from_slice(out0_re.as_array());
                    im_s0.copy_from_slice(out0_im.as_array());
                    re_s1.copy_from_slice(out1_re.as_array());
                    im_s1.copy_from_slice(out1_im.as_array());
                });
        });
}

/// General DIT butterfly for f32
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_dit_32_chunk_n_simd(
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    dist: usize,
) {
    const LANES: usize = 16;
    let chunk_size = dist << 1;
    assert!(chunk_size >= LANES * 2);

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            (reals_s0.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(reals_s1.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(imags_s0.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(imags_s1.as_chunks_mut::<LANES>().0.iter_mut())
                .zip(twiddles_re.as_chunks::<LANES>().0.iter())
                .zip(twiddles_im.as_chunks::<LANES>().0.iter())
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), tw_re), tw_im)| {
                    let two = f32x16::splat(2.0);
                    let in0_re = f32x16::new(*re_s0);
                    let in1_re = f32x16::new(*re_s1);
                    let in0_im = f32x16::new(*im_s0);
                    let in1_im = f32x16::new(*im_s1);

                    let tw_re = f32x16::new(*tw_re);
                    let tw_im = f32x16::new(*tw_im);

                    // out0.re = (in0.re + tw_re * in1.re) - tw_im * in1.im
                    let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                    // out0.im = (in0.im + tw_re * in1.im) + tw_im * in1.re
                    let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));

                    // Use FMA for out1 = 2*in0 - out0
                    let out1_re = two.mul_sub(in0_re, out0_re);
                    let out1_im = two.mul_sub(in0_im, out0_im);

                    re_s0.copy_from_slice(out0_re.as_array());
                    im_s0.copy_from_slice(out0_im.as_array());
                    re_s1.copy_from_slice(out1_re.as_array());
                    im_s1.copy_from_slice(out1_im.as_array());
                });
        });
}
