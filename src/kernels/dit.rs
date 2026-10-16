//! DIT-specific FFT Kernels
//!
//! FFT kernels for the Decimation-in-Time algorithm.
//!
use fearless_simd::{f32x16, f32x4, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};
use num_traits::Float;

/// DIT butterflies for stage 0 (i.e., where `chunk_size == 2`)
/// Identical to DIF version (no twiddles at size 2)
#[inline(always)] // required by fearless_simd
pub fn fft_dit_chunk_2<S: Simd, T: Float>(_simd: S, reals: &mut [T], imags: &mut [T]) {
    reals
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(imags.as_chunks_mut::<2>().0.iter_mut())
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
            1.0_f32,                          // W_32^0 = 1
            0.980_785_25_f32,                 // W_32^1 = cos(π/16)
            0.923_879_5_f32,                  // W_32^2 = cos(π/8)
            0.831_469_6_f32,                  // W_32^3 = cos(3π/16)
            std::f32::consts::FRAC_1_SQRT_2,  // W_32^4 = sqrt(2)/2
            0.555_570_24_f32,                 // W_32^5 = cos(5π/16)
            0.382_683_43_f32,                 // W_32^6 = cos(3π/8)
            0.195_090_32_f32,                 // W_32^7 = cos(7π/16)
            0.0_f32,                          // W_32^8 = 0
            -0.195_090_32_f32,                // W_32^9
            -0.382_683_43_f32,                // W_32^10
            -0.555_570_24_f32,                // W_32^11
            -std::f32::consts::FRAC_1_SQRT_2, // W_32^12
            -0.831_469_6_f32,                 // W_32^13
            -0.923_879_5_f32,                 // W_32^14
            -0.980_785_25_f32,                // W_32^15
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
    let two = f64x8::splat(simd, 2.0);

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

/// Twiddle slices for two consecutive stages. The next stage only needs its
/// first `dist_s` entries; the second half is derived with a -j rotation.
#[derive(Clone, Copy)]
pub(crate) struct FusedTwiddles<'a, T> {
    pub stage_re: &'a [T],
    pub stage_im: &'a [T],
    pub next_stage_re: &'a [T],
    pub next_stage_im: &'a [T],
}

// Expand the same butterfly for both precisions and traversals. A macro keeps
// every operation inside the caller's SIMD target-feature boundary without
// introducing helper calls or relying on the optimizer's inlining heuristics.
macro_rules! fused_butterfly {
    ($vector:ident, $simd:expr,
     [$ra:expr, $rb:expr, $rc:expr, $rd:expr],
     [$ia:expr, $ib:expr, $ic:expr, $id:expr],
     [$tw_s_re:expr, $tw_s_im:expr, $tw1_re:expr, $tw1_im:expr]) => {{
        let two = $vector::splat($simd, 2.0);

        // Load 4 data sub-blocks
        let a_re = $vector::simd_from($simd, *$ra);
        let a_im = $vector::simd_from($simd, *$ia);
        let b_re = $vector::simd_from($simd, *$rb);
        let b_im = $vector::simd_from($simd, *$ib);
        let c_re = $vector::simd_from($simd, *$rc);
        let c_im = $vector::simd_from($simd, *$ic);
        let d_re = $vector::simd_from($simd, *$rd);
        let d_im = $vector::simd_from($simd, *$id);

        // Load stage-s twiddles
        let tws_re = $vector::simd_from($simd, *$tw_s_re);
        let tws_im = $vector::simd_from($simd, *$tw_s_im);

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
        let tw1r = $vector::simd_from($simd, *$tw1_re);
        let tw1i = $vector::simd_from($simd, *$tw1_im);

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
        out_a_re.store_slice($ra);
        out_a_im.store_slice($ia);
        out_b_re.store_slice($rb);
        out_b_im.store_slice($ib);
        out_c_re.store_slice($rc);
        out_c_im.store_slice($ic);
        out_d_re.store_slice($rd);
        out_d_im.store_slice($id);
    }};
}

/// Generate serial and parallel fused two-stage DIT kernels for one vector width.
///
/// Processes two consecutive DIT stages in a single pass over memory,
/// halving memory traffic for the fused stages. Uses the radix-2² identity
/// `W_{4D}^{k+D} = -j * W_{4D}^k` to derive the second-half twiddle factors.
/// Only the first `dist_s` entries of each twiddle slice are read.
///
/// The vector type and literal lane count are explicit so additional widths can
/// be generated without changing the traversal. `as_chunks::<LANES>()` needs a
/// const argument; an associated constant depending on generic `S` cannot be
/// used there on stable Rust. Width selection belongs outside these kernels.
macro_rules! impl_fft_dit_fused_2stage {
    ($float:ty, $vector:ident, $lanes:literal,
     $serial:ident, $serial_simd:ident,
     $parallel:ident, $parallel_simd:ident) => {
        /// Fused two-stage DIT butterfly (radix-2²).
        #[inline(never)]
        pub(crate) fn $serial<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: FusedTwiddles<'_, $float>,
            dist_s: usize,
        ) {
            simd.vectorize(
                #[inline(always)]
                || $serial_simd(simd, reals, imags, twiddles, dist_s),
            )
        }

        /// Fused two-stage DIT traversal with serial SIMD butterflies.
        #[inline(always)]
        fn $serial_simd<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: FusedTwiddles<'_, $float>,
            dist_s: usize,
        ) {
            const LANES: usize = $lanes;
            let block_size = dist_s * 4; // two stages: 4 sub-blocks of dist_s
            assert!(dist_s >= LANES);

            // Outer loop over blocks of 4*dist_s elements.
            // Using `for` (not `for_each`) for the same inlining reasons
            // as in the single-stage SIMD kernels.
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
                    .zip(twiddles.stage_re.as_chunks::<LANES>().0.iter())
                    .zip(twiddles.stage_im.as_chunks::<LANES>().0.iter())
                    .zip(twiddles.next_stage_re.as_chunks::<LANES>().0.iter())
                    .zip(twiddles.next_stage_im.as_chunks::<LANES>().0.iter())
                    .for_each(
                        |(
                            (
                                (((((((((ra, rb), rc), rd), ia), ib), ic), id), tw_s_re), tw_s_im),
                                tw1_re,
                            ),
                            tw1_im,
                        )| {
                            fused_butterfly!(
                                $vector,
                                simd,
                                [ra, rb, rc, rd],
                                [ia, ib, ic, id],
                                [tw_s_re, tw_s_im, tw1_re, tw1_im]
                            );
                        },
                    );
            }
        }

        /// Fused two-stage DIT butterfly (radix-2²), executed in parallel.
        #[cfg(feature = "parallel")]
        #[inline(never)]
        pub(crate) fn $parallel<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: FusedTwiddles<'_, $float>,
            dist_s: usize,
        ) {
            simd.vectorize(
                #[inline(always)]
                || $parallel_simd(simd, reals, imags, twiddles, dist_s),
            )
        }

        /// Fused two-stage DIT traversal with parallel SIMD butterflies.
        #[cfg(feature = "parallel")]
        #[inline(always)]
        fn $parallel_simd<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: FusedTwiddles<'_, $float>,
            dist_s: usize,
        ) {
            use rayon::prelude::*;
            const LANES: usize = $lanes;
            let block_size = dist_s * 4; // two stages: 4 sub-blocks of dist_s
            assert!(dist_s >= LANES);

            // Outer loop over blocks of 4*dist_s elements.
            // Using `for` (not `for_each`) for the same inlining reasons
            // as in the single-stage SIMD kernels.
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
                    .zip(twiddles.stage_re.as_chunks::<LANES>().0.par_iter())
                    .zip(twiddles.stage_im.as_chunks::<LANES>().0.par_iter())
                    .zip(twiddles.next_stage_re.as_chunks::<LANES>().0.par_iter())
                    .zip(twiddles.next_stage_im.as_chunks::<LANES>().0.par_iter())
                    .for_each(
                        |(
                            (
                                (((((((((ra, rb), rc), rd), ia), ib), ic), id), tw_s_re), tw_s_im),
                                tw1_re,
                            ),
                            tw1_im,
                        )| {
                            simd.vectorize(
                                #[inline(always)]
                                || {
                                    fused_butterfly!(
                                        $vector,
                                        simd,
                                        [ra, rb, rc, rd],
                                        [ia, ib, ic, id],
                                        [tw_s_re, tw_s_im, tw1_re, tw1_im]
                                    );
                                },
                            );
                        },
                    );
            }
        }
    };
}

impl_fft_dit_fused_2stage!(
    f64,
    f64x4,
    4,
    fft_dit_fused_2stage_f64_narrow,
    fft_dit_fused_2stage_simd_f64_narrow,
    fft_dit_fused_2stage_f64_narrow_parallel,
    fft_dit_fused_2stage_simd_f64_narrow_parallel
);

impl_fft_dit_fused_2stage!(
    f32,
    f32x8,
    8,
    fft_dit_fused_2stage_f32_narrow,
    fft_dit_fused_2stage_simd_f32_narrow,
    fft_dit_fused_2stage_f32_narrow_parallel,
    fft_dit_fused_2stage_simd_f32_narrow_parallel
);

/// Base twiddles for four consecutive DIT stages, from the final stage's table.
/// Only the first `dist_s` entries of each slice are read. For `D = dist_s`,
/// `w = W_{16D}^k` supplies the stage bases `w^8`, `w^4`, `w^2`, and `w`.
/// Powers and fixed rotations are computed locally in the butterfly.
#[derive(Clone, Copy)]
pub(crate) struct Fused4Twiddles<'a, T> {
    pub re: &'a [T],
    pub im: &'a [T],
}

// Expand the complete radix-2⁴ network inside the SIMD target-feature boundary.
// Each lane is an independent 16-point butterfly across sub-blocks spaced D
// apart. Intermediates stay in local vectors rather than writing to the input.
macro_rules! fused_4stage_butterfly {
    ($float:ty, $vector:ident, $simd:expr, [$base_re:expr, $base_im:expr],
     [($r0:expr, $i0:expr), ($r1:expr, $i1:expr), ($r2:expr, $i2:expr), ($r3:expr, $i3:expr),
      ($r4:expr, $i4:expr), ($r5:expr, $i5:expr), ($r6:expr, $i6:expr), ($r7:expr, $i7:expr),
      ($r8:expr, $i8:expr), ($r9:expr, $i9:expr), ($r10:expr, $i10:expr), ($r11:expr, $i11:expr),
      ($r12:expr, $i12:expr), ($r13:expr, $i13:expr), ($r14:expr, $i14:expr), ($r15:expr, $i15:expr)]) => {{
        let two = $vector::splat($simd, 2.0);
        // One complex vector load for all four stages. Squaring is local to
        // this SIMD position, so rounding error cannot accumulate across k.
        let w_re = $vector::simd_from($simd, *$base_re);
        let w_im = $vector::simd_from($simd, *$base_im);
        let w2_re = w_re.mul_sub(w_re, w_im * w_im);
        let w2_im = (w_re + w_re) * w_im;
        let w4_re = w2_re.mul_sub(w2_re, w2_im * w2_im);
        let w4_im = (w2_re + w2_re) * w2_im;
        let w8_re = w4_re.mul_sub(w4_re, w4_im * w4_im);
        let w8_im = (w4_re + w4_re) * w4_im;
        let mut r0 = $vector::simd_from($simd, *$r0);
        let mut i0 = $vector::simd_from($simd, *$i0);
        let mut r1 = $vector::simd_from($simd, *$r1);
        let mut i1 = $vector::simd_from($simd, *$i1);
        let mut r2 = $vector::simd_from($simd, *$r2);
        let mut i2 = $vector::simd_from($simd, *$i2);
        let mut r3 = $vector::simd_from($simd, *$r3);
        let mut i3 = $vector::simd_from($simd, *$i3);
        let mut r4 = $vector::simd_from($simd, *$r4);
        let mut i4 = $vector::simd_from($simd, *$i4);
        let mut r5 = $vector::simd_from($simd, *$r5);
        let mut i5 = $vector::simd_from($simd, *$i5);
        let mut r6 = $vector::simd_from($simd, *$r6);
        let mut i6 = $vector::simd_from($simd, *$i6);
        let mut r7 = $vector::simd_from($simd, *$r7);
        let mut i7 = $vector::simd_from($simd, *$i7);
        let mut r8 = $vector::simd_from($simd, *$r8);
        let mut i8 = $vector::simd_from($simd, *$i8);
        let mut r9 = $vector::simd_from($simd, *$r9);
        let mut i9 = $vector::simd_from($simd, *$i9);
        let mut r10 = $vector::simd_from($simd, *$r10);
        let mut i10 = $vector::simd_from($simd, *$i10);
        let mut r11 = $vector::simd_from($simd, *$r11);
        let mut i11 = $vector::simd_from($simd, *$i11);
        let mut r12 = $vector::simd_from($simd, *$r12);
        let mut i12 = $vector::simd_from($simd, *$i12);
        let mut r13 = $vector::simd_from($simd, *$r13);
        let mut i13 = $vector::simd_from($simd, *$i13);
        let mut r14 = $vector::simd_from($simd, *$r14);
        let mut i14 = $vector::simd_from($simd, *$i14);
        let mut r15 = $vector::simd_from($simd, *$r15);
        let mut i15 = $vector::simd_from($simd, *$i15);

        macro_rules! butterfly {
            ($ar:ident, $ai:ident, $br:ident, $bi:ident, $wr:expr, $wi:expr) => {{
                let out_re = $wi.mul_add(-$bi, $wr.mul_add($br, $ar));
                let out_im = $wi.mul_add($br, $wr.mul_add($bi, $ai));
                $br = two.mul_sub($ar, out_re);
                $bi = two.mul_sub($ai, out_im);
                $ar = out_re;
                $ai = out_im;
            }};
        }

        // Stage s + 0: pair sub-blocks 1 apart.
        let wr = w8_re;
        let wi = w8_im;
        butterfly!(r0, i0, r1, i1, wr, wi);
        butterfly!(r2, i2, r3, i3, wr, wi);
        butterfly!(r4, i4, r5, i5, wr, wi);
        butterfly!(r6, i6, r7, i7, wr, wi);
        butterfly!(r8, i8, r9, i9, wr, wi);
        butterfly!(r10, i10, r11, i11, wr, wi);
        butterfly!(r12, i12, r13, i13, wr, wi);
        butterfly!(r14, i14, r15, i15, wr, wi);

        // Stage s + 1: pair sub-blocks 2 apart.
        let wr = w4_re;
        let wi = w4_im;
        butterfly!(r0, i0, r2, i2, wr, wi);
        butterfly!(r4, i4, r6, i6, wr, wi);
        butterfly!(r8, i8, r10, i10, wr, wi);
        butterfly!(r12, i12, r14, i14, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r1, i1, r3, i3, wi, (-wr));
        butterfly!(r5, i5, r7, i7, wi, (-wr));
        butterfly!(r9, i9, r11, i11, wi, (-wr));
        butterfly!(r13, i13, r15, i15, wi, (-wr));

        // Stage s + 2: pair sub-blocks 4 apart.
        let wr = w2_re;
        let wi = w2_im;
        butterfly!(r0, i0, r4, i4, wr, wi);
        butterfly!(r8, i8, r12, i12, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r2, i2, r6, i6, wi, (-wr));
        butterfly!(r10, i10, r14, i14, wi, (-wr));
        // W_{8D}^{k+D} = w^2 * exp(-i*pi/4).
        let sqrt_half = $vector::splat($simd, std::f64::consts::FRAC_1_SQRT_2 as $float);
        let wr = (w2_re + w2_im) * sqrt_half;
        let wi = (w2_im - w2_re) * sqrt_half;
        butterfly!(r1, i1, r5, i5, wr, wi);
        butterfly!(r9, i9, r13, i13, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r3, i3, r7, i7, wi, (-wr));
        butterfly!(r11, i11, r15, i15, wi, (-wr));

        // Stage s + 3: pair sub-blocks 8 apart.
        let wr = w_re;
        let wi = w_im;
        butterfly!(r0, i0, r8, i8, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r4, i4, r12, i12, wi, (-wr));
        // W_{16D}^{k+D} = w * exp(-i*pi/8).
        let cos_pi_8 = $vector::splat($simd, 0.923_879_532_511_286_7_f64 as $float);
        let sin_pi_8 = $vector::splat($simd, 0.382_683_432_365_089_8_f64 as $float);
        let wr = w_re.mul_add(cos_pi_8, w_im * sin_pi_8);
        let wi = w_im.mul_sub(cos_pi_8, w_re * sin_pi_8);
        butterfly!(r1, i1, r9, i9, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r5, i5, r13, i13, wi, (-wr));
        // W_{16D}^{k+2D} = w * exp(-i*pi/4).
        let wr = (w_re + w_im) * sqrt_half;
        let wi = (w_im - w_re) * sqrt_half;
        butterfly!(r2, i2, r10, i10, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r6, i6, r14, i14, wi, (-wr));
        // W_{16D}^{k+3D} = w * exp(-i*3*pi/8).
        let wr = w_re.mul_add(sin_pi_8, w_im * cos_pi_8);
        let wi = w_im.mul_sub(sin_pi_8, w_re * cos_pi_8);
        butterfly!(r3, i3, r11, i11, wr, wi);
        // W[k + half_stage_distance] = -j * W[k].
        butterfly!(r7, i7, r15, i15, wi, (-wr));
        r0.store_slice($r0);
        i0.store_slice($i0);
        r1.store_slice($r1);
        i1.store_slice($i1);
        r2.store_slice($r2);
        i2.store_slice($i2);
        r3.store_slice($r3);
        i3.store_slice($i3);
        r4.store_slice($r4);
        i4.store_slice($i4);
        r5.store_slice($r5);
        i5.store_slice($i5);
        r6.store_slice($r6);
        i6.store_slice($i6);
        r7.store_slice($r7);
        i7.store_slice($i7);
        r8.store_slice($r8);
        i8.store_slice($i8);
        r9.store_slice($r9);
        i9.store_slice($i9);
        r10.store_slice($r10);
        i10.store_slice($i10);
        r11.store_slice($r11);
        i11.store_slice($i11);
        r12.store_slice($r12);
        i12.store_slice($i12);
        r13.store_slice($r13);
        i13.store_slice($i13);
        r14.store_slice($r14);
        i14.store_slice($i14);
        r15.store_slice($r15);
        i15.store_slice($i15);
    }};
}

// Serial and Rayon traversals use the same disjoint slices and arithmetic.
// Only the iterator method differs. Worker closures enter their SIMD boundary
// explicitly, since Rayon workers do not inherit the caller's target features.
macro_rules! fused_4stage_traversal {
    ($float:ty, $vector:ident, $lanes:literal,
     $simd:ident, $reals:ident, $imags:ident, $tw:ident, $dist:ident, $iter:ident, $tw_iter:ident) => {{
        const LANES: usize = $lanes;
        assert_eq!($reals.len(), $imags.len());
        assert!($dist >= LANES && $dist % LANES == 0);
        let block_size = $dist * 16;
        assert_eq!($reals.len() % block_size, 0);
        let tw_re = $tw.re[..$dist].as_chunks::<LANES>().0;
        let tw_im = $tw.im[..$dist].as_chunks::<LANES>().0;

        for (re, im) in $reals.chunks_exact_mut(block_size).zip($imags.chunks_exact_mut(block_size)) {
            let mut re = re.chunks_exact_mut($dist);
            let mut im = im.chunks_exact_mut($dist);
            let [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15]: [&mut [$float]; 16] =
                std::array::from_fn(|_| re.next().unwrap());
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]: [&mut [$float]; 16] =
                std::array::from_fn(|_| im.next().unwrap());
            let p0 = r0.as_chunks_mut::<LANES>().0.$iter().zip(i0.as_chunks_mut::<LANES>().0.$iter());
            let p1 = r1.as_chunks_mut::<LANES>().0.$iter().zip(i1.as_chunks_mut::<LANES>().0.$iter());
            let p2 = r2.as_chunks_mut::<LANES>().0.$iter().zip(i2.as_chunks_mut::<LANES>().0.$iter());
            let p3 = r3.as_chunks_mut::<LANES>().0.$iter().zip(i3.as_chunks_mut::<LANES>().0.$iter());
            let p4 = r4.as_chunks_mut::<LANES>().0.$iter().zip(i4.as_chunks_mut::<LANES>().0.$iter());
            let p5 = r5.as_chunks_mut::<LANES>().0.$iter().zip(i5.as_chunks_mut::<LANES>().0.$iter());
            let p6 = r6.as_chunks_mut::<LANES>().0.$iter().zip(i6.as_chunks_mut::<LANES>().0.$iter());
            let p7 = r7.as_chunks_mut::<LANES>().0.$iter().zip(i7.as_chunks_mut::<LANES>().0.$iter());
            let p8 = r8.as_chunks_mut::<LANES>().0.$iter().zip(i8.as_chunks_mut::<LANES>().0.$iter());
            let p9 = r9.as_chunks_mut::<LANES>().0.$iter().zip(i9.as_chunks_mut::<LANES>().0.$iter());
            let p10 = r10.as_chunks_mut::<LANES>().0.$iter().zip(i10.as_chunks_mut::<LANES>().0.$iter());
            let p11 = r11.as_chunks_mut::<LANES>().0.$iter().zip(i11.as_chunks_mut::<LANES>().0.$iter());
            let p12 = r12.as_chunks_mut::<LANES>().0.$iter().zip(i12.as_chunks_mut::<LANES>().0.$iter());
            let p13 = r13.as_chunks_mut::<LANES>().0.$iter().zip(i13.as_chunks_mut::<LANES>().0.$iter());
            let p14 = r14.as_chunks_mut::<LANES>().0.$iter().zip(i14.as_chunks_mut::<LANES>().0.$iter());
            let p15 = r15.as_chunks_mut::<LANES>().0.$iter().zip(i15.as_chunks_mut::<LANES>().0.$iter());
            let q0 = p0.zip(p1).zip(p2.zip(p3));
            let q1 = p4.zip(p5).zip(p6.zip(p7));
            let q2 = p8.zip(p9).zip(p10.zip(p11));
            let q3 = p12.zip(p13).zip(p14.zip(p15));
            let tw = tw_re.$tw_iter().zip(tw_im.$tw_iter());
            q0.zip(q1).zip(q2.zip(q3)).zip(tw)
                .for_each(|(((q0, q1), (q2, q3)), (base_re, base_im))| {
                    let (((r0, i0), (r1, i1)), ((r2, i2), (r3, i3))) = q0;
                    let (((r4, i4), (r5, i5)), ((r6, i6), (r7, i7))) = q1;
                    let (((r8, i8), (r9, i9)), ((r10, i10), (r11, i11))) = q2;
                    let (((r12, i12), (r13, i13)), ((r14, i14), (r15, i15))) = q3;
                    $simd.vectorize(
                        #[inline(always)]
                        || {
                            fused_4stage_butterfly!($float, $vector, $simd, [base_re, base_im],
                                [(r0, i0), (r1, i1), (r2, i2), (r3, i3),
                                 (r4, i4), (r5, i5), (r6, i6), (r7, i7),
                                 (r8, i8), (r9, i9), (r10, i10), (r11, i11),
                                 (r12, i12), (r13, i13), (r14, i14), (r15, i15)]);
                        },
                    );
                });
        }
    }};
}

/// Generate four-stage kernels for one precision and vector width.
macro_rules! impl_fft_dit_fused_4stage {
    ($float:ty, $vector:ident, $lanes:literal,
     $serial:ident, $serial_simd:ident, $parallel:ident, $parallel_simd:ident) => {
        /// Four consecutive DIT stages (radix-2⁴) in one pass over the input.
        #[inline(never)]
        pub(crate) fn $serial<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: Fused4Twiddles<'_, $float>,
            dist_s: usize,
        ) {
            simd.vectorize(
                #[inline(always)]
                || $serial_simd(simd, reals, imags, twiddles, dist_s),
            )
        }

        #[inline(always)]
        fn $serial_simd<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: Fused4Twiddles<'_, $float>,
            dist_s: usize,
        ) {
            fused_4stage_traversal!(
                $float, $vector, $lanes, simd, reals, imags, twiddles, dist_s, iter_mut, iter
            );
        }

        /// Four consecutive DIT stages, distributed across Rayon workers.
        #[cfg(feature = "parallel")]
        #[inline(never)]
        pub(crate) fn $parallel<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: Fused4Twiddles<'_, $float>,
            dist_s: usize,
        ) {
            simd.vectorize(
                #[inline(always)]
                || $parallel_simd(simd, reals, imags, twiddles, dist_s),
            )
        }

        #[cfg(feature = "parallel")]
        #[inline(always)]
        fn $parallel_simd<S: Simd>(
            simd: S,
            reals: &mut [$float],
            imags: &mut [$float],
            twiddles: Fused4Twiddles<'_, $float>,
            dist_s: usize,
        ) {
            use rayon::prelude::*;
            fused_4stage_traversal!(
                $float,
                $vector,
                $lanes,
                simd,
                reals,
                imags,
                twiddles,
                dist_s,
                par_iter_mut,
                par_iter
            );
        }
    };
}

impl_fft_dit_fused_4stage!(
    f64,
    f64x4,
    4,
    fft_dit_fused_4stage_f64_narrow,
    fft_dit_fused_4stage_simd_f64_narrow,
    fft_dit_fused_4stage_f64_narrow_parallel,
    fft_dit_fused_4stage_simd_f64_narrow_parallel
);

impl_fft_dit_fused_4stage!(
    f32,
    f32x8,
    8,
    fft_dit_fused_4stage_f32_narrow,
    fft_dit_fused_4stage_simd_f32_narrow,
    fft_dit_fused_4stage_f32_narrow_parallel,
    fft_dit_fused_4stage_simd_f32_narrow_parallel
);

#[cfg(test)]
mod tests {
    use fearless_simd::{dispatch, Level};

    use super::*;

    fn supported_levels() -> Vec<Level> {
        let detected = Level::new();
        #[allow(unused_mut)]
        let mut levels = vec![Level::baseline(), detected];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            levels.extend(detected.as_sse2().map(Level::Sse2));
            levels.extend(detected.as_sse4_2().map(Level::Sse4_2));
            levels.extend(detected.as_avx2().map(Level::Avx2));
            levels.extend(detected.as_avx512().map(Level::Avx512));
        }
        levels
    }

    macro_rules! test_four_stages {
        ($name:ident, $float:ident, $lanes:literal,
         $single:ident, $small:ident, $fused:ident, $parallel:ident, $tolerance:expr) => {
            #[test]
            fn $name() {
                for level in supported_levels() {
                    for dist in [$lanes, 2 * $lanes, 64, 256, 1024] {
                        let tables: [(Vec<$float>, Vec<$float>); 4] =
                            std::array::from_fn(|stage| {
                                let distance = dist << stage;
                                (0..distance)
                                    .map(|k| {
                                        let angle = -std::$float::consts::PI * k as $float
                                            / distance as $float;
                                        (angle.cos(), angle.sin())
                                    })
                                    .unzip()
                            });
                        // Supply only the final stage's first D entries. The
                        // kernel must derive every other twiddle locally, while
                        // the reference uses independent tables for all stages.
                        let twiddles = Fused4Twiddles {
                            re: &tables[3].0[..dist],
                            im: &tables[3].1[..dist],
                        };
                        for blocks in [1, 3] {
                            let n = 16 * dist * blocks;
                            let input_re: Vec<$float> = (0..n)
                                .map(|i| ((i * 17 % 257) as $float - 128.0) / 128.0)
                                .collect();
                            let input_im: Vec<$float> = (0..n)
                                .map(|i| ((i * 31 % 251) as $float - 125.0) / 125.0)
                                .collect();
                            let mut expected_re = input_re.clone();
                            let mut expected_im = input_im.clone();
                            dispatch!(level, simd => {
                                for (stage, (tw_re, tw_im)) in tables.iter().enumerate() {
                                    let distance = dist << stage;
                                    // The existing general kernels use twice the
                                    // lane width of the fused kernels.
                                    if distance == $lanes {
                                        $small(simd, &mut expected_re, &mut expected_im);
                                    } else {
                                        $single(simd, &mut expected_re, &mut expected_im,
                                            tw_re, tw_im, distance);
                                    }
                                }
                            });
                            let check = |actual_re: &[$float], actual_im: &[$float]| {
                                let mut error = 0.0;
                                let mut norm = 0.0;
                                for ((re, im), (er, ei)) in actual_re.iter().zip(actual_im)
                                    .zip(expected_re.iter().zip(&expected_im))
                                {
                                    assert!(re.is_finite() && im.is_finite());
                                    error += (*re as f64 - *er as f64).powi(2)
                                        + (*im as f64 - *ei as f64).powi(2);
                                    norm += (*er as f64).powi(2) + (*ei as f64).powi(2);
                                }
                                assert!((error / norm).sqrt() < $tolerance,
                                    "{level:?}, dist={dist}, blocks={blocks}: error={error}, norm={norm}");
                            };
                            let mut re = input_re.clone();
                            let mut im = input_im.clone();
                            dispatch!(level, simd => $fused(simd, &mut re, &mut im, twiddles, dist));
                            check(&re, &im);
                            #[cfg(feature = "parallel")]
                            {
                                let mut re = input_re.clone();
                                let mut im = input_im.clone();
                                dispatch!(level, simd => $parallel(simd, &mut re, &mut im, twiddles, dist));
                                check(&re, &im);
                            }
                        }
                    }
                }
            }
        };
    }

    test_four_stages!(
        fused_four_stages_f64,
        f64,
        4,
        fft_dit_chunk_n_f64,
        fft_dit_chunk_8_f64,
        fft_dit_fused_4stage_f64_narrow,
        fft_dit_fused_4stage_f64_narrow_parallel,
        1e-12
    );
    test_four_stages!(
        fused_four_stages_f32,
        f32,
        8,
        fft_dit_chunk_n_f32,
        fft_dit_chunk_16_f32,
        fft_dit_fused_4stage_f32_narrow,
        fft_dit_fused_4stage_f32_narrow_parallel,
        2e-6
    );
}
