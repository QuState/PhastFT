//! Radix-2^2 DIT FFT Kernels
//!
//! Each kernel fuses two consecutive radix-2 butterfly stages into one pass.
//! The radix-2^2 decomposition (He & Torkelson 1996) means only every other
//! stage pair needs non-trivial twiddle multiplications, and the inter-butterfly
//! twiddle within a pair is just multiplication by -j (a free swap+negate).

use fearless_simd::{f32x16, f32x4, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};

// ---------------------------------------------------------------------------
// Pair 0: fuses stages 0+1 (chunk = 4, D = 1)
// No inter-pair twiddle needed (first pair).
// 4-point DIT butterfly with only trivial operations.
// ---------------------------------------------------------------------------

/// Radix-2^2 fused pair 0 for f64: stages 0+1, chunk_size = 4
#[inline(never)]
pub fn fft_dit_r22_pair0_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_r22_pair0_simd_f64(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_r22_pair0_simd_f64<S: Simd>(_simd: S, reals: &mut [f64], imags: &mut [f64]) {
    // D=1, chunk=4. Process 4 elements at a time.
    reals
        .chunks_exact_mut(4)
        .zip(imags.chunks_exact_mut(4))
        .for_each(|(re, im)| {
            let a_re = re[0];
            let a_im = im[0];
            let b_re = re[1];
            let b_im = im[1];
            let c_re = re[2];
            let c_im = im[2];
            let d_re = re[3];
            let d_im = im[3];

            // Step 1: First radix-2 BF (dist=1): pairs (a,b) and (c,d)
            let t0_re = a_re + b_re;
            let t0_im = a_im + b_im;
            let t1_re = a_re - b_re;
            let t1_im = a_im - b_im;
            let t2_re = c_re + d_re;
            let t2_im = c_im + d_im;
            let t3_re = c_re - d_re;
            let t3_im = c_im - d_im;

            // Step 2: Trivial -j on position 3 (4th quarter of chunk)
            // -j * (t3_re + j*t3_im) = t3_im - j*t3_re
            let t3j_re = t3_im;
            let t3j_im = -t3_re;

            // Step 3: Second radix-2 BF (dist=2): pairs (t0,t2) and (t1,t3')
            re[0] = t0_re + t2_re;
            im[0] = t0_im + t2_im;
            re[1] = t1_re + t3j_re;
            im[1] = t1_im + t3j_im;
            re[2] = t0_re - t2_re;
            im[2] = t0_im - t2_im;
            re[3] = t1_re - t3j_re;
            im[3] = t1_im - t3j_im;
        });
}

/// Radix-2^2 fused pair 0 for f32: stages 0+1, chunk_size = 4
#[inline(never)]
pub fn fft_dit_r22_pair0_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_r22_pair0_simd_f32(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_r22_pair0_simd_f32<S: Simd>(_simd: S, reals: &mut [f32], imags: &mut [f32]) {
    reals
        .chunks_exact_mut(4)
        .zip(imags.chunks_exact_mut(4))
        .for_each(|(re, im)| {
            let a_re = re[0];
            let a_im = im[0];
            let b_re = re[1];
            let b_im = im[1];
            let c_re = re[2];
            let c_im = im[2];
            let d_re = re[3];
            let d_im = im[3];

            let t0_re = a_re + b_re;
            let t0_im = a_im + b_im;
            let t1_re = a_re - b_re;
            let t1_im = a_im - b_im;
            let t2_re = c_re + d_re;
            let t2_im = c_im + d_im;
            let t3_re = c_re - d_re;
            let t3_im = c_im - d_im;

            let t3j_re = t3_im;
            let t3j_im = -t3_re;

            re[0] = t0_re + t2_re;
            im[0] = t0_im + t2_im;
            re[1] = t1_re + t3j_re;
            im[1] = t1_im + t3j_im;
            re[2] = t0_re - t2_re;
            im[2] = t0_im - t2_im;
            re[3] = t1_re - t3j_re;
            im[3] = t1_im - t3j_im;
        });
}

// ---------------------------------------------------------------------------
// Pair 1: fuses stages 2+3 (chunk = 16, D = 4)
// Inter-pair twiddle: W_16^{q*k} for q=1,2,3, k=0..3 (hardcoded)
// ---------------------------------------------------------------------------

/// Radix-2^2 fused pair 1 for f64: stages 2+3, chunk_size = 16
#[inline(never)]
pub fn fft_dit_r22_pair1_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_r22_pair1_simd_f64(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_r22_pair1_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const D: usize = 4;
    const CHUNK: usize = 16;

    // Inter-pair twiddle factors W_16^{q*k} for k=0..3
    // Quarter 1: W_16^{2k} for k=0..3
    let q1_tw_re = f64x4::simd_from(
        simd,
        [
            1.0,                              // W_16^0
            std::f64::consts::FRAC_1_SQRT_2,  // W_16^2
            0.0,                              // W_16^4
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
        ],
    );
    let q1_tw_im = f64x4::simd_from(
        simd,
        [
            0.0,                              // W_16^0
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^2
            -1.0,                             // W_16^4
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
        ],
    );

    // Quarter 2: W_16^k for k=0..3
    let q2_tw_re = f64x4::simd_from(
        simd,
        [
            1.0,                              // W_16^0
            0.9238795325112867,               // W_16^1
            std::f64::consts::FRAC_1_SQRT_2,  // W_16^2
            0.38268343236508984,              // W_16^3
        ],
    );
    let q2_tw_im = f64x4::simd_from(
        simd,
        [
            0.0,                              // W_16^0
            -0.38268343236508984,             // W_16^1
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^2
            -0.9238795325112867,              // W_16^3
        ],
    );

    // Quarter 3: W_16^{3k} for k=0..3
    let q3_tw_re = f64x4::simd_from(
        simd,
        [
            1.0,                              // W_16^0
            0.38268343236508984,              // W_16^3
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
            -0.9238795325112867,              // W_16^9
        ],
    );
    let q3_tw_im = f64x4::simd_from(
        simd,
        [
            0.0,                              // W_16^0
            -0.9238795325112867,              // W_16^3
            -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
            0.38268343236508984,              // W_16^9
        ],
    );

    let two = f64x4::splat(simd, 2.0);

    reals
        .chunks_exact_mut(CHUNK)
        .zip(imags.chunks_exact_mut(CHUNK))
        .for_each(|(re_chunk, im_chunk)| {
            // Step 1: Inter-pair twiddle on quarters 1, 2, 3
            // Quarter 0 (indices 0..4): identity, skip
            // Quarter 1 (indices 4..8)
            let r1 = f64x4::from_slice(simd, &re_chunk[D..2 * D]);
            let i1 = f64x4::from_slice(simd, &im_chunk[D..2 * D]);
            let tw_r1 = q1_tw_im.mul_add(-i1, q1_tw_re * r1);
            let tw_i1 = q1_tw_im.mul_add(r1, q1_tw_re * i1);
            tw_r1.store_slice(&mut re_chunk[D..2 * D]);
            tw_i1.store_slice(&mut im_chunk[D..2 * D]);

            // Quarter 2 (indices 8..12)
            let r2 = f64x4::from_slice(simd, &re_chunk[2 * D..3 * D]);
            let i2 = f64x4::from_slice(simd, &im_chunk[2 * D..3 * D]);
            let tw_r2 = q2_tw_im.mul_add(-i2, q2_tw_re * r2);
            let tw_i2 = q2_tw_im.mul_add(r2, q2_tw_re * i2);
            tw_r2.store_slice(&mut re_chunk[2 * D..3 * D]);
            tw_i2.store_slice(&mut im_chunk[2 * D..3 * D]);

            // Quarter 3 (indices 12..16)
            let r3 = f64x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
            let i3 = f64x4::from_slice(simd, &im_chunk[3 * D..4 * D]);
            let tw_r3 = q3_tw_im.mul_add(-i3, q3_tw_re * r3);
            let tw_i3 = q3_tw_im.mul_add(r3, q3_tw_re * i3);
            tw_r3.store_slice(&mut re_chunk[3 * D..4 * D]);
            tw_i3.store_slice(&mut im_chunk[3 * D..4 * D]);

            // Step 2: First radix-2 BF (dist=4) in two sub-chunks of 8
            // Sub-chunk 0: indices [0..4] and [4..8]
            {
                let in0_re = f64x4::from_slice(simd, &re_chunk[0..D]);
                let in1_re = f64x4::from_slice(simd, &re_chunk[D..2 * D]);
                let in0_im = f64x4::from_slice(simd, &im_chunk[0..D]);
                let in1_im = f64x4::from_slice(simd, &im_chunk[D..2 * D]);

                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_chunk[0..D]);
                out0_im.store_slice(&mut im_chunk[0..D]);
                out1_re.store_slice(&mut re_chunk[D..2 * D]);
                out1_im.store_slice(&mut im_chunk[D..2 * D]);
            }
            // Sub-chunk 1: indices [8..12] and [12..16]
            {
                let in0_re = f64x4::from_slice(simd, &re_chunk[2 * D..3 * D]);
                let in1_re = f64x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let in0_im = f64x4::from_slice(simd, &im_chunk[2 * D..3 * D]);
                let in1_im = f64x4::from_slice(simd, &im_chunk[3 * D..4 * D]);

                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_chunk[2 * D..3 * D]);
                out0_im.store_slice(&mut im_chunk[2 * D..3 * D]);
                out1_re.store_slice(&mut re_chunk[3 * D..4 * D]);
                out1_im.store_slice(&mut im_chunk[3 * D..4 * D]);
            }

            // Step 3: Trivial -j on 4th quarter (indices 12..16)
            {
                let r = f64x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let i = f64x4::from_slice(simd, &im_chunk[3 * D..4 * D]);
                // -j * (r + j*i) = i - j*r
                i.store_slice(&mut re_chunk[3 * D..4 * D]);
                (-r).store_slice(&mut im_chunk[3 * D..4 * D]);
            }

            // Step 4: Second radix-2 BF (dist=8)
            // Pairs: [0..4] with [8..12], and [4..8] with [12..16]
            {
                let in0_re = f64x4::from_slice(simd, &re_chunk[0..D]);
                let in1_re = f64x4::from_slice(simd, &re_chunk[2 * D..3 * D]);
                let in0_im = f64x4::from_slice(simd, &im_chunk[0..D]);
                let in1_im = f64x4::from_slice(simd, &im_chunk[2 * D..3 * D]);

                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_chunk[0..D]);
                out0_im.store_slice(&mut im_chunk[0..D]);
                out1_re.store_slice(&mut re_chunk[2 * D..3 * D]);
                out1_im.store_slice(&mut im_chunk[2 * D..3 * D]);
            }
            {
                let in0_re = f64x4::from_slice(simd, &re_chunk[D..2 * D]);
                let in1_re = f64x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let in0_im = f64x4::from_slice(simd, &im_chunk[D..2 * D]);
                let in1_im = f64x4::from_slice(simd, &im_chunk[3 * D..4 * D]);

                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_chunk[D..2 * D]);
                out0_im.store_slice(&mut im_chunk[D..2 * D]);
                out1_re.store_slice(&mut re_chunk[3 * D..4 * D]);
                out1_im.store_slice(&mut im_chunk[3 * D..4 * D]);
            }
        });
}

/// Radix-2^2 fused pair 1 for f32: stages 2+3, chunk_size = 16
#[inline(never)]
pub fn fft_dit_r22_pair1_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_r22_pair1_simd_f32(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_r22_pair1_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const D: usize = 4;
    const CHUNK: usize = 16;

    let q1_tw_re = f32x4::simd_from(
        simd,
        [1.0_f32, std::f32::consts::FRAC_1_SQRT_2, 0.0_f32, -std::f32::consts::FRAC_1_SQRT_2],
    );
    let q1_tw_im = f32x4::simd_from(
        simd,
        [0.0_f32, -std::f32::consts::FRAC_1_SQRT_2, -1.0_f32, -std::f32::consts::FRAC_1_SQRT_2],
    );
    let q2_tw_re = f32x4::simd_from(
        simd,
        [1.0_f32, 0.923_879_5_f32, std::f32::consts::FRAC_1_SQRT_2, 0.382_683_43_f32],
    );
    let q2_tw_im = f32x4::simd_from(
        simd,
        [0.0_f32, -0.382_683_43_f32, -std::f32::consts::FRAC_1_SQRT_2, -0.923_879_5_f32],
    );
    let q3_tw_re = f32x4::simd_from(
        simd,
        [1.0_f32, 0.382_683_43_f32, -std::f32::consts::FRAC_1_SQRT_2, -0.923_879_5_f32],
    );
    let q3_tw_im = f32x4::simd_from(
        simd,
        [0.0_f32, -0.923_879_5_f32, -std::f32::consts::FRAC_1_SQRT_2, 0.382_683_43_f32],
    );

    let two = f32x4::splat(simd, 2.0);

    reals
        .chunks_exact_mut(CHUNK)
        .zip(imags.chunks_exact_mut(CHUNK))
        .for_each(|(re_chunk, im_chunk)| {
            // Step 1: Inter-pair twiddle on quarters 1, 2, 3
            {
                let r = f32x4::from_slice(simd, &re_chunk[D..2 * D]);
                let i = f32x4::from_slice(simd, &im_chunk[D..2 * D]);
                let tw_r = q1_tw_im.mul_add(-i, q1_tw_re * r);
                let tw_i = q1_tw_im.mul_add(r, q1_tw_re * i);
                tw_r.store_slice(&mut re_chunk[D..2 * D]);
                tw_i.store_slice(&mut im_chunk[D..2 * D]);
            }
            {
                let r = f32x4::from_slice(simd, &re_chunk[2 * D..3 * D]);
                let i = f32x4::from_slice(simd, &im_chunk[2 * D..3 * D]);
                let tw_r = q2_tw_im.mul_add(-i, q2_tw_re * r);
                let tw_i = q2_tw_im.mul_add(r, q2_tw_re * i);
                tw_r.store_slice(&mut re_chunk[2 * D..3 * D]);
                tw_i.store_slice(&mut im_chunk[2 * D..3 * D]);
            }
            {
                let r = f32x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let i = f32x4::from_slice(simd, &im_chunk[3 * D..4 * D]);
                let tw_r = q3_tw_im.mul_add(-i, q3_tw_re * r);
                let tw_i = q3_tw_im.mul_add(r, q3_tw_re * i);
                tw_r.store_slice(&mut re_chunk[3 * D..4 * D]);
                tw_i.store_slice(&mut im_chunk[3 * D..4 * D]);
            }

            // Step 2: First radix-2 BF (dist=4)
            {
                let in0_re = f32x4::from_slice(simd, &re_chunk[0..D]);
                let in1_re = f32x4::from_slice(simd, &re_chunk[D..2 * D]);
                let in0_im = f32x4::from_slice(simd, &im_chunk[0..D]);
                let in1_im = f32x4::from_slice(simd, &im_chunk[D..2 * D]);
                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);
                out0_re.store_slice(&mut re_chunk[0..D]);
                out0_im.store_slice(&mut im_chunk[0..D]);
                out1_re.store_slice(&mut re_chunk[D..2 * D]);
                out1_im.store_slice(&mut im_chunk[D..2 * D]);
            }
            {
                let in0_re = f32x4::from_slice(simd, &re_chunk[2 * D..3 * D]);
                let in1_re = f32x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let in0_im = f32x4::from_slice(simd, &im_chunk[2 * D..3 * D]);
                let in1_im = f32x4::from_slice(simd, &im_chunk[3 * D..4 * D]);
                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);
                out0_re.store_slice(&mut re_chunk[2 * D..3 * D]);
                out0_im.store_slice(&mut im_chunk[2 * D..3 * D]);
                out1_re.store_slice(&mut re_chunk[3 * D..4 * D]);
                out1_im.store_slice(&mut im_chunk[3 * D..4 * D]);
            }

            // Step 3: Trivial -j on 4th quarter (indices 12..16)
            {
                let r = f32x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let i = f32x4::from_slice(simd, &im_chunk[3 * D..4 * D]);
                i.store_slice(&mut re_chunk[3 * D..4 * D]);
                (-r).store_slice(&mut im_chunk[3 * D..4 * D]);
            }

            // Step 4: Second radix-2 BF (dist=8)
            {
                let in0_re = f32x4::from_slice(simd, &re_chunk[0..D]);
                let in1_re = f32x4::from_slice(simd, &re_chunk[2 * D..3 * D]);
                let in0_im = f32x4::from_slice(simd, &im_chunk[0..D]);
                let in1_im = f32x4::from_slice(simd, &im_chunk[2 * D..3 * D]);
                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);
                out0_re.store_slice(&mut re_chunk[0..D]);
                out0_im.store_slice(&mut im_chunk[0..D]);
                out1_re.store_slice(&mut re_chunk[2 * D..3 * D]);
                out1_im.store_slice(&mut im_chunk[2 * D..3 * D]);
            }
            {
                let in0_re = f32x4::from_slice(simd, &re_chunk[D..2 * D]);
                let in1_re = f32x4::from_slice(simd, &re_chunk[3 * D..4 * D]);
                let in0_im = f32x4::from_slice(simd, &im_chunk[D..2 * D]);
                let in1_im = f32x4::from_slice(simd, &im_chunk[3 * D..4 * D]);
                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);
                out0_re.store_slice(&mut re_chunk[D..2 * D]);
                out0_im.store_slice(&mut im_chunk[D..2 * D]);
                out1_re.store_slice(&mut re_chunk[3 * D..4 * D]);
                out1_im.store_slice(&mut im_chunk[3 * D..4 * D]);
            }
        });
}

// ---------------------------------------------------------------------------
// Pair 2: fuses stages 4+5 (chunk = 64, D = 16)
// Inter-pair twiddle: W_64^{q*k} for q=1,2,3, k=0..15 (hardcoded)
// ---------------------------------------------------------------------------

/// Radix-2^2 fused pair 2 for f64: stages 4+5, chunk_size = 64
#[inline(never)]
pub fn fft_dit_r22_pair2_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_r22_pair2_simd_f64(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_r22_pair2_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    const D: usize = 16;
    const CHUNK: usize = 64;

    // W_64^{2k} for k=0..7 = W_32^k for k=0..7
    let q1_tw_re_0 = f64x8::simd_from(
        simd,
        [
            1.0, 0.9807852804032304, 0.9238795325112867, 0.8314696123025452,
            std::f64::consts::FRAC_1_SQRT_2, 0.5555702330196022,
            0.3826834323650898, 0.19509032201612825,
        ],
    );
    let q1_tw_im_0 = f64x8::simd_from(
        simd,
        [
            0.0, -0.19509032201612825, -0.3826834323650898, -0.5555702330196022,
            -std::f64::consts::FRAC_1_SQRT_2, -0.8314696123025452,
            -0.9238795325112867, -0.9807852804032304,
        ],
    );
    // W_64^{2k} for k=8..15 = W_32^k for k=8..15
    let q1_tw_re_1 = f64x8::simd_from(
        simd,
        [
            0.0, -0.19509032201612825, -0.3826834323650898, -0.5555702330196022,
            -std::f64::consts::FRAC_1_SQRT_2, -0.8314696123025452,
            -0.9238795325112867, -0.9807852804032304,
        ],
    );
    let q1_tw_im_1 = f64x8::simd_from(
        simd,
        [
            -1.0, -0.9807852804032304, -0.9238795325112867, -0.8314696123025452,
            -std::f64::consts::FRAC_1_SQRT_2, -0.5555702330196022,
            -0.3826834323650898, -0.19509032201612825,
        ],
    );

    // W_64^k for k=0..7
    let q2_tw_re_0 = f64x8::simd_from(
        simd,
        [
            1.0, 0.9951847266721969, 0.9807852804032304, 0.9569403357322089,
            0.9238795325112867, 0.8819212643483549, 0.8314696123025452, 0.773010453362737,
        ],
    );
    let q2_tw_im_0 = f64x8::simd_from(
        simd,
        [
            0.0, -0.0980171403295606, -0.19509032201612825, -0.29028467725446233,
            -0.3826834323650898, -0.47139673682599764, -0.5555702330196022, -0.6343932841636455,
        ],
    );
    // W_64^k for k=8..15
    let q2_tw_re_1 = f64x8::simd_from(
        simd,
        [
            std::f64::consts::FRAC_1_SQRT_2, 0.6343932841636455,
            0.5555702330196022, 0.47139673682599764,
            0.3826834323650898, 0.29028467725446233,
            0.19509032201612825, 0.0980171403295606,
        ],
    );
    let q2_tw_im_1 = f64x8::simd_from(
        simd,
        [
            -std::f64::consts::FRAC_1_SQRT_2, -0.773010453362737,
            -0.8314696123025452, -0.8819212643483549,
            -0.9238795325112867, -0.9569403357322089,
            -0.9807852804032304, -0.9951847266721969,
        ],
    );

    // W_64^{3k} for k=0..7
    let q3_tw_re_0 = f64x8::simd_from(
        simd,
        [
            1.0, 0.9569403357322089, 0.8314696123025452, 0.6343932841636455,
            0.3826834323650898, 0.0980171403295606,
            -0.19509032201612825, -0.47139673682599764,
        ],
    );
    let q3_tw_im_0 = f64x8::simd_from(
        simd,
        [
            0.0, -0.29028467725446233, -0.5555702330196022, -0.773010453362737,
            -0.9238795325112867, -0.9951847266721969,
            -0.9807852804032304, -0.8819212643483549,
        ],
    );
    // W_64^{3k} for k=8..15
    let q3_tw_re_1 = f64x8::simd_from(
        simd,
        [
            -std::f64::consts::FRAC_1_SQRT_2, -0.8819212643483549,
            -0.9807852804032304, -0.9951847266721969,
            -0.9238795325112867, -0.773010453362737,
            -0.5555702330196022, -0.29028467725446233,
        ],
    );
    let q3_tw_im_1 = f64x8::simd_from(
        simd,
        [
            -std::f64::consts::FRAC_1_SQRT_2, -0.47139673682599764,
            -0.19509032201612825, 0.0980171403295606,
            0.3826834323650898, 0.6343932841636455,
            0.8314696123025452, 0.9569403357322089,
        ],
    );

    let two = f64x8::splat(simd, 2.0);

    reals
        .chunks_exact_mut(CHUNK)
        .zip(imags.chunks_exact_mut(CHUNK))
        .for_each(|(re, im)| {
            // Step 1: Inter-pair twiddle on quarters 1, 2, 3
            for (tw_re, tw_im, offset) in [
                (&q1_tw_re_0, &q1_tw_im_0, D),
                (&q1_tw_re_1, &q1_tw_im_1, D + 8),
                (&q2_tw_re_0, &q2_tw_im_0, 2 * D),
                (&q2_tw_re_1, &q2_tw_im_1, 2 * D + 8),
                (&q3_tw_re_0, &q3_tw_im_0, 3 * D),
                (&q3_tw_re_1, &q3_tw_im_1, 3 * D + 8),
            ] {
                let r = f64x8::from_slice(simd, &re[offset..offset + 8]);
                let i = f64x8::from_slice(simd, &im[offset..offset + 8]);
                let tw_r = tw_im.mul_add(-i, *tw_re * r);
                let tw_i = tw_im.mul_add(r, *tw_re * i);
                tw_r.store_slice(&mut re[offset..offset + 8]);
                tw_i.store_slice(&mut im[offset..offset + 8]);
            }

            // Step 2: First radix-2 BF (dist=16) in two sub-chunks
            for sub in 0..2 {
                let base_lo = sub * 2 * D;
                let base_hi = base_lo + D;
                for lane_off in (0..D).step_by(8) {
                    let lo = base_lo + lane_off;
                    let hi = base_hi + lane_off;
                    let in0_re = f64x8::from_slice(simd, &re[lo..lo + 8]);
                    let in1_re = f64x8::from_slice(simd, &re[hi..hi + 8]);
                    let in0_im = f64x8::from_slice(simd, &im[lo..lo + 8]);
                    let in1_im = f64x8::from_slice(simd, &im[hi..hi + 8]);
                    let out0_re = in0_re + in1_re;
                    let out0_im = in0_im + in1_im;
                    let out1_re = two.mul_sub(in0_re, out0_re);
                    let out1_im = two.mul_sub(in0_im, out0_im);
                    out0_re.store_slice(&mut re[lo..lo + 8]);
                    out0_im.store_slice(&mut im[lo..lo + 8]);
                    out1_re.store_slice(&mut re[hi..hi + 8]);
                    out1_im.store_slice(&mut im[hi..hi + 8]);
                }
            }

            // Step 3: Trivial -j on 4th quarter (indices 48..64)
            for lane_off in (0..D).step_by(8) {
                let off = 3 * D + lane_off;
                let r = f64x8::from_slice(simd, &re[off..off + 8]);
                let i = f64x8::from_slice(simd, &im[off..off + 8]);
                i.store_slice(&mut re[off..off + 8]);
                (-r).store_slice(&mut im[off..off + 8]);
            }

            // Step 4: Second radix-2 BF (dist=32)
            // Pairs: [0..16] with [32..48], [16..32] with [48..64]
            for sub in 0..2 {
                let base_lo = sub * D;
                let base_hi = base_lo + 2 * D;
                for lane_off in (0..D).step_by(8) {
                    let lo = base_lo + lane_off;
                    let hi = base_hi + lane_off;
                    let in0_re = f64x8::from_slice(simd, &re[lo..lo + 8]);
                    let in1_re = f64x8::from_slice(simd, &re[hi..hi + 8]);
                    let in0_im = f64x8::from_slice(simd, &im[lo..lo + 8]);
                    let in1_im = f64x8::from_slice(simd, &im[hi..hi + 8]);
                    let out0_re = in0_re + in1_re;
                    let out0_im = in0_im + in1_im;
                    let out1_re = two.mul_sub(in0_re, out0_re);
                    let out1_im = two.mul_sub(in0_im, out0_im);
                    out0_re.store_slice(&mut re[lo..lo + 8]);
                    out0_im.store_slice(&mut im[lo..lo + 8]);
                    out1_re.store_slice(&mut re[hi..hi + 8]);
                    out1_im.store_slice(&mut im[hi..hi + 8]);
                }
            }
        });
}

/// Radix-2^2 fused pair 2 for f32: stages 4+5, chunk_size = 64
#[inline(never)]
pub fn fft_dit_r22_pair2_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_r22_pair2_simd_f32(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_r22_pair2_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    const D: usize = 16;
    const CHUNK: usize = 64;

    // W_64^{2k} = W_32^k for k=0..15
    let q1_tw_re = f32x16::simd_from(
        simd,
        [
            1.0_f32, 0.980_785_25_f32, 0.923_879_5_f32, 0.831_469_6_f32,
            std::f32::consts::FRAC_1_SQRT_2, 0.555_570_24_f32,
            0.382_683_43_f32, 0.195_090_32_f32,
            0.0_f32, -0.195_090_32_f32,
            -0.382_683_43_f32, -0.555_570_24_f32,
            -std::f32::consts::FRAC_1_SQRT_2, -0.831_469_6_f32,
            -0.923_879_5_f32, -0.980_785_25_f32,
        ],
    );
    let q1_tw_im = f32x16::simd_from(
        simd,
        [
            0.0_f32, -0.195_090_32_f32, -0.382_683_43_f32, -0.555_570_24_f32,
            -std::f32::consts::FRAC_1_SQRT_2, -0.831_469_6_f32,
            -0.923_879_5_f32, -0.980_785_25_f32,
            -1.0_f32, -0.980_785_25_f32,
            -0.923_879_5_f32, -0.831_469_6_f32,
            -std::f32::consts::FRAC_1_SQRT_2, -0.555_570_24_f32,
            -0.382_683_43_f32, -0.195_090_32_f32,
        ],
    );

    // W_64^k for k=0..15
    let q2_tw_re = f32x16::simd_from(
        simd,
        [
            1.0_f32, 0.995_184_7_f32, 0.980_785_25_f32, 0.956_940_35_f32,
            0.923_879_5_f32, 0.881_921_3_f32, 0.831_469_6_f32, 0.773_010_43_f32,
            std::f32::consts::FRAC_1_SQRT_2, 0.634_393_3_f32,
            0.555_570_24_f32, 0.471_396_74_f32,
            0.382_683_43_f32, 0.290_284_66_f32,
            0.195_090_32_f32, 0.098_017_14_f32,
        ],
    );
    let q2_tw_im = f32x16::simd_from(
        simd,
        [
            0.0_f32, -0.098_017_14_f32, -0.195_090_32_f32, -0.290_284_66_f32,
            -0.382_683_43_f32, -0.471_396_74_f32, -0.555_570_24_f32, -0.634_393_3_f32,
            -std::f32::consts::FRAC_1_SQRT_2, -0.773_010_43_f32,
            -0.831_469_6_f32, -0.881_921_3_f32,
            -0.923_879_5_f32, -0.956_940_35_f32,
            -0.980_785_25_f32, -0.995_184_7_f32,
        ],
    );

    // W_64^{3k} for k=0..15
    let q3_tw_re = f32x16::simd_from(
        simd,
        [
            1.0_f32, 0.956_940_35_f32, 0.831_469_6_f32, 0.634_393_3_f32,
            0.382_683_43_f32, 0.098_017_14_f32,
            -0.195_090_32_f32, -0.471_396_74_f32,
            -std::f32::consts::FRAC_1_SQRT_2, -0.881_921_3_f32,
            -0.980_785_25_f32, -0.995_184_7_f32,
            -0.923_879_5_f32, -0.773_010_43_f32,
            -0.555_570_24_f32, -0.290_284_66_f32,
        ],
    );
    let q3_tw_im = f32x16::simd_from(
        simd,
        [
            0.0_f32, -0.290_284_66_f32, -0.555_570_24_f32, -0.773_010_43_f32,
            -0.923_879_5_f32, -0.995_184_7_f32,
            -0.980_785_25_f32, -0.881_921_3_f32,
            -std::f32::consts::FRAC_1_SQRT_2, -0.471_396_74_f32,
            -0.195_090_32_f32, 0.098_017_14_f32,
            0.382_683_43_f32, 0.634_393_3_f32,
            0.831_469_6_f32, 0.956_940_35_f32,
        ],
    );

    let two = f32x16::splat(simd, 2.0);

    reals
        .chunks_exact_mut(CHUNK)
        .zip(imags.chunks_exact_mut(CHUNK))
        .for_each(|(re, im)| {
            // Step 1: Inter-pair twiddle
            for (tw_re, tw_im, offset) in [
                (&q1_tw_re, &q1_tw_im, D),
                (&q2_tw_re, &q2_tw_im, 2 * D),
                (&q3_tw_re, &q3_tw_im, 3 * D),
            ] {
                let r = f32x16::from_slice(simd, &re[offset..offset + 16]);
                let i = f32x16::from_slice(simd, &im[offset..offset + 16]);
                let tw_r = tw_im.mul_add(-i, *tw_re * r);
                let tw_i = tw_im.mul_add(r, *tw_re * i);
                tw_r.store_slice(&mut re[offset..offset + 16]);
                tw_i.store_slice(&mut im[offset..offset + 16]);
            }

            // Step 2: First radix-2 BF (dist=16)
            for sub in 0..2 {
                let lo = sub * 2 * D;
                let hi = lo + D;
                let in0_re = f32x16::from_slice(simd, &re[lo..lo + 16]);
                let in1_re = f32x16::from_slice(simd, &re[hi..hi + 16]);
                let in0_im = f32x16::from_slice(simd, &im[lo..lo + 16]);
                let in1_im = f32x16::from_slice(simd, &im[hi..hi + 16]);
                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);
                out0_re.store_slice(&mut re[lo..lo + 16]);
                out0_im.store_slice(&mut im[lo..lo + 16]);
                out1_re.store_slice(&mut re[hi..hi + 16]);
                out1_im.store_slice(&mut im[hi..hi + 16]);
            }

            // Step 3: Trivial -j on 4th quarter (indices 48..64)
            {
                let r = f32x16::from_slice(simd, &re[3 * D..4 * D]);
                let i = f32x16::from_slice(simd, &im[3 * D..4 * D]);
                i.store_slice(&mut re[3 * D..4 * D]);
                (-r).store_slice(&mut im[3 * D..4 * D]);
            }

            // Step 4: Second radix-2 BF (dist=32)
            for sub in 0..2 {
                let lo = sub * D;
                let hi = lo + 2 * D;
                let in0_re = f32x16::from_slice(simd, &re[lo..lo + 16]);
                let in1_re = f32x16::from_slice(simd, &re[hi..hi + 16]);
                let in0_im = f32x16::from_slice(simd, &im[lo..lo + 16]);
                let in1_im = f32x16::from_slice(simd, &im[hi..hi + 16]);
                let out0_re = in0_re + in1_re;
                let out0_im = in0_im + in1_im;
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);
                out0_re.store_slice(&mut re[lo..lo + 16]);
                out0_im.store_slice(&mut im[lo..lo + 16]);
                out1_re.store_slice(&mut re[hi..hi + 16]);
                out1_im.store_slice(&mut im[hi..hi + 16]);
            }
        });
}

