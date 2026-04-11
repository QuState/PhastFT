//! FFT Codelets
//!
//! A codelet is a self-contained FFT kernel that fuses multiple stages into a single function
//! call, eliminating per-stage function call overhead and giving LLVM a wider optimization window.
//!
use fearless_simd::{f32x16, f32x4, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};

/// Legacy FFT-32 codelet for `f64`: stage-by-stage in-place with intermediate stores.
#[inline(never)]
pub fn fft_dit_codelet_32_staged_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_codelet_32_staged_simd_f64(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_codelet_32_staged_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    // Fused stages 0+1: radix-2^2 4-point DIT in a single sweep
    reals
        .chunks_exact_mut(4)
        .zip(imags.chunks_exact_mut(4))
        .for_each(|(re, im)| {
            let (a_re, a_im) = (re[0], im[0]);
            let (b_re, b_im) = (re[1], im[1]);
            let (c_re, c_im) = (re[2], im[2]);
            let (d_re, d_im) = (re[3], im[3]);

            // Stage 0: dist=1 butterflies on pairs (a,b) and (c,d)
            let (t0_re, t0_im) = (a_re + b_re, a_im + b_im);
            let (t1_re, t1_im) = (a_re - b_re, a_im - b_im);
            let (t2_re, t2_im) = (c_re + d_re, c_im + d_im);
            let (t3_re, t3_im) = (c_re - d_re, c_im - d_im);

            // Stage 1: -j multiply on t3, then dist=2 butterflies
            let (t3j_re, t3j_im) = (t3_im, -t3_re);

            re[0] = t0_re + t2_re;
            im[0] = t0_im + t2_im;
            re[1] = t1_re + t3j_re;
            im[1] = t1_im + t3j_im;
            re[2] = t0_re - t2_re;
            im[2] = t0_im - t2_im;
            re[3] = t1_re - t3j_re;
            im[3] = t1_im - t3j_im;
        });

    // Stage 2: dist=4, chunk_size=8, W_8 twiddles via f64x4
    {
        let tw_re = f64x4::simd_from(
            simd,
            [
                1.0,                              // W_8^0
                std::f64::consts::FRAC_1_SQRT_2,  // W_8^1
                0.0,                              // W_8^2
                -std::f64::consts::FRAC_1_SQRT_2, // W_8^3
            ],
        );
        let tw_im = f64x4::simd_from(
            simd,
            [
                0.0,                              // W_8^0
                -std::f64::consts::FRAC_1_SQRT_2, // W_8^1
                -1.0,                             // W_8^2
                -std::f64::consts::FRAC_1_SQRT_2, // W_8^3
            ],
        );
        let two = f64x4::splat(simd, 2.0);

        (reals.as_chunks_mut::<8>().0.iter_mut())
            .zip(imags.as_chunks_mut::<8>().0.iter_mut())
            .for_each(|(re8, im8)| {
                let (re_lo, re_hi) = re8.split_at_mut(4);
                let (im_lo, im_hi) = im8.split_at_mut(4);

                let in0_re = f64x4::from_slice(simd, re_lo);
                let in1_re = f64x4::from_slice(simd, re_hi);
                let in0_im = f64x4::from_slice(simd, im_lo);
                let in1_im = f64x4::from_slice(simd, im_hi);

                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_lo);
                out0_im.store_slice(im_lo);
                out1_re.store_slice(re_hi);
                out1_im.store_slice(im_hi);
            });
    }

    // Stage 3: dist=8, chunk_size=16, W_16 twiddles via f64x8
    {
        let tw_re = f64x8::simd_from(
            simd,
            [
                1.0,                              // W_16^0
                0.9238795325112867,               // W_16^1
                std::f64::consts::FRAC_1_SQRT_2,  // W_16^2
                0.38268343236508984,              // W_16^3
                0.0,                              // W_16^4
                -0.38268343236508984,             // W_16^5
                -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
                -0.9238795325112867,              // W_16^7
            ],
        );
        let tw_im = f64x8::simd_from(
            simd,
            [
                0.0,                              // W_16^0
                -0.38268343236508984,             // W_16^1
                -std::f64::consts::FRAC_1_SQRT_2, // W_16^2
                -0.9238795325112867,              // W_16^3
                -1.0,                             // W_16^4
                -0.9238795325112867,              // W_16^5
                -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
                -0.38268343236508984,             // W_16^7
            ],
        );
        let two = f64x8::splat(simd, 2.0);

        (reals.as_chunks_mut::<16>().0.iter_mut())
            .zip(imags.as_chunks_mut::<16>().0.iter_mut())
            .for_each(|(re16, im16)| {
                let (re_lo, re_hi) = re16.split_at_mut(8);
                let (im_lo, im_hi) = im16.split_at_mut(8);

                let in0_re = f64x8::from_slice(simd, re_lo);
                let in1_re = f64x8::from_slice(simd, re_hi);
                let in0_im = f64x8::from_slice(simd, im_lo);
                let in1_im = f64x8::from_slice(simd, im_hi);

                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_lo);
                out0_im.store_slice(im_lo);
                out1_re.store_slice(re_hi);
                out1_im.store_slice(im_hi);
            });
    }

    // Stage 4: dist=16, chunk_size=32, W_32 twiddles via 2x f64x8
    {
        let tw_re_0_7 = f64x8::simd_from(
            simd,
            [
                1.0,                             // W_32^0
                0.9807852804032304,              // W_32^1
                0.9238795325112867,              // W_32^2
                0.8314696123025452,              // W_32^3
                std::f64::consts::FRAC_1_SQRT_2, // W_32^4
                0.5555702330196022,              // W_32^5
                0.3826834323650898,              // W_32^6
                0.19509032201612825,             // W_32^7
            ],
        );
        let tw_im_0_7 = f64x8::simd_from(
            simd,
            [
                0.0,                              // W_32^0
                -0.19509032201612825,             // W_32^1
                -0.3826834323650898,              // W_32^2
                -0.5555702330196022,              // W_32^3
                -std::f64::consts::FRAC_1_SQRT_2, // W_32^4
                -0.8314696123025452,              // W_32^5
                -0.9238795325112867,              // W_32^6
                -0.9807852804032304,              // W_32^7
            ],
        );
        let tw_re_8_15 = f64x8::simd_from(
            simd,
            [
                0.0,                              // W_32^8
                -0.19509032201612825,             // W_32^9
                -0.3826834323650898,              // W_32^10
                -0.5555702330196022,              // W_32^11
                -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
                -0.8314696123025452,              // W_32^13
                -0.9238795325112867,              // W_32^14
                -0.9807852804032304,              // W_32^15
            ],
        );
        let tw_im_8_15 = f64x8::simd_from(
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
        let two = f64x8::splat(simd, 2.0);

        for (re32, im32) in reals
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(imags.as_chunks_mut::<32>().0.iter_mut())
        {
            let (re_lo, re_hi) = re32.split_at_mut(16);
            let (im_lo, im_hi) = im32.split_at_mut(16);

            // Batch 0: elements [0..8] and [16..24] with W_32^{0..7}
            let in0_re = f64x8::from_slice(simd, &re_lo[0..8]);
            let in1_re = f64x8::from_slice(simd, &re_hi[0..8]);
            let in0_im = f64x8::from_slice(simd, &im_lo[0..8]);
            let in1_im = f64x8::from_slice(simd, &im_hi[0..8]);

            let out0_re = tw_im_0_7.mul_add(-in1_im, tw_re_0_7.mul_add(in1_re, in0_re));
            let out0_im = tw_im_0_7.mul_add(in1_re, tw_re_0_7.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut re_lo[0..8]);
            out0_im.store_slice(&mut im_lo[0..8]);
            out1_re.store_slice(&mut re_hi[0..8]);
            out1_im.store_slice(&mut im_hi[0..8]);

            // Batch 1: elements [8..16] and [24..32] with W_32^{8..15}
            let in0_re = f64x8::from_slice(simd, &re_lo[8..16]);
            let in1_re = f64x8::from_slice(simd, &re_hi[8..16]);
            let in0_im = f64x8::from_slice(simd, &im_lo[8..16]);
            let in1_im = f64x8::from_slice(simd, &im_hi[8..16]);

            let out0_re = tw_im_8_15.mul_add(-in1_im, tw_re_8_15.mul_add(in1_re, in0_re));
            let out0_im = tw_im_8_15.mul_add(in1_re, tw_re_8_15.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut re_lo[8..16]);
            out0_im.store_slice(&mut im_lo[8..16]);
            out1_re.store_slice(&mut re_hi[8..16]);
            out1_im.store_slice(&mut im_hi[8..16]);
        }
    }
}

/// FFT-32 codelet for `f64`: executes stages 0-4 (chunk_size 2 through 32) in a single function.
///
/// Register-resident implementation: all 32 complex values are loaded into f64x4 vectors,
/// all 5 butterfly stages execute in registers with no intermediate memory traffic,
/// then results are stored back.
#[inline(never)]
pub fn fft_dit_codelet_32_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_codelet_32_simd_f64(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_codelet_32_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    assert_eq!(reals.len(), imags.len());

    let two = f64x4::splat(simd, 2.0);

    for (re, im) in reals.chunks_exact_mut(32).zip(imags.chunks_exact_mut(32)) {
        // ---- Load into 8 f64x4 register pairs (re + im) ----
        // v_k holds elements [4k .. 4k+3]
        let mut v0_re = f64x4::from_slice(simd, &re[0..4]);
        let mut v1_re = f64x4::from_slice(simd, &re[4..8]);
        let mut v2_re = f64x4::from_slice(simd, &re[8..12]);
        let mut v3_re = f64x4::from_slice(simd, &re[12..16]);
        let mut v4_re = f64x4::from_slice(simd, &re[16..20]);
        let mut v5_re = f64x4::from_slice(simd, &re[20..24]);
        let mut v6_re = f64x4::from_slice(simd, &re[24..28]);
        let mut v7_re = f64x4::from_slice(simd, &re[28..32]);

        let mut v0_im = f64x4::from_slice(simd, &im[0..4]);
        let mut v1_im = f64x4::from_slice(simd, &im[4..8]);
        let mut v2_im = f64x4::from_slice(simd, &im[8..12]);
        let mut v3_im = f64x4::from_slice(simd, &im[12..16]);
        let mut v4_im = f64x4::from_slice(simd, &im[16..20]);
        let mut v5_im = f64x4::from_slice(simd, &im[20..24]);
        let mut v6_im = f64x4::from_slice(simd, &im[24..28]);
        let mut v7_im = f64x4::from_slice(simd, &im[28..32]);

        // ---- Stages 0+1 fused: radix-4 DIT on each group of 4 (within each vector) ----
        // Extract scalars, compute radix-4 butterfly, repack into f64x4.
        macro_rules! radix4_inplace {
            ($v_re:expr, $v_im:expr) => {{
                let mut re_arr = [0.0f64; 4];
                let mut im_arr = [0.0f64; 4];
                $v_re.store_slice(&mut re_arr);
                $v_im.store_slice(&mut im_arr);

                let (a_re, a_im) = (re_arr[0], im_arr[0]);
                let (b_re, b_im) = (re_arr[1], im_arr[1]);
                let (c_re, c_im) = (re_arr[2], im_arr[2]);
                let (d_re, d_im) = (re_arr[3], im_arr[3]);

                // Stage 0: dist=1 butterflies
                let (t0_re, t0_im) = (a_re + b_re, a_im + b_im);
                let (t1_re, t1_im) = (a_re - b_re, a_im - b_im);
                let (t2_re, t2_im) = (c_re + d_re, c_im + d_im);
                let (t3_re, t3_im) = (c_re - d_re, c_im - d_im);

                // Stage 1: -j multiply on t3, then dist=2 butterflies
                let (t3j_re, t3j_im) = (t3_im, -t3_re);

                $v_re = f64x4::simd_from(
                    simd,
                    [t0_re + t2_re, t1_re + t3j_re, t0_re - t2_re, t1_re - t3j_re],
                );
                $v_im = f64x4::simd_from(
                    simd,
                    [t0_im + t2_im, t1_im + t3j_im, t0_im - t2_im, t1_im - t3j_im],
                );
            }};
        }

        radix4_inplace!(v0_re, v0_im);
        radix4_inplace!(v1_re, v1_im);
        radix4_inplace!(v2_re, v2_im);
        radix4_inplace!(v3_re, v3_im);
        radix4_inplace!(v4_re, v4_im);
        radix4_inplace!(v5_re, v5_im);
        radix4_inplace!(v6_re, v6_im);
        radix4_inplace!(v7_re, v7_im);

        // Butterfly macro: twiddle-multiply hi, then add/sub with lo.
        // out_lo = lo + tw*hi, out_hi = lo - tw*hi (via 2*lo - out_lo).
        macro_rules! butterfly {
            ($lo_re:expr, $lo_im:expr, $hi_re:expr, $hi_im:expr, $tw_re:expr, $tw_im:expr) => {{
                let out_lo_re = $tw_im.mul_add(-$hi_im, $tw_re.mul_add($hi_re, $lo_re));
                let out_lo_im = $tw_im.mul_add($hi_re, $tw_re.mul_add($hi_im, $lo_im));
                let out_hi_re = two.mul_sub($lo_re, out_lo_re);
                let out_hi_im = two.mul_sub($lo_im, out_lo_im);
                $lo_re = out_lo_re;
                $lo_im = out_lo_im;
                $hi_re = out_hi_re;
                $hi_im = out_hi_im;
            }};
        }

        // ---- Stage 2: dist=4, W_8 twiddles ----
        // Butterfly pairs: (v0,v1), (v2,v3), (v4,v5), (v6,v7)
        // All pairs use the same twiddle: W_8^{0,1,2,3}
        {
            let tw_re = f64x4::simd_from(
                simd,
                [
                    1.0,                              // W_8^0
                    std::f64::consts::FRAC_1_SQRT_2,  // W_8^1
                    0.0,                              // W_8^2
                    -std::f64::consts::FRAC_1_SQRT_2, // W_8^3
                ],
            );
            let tw_im = f64x4::simd_from(
                simd,
                [
                    0.0,                              // W_8^0
                    -std::f64::consts::FRAC_1_SQRT_2, // W_8^1
                    -1.0,                             // W_8^2
                    -std::f64::consts::FRAC_1_SQRT_2, // W_8^3
                ],
            );

            butterfly!(v0_re, v0_im, v1_re, v1_im, tw_re, tw_im);
            butterfly!(v2_re, v2_im, v3_re, v3_im, tw_re, tw_im);
            butterfly!(v4_re, v4_im, v5_re, v5_im, tw_re, tw_im);
            butterfly!(v6_re, v6_im, v7_re, v7_im, tw_re, tw_im);
        }

        // ---- Stage 3: dist=8, W_16 twiddles ----
        // Butterfly pairs: (v0,v2), (v1,v3), (v4,v6), (v5,v7)
        // (v0,v2) and (v4,v6) use W_16^{0,1,2,3}
        // (v1,v3) and (v5,v7) use W_16^{4,5,6,7}
        {
            let tw_lo_re = f64x4::simd_from(
                simd,
                [
                    1.0,                             // W_16^0
                    0.9238795325112867,              // W_16^1
                    std::f64::consts::FRAC_1_SQRT_2, // W_16^2
                    0.38268343236508984,             // W_16^3
                ],
            );
            let tw_lo_im = f64x4::simd_from(
                simd,
                [
                    0.0,                              // W_16^0
                    -0.38268343236508984,             // W_16^1
                    -std::f64::consts::FRAC_1_SQRT_2, // W_16^2
                    -0.9238795325112867,              // W_16^3
                ],
            );
            let tw_hi_re = f64x4::simd_from(
                simd,
                [
                    0.0,                              // W_16^4
                    -0.38268343236508984,             // W_16^5
                    -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
                    -0.9238795325112867,              // W_16^7
                ],
            );
            let tw_hi_im = f64x4::simd_from(
                simd,
                [
                    -1.0,                             // W_16^4
                    -0.9238795325112867,              // W_16^5
                    -std::f64::consts::FRAC_1_SQRT_2, // W_16^6
                    -0.38268343236508984,             // W_16^7
                ],
            );

            butterfly!(v0_re, v0_im, v2_re, v2_im, tw_lo_re, tw_lo_im);
            butterfly!(v1_re, v1_im, v3_re, v3_im, tw_hi_re, tw_hi_im);
            butterfly!(v4_re, v4_im, v6_re, v6_im, tw_lo_re, tw_lo_im);
            butterfly!(v5_re, v5_im, v7_re, v7_im, tw_hi_re, tw_hi_im);
        }

        // ---- Stage 4: dist=16, W_32 twiddles ----
        // Butterfly pairs: (v0,v4), (v1,v5), (v2,v6), (v3,v7)
        // Each pair uses its own twiddle: W_32^{0..3}, W_32^{4..7}, W_32^{8..11}, W_32^{12..15}
        {
            let tw0_re = f64x4::simd_from(
                simd,
                [
                    1.0,                // W_32^0
                    0.9807852804032304, // W_32^1
                    0.9238795325112867, // W_32^2
                    0.8314696123025452, // W_32^3
                ],
            );
            let tw0_im = f64x4::simd_from(
                simd,
                [
                    0.0,                  // W_32^0
                    -0.19509032201612825, // W_32^1
                    -0.3826834323650898,  // W_32^2
                    -0.5555702330196022,  // W_32^3
                ],
            );

            let tw1_re = f64x4::simd_from(
                simd,
                [
                    std::f64::consts::FRAC_1_SQRT_2, // W_32^4
                    0.5555702330196022,              // W_32^5
                    0.3826834323650898,              // W_32^6
                    0.19509032201612825,             // W_32^7
                ],
            );
            let tw1_im = f64x4::simd_from(
                simd,
                [
                    -std::f64::consts::FRAC_1_SQRT_2, // W_32^4
                    -0.8314696123025452,              // W_32^5
                    -0.9238795325112867,              // W_32^6
                    -0.9807852804032304,              // W_32^7
                ],
            );

            let tw2_re = f64x4::simd_from(
                simd,
                [
                    0.0,                  // W_32^8
                    -0.19509032201612825, // W_32^9
                    -0.3826834323650898,  // W_32^10
                    -0.5555702330196022,  // W_32^11
                ],
            );
            let tw2_im = f64x4::simd_from(
                simd,
                [
                    -1.0,                // W_32^8
                    -0.9807852804032304, // W_32^9
                    -0.9238795325112867, // W_32^10
                    -0.8314696123025452, // W_32^11
                ],
            );

            let tw3_re = f64x4::simd_from(
                simd,
                [
                    -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
                    -0.8314696123025452,              // W_32^13
                    -0.9238795325112867,              // W_32^14
                    -0.9807852804032304,              // W_32^15
                ],
            );
            let tw3_im = f64x4::simd_from(
                simd,
                [
                    -std::f64::consts::FRAC_1_SQRT_2, // W_32^12
                    -0.5555702330196022,              // W_32^13
                    -0.3826834323650898,              // W_32^14
                    -0.19509032201612825,             // W_32^15
                ],
            );

            butterfly!(v0_re, v0_im, v4_re, v4_im, tw0_re, tw0_im);
            butterfly!(v1_re, v1_im, v5_re, v5_im, tw1_re, tw1_im);
            butterfly!(v2_re, v2_im, v6_re, v6_im, tw2_re, tw2_im);
            butterfly!(v3_re, v3_im, v7_re, v7_im, tw3_re, tw3_im);
        }

        // ---- Store all vectors back ----
        v0_re.store_slice(&mut re[0..4]);
        v1_re.store_slice(&mut re[4..8]);
        v2_re.store_slice(&mut re[8..12]);
        v3_re.store_slice(&mut re[12..16]);
        v4_re.store_slice(&mut re[16..20]);
        v5_re.store_slice(&mut re[20..24]);
        v6_re.store_slice(&mut re[24..28]);
        v7_re.store_slice(&mut re[28..32]);

        v0_im.store_slice(&mut im[0..4]);
        v1_im.store_slice(&mut im[4..8]);
        v2_im.store_slice(&mut im[8..12]);
        v3_im.store_slice(&mut im[12..16]);
        v4_im.store_slice(&mut im[16..20]);
        v5_im.store_slice(&mut im[20..24]);
        v6_im.store_slice(&mut im[24..28]);
        v7_im.store_slice(&mut im[28..32]);
    }
}

/// Legacy FFT-32 codelet for `f32`: stage-by-stage in-place with intermediate stores.
#[inline(never)]
pub fn fft_dit_codelet_32_staged_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_codelet_32_staged_simd_f32(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_codelet_32_staged_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    // Fused stages 0+1: radix-2^2 4-point DIT in a single sweep
    reals
        .chunks_exact_mut(4)
        .zip(imags.chunks_exact_mut(4))
        .for_each(|(re, im)| {
            let (a_re, a_im) = (re[0], im[0]);
            let (b_re, b_im) = (re[1], im[1]);
            let (c_re, c_im) = (re[2], im[2]);
            let (d_re, d_im) = (re[3], im[3]);

            // Stage 0: dist=1 butterflies on pairs (a,b) and (c,d)
            let (t0_re, t0_im) = (a_re + b_re, a_im + b_im);
            let (t1_re, t1_im) = (a_re - b_re, a_im - b_im);
            let (t2_re, t2_im) = (c_re + d_re, c_im + d_im);
            let (t3_re, t3_im) = (c_re - d_re, c_im - d_im);

            // Stage 1: -j multiply on t3, then dist=2 butterflies
            let (t3j_re, t3j_im) = (t3_im, -t3_re);

            re[0] = t0_re + t2_re;
            im[0] = t0_im + t2_im;
            re[1] = t1_re + t3j_re;
            im[1] = t1_im + t3j_im;
            re[2] = t0_re - t2_re;
            im[2] = t0_im - t2_im;
            re[3] = t1_re - t3j_re;
            im[3] = t1_im - t3j_im;
        });

    // Stage 2: dist=4, chunk_size=8, W_8 twiddles via f32x4
    {
        let tw_re = f32x4::simd_from(
            simd,
            [
                1.0_f32,                          // W_8^0
                std::f32::consts::FRAC_1_SQRT_2,  // W_8^1
                0.0_f32,                          // W_8^2
                -std::f32::consts::FRAC_1_SQRT_2, // W_8^3
            ],
        );
        let tw_im = f32x4::simd_from(
            simd,
            [
                0.0_f32,                          // W_8^0
                -std::f32::consts::FRAC_1_SQRT_2, // W_8^1
                -1.0_f32,                         // W_8^2
                -std::f32::consts::FRAC_1_SQRT_2, // W_8^3
            ],
        );
        let two = f32x4::splat(simd, 2.0);

        (reals.as_chunks_mut::<8>().0.iter_mut())
            .zip(imags.as_chunks_mut::<8>().0.iter_mut())
            .for_each(|(re8, im8)| {
                let (re_lo, re_hi) = re8.split_at_mut(4);
                let (im_lo, im_hi) = im8.split_at_mut(4);

                let in0_re = f32x4::from_slice(simd, re_lo);
                let in1_re = f32x4::from_slice(simd, re_hi);
                let in0_im = f32x4::from_slice(simd, im_lo);
                let in1_im = f32x4::from_slice(simd, im_hi);

                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_lo);
                out0_im.store_slice(im_lo);
                out1_re.store_slice(re_hi);
                out1_im.store_slice(im_hi);
            });
    }

    // Stage 3: dist=8, chunk_size=16, W_16 twiddles via f32x8
    {
        let tw_re = f32x8::simd_from(
            simd,
            [
                1.0_f32,                          // W_16^0
                0.923_879_5_f32,                  // W_16^1
                std::f32::consts::FRAC_1_SQRT_2,  // W_16^2
                0.382_683_43_f32,                 // W_16^3
                0.0_f32,                          // W_16^4
                -0.382_683_43_f32,                // W_16^5
                -std::f32::consts::FRAC_1_SQRT_2, // W_16^6
                -0.923_879_5_f32,                 // W_16^7
            ],
        );
        let tw_im = f32x8::simd_from(
            simd,
            [
                0.0_f32,                          // W_16^0
                -0.382_683_43_f32,                // W_16^1
                -std::f32::consts::FRAC_1_SQRT_2, // W_16^2
                -0.923_879_5_f32,                 // W_16^3
                -1.0_f32,                         // W_16^4
                -0.923_879_5_f32,                 // W_16^5
                -std::f32::consts::FRAC_1_SQRT_2, // W_16^6
                -0.382_683_43_f32,                // W_16^7
            ],
        );
        let two = f32x8::splat(simd, 2.0);

        (reals.as_chunks_mut::<16>().0.iter_mut())
            .zip(imags.as_chunks_mut::<16>().0.iter_mut())
            .for_each(|(re16, im16)| {
                let (re_lo, re_hi) = re16.split_at_mut(8);
                let (im_lo, im_hi) = im16.split_at_mut(8);

                let in0_re = f32x8::from_slice(simd, re_lo);
                let in1_re = f32x8::from_slice(simd, re_hi);
                let in0_im = f32x8::from_slice(simd, im_lo);
                let in1_im = f32x8::from_slice(simd, im_hi);

                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_lo);
                out0_im.store_slice(im_lo);
                out1_re.store_slice(re_hi);
                out1_im.store_slice(im_hi);
            });
    }

    // Stage 4: dist=16, chunk_size=32, W_32 twiddles via f32x16
    {
        let tw_re = f32x16::simd_from(
            simd,
            [
                1.0_f32,                          // W_32^0
                0.980_785_25_f32,                 // W_32^1
                0.923_879_5_f32,                  // W_32^2
                0.831_469_6_f32,                  // W_32^3
                std::f32::consts::FRAC_1_SQRT_2,  // W_32^4
                0.555_570_24_f32,                 // W_32^5
                0.382_683_43_f32,                 // W_32^6
                0.195_090_32_f32,                 // W_32^7
                0.0_f32,                          // W_32^8
                -0.195_090_32_f32,                // W_32^9
                -0.382_683_43_f32,                // W_32^10
                -0.555_570_24_f32,                // W_32^11
                -std::f32::consts::FRAC_1_SQRT_2, // W_32^12
                -0.831_469_6_f32,                 // W_32^13
                -0.923_879_5_f32,                 // W_32^14
                -0.980_785_25_f32,                // W_32^15
            ],
        );
        let tw_im = f32x16::simd_from(
            simd,
            [
                0.0_f32,                          // W_32^0
                -0.195_090_32_f32,                // W_32^1
                -0.382_683_43_f32,                // W_32^2
                -0.555_570_24_f32,                // W_32^3
                -std::f32::consts::FRAC_1_SQRT_2, // W_32^4
                -0.831_469_6_f32,                 // W_32^5
                -0.923_879_5_f32,                 // W_32^6
                -0.980_785_25_f32,                // W_32^7
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
        let two = f32x16::splat(simd, 2.0);

        (reals.as_chunks_mut::<32>().0.iter_mut())
            .zip(imags.as_chunks_mut::<32>().0.iter_mut())
            .for_each(|(re32, im32)| {
                let (re_lo, re_hi) = re32.split_at_mut(16);
                let (im_lo, im_hi) = im32.split_at_mut(16);

                let in0_re = f32x16::from_slice(simd, re_lo);
                let in1_re = f32x16::from_slice(simd, re_hi);
                let in0_im = f32x16::from_slice(simd, im_lo);
                let in1_im = f32x16::from_slice(simd, im_hi);

                let out0_re = tw_im.mul_add(-in1_im, tw_re.mul_add(in1_re, in0_re));
                let out0_im = tw_im.mul_add(in1_re, tw_re.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(re_lo);
                out0_im.store_slice(im_lo);
                out1_re.store_slice(re_hi);
                out1_im.store_slice(im_hi);
            });
    }
}

/// FFT-32 codelet for `f32`: executes stages 0-4 (chunk_size 2 through 32) in a single function.
///
/// Register-resident implementation using `f32x8`: all 32 complex values are loaded into
/// 4 `f32x8` re + 4 `f32x8` im vectors, all 5 butterfly stages execute in registers with
/// no intermediate memory traffic, then results are stored back.
#[inline(never)]
pub fn fft_dit_codelet_32_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_codelet_32_simd_f32(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_codelet_32_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    assert_eq!(reals.len(), imags.len());

    let two = f32x8::splat(simd, 2.0);

    for (re, im) in reals.chunks_exact_mut(32).zip(imags.chunks_exact_mut(32)) {
        // ---- Load into 4 f32x8 register pairs (re + im) ----
        // v_k holds elements [8k .. 8k+7]
        let mut v0_re = f32x8::from_slice(simd, &re[0..8]);
        let mut v1_re = f32x8::from_slice(simd, &re[8..16]);
        let mut v2_re = f32x8::from_slice(simd, &re[16..24]);
        let mut v3_re = f32x8::from_slice(simd, &re[24..32]);

        let mut v0_im = f32x8::from_slice(simd, &im[0..8]);
        let mut v1_im = f32x8::from_slice(simd, &im[8..16]);
        let mut v2_im = f32x8::from_slice(simd, &im[16..24]);
        let mut v3_im = f32x8::from_slice(simd, &im[24..32]);

        // ---- Stages 0+1+2 fused: 8-point DIT on each vector (scalar) ----
        // Each f32x8 holds 8 consecutive elements. Stages 0 (dist=1), 1 (dist=2),
        // and 2 (dist=4) all have butterfly pairs within the same 8-element group.
        // We extract to scalars, do the full 8-point DIT, and repack.
        macro_rules! dit8_inplace {
            ($v_re:expr, $v_im:expr) => {{
                let mut r = [0.0f32; 8];
                let mut i = [0.0f32; 8];
                $v_re.store_slice(&mut r);
                $v_im.store_slice(&mut i);

                // Stage 0+1 fused: radix-4 on elements [0,1,2,3]
                let (a_re, a_im) = (r[0] + r[1], i[0] + i[1]);
                let (b_re, b_im) = (r[0] - r[1], i[0] - i[1]);
                let (c_re, c_im) = (r[2] + r[3], i[2] + i[3]);
                let (d_re, d_im) = (r[2] - r[3], i[2] - i[3]);
                let (dj_re, dj_im) = (d_im, -d_re); // -j * d
                r[0] = a_re + c_re;
                i[0] = a_im + c_im;
                r[1] = b_re + dj_re;
                i[1] = b_im + dj_im;
                r[2] = a_re - c_re;
                i[2] = a_im - c_im;
                r[3] = b_re - dj_re;
                i[3] = b_im - dj_im;

                // Stage 0+1 fused: radix-4 on elements [4,5,6,7]
                let (a_re, a_im) = (r[4] + r[5], i[4] + i[5]);
                let (b_re, b_im) = (r[4] - r[5], i[4] - i[5]);
                let (c_re, c_im) = (r[6] + r[7], i[6] + i[7]);
                let (d_re, d_im) = (r[6] - r[7], i[6] - i[7]);
                let (dj_re, dj_im) = (d_im, -d_re);
                r[4] = a_re + c_re;
                i[4] = a_im + c_im;
                r[5] = b_re + dj_re;
                i[5] = b_im + dj_im;
                r[6] = a_re - c_re;
                i[6] = a_im - c_im;
                r[7] = b_re - dj_re;
                i[7] = b_im - dj_im;

                // Stage 2: dist=4 butterflies between (k, k+4) with W_8 twiddles
                // W_8^0 = 1,         W_8^1 = (1-j)/√2,   W_8^2 = -j,   W_8^3 = (-1-j)/√2
                const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
                let tw_re: [f32; 4] = [1.0, FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2];
                let tw_im: [f32; 4] = [0.0, -FRAC_1_SQRT_2, -1.0, -FRAC_1_SQRT_2];

                for k in 0..4 {
                    let hi_re = tw_re[k] * r[k + 4] - tw_im[k] * i[k + 4];
                    let hi_im = tw_re[k] * i[k + 4] + tw_im[k] * r[k + 4];
                    let lo_re = r[k];
                    let lo_im = i[k];
                    r[k] = lo_re + hi_re;
                    i[k] = lo_im + hi_im;
                    r[k + 4] = lo_re - hi_re;
                    i[k + 4] = lo_im - hi_im;
                }

                $v_re = f32x8::simd_from(simd, r);
                $v_im = f32x8::simd_from(simd, i);
            }};
        }

        dit8_inplace!(v0_re, v0_im);
        dit8_inplace!(v1_re, v1_im);
        dit8_inplace!(v2_re, v2_im);
        dit8_inplace!(v3_re, v3_im);

        // Butterfly macro: twiddle-multiply hi, then add/sub with lo.
        // out_lo = lo + tw*hi, out_hi = lo - tw*hi (via 2*lo - out_lo).
        macro_rules! butterfly {
            ($lo_re:expr, $lo_im:expr, $hi_re:expr, $hi_im:expr, $tw_re:expr, $tw_im:expr) => {{
                let out_lo_re = $tw_im.mul_add(-$hi_im, $tw_re.mul_add($hi_re, $lo_re));
                let out_lo_im = $tw_im.mul_add($hi_re, $tw_re.mul_add($hi_im, $lo_im));
                let out_hi_re = two.mul_sub($lo_re, out_lo_re);
                let out_hi_im = two.mul_sub($lo_im, out_lo_im);
                $lo_re = out_lo_re;
                $lo_im = out_lo_im;
                $hi_re = out_hi_re;
                $hi_im = out_hi_im;
            }};
        }

        // ---- Stage 3: dist=8, W_16 twiddles ----
        // Butterfly pairs: (v0,v1), (v2,v3)
        // Both pairs use the same twiddle: W_16^{0..7}
        {
            let tw_re = f32x8::simd_from(
                simd,
                [
                    1.0_f32,                          // W_16^0
                    0.923_879_5_f32,                  // W_16^1
                    std::f32::consts::FRAC_1_SQRT_2,  // W_16^2
                    0.382_683_43_f32,                 // W_16^3
                    0.0_f32,                          // W_16^4
                    -0.382_683_43_f32,                // W_16^5
                    -std::f32::consts::FRAC_1_SQRT_2, // W_16^6
                    -0.923_879_5_f32,                 // W_16^7
                ],
            );
            let tw_im = f32x8::simd_from(
                simd,
                [
                    0.0_f32,                          // W_16^0
                    -0.382_683_43_f32,                // W_16^1
                    -std::f32::consts::FRAC_1_SQRT_2, // W_16^2
                    -0.923_879_5_f32,                 // W_16^3
                    -1.0_f32,                         // W_16^4
                    -0.923_879_5_f32,                 // W_16^5
                    -std::f32::consts::FRAC_1_SQRT_2, // W_16^6
                    -0.382_683_43_f32,                // W_16^7
                ],
            );

            butterfly!(v0_re, v0_im, v1_re, v1_im, tw_re, tw_im);
            butterfly!(v2_re, v2_im, v3_re, v3_im, tw_re, tw_im);
        }

        // ---- Stage 4: dist=16, W_32 twiddles ----
        // Butterfly pairs: (v0,v2), (v1,v3)
        // (v0,v2) uses W_32^{0..7}, (v1,v3) uses W_32^{8..15}
        {
            let tw_lo_re = f32x8::simd_from(
                simd,
                [
                    1.0_f32,                         // W_32^0
                    0.980_785_25_f32,                // W_32^1
                    0.923_879_5_f32,                 // W_32^2
                    0.831_469_6_f32,                 // W_32^3
                    std::f32::consts::FRAC_1_SQRT_2, // W_32^4
                    0.555_570_24_f32,                // W_32^5
                    0.382_683_43_f32,                // W_32^6
                    0.195_090_32_f32,                // W_32^7
                ],
            );
            let tw_lo_im = f32x8::simd_from(
                simd,
                [
                    0.0_f32,                          // W_32^0
                    -0.195_090_32_f32,                // W_32^1
                    -0.382_683_43_f32,                // W_32^2
                    -0.555_570_24_f32,                // W_32^3
                    -std::f32::consts::FRAC_1_SQRT_2, // W_32^4
                    -0.831_469_6_f32,                 // W_32^5
                    -0.923_879_5_f32,                 // W_32^6
                    -0.980_785_25_f32,                // W_32^7
                ],
            );
            let tw_hi_re = f32x8::simd_from(
                simd,
                [
                    0.0_f32,                          // W_32^8
                    -0.195_090_32_f32,                // W_32^9
                    -0.382_683_43_f32,                // W_32^10
                    -0.555_570_24_f32,                // W_32^11
                    -std::f32::consts::FRAC_1_SQRT_2, // W_32^12
                    -0.831_469_6_f32,                 // W_32^13
                    -0.923_879_5_f32,                 // W_32^14
                    -0.980_785_25_f32,                // W_32^15
                ],
            );
            let tw_hi_im = f32x8::simd_from(
                simd,
                [
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

            butterfly!(v0_re, v0_im, v2_re, v2_im, tw_lo_re, tw_lo_im);
            butterfly!(v1_re, v1_im, v3_re, v3_im, tw_hi_re, tw_hi_im);
        }

        // ---- Store all vectors back ----
        v0_re.store_slice(&mut re[0..8]);
        v1_re.store_slice(&mut re[8..16]);
        v2_re.store_slice(&mut re[16..24]);
        v3_re.store_slice(&mut re[24..32]);

        v0_im.store_slice(&mut im[0..8]);
        v1_im.store_slice(&mut im[8..16]);
        v2_im.store_slice(&mut im[16..24]);
        v3_im.store_slice(&mut im[24..32]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dit::*;

    fn run_stages_0_to_4_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
        assert_eq!(reals.len(), 32);
        fft_dit_chunk_2(simd, reals, imags);
        fft_dit_chunk_4_f64(simd, reals, imags);
        fft_dit_chunk_8_f64(simd, reals, imags);
        fft_dit_chunk_16_f64(simd, reals, imags);
        fft_dit_chunk_32_f64(simd, reals, imags);
    }

    fn run_stages_0_to_4_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
        assert_eq!(reals.len(), 32);
        fft_dit_chunk_2(simd, reals, imags);
        fft_dit_chunk_4_f32(simd, reals, imags);
        fft_dit_chunk_8_f32(simd, reals, imags);
        fft_dit_chunk_16_f32(simd, reals, imags);
        fft_dit_chunk_32_f32(simd, reals, imags);
    }

    #[test]
    fn codelet_32_f64_matches_staged() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        // Test with a simple impulse signal
        let mut re_staged = vec![0.0f64; 32];
        let mut im_staged = vec![0.0f64; 32];
        re_staged[0] = 1.0;

        let mut re_codelet = re_staged.clone();
        let mut im_codelet = im_staged.clone();

        dispatch!(simd_level, simd => {
            simd.vectorize(
                #[inline(always)]
                || run_stages_0_to_4_f64(simd, &mut re_staged, &mut im_staged),
            );
            fft_dit_codelet_32_f64(simd, &mut re_codelet, &mut im_codelet);
        });

        for i in 0..32 {
            assert!(
                (re_staged[i] - re_codelet[i]).abs() < 1e-14,
                "re[{i}]: staged={}, codelet={}",
                re_staged[i],
                re_codelet[i]
            );
            assert!(
                (im_staged[i] - im_codelet[i]).abs() < 1e-14,
                "im[{i}]: staged={}, codelet={}",
                im_staged[i],
                im_codelet[i]
            );
        }

        let mut re_staged: Vec<f64> = (1..=32).map(|i| i as f64).collect();
        let mut im_staged: Vec<f64> = (1..=32).map(|i| -(i as f64) * 0.5).collect();

        let mut re_codelet = re_staged.clone();
        let mut im_codelet = im_staged.clone();

        dispatch!(simd_level, simd => {
            simd.vectorize(
                #[inline(always)]
                || run_stages_0_to_4_f64(simd, &mut re_staged, &mut im_staged),
            );
            fft_dit_codelet_32_f64(simd, &mut re_codelet, &mut im_codelet);
        });

        for i in 0..32 {
            assert!(
                (re_staged[i] - re_codelet[i]).abs() < 1e-10,
                "re[{i}]: staged={}, codelet={}",
                re_staged[i],
                re_codelet[i]
            );
            assert!(
                (im_staged[i] - im_codelet[i]).abs() < 1e-10,
                "im[{i}]: staged={}, codelet={}",
                im_staged[i],
                im_codelet[i]
            );
        }
    }

    #[test]
    fn codelet_32_f32_matches_staged() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        let mut re_staged = vec![0.0f32; 32];
        let mut im_staged = vec![0.0f32; 32];
        re_staged[0] = 1.0;

        let mut re_codelet = re_staged.clone();
        let mut im_codelet = im_staged.clone();

        dispatch!(simd_level, simd => {
            simd.vectorize(
                #[inline(always)]
                || run_stages_0_to_4_f32(simd, &mut re_staged, &mut im_staged),
            );
            fft_dit_codelet_32_f32(simd, &mut re_codelet, &mut im_codelet);
        });

        for i in 0..32 {
            assert!(
                (re_staged[i] - re_codelet[i]).abs() < 1e-5,
                "re[{i}]: staged={}, codelet={}",
                re_staged[i],
                re_codelet[i]
            );
            assert!(
                (im_staged[i] - im_codelet[i]).abs() < 1e-5,
                "im[{i}]: staged={}, codelet={}",
                im_staged[i],
                im_codelet[i]
            );
        }

        let mut re_staged: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let mut im_staged: Vec<f32> = (1..=32).map(|i| -(i as f32) * 0.5).collect();

        let mut re_codelet = re_staged.clone();
        let mut im_codelet = im_staged.clone();

        dispatch!(simd_level, simd => {
            simd.vectorize(
                #[inline(always)]
                || run_stages_0_to_4_f32(simd, &mut re_staged, &mut im_staged),
            );
            fft_dit_codelet_32_f32(simd, &mut re_codelet, &mut im_codelet);
        });

        for i in 0..32 {
            assert!(
                (re_staged[i] - re_codelet[i]).abs() < 1e-4,
                "re[{i}]: staged={}, codelet={}",
                re_staged[i],
                re_codelet[i]
            );
            assert!(
                (im_staged[i] - im_codelet[i]).abs() < 1e-4,
                "im[{i}]: staged={}, codelet={}",
                im_staged[i],
                im_codelet[i]
            );
        }
    }

    #[test]
    fn codelet_32_f64_matches_legacy() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        // Test with impulse signal
        let mut re_new = vec![0.0f64; 32];
        let mut im_new = vec![0.0f64; 32];
        re_new[0] = 1.0;

        let mut re_legacy = re_new.clone();
        let mut im_legacy = im_new.clone();

        dispatch!(simd_level, simd => {
            fft_dit_codelet_32_f64(simd, &mut re_new, &mut im_new);
            fft_dit_codelet_32_staged_f64(simd, &mut re_legacy, &mut im_legacy);
        });

        for i in 0..32 {
            assert!(
                (re_new[i] - re_legacy[i]).abs() < 1e-14,
                "re[{i}]: new={}, legacy={}",
                re_new[i],
                re_legacy[i]
            );
            assert!(
                (im_new[i] - im_legacy[i]).abs() < 1e-14,
                "im[{i}]: new={}, legacy={}",
                im_new[i],
                im_legacy[i]
            );
        }

        // Test with non-trivial signal
        let mut re_new: Vec<f64> = (1..=32).map(|i| i as f64).collect();
        let mut im_new: Vec<f64> = (1..=32).map(|i| -(i as f64) * 0.5).collect();

        let mut re_legacy = re_new.clone();
        let mut im_legacy = im_new.clone();

        dispatch!(simd_level, simd => {
            fft_dit_codelet_32_f64(simd, &mut re_new, &mut im_new);
            fft_dit_codelet_32_staged_f64(simd, &mut re_legacy, &mut im_legacy);
        });

        for i in 0..32 {
            assert!(
                (re_new[i] - re_legacy[i]).abs() < 1e-10,
                "re[{i}]: new={}, legacy={}",
                re_new[i],
                re_legacy[i]
            );
            assert!(
                (im_new[i] - im_legacy[i]).abs() < 1e-10,
                "im[{i}]: new={}, legacy={}",
                im_new[i],
                im_legacy[i]
            );
        }
    }

    #[test]
    fn codelet_32_f64_matches_legacy_multi_chunk() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        let n = 128;
        let mut re_new: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let mut im_new: Vec<f64> = (0..n).map(|i| -(i as f64) * 0.05).collect();

        let mut re_legacy = re_new.clone();
        let mut im_legacy = im_new.clone();

        dispatch!(simd_level, simd => {
            fft_dit_codelet_32_f64(simd, &mut re_new, &mut im_new);
            fft_dit_codelet_32_staged_f64(simd, &mut re_legacy, &mut im_legacy);
        });

        for i in 0..n {
            assert!(
                (re_new[i] - re_legacy[i]).abs() < 1e-10,
                "re[{i}]: new={}, legacy={}",
                re_new[i],
                re_legacy[i]
            );
            assert!(
                (im_new[i] - im_legacy[i]).abs() < 1e-10,
                "im[{i}]: new={}, legacy={}",
                im_new[i],
                im_legacy[i]
            );
        }
    }

    #[test]
    fn codelet_32_f64_multi_chunk() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        // Test that the codelet correctly processes multiple 32-element chunks
        let n = 128;
        let mut re_staged: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let mut im_staged: Vec<f64> = (0..n).map(|i| -(i as f64) * 0.05).collect();

        let mut re_codelet = re_staged.clone();
        let mut im_codelet = im_staged.clone();

        dispatch!(simd_level, simd => {
            // Run individual stage kernels on all chunks
            for chunk_start in (0..n).step_by(32) {
                let re = &mut re_staged[chunk_start..chunk_start + 32];
                let im = &mut im_staged[chunk_start..chunk_start + 32];
                simd.vectorize(
                    #[inline(always)]
                    || run_stages_0_to_4_f64(simd, re, im),
                );
            }

            // Run codelet on the full array
            fft_dit_codelet_32_f64(simd, &mut re_codelet, &mut im_codelet);
        });

        for i in 0..n {
            assert!(
                (re_staged[i] - re_codelet[i]).abs() < 1e-10,
                "re[{i}]: staged={}, codelet={}",
                re_staged[i],
                re_codelet[i]
            );
            assert!(
                (im_staged[i] - im_codelet[i]).abs() < 1e-10,
                "im[{i}]: staged={}, codelet={}",
                im_staged[i],
                im_codelet[i]
            );
        }
    }

    #[test]
    fn codelet_32_f32_matches_legacy() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        // Test with impulse signal
        let mut re_new = vec![0.0f32; 32];
        let mut im_new = vec![0.0f32; 32];
        re_new[0] = 1.0;

        let mut re_legacy = re_new.clone();
        let mut im_legacy = im_new.clone();

        dispatch!(simd_level, simd => {
            fft_dit_codelet_32_f32(simd, &mut re_new, &mut im_new);
            fft_dit_codelet_32_staged_f32(simd, &mut re_legacy, &mut im_legacy);
        });

        for i in 0..32 {
            assert!(
                (re_new[i] - re_legacy[i]).abs() < 1e-5,
                "re[{i}]: new={}, legacy={}",
                re_new[i],
                re_legacy[i]
            );
            assert!(
                (im_new[i] - im_legacy[i]).abs() < 1e-5,
                "im[{i}]: new={}, legacy={}",
                im_new[i],
                im_legacy[i]
            );
        }

        // Test with non-trivial signal
        let mut re_new: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let mut im_new: Vec<f32> = (1..=32).map(|i| -(i as f32) * 0.5).collect();

        let mut re_legacy = re_new.clone();
        let mut im_legacy = im_new.clone();

        dispatch!(simd_level, simd => {
            fft_dit_codelet_32_f32(simd, &mut re_new, &mut im_new);
            fft_dit_codelet_32_staged_f32(simd, &mut re_legacy, &mut im_legacy);
        });

        for i in 0..32 {
            assert!(
                (re_new[i] - re_legacy[i]).abs() < 1e-4,
                "re[{i}]: new={}, legacy={}",
                re_new[i],
                re_legacy[i]
            );
            assert!(
                (im_new[i] - im_legacy[i]).abs() < 1e-4,
                "im[{i}]: new={}, legacy={}",
                im_new[i],
                im_legacy[i]
            );
        }
    }

    #[test]
    fn codelet_32_f32_matches_legacy_multi_chunk() {
        use fearless_simd::dispatch;

        let simd_level = fearless_simd::Level::new();

        let n = 128;
        let mut re_new: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut im_new: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.05).collect();

        let mut re_legacy = re_new.clone();
        let mut im_legacy = im_new.clone();

        dispatch!(simd_level, simd => {
            fft_dit_codelet_32_f32(simd, &mut re_new, &mut im_new);
            fft_dit_codelet_32_staged_f32(simd, &mut re_legacy, &mut im_legacy);
        });

        for i in 0..n {
            assert!(
                (re_new[i] - re_legacy[i]).abs() < 1e-4,
                "re[{i}]: new={}, legacy={}",
                re_new[i],
                re_legacy[i]
            );
            assert!(
                (im_new[i] - im_legacy[i]).abs() < 1e-4,
                "im[{i}]: new={}, legacy={}",
                im_new[i],
                im_legacy[i]
            );
        }
    }
}
