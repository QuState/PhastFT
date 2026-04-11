//! FFT Codelets
//!
//! A codelet is a self-contained FFT kernel that fuses multiple stages into a single function
//! call, eliminating per-stage function call overhead and giving LLVM a wider optimization window.
//!
use fearless_simd::{f32x4, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};

/// FFT-32 codelet for `f64`: executes stages 0-4 (chunk_size 2 through 32) in a single function.
#[inline(never)]
pub fn fft_dit_codelet_32_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_codelet_32_simd_f64(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_codelet_32_simd_f64<S: Simd>(simd: S, reals: &mut [f64], imags: &mut [f64]) {
    // Fuse stages 0+1 into a radix-2^2 4-point DIT in a single sweep
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

    // Stage 3: dist=8, chunk_size=16, W_16 twiddles via 2x f64x4
    {
        let tw_re_0_3 = f64x4::simd_from(
            simd,
            [
                1.0,                             // W_16^0
                0.9238795325112867,              // W_16^1
                std::f64::consts::FRAC_1_SQRT_2, // W_16^2
                0.38268343236508984,             // W_16^3
            ],
        );
        let tw_im_0_3 = f64x4::simd_from(
            simd,
            [
                0.0,                              // W_16^0
                -0.38268343236508984,             // W_16^1
                -std::f64::consts::FRAC_1_SQRT_2, // W_16^2
                -0.9238795325112867,              // W_16^3
            ],
        );
        // W_16^{4..7} derived from W_16^{0..3} via W_16^(k+4) = -i * W_16^k
        let two = f64x4::splat(simd, 2.0);

        (reals.as_chunks_mut::<16>().0.iter_mut())
            .zip(imags.as_chunks_mut::<16>().0.iter_mut())
            .for_each(|(re16, im16)| {
                let (re_lo, re_hi) = re16.split_at_mut(8);
                let (im_lo, im_hi) = im16.split_at_mut(8);

                // Batch 0: elements [0..4] and [8..12] with W_16^{0..3}
                let in0_re = f64x4::from_slice(simd, &re_lo[0..4]);
                let in1_re = f64x4::from_slice(simd, &re_hi[0..4]);
                let in0_im = f64x4::from_slice(simd, &im_lo[0..4]);
                let in1_im = f64x4::from_slice(simd, &im_hi[0..4]);

                let out0_re = tw_im_0_3.mul_add(-in1_im, tw_re_0_3.mul_add(in1_re, in0_re));
                let out0_im = tw_im_0_3.mul_add(in1_re, tw_re_0_3.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_lo[0..4]);
                out0_im.store_slice(&mut im_lo[0..4]);
                out1_re.store_slice(&mut re_hi[0..4]);
                out1_im.store_slice(&mut im_hi[0..4]);

                // Batch 1: elements [4..8] and [12..16] — W_16^(k+4) = -i * W_16^k
                //   out0_re = in0_re + tw_im·in1_re + tw_re·in1_im
                //   out0_im = in0_im + tw_im·in1_im - tw_re·in1_re
                let in0_re = f64x4::from_slice(simd, &re_lo[4..8]);
                let in1_re = f64x4::from_slice(simd, &re_hi[4..8]);
                let in0_im = f64x4::from_slice(simd, &im_lo[4..8]);
                let in1_im = f64x4::from_slice(simd, &im_hi[4..8]);

                let out0_re = tw_re_0_3.mul_add(in1_im, tw_im_0_3.mul_add(in1_re, in0_re));
                let out0_im = tw_re_0_3.mul_add(-in1_re, tw_im_0_3.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_lo[4..8]);
                out0_im.store_slice(&mut im_lo[4..8]);
                out1_re.store_slice(&mut re_hi[4..8]);
                out1_im.store_slice(&mut im_hi[4..8]);
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
        // W_32^{8..15} derived from W_32^{0..7} via W_32^(k+8) = -i * W_32^k
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

            // Batch 1: elements [8..16] and [24..32] — W_32^(k+8) = -i * W_32^k
            // Rearranged: out0 = in0 + (-i·W)·in1
            //   out0_re = in0_re + tw_im·in1_re + tw_re·in1_im
            //   out0_im = in0_im + tw_im·in1_im - tw_re·in1_re
            let in0_re = f64x8::from_slice(simd, &re_lo[8..16]);
            let in1_re = f64x8::from_slice(simd, &re_hi[8..16]);
            let in0_im = f64x8::from_slice(simd, &im_lo[8..16]);
            let in1_im = f64x8::from_slice(simd, &im_hi[8..16]);

            let out0_re = tw_re_0_7.mul_add(in1_im, tw_im_0_7.mul_add(in1_re, in0_re));
            let out0_im = tw_re_0_7.mul_add(-in1_re, tw_im_0_7.mul_add(in1_im, in0_im));
            let out1_re = two.mul_sub(in0_re, out0_re);
            let out1_im = two.mul_sub(in0_im, out0_im);

            out0_re.store_slice(&mut re_lo[8..16]);
            out0_im.store_slice(&mut im_lo[8..16]);
            out1_re.store_slice(&mut re_hi[8..16]);
            out1_im.store_slice(&mut im_hi[8..16]);
        }
    }
}

/// FFT-32 codelet for f32: executes stages 0-4 in a single function.
#[inline(never)]
pub fn fft_dit_codelet_32_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
    simd.vectorize(
        #[inline(always)]
        || fft_dit_codelet_32_simd_f32(simd, reals, imags),
    )
}

#[inline(always)]
fn fft_dit_codelet_32_simd_f32<S: Simd>(simd: S, reals: &mut [f32], imags: &mut [f32]) {
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

    // Stage 4: dist=16, chunk_size=32, W_32 twiddles via 2x f32x8
    {
        let tw_re_0_7 = f32x8::simd_from(
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
        let tw_im_0_7 = f32x8::simd_from(
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
        // W_32^{8..15} derived from W_32^{0..7} via W_32^(k+8) = -i * W_32^k
        let two = f32x8::splat(simd, 2.0);

        (reals.as_chunks_mut::<32>().0.iter_mut())
            .zip(imags.as_chunks_mut::<32>().0.iter_mut())
            .for_each(|(re32, im32)| {
                let (re_lo, re_hi) = re32.split_at_mut(16);
                let (im_lo, im_hi) = im32.split_at_mut(16);

                // Batch 0: elements [0..8] and [16..24] with W_32^{0..7}
                let in0_re = f32x8::from_slice(simd, &re_lo[0..8]);
                let in1_re = f32x8::from_slice(simd, &re_hi[0..8]);
                let in0_im = f32x8::from_slice(simd, &im_lo[0..8]);
                let in1_im = f32x8::from_slice(simd, &im_hi[0..8]);

                let out0_re = tw_im_0_7.mul_add(-in1_im, tw_re_0_7.mul_add(in1_re, in0_re));
                let out0_im = tw_im_0_7.mul_add(in1_re, tw_re_0_7.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_lo[0..8]);
                out0_im.store_slice(&mut im_lo[0..8]);
                out1_re.store_slice(&mut re_hi[0..8]);
                out1_im.store_slice(&mut im_hi[0..8]);

                // Batch 1: elements [8..16] and [24..32] — W_32^(k+8) = -i * W_32^k
                //   out0_re = in0_re + tw_im·in1_re + tw_re·in1_im
                //   out0_im = in0_im + tw_im·in1_im - tw_re·in1_re
                let in0_re = f32x8::from_slice(simd, &re_lo[8..16]);
                let in1_re = f32x8::from_slice(simd, &re_hi[8..16]);
                let in0_im = f32x8::from_slice(simd, &im_lo[8..16]);
                let in1_im = f32x8::from_slice(simd, &im_hi[8..16]);

                let out0_re = tw_re_0_7.mul_add(in1_im, tw_im_0_7.mul_add(in1_re, in0_re));
                let out0_im = tw_re_0_7.mul_add(-in1_re, tw_im_0_7.mul_add(in1_im, in0_im));
                let out1_re = two.mul_sub(in0_re, out0_re);
                let out1_im = two.mul_sub(in0_im, out0_im);

                out0_re.store_slice(&mut re_lo[8..16]);
                out0_im.store_slice(&mut im_lo[8..16]);
                out1_re.store_slice(&mut re_hi[8..16]);
                out1_im.store_slice(&mut im_hi[8..16]);
            });
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
}
