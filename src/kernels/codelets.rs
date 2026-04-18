//! FFT Codelets
//!
//! A codelet is a self-contained FFT kernel that fuses multiple stages into a single function
//! call, eliminating per-stage function call overhead and giving LLVM a wider optimization window.
//!
use fearless_simd::{
    f32x4, f32x8, f64x4, Simd, SimdBase, SimdCombine, SimdFloat, SimdFrom, SimdSplit,
};

/// FFT-32 codelet for `f64`: executes stages 0-4 (chunk_size 2 through 32) in a single function.
///
/// Register-resident implementation: all 32 complex values are loaded into f64x4 vectors,
/// all 5 butterfly stages execute in registers with no intermediate memory traffic,
/// then results are stored back.
///
/// In reality there is a lot of spilling to the stack,
/// but empirically it performs consistently better than loading/storing after every step
/// or even reducing the load/store traffic with radix-2^2.
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
        macro_rules! transpose4x4_f64 {
            ($g0:expr, $g1:expr, $g2:expr, $g3:expr) => {{
                let (t0, t1) = $g0.interleave($g2);
                let (t2, t3) = $g1.interleave($g3);
                let (r0, r1) = t0.interleave(t2);
                let (r2, r3) = t1.interleave(t3);
                (r0, r1, r2, r3)
            }};
        }

        macro_rules! radix4_transpose {
            ($va_re:expr, $vb_re:expr, $vc_re:expr, $vd_re:expr,
             $va_im:expr, $vb_im:expr, $vc_im:expr, $vd_im:expr) => {{
                let (e0_re, e1_re, e2_re, e3_re) =
                    transpose4x4_f64!($va_re, $vb_re, $vc_re, $vd_re);
                let (e0_im, e1_im, e2_im, e3_im) =
                    transpose4x4_f64!($va_im, $vb_im, $vc_im, $vd_im);

                let s01_re = e0_re + e1_re;
                let d01_re = e0_re - e1_re;
                let s23_re = e2_re + e3_re;
                let d23_re = e2_re - e3_re;
                let s01_im = e0_im + e1_im;
                let d01_im = e0_im - e1_im;
                let s23_im = e2_im + e3_im;
                let d23_im = e2_im - e3_im;

                let p0_re = s01_re + s23_re;
                let p2_re = s01_re - s23_re;
                let p0_im = s01_im + s23_im;
                let p2_im = s01_im - s23_im;

                let p1_re = d01_re + d23_im;
                let p3_re = d01_re - d23_im;
                let p1_im = d01_im - d23_re;
                let p3_im = d01_im + d23_re;

                let (r0_re, r1_re, r2_re, r3_re) = transpose4x4_f64!(p0_re, p1_re, p2_re, p3_re);
                let (r0_im, r1_im, r2_im, r3_im) = transpose4x4_f64!(p0_im, p1_im, p2_im, p3_im);

                $va_re = r0_re;
                $vb_re = r1_re;
                $vc_re = r2_re;
                $vd_re = r3_re;
                $va_im = r0_im;
                $vb_im = r1_im;
                $vc_im = r2_im;
                $vd_im = r3_im;
            }};
        }

        // ---- Load first group and do stages 0+1 ----
        // Deferring v4-v7 loads reduces peak register pressure during transpose.
        let mut v0_re = f64x4::from_slice(simd, &re[0..4]);
        let mut v1_re = f64x4::from_slice(simd, &re[4..8]);
        let mut v2_re = f64x4::from_slice(simd, &re[8..12]);
        let mut v3_re = f64x4::from_slice(simd, &re[12..16]);
        let mut v0_im = f64x4::from_slice(simd, &im[0..4]);
        let mut v1_im = f64x4::from_slice(simd, &im[4..8]);
        let mut v2_im = f64x4::from_slice(simd, &im[8..12]);
        let mut v3_im = f64x4::from_slice(simd, &im[12..16]);

        radix4_transpose!(v0_re, v1_re, v2_re, v3_re, v0_im, v1_im, v2_im, v3_im);

        // ---- Load second group and do stages 0+1 ----
        let mut v4_re = f64x4::from_slice(simd, &re[16..20]);
        let mut v5_re = f64x4::from_slice(simd, &re[20..24]);
        let mut v6_re = f64x4::from_slice(simd, &re[24..28]);
        let mut v7_re = f64x4::from_slice(simd, &re[28..32]);
        let mut v4_im = f64x4::from_slice(simd, &im[16..20]);
        let mut v5_im = f64x4::from_slice(simd, &im[20..24]);
        let mut v6_im = f64x4::from_slice(simd, &im[24..28]);
        let mut v7_im = f64x4::from_slice(simd, &im[28..32]);

        radix4_transpose!(v4_re, v5_re, v6_re, v7_re, v4_im, v5_im, v6_im, v7_im);

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

        // ---- Stages 0+1+2 fused: 8-point DIT on all 4 vectors via transpose ----
        // To reduce register pressure, process lo halves (elements 0-3) through
        // stages 0+1 first, then hi halves (elements 4-7) through stages 0+1.
        // Data stays in transposed (per-element) layout between stages 0+1 and
        // stage 2, avoiding redundant transpose pairs. Only one inverse transpose
        // at the end. Peak active register usage during stages 0+1: ~8 f32x4.
        {
            macro_rules! transpose4x4 {
                ($g0:expr, $g1:expr, $g2:expr, $g3:expr) => {{
                    let (t0, t1) = $g0.interleave($g2);
                    let (t2, t3) = $g1.interleave($g3);
                    let (r0, r1) = t0.interleave(t2);
                    let (r2, r3) = t1.interleave(t3);
                    (r0, r1, r2, r3)
                }};
            }

            // Stages 0+1 (radix-4 DIT) on transposed data.
            // Input: 4 f32x4 in per-group layout. Output: 4 f32x4 in per-element
            // (transposed) layout — p0, p1, p2, p3 where each lane is one group.
            macro_rules! radix4_transpose_fwd {
                ($ga_re:expr, $gb_re:expr, $gc_re:expr, $gd_re:expr,
                 $ga_im:expr, $gb_im:expr, $gc_im:expr, $gd_im:expr) => {{
                    let (e0_re, e1_re, e2_re, e3_re) =
                        transpose4x4!($ga_re, $gb_re, $gc_re, $gd_re);
                    let (e0_im, e1_im, e2_im, e3_im) =
                        transpose4x4!($ga_im, $gb_im, $gc_im, $gd_im);

                    // Stage 0 (dist=1)
                    let s01_re = e0_re + e1_re;
                    let d01_re = e0_re - e1_re;
                    let s23_re = e2_re + e3_re;
                    let d23_re = e2_re - e3_re;
                    let s01_im = e0_im + e1_im;
                    let d01_im = e0_im - e1_im;
                    let s23_im = e2_im + e3_im;
                    let d23_im = e2_im - e3_im;

                    // Stage 1 (dist=2): W4^0=1, W4^1=-j
                    let p0_re = s01_re + s23_re;
                    let p2_re = s01_re - s23_re;
                    let p0_im = s01_im + s23_im;
                    let p2_im = s01_im - s23_im;

                    let p1_re = d01_re + d23_im;
                    let p3_re = d01_re - d23_im;
                    let p1_im = d01_im - d23_re;
                    let p3_im = d01_im + d23_re;

                    // Return in per-element (transposed) layout
                    (p0_re, p1_re, p2_re, p3_re, p0_im, p1_im, p2_im, p3_im)
                }};
            }

            // Process lo halves (elements 0-3) — result in transposed layout
            let (g0_lo_re, g0_hi_re) = v0_re.split();
            let (g1_lo_re, g1_hi_re) = v1_re.split();
            let (g2_lo_re, g2_hi_re) = v2_re.split();
            let (g3_lo_re, g3_hi_re) = v3_re.split();
            let (g0_lo_im, g0_hi_im) = v0_im.split();
            let (g1_lo_im, g1_hi_im) = v1_im.split();
            let (g2_lo_im, g2_hi_im) = v2_im.split();
            let (g3_lo_im, g3_hi_im) = v3_im.split();

            let (p0_re, p1_re, p2_re, p3_re, p0_im, p1_im, p2_im, p3_im) = radix4_transpose_fwd!(
                g0_lo_re, g1_lo_re, g2_lo_re, g3_lo_re, g0_lo_im, g1_lo_im, g2_lo_im, g3_lo_im
            );

            // Process hi halves (elements 4-7) — result in transposed layout
            let (p4_re, p5_re, p6_re, p7_re, p4_im, p5_im, p6_im, p7_im) = radix4_transpose_fwd!(
                g0_hi_re, g1_hi_re, g2_hi_re, g3_hi_re, g0_hi_im, g1_hi_im, g2_hi_im, g3_hi_im
            );

            // Stage 2 (dist=4) — W8^k twiddles, already in per-element layout

            // W8^0 = 1+0j: just add/sub
            let r0_re = p0_re + p4_re;
            let r4_re = p0_re - p4_re;
            let r0_im = p0_im + p4_im;
            let r4_im = p0_im - p4_im;

            // W8^1 = (1-j)/√2: twiddle * p5, expressed to avoid double-negation.
            // tw = (s, -s) where s = 1/√2.
            // tw*p5 = (s*p5_re + s*p5_im, s*p5_im - s*p5_re)
            const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
            let s = f32x4::splat(simd, FRAC_1_SQRT_2);
            let tw_p5_re = s.mul_add(p5_im, s * p5_re);
            let tw_p5_im = s.mul_sub(p5_im, s * p5_re);
            let r1_re = p1_re + tw_p5_re;
            let r5_re = p1_re - tw_p5_re;
            let r1_im = p1_im + tw_p5_im;
            let r5_im = p1_im - tw_p5_im;

            // W8^2 = 0-j: -j*(re+j*im) = (im, -re)
            let r2_re = p2_re + p6_im;
            let r6_re = p2_re - p6_im;
            let r2_im = p2_im - p6_re;
            let r6_im = p2_im + p6_re;

            // W8^3 = (-1-j)/√2: twiddle * p7, expressed to avoid double-negation.
            // tw = (-s, -s) where s = 1/√2.
            // tw*p7_re = s*p7_im - s*p7_re  (fmsub, clean)
            // tw*p7_im = -(s*p7_im + s*p7_re), but we compute the positive form
            //   neg_tw_p7_im = s*p7_im + s*p7_re  (fmadd, clean)
            // and swap the +/- in the butterfly for the im component.
            let tw_p7_re = s.mul_sub(p7_im, s * p7_re);
            let neg_tw_p7_im = s.mul_add(p7_im, s * p7_re);
            let r3_re = p3_re + tw_p7_re;
            let r7_re = p3_re - tw_p7_re;
            let r3_im = p3_im - neg_tw_p7_im;
            let r7_im = p3_im + neg_tw_p7_im;

            // Single inverse transpose and recombine f32x4 → f32x8
            let (g0_lo_re, g1_lo_re, g2_lo_re, g3_lo_re) =
                transpose4x4!(r0_re, r1_re, r2_re, r3_re);
            let (g0_hi_re, g1_hi_re, g2_hi_re, g3_hi_re) =
                transpose4x4!(r4_re, r5_re, r6_re, r7_re);
            let (g0_lo_im, g1_lo_im, g2_lo_im, g3_lo_im) =
                transpose4x4!(r0_im, r1_im, r2_im, r3_im);
            let (g0_hi_im, g1_hi_im, g2_hi_im, g3_hi_im) =
                transpose4x4!(r4_im, r5_im, r6_im, r7_im);

            v0_re = g0_lo_re.combine(g0_hi_re);
            v1_re = g1_lo_re.combine(g1_hi_re);
            v2_re = g2_lo_re.combine(g2_hi_re);
            v3_re = g3_lo_re.combine(g3_hi_re);
            v0_im = g0_lo_im.combine(g0_hi_im);
            v1_im = g1_lo_im.combine(g1_hi_im);
            v2_im = g2_lo_im.combine(g2_hi_im);
            v3_im = g3_lo_im.combine(g3_hi_im);
        }

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
