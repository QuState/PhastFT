//! Fixed, natural-order FFTs for split complex arrays.
//!
//! Radix-4/8 column butterflies process independent columns in separate
//! real/imaginary vectors. Connecting twiddles are fused into that pass. Small
//! register transposes put the result in frequency order without bit reversal.
//! The 16-point leaf stays in registers; larger kernels stage through bounded
//! stack arrays to limit register pressure. No execution-time heap allocation.
//!
//! Factorizations (first component runs down columns): 16 = 4×4, 32 = 2×16,
//! 64 = 8×8, 128 = 4×32, 256 = 4×64, 512 = 8×64. The 64-point kernel fuses
//! the connecting twiddles and transpose into its first phase. Larger kernels
//! reuse the fixed leaves and finish with packed transposes into natural order.
//!
//! Wide stages use eight f32 lanes, four f64 lanes on AVX2, and eight f64 lanes
//! on AVX-512. Keeping the 16-point leaf at four lanes avoids padding and makes
//! its transpose a 4×4 operation. All helpers inline into the caller's SIMD
//! context, as required by fearless_simd.

use fearless_simd::{f32x4, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};

pub(crate) const MAX_SIZE: usize = 512;

/// Twiddles are shared by the AVX2 and AVX-512 layouts. Each non-unity
/// row is contiguous, so either four or eight columns can be loaded at once.
#[derive(Clone, Default)]
pub(crate) struct Plan<T> {
    re: Vec<T>,
    im: Vec<T>,
}

impl<T: num_traits::Float> Plan<T> {
    pub(crate) fn new(mut n: usize) -> Self {
        let mut plan = Self {
            re: Vec::new(),
            im: Vec::new(),
        };
        if n > MAX_SIZE {
            return plan;
        }
        while n >= 32 {
            let radix = match n {
                32 => 2,
                64 | 512 => 8,
                _ => 4,
            };
            let m = n / radix;
            for row in 1..radix {
                for k in 0..m {
                    // Compute in f64 even for f32 plans.
                    let angle = -std::f64::consts::TAU * (row * k) as f64 / n as f64;
                    let (im, re) = angle.sin_cos();
                    plan.re.push(T::from(re).unwrap());
                    plan.im.push(T::from(im).unwrap());
                }
            }
            n = m;
        }
        plan
    }
}

macro_rules! precision {
    ($module:ident, $float:ident, $vector:ident, $wide:ident, $lanes:literal) => {
        pub(crate) mod $module {
            use super::*;

            type Plan = super::Plan<$float>;

            type V<S> = $vector<S>;
            type C<S> = (V<S>, V<S>);

            #[inline(always)]
            fn butterfly4<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>>(
                [a, b, c, d]: [(T, T); 4],
            ) -> [(T, T); 4] {
                let ac = (a.0 + c.0, a.1 + c.1);
                let bd = (b.0 + d.0, b.1 + d.1);
                let amc = (a.0 - c.0, a.1 - c.1);
                let bmd = (b.0 - d.0, b.1 - d.1);
                [
                    (ac.0 + bd.0, ac.1 + bd.1),
                    (amc.0 + bmd.1, amc.1 - bmd.0),
                    (ac.0 - bd.0, ac.1 - bd.1),
                    (amc.0 - bmd.1, amc.1 + bmd.0),
                ]
            }

            #[inline(always)]
            fn transpose<S: Simd>([a, b, c, d]: [V<S>; 4]) -> [V<S>; 4] {
                let (ac0, ac1) = a.interleave(c);
                let (bd0, bd1) = b.interleave(d);
                let (r0, r1) = ac0.interleave(bd0);
                let (r2, r3) = ac1.interleave(bd1);
                [r0, r1, r2, r3]
            }

            #[inline(always)]
            fn transpose_complex<S: Simd>(v: [C<S>; 4]) -> [C<S>; 4] {
                let re = transpose(v.map(|v| v.0));
                let im = transpose(v.map(|v| v.1));
                std::array::from_fn(|i| (re[i], im[i]))
            }

            #[inline(always)]
            fn multiply<S: Simd>(v: C<S>, re: V<S>, im: V<S>) -> C<S> {
                (v.0.mul_sub(re, v.1 * im), v.1.mul_add(re, v.0 * im))
            }

            #[inline(always)]
            fn load<S: Simd>(simd: S, re: &[$float], im: &[$float]) -> C<S> {
                (V::from_slice(simd, re), V::from_slice(simd, im))
            }

            #[inline(always)]
            fn store<S: Simd>(v: C<S>, re: &mut [$float], im: &mut [$float], scale: $float) {
                (v.0 * scale).store_slice(re);
                (v.1 * scale).store_slice(im);
            }

            #[inline(always)]
            fn fft16<S: Simd>(simd: S, re: &mut [$float], im: &mut [$float], scale: $float) {
                let re: &mut [$float; 16] = re.try_into().unwrap();
                let im: &mut [$float; 16] = im.try_into().unwrap();
                let mut v = butterfly4([
                    load(simd, &re[0..4], &im[0..4]),
                    load(simd, &re[4..8], &im[4..8]),
                    load(simd, &re[8..12], &im[8..12]),
                    load(simd, &re[12..16], &im[12..16]),
                ]);
                const C: $float = 0.9238795325112867f64 as $float;
                const S: $float = 0.3826834323650898f64 as $float;
                const H: $float = std::$float::consts::FRAC_1_SQRT_2;
                v[1] = multiply(
                    v[1],
                    V::simd_from(simd, [1.0, C, H, S]),
                    V::simd_from(simd, [0.0, -S, -H, -C]),
                );
                v[2] = multiply(
                    v[2],
                    V::simd_from(simd, [1.0, H, 0.0, -H]),
                    V::simd_from(simd, [0.0, -H, -1.0, -H]),
                );
                v[3] = multiply(
                    v[3],
                    V::simd_from(simd, [1.0, S, -H, -C]),
                    V::simd_from(simd, [0.0, -C, -H, S]),
                );
                let v = butterfly4(transpose_complex(v));
                for (i, v) in v.into_iter().enumerate() {
                    store(
                        v,
                        &mut re[4 * i..4 * i + 4],
                        &mut im[4 * i..4 * i + 4],
                        scale,
                    );
                }
            }

            #[inline(always)]
            fn scalar4(re: [$float; 4], im: [$float; 4]) -> ([$float; 4], [$float; 4]) {
                let ar = re[0] + re[2];
                let ai = im[0] + im[2];
                let br = re[1] + re[3];
                let bi = im[1] + im[3];
                let cr = re[0] - re[2];
                let ci = im[0] - im[2];
                let dr = re[1] - re[3];
                let di = im[1] - im[3];
                (
                    [ar + br, cr + di, ar - br, cr - di],
                    [ai + bi, ci - dr, ai - bi, ci + dr],
                )
            }

            #[inline(always)]
            fn fft8<S: Simd>(simd: S, re: &mut [$float], im: &mut [$float], scale: $float) {
                let re: &mut [$float; 8] = re.try_into().unwrap();
                let im: &mut [$float; 8] = im.try_into().unwrap();
                let a = scalar4([re[0], re[2], re[4], re[6]], [im[0], im[2], im[4], im[6]]);
                let b = scalar4([re[1], re[3], re[5], re[7]], [im[1], im[3], im[5], im[7]]);
                let h = std::$float::consts::FRAC_1_SQRT_2;
                // Unity, eighth roots, and -i use only sums, swaps, and a real scale.
                let br = [b.0[0], (b.0[1] + b.1[1]) * h, b.1[2], (b.1[3] - b.0[3]) * h];
                let bi = [
                    b.1[0],
                    (b.1[1] - b.0[1]) * h,
                    -b.0[2],
                    -(b.0[3] + b.1[3]) * h,
                ];
                let ar = V::simd_from(simd, a.0);
                let ai = V::simd_from(simd, a.1);
                let br = V::simd_from(simd, br);
                let bi = V::simd_from(simd, bi);
                store((ar + br, ai + bi), &mut re[..4], &mut im[..4], scale);
                store((ar - br, ai - bi), &mut re[4..], &mut im[4..], scale);
            }

            // Each instantiation fixes the factorization at compile time. Only
            // the top-level size match runs at execution time.
            macro_rules! composite {
                ($name:ident, $n:literal, $inner:ident) => {
                    #[inline(always)]
                    fn $name<S: Simd>(
                        simd: S,
                        re: &mut [$float],
                        im: &mut [$float],
                        scale: $float,
                        wr: &[$float],
                        wi: &[$float],
                    ) {
                        const M: usize = $n / 4;
                        let re: &mut [$float; $n] = re.try_into().unwrap();
                        let im: &mut [$float; $n] = im.try_into().unwrap();
                        let (wr, inner_wr) = wr.split_at(3 * M);
                        let (wi, inner_wi) = wi.split_at(3 * M);
                        let mut tmp_re = [0.0; $n];
                        let mut tmp_im = [0.0; $n];
                        for b in (0..M).step_by($lanes) {
                            let mut v = butterfly4(std::array::from_fn(|row| {
                                (
                                    $wide::from_slice(simd, &re[row * M + b..row * M + b + $lanes]),
                                    $wide::from_slice(simd, &im[row * M + b..row * M + b + $lanes]),
                                )
                            }));
                            for row in 1..4 {
                                let offset = (row - 1) * M + b;
                                let wr = $wide::from_slice(simd, &wr[offset..offset + $lanes]);
                                let wi = $wide::from_slice(simd, &wi[offset..offset + $lanes]);
                                let (r, i) = v[row];
                                v[row] = (r.mul_sub(wr, i * wi), i.mul_add(wr, r * wi));
                            }
                            for row in 0..4 {
                                v[row]
                                    .0
                                    .store_slice(&mut tmp_re[row * M + b..row * M + b + $lanes]);
                                v[row]
                                    .1
                                    .store_slice(&mut tmp_im[row * M + b..row * M + b + $lanes]);
                            }
                        }
                        for row in 0..4 {
                            $inner(
                                simd,
                                &mut tmp_re[row * M..(row + 1) * M],
                                &mut tmp_im[row * M..(row + 1) * M],
                                1.0,
                                inner_wr,
                                inner_wi,
                            );
                        }
                        // X[row + 4*k] = FFT_M(tmp[row])[k]. Transpose tiles
                        // in registers so all output stores remain contiguous.
                        for k in (0..M).step_by(4) {
                            let v = transpose_complex(std::array::from_fn(|row| {
                                load(
                                    simd,
                                    &tmp_re[row * M + k..row * M + k + 4],
                                    &tmp_im[row * M + k..row * M + k + 4],
                                )
                            }));
                            for lane in 0..4 {
                                let offset = 4 * (k + lane);
                                store(
                                    v[lane],
                                    &mut re[offset..offset + 4],
                                    &mut im[offset..offset + 4],
                                    scale,
                                );
                            }
                        }
                    }
                };
            }
            #[inline(always)]
            // Radix 2 followed by two register-resident 16-point transforms.
            fn fft32<S: Simd>(
                simd: S,
                re: &mut [$float],
                im: &mut [$float],
                scale: $float,
                wr: &[$float],
                wi: &[$float],
            ) {
                let re: &mut [$float; 32] = re.try_into().unwrap();
                let im: &mut [$float; 32] = im.try_into().unwrap();
                let mut tr = [0.0; 32];
                let mut ti = [0.0; 32];
                for k in (0..16).step_by(4) {
                    let a = load(simd, &re[k..k + 4], &im[k..k + 4]);
                    let b = load(simd, &re[k + 16..k + 20], &im[k + 16..k + 20]);
                    store(
                        (a.0 + b.0, a.1 + b.1),
                        &mut tr[k..k + 4],
                        &mut ti[k..k + 4],
                        1.0,
                    );
                    let d = multiply(
                        (a.0 - b.0, a.1 - b.1),
                        V::from_slice(simd, &wr[k..k + 4]),
                        V::from_slice(simd, &wi[k..k + 4]),
                    );
                    store(d, &mut tr[k + 16..k + 20], &mut ti[k + 16..k + 20], 1.0);
                }
                fft16(simd, &mut tr[..16], &mut ti[..16], 1.0);
                fft16(simd, &mut tr[16..], &mut ti[16..], 1.0);
                for k in (0..16).step_by(4) {
                    let a = load(simd, &tr[k..k + 4], &ti[k..k + 4]);
                    let b = load(simd, &tr[k + 16..k + 20], &ti[k + 16..k + 20]);
                    let r = a.0.interleave(b.0);
                    let i = a.1.interleave(b.1);
                    store(
                        (r.0, i.0),
                        &mut re[2 * k..2 * k + 4],
                        &mut im[2 * k..2 * k + 4],
                        scale,
                    );
                    store(
                        (r.1, i.1),
                        &mut re[2 * k + 4..2 * k + 8],
                        &mut im[2 * k + 4..2 * k + 8],
                        scale,
                    );
                }
            }
            #[inline(always)]
            fn column8_parts<S: Simd>(
                v: [($wide<S>, $wide<S>); 8],
            ) -> ([($wide<S>, $wide<S>); 4], [($wide<S>, $wide<S>); 4]) {
                let a = butterfly4([v[0], v[2], v[4], v[6]]);
                let mut b = butterfly4([v[1], v[3], v[5], v[7]]);
                let h = std::$float::consts::FRAC_1_SQRT_2;
                b[1] = ((b[1].0 + b[1].1) * h, (b[1].1 - b[1].0) * h);
                b[2] = (b[2].1, -b[2].0);
                b[3] = ((b[3].1 - b[3].0) * h, -(b[3].0 + b[3].1) * h);
                (a, b)
            }

            #[inline(always)]
            fn column8<S: Simd>(v: [($wide<S>, $wide<S>); 8]) -> [($wide<S>, $wide<S>); 8] {
                let (a, b) = column8_parts(v);
                [
                    (a[0].0 + b[0].0, a[0].1 + b[0].1),
                    (a[1].0 + b[1].0, a[1].1 + b[1].1),
                    (a[2].0 + b[2].0, a[2].1 + b[2].1),
                    (a[3].0 + b[3].0, a[3].1 + b[3].1),
                    (a[0].0 - b[0].0, a[0].1 - b[0].1),
                    (a[1].0 - b[1].0, a[1].1 - b[1].1),
                    (a[2].0 - b[2].0, a[2].1 - b[2].1),
                    (a[3].0 - b[3].0, a[3].1 - b[3].1),
                ]
            }

            #[inline(always)]
            fn transpose_wide<S: Simd>([a, b, c, d]: [$wide<S>; 4]) -> [$wide<S>; 4] {
                let (ac0, ac1) = a.interleave(c);
                let (bd0, bd1) = b.interleave(d);
                let (r0, r1) = ac0.interleave(bd0);
                let (r2, r3) = ac1.interleave(bd1);
                [r0, r1, r2, r3]
            }

            #[inline(always)]
            fn transpose8<S: Simd>(v: [$wide<S>; 8]) -> [$wide<S>; 8] {
                let (a0, a1) = v[0].interleave(v[4]);
                let (b0, b1) = v[1].interleave(v[5]);
                let (c0, c1) = v[2].interleave(v[6]);
                let (d0, d1) = v[3].interleave(v[7]);
                let lo = transpose_wide([a0, b0, c0, d0]);
                let hi = transpose_wide([a1, b1, c1, d1]);
                [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
            }

            #[inline(always)]
            fn store_transposed8<S: Simd>(
                v: [($wide<S>, $wide<S>); 8],
                re: &mut [$float],
                im: &mut [$float],
                scale: $float,
            ) {
                if $lanes == 8 {
                    let r = transpose8(v.map(|v| v.0));
                    let i = transpose8(v.map(|v| v.1));
                    for k in 0..8 {
                        (r[k] * scale).store_slice(&mut re[k * 8..k * 8 + 8]);
                        (i[k] * scale).store_slice(&mut im[k * 8..k * 8 + 8]);
                    }
                } else {
                    for j in [0, 4] {
                        let r = transpose_wide([v[j].0, v[j + 1].0, v[j + 2].0, v[j + 3].0]);
                        let i = transpose_wide([v[j].1, v[j + 1].1, v[j + 2].1, v[j + 3].1]);
                        for k in 0..4 {
                            let offset = k * 8 + j;
                            (r[k] * scale).store_slice(&mut re[offset..offset + 4]);
                            (i[k] * scale).store_slice(&mut im[offset..offset + 4]);
                        }
                    }
                }
            }

            #[inline(always)]
            // An 8×8 transpose lets both phases use column butterflies.
            fn fft64<S: Simd>(
                simd: S,
                re: &mut [$float],
                im: &mut [$float],
                scale: $float,
                wr: &[$float],
                wi: &[$float],
            ) {
                let re: &mut [$float; 64] = re.try_into().unwrap();
                let im: &mut [$float; 64] = im.try_into().unwrap();
                let mut tr = [0.0; 64];
                let mut ti = [0.0; 64];
                for b in (0..8).step_by($lanes) {
                    let mut v = column8(std::array::from_fn(|row| {
                        (
                            $wide::from_slice(simd, &re[row * 8 + b..row * 8 + b + $lanes]),
                            $wide::from_slice(simd, &im[row * 8 + b..row * 8 + b + $lanes]),
                        )
                    }));
                    for row in 1..8 {
                        let offset = (row - 1) * 8 + b;
                        let wr = $wide::from_slice(simd, &wr[offset..offset + $lanes]);
                        let wi = $wide::from_slice(simd, &wi[offset..offset + $lanes]);
                        let (r, i) = v[row];
                        v[row] = (r.mul_sub(wr, i * wi), i.mul_add(wr, r * wi));
                    }
                    store_transposed8(
                        v,
                        &mut tr[b * 8..(b + $lanes) * 8],
                        &mut ti[b * 8..(b + $lanes) * 8],
                        1.0,
                    );
                }
                for j in (0..8).step_by($lanes) {
                    let v = column8(std::array::from_fn(|k| {
                        (
                            $wide::from_slice(simd, &tr[k * 8 + j..k * 8 + j + $lanes]),
                            $wide::from_slice(simd, &ti[k * 8 + j..k * 8 + j + $lanes]),
                        )
                    }));
                    for k in 0..8 {
                        (v[k].0 * scale).store_slice(&mut re[k * 8 + j..k * 8 + j + $lanes]);
                        (v[k].1 * scale).store_slice(&mut im[k * 8 + j..k * 8 + j + $lanes]);
                    }
                }
            }

            composite!(fft128, 128, fft32);
            composite!(fft256, 256, fft64);
            #[inline(always)]
            // Eight 64-point leaves avoid the extra stages of four 128-point leaves.
            fn fft512<S: Simd>(
                simd: S,
                re: &mut [$float],
                im: &mut [$float],
                scale: $float,
                wr: &[$float],
                wi: &[$float],
            ) {
                let re: &mut [$float; 512] = re.try_into().unwrap();
                let im: &mut [$float; 512] = im.try_into().unwrap();
                let (wr, inner_wr) = wr.split_at(448);
                let (wi, inner_wi) = wi.split_at(448);
                let mut tr = [0.0; 512];
                let mut ti = [0.0; 512];
                for b in (0..64).step_by($lanes) {
                    let (even, odd) = column8_parts(std::array::from_fn(|row| {
                        (
                            $wide::from_slice(simd, &re[row * 64 + b..row * 64 + b + $lanes]),
                            $wide::from_slice(simd, &im[row * 64 + b..row * 64 + b + $lanes]),
                        )
                    }));
                    // Consume paired outputs immediately, leaving registers for
                    // twiddles instead of keeping all eight output rows live.
                    for k in 0..4 {
                        let a = even[k];
                        let c = odd[k];
                        for (row, (mut r, mut i)) in
                            [(k, (a.0 + c.0, a.1 + c.1)), (k + 4, (a.0 - c.0, a.1 - c.1))]
                        {
                            if row != 0 {
                                let offset = (row - 1) * 64 + b;
                                let wr = $wide::from_slice(simd, &wr[offset..offset + $lanes]);
                                let wi = $wide::from_slice(simd, &wi[offset..offset + $lanes]);
                                (r, i) = (r.mul_sub(wr, i * wi), i.mul_add(wr, r * wi));
                            }
                            r.store_slice(&mut tr[row * 64 + b..row * 64 + b + $lanes]);
                            i.store_slice(&mut ti[row * 64 + b..row * 64 + b + $lanes]);
                        }
                    }
                }
                for row in 0..8 {
                    fft64(
                        simd,
                        &mut tr[row * 64..row * 64 + 64],
                        &mut ti[row * 64..row * 64 + 64],
                        1.0,
                        inner_wr,
                        inner_wi,
                    );
                }
                for k in (0..64).step_by($lanes) {
                    let v = std::array::from_fn(|row| {
                        (
                            $wide::from_slice(simd, &tr[row * 64 + k..row * 64 + k + $lanes]),
                            $wide::from_slice(simd, &ti[row * 64 + k..row * 64 + k + $lanes]),
                        )
                    });
                    store_transposed8(
                        v,
                        &mut re[8 * k..8 * (k + $lanes)],
                        &mut im[8 * k..8 * (k + $lanes)],
                        scale,
                    );
                }
            }

            #[inline(always)]
            pub(crate) fn fft<S: Simd>(
                simd: S,
                re: &mut [$float],
                im: &mut [$float],
                scale: $float,
                plan: &Plan,
            ) {
                match re.len() {
                    1 => {}
                    2 => {
                        let r = (re[0] + re[1], re[0] - re[1]);
                        let i = (im[0] + im[1], im[0] - im[1]);
                        re.copy_from_slice(&[r.0 * scale, r.1 * scale]);
                        im.copy_from_slice(&[i.0 * scale, i.1 * scale]);
                    }
                    4 => {
                        let (r, i) = scalar4(re.try_into().unwrap(), im.try_into().unwrap());
                        store(
                            (V::simd_from(simd, r), V::simd_from(simd, i)),
                            re,
                            im,
                            scale,
                        );
                    }
                    8 => fft8(simd, re, im, scale),
                    16 => fft16(simd, re, im, scale),
                    32 => fft32(simd, re, im, scale, &plan.re, &plan.im),
                    64 => fft64(simd, re, im, scale, &plan.re, &plan.im),
                    128 => fft128(simd, re, im, scale, &plan.re, &plan.im),
                    256 => fft256(simd, re, im, scale, &plan.re, &plan.im),
                    512 => fft512(simd, re, im, scale, &plan.re, &plan.im),
                    _ => unreachable!("unsupported small FFT size"),
                }
            }
        }
    };
}

precision!(single, f32, f32x4, f32x8, 8);
precision!(double, f64, f64x4, f64x4, 4);
precision!(double_wide, f64, f64x4, f64x8, 8);

#[cfg(test)]
mod tests {
    use crate::options::Options;
    use crate::planner::{Direction, PlannerDit32, PlannerDit64};
    use crate::{fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts};
    use fearless_simd::Level;
    use utilities::rustfft::{num_complex::Complex, FftPlanner};

    fn levels() -> Vec<Level> {
        let level = Level::new();
        let mut levels = vec![level];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(sse2) = level.as_sse2() {
                levels.push(Level::Sse2(sse2));
            }
            if let Some(avx2) = level.as_avx2() {
                levels.push(Level::Avx2(avx2));
            }
        }
        levels
    }

    macro_rules! correctness {
        ($name:ident, $float:ty, $planner:ty, $fft:ident, $tolerance:expr) => {
            #[test]
            fn $name() {
                let mut reference_planner = FftPlanner::<$float>::new();
                // Include the first size beyond the small path. Exercise every
                // available backend, including SSE2 without FMA on x86.
                for level in levels() {
                    for log_n in 0..=10 {
                        let n = 1 << log_n;
                        let mut planner = <$planner>::new(n);
                        planner.simd_level = level;
                        let planner = planner.clone();
                        let mut opts = Options::guess_options(n);
                        opts.multithreaded_bit_reversal = true;
                        opts.smallest_parallel_chunk_size = 0;
                        for case in 0..4 {
                            let input: Vec<Complex<$float>> = (0..n).map(|i| {
                                match case {
                                    0 => Complex::default(),
                                    1 if i == n / 3 => Complex::new(0.75, -0.25),
                                    1 => Complex::default(),
                                    2 => Complex::new(
                                        ((i * 17 % 257) as $float - 128.0) / 128.0,
                                        ((i * 31 % 251) as $float - 125.0) / 125.0,
                                    ),
                                    _ => {
                                        let phase = std::f64::consts::TAU * (i * (n / 4 + 1)) as f64 / n as f64;
                                        Complex::new(phase.cos() as $float, phase.sin() as $float)
                                    },
                                }
                            }).collect();
                            for direction in [Direction::Forward, Direction::Inverse] {
                                let mut expected = input.clone();
                                let reference = match direction {
                                    Direction::Forward => reference_planner.plan_fft_forward(n),
                                    Direction::Inverse => reference_planner.plan_fft_inverse(n),
                                };
                                reference.process(&mut expected);
                                if direction == Direction::Inverse {
                                    for z in &mut expected {
                                        *z /= n as $float;
                                    }
                                }
                                // Offset the slices independently and protect their ends.
                                let mut re = vec![12345.0; n + 3];
                                let mut im = vec![12345.0; n + 5];
                                for (i, z) in input.iter().enumerate() {
                                    re[i + 1] = z.re;
                                    im[i + 3] = z.im;
                                }
                                $fft(&mut re[1..n+1], &mut im[3..n+3], direction, &planner, &opts);
                                let scale = if direction == Direction::Inverse { 1.0 / n as f64 } else { 1.0 };
                                for k in 0..n {
                                    let error = (re[k+1] as f64 - expected[k].re as f64).hypot(im[k+3] as f64 - expected[k].im as f64);
                                    assert!(error <= $tolerance * n as f64 * scale,
                                        "n={n}, k={k}, case={case}, {direction:?}, {level:?}: error {error}");
                                }
                                assert!(re[..1].iter().chain(&re[n+1..]).all(|&x| x == 12345.0));
                                assert!(im[..3].iter().chain(&im[n+3..]).all(|&x| x == 12345.0));
                            }
                        }
                    }
                }
            }
        };
    }

    correctness!(
        small_f32,
        f32,
        PlannerDit32,
        fft_f32_dit_with_planner_and_opts,
        5e-7
    );
    correctness!(
        small_f64,
        f64,
        PlannerDit64,
        fft_f64_dit_with_planner_and_opts,
        3e-15
    );

    #[test]
    #[should_panic]
    fn small_rejects_wrong_planner() {
        fft_f64_dit_with_planner_and_opts(
            &mut [0.0; 8],
            &mut [0.0; 8],
            Direction::Forward,
            &PlannerDit64::new(16),
            &Options::default(),
        );
    }

    #[test]
    #[should_panic]
    fn small_rejects_mismatched_slices() {
        fft_f32_dit_with_planner_and_opts(
            &mut [0.0; 8],
            &mut [0.0; 4],
            Direction::Forward,
            &PlannerDit32::new(8),
            &Options::default(),
        );
    }
}
