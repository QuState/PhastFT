//! Compact factors for generating DIT twiddles without a running recurrence.
//!
//! For k = anchor + q * LANES + r, multiply W^anchor, W^(q * LANES),
//! and W^r. Both tables are computed directly, and each anchor is evaluated
//! independently. Each result therefore has at most two complex multiplies
//! of rounding error, regardless of the transform size.

use fearless_simd::{f32x16, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat};

/// A statically dispatched source of SIMD twiddles. Each block is independent,
/// so serial and Rayon traversals can use the same interface without sharing
/// mutable state. Vector offsets are relative to the start of the block.
pub(crate) trait TwiddleSource<S: Simd, V>: Sync {
    const BLOCK_LEN: usize;
    type Block;

    fn block(&self, simd: S, offset: usize) -> Self::Block;
    fn vector(&self, simd: S, block: &Self::Block, offset: usize) -> (V, V);
}

pub(crate) const COARSE_LEN: usize = 64;

#[derive(Clone)]
pub(crate) struct TwiddleTable<T, const LANES: usize> {
    pub(crate) fine_re: [T; LANES],
    pub(crate) fine_im: [T; LANES],
    pub(crate) coarse_re: [T; COARSE_LEN],
    pub(crate) coarse_im: [T; COARSE_LEN],
    angle_step: f64,
}

pub(crate) type TwiddleTable64 = TwiddleTable<f64, 8>;
pub(crate) type TwiddleTable32 = TwiddleTable<f32, 16>;

impl<T, const LANES: usize> TwiddleTable<T, LANES> {
    pub(crate) const LANES: usize = LANES;
    pub(crate) const BLOCK_LEN: usize = COARSE_LEN * LANES;
}

macro_rules! impl_twiddle_table {
    ($float:ty, $lanes:literal) => {
        impl TwiddleTable<$float, $lanes> {
            pub(crate) fn new(num_points: usize) -> Self {
                let angle_step = -std::f64::consts::TAU / num_points as f64;
                let mut table = Self {
                    fine_re: [0.0; $lanes],
                    fine_im: [0.0; $lanes],
                    coarse_re: [0.0; COARSE_LEN],
                    coarse_im: [0.0; COARSE_LEN],
                    angle_step,
                };
                for k in 0..$lanes {
                    let (s, c) = (angle_step * k as f64).sin_cos();
                    table.fine_re[k] = c as $float;
                    table.fine_im[k] = s as $float;
                }
                for q in 0..COARSE_LEN {
                    let (s, c) = (angle_step * (q * $lanes) as f64).sin_cos();
                    table.coarse_re[q] = c as $float;
                    table.coarse_im[q] = s as $float;
                }
                table
            }

            // Only called at block boundaries, including arbitrary Rayon chunks.
            // Compute angles in f64 even for f32 transforms.
            #[inline]
            pub(crate) fn anchor(&self, offset: usize) -> ($float, $float) {
                if offset == 0 {
                    return (1.0, 0.0);
                }
                let (s, c) = (self.angle_step * offset as f64).sin_cos();
                (c as $float, s as $float)
            }
        }
    };
}

impl_twiddle_table!(f64, 8);
impl_twiddle_table!(f32, 16);

macro_rules! impl_compact_source {
    ($table:ty, $vector:ident, $lanes:literal) => {
        impl<S: Simd> TwiddleSource<S, $vector<S>> for $table {
            const BLOCK_LEN: usize = Self::BLOCK_LEN;
            type Block = ($vector<S>, $vector<S>);

            #[inline(always)] // Keep SIMD operations inside the vectorize boundary.
            fn block(&self, simd: S, offset: usize) -> Self::Block {
                let (re, im) = self.anchor(offset);
                ($vector::splat(simd, re), $vector::splat(simd, im))
            }

            #[inline(always)]
            fn vector(
                &self,
                simd: S,
                block: &Self::Block,
                offset: usize,
            ) -> ($vector<S>, $vector<S>) {
                let q = offset / Self::LANES;
                let r = offset % Self::LANES;
                let (anchor_re, anchor_im) = *block;
                let coarse_re = $vector::splat(simd, self.coarse_re[q]);
                let coarse_im = $vector::splat(simd, self.coarse_im[q]);
                let re = anchor_im.mul_add(-coarse_im, anchor_re * coarse_re);
                let im = anchor_im.mul_add(coarse_re, anchor_re * coarse_im);
                let fine_re = $vector::from_slice(simd, &self.fine_re[r..r + $lanes]);
                let fine_im = $vector::from_slice(simd, &self.fine_im[r..r + $lanes]);
                (
                    im.mul_add(-fine_im, re * fine_re),
                    im.mul_add(fine_re, re * fine_im),
                )
            }
        }
    };
}

impl_compact_source!(TwiddleTable64, f64x8, 8);
impl_compact_source!(TwiddleTable64, f64x4, 4);
impl_compact_source!(TwiddleTable32, f32x16, 16);
impl_compact_source!(TwiddleTable32, f32x8, 8);
