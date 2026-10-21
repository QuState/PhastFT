//! Compact factors for generating DIT twiddles without a running recurrence.
//!
//! For k = anchor + q * LANES + r, multiply W^anchor, W^(q * LANES),
//! and W^r. Both tables are computed directly, and each anchor is evaluated
//! independently. Each result therefore has at most two complex multiplies
//! of rounding error, regardless of the transform size.

use fearless_simd::{f32x16, f32x8, f64x4, f64x8, Simd, SimdBase, SimdFloat, SimdFrom};

/// A statically dispatched source of SIMD twiddles. Blocks are independent,
/// so serial and Rayon traversals need no shared mutable state. A block may
/// start at any absolute index, and its length must not exceed `BLOCK_LEN`.
///
/// `chunks` yields plain data (stored arrays or indices). SIMD work happens in
/// `vector`, explicitly inlined into the kernel: doing it inside an iterator
/// adapter can prevent inlining across fearless_simd's target-feature boundary.
pub(crate) trait TwiddleSource<S: Simd, V>: Sync {
    /// Maximum span of one block. Stored sources can traverse the whole stage.
    const BLOCK_LEN: usize;
    /// Work assigned to a Rayon item; bounded even for stored sources.
    #[cfg(feature = "parallel")]
    const PARALLEL_BLOCK_LEN: usize;
    type Block;
    type Chunk;

    fn block(&self, simd: S, offset: usize) -> Self::Block;
    fn chunks(&self, offset: usize, len: usize) -> impl Iterator<Item = Self::Chunk>;
    fn vector(&self, simd: S, block: &Self::Block, chunk: Self::Chunk) -> (V, V);
}

/// Scalar access for real-transform endpoints and tails (at most one vector).
pub(crate) trait ScalarTwiddleSource<T> {
    fn value(&self, offset: usize) -> (T, T);
}

pub(crate) const COARSE_LEN: usize = 64;

#[derive(Clone)]
pub(crate) struct StoredTwiddles<T> {
    re: Vec<T>,
    im: Vec<T>,
}

macro_rules! impl_stored_twiddles {
    ($float:ty) => {
        impl StoredTwiddles<$float> {
            pub(crate) fn new(num_points: usize) -> Self {
                Self::new_scaled(num_points, 1.0)
            }

            pub(crate) fn new_scaled(num_points: usize, scale: f64) -> Self {
                let angle_step = -std::f64::consts::TAU / num_points as f64;
                let mut re = Vec::with_capacity(num_points / 2);
                let mut im = Vec::with_capacity(num_points / 2);
                for k in 0..num_points / 2 {
                    // Independent angles in f64 also keep f32 tables accurate.
                    let (s, c) = (angle_step * k as f64).sin_cos();
                    re.push((scale * c) as $float);
                    im.push((scale * s) as $float);
                }
                Self { re, im }
            }
        }

        impl ScalarTwiddleSource<$float> for StoredTwiddles<$float> {
            #[inline(always)]
            fn value(&self, offset: usize) -> ($float, $float) {
                (self.re[offset], self.im[offset])
            }
        }
    };
}

impl_stored_twiddles!(f64);
impl_stored_twiddles!(f32);

macro_rules! impl_stored_source {
    ($float:ty, $vector:ident, $lanes:literal, $block_len:expr) => {
        impl<S: Simd> TwiddleSource<S, $vector<S>> for StoredTwiddles<$float> {
            const BLOCK_LEN: usize = usize::MAX;
            #[cfg(feature = "parallel")]
            const PARALLEL_BLOCK_LEN: usize = $block_len;
            type Block = ();
            type Chunk = ([$float; $lanes], [$float; $lanes]);

            #[inline(always)]
            fn block(&self, _simd: S, _offset: usize) {}

            #[inline(always)]
            fn chunks(&self, offset: usize, len: usize) -> impl Iterator<Item = Self::Chunk> {
                self.re[offset..offset + len]
                    .as_chunks::<$lanes>()
                    .0
                    .iter()
                    .zip(self.im[offset..offset + len].as_chunks::<$lanes>().0.iter())
                    .map(|(re, im)| (*re, *im))
            }

            #[inline(always)]
            fn vector(
                &self,
                simd: S,
                _block: &Self::Block,
                (re, im): Self::Chunk,
            ) -> ($vector<S>, $vector<S>) {
                ($vector::simd_from(simd, re), $vector::simd_from(simd, im))
            }
        }
    };
}

impl_stored_source!(f64, f64x8, 8, TwiddleTable64::BLOCK_LEN);
impl_stored_source!(f64, f64x4, 4, TwiddleTable64::BLOCK_LEN);
impl_stored_source!(f32, f32x16, 16, TwiddleTable32::BLOCK_LEN);
impl_stored_source!(f32, f32x8, 8, TwiddleTable32::BLOCK_LEN);

#[derive(Clone)]
pub(crate) struct TwiddleTable<T, const LANES: usize> {
    pub(crate) fine_re: [T; LANES],
    pub(crate) fine_im: [T; LANES],
    pub(crate) coarse_re: [T; COARSE_LEN],
    pub(crate) coarse_im: [T; COARSE_LEN],
    angle_step: f64,
    scale: f64,
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
                Self::new_scaled(num_points, 1.0)
            }

            pub(crate) fn new_scaled(num_points: usize, scale: f64) -> Self {
                let angle_step = -std::f64::consts::TAU / num_points as f64;
                let mut table = Self {
                    fine_re: [0.0; $lanes],
                    fine_im: [0.0; $lanes],
                    coarse_re: [0.0; COARSE_LEN],
                    coarse_im: [0.0; COARSE_LEN],
                    angle_step,
                    scale,
                };
                for k in 0..$lanes {
                    let (s, c) = (angle_step * k as f64).sin_cos();
                    // Fold real-transform scaling into the fine table, keeping
                    // the vector hot loop identical to the complex transform.
                    table.fine_re[k] = (scale * c) as $float;
                    table.fine_im[k] = (scale * s) as $float;
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

        impl ScalarTwiddleSource<$float> for TwiddleTable<$float, $lanes> {
            #[inline]
            fn value(&self, offset: usize) -> ($float, $float) {
                let (s, c) = (self.angle_step * offset as f64).sin_cos();
                ((self.scale * c) as $float, (self.scale * s) as $float)
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
            #[cfg(feature = "parallel")]
            const PARALLEL_BLOCK_LEN: usize = Self::BLOCK_LEN;
            type Block = ($vector<S>, $vector<S>);
            type Chunk = usize;

            #[inline(always)] // Keep SIMD operations inside the vectorize boundary.
            fn block(&self, simd: S, offset: usize) -> Self::Block {
                let (re, im) = self.anchor(offset);
                ($vector::splat(simd, re), $vector::splat(simd, im))
            }

            #[inline(always)]
            fn chunks(&self, _offset: usize, len: usize) -> impl Iterator<Item = Self::Chunk> {
                assert!(len <= Self::BLOCK_LEN);
                0..len / $lanes
            }

            #[inline(always)]
            fn vector(
                &self,
                simd: S,
                block: &Self::Block,
                chunk: usize,
            ) -> ($vector<S>, $vector<S>) {
                let offset = chunk * $lanes;
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
