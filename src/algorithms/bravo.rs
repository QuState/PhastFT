//! CO-BRAVO: Cache-Optimal Bit-Reversal Algorithm using Vector permute Operations
//!
//! This implements the algorithm from "Optimal Bit-Reversal Using Vector Permutations"
//! by Lokhmotov and Mycroft (SPAA'07).
//!
//! The algorithm uses vector interleaving operations to perform bit-reversal permutation.
//! For `N = 2^n` elements with `W`-element vectors, the algorithm performs `log_2(N)` rounds
//! of in-place interleave operations on pairs of vectors.

#![allow(clippy::manual_swap)] // cannot be applied to expressions, so needs to be defined here

use fearless_simd::prelude::*;
use fearless_simd::{f32x4, f32x8, f64x4, f64x8, Simd};

// Tile side length for COBRAVO, per element type.
// B² × sizeof(T) must fit comfortably in L1d as a stack buffer.
// Constraint: B must be a power of two and B >= LANES (the SIMD vector width).

const TILE_SIDE_F32: usize = 64; // 64² × 4 = 16 KB
const TILE_SIDE_F64: usize = 32; // 32² × 8 = 8 KB

// Minimum number of tiles before engaging the tile loop.
// With fewer tiles the staging overhead dominates.
const MIN_TILES: usize = 16;

/// Copy TILE_SIDE strips from `data` into `buf`.
///
/// Tile layout: strip `u` of tile `tile` is `data_chunks[u * strip_stride + tile]`,
/// and maps to `buf[u]`.
#[inline(always)]
fn stage_in<T: Copy, const TILE_SIDE: usize>(
    data: &[[T; TILE_SIDE]],
    buf: &mut [[T; TILE_SIDE]; TILE_SIDE],
    tile: usize,
) {
    let strip_stride = data.len() / TILE_SIDE;
    for u in 0..TILE_SIDE {
        buf[u] = data[u * strip_stride + tile];
    }
}

/// Copy `buf` back into TILE_SIDE strips in `data`.
#[inline(always)]
fn stage_out<T: Copy, const TILE_SIDE: usize>(
    buf: &[[T; TILE_SIDE]; TILE_SIDE],
    data: &mut [[T; TILE_SIDE]],
    tile: usize,
) {
    let strip_stride = data.len() / TILE_SIDE;
    for u in 0..TILE_SIDE {
        data[u * strip_stride + tile] = buf[u];
    }
}

/// Swap `buf` contents with tile `tile_rev`'s strips in `data`.
#[inline(always)]
fn stage_swap<T: Copy, const TILE_SIDE: usize>(
    data: &mut [[T; TILE_SIDE]],
    buf: &mut [[T; TILE_SIDE]; TILE_SIDE],
    tile_rev: usize,
) {
    let strip_stride = data.len() / TILE_SIDE;
    #[allow(clippy::needless_range_loop)] // the clippy version is not clearer
    for u in 0..TILE_SIDE {
        let data_idx = u * strip_stride + tile_rev;
        // Clippy wants to use std::mem::swap here,
        // but that's much slower on benchmarks than implementing it manually.
        // It's not clear why, possibly because fixed sizes don't propagate.
        let tmp = buf[u];
        buf[u] = data[data_idx];
        data[data_idx] = tmp;
    }
}

/// Macro to generate bit_rev_bravo implementations for concrete types.
/// Used instead of generics because `fearless_simd` doesn't let us be generic over the exact float type.
macro_rules! impl_bit_rev_bravo {
    ($fn_name:ident, $buf_fn_name:ident, $elem_ty:ty, $vec_ty:ty, $lanes:expr, $tile_side:expr) => {
        /// Inner helper: runs the BRAVO class loop on a contiguous slice.
        /// The slice must have length 2^n and at least LANES² elements.
        #[inline(always)]
        fn $buf_fn_name<S: Simd>(simd: S, data: &mut [$elem_ty], n: usize) {
            type Chunk<S> = $vec_ty;
            const LANES: usize = $lanes; // Vector width W

            // as of Rust 1.93 we cannot use an associated constant for array lengths
            assert!(<Chunk<S>>::N == LANES);

            let big_n = 1usize << n; // 2.pow(n)
            assert_eq!(data.len(), big_n, "Data length must be 2^n");

            const LOG_W: usize = LANES.ilog2() as usize;

            // π = N / W² = number of equivalence classes
            let num_classes = big_n / (LANES * LANES);
            let class_bits = n - 2 * LOG_W;

            let (data_chunks, _) = data.as_chunks_mut::<LANES>();

            // Process each equivalence class.
            // For in-place operation, we handle class pairs that swap with each other.
            // We only process when class_idx <= class_idx_rev to avoid double processing.

            // Hoisted out of the loop to avoid re-initializing on every iteration.
            // Every element is overwritten by from_slice before being read.
            let mut chunks_a: [Chunk<S>; LANES] = [Chunk::splat(simd, Default::default()); LANES];
            let mut chunks_b: [Chunk<S>; LANES] = [Chunk::splat(simd, Default::default()); LANES];

            for class_idx in 0..num_classes {
                let class_idx_rev = if class_bits > 0 {
                    class_idx.reverse_bits() >> (usize::BITS - class_bits as u32)
                } else {
                    0
                };

                // Only process if this is the "first" of a swapping pair (or self-mapping)
                if class_idx > class_idx_rev {
                    continue;
                }

                // Load vectors for class A
                for j in 0..LANES {
                    chunks_a[j] =
                        Chunk::from_slice(simd, &data_chunks[class_idx + j * num_classes]);
                }

                // Perform interleave rounds for class A
                // Each pair reads (idx0, idx1) and writes to the same (idx0, idx1),
                // so no aliasing conflict occurs within a round.
                for round in 0..LOG_W {
                    let stride = 1 << round; // 2.pow(round)

                    let mut i = 0;
                    while i < LANES {
                        for offset in 0..stride {
                            let idx0 = i + offset;
                            let idx1 = i + offset + stride;
                            let vec0 = chunks_a[idx0];
                            let vec1 = chunks_a[idx1];
                            chunks_a[idx0] = vec0.zip_low(vec1);
                            chunks_a[idx1] = vec0.zip_high(vec1);
                        }
                        i += stride * 2;
                    }
                }

                if class_idx == class_idx_rev {
                    // Self-mapping class - just write back to same location
                    for j in 0..LANES {
                        chunks_a[j].store_slice(&mut data_chunks[class_idx + j * num_classes]);
                    }
                } else {
                    // Swapping pair - load class B, process it, then swap both
                    for j in 0..LANES {
                        chunks_b[j] =
                            Chunk::from_slice(simd, &data_chunks[class_idx_rev + j * num_classes]);
                    }

                    // Perform interleave rounds for class B
                    for round in 0..LOG_W {
                        let stride = 1 << round; // 2.pow(round)

                        let mut i = 0;
                        while i < LANES {
                            for offset in 0..stride {
                                let idx0 = i + offset;
                                let idx1 = i + offset + stride;
                                let vec0 = chunks_b[idx0];
                                let vec1 = chunks_b[idx1];
                                chunks_b[idx0] = vec0.zip_low(vec1);
                                chunks_b[idx1] = vec0.zip_high(vec1);
                            }
                            i += stride * 2;
                        }
                    }

                    // Swap: write A's result to B's location and vice versa
                    for j in 0..LANES {
                        chunks_a[j].store_slice(&mut data_chunks[class_idx_rev + j * num_classes]);
                        chunks_b[j].store_slice(&mut data_chunks[class_idx + j * num_classes]);
                    }
                }
            }
        }

        /// Outer function: COBRAVO tile loop for large arrays, direct BRAVO for
        /// medium arrays, scalar fallback for tiny arrays.
        #[inline(always)] // required by fearless_simd
        fn $fn_name<S: Simd>(simd: S, data: &mut [$elem_ty], n: usize) {
            const LANES: usize = $lanes;
            let big_n = 1usize << n;
            assert_eq!(data.len(), big_n, "Data length must be 2^n");

            // For very small arrays, fall back to scalar bit-reversal
            if big_n < LANES * LANES {
                scalar_bit_reversal(data, n);
                return;
            }

            // Below tile threshold: not enough tiles to amortize staging overhead,
            // or data fits in cache. Run BRAVO directly.
            const TILE_SIDE: usize = $tile_side;
            if big_n <= TILE_SIDE * TILE_SIDE * MIN_TILES {
                $buf_fn_name(simd, data, n);
                return;
            }

            // COBRAVO: gather strips of data in a contiguous buffer,
            // then operate on it for better cache locality.
            // Each tile is TILE_SIDE strips of TILE_SIDE contiguous elements.
            // The buffer fits in L1d, so the strided BRAVO loads stay in cache.
            const N_BUF: usize = 2 * TILE_SIDE.ilog2() as usize; // log2(TILE_SIDE²)
            let tile_bits = n - N_BUF;
            let num_tiles = 1usize << tile_bits;

            let (data_tiles, _) = data.as_chunks_mut::<TILE_SIDE>();
            let mut buf = [[<$elem_ty>::default(); TILE_SIDE]; TILE_SIDE];

            for tile in 0..num_tiles {
                let tile_rev = reverse_bits_scalar(tile, tile_bits as u32);
                if tile > tile_rev {
                    continue;
                }

                stage_in(data_tiles, &mut buf, tile);
                $buf_fn_name(simd, buf.as_flattened_mut(), N_BUF);

                if tile == tile_rev {
                    stage_out(&buf, data_tiles, tile);
                } else {
                    // Swap-pair dance: buf holds BRAVO(tile t), swap with tile_rev,
                    // then BRAVO the swapped-in data and write to tile t.
                    stage_swap(data_tiles, &mut buf, tile_rev);
                    $buf_fn_name(simd, buf.as_flattened_mut(), N_BUF);
                    stage_out(&buf, data_tiles, tile);
                }
            }
        }
    };
}

// Generate concrete implementations for f32 and f64
// This needed for two reasons:
// 1. fearless_simd doesn't support being generic over the element type
// 2. As of Rust 1.93 we cannot use an associated constant for array lengths,
//    which is necessary for using the native vector width
impl_bit_rev_bravo!(
    bit_rev_bravo_chunk_4_f32,
    bravo_on_buf_chunk_4_f32,
    f32,
    f32x4<S>,
    4,
    TILE_SIDE_F32
);
impl_bit_rev_bravo!(
    bit_rev_bravo_chunk_8_f32,
    bravo_on_buf_chunk_8_f32,
    f32,
    f32x8<S>,
    8,
    TILE_SIDE_F32
);
impl_bit_rev_bravo!(
    bit_rev_bravo_chunk_4_f64,
    bravo_on_buf_chunk_4_f64,
    f64,
    f64x4<S>,
    4,
    TILE_SIDE_F64
);
impl_bit_rev_bravo!(
    bit_rev_bravo_chunk_8_f64,
    bravo_on_buf_chunk_8_f64,
    f64,
    f64x8<S>,
    8,
    TILE_SIDE_F64
);

/// Performs in-place bit-reversal permutation using the CO-BRAVO algorithm.
///
/// # Arguments
/// * `data` - The slice to permute in-place. Length must be a power of 2 and >= LANES².
/// * `n` - The log₂ of the data length (i.e., data.len() == 2^n)
#[inline(always)] // required by fearless_simd
pub fn bit_rev_bravo_f32<S: Simd>(simd: S, data: &mut [f32], n: usize) {
    match <S::f32s>::N {
        4 => bit_rev_bravo_chunk_4_f32(simd, data, n), // SSE, NEON and fallback
        _ => bit_rev_bravo_chunk_8_f32(simd, data, n),
        // fearless_simd has no native support for AVX-512 yet
    }
}

/// Performs in-place bit-reversal permutation using the CO-BRAVO algorithm.
///
/// # Arguments
/// * `data` - The slice to permute in-place. Length must be a power of 2 and >= LANES².
/// * `n` - The log₂ of the data length (i.e., data.len() == 2^n)
#[inline(always)] // required by fearless_simd
pub fn bit_rev_bravo_f64<S: Simd>(simd: S, data: &mut [f64], n: usize) {
    match <S::f64s>::N {
        // despite exceeding the native vector width, it is profitable to use larger chunks
        // according to benchmarks on both Zen4 and Apple M4
        2 => bit_rev_bravo_chunk_4_f64(simd, data, n), // SSE, NEON and fallback
        _ => bit_rev_bravo_chunk_8_f64(simd, data, n),
        // fearless_simd has no native support for AVX-512 yet
    }
}

/// Scalar bit-reversal for small arrays
fn scalar_bit_reversal<T: Default + Copy + Clone>(data: &mut [T], n: usize) {
    let big_n = data.len();

    for i in 0..big_n {
        let j = reverse_bits_scalar(i, n as u32);
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of `x`
const fn reverse_bits_scalar(x: usize, bits: u32) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS - bits)
}

#[cfg(test)]
mod tests {
    use fearless_simd::{dispatch, Level};

    use super::*;

    /// Top down bit reverse interleaving. This is a very simple and well known approach that we
    /// only use for testing due to its lackluster performance.
    fn top_down_bit_reverse_permutation<T: Copy + Clone>(x: &[T]) -> Vec<T> {
        if x.len() == 1 {
            return x.to_vec();
        }
        let mut y = Vec::with_capacity(x.len());
        let mut evens = Vec::with_capacity(x.len() >> 1);
        let mut odds = Vec::with_capacity(x.len() >> 1);
        let mut i = 1;
        while i < x.len() {
            evens.push(x[i - 1]);
            odds.push(x[i]);
            i += 2;
        }
        y.extend_from_slice(&top_down_bit_reverse_permutation(&evens));
        y.extend_from_slice(&top_down_bit_reverse_permutation(&odds));
        y
    }

    #[test]
    fn test_bravo_bit_reversal_f64() {
        for n in 2..24 {
            let big_n = 1 << n; // 2.pow(n)
            let mut actual_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let mut actual_im: Vec<f64> = (0..big_n).map(f64::from).collect();
            let simd_level = Level::new();
            dispatch!(simd_level, simd => bit_rev_bravo_f64(simd, &mut actual_re, n));
            dispatch!(simd_level, simd => bit_rev_bravo_f64(simd, &mut actual_im, n));
            let input_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_re = top_down_bit_reverse_permutation(&input_re);
            assert_eq!(actual_re, expected_re);
            let input_im: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_im = top_down_bit_reverse_permutation(&input_im);
            assert_eq!(actual_im, expected_im);
        }
    }

    #[test]
    fn test_bravo_bit_reversal_f32() {
        for n in 2..24 {
            let big_n = 1 << n; // 2.pow(n)
            let mut actual_re: Vec<f32> = (0..big_n).map(|i| i as f32).collect();
            let mut actual_im: Vec<f32> = (0..big_n).map(|i| i as f32).collect();
            let simd_level = Level::new();
            dispatch!(simd_level, simd => bit_rev_bravo_f32(simd, &mut actual_re, n));
            dispatch!(simd_level, simd => bit_rev_bravo_f32(simd, &mut actual_im, n));
            let input_re: Vec<f32> = (0..big_n).map(|i| i as f32).collect();
            let expected_re = top_down_bit_reverse_permutation(&input_re);
            assert_eq!(actual_re, expected_re);
            let input_im: Vec<f32> = (0..big_n).map(|i| i as f32).collect();
            let expected_im = top_down_bit_reverse_permutation(&input_im);
            assert_eq!(actual_im, expected_im);
        }
    }
}
