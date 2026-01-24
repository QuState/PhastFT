/// CO-BRAVO: Cache-Optimal Bit-Reversal Algorithm using Vector permute Operations
///
/// This implements the algorithm from "Optimal Bit-Reversal Using Vector Permutations"
/// by Lokhmotov and Mycroft (SPAA'07).
///
/// The algorithm uses vector interleaving operations to perform bit-reversal permutation.
/// For N = 2^n elements with W-element vectors, the algorithm performs log₂(N) rounds
/// of in-place interleave operations on pairs of vectors.
///
/// The initial implementation was translated from mathematical notation in the paper
/// to Rust by Claude 4.5 Opus.

use fearless_simd::prelude::*;
use fearless_simd::{f32x4, f32x8, f64x2, f64x4, Simd};

/// Macro to generate bit_rev_bravo implementations for concrete types.
/// Used instead of generics because `fearless_simd` doesn't let us be generic over the exact float type.
macro_rules! impl_bit_rev_bravo {
    ($fn_name:ident, $elem_ty:ty, $vec_ty:ty, $lanes:expr) => {
        #[inline(always)] // required by fearless_simd
        fn $fn_name<S: Simd>(simd: S, data: &mut [$elem_ty], n: usize) {
            type Chunk<S> = $vec_ty;
            const LANES: usize = $lanes; // Vector width W

            // as of Rust 1.93 we cannot use an associated constant for array lengths
            assert!(<Chunk<S>>::N == LANES);

            let big_n = 1usize << n;
            assert_eq!(data.len(), big_n, "Data length must be 2^n");

            // For very small arrays, fall back to scalar bit-reversal
            if big_n < LANES * LANES {
                scalar_bit_reversal(data, n);
                return;
            }

            let w = LANES;
            let log_w = w.ilog2() as usize; // = 2 for W=4

            // π = N / W² = number of equivalence classes
            let num_classes = big_n / (w * w);
            let class_bits = n - 2 * log_w;

            // Process each equivalence class.
            // For in-place operation, we handle class pairs that swap with each other.
            // We only process when class_idx <= class_idx_rev to avoid double processing.

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
                let mut chunks_a: [Chunk<S>; LANES] = [Chunk::splat(simd, Default::default()); LANES];
                for j in 0..w {
                    let base_idx = (class_idx + j * num_classes) * w;
                    chunks_a[j] = Chunk::from_slice(simd, &data[base_idx..base_idx + w]);
                }

                // Perform interleave rounds for class A
                for round in 0..log_w {
                    let mut new_chunks: [Chunk<S>; LANES] = [Chunk::splat(simd, Default::default()); LANES];
                    let stride = 1 << round;

                    let mut pair_idx = 0;
                    let mut i = 0;
                    while i < w {
                        for offset in 0..stride {
                            let idx0 = i + offset;
                            let idx1 = i + offset + stride;
                            let vec0 = chunks_a[idx0];
                            let vec1 = chunks_a[idx1];
                            let lo = vec0.zip_low(vec1);
                            let hi = vec0.zip_high(vec1);
                            let base = (pair_idx % stride) + (pair_idx / stride) * stride * 2;
                            new_chunks[base] = lo;
                            new_chunks[base + stride] = hi;
                            pair_idx += 1;
                        }
                        i += stride * 2;
                    }

                    chunks_a = new_chunks;
                }

                if class_idx == class_idx_rev {
                    // Self-mapping class - just write back to same location
                    for j in 0..w {
                        let base_idx = (class_idx + j * num_classes) * w;
                        chunks_a[j].store_slice(&mut data[base_idx..base_idx + w]);
                    }
                } else {
                    // Swapping pair - load class B, process it, then swap both
                    let mut chunks_b: [Chunk<S>; LANES] = [Chunk::splat(simd, Default::default()); LANES];
                    for j in 0..w {
                        let base_idx = (class_idx_rev + j * num_classes) * w;
                        chunks_b[j] = Chunk::from_slice(simd, &data[base_idx..base_idx + w]);
                    }

                    // Perform interleave rounds for class B
                    for round in 0..log_w {
                        let mut new_chunks: [Chunk<S>; LANES] = [Chunk::splat(simd, Default::default()); LANES];
                        let stride = 1 << round;

                        let mut pair_idx = 0;
                        let mut i = 0;
                        while i < w {
                            for offset in 0..stride {
                                let idx0 = i + offset;
                                let idx1 = i + offset + stride;
                                let vec0 = chunks_b[idx0];
                                let vec1 = chunks_b[idx1];
                                let lo = vec0.zip_low(vec1);
                                let hi = vec0.zip_high(vec1);
                                let base = (pair_idx % stride) + (pair_idx / stride) * stride * 2;
                                new_chunks[base] = lo;
                                new_chunks[base + stride] = hi;
                                pair_idx += 1;
                            }
                            i += stride * 2;
                        }

                        chunks_b = new_chunks;
                    }

                    // Swap: write A's result to B's location and vice versa
                    for j in 0..w {
                        let base_idx_a = (class_idx + j * num_classes) * w;
                        let base_idx_b = (class_idx_rev + j * num_classes) * w;
                        chunks_a[j].store_slice(&mut data[base_idx_b..base_idx_b + w]);
                        chunks_b[j].store_slice(&mut data[base_idx_a..base_idx_a + w]);
                    }
                }
            }
        }
    };
}

// Generate concrete implementations for f32 and f64
impl_bit_rev_bravo!(bit_rev_bravo_chunk_4_f32, f32, f32x4<S>, 4);
impl_bit_rev_bravo!(bit_rev_bravo_chunk_8_f32, f32, f32x8<S>, 8);
impl_bit_rev_bravo!(bit_rev_bravo_chunk_2_f64, f64, f64x2<S>, 2);
impl_bit_rev_bravo!(bit_rev_bravo_chunk_4_f64, f64, f64x4<S>, 4);

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
pub fn bit_rev_bravo_f64<S: Simd>(simd: S, data: &mut [f64], n: usize) {
    match <S::f64s>::N {
        2 => bit_rev_bravo_chunk_2_f64(simd, data, n), // SSE, NEON and fallback
        _ => bit_rev_bravo_chunk_4_f64(simd, data, n),
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
fn reverse_bits_scalar(x: usize, bits: u32) -> usize {
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
            let big_n = 1 << n;
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
            let big_n = 1 << n;
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
