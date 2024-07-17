//! Utility functions such as interleave/deinterleave

#[cfg(feature = "complex-nums")]
use num_complex::Complex;

#[cfg(feature = "complex-nums")]
use num_traits::Float;

#[cfg(feature = "complex-nums")]
use bytemuck::cast_slice;

use std::simd::{prelude::Simd, simd_swizzle, SimdElement};

// We don't multiversion for AVX-512 here and keep the chunk size below AVX-512
// because we haven't seen any gains from it in benchmarks.
// This might be due to us running benchmarks on Zen4 which implements AVX-512
// on top of 256-bit wide execution units.
//
// If benchmarks on "real" AVX-512 show improvement on AVX-512
// without degrading AVX2 machines due to larger chunk size,
// the AVX-512 specialization should be re-enabled.
#[multiversion::multiversion(
    targets(
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length
pub(crate) fn deinterleave<T: Copy + Default + SimdElement>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;
    const DOUBLE_CHUNK: usize = CHUNK_SIZE * 2;

    let out_len = input.len() / 2;
    // We've benchmarked, and it turns out that this approach with zeroed memory
    // is faster than using uninit memory and bumping the length once in a while!
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(DOUBLE_CHUNK)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            let in_simd: Simd<T, DOUBLE_CHUNK> = Simd::from_array(in_chunk.try_into().unwrap());
            // This generates *slightly* faster code than just assigning values by index.
            // You'd think simd::deinterleave would be appropriate, but it does something different!
            let result = simd_swizzle!(in_simd, [0, 2, 4, 6, 1, 3, 5, 7]);
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    // Process the remainder, too small for the vectorized loop
    let input_rem = input.chunks_exact(DOUBLE_CHUNK).remainder();
    let odds_rem = out_odd.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let evens_rem = out_even.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    input_rem
        .chunks_exact(2)
        .zip(odds_rem.iter_mut())
        .zip(evens_rem.iter_mut())
        .for_each(|((inp, odd), even)| {
            *odd = inp[0];
            *even = inp[1];
        });

    (out_odd, out_even)
}

/// Utility function to separate a slice of [`Complex64``]
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
#[cfg(feature = "complex-nums")]
pub(crate) fn deinterleave_complex64(signal: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
    let complex_t: &[f64] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Utility function to separate a slice of [`Complex32``]
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
#[cfg(feature = "complex-nums")]
pub(crate) fn deinterleave_complex32(signal: &[Complex<f32>]) -> (Vec<f32>, Vec<f32>) {
    let complex_t: &[f32] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Utility function to combine separate vectors of real and imaginary components
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
#[cfg(feature = "complex-nums")]
pub(crate) fn combine_re_im<T: Float>(reals: &[T], imags: &[T]) -> Vec<Complex<T>> {
    assert_eq!(reals.len(), imags.len());

    reals
        .iter()
        .zip(imags.iter())
        .map(|(z_re, z_im)| Complex::new(*z_re, *z_im))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_test_vec(len: usize) -> Vec<usize> {
        (0..len).collect()
    }

    /// Slow but obviously correct implementation of deinterleaving,
    /// to be used in tests
    fn deinterleave_naive<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
        input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
    }

    #[test]
    fn deinterleaving_correctness() {
        for len in [0, 1, 2, 3, 15, 16, 17, 127, 128, 129, 130, 135, 100500] {
            let input = gen_test_vec(len);
            let (naive_a, naive_b) = deinterleave_naive(&input);
            let (opt_a, opt_b) = deinterleave(&input);
            assert_eq!(naive_a, opt_a);
            assert_eq!(naive_b, opt_b);
        }
    }

    #[cfg(feature = "complex-nums")]
    #[test]
    fn test_separate_and_combine_re_im() {
        let complex_vec: Vec<_> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];

        let (reals, imags) = deinterleave_complex64(&complex_vec);

        let recombined_vec = combine_re_im(&reals, &imags);

        assert_eq!(complex_vec, recombined_vec);
    }
}
