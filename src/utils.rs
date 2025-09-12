//! Utility functions such as interleave/deinterleave

#[cfg(feature = "complex-nums")]
use bytemuck::cast_slice;
#[cfg(feature = "complex-nums")]
use num_complex::Complex;
#[cfg(feature = "complex-nums")]
use num_traits::Float;

// Note: Since wide doesn't have simd_swizzle equivalent, we use an optimized scalar approach
// The compiler should still be able to vectorize this pattern when beneficial
#[multiversion::multiversion(
    targets(
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
    ))]
/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length
pub(crate) fn deinterleave<T: Copy + Default>(input: &[T]) -> (Vec<T>, Vec<T>) {
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
            // Manual deinterleaving that compiler can often vectorize
            odds[0] = in_chunk[0];
            odds[1] = in_chunk[2];
            odds[2] = in_chunk[4];
            odds[3] = in_chunk[6];
            evens[0] = in_chunk[1];
            evens[1] = in_chunk[3];
            evens[2] = in_chunk[5];
            evens[3] = in_chunk[7];
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
