//! Functions for complex numbers such as interleave/deinterleave
//!
//! They are not part of the public API because the module they're in is private.
//! They can be accessed with `--features bench-internals` for benchmarking only.
use bytemuck::cast_slice;
use num_complex::Complex;

/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any *even* length
pub fn deinterleave<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    // Despite relying on autovectorization, this is the fastest approach
    // because we don't need to initialize the output Vecs.
    // The Vecs are also allocated up front without intermediate reallocations.
    // This is faster than any implementation we could write without `unsafe`.
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}

/// Splits a slice of [`Complex<f64>`] into separate real and imaginary vectors.
///
/// Inverse of [`combine_re_im`].
pub fn deinterleave_complex64(signal: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
    let complex_t: &[f64] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Splits a slice of [`Complex<f32>`] into separate real and imaginary vectors.
///
/// Inverse of [`combine_re_im`].
pub fn deinterleave_complex32(signal: &[Complex<f32>]) -> (Vec<f32>, Vec<f32>) {
    let complex_t: &[f32] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Utility function to combine separate vectors of real and imaginary components
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn combine_re_im<T>(reals: &[T], imags: &[T]) -> Vec<Complex<T>>
where
    T: Copy,
{
    assert_eq!(reals.len(), imags.len());

    reals
        .iter()
        .zip(imags.iter())
        .map(|(z_re, z_im)| Complex::new(*z_re, *z_im))
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::distr::StandardUniform;
    use rand::rngs::SmallRng;
    use rand::RngExt;

    use super::*;

    fn gen_test_vec(len: usize) -> Vec<usize> {
        (0..len).collect()
    }

    /// Slow but obviously correct implementation of deinterleaving,
    /// to be used in tests
    fn deinterleave_naive<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
        let half_len = input.len() / 2;
        let mut evens = Vec::with_capacity(half_len);
        let mut odds = Vec::with_capacity(half_len);

        for i in 0..half_len {
            evens.push(input[2 * i]);
            odds.push(input[2 * i + 1])
        }
        // assert_eq!(evens.len() + odds.len(), input.len());
        (evens, odds)
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

    #[test]
    fn test_separate_and_combine_re_im() {
        let mut rng = rand::make_rng::<SmallRng>();
        let complex_vec: Vec<f32> = (&mut rng)
            .sample_iter(StandardUniform)
            .take(16384)
            .collect();

        let (reals, imags) = deinterleave(&complex_vec);

        let recombined_vec = combine_re_im(&reals, &imags);

        let recombined_flat: &[f32] = cast_slice(recombined_vec.as_slice());
        assert_eq!(complex_vec, recombined_flat);
    }
}
