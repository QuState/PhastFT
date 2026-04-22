//! Functions for complex numbers such as interleave/deinterleave
//!
//! They are not part of the public API because the module they're in is private.
//! They can be accessed with `--features bench-internals` for benchmarking only.

use bytemuck::cast_slice;
use num_complex::Complex;
use num_traits::Float;

#[cfg(feature = "parallel")]
fn deinterleave_parallel<T: Copy + Send + Sync>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 8;
    let out_vec_len = input.len() * 2 / CHUNK_SIZE;
    let (mut re, mut im) = (
        Vec::with_capacity(out_vec_len),
        Vec::with_capacity(out_vec_len),
    );
    use rayon::prelude::*;
    input
        .as_chunks::<CHUNK_SIZE>()
        .0
        .par_iter()
        .map(|c| ([c[0], c[2], c[4], c[6]], [c[1], c[3], c[5], c[7]]))
        .unzip_into_vecs(&mut re, &mut im);
    (re.into_flattened(), im.into_flattened())
}

/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length
pub fn deinterleave<T: Copy + Send + Sync>(input: &[T]) -> (Vec<T>, Vec<T>) {
    #[cfg(not(feature = "parallel"))]
    return deinterleave_sequential(input);
    #[cfg(feature = "parallel")]
    return deinterleave_parallel(input);
}

fn deinterleave_sequential<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    // Despite relying on autovectorization, this is the fastest approach
    // because we don't need to initialize the output Vecs.
    // The Vecs are also allocated up front without intermediate reallocations.
    // This is faster than any implementation we could write without `unsafe`.
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}

/// Utility function to separate a slice of [`Complex64``]
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn deinterleave_complex64(signal: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
    let complex_t: &[f64] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Utility function to separate a slice of [`Complex32``]
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn deinterleave_complex32(signal: &[Complex<f32>]) -> (Vec<f32>, Vec<f32>) {
    let complex_t: &[f32] = cast_slice(signal);
    deinterleave(complex_t)
}

#[cfg(feature = "parallel")]
fn combine_re_im_parallel<T: Float + Send + Sync>(reals: &[T], imags: &[T]) -> Vec<Complex<T>> {
    assert_eq!(reals.len(), imags.len());

    const CHUNK_SIZE: usize = 4;
    use rayon::prelude::*;
    let mut output: Vec<[Complex<T>; CHUNK_SIZE]> = Vec::with_capacity(reals.len() / CHUNK_SIZE);
    reals
        .as_chunks::<CHUNK_SIZE>()
        .0
        .par_iter()
        .zip(imags.as_chunks::<CHUNK_SIZE>().0.par_iter())
        .map(|(re, im)| {
            [
                Complex::new(re[0], im[0]),
                Complex::new(re[1], im[1]),
                Complex::new(re[2], im[2]),
                Complex::new(re[3], im[3]),
            ]
        })
        .collect_into_vec(&mut output);
    output.into_flattened()
}

/// Utility function to combine separate vectors of real and imaginary components
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn combine_re_im<T: Float + Send + Sync>(reals: &[T], imags: &[T]) -> Vec<Complex<T>> {
    #[cfg(not(feature = "parallel"))]
    return combine_re_im_sequential(reals, imags);
    #[cfg(feature = "parallel")]
    return combine_re_im_parallel(reals, imags);
}

fn combine_re_im_sequential<T: Float>(reals: &[T], imags: &[T]) -> Vec<Complex<T>> {
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
