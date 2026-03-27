//! Functions for complex numbers such as interleave/deinterleave
//!
//! They are not part of the public API because the module they're in is private.
//! They can be accessed with `--cfg phastft_bench` for benchmarking only.

use bytemuck::cast_slice;
use num_complex::Complex;

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
fn combine_re_im_into_parallel<T: Copy + Send + Sync>(reals: &[T], imags: &[T], output: &mut [T]) {
    const CHUNK_SIZE: usize = 8;
    use rayon::prelude::*;
    output
        .as_chunks_mut::<CHUNK_SIZE>()
        .0
        .par_iter_mut()
        .zip(reals.as_chunks::<{ CHUNK_SIZE / 2 }>().0.par_iter())
        .zip(imags.as_chunks::<{ CHUNK_SIZE / 2 }>().0.par_iter())
        .for_each(|((out, re), im)| {
            out[0] = re[0];
            out[1] = im[0];
            out[2] = re[1];
            out[3] = im[1];
            out[4] = re[2];
            out[5] = im[2];
            out[6] = re[3];
            out[7] = im[3];
        });
}

/// Combines parallel reals and imags arrays into an interleaved output.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()` or `output.len() != reals.len() * 2`.
pub fn combine_re_im_into<T: Copy + Send + Sync>(reals: &[T], imags: &[T], output: &mut [T]) {
    assert_eq!(reals.len(), imags.len());
    assert_eq!(output.len(), reals.len() * 2);

    #[cfg(not(feature = "parallel"))]
    combine_re_im_into_sequential(reals, imags, output);
    #[cfg(feature = "parallel")]
    combine_re_im_into_parallel(reals, imags, output);
}

fn combine_re_im_into_sequential<T: Copy>(reals: &[T], imags: &[T], output: &mut [T]) {
    for ((out, z_re), z_im) in output
        .chunks_exact_mut(2)
        .zip(reals.iter())
        .zip(imags.iter())
    {
        out[0] = *z_re;
        out[1] = *z_im;
    }
}

#[cfg(test)]
mod tests {
    use rand::distr::StandardUniform;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

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
        let mut rng = SmallRng::from_os_rng();
        let complex_vec: Vec<f32> = (&mut rng)
            .sample_iter(StandardUniform)
            .take(16384)
            .collect();

        let (reals, imags) = deinterleave(&complex_vec);

        let mut recombined = vec![0.0f32; complex_vec.len()];
        combine_re_im_into(&reals, &imags, &mut recombined);

        assert_eq!(complex_vec, recombined);
    }
}
