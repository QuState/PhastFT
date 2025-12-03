//! Common FFT Kernels
//!
//! FFT kernels shared by dif and dit versions of the FFT.
//!
use num_traits::Float;

/// Simple butterfly for chunk_size == 2
/// This is the same for both, the DIF and DIT algorithms
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
))]
pub fn fft_chunk_2<T: Float>(reals: &mut [T], imags: &mut [T]) {
    reals
        .chunks_exact_mut(2)
        .zip(imags.chunks_exact_mut(2))
        .for_each(|(reals_chunk, imags_chunk)| {
            let z0_re = reals_chunk[0];
            let z0_im = imags_chunk[0];
            let z1_re = reals_chunk[1];
            let z1_im = imags_chunk[1];

            reals_chunk[0] = z0_re + z1_re;
            imags_chunk[0] = z0_im + z1_im;
            reals_chunk[1] = z0_re - z1_re;
            imags_chunk[1] = z0_im - z1_im;
        });
}
