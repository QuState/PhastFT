//! Common FFT Kernels
//!
//! FFT kernels shared by dif and dit versions of the FFT.
//!
use num_traits::Float;

/// Simple butterfly for chunk_size == 2
/// This is the same for both, the DIF and DIT algorithms
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
#[inline]
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

/// DIT butterfly for chunk_size == 2 (identical to DIF version)
#[inline]
pub fn fft_dit_chunk_2<T: Float>(reals: &mut [T], imags: &mut [T]) {
    fft_chunk_2(reals, imags);
}

/// DIF butterfly for chunk_size == 4 with hard-coded twiddle factors
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
#[inline]
pub fn fft_chunk_4<T: Float>(reals: &mut [T], imags: &mut [T]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST << 1;

    reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // First pair (W_4^0 = 1)
            let real_c0 = reals_s0[0];
            let real_c1 = reals_s1[0];
            let imag_c0 = imags_s0[0];
            let imag_c1 = imags_s1[0];

            reals_s0[0] = real_c0 + real_c1;
            imags_s0[0] = imag_c0 + imag_c1;
            reals_s1[0] = real_c0 - real_c1;
            imags_s1[0] = imag_c0 - imag_c1;

            // Second pair (W_4^1 = -i)
            let real_c0 = reals_s0[1];
            let real_c1 = reals_s1[1];
            let imag_c0 = imags_s0[1];
            let imag_c1 = imags_s1[1];

            reals_s0[1] = real_c0 + real_c1;
            imags_s0[1] = imag_c0 + imag_c1;
            // Multiply by -i: (a + bi) * (-i) = b - ai
            reals_s1[1] = imag_c0 - imag_c1;
            imags_s1[1] = -(real_c0 - real_c1);
        });
}

/// DIT butterfly for chunk_size == 4 with FMA optimization
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
#[inline]
pub fn fft_dit_chunk_4<T: Float>(reals: &mut [T], imags: &mut [T]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = T::from(2.0).unwrap();

    reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // First pair (W_4^0 = 1)
            let in0_re = reals_s0[0];
            let in1_re = reals_s1[0];
            let in0_im = imags_s0[0];
            let in1_im = imags_s1[0];

            reals_s0[0] = in0_re + in1_re;
            imags_s0[0] = in0_im + in1_im;
            // Use FMA: out1 = 2*in0 - out0
            reals_s1[0] = two * in0_re - reals_s0[0];
            imags_s1[0] = two * in0_im - imags_s0[0];

            // Second pair (W_4^1 = -i)
            let in0_re = reals_s0[1];
            let in1_re = reals_s1[1];
            let in0_im = imags_s0[1];
            let in1_im = imags_s1[1];

            // Apply twiddle W_4^1 = -i: (re + i*im) * (-i) = im - i*re
            reals_s0[1] = in0_re + in1_im;
            imags_s0[1] = in0_im - in1_re;
            // Use FMA: out1 = 2*in0 - out0
            reals_s1[1] = two * in0_re - reals_s0[1];
            imags_s1[1] = two * in0_im - imags_s0[1];
        });
}
