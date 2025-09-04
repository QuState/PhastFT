use std::simd::{f32x16, f64x8, StdFloat};

use num_traits::Float;

macro_rules! fft_butterfly_n_simd {
    ($func_name:ident, $precision:ty, $lanes:literal, $simd_vector:ty) => {
        #[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                                     "x86_64+avx2+fma", // x86_64-v3
                                                     "x86_64+sse4.2", // x86_64-v2
                                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                                     "x86+avx2+fma",
                                                     "x86+sse4.2",
                                                     "x86+sse2",
                                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
                ))]
        #[inline]
        pub fn $func_name(
            reals: &mut [$precision],
            imags: &mut [$precision],
            twiddles_re: &[$precision],
            twiddles_im: &[$precision],
            dist: usize,
        ) {
            let chunk_size = dist << 1;
            assert!(chunk_size >= $lanes * 2);
            reals
                .chunks_exact_mut(chunk_size)
                .zip(imags.chunks_exact_mut(chunk_size))
                .for_each(|(reals_chunk, imags_chunk)| {
                    let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
                    let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

                    reals_s0
                        .chunks_exact_mut($lanes)
                        .zip(reals_s1.chunks_exact_mut($lanes))
                        .zip(imags_s0.chunks_exact_mut($lanes))
                        .zip(imags_s1.chunks_exact_mut($lanes))
                        .zip(twiddles_re.chunks_exact($lanes))
                        .zip(twiddles_im.chunks_exact($lanes))
                        .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                            let real_c0 = <$simd_vector>::from_slice(re_s0);
                            let real_c1 = <$simd_vector>::from_slice(re_s1);
                            let imag_c0 = <$simd_vector>::from_slice(im_s0);
                            let imag_c1 = <$simd_vector>::from_slice(im_s1);

                            let tw_re = <$simd_vector>::from_slice(w_re);
                            let tw_im = <$simd_vector>::from_slice(w_im);

                            re_s0.copy_from_slice((real_c0 + real_c1).as_array());
                            im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
                            let v_re = real_c0 - real_c1;
                            let v_im = imag_c0 - imag_c1;
                            re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array());
                            im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array());
                        });
                });
        }
    };
}

fft_butterfly_n_simd!(fft_64_chunk_n_simd, f64, 8, f64x8);
fft_butterfly_n_simd!(fft_32_chunk_n_simd, f32, 16, f32x16);

#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
))]
#[inline]
pub(crate) fn fft_chunk_n<T: Float>(
    reals: &mut [T],
    imags: &mut [T],
    twiddles_re: &[T],
    twiddles_im: &[T],
    dist: usize,
) {
    let chunk_size = dist << 1;

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .iter_mut()
                .zip(reals_s1.iter_mut())
                .zip(imags_s0.iter_mut())
                .zip(imags_s1.iter_mut())
                .zip(twiddles_re.iter())
                .zip(twiddles_im.iter())
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = *re_s0;
                    let real_c1 = *re_s1;
                    let imag_c0 = *im_s0;
                    let imag_c1 = *im_s1;

                    *re_s0 = real_c0 + real_c1;
                    *im_s0 = imag_c0 + imag_c1;
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    *re_s1 = v_re * *w_re - v_im * *w_im;
                    *im_s1 = v_re * *w_im + v_im * *w_re;
                });
        });
}

/// `chunk_size == 4`, so hard code twiddle factors
#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
))]
#[inline]
pub(crate) fn fft_chunk_4<T: Float>(reals: &mut [T], imags: &mut [T]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST << 1;

    reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            let real_c0 = reals_s0[0];
            let real_c1 = reals_s1[0];
            let imag_c0 = imags_s0[0];
            let imag_c1 = imags_s1[0];

            reals_s0[0] = real_c0 + real_c1;
            imags_s0[0] = imag_c0 + imag_c1;
            reals_s1[0] = real_c0 - real_c1;
            imags_s1[0] = imag_c0 - imag_c1;

            let real_c0 = reals_s0[1];
            let real_c1 = reals_s1[1];
            let imag_c0 = imags_s0[1];
            let imag_c1 = imags_s1[1];

            reals_s0[1] = real_c0 + real_c1;
            imags_s0[1] = imag_c0 + imag_c1;
            reals_s1[1] = imag_c0 - imag_c1;
            imags_s1[1] = -(real_c0 - real_c1);
        });
}

/// `chunk_size == 2`, so skip phase
#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
))]
#[inline]
pub(crate) fn fft_chunk_2<T: Float>(reals: &mut [T], imags: &mut [T]) {
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

#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
))]
#[inline]
pub(crate) fn fft_dit_chunk_2<T: Float>(reals: &mut [T], imags: &mut [T]) {
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

/// DIT butterfly for chunk_size == 4 with FMA optimization
#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
))]
#[inline]
pub(crate) fn fft_dit_chunk_4<T: Float>(reals: &mut [T], imags: &mut [T]) {
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
            // out0_re = in0_re + in1_im
            // out0_im = in0_im - in1_re
            reals_s0[1] = in0_re + in1_im;
            imags_s0[1] = in0_im - in1_re;
            // Use FMA: out1 = 2*in0 - out0
            reals_s1[1] = two * in0_re - reals_s0[1];
            imags_s1[1] = two * in0_im - imags_s0[1];
        });
}

macro_rules! fft_dit_butterfly_n_simd {
    ($func_name:ident, $precision:ty, $lanes:literal, $simd_vector:ty) => {
        #[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                                     "x86_64+avx2+fma", // x86_64-v3
                                                     "x86_64+sse4.2", // x86_64-v2
                                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                                     "x86+avx2+fma",
                                                     "x86+sse4.2",
                                                     "x86+sse2",
                                                     "aarch64+neon", // ARM64 with NEON (Apple Silicon M1/M2)
                ))]
        #[inline]
        pub fn $func_name(
            reals: &mut [$precision],
            imags: &mut [$precision],
            twiddles_re: &[$precision],
            twiddles_im: &[$precision],
            dist: usize,
        ) {
            let chunk_size = dist << 1;
            let two = <$simd_vector>::splat(2.0 as $precision);
            assert!(chunk_size >= $lanes * 2);

            reals
                .chunks_exact_mut(chunk_size)
                .zip(imags.chunks_exact_mut(chunk_size))
                .for_each(|(reals_chunk, imags_chunk)| {
                    let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
                    let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

                    reals_s0
                        .chunks_exact_mut($lanes)
                        .zip(reals_s1.chunks_exact_mut($lanes))
                        .zip(imags_s0.chunks_exact_mut($lanes))
                        .zip(imags_s1.chunks_exact_mut($lanes))
                        .zip(twiddles_re.chunks_exact($lanes))
                        .zip(twiddles_im.chunks_exact($lanes))
                        .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                            let in0_re = <$simd_vector>::from_slice(re_s0);
                            let in1_re = <$simd_vector>::from_slice(re_s1);
                            let in0_im = <$simd_vector>::from_slice(im_s0);
                            let in1_im = <$simd_vector>::from_slice(im_s1);

                            let tw_re = <$simd_vector>::from_slice(w_re);
                            let tw_im = <$simd_vector>::from_slice(w_im);

                            // Apply twiddle factors with explicit FMA operations
                            // out0_re = in0_re + (tw_re * in1_re - tw_im * in1_im)
                            // out0_im = in0_im + (tw_re * in1_im + tw_im * in1_re)
                            let twiddle_re_part = tw_re.mul_add(in1_re, -(tw_im * in1_im));
                            let twiddle_im_part = tw_re.mul_add(in1_im, tw_im * in1_re);

                            let out0_re = in0_re + twiddle_re_part;
                            let out0_im = in0_im + twiddle_im_part;

                            // Use FMA for out1 = 2*in0 - out0
                            let out1_re = two.mul_add(in0_re, -out0_re);
                            let out1_im = two.mul_add(in0_im, -out0_im);

                            re_s0.copy_from_slice(out0_re.as_array());
                            im_s0.copy_from_slice(out0_im.as_array());
                            re_s1.copy_from_slice(out1_re.as_array());
                            im_s1.copy_from_slice(out1_im.as_array());
                        });
                });
        }
    };
}

fft_dit_butterfly_n_simd!(fft_dit_64_chunk_n_simd, f64, 8, f64x8);
fft_dit_butterfly_n_simd!(fft_dit_32_chunk_n_simd, f32, 16, f32x16);

// Include the optimized SIMD implementations
include!("kernels_dit_simd.rs");
