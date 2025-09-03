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

/// DIT butterfly for chunk_size == 8 with FMA optimization
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
pub(crate) fn fft_dit_chunk_8<T: Float>(reals: &mut [T], imags: &mut [T]) {
    const DIST: usize = 4;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = T::from(2.0).unwrap();

    reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Hardcoded twiddle factor for W_8
            let sqrt2_2 = T::from(0.7071067811865476).unwrap(); // sqrt(2)/2

            // k=0: W_8^0 = 1
            let in0_re = reals_s0[0];
            let in1_re = reals_s1[0];
            let in0_im = imags_s0[0];
            let in1_im = imags_s1[0];

            reals_s0[0] = in0_re + in1_re;
            imags_s0[0] = in0_im + in1_im;
            reals_s1[0] = in0_re.mul_add(two, -reals_s0[0]);
            imags_s1[0] = in0_im.mul_add(two, -imags_s0[0]);

            // k=1: W_8^1 = sqrt(2)/2 - i*sqrt(2)/2
            let in0_re = reals_s0[1];
            let in1_re = reals_s1[1];
            let in0_im = imags_s0[1];
            let in1_im = imags_s1[1];

            // out0_re = in0_re + (in1_re * sqrt2_2 + in1_im * sqrt2_2)
            // out0_im = in0_im + (in1_im * sqrt2_2 - in1_re * sqrt2_2)
            reals_s0[1] = in1_im.mul_add(sqrt2_2, in1_re.mul_add(sqrt2_2, in0_re));
            imags_s0[1] = in1_im.mul_add(sqrt2_2, -(in1_re * sqrt2_2)) + in0_im;
            reals_s1[1] = in0_re.mul_add(two, -reals_s0[1]);
            imags_s1[1] = in0_im.mul_add(two, -imags_s0[1]);

            // k=2: W_8^2 = -i
            let in0_re = reals_s0[2];
            let in1_re = reals_s1[2];
            let in0_im = imags_s0[2];
            let in1_im = imags_s1[2];

            reals_s0[2] = in0_re + in1_im;
            imags_s0[2] = in0_im - in1_re;
            reals_s1[2] = in0_re.mul_add(two, -reals_s0[2]);
            imags_s1[2] = in0_im.mul_add(two, -imags_s0[2]);

            // k=3: W_8^3 = -sqrt(2)/2 - i*sqrt(2)/2
            let in0_re = reals_s0[3];
            let in1_re = reals_s1[3];
            let in0_im = imags_s0[3];
            let in1_im = imags_s1[3];

            // out0_re = in0_re + (-in1_re * sqrt2_2 + in1_im * sqrt2_2)
            // out0_im = in0_im + (-in1_im * sqrt2_2 - in1_re * sqrt2_2)
            reals_s0[3] = in1_im.mul_add(sqrt2_2, -(in1_re * sqrt2_2)) + in0_re;
            imags_s0[3] = in0_im - (in1_im.mul_add(sqrt2_2, in1_re * sqrt2_2));
            reals_s1[3] = in0_re.mul_add(two, -reals_s0[3]);
            imags_s1[3] = in0_im.mul_add(two, -imags_s0[3]);
        });
}

/// DIT butterfly for chunk_size == 16  
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
pub(crate) fn fft_dit_chunk_16<T: Float>(reals: &mut [T], imags: &mut [T]) {
    const DIST: usize = 8;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = T::from(2.0).unwrap();

    reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Hardcoded twiddle factors for W_16
            let cos_pi_8 = T::from(0.9238795325112867).unwrap(); // cos(π/8)
            let sin_pi_8 = T::from(0.3826834323650898).unwrap(); // sin(π/8)
            let sqrt2_2 = T::from(0.7071067811865476).unwrap(); // sqrt(2)/2

            // k=0: W_16^0 = 1
            let in0_re = reals_s0[0];
            let in1_re = reals_s1[0];
            let in0_im = imags_s0[0];
            let in1_im = imags_s1[0];

            reals_s0[0] = in0_re + in1_re;
            imags_s0[0] = in0_im + in1_im;
            reals_s1[0] = in0_re.mul_add(two, -reals_s0[0]);
            imags_s1[0] = in0_im.mul_add(two, -imags_s0[0]);

            // k=1: W_16^1 = cos(π/8) - i*sin(π/8)
            let in0_re = reals_s0[1];
            let in1_re = reals_s1[1];
            let in0_im = imags_s0[1];
            let in1_im = imags_s1[1];

            // Apply twiddle with FMA
            reals_s0[1] = in0_re + in1_re.mul_add(cos_pi_8, in1_im * sin_pi_8);
            imags_s0[1] = in0_im + in1_im.mul_add(cos_pi_8, -(in1_re * sin_pi_8));
            reals_s1[1] = in0_re.mul_add(two, -reals_s0[1]);
            imags_s1[1] = in0_im.mul_add(two, -imags_s0[1]);

            // k=2: W_16^2 = sqrt(2)/2 - i*sqrt(2)/2
            let in0_re = reals_s0[2];
            let in1_re = reals_s1[2];
            let in0_im = imags_s0[2];
            let in1_im = imags_s1[2];

            reals_s0[2] = in0_re + in1_re.mul_add(sqrt2_2, in1_im * sqrt2_2);
            imags_s0[2] = in0_im + in1_im.mul_add(sqrt2_2, -(in1_re * sqrt2_2));
            reals_s1[2] = in0_re.mul_add(two, -reals_s0[2]);
            imags_s1[2] = in0_im.mul_add(two, -imags_s0[2]);

            // k=3: W_16^3 = sin(π/8) - i*cos(π/8)
            let in0_re = reals_s0[3];
            let in1_re = reals_s1[3];
            let in0_im = imags_s0[3];
            let in1_im = imags_s1[3];

            reals_s0[3] = in0_re + in1_re.mul_add(sin_pi_8, in1_im * cos_pi_8);
            imags_s0[3] = in0_im + in1_im.mul_add(sin_pi_8, -(in1_re * cos_pi_8));
            reals_s1[3] = in0_re.mul_add(two, -reals_s0[3]);
            imags_s1[3] = in0_im.mul_add(two, -imags_s0[3]);

            // k=4: W_16^4 = -i
            let in0_re = reals_s0[4];
            let in1_re = reals_s1[4];
            let in0_im = imags_s0[4];
            let in1_im = imags_s1[4];

            reals_s0[4] = in0_re + in1_im;
            imags_s0[4] = in0_im - in1_re;
            reals_s1[4] = in0_re.mul_add(two, -reals_s0[4]);
            imags_s1[4] = in0_im.mul_add(two, -imags_s0[4]);

            // k=5: W_16^5 = -sin(π/8) - i*cos(π/8)
            let in0_re = reals_s0[5];
            let in1_re = reals_s1[5];
            let in0_im = imags_s0[5];
            let in1_im = imags_s1[5];

            reals_s0[5] = in0_re + in1_im.mul_add(cos_pi_8, -(in1_re * sin_pi_8));
            imags_s0[5] = in0_im - (in1_im.mul_add(sin_pi_8, in1_re * cos_pi_8));
            reals_s1[5] = in0_re.mul_add(two, -reals_s0[5]);
            imags_s1[5] = in0_im.mul_add(two, -imags_s0[5]);

            // k=6: W_16^6 = -sqrt(2)/2 - i*sqrt(2)/2
            let in0_re = reals_s0[6];
            let in1_re = reals_s1[6];
            let in0_im = imags_s0[6];
            let in1_im = imags_s1[6];

            reals_s0[6] = in0_re + in1_im.mul_add(sqrt2_2, -(in1_re * sqrt2_2));
            imags_s0[6] = in0_im - (in1_im.mul_add(sqrt2_2, in1_re * sqrt2_2));
            reals_s1[6] = in0_re.mul_add(two, -reals_s0[6]);
            imags_s1[6] = in0_im.mul_add(two, -imags_s0[6]);

            // k=7: W_16^7 = -cos(π/8) - i*sin(π/8)
            let in0_re = reals_s0[7];
            let in1_re = reals_s1[7];
            let in0_im = imags_s0[7];
            let in1_im = imags_s1[7];

            reals_s0[7] = in0_re + in1_im.mul_add(sin_pi_8, -(in1_re * cos_pi_8));
            imags_s0[7] = in0_im - (in1_im.mul_add(cos_pi_8, in1_re * sin_pi_8));
            reals_s1[7] = in0_re.mul_add(two, -reals_s0[7]);
            imags_s1[7] = in0_im.mul_add(two, -imags_s0[7]);
        });
}

/// DIT butterfly for chunk_size == 32
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
pub(crate) fn fft_dit_chunk_32<T: Float>(reals: &mut [T], imags: &mut [T]) {
    const DIST: usize = 16;
    const CHUNK_SIZE: usize = DIST << 1;

    let two = T::from(2.0).unwrap();

    reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Hardcoded twiddle factors for W_32
            let cos_pi_16 = T::from(0.9807852804032304).unwrap(); // cos(π/16)
            let sin_pi_16 = T::from(0.19509032201612825).unwrap(); // sin(π/16)
            let cos_pi_8 = T::from(0.9238795325112867).unwrap(); // cos(π/8)
            let sin_pi_8 = T::from(0.3826834323650898).unwrap(); // sin(π/8)
            let cos_3pi_16 = T::from(0.8314696123025452).unwrap(); // cos(3π/16)
            let sin_3pi_16 = T::from(0.5555702330196022).unwrap(); // sin(3π/16)
            let sqrt2_2 = T::from(0.7071067811865476).unwrap(); // sqrt(2)/2
            let cos_5pi_16 = T::from(0.5555702330196022).unwrap(); // cos(5π/16)
            let sin_5pi_16 = T::from(0.8314696123025452).unwrap(); // sin(5π/16)
            let cos_3pi_8 = T::from(0.3826834323650898).unwrap(); // cos(3π/8)
            let sin_3pi_8 = T::from(0.9238795325112867).unwrap(); // sin(3π/8)
            let cos_7pi_16 = T::from(0.19509032201612825).unwrap(); // cos(7π/16)
            let sin_7pi_16 = T::from(0.9807852804032304).unwrap(); // sin(7π/16)

            // Manually unroll all 16 butterflies
            // k=0: W_32^0 = 1
            let in0_re = reals_s0[0];
            let in1_re = reals_s1[0];
            let in0_im = imags_s0[0];
            let in1_im = imags_s1[0];

            reals_s0[0] = in0_re + in1_re;
            imags_s0[0] = in0_im + in1_im;
            reals_s1[0] = in0_re.mul_add(two, -reals_s0[0]);
            imags_s1[0] = in0_im.mul_add(two, -imags_s0[0]);

            // k=1: W_32^1 = cos(π/16) - i*sin(π/16)
            let in0_re = reals_s0[1];
            let in1_re = reals_s1[1];
            let in0_im = imags_s0[1];
            let in1_im = imags_s1[1];

            reals_s0[1] = in0_re + in1_re.mul_add(cos_pi_16, in1_im * sin_pi_16);
            imags_s0[1] = in0_im + in1_im.mul_add(cos_pi_16, -(in1_re * sin_pi_16));
            reals_s1[1] = in0_re.mul_add(two, -reals_s0[1]);
            imags_s1[1] = in0_im.mul_add(two, -imags_s0[1]);

            // k=2: W_32^2 = cos(π/8) - i*sin(π/8)
            let in0_re = reals_s0[2];
            let in1_re = reals_s1[2];
            let in0_im = imags_s0[2];
            let in1_im = imags_s1[2];

            reals_s0[2] = in0_re + in1_re.mul_add(cos_pi_8, in1_im * sin_pi_8);
            imags_s0[2] = in0_im + in1_im.mul_add(cos_pi_8, -(in1_re * sin_pi_8));
            reals_s1[2] = in0_re.mul_add(two, -reals_s0[2]);
            imags_s1[2] = in0_im.mul_add(two, -imags_s0[2]);

            // k=3: W_32^3 = cos(3π/16) - i*sin(3π/16)
            let in0_re = reals_s0[3];
            let in1_re = reals_s1[3];
            let in0_im = imags_s0[3];
            let in1_im = imags_s1[3];

            reals_s0[3] = in0_re + in1_re.mul_add(cos_3pi_16, in1_im * sin_3pi_16);
            imags_s0[3] = in0_im + in1_im.mul_add(cos_3pi_16, -(in1_re * sin_3pi_16));
            reals_s1[3] = in0_re.mul_add(two, -reals_s0[3]);
            imags_s1[3] = in0_im.mul_add(two, -imags_s0[3]);

            // k=4: W_32^4 = sqrt(2)/2 - i*sqrt(2)/2
            let in0_re = reals_s0[4];
            let in1_re = reals_s1[4];
            let in0_im = imags_s0[4];
            let in1_im = imags_s1[4];

            reals_s0[4] = in0_re + in1_re.mul_add(sqrt2_2, in1_im * sqrt2_2);
            imags_s0[4] = in0_im + in1_im.mul_add(sqrt2_2, -(in1_re * sqrt2_2));
            reals_s1[4] = in0_re.mul_add(two, -reals_s0[4]);
            imags_s1[4] = in0_im.mul_add(two, -imags_s0[4]);

            // k=5: W_32^5 = cos(5π/16) - i*sin(5π/16)
            let in0_re = reals_s0[5];
            let in1_re = reals_s1[5];
            let in0_im = imags_s0[5];
            let in1_im = imags_s1[5];

            reals_s0[5] = in0_re + in1_re.mul_add(cos_5pi_16, in1_im * sin_5pi_16);
            imags_s0[5] = in0_im + in1_im.mul_add(cos_5pi_16, -(in1_re * sin_5pi_16));
            reals_s1[5] = in0_re.mul_add(two, -reals_s0[5]);
            imags_s1[5] = in0_im.mul_add(two, -imags_s0[5]);

            // k=6: W_32^6 = cos(3π/8) - i*sin(3π/8)
            let in0_re = reals_s0[6];
            let in1_re = reals_s1[6];
            let in0_im = imags_s0[6];
            let in1_im = imags_s1[6];

            reals_s0[6] = in0_re + in1_re.mul_add(cos_3pi_8, in1_im * sin_3pi_8);
            imags_s0[6] = in0_im + in1_im.mul_add(cos_3pi_8, -(in1_re * sin_3pi_8));
            reals_s1[6] = in0_re.mul_add(two, -reals_s0[6]);
            imags_s1[6] = in0_im.mul_add(two, -imags_s0[6]);

            // k=7: W_32^7 = cos(7π/16) - i*sin(7π/16)
            let in0_re = reals_s0[7];
            let in1_re = reals_s1[7];
            let in0_im = imags_s0[7];
            let in1_im = imags_s1[7];

            reals_s0[7] = in0_re + in1_re.mul_add(cos_7pi_16, in1_im * sin_7pi_16);
            imags_s0[7] = in0_im + in1_im.mul_add(cos_7pi_16, -(in1_re * sin_7pi_16));
            reals_s1[7] = in0_re.mul_add(two, -reals_s0[7]);
            imags_s1[7] = in0_im.mul_add(two, -imags_s0[7]);

            // k=8: W_32^8 = -i
            let in0_re = reals_s0[8];
            let in1_re = reals_s1[8];
            let in0_im = imags_s0[8];
            let in1_im = imags_s1[8];

            reals_s0[8] = in0_re + in1_im;
            imags_s0[8] = in0_im - in1_re;
            reals_s1[8] = in0_re.mul_add(two, -reals_s0[8]);
            imags_s1[8] = in0_im.mul_add(two, -imags_s0[8]);

            // k=9: W_32^9 = -cos(7π/16) - i*sin(7π/16)
            let in0_re = reals_s0[9];
            let in1_re = reals_s1[9];
            let in0_im = imags_s0[9];
            let in1_im = imags_s1[9];

            reals_s0[9] = in0_re + in1_im.mul_add(sin_7pi_16, -(in1_re * cos_7pi_16));
            imags_s0[9] = in0_im - (in1_im.mul_add(cos_7pi_16, in1_re * sin_7pi_16));
            reals_s1[9] = in0_re.mul_add(two, -reals_s0[9]);
            imags_s1[9] = in0_im.mul_add(two, -imags_s0[9]);

            // k=10: W_32^10 = -cos(3π/8) - i*sin(3π/8)
            let in0_re = reals_s0[10];
            let in1_re = reals_s1[10];
            let in0_im = imags_s0[10];
            let in1_im = imags_s1[10];

            reals_s0[10] = in0_re + in1_im.mul_add(sin_3pi_8, -(in1_re * cos_3pi_8));
            imags_s0[10] = in0_im - (in1_im.mul_add(cos_3pi_8, in1_re * sin_3pi_8));
            reals_s1[10] = in0_re.mul_add(two, -reals_s0[10]);
            imags_s1[10] = in0_im.mul_add(two, -imags_s0[10]);

            // k=11: W_32^11 = -cos(5π/16) - i*sin(5π/16)
            let in0_re = reals_s0[11];
            let in1_re = reals_s1[11];
            let in0_im = imags_s0[11];
            let in1_im = imags_s1[11];

            reals_s0[11] = in0_re + in1_im.mul_add(sin_5pi_16, -(in1_re * cos_5pi_16));
            imags_s0[11] = in0_im - (in1_im.mul_add(cos_5pi_16, in1_re * sin_5pi_16));
            reals_s1[11] = in0_re.mul_add(two, -reals_s0[11]);
            imags_s1[11] = in0_im.mul_add(two, -imags_s0[11]);

            // k=12: W_32^12 = -sqrt(2)/2 - i*sqrt(2)/2
            let in0_re = reals_s0[12];
            let in1_re = reals_s1[12];
            let in0_im = imags_s0[12];
            let in1_im = imags_s1[12];

            reals_s0[12] = in0_re + in1_im.mul_add(sqrt2_2, -(in1_re * sqrt2_2));
            imags_s0[12] = in0_im - (in1_im.mul_add(sqrt2_2, in1_re * sqrt2_2));
            reals_s1[12] = in0_re.mul_add(two, -reals_s0[12]);
            imags_s1[12] = in0_im.mul_add(two, -imags_s0[12]);

            // k=13: W_32^13 = -cos(3π/16) - i*sin(3π/16)
            let in0_re = reals_s0[13];
            let in1_re = reals_s1[13];
            let in0_im = imags_s0[13];
            let in1_im = imags_s1[13];

            reals_s0[13] = in0_re + in1_im.mul_add(sin_3pi_16, -(in1_re * cos_3pi_16));
            imags_s0[13] = in0_im - (in1_im.mul_add(cos_3pi_16, in1_re * sin_3pi_16));
            reals_s1[13] = in0_re.mul_add(two, -reals_s0[13]);
            imags_s1[13] = in0_im.mul_add(two, -imags_s0[13]);

            // k=14: W_32^14 = -cos(π/8) - i*sin(π/8)
            let in0_re = reals_s0[14];
            let in1_re = reals_s1[14];
            let in0_im = imags_s0[14];
            let in1_im = imags_s1[14];

            reals_s0[14] = in0_re + in1_im.mul_add(sin_pi_8, -(in1_re * cos_pi_8));
            imags_s0[14] = in0_im - (in1_im.mul_add(cos_pi_8, in1_re * sin_pi_8));
            reals_s1[14] = in0_re.mul_add(two, -reals_s0[14]);
            imags_s1[14] = in0_im.mul_add(two, -imags_s0[14]);

            // k=15: W_32^15 = -cos(π/16) - i*sin(π/16)
            let in0_re = reals_s0[15];
            let in1_re = reals_s1[15];
            let in0_im = imags_s0[15];
            let in1_im = imags_s1[15];

            reals_s0[15] = in0_re + in1_im.mul_add(sin_pi_16, -(in1_re * cos_pi_16));
            imags_s0[15] = in0_im - (in1_im.mul_add(cos_pi_16, in1_re * sin_pi_16));
            reals_s1[15] = in0_re.mul_add(two, -reals_s0[15]);
            imags_s1[15] = in0_im.mul_add(two, -imags_s0[15]);
        });
}

/// Generic DIT butterfly kernel with twiddle factors
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
pub(crate) fn fft_dit_chunk_n<T: Float>(
    reals: &mut [T],
    imags: &mut [T],
    twiddles_re: &[T],
    twiddles_im: &[T],
    dist: usize,
) {
    let chunk_size = dist << 1;
    let two = T::from(2.0).unwrap();

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
                    let in0_re = *re_s0;
                    let in1_re = *re_s1;
                    let in0_im = *im_s0;
                    let in1_im = *im_s1;

                    // Apply twiddle factor using optimized FMA
                    // Compute twiddle multiplication with minimal operations
                    let twiddle_re = in1_re.mul_add(*w_re, -(in1_im * *w_im));
                    let twiddle_im = in1_im.mul_add(*w_re, in1_re * *w_im);

                    // Compute outputs
                    let out0_re = in0_re + twiddle_re;
                    let out0_im = in0_im + twiddle_im;

                    *re_s0 = out0_re;
                    *im_s0 = out0_im;

                    // Use FMA for out1 = 2*in0 - out0
                    *re_s1 = two.mul_add(in0_re, -out0_re);
                    *im_s1 = two.mul_add(in0_im, -out0_im);
                });
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
