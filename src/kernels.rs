use std::simd::f64x8;

pub type Float = f64;

pub(crate) fn fft_chunk_n_simd(
    reals: &mut [Float],
    imags: &mut [Float],
    twiddles_re: &[Float],
    twiddles_im: &[Float],
    dist: usize,
) {
    let chunk_size = dist << 1;
    assert!(chunk_size >= 16);

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .array_chunks_mut::<8>()
                .zip(reals_s1.array_chunks_mut::<8>())
                .zip(imags_s0.array_chunks_mut::<8>())
                .zip(imags_s1.array_chunks_mut::<8>())
                .zip(twiddles_re.array_chunks::<8>())
                .zip(twiddles_im.array_chunks::<8>())
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = f64x8::from_array(*re_s0);
                    let real_c1 = f64x8::from_array(*re_s1);
                    let imag_c0 = f64x8::from_array(*im_s0);
                    let imag_c1 = f64x8::from_array(*im_s1);

                    let tw_re = f64x8::from_array(*w_re);
                    let tw_im = f64x8::from_array(*w_im);

                    re_s0.copy_from_slice((real_c0 + real_c1).as_array());
                    im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array());
                    im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array());
                });
        });
}

// TODO(saveliy): parallelize
pub(crate) fn fft_chunk_n(
    reals: &mut [Float],
    imags: &mut [Float],
    twiddles_re: &[Float],
    twiddles_im: &[Float],
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
                    *re_s1 = v_re * w_re - v_im * w_im;
                    *im_s1 = v_re * w_im + v_im * w_re;
                });
        });
}

/// `chunk_size == 4`, so hard code twiddle factors
pub(crate) fn fft_chunk_4(reals: &mut [Float], imags: &mut [Float]) {
    const DIST: usize = 2;
    const CHUNK_SIZE: usize = DIST << 1;

    reals
        .array_chunks_mut::<CHUNK_SIZE>()
        .zip(imags.array_chunks_mut::<CHUNK_SIZE>())
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
pub(crate) fn fft_chunk_2(reals: &mut [Float], imags: &mut [Float]) {
    reals
        .array_chunks_mut::<2>()
        .zip(imags.array_chunks_mut::<2>())
        .for_each(
            |(reals_chunk @ &mut [z0_re, z1_re], imags_chunk @ &mut [z0_im, z1_im])| {
                reals_chunk[0] = z0_re + z1_re;
                imags_chunk[0] = z0_im + z1_im;
                reals_chunk[1] = z0_re - z1_re;
                imags_chunk[1] = z0_im - z1_im;
            },
        );
}
