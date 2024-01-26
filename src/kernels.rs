use std::simd::f64x8;

use spinoza::core::State;
use spinoza::math::Float;

pub(crate) fn fft_chunk_n_simd(
    state: &mut State,
    twiddles_re: &[Float],
    twiddles_im: &[Float],
    dist: usize,
) {
    let chunk_size = dist << 1;
    assert!(chunk_size >= 16);

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .chunks_exact_mut(8)
                .zip(reals_s1.chunks_exact_mut(8))
                .zip(imags_s0.chunks_exact_mut(8))
                .zip(imags_s1.chunks_exact_mut(8))
                .zip(twiddles_re.chunks_exact(8))
                .zip(twiddles_im.chunks_exact(8))
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = f64x8::from_slice(re_s0);
                    let real_c1 = f64x8::from_slice(re_s1);
                    let imag_c0 = f64x8::from_slice(im_s0);
                    let imag_c1 = f64x8::from_slice(im_s1);

                    let tw_re = f64x8::from_slice(w_re);
                    let tw_im = f64x8::from_slice(w_im);

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
    state: &mut State,
    twiddles_re: &[Float],
    twiddles_im: &[Float],
    dist: usize,
) {
    let chunk_size = dist << 1;

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
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

/// `chunk_size == 16`, so hard code twiddle factors
pub(crate) fn fft_chunk_16(state: &mut State) {
    const CHUNK_SIZE: usize = 16;
    const DIST: usize = CHUNK_SIZE >> 1;

    const TWIDDLES_RE: [Float; DIST] = [
        1.0,
        0.923_879_532_511_286_7,
        std::f64::consts::FRAC_1_SQRT_2,
        0.382_683_432_365_089_67,
        0.0,
        -0.382_683_432_365_089_9,
        -std::f64::consts::FRAC_1_SQRT_2,
        -0.923_879_532_511_286_8,
    ];
    const TWIDDLES_IM: [Float; DIST] = [
        0.0,
        -0.382_683_432_365_089_8,
        -std::f64::consts::FRAC_1_SQRT_2,
        -0.923_879_532_511_286_7,
        -1.0,
        -0.923_879_532_511_286_7,
        -std::f64::consts::FRAC_1_SQRT_2,
        -0.382_683_432_365_089_6,
    ];

    state
        .reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(state.imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);
            let mut i = 0;

            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];

            // Butterfly operation for 0th pair -- multiply by 1 + i0, so twiddle is unneeded
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            reals_s1[i] = real_c0 - real_c1;
            imags_s1[i] = imag_c0 - imag_c1;
            i += 1;

            // Butterfly operation for 1st pair
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];
            let w_re = TWIDDLES_RE[i];
            let w_im = TWIDDLES_IM[i];
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[i] = v_re * w_re - v_im * w_im;
            imags_s1[i] = v_re * w_im + v_im * w_re;
            i += 1;

            // Butterfly operation for 2nd pair
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];
            let w_re = TWIDDLES_RE[i];
            let w_im = TWIDDLES_IM[i];
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[i] = v_re * w_re - v_im * w_im;
            imags_s1[i] = v_re * w_im + v_im * w_re;
            i += 1;

            // Butterfly operation for 3rd pair -- multiply by 0 + -i, so twiddle is unneeded
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];

            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            reals_s1[i] = imag_c0 - imag_c1;
            imags_s1[i] = -(real_c0 - real_c1);
            i += 1;

            // Butterfly operation for 4th pair
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];
            let w_re = TWIDDLES_RE[i];
            let w_im = TWIDDLES_IM[i];
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[i] = v_re * w_re - v_im * w_im;
            imags_s1[i] = v_re * w_im + v_im * w_re;
            i += 1;

            // Butterfly operation for 5th pair
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];
            let w_re = TWIDDLES_RE[i];
            let w_im = TWIDDLES_IM[i];
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[i] = v_re * w_re - v_im * w_im;
            imags_s1[i] = v_re * w_im + v_im * w_re;
            i += 1;

            // Butterfly operation for 6th pair
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];
            let w_re = TWIDDLES_RE[i];
            let w_im = TWIDDLES_IM[i];
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[i] = v_re * w_re - v_im * w_im;
            imags_s1[i] = v_re * w_im + v_im * w_re;
            i += 1;

            // Butterfly operation for 7th pair
            let real_c0 = reals_s0[i];
            let real_c1 = reals_s1[i];
            let imag_c0 = imags_s0[i];
            let imag_c1 = imags_s1[i];
            let w_re = TWIDDLES_RE[i];
            let w_im = TWIDDLES_IM[i];
            reals_s0[i] = real_c0 + real_c1;
            imags_s0[i] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[i] = v_re * w_re - v_im * w_im;
            imags_s1[i] = v_re * w_im + v_im * w_re;
        });
}

/// `chunk_size == 8`, so hard code twiddle factors
pub(crate) fn fft_chunk_8(state: &mut State) {
    const CHUNK_SIZE: usize = 8;
    const DIST: usize = CHUNK_SIZE >> 1;

    const TWIDDLES_RE: [Float; DIST] = [
        1.0,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
        -std::f64::consts::FRAC_1_SQRT_2,
    ];
    const TWIDDLES_IM: [Float; DIST] = [
        0.0,
        -std::f64::consts::FRAC_1_SQRT_2,
        -1.0,
        -std::f64::consts::FRAC_1_SQRT_2,
    ];

    state
        .reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(state.imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

            // Butterfly operation for 0th pair
            let real_c0 = reals_s0[0];
            let real_c1 = reals_s1[0];
            let imag_c0 = imags_s0[0];
            let imag_c1 = imags_s1[0];
            reals_s0[0] = real_c0 + real_c1;
            imags_s0[0] = imag_c0 + imag_c1;
            reals_s1[0] = real_c0 - real_c1;
            imags_s1[0] = imag_c0 - imag_c1;

            // Butterfly operation for 1st pair
            let real_c0 = reals_s0[1];
            let real_c1 = reals_s1[1];
            let imag_c0 = imags_s0[1];
            let imag_c1 = imags_s1[1];
            let w_re = TWIDDLES_RE[1];
            let w_im = TWIDDLES_IM[1];
            reals_s0[1] = real_c0 + real_c1;
            imags_s0[1] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[1] = v_re * w_re - v_im * w_im;
            imags_s1[1] = v_re * w_im + v_im * w_re;

            // Butterfly operation for 2nd pair
            let real_c0 = reals_s0[2];
            let real_c1 = reals_s1[2];
            let imag_c0 = imags_s0[2];
            let imag_c1 = imags_s1[2];
            reals_s0[2] = real_c0 + real_c1;
            imags_s0[2] = imag_c0 + imag_c1;
            reals_s1[2] = imag_c0 - imag_c1;
            imags_s1[2] = -(real_c0 - real_c1);

            // Butterfly operation for 3rd pair
            let real_c0 = reals_s0[3];
            let real_c1 = reals_s1[3];
            let imag_c0 = imags_s0[3];
            let imag_c1 = imags_s1[3];
            let w_re = TWIDDLES_RE[3];
            let w_im = TWIDDLES_IM[3];
            reals_s0[3] = real_c0 + real_c1;
            imags_s0[3] = imag_c0 + imag_c1;
            let v_re = real_c0 - real_c1;
            let v_im = imag_c0 - imag_c1;
            reals_s1[3] = v_re * w_re - v_im * w_im;
            imags_s1[3] = v_re * w_im + v_im * w_re;
        });
}

/// chunk_size == 4, so hard code twiddle factors
pub(crate) fn fft_chunk_4(state: &mut State) {
    let dist = 2;
    let chunk_size = dist << 1;

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

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

/// chunk_size == 2, so skip phase
pub(crate) fn fft_chunk_2(state: &mut State) {
    let dist = 1;
    state
        .reals
        .chunks_exact_mut(2)
        .zip(state.imags.chunks_exact_mut(2))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .iter_mut()
                .zip(reals_s1.iter_mut())
                .zip(imags_s0.iter_mut())
                .zip(imags_s1.iter_mut())
                .for_each(|(((re_s0, re_s1), im_s0), im_s1)| {
                    let real_c0 = *re_s0;
                    let real_c1 = *re_s1;
                    let imag_c0 = *im_s0;
                    let imag_c1 = *im_s1;

                    *re_s0 = real_c0 + real_c1;
                    *im_s0 = imag_c0 + imag_c1;
                    *re_s1 = real_c0 - real_c1;
                    *im_s1 = imag_c0 - imag_c1;
                });
        });
}
