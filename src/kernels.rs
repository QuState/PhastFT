use spinoza::core::State;
use spinoza::math::Float;

// pub(crate) fn fft_chunk_n_simd(
//     state: &mut State,
//     twiddles_re: &mut [Float],
//     twiddles_im: &mut [Float],
//     dist: usize,
// ) {
//     let half_dist = dist / 2;
//     let chunk_size = dist << 1;
//     assert!(chunk_size > 16);
//     assert!(twiddles_re.len() == twiddles_im.len() && twiddles_re.len() == (dist / 2));
//
//     state
//         .reals
//         .chunks_exact_mut(chunk_size)
//         .zip(state.imags.chunks_exact_mut(chunk_size))
//         .for_each(|(reals_chunk, imags_chunk)| {
//             let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
//             let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);
//             let mut start = 0;
//             let mut end = 8;
//
//             reals_s0
//                 .chunks_exact_mut(8)
//                 .zip(reals_s1.chunks_exact_mut(8))
//                 .zip(imags_s0.chunks_exact_mut(8))
//                 .zip(imags_s1.chunks_exact_mut(8))
//                 .for_each(|(((re_s0, re_s1), im_s0), im_s1)| {
//                     let real_c0 = f64x8::from_slice(re_s0);
//                     let real_c1 = f64x8::from_slice(re_s1);
//                     let imag_c0 = f64x8::from_slice(im_s0);
//                     let imag_c1 = f64x8::from_slice(im_s1);
//
//                     let tw_re = f64x8::from_slice(&twiddles_re[start..end]);
//                     let tw_im = f64x8::from_slice(&twiddles_im[start..end]);
//
//                     re_s0.copy_from_slice((real_c0 + real_c1).as_array());
//                     im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
//                     let v_re = real_c0 - real_c1;
//                     let v_im = imag_c0 - imag_c1;
//                     re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array());
//                     im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array());
//
//                     for i in start..end {
//                         std::mem::swap(&mut twiddles_re[i], &mut twiddles_im[i]);
//                         twiddles_im[i] = -twiddles_im[i];
//                     }
//                     start += 8;
//                     end += 8;
//                     start %= half_dist;
//                     end %= half_dist;
//                 });
//         });
// }

pub(crate) fn fft_chunk_n(
    state: &mut State,
    twiddles_re: &mut [Float],
    twiddles_im: &mut [Float],
    dist: usize,
) {
    assert!(twiddles_re.len() == twiddles_im.len() && twiddles_re.len() == 1 + dist / 2);
    let chunk_size = dist << 1;
    let half_dist = (dist) / 2;

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);
            let mut i = 0;
            reals_s0
                .iter_mut()
                .take(half_dist + 1)
                .zip(reals_s1.iter_mut().take(half_dist + 1))
                .zip(imags_s0.iter_mut().take(half_dist + 1))
                .zip(imags_s1.iter_mut().take(half_dist + 1))
                .for_each(|(((re_s0, re_s1), im_s0), im_s1)| {
                    let w_re = twiddles_re[i];
                    let w_im = twiddles_im[i];
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
                    i += 1;
                });

            reals_s0
                .iter_mut()
                .skip(half_dist + 1)
                .zip(reals_s1.iter_mut().skip(half_dist + 1))
                .zip(imags_s0.iter_mut().skip(half_dist + 1))
                .zip(imags_s1.iter_mut().skip(half_dist + 1))
                .for_each(|(((re_s0, re_s1), im_s0), im_s1)| {
                    let w_re = -twiddles_re[dist - i];
                    let w_im = twiddles_im[dist - i];
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
                    i += 1;
                });

            /*
            let mut i = 0;
            reals_s0
                .iter_mut()
                .zip(reals_s1.iter_mut())
                .zip(imags_s0.iter_mut())
                .zip(imags_s1.iter_mut())
                .for_each(|(((re_s0, re_s1), im_s0), im_s1)| {
                    let (w_re, w_im) = if i > half_dist {
                        // println!("dist - i == {dist} - {i} == {}", dist - i);
                        println!("in second half:");
                        println!(
                            "{} w_im: {}\n-------------------------",
                            -twiddles_re[dist - i],
                            twiddles_im[dist - i]
                        );
                        println!("twiddles at: {}", dist - i);
                        (-twiddles_re[dist - i], twiddles_im[dist - i])
                    } else {
                        println!("in first half:");
                        println!("w_re: {} w_im: {}", twiddles_re[i], twiddles_im[i]);
                        println!("twiddles at: {i}");
                        (twiddles_re[i], twiddles_im[i])
                    };

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
                    i += 1;
                });
             */
        });
}

// fn fft_chunk_8(state: &mut State) {
//     let CHUNK_SIZE = 8;
//     let DIST = CHUNK_SIZE >> 1;
//
//     state
//         .reals
//         .chunks_exact_mut(CHUNK_SIZE)
//         .zip(state.imags.chunks_exact_mut(CHUNK_SIZE))
//         .for_each(|(reals_chunk, imags_chunk)| {
//             let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
//             let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);
//
//             reals_s0
//                 .iter_mut()
//                 .zip(reals_s1.iter_mut())
//                 .zip(imags_s0.iter_mut())
//                 .zip(imags_s1.iter_mut())
//                 .for_each(|(((((re_s0, re_s1), im_s0), im_s1)| {
//                     let real_c0 = *re_s0;
//                     let real_c1 = *re_s1;
//                     let imag_c0 = *im_s0;
//                     let imag_c1 = *im_s1;
//
//                     *re_s0 = real_c0 + real_c1;
//                     *im_s0 = imag_c0 + imag_c1;
//                     let v_re = real_c0 - real_c1;
//                     let v_im = imag_c0 - imag_c1;
//                     *re_s1 = v_re * w_re - v_im * w_im;
//                     *im_s1 = v_re * w_im + v_im * w_re;
//                 });
//         });
// }

/// chunk_size == 4, so hard code twiddle factors
pub(crate) fn fft_chunk_4(state: &mut State) {
    const CHUNK_SIZE: usize = 4;
    const DIST: usize = CHUNK_SIZE >> 1;

    state
        .reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(state.imags.chunks_exact_mut(CHUNK_SIZE))
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

/// chunk_size == 2, so skip phase
pub(crate) fn fft_chunk_2(state: &mut State) {
    const CHUNK_SIZE: usize = 2;
    const DIST: usize = CHUNK_SIZE >> 1;

    state
        .reals
        .chunks_exact_mut(CHUNK_SIZE)
        .zip(state.imags.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(DIST);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(DIST);

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
