#![feature(portable_simd, avx512_target_feature)]

use std::simd::{simd_swizzle, Simd, SimdElement};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use phastft::utils::deinterleave_naive;

// fn criterion_benchmark(c: &mut Criterion) {
//     let sizes = vec![1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];
//
//     let mut group = c.benchmark_group("r2c_versus_c2c");
//     for &size in &sizes {
//         group.throughput(Throughput::Elements(size as u64));
//
//         group.bench_with_input(BenchmarkId::new("r2c_fft", size), &size, |b, &size| {
//             let mut s_re = vec![0.0; size];
//             let mut s_im = vec![0.0; size];
//             gen_random_signal(&mut s_re, &mut s_im);
//
//             b.iter(|| {
//                 let mut output_re = vec![0.0; size];
//                 let mut output_im = vec![0.0; size];
//                 r2c_fft_f64(
//                     black_box(&mut s_re),
//                     black_box(&mut output_re),
//                     black_box(&mut output_im),
//                 );
//             });
//         });
//
//         group.bench_with_input(BenchmarkId::new("c2c_fft", size), &size, |b, &size| {
//             let mut s_re = vec![0.0; size];
//             let mut s_im = vec![0.0; size];
//             gen_random_signal(&mut s_re, &mut s_im);
//             s_im = vec![0.0; size];
//
//             b.iter(|| {
//                 fft_64(
//                     black_box(&mut s_re),
//                     black_box(&mut s_im),
//                     Direction::Forward,
//                 );
//             });
//         });
//
//         group.bench_with_input(BenchmarkId::new("real_fft", size), &size, |b, &size| {
//             let mut s_re = vec![0.0; size];
//             let mut s_im = vec![0.0; size];
//             gen_random_signal(&mut s_re, &mut s_im);
//             let mut output = vec![Complex::default(); s_re.len() / 2 + 1];
//
//             b.iter(|| {
//                 let mut planner = RealFftPlanner::<f64>::new();
//                 let fft = planner.plan_fft_forward(s_re.len());
//                 fft.process(&mut s_re, &mut output)
//                     .expect("fft.process() failed!");
//             });
//         });
//     }
//     group.finish();
// }

fn deinterleave<T: Copy + Default>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;

    let out_len = input.len() / 2;
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(CHUNK_SIZE * 2)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            odds[0] = in_chunk[0];
            evens[0] = in_chunk[1];
            odds[1] = in_chunk[2];
            evens[1] = in_chunk[3];
            odds[2] = in_chunk[4];
            evens[2] = in_chunk[5];
            odds[3] = in_chunk[6];
            evens[3] = in_chunk[7];
        });

    (out_odd, out_even)
}

fn deinterleave_simd_swizzle<T: Copy + Default + SimdElement>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 8;
    const DOUBLE_CHUNK: usize = CHUNK_SIZE * 2;

    let out_len = input.len() / 2;
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(DOUBLE_CHUNK)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            let in_simd: Simd<T, DOUBLE_CHUNK> = Simd::from_array(in_chunk.try_into().unwrap());
            let result = simd_swizzle!(
                in_simd,
                [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
            );
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    (out_odd, out_even)
}

fn deinterleave_portable_simd_deintlv<T: Copy + Default + SimdElement>(
    input: &[T],
) -> (Vec<T>, Vec<T>) {
    let n = input.len();
    let mut evens = vec![T::default(); n / 2];
    let mut odds = vec![T::default(); n / 2];
    const CHUNK_SIZE: usize = 8;

    for ((chunk, chunk_re), chunk_im) in input
        .chunks_exact(CHUNK_SIZE * 2)
        .zip(evens.chunks_exact_mut(CHUNK_SIZE))
        .zip(odds.chunks_exact_mut(CHUNK_SIZE))
    {
        let (first_half, second_half) = chunk.split_at(8);

        let a: Simd<T, CHUNK_SIZE> = Simd::from_slice(first_half);
        let b: Simd<T, CHUNK_SIZE> = Simd::from_slice(second_half);
        let (re_deinterleaved, im_deinterleaved) = a.deinterleave(b);

        chunk_re.copy_from_slice(&re_deinterleaved.to_array());
        chunk_im.copy_from_slice(&im_deinterleaved.to_array());
    }
    (evens, odds)
}

fn interleave_benchmark(c: &mut Criterion) {
    let input: Vec<f64> = (0..(1 << 20)).map(|x| x as f64).collect();

    c.bench_function("deinterleave_naive", |b| {
        b.iter(|| deinterleave_naive(black_box(&input)))
    });

    c.bench_function("deinterleave_simple", |b| {
        b.iter(|| deinterleave(black_box(&input)))
    });

    c.bench_function("deinterleave_simd_swizzle", |b| {
        b.iter(|| deinterleave_simd_swizzle(black_box(&input)))
    });

    c.bench_function("deinterleave_portable_simd_deinterleave", |b| {
        b.iter(|| deinterleave_portable_simd_deintlv(black_box(&input)))
    });
}

criterion_group!(benches, interleave_benchmark);
criterion_main!(benches);
