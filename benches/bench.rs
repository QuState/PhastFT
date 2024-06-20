#![feature(portable_simd, avx512_target_feature)]

use std::simd::{simd_swizzle, Simd, SimdElement};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

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

#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
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

#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
fn deinterleave_simd_swizzle<T: Copy + Default + SimdElement>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;
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
            let result = simd_swizzle!(in_simd, [0, 2, 4, 6, 1, 3, 5, 7]);
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    (out_odd, out_even)
}

// We don't multiversion for AVX-512 here and keep the chunk size below AVX-512
// because we haven't seen any gains from it in benchmarks.
// This might be due to us running benchmarks on Zen4 which implements AVX-512
// on top of 256-bit wide execution units.
//
// If benchmarks on "real" AVX-512 show improvement on AVX-512
// without degrading AVX2 machines due to larger chunk size,
// the AVX-512 specialization should be re-enabled.
#[multiversion::multiversion(
    targets(
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length
pub(crate) fn deinterleave_from_pr<T: Copy + Default + SimdElement>(
    input: &[T],
) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;
    const DOUBLE_CHUNK: usize = CHUNK_SIZE * 2;

    let out_len = input.len() / 2;
    // We've benchmarked, and it turns out that this approach with zeroed memory
    // is faster than using uninit memory and bumping the length once in a while!
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(DOUBLE_CHUNK)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            let in_simd: Simd<T, DOUBLE_CHUNK> = Simd::from_array(in_chunk.try_into().unwrap());
            // This generates *slightly* faster code than just assigning values by index.
            // You'd think simd::deinterleave would be appropriate, but it does something different!
            let result = simd_swizzle!(in_simd, [0, 2, 4, 6, 1, 3, 5, 7]);
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    // Process the remainder, too small for the vectorized loop
    let input_rem = input.chunks_exact(DOUBLE_CHUNK).remainder();
    let odds_rem = out_odd.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let evens_rem = out_even.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    input_rem
        .chunks_exact(2)
        .zip(odds_rem.iter_mut())
        .zip(evens_rem.iter_mut())
        .for_each(|((inp, odd), even)| {
            *odd = inp[0];
            *even = inp[1];
        });

    (out_odd, out_even)
}

fn benchmark_deinterleave(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("deinterleave"));

    for s in (4..=28).step_by(4) {
        let size = 1 << s;
        let input: Vec<f64> = (0..size).map(|x| x as f64).collect();

        group.bench_with_input(
            BenchmarkId::new("Naive deinterleave", size),
            &input,
            |b, input| b.iter(|| deinterleave_naive(black_box(input))),
        );

        group.bench_with_input(
            BenchmarkId::new("Autovectorized deinterleave", size),
            &input,
            |b, input| b.iter(|| deinterleave(black_box(input))),
        );

        group.bench_with_input(
            BenchmarkId::new("Simd Swizzle deinterleave", size),
            &input,
            |b, input| b.iter(|| deinterleave_simd_swizzle(black_box(input))),
        );

        group.bench_with_input(
            BenchmarkId::new("PR deinterleave", size),
            &input,
            |b, input| b.iter(|| deinterleave_simd_swizzle(black_box(input))),
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_deinterleave);
criterion_main!(benches);
