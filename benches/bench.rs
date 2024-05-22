#![feature(portable_simd, avx512_target_feature)]

use std::simd::{f32x16, f64x8};

use bytemuck::cast_slice;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex;

use phastft::separate_re_im_f64;

macro_rules! impl_separate_re_im_with_capacity {
    ($func_name:ident, $precision:ty, $lanes:literal, $simd_vec:ty) => {
        /// Utility function to separate interleaved format signals (i.e., Vector of Complex Number Structs)
        /// into separate vectors for the corresponding real and imaginary components.
        #[multiversion::multiversion(
                                    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                            "x86_64+avx2+fma", // x86_64-v3
                                            "x86_64+sse4.2", // x86_64-v2
                                            "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                            "x86+avx2+fma",
                                            "x86+sse4.2",
                                            "x86+sse2",
                                        ))]
        pub fn $func_name(
            signal: &[Complex<$precision>],
        ) -> (Vec<$precision>, Vec<$precision>) {
            let n = signal.len();
            let mut reals = Vec::with_capacity(n);
            let mut imags = Vec::with_capacity(n);

            let complex_t: &[$precision] = cast_slice(signal);
            const CHUNK_SIZE: usize = $lanes * 2;

            for chunk in complex_t
                .chunks_exact(CHUNK_SIZE)
            {
                let (first_half, second_half) = chunk.split_at($lanes);

                let a = <$simd_vec>::from_slice(&first_half);
                let b = <$simd_vec>::from_slice(&second_half);
                let (re_deinterleaved, im_deinterleaved) = a.deinterleave(b);

                reals.extend_from_slice(&re_deinterleaved.to_array());
                imags.extend_from_slice(&im_deinterleaved.to_array());
            }

    let remainder = complex_t.chunks_exact(CHUNK_SIZE).remainder();
    if !remainder.is_empty() {
        let i = n - remainder.len() / 2;
        for (j, chunk) in remainder.chunks_exact(2).enumerate() {
            reals.push(chunk[0]);
            imags.push(chunk[1]);
        }
    }

            (reals, imags)
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_separate_re_im_with_capacity!(separate_re_im_f32_with_cap, f32, 16, f32x16);

#[cfg(feature = "complex-nums")]
impl_separate_re_im_with_capacity!(separate_re_im_f64_with_cap, f64, 8, f64x8);

fn criterion_benchmark(c: &mut Criterion) {
    let signal: Vec<Complex<f64>> = (0..(1 << 20))
        .map(|x| Complex::new(x as f64, (x * 2) as f64))
        .collect();

    c.bench_function("separate_re_im_with_zeroed_vecs", |b| {
        b.iter(|| separate_re_im_f64(black_box(&signal)));
    });

    c.bench_function("separate_re_im_with_capacity", |b| {
        b.iter(|| separate_re_im_f64_with_cap(black_box(&signal)));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
