use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fearless_simd::{dispatch, Level};
use phastft::algorithms::bravo::{bit_rev_bravo_f32, bit_rev_bravo_f64};
use rand::distr::StandardUniform;
use rand::prelude::*;

const LENGTHS: &[usize] = &[10, 14, 18, 20, 22, 24];

fn random_vec<T>(n: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = SmallRng::seed_from_u64(0xCAFE);
    (&mut rng)
        .sample_iter(StandardUniform)
        .take(1 << n)
        .collect()
}

// Old (pre-tiling) BRAVO inlined comparison to COBRAVO. Can be deleted later on.
mod old_bravo {
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f64x4, f64x8, Simd};

    fn scalar_bit_reversal<T: Default + Copy + Clone>(data: &mut [T], n: usize) {
        let big_n = data.len();
        for i in 0..big_n {
            let j = reverse_bits_scalar(i, n as u32);
            if i < j {
                data.swap(i, j);
            }
        }
    }

    fn reverse_bits_scalar(x: usize, bits: u32) -> usize {
        if bits == 0 {
            return 0;
        }
        x.reverse_bits() >> (usize::BITS - bits)
    }

    macro_rules! impl_bit_rev_bravo {
        ($fn_name:ident, $elem_ty:ty, $vec_ty:ty, $lanes:expr) => {
            #[inline(always)]
            fn $fn_name<S: Simd>(simd: S, data: &mut [$elem_ty], n: usize) {
                type Chunk<S> = $vec_ty;
                const LANES: usize = $lanes;
                assert!(<Chunk<S>>::N == LANES);

                let big_n = 1usize << n;
                assert_eq!(data.len(), big_n);

                if big_n < LANES * LANES {
                    scalar_bit_reversal(data, n);
                    return;
                }

                let log_w = LANES.ilog2() as usize;
                let num_classes = big_n / (LANES * LANES);
                let class_bits = n - 2 * log_w;
                let (data_chunks, _) = data.as_chunks_mut::<LANES>();

                let mut chunks_a: [Chunk<S>; LANES] =
                    [Chunk::splat(simd, Default::default()); LANES];
                let mut chunks_b: [Chunk<S>; LANES] =
                    [Chunk::splat(simd, Default::default()); LANES];

                for class_idx in 0..num_classes {
                    let class_idx_rev = if class_bits > 0 {
                        class_idx.reverse_bits() >> (usize::BITS - class_bits as u32)
                    } else {
                        0
                    };
                    if class_idx > class_idx_rev {
                        continue;
                    }

                    for j in 0..LANES {
                        chunks_a[j] =
                            Chunk::from_slice(simd, &data_chunks[class_idx + j * num_classes]);
                    }

                    for round in 0..log_w {
                        let stride = 1 << round;
                        let mut i = 0;
                        while i < LANES {
                            for offset in 0..stride {
                                let idx0 = i + offset;
                                let idx1 = i + offset + stride;
                                let vec0 = chunks_a[idx0];
                                let vec1 = chunks_a[idx1];
                                chunks_a[idx0] = vec0.zip_low(vec1);
                                chunks_a[idx1] = vec0.zip_high(vec1);
                            }
                            i += stride * 2;
                        }
                    }

                    if class_idx == class_idx_rev {
                        for j in 0..LANES {
                            chunks_a[j].store_slice(&mut data_chunks[class_idx + j * num_classes]);
                        }
                    } else {
                        for j in 0..LANES {
                            chunks_b[j] = Chunk::from_slice(
                                simd,
                                &data_chunks[class_idx_rev + j * num_classes],
                            );
                        }

                        for round in 0..log_w {
                            let stride = 1 << round;
                            let mut i = 0;
                            while i < LANES {
                                for offset in 0..stride {
                                    let idx0 = i + offset;
                                    let idx1 = i + offset + stride;
                                    let vec0 = chunks_b[idx0];
                                    let vec1 = chunks_b[idx1];
                                    chunks_b[idx0] = vec0.zip_low(vec1);
                                    chunks_b[idx1] = vec0.zip_high(vec1);
                                }
                                i += stride * 2;
                            }
                        }

                        for j in 0..LANES {
                            chunks_a[j]
                                .store_slice(&mut data_chunks[class_idx_rev + j * num_classes]);
                            chunks_b[j].store_slice(&mut data_chunks[class_idx + j * num_classes]);
                        }
                    }
                }
            }
        };
    }

    impl_bit_rev_bravo!(bit_rev_bravo_chunk_4_f32, f32, f32x4<S>, 4);
    impl_bit_rev_bravo!(bit_rev_bravo_chunk_8_f32, f32, f32x8<S>, 8);
    impl_bit_rev_bravo!(bit_rev_bravo_chunk_4_f64, f64, f64x4<S>, 4);
    impl_bit_rev_bravo!(bit_rev_bravo_chunk_8_f64, f64, f64x8<S>, 8);

    #[inline(always)]
    pub fn bit_rev_bravo_f32<S: Simd>(simd: S, data: &mut [f32], n: usize) {
        match <S::f32s>::N {
            4 => bit_rev_bravo_chunk_4_f32(simd, data, n),
            _ => bit_rev_bravo_chunk_8_f32(simd, data, n),
        }
    }

    #[inline(always)]
    pub fn bit_rev_bravo_f64<S: Simd>(simd: S, data: &mut [f64], n: usize) {
        match <S::f64s>::N {
            2 => bit_rev_bravo_chunk_4_f64(simd, data, n),
            _ => bit_rev_bravo_chunk_8_f64(simd, data, n),
        }
    }
}

fn bench_bit_reversal_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit reversal f64");
    group.sample_size(20);

    for &n in LENGTHS {
        let len = 1usize << n;
        group.throughput(Throughput::Bytes((len * size_of::<f64>()) as u64));

        group.bench_function(BenchmarkId::new("COBRAVO", len), |b| {
            let simd_level = Level::new();
            b.iter_batched(
                || random_vec::<f64>(n),
                |mut data| {
                    dispatch!(simd_level, simd => bit_rev_bravo_f64(simd, &mut data, n));
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_function(BenchmarkId::new("BRAVO", len), |b| {
            let simd_level = Level::new();
            b.iter_batched(
                || random_vec::<f64>(n),
                |mut data| {
                    dispatch!(simd_level, simd => old_bravo::bit_rev_bravo_f64(simd, &mut data, n));
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_bit_reversal_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit reversal f32");
    group.sample_size(20);

    for &n in LENGTHS {
        let len = 1usize << n;
        group.throughput(Throughput::Bytes((len * size_of::<f32>()) as u64));

        group.bench_function(BenchmarkId::new("COBRAVO", len), |b| {
            let simd_level = Level::new();
            b.iter_batched(
                || random_vec::<f32>(n),
                |mut data| {
                    dispatch!(simd_level, simd => bit_rev_bravo_f32(simd, &mut data, n));
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_function(BenchmarkId::new("BRAVO", len), |b| {
            let simd_level = Level::new();
            b.iter_batched(
                || random_vec::<f32>(n),
                |mut data| {
                    dispatch!(simd_level, simd => old_bravo::bit_rev_bravo_f32(simd, &mut data, n));
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bit_reversal_f64, bench_bit_reversal_f32);
criterion_main!(benches);
