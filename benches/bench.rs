use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use phastft::{fft::r2c_fft_f64, fft_64, planner::Direction};
use utilities::gen_random_signal;

fn criterion_benchmark(c: &mut Criterion) {
    let sizes = vec![1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];

    let mut group = c.benchmark_group("r2c_versus_c2c");
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("r2c_fft", size), &size, |b, &size| {
            let mut s_re = vec![0.0; size];
            let mut s_im = vec![0.0; size];
            gen_random_signal(&mut s_re, &mut s_im);

            b.iter(|| {
                let mut output_re = vec![0.0; size];
                let mut output_im = vec![0.0; size];
                r2c_fft_f64(
                    black_box(&mut s_re),
                    black_box(&mut output_re),
                    black_box(&mut output_im),
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("c2c_fft", size), &size, |b, &size| {
            let mut s_re = vec![0.0; size];
            let mut s_im = vec![0.0; size];
            gen_random_signal(&mut s_re, &mut s_im);
            s_im = vec![0.0; size];

            b.iter(|| {
                fft_64(
                    black_box(&mut s_re),
                    black_box(&mut s_im),
                    Direction::Forward,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
