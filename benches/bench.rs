use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::Float;
use phastft::{
    fft_32_with_opts_and_plan, fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner32, Planner64},
};
use rand::{
    distributions::{Distribution, Standard},
    thread_rng, Rng,
};

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
];

fn generate_numbers<T: Float>(n: usize) -> (Vec<T>, Vec<T>)
where
    Standard: Distribution<T>,
{
    let mut rng = thread_rng();

    let samples: Vec<T> = (&mut rng).sample_iter(Standard).take(2 * n).collect();

    let mut reals = vec![T::zero(); n];
    let mut imags = vec![T::zero(); n];

    for ((z_re, z_im), rand_chunk) in reals
        .iter_mut()
        .zip(imags.iter_mut())
        .zip(samples.chunks_exact(2))
    {
        *z_re = rand_chunk[0];
        *z_im = rand_chunk[1];
    }

    (reals, imags)
}

fn benchmark_forward_f32(c: &mut Criterion) {
    let options = Options::default();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        let id = format!("FFT Forward f32 {} elements", len);
        c.bench_function(&id, |b| {
            let (mut reals, mut imags) = generate_numbers(len);
            let planner = Planner32::new(len, Direction::Forward);
            b.iter(|| {
                black_box(fft_32_with_opts_and_plan(
                    &mut reals, &mut imags, &options, &planner,
                ));
            });
        });
    }
}

fn benchmark_inverse_f32(c: &mut Criterion) {
    let options = Options::default();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        let id = format!("FFT Inverse f32 {} elements", len);
        c.bench_function(&id, |b| {
            let (mut reals, mut imags) = generate_numbers(len);
            let planner = Planner32::new(len, Direction::Reverse);
            b.iter(|| {
                black_box(fft_32_with_opts_and_plan(
                    &mut reals, &mut imags, &options, &planner,
                ));
            });
        });
    }
}

fn benchmark_forward_f64(c: &mut Criterion) {
    let options = Options::default();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        let id = format!("FFT Forward f64 {} elements", len);
        c.bench_function(&id, |b| {
            let (mut reals, mut imags) = generate_numbers(len);
            let planner = Planner64::new(len, Direction::Forward);
            b.iter(|| {
                black_box(fft_64_with_opts_and_plan(
                    &mut reals, &mut imags, &options, &planner,
                ));
            });
        });
    }
}

fn benchmark_inverse_f64(c: &mut Criterion) {
    let options = Options::default();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        let id = format!("FFT Inverse f64 {} elements", len);
        c.bench_function(&id, |b| {
            let (mut reals, mut imags) = generate_numbers(len);
            let planner = Planner64::new(len, Direction::Reverse);
            b.iter(|| {
                black_box(fft_64_with_opts_and_plan(
                    &mut reals, &mut imags, &options, &planner,
                ));
            });
        });
    }
}

criterion_group!(
    benches,
    benchmark_forward_f32,
    benchmark_inverse_f32,
    benchmark_forward_f64,
    benchmark_inverse_f64
);
criterion_main!(benches);
