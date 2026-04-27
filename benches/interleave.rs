//! Requires the `bench-internals` feature to access private functions in the
//! crate root. Run with:
//!
//! ```sh
//! cargo bench --bench interleave --features bench-internals
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use phastft::complex_nums::{combine_re_im, deinterleave};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::RngExt;

mod common;
use common::{groups, ids, sweep_complex, LENGTHS};

fn interleaved<T>(n: usize) -> Vec<T>
where
    StandardUniform: rand::prelude::Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();
    (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect()
}

fn re_im_pair<T>(n: usize) -> (Vec<T>, Vec<T>)
where
    StandardUniform: rand::prelude::Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();
    let reals: Vec<T> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    let imags: Vec<T> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    (reals, imags)
}

macro_rules! deinterleave_bench {
    ($name:ident, $float:ty, $group:expr) => {
        fn $name(c: &mut Criterion) {
            sweep_complex::<$float, _>(c, $group, LENGTHS, |g, len| {
                g.bench_function(BenchmarkId::new(ids::DEINTERLEAVE, len), |b| {
                    b.iter_batched(
                        || interleaved::<$float>(len),
                        |input| deinterleave(&input),
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

macro_rules! combine_bench {
    ($name:ident, $float:ty, $group:expr) => {
        fn $name(c: &mut Criterion) {
            sweep_complex::<$float, _>(c, $group, LENGTHS, |g, len| {
                g.bench_function(BenchmarkId::new(ids::COMBINE_RE_IM, len), |b| {
                    b.iter_batched(
                        || re_im_pair::<$float>(len),
                        |(reals, imags)| combine_re_im::<$float>(&reals, &imags),
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

deinterleave_bench!(deinterleave_f32, f32, groups::KERNEL_DEINTERLEAVE_F32);
deinterleave_bench!(deinterleave_f64, f64, groups::KERNEL_DEINTERLEAVE_F64);
combine_bench!(combine_f32, f32, groups::KERNEL_COMBINE_RE_IM_F32);
combine_bench!(combine_f64, f64, groups::KERNEL_COMBINE_RE_IM_F64);

criterion_group!(
    benches,
    deinterleave_f32,
    deinterleave_f64,
    combine_f32,
    combine_f64,
);
criterion_main!(benches);
