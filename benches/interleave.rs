//! Requires the `bench-internals` feature to access private functions in the
//! crate root. Run with:
//!
//! ```sh
//! cargo bench --bench interleave --features bench-internals
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use phastft::complex_nums::{combine_re_im, deinterleave_complex32, deinterleave_complex64};

mod common;
use common::{
    bench_at_sizes, groups, ids, interleaved_complex, split_complex, throughput_complex, LENGTHS,
};

macro_rules! deinterleave_bench {
    ($name:ident, $float:ty, $deinter_fn:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    g.bench_function(BenchmarkId::new(ids::DEINTERLEAVE, len), |b| {
                        b.iter_batched(
                            || interleaved_complex::<$float>(len),
                            |input| $deinter_fn(&input),
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

macro_rules! combine_bench {
    ($name:ident, $float:ty, $group:expr) => {
        fn $name(c: &mut Criterion) {
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    g.bench_function(BenchmarkId::new(ids::COMBINE_RE_IM, len), |b| {
                        b.iter_batched(
                            || split_complex::<$float>(len),
                            |(reals, imags)| combine_re_im::<$float>(&reals, &imags),
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

deinterleave_bench!(
    deinterleave_f32,
    f32,
    deinterleave_complex32,
    groups::KERNEL_DEINTERLEAVE_F32
);
deinterleave_bench!(
    deinterleave_f64,
    f64,
    deinterleave_complex64,
    groups::KERNEL_DEINTERLEAVE_F64
);
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
