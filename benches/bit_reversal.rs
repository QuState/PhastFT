use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use phastft::cobra_apply;

pub fn cobra(c: &mut Criterion) {
    let mut group = c.benchmark_group("cobra_apply");

    for n in 15..20 {
        let big_n = 1 << n;
        let mut v: Vec<_> = (0..big_n).collect();
        group.bench_with_input(criterion::BenchmarkId::new("cobra", n), &n, |b, n| {
            b.iter(|| cobra_apply(black_box(&mut v), black_box(*n)))
        });
    }

    group.finish();
}

// pub fn cobra(c: &mut Criterion) {
//     let big_n = 1 << 18;
//     let mut v: Vec<_> = (0..big_n).collect();
//     c.bench_function("cobra18", |b| {
//         b.iter(|| cobra_apply(black_box(&mut v), black_box(18)))
//     });
// }

criterion_group!(benches, cobra);
criterion_main!(benches);
