//! Shared helpers for the criterion bench targets in `benches/`.
//!
//! Each entry in `Cargo.toml`'s `[[bench]]` block is its own binary so
//! FFTW's per-process wisdom cache cannot leak between planning modes. The
//! size sweep, sample distribution, throughput reporting, and group / ID
//! spellings live here so renaming requires touching one file.
//!
//! ## Group naming convention
//!
//! Group names are **snake_case** with category prefixes (`c2c_*`, `r2c_*`,
//! `c2r_*`, `planner_*`, `kernel_*`) so they survive criterion's filename
//! sanitizer (which replaces `?"/\*<>:|^` with `_`) unchanged, can be
//! passed to `cargo bench --bench <name> <filter>` and
//! `plot_criterion_overlay.py --groups <list>` without shell quoting, and
//! tab-complete in shells. The Python overlay script humanizes them at
//! plot time via the `GROUPS` registry — chart titles read "C2C Forward
//! (f32)" or "R2C (f32)".

#![allow(dead_code)]

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, PlotConfiguration, Throughput};
use num_traits::Float;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::RngExt;
use utilities::rustfft::num_complex::Complex;

/// Default power-of-2 size sweep (log2). Every cross-library and
/// PhastFT-internal FFT group iterates this list unless it provides a
/// reason to override (see `BIT_REVERSAL_LENGTHS`, `PLANNER_MODE_LENGTHS`).
pub const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

/// Bit-reversal kernel only kicks in at `n >= 10` (the SIMD path needs at
/// least one full `LANES * LANES` chunk), so it has its own floor.
pub const BIT_REVERSAL_LENGTHS: &[usize] =
    &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];

/// Planner-mode comparison stops at 2^18 — larger sizes blow out the
/// per-bench budget when both `Heuristic` and `Tune` planners are
/// constructed for the same `n`.
pub const PLANNER_MODE_LENGTHS: &[usize] = &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

/// Number of samples criterion collects per (group, id, size). 20 is a
/// compromise between run time and stable medians; tighter convergence comes
/// from running the bench multiple times rather than raising this number.
pub const SAMPLE_SIZE: usize = 20;

/// Build a benchmark group with the project's standard config: log-x
/// summary scale and `SAMPLE_SIZE` samples per (id, size).
pub fn make_group<'c>(c: &'c mut Criterion, name: &str) -> BenchmarkGroup<'c, WallTime> {
    let mut group = c.benchmark_group(name);
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));
    group.sample_size(SAMPLE_SIZE);
    group
}

/// Run `bench_at_size(group, len)` for every `len = 2^n` in `lengths`,
/// setting `group.throughput(throughput(len))` per size. Wraps the
/// LENGTHS loop + `make_group` + per-size `throughput()` + `group.finish()`
/// so each bench file just declares the per-size body.
///
/// `throughput` selects the accounting model independently of the input
/// data layout — pass `throughput_complex::<T>` when each element is a
/// real/imag pair (split or interleaved layout), `throughput_real::<T>`
/// when each element is a single scalar (R2C / C2R / bit-reversal).
pub fn bench_at_sizes<F>(
    c: &mut Criterion,
    group_name: &str,
    lengths: &[usize],
    throughput: impl Fn(usize) -> Throughput,
    mut bench_at_size: F,
) where
    F: FnMut(&mut BenchmarkGroup<'_, WallTime>, usize),
{
    let mut group = make_group(c, group_name);
    for &n in lengths {
        let len = 1 << n;
        group.throughput(throughput(len));
        bench_at_size(&mut group, len);
    }
    group.finish();
}

/// Throughput for an N-element complex signal (split or interleaved):
/// `N` elements, `2 * N * size_of::<T>()` bytes.
pub fn throughput_complex<T>(len: usize) -> Throughput {
    Throughput::ElementsAndBytes {
        elements: len as u64,
        bytes: (2 * len * size_of::<T>()) as u64,
    }
}

/// Throughput for an N-element real signal (R2C input / C2R output):
/// `N` elements, `N * size_of::<T>()` bytes.
pub fn throughput_real<T>(len: usize) -> Throughput {
    Throughput::ElementsAndBytes {
        elements: len as u64,
        bytes: (len * size_of::<T>()) as u64,
    }
}

/// Random complex signal in two split arrays `(reals, imags)`.
///
/// Distribution: `StandardUniform` over `[0, 1)` with `SmallRng`. Used by the
/// PhastFT split-array API and FFTW's `(reals, imags) -> AlignedVec<Complex>`
/// staging in the FFTW benches.
pub fn split_complex<T: Float>(n: usize) -> (Vec<T>, Vec<T>)
where
    StandardUniform: Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();
    let samples: Vec<T> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect();

    let mut reals = vec![T::zero(); n];
    let mut imags = vec![T::zero(); n];

    for ((z_re, z_im), chunk) in reals
        .iter_mut()
        .zip(imags.iter_mut())
        .zip(samples.chunks_exact(2))
    {
        *z_re = chunk[0];
        *z_im = chunk[1];
    }
    (reals, imags)
}

/// Random complex signal as interleaved `Complex<T>`. Same distribution as
/// `split_complex` so the two forms can be cross-checked.
pub fn interleaved_complex<T: Float>(n: usize) -> Vec<Complex<T>>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();
    let samples: Vec<T> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect();

    let mut signal = vec![Complex::new(T::zero(), T::zero()); n];
    for (z, chunk) in signal.iter_mut().zip(samples.chunks_exact(2)) {
        z.re = chunk[0];
        z.im = chunk[1];
    }
    signal
}

/// Random real signal of length `N`. The R2C input shape both PhastFT and
/// the realfft crate consume.
pub fn real_signal<T: Float>(n: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();
    (&mut rng).sample_iter(StandardUniform).take(n).collect()
}

/// Random "valid" half-spectrum (`N/2 + 1` complex bins) in split-array
/// form — the PhastFT C2R input shape.
///
/// DC (`k = 0`) and Nyquist (`k = N/2`) imaginary parts are zeroed: a
/// real-input DFT has `X[0]` and `X[N/2]` purely real, and some realfft
/// versions assert it. Zeroing here keeps the two libraries' inputs
/// comparable.
pub fn spectrum_split<T: Float>(n_real: usize) -> (Vec<T>, Vec<T>)
where
    StandardUniform: Distribution<T>,
{
    let len = n_real / 2 + 1;
    let mut rng = rand::make_rng::<SmallRng>();
    let samples: Vec<T> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * len)
        .collect();
    let mut re = vec![T::zero(); len];
    let mut im = vec![T::zero(); len];
    for ((r, i), c) in re
        .iter_mut()
        .zip(im.iter_mut())
        .zip(samples.chunks_exact(2))
    {
        *r = c[0];
        *i = c[1];
    }
    im[0] = T::zero();
    im[len - 1] = T::zero();
    (re, im)
}

/// Same data shape as `spectrum_split`, but interleaved as `Complex<T>` —
/// the realfft crate's C2R input shape.
pub fn spectrum_interleaved<T: Float>(n_real: usize) -> Vec<Complex<T>>
where
    StandardUniform: Distribution<T>,
{
    let len = n_real / 2 + 1;
    let mut rng = rand::make_rng::<SmallRng>();
    let samples: Vec<T> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * len)
        .collect();
    let mut signal = vec![Complex::new(T::zero(), T::zero()); len];
    for (z, c) in signal.iter_mut().zip(samples.chunks_exact(2)) {
        z.re = c[0];
        z.im = c[1];
    }
    signal[0].im = T::zero();
    signal[len - 1].im = T::zero();
    signal
}

/// Snake_case group names — round-trip clean through criterion's filename
/// sanitizer, shell args, and tab-completion. Humanized to "C2C Forward
/// (f32)" etc. by `plot_criterion_overlay.py` at plot time.
///
/// Naming convention:
/// - `c2c_*` — complex-to-complex FFT (the standard Cooley-Tukey path)
/// - `r2c_*` / `c2r_*` — real-input / real-output FFT (direction is
///   implicit: R2C is forward, C2R is inverse)
/// - `planner_*` — planner construction cost
/// - `kernel_*` — internal SIMD kernels exposed for micro-benchmarking,
///   not part of the public FFT pipeline
pub mod groups {
    pub const C2C_FORWARD_F32: &str = "c2c_forward_f32";
    pub const C2C_FORWARD_F64: &str = "c2c_forward_f64";
    pub const C2C_INVERSE_F32: &str = "c2c_inverse_f32";
    pub const C2C_INVERSE_F64: &str = "c2c_inverse_f64";

    pub const R2C_F32: &str = "r2c_f32";
    pub const R2C_F64: &str = "r2c_f64";
    pub const C2R_F32: &str = "c2r_f32";
    pub const C2R_F64: &str = "c2r_f64";

    pub const PLANNER_F32: &str = "planner_f32";
    pub const PLANNER_F64: &str = "planner_f64";

    pub const PLANNER_MODE_F32: &str = "planner_mode_f32";
    pub const PLANNER_MODE_F64: &str = "planner_mode_f64";

    pub const KERNEL_DEINTERLEAVE_F32: &str = "kernel_deinterleave_f32";
    pub const KERNEL_DEINTERLEAVE_F64: &str = "kernel_deinterleave_f64";
    pub const KERNEL_COMBINE_RE_IM_F32: &str = "kernel_combine_re_im_f32";
    pub const KERNEL_COMBINE_RE_IM_F64: &str = "kernel_combine_re_im_f64";

    pub const KERNEL_BIT_REVERSAL_F32: &str = "kernel_bit_reversal_f32";
    pub const KERNEL_BIT_REVERSAL_F64: &str = "kernel_bit_reversal_f64";
}

/// Series IDs (the inner directory under `target/criterion/<group>/`).
/// Human-readable PascalCase / lowercase-crate-name; never used as a CLI
/// filter argument, so spaces and case mixing are fine.
pub mod ids {
    pub const PHASTFT_DIT: &str = "PhastFT DIT";
    pub const PHASTFT_R2C: &str = "PhastFT R2C";
    pub const PHASTFT_C2R: &str = "PhastFT C2R";

    pub const RUSTFFT: &str = "RustFFT";
    pub const REALFFT: &str = "realfft";

    pub const FFTW_ESTIMATE: &str = "FFTW Estimate";
    pub const FFTW_MEASURE: &str = "FFTW Measure";
    pub const FFTW_CONSERVE: &str = "FFTW Conserve";

    pub const HEURISTIC: &str = "Heuristic";
    pub const TUNE: &str = "Tune";

    pub const COBRAVO: &str = "COBRAVO";
    pub const BRAVO: &str = "BRAVO";

    pub const DEINTERLEAVE: &str = "deinterleave";
    pub const COMBINE_RE_IM: &str = "combine_re_im";
}
