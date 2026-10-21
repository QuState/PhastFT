//! Experimental out-of-place FFT with four full-array sweeps.
//!
//! For N = rows * cols, n = j1 * cols + j2 and k = k1 + rows * k2:
//! X[k] = sum_j2 (sum_j1 x[n] W_rows^(j1*k1)) W_N^(j2*k1) W_cols^(j2*k2).
//! The four sweeps are:
//! 1. Gather columns in cache-line-wide batches, FFT each, apply twiddles.
//! 2. Transpose to make the second set of FFTs contiguous.
//! 3. FFT those rows.
//! 4. Transpose into natural frequency order in the scratch buffers.
//!
//! Local FFTs make multiple cache passes; "four" counts full-array sweeps,
//! not literal memory transactions. Input is destroyed, output is in scratch.
//! The two small DIT plans and twiddle steps take O(sqrt(N)) space; scratch
//! takes N complex elements. No allocations occur during execution.

use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Sixteen adjacent elements cover a cache line in either precision.
const COLUMNS: usize = 16;
const TILE: usize = 32;

fn chunks<T: Send, F>(re: &mut [T], im: &mut [T], size: usize, f: F)
where
    F: Fn(usize, &mut [T], &mut [T]) + Send + Sync,
{
    #[cfg(feature = "parallel")]
    re.par_chunks_mut(size)
        .zip(im.par_chunks_mut(size))
        .enumerate()
        .for_each(|(i, (re, im))| f(i, re, im));
    #[cfg(not(feature = "parallel"))]
    re.chunks_mut(size)
        .zip(im.chunks_mut(size))
        .enumerate()
        .for_each(|(i, (re, im))| f(i, re, im));
}

fn transpose<T: Copy + Send + Sync>(
    re: &[T],
    im: &[T],
    out_re: &mut [T],
    out_im: &mut [T],
    rows: usize,
    cols: usize,
) {
    chunks(out_re, out_im, TILE * rows, |batch, dr, di| {
        let first_col = batch * TILE;
        for r0 in (0..rows).step_by(TILE) {
            for c in 0..dr.len() / rows {
                for r in r0..(r0 + TILE).min(rows) {
                    dr[c * rows + r] = re[r * cols + first_col + c];
                    di[c * rows + r] = im[r * cols + first_col + c];
                }
            }
        }
    });
}

macro_rules! implementation {
    ($name:ident, $float:ty, $dit:ident, $fft:ident) => {
        pub struct $name {
            rows: usize,
            cols: usize,
            column_plan: $dit,
            row_plan: $dit,
            // f64 recurrence even for f32, restarted every 256 elements.
            steps: Vec<(f64, f64)>,
            opts: Options,
        }

        impl $name {
            pub fn new(n: usize) -> Self {
                assert!(n.is_power_of_two());
                let rows = 1 << (n.ilog2() / 2);
                let cols = n / rows;
                let mut opts = Options::default();
                opts.smallest_parallel_chunk_size = usize::MAX;
                opts.multithreaded_bit_reversal = false;
                Self {
                    rows,
                    cols,
                    column_plan: $dit::new(rows),
                    row_plan: $dit::new(cols),
                    steps: (0..cols)
                        .map(|j| {
                            let (im, re) = (-std::f64::consts::TAU * j as f64 / n as f64).sin_cos();
                            (re, im)
                        })
                        .collect(),
                    opts,
                }
            }

            /// Output is in `scratch_re` / `scratch_im`, in natural order.
            /// Inverse transforms are normalized by 1/N, like PhastFT's DIT.
            pub fn process(
                &self,
                re: &mut [$float],
                im: &mut [$float],
                scratch_re: &mut [$float],
                scratch_im: &mut [$float],
                direction: Direction,
            ) {
                let n = self.rows * self.cols;
                for len in [re.len(), im.len(), scratch_re.len(), scratch_im.len()] {
                    assert_eq!(len, n);
                }
                let rows = self.rows;
                let cols = self.cols;
                let sign = match direction {
                    Direction::Forward => -1.0,
                    Direction::Inverse => 1.0,
                };

                // Sweep 1: tiled gather, local column FFTs and twiddles. Each
                // task owns contiguous scratch rows, so no unsafe scatter is needed.
                chunks(scratch_re, scratch_im, COLUMNS * rows, |batch, dr, di| {
                    let first_col = batch * COLUMNS;
                    for r0 in (0..rows).step_by(TILE) {
                        for c in 0..dr.len() / rows {
                            for r in r0..(r0 + TILE).min(rows) {
                                dr[c * rows + r] = re[r * cols + first_col + c];
                                di[c * rows + r] = im[r * cols + first_col + c];
                            }
                        }
                    }
                    for (c, (rr, ii)) in dr.chunks_mut(rows).zip(di.chunks_mut(rows)).enumerate() {
                        $fft(rr, ii, direction, &self.column_plan, &self.opts);
                        let j = first_col + c;
                        if j == 0 {
                            continue;
                        }
                        let (step_re, forward_im) = self.steps[j];
                        let step_im = -sign * forward_im;
                        for start in (0..rows).step_by(256) {
                            let angle =
                                sign * std::f64::consts::TAU * j as f64 * start as f64 / n as f64;
                            let (mut wi, mut wr) = angle.sin_cos();
                            for k in start..(start + 256).min(rows) {
                                let ar = rr[k];
                                let ai = ii[k];
                                rr[k] = ar * wr as $float - ai * wi as $float;
                                ii[k] = ar * wi as $float + ai * wr as $float;
                                (wr, wi) =
                                    (wr * step_re - wi * step_im, wr * step_im + wi * step_re);
                            }
                        }
                    }
                });

                // Sweep 2: A[j2][k1] -> B[k1][j2].
                transpose(scratch_re, scratch_im, re, im, cols, rows);
                // Sweep 3: local row FFTs. Parallelism is across rows only.
                chunks(re, im, cols, |_, rr, ii| {
                    $fft(rr, ii, direction, &self.row_plan, &self.opts);
                });
                // Sweep 4: B[k1][k2] -> X[k2 * rows + k1].
                transpose(re, im, scratch_re, scratch_im, rows, cols);
            }
        }
    };
}

implementation!(
    FourPass32,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts
);
implementation!(
    FourPass64,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts
);

#[cfg(test)]
mod tests {
    use super::*;
    use utilities::rustfft::{num_complex::Complex, FftPlanner};

    macro_rules! correctness {
        ($name:ident, $float:ty, $plan:ident, $tol:expr) => {
            #[test]
            fn $name() {
                let mut reference = FftPlanner::<$float>::new();
                // Square/rectangular matrices, partial tiles, and recurrence restarts.
                for log_n in 0..=20 {
                    let n = 1 << log_n;
                    let input: Vec<_> = (0..n)
                        .map(|i| {
                            Complex::new(
                                ((i * 17 % 257) as $float - 128.0) / 128.0,
                                ((i * 31 % 251) as $float - 125.0) / 125.0,
                            )
                        })
                        .collect();
                    let plan = $plan::new(n);
                    for direction in [Direction::Forward, Direction::Inverse] {
                        let mut expected = input.clone();
                        match direction {
                            Direction::Forward => reference.plan_fft_forward(n),
                            Direction::Inverse => reference.plan_fft_inverse(n),
                        }
                        .process(&mut expected);
                        if direction == Direction::Inverse {
                            for x in &mut expected {
                                *x /= n as $float;
                            }
                        }
                        let mut re: Vec<_> = input.iter().map(|x| x.re).collect();
                        let mut im: Vec<_> = input.iter().map(|x| x.im).collect();
                        let mut sr = vec![0.0; n];
                        let mut si = vec![0.0; n];
                        plan.process(&mut re, &mut im, &mut sr, &mut si, direction);
                        let mut error = 0.0f64;
                        let mut energy = 0.0f64;
                        let mut max_error = 0.0f64;
                        let mut max_expected = 0.0f64;
                        for k in 0..n {
                            let dr = (sr[k] - expected[k].re) as f64;
                            let di = (si[k] - expected[k].im) as f64;
                            error += dr * dr + di * di;
                            energy += expected[k].norm_sqr() as f64;
                            max_error = max_error.max(dr.hypot(di));
                            max_expected = max_expected.max(expected[k].norm() as f64);
                        }
                        assert!(
                            (error / energy).sqrt() < $tol,
                            "N={n} {direction:?}: relative L2 error {}",
                            (error / energy).sqrt()
                        );
                        assert!(
                            max_error / max_expected < $tol,
                            "N={n} {direction:?}: relative peak error {}",
                            max_error / max_expected
                        );
                    }
                }
            }
        };
    }
    correctness!(matches_rustfft_f32, f32, FourPass32, 2e-6);
    correctness!(matches_rustfft_f64, f64, FourPass64, 2e-13);
}
