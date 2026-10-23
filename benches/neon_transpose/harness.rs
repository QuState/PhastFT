#![allow(dead_code)]
mod dedicated;
mod generic;
use fearless_simd::Level;
use phastft::fft_f32_dit_with_planner_and_opts;
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32};
use std::hint::black_box;

fn main() {
    let neon = Level::new().as_neon().unwrap();
    // Include payload NaNs, signed zero, infinities and subnormals: a transpose
    // must preserve bits, independent of the floating-point interpretation.
    for pattern in 0..2 {
        let input = std::array::from_fn(|k| {
            if pattern == 0 {
                k as f32
            } else {
                f32::from_bits(
                    [
                        0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1, 0x80000001,
                        0x00800000,
                    ][k % 8],
                )
            }
        });
        let mut a = [0.0; 64];
        let mut b = [0.0; 64];
        dedicated::probe_transpose(neon, black_box(&input), black_box(&mut a));
        generic::probe_transpose(neon, black_box(&input), black_box(&mut b));
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(a[row * 8 + col].to_bits(), input[col * 8 + row].to_bits());
                assert_eq!(a[row * 8 + col].to_bits(), b[row * 8 + col].to_bits());
            }
        }
    }
    // Keep both FFT probes in the generated assembly, with unknown inputs and
    // plan values. The public API is checked separately below.
    let input = std::array::from_fn(|k| k as f32);
    let mut r = input;
    let mut i = input;
    let plan = dedicated::Plan::new(64);
    dedicated::probe_fft64(neon, black_box(&mut r), black_box(&mut i), black_box(&plan));
    let mut r2 = input;
    let mut i2 = input;
    let plan = generic::Plan::new(64);
    generic::probe_fft64(
        neon,
        black_box(&mut r2),
        black_box(&mut i2),
        black_box(&plan),
    );
    assert_eq!(r, r2);
    assert_eq!(i, i2);

    for log_n in 0..=9 {
        let n = 1 << log_n;
        let planner = PlannerDit32::new(n);
        let opts = Options::guess_options(n);
        for case in 0..4 {
            let input: Vec<(f32, f32)> = (0..n)
                .map(|k| match case {
                    0 => (0.0, 0.0),
                    1 if k == n / 3 => (0.75, -0.25),
                    1 => (0.0, 0.0),
                    2 => (
                        ((k * 17 % 257) as f32 - 128.0) / 128.0,
                        ((k * 31 % 251) as f32 - 125.0) / 125.0,
                    ),
                    _ => {
                        let phase = std::f64::consts::TAU * (k * (n / 4 + 1)) as f64 / n as f64;
                        (phase.cos() as f32, phase.sin() as f32)
                    }
                })
                .collect();
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut re = vec![12345.0; n + 3];
                let mut im = vec![12345.0; n + 5];
                for (k, &(r, i)) in input.iter().enumerate() {
                    re[k + 1] = r;
                    im[k + 3] = i;
                }
                fft_f32_dit_with_planner_and_opts(
                    &mut re[1..n + 1],
                    &mut im[3..n + 3],
                    direction,
                    &planner,
                    &opts,
                );
                let sign = if direction == Direction::Forward {
                    -1.0
                } else {
                    1.0
                };
                let scale = if direction == Direction::Forward {
                    1.0
                } else {
                    1.0 / n as f64
                };
                for k in 0..n {
                    let (mut r, mut i) = (0.0, 0.0);
                    for (j, &(a, b)) in input.iter().enumerate() {
                        let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                        let (s, c) = angle.sin_cos();
                        r += a as f64 * c - b as f64 * s;
                        i += a as f64 * s + b as f64 * c;
                    }
                    let error = (re[k + 1] as f64 - r * scale).hypot(im[k + 3] as f64 - i * scale);
                    assert!(
                        error <= 5e-7 * n as f64 * scale,
                        "n={n}, case={case}, k={k}, {direction:?}, error={error}"
                    );
                }
                assert!(re[..1].iter().chain(&re[n + 1..]).all(|&x| x == 12345.0));
                assert!(im[..3].iter().chain(&im[n + 3..]).all(|&x| x == 12345.0));
            }
        }
    }
    println!("NEON bitwise transpose checks and public-API f32 FFT/DFT checks (1–512, both directions, four inputs, unaligned slices) passed");
}
