//! Compare the four-pass prototype with DIT, excluding setup and input generation.
//! Usage: benchmark_four_pass <32|64> <log2(N)> <iterations> [both|dit|four-pass]
//! Two GiB of complex input: f32 log2(N)=28, f64 log2(N)=27.
//! RAYON_NUM_THREADS controls both implementations. Output is CSV nanoseconds.

#[path = "support/four_pass.rs"]
mod four_pass;

use std::hint::black_box;
use std::time::Instant;

use four_pass::{FourPass32, FourPass64};
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_f32_dit_with_planner, fft_f64_dit_with_planner};
use rand::{rngs::SmallRng, RngExt, SeedableRng};

macro_rules! benchmark {
    ($name:ident, $float:ty, $four:ident, $dit:ident, $fft:ident) => {
        fn $name(n: usize, iterations: usize, mode: &str) {
            let mut re = vec![0.0 as $float; n];
            let mut im = vec![0.0 as $float; n];
            let start = Instant::now();
            let dit = (mode != "four-pass").then(|| $dit::new(n));
            let four = (mode != "dit").then(|| $four::new(n));
            let mut sr = vec![0.0; if four.is_some() { n } else { 0 }];
            let mut si = vec![0.0; sr.len()];
            // Touch scratch before timing so first-write page faults are excluded.
            sr.fill(1.0);
            si.fill(1.0);
            eprintln!(
                "input_bytes={}, points={}, setup_seconds={:.6}",
                n * 2 * std::mem::size_of::<$float>(),
                n,
                start.elapsed().as_secs_f64()
            );
            #[cfg(feature = "parallel")]
            eprintln!("rayon_threads={}", rayon::current_num_threads());

            let sample_count = n.min(4096);
            let mut reference_samples = Vec::new();

            // One warm-up per implementation, then alternate execution order.
            // Reset the same seeded input outside every timed region.
            for iteration in 0..=iterations {
                let order = if iteration % 2 == 0 {
                    ["dit", "four-pass"]
                } else {
                    ["four-pass", "dit"]
                };
                for algorithm in order {
                    if mode != "both" && mode != algorithm {
                        continue;
                    }
                    let mut rng = SmallRng::seed_from_u64(0x4f4654);
                    for (r, i) in re.iter_mut().zip(&mut im) {
                        *r = rng.random::<$float>() - 0.5;
                        *i = rng.random::<$float>() - 0.5;
                    }
                    let start = Instant::now();
                    if algorithm == "dit" {
                        $fft(&mut re, &mut im, Direction::Forward, dit.as_ref().unwrap());
                        black_box((&re, &im));
                    } else {
                        four.as_ref().unwrap().process(
                            &mut re,
                            &mut im,
                            &mut sr,
                            &mut si,
                            Direction::Forward,
                        );
                        black_box((&sr, &si));
                    }
                    let elapsed = start.elapsed().as_nanos();
                    // Compare distributed output bins at the actual benchmark size.
                    // This is outside timing and supplements the full RustFFT tests.
                    if mode == "both" && iteration == 0 {
                        let (out_re, out_im) = if algorithm == "dit" {
                            (&re, &im)
                        } else {
                            (&sr, &si)
                        };
                        let samples: Vec<(f64, f64)> = (0..sample_count)
                            .map(|j| {
                                // Include both endpoints, avoiding only sampling
                                // power-of-two-aligned frequency bins.
                                let k = j * (n - 1) / (sample_count - 1).max(1);
                                (out_re[k] as f64, out_im[k] as f64)
                            })
                            .collect();
                        if algorithm == "dit" {
                            reference_samples = samples;
                        } else {
                            let mut error = 0.0;
                            let mut energy = 0.0;
                            for ((ar, ai), (br, bi)) in samples.iter().zip(&reference_samples) {
                                error += (ar - br).powi(2) + (ai - bi).powi(2);
                                energy += br * br + bi * bi;
                            }
                            let relative_error = (error / energy).sqrt();
                            eprintln!("sampled_bins={sample_count}, relative_l2_vs_dit={relative_error:.6e}");
                            let tolerance = if std::mem::size_of::<$float>() == 4 {
                                2e-6
                            } else {
                                2e-12
                            };
                            assert!(relative_error < tolerance, "four-pass disagrees with DIT");
                        }
                    }
                    if iteration != 0 {
                        println!("{algorithm},{iteration},{elapsed}");
                    }
                }
            }
        }
    };
}

benchmark!(
    bench32,
    f32,
    FourPass32,
    PlannerDit32,
    fft_f32_dit_with_planner
);
benchmark!(
    bench64,
    f64,
    FourPass64,
    PlannerDit64,
    fft_f64_dit_with_planner
);

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (4..=5).contains(&args.len()),
        "Usage: {} <32|64> <log2(N)> <iterations> [both|dit|four-pass]",
        args[0]
    );
    let log_n: u32 = args[2].parse().expect("invalid exponent");
    let n = 1usize.checked_shl(log_n).expect("exponent too large");
    let iterations: usize = args[3].parse().expect("invalid iterations");
    assert!(iterations > 0);
    let mode = args.get(4).map(String::as_str).unwrap_or("both");
    assert!(["both", "dit", "four-pass"].contains(&mode));
    println!("algorithm,iteration,nanoseconds");
    match args[1].as_str() {
        "32" => bench32(n, iterations, mode),
        "64" => bench64(n, iterations, mode),
        _ => panic!("precision must be 32 or 64"),
    }
}
