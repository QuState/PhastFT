use std::env;
use std::str::FromStr;

use utilities::rustfft::num_complex::{Complex32, Complex64};
use utilities::rustfft::num_traits::Zero;
use utilities::rustfft::FftPlanner;
use utilities::{gen_random_signal_f32, gen_random_signal_f64};

// See examples/benchmark.rs for the rationale behind batch-timing.
fn batch_size(n: usize) -> usize {
    match n {
        0..=7 => 256,
        8..=9 => 32,
        _ => 1,
    }
}

fn bench32(n: usize, iterations: usize) {
    let big_n = 1usize << n;

    let mut reals = vec![0.0f32; big_n];
    let mut imags = vec![0.0f32; big_n];
    let mut signal = vec![Complex32::default(); big_n];

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(big_n);
    let mut scratch = vec![Complex32::zero(); fft.get_inplace_scratch_len()];

    let batch = batch_size(n).min(iterations.max(1));
    let batches = (iterations / batch).max(1);
    for _ in 0..batches {
        gen_random_signal_f32(&mut reals, &mut imags);
        signal
            .iter_mut()
            .zip(reals.iter())
            .zip(imags.iter())
            .for_each(|((z, &re), &im)| {
                z.re = re;
                z.im = im;
            });

        let now = std::time::Instant::now();
        for _ in 0..batch {
            fft.process_with_scratch(&mut signal, scratch.as_mut_slice());
            std::hint::black_box(&mut signal);
        }
        let elapsed = now.elapsed().as_nanos();
        println!("{}", elapsed / batch as u128);
    }
}

fn bench64(n: usize, iterations: usize) {
    let big_n = 1usize << n;

    let mut reals = vec![0.0f64; big_n];
    let mut imags = vec![0.0f64; big_n];
    let mut signal = vec![Complex64::default(); big_n];

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(big_n);
    let mut scratch = vec![Complex64::zero(); fft.get_inplace_scratch_len()];

    let batch = batch_size(n).min(iterations.max(1));
    let batches = (iterations / batch).max(1);
    for _ in 0..batches {
        gen_random_signal_f64(&mut reals, &mut imags);
        signal
            .iter_mut()
            .zip(reals.iter())
            .zip(imags.iter())
            .for_each(|((z, &re), &im)| {
                z.re = re;
                z.im = im;
            });

        let now = std::time::Instant::now();
        for _ in 0..batch {
            fft.process_with_scratch(&mut signal, scratch.as_mut_slice());
            std::hint::black_box(&mut signal);
        }
        let elapsed = now.elapsed().as_nanos();
        println!("{}", elapsed / batch as u128);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() >= 4,
        "Usage: {} <32|64> <n> <iterations>",
        args[0]
    );

    let n = usize::from_str(&args[2]).unwrap();
    let iterations = usize::from_str(&args[3]).unwrap();

    match args[1].as_str() {
        "32" => bench32(n, iterations),
        "64" => bench64(n, iterations),
        other => panic!("Invalid precision: {other}. Use 32 or 64"),
    }
}
