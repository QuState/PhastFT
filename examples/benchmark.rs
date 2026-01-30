use std::env;
use std::str::FromStr;

use phastft::options::Options;
use phastft::planner::{Direction, Planner64, PlannerDit64};
use phastft::{fft_64_dit_with_planner, fft_64_with_opts_and_plan};
use utilities::gen_random_signal;

fn benchmark_fft_64(n: usize) {
    let big_n = 1 << n; // 2.pow(n)
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal(&mut reals, &mut imags);

    // Pre-create planner and options for DIF
    let planner = Planner64::new(reals.len(), Direction::Forward);
    let opts = Options::guess_options(reals.len());

    let now = std::time::Instant::now();
    fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &planner);
    let elapsed = now.elapsed().as_nanos();
    println!("{elapsed}");
}

fn benchmark_fft_64_dit(n: usize) {
    let big_n = 1 << n; // 2.pow(n)
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal(&mut reals, &mut imags);

    // Pre-create planner for DIT
    let planner = PlannerDit64::new(reals.len(), Direction::Forward);

    let now = std::time::Instant::now();
    fft_64_dit_with_planner(&mut reals, &mut imags, &planner);
    let elapsed = now.elapsed().as_nanos();
    println!("{elapsed}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 2, "Usage {} <n> [--dit]", args[0]);

    let n = usize::from_str(&args[1]).unwrap();
    let use_dit = args.len() > 2 && args[2] == "--dit";

    if use_dit {
        benchmark_fft_64_dit(n);
    } else {
        benchmark_fft_64(n);
    }
}
