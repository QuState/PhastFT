use std::env;
use std::str::FromStr;

use utilities::gen_random_signal;

use phastft::fft_64_with_opts_and_plan;
use phastft::options::Options;
use phastft::planner::{Direction, Planner64};

fn benchmark_fft_64(n: usize) {
    let big_n = 1 << n;
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal(&mut reals, &mut imags);

    let planner = Planner64::new(reals.len(), Direction::Forward);
    let opts = Options::guess_options(reals.len());

    fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &planner);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();

    benchmark_fft_64(n);
}
