use std::env;
use std::str::FromStr;

use utilities::gen_random_signal;

use phastft::fft_64;
use phastft::planner::Direction;

fn benchmark_fft_64(n: usize) {
    let big_n = 1 << n;
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal(&mut reals, &mut imags);

    let now = std::time::Instant::now();
    fft_64(&mut reals, &mut imags, Direction::Forward);
    let elapsed = now.elapsed().as_micros();
    println!("{elapsed}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();

    benchmark_fft_64(n);
}
