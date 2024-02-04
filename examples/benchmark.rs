use std::env;
use std::str::FromStr;

use phastft::fft;
use phastft::planner::Planner;
use utilities::gen_random_signal;

fn benchmark_fft(n: usize) {
    let big_n = 1 << n;
    let mut reals = vec![0.0; big_n];
    let mut imags = vec![0.0; big_n];
    gen_random_signal(&mut reals, &mut imags);

    let now = std::time::Instant::now();
    let mut planner = Planner::new(n);
    fft(&mut reals, &mut imags, &mut planner);
    let elapsed = now.elapsed().as_micros();
    println!("{elapsed}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Usage {} <n>", args[0]);

    let n = usize::from_str(&args[1]).unwrap();

    benchmark_fft(n);
}
