use rustfft::{num_complex::Complex64, FftPlanner};

fn main() {
    let big_n = 31;

    for i in 4..big_n {
        println!("run RustFFT with {i} qubits");
        let n = 1 << i;
        let mut buffer: Vec<Complex64> = (1..n + 1)
            .map(|i| Complex64::new(i as f64, i as f64))
            .collect();

        let now = std::time::Instant::now();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("time elapsed: {elapsed} us\n----------------------------");
    }
}
