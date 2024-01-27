use phastft::fft_dif;

fn bm_fft(num_qubits: usize) {
    for i in 4..num_qubits {
        println!("run PhastFT with {i} qubits");
        let n = 1 << i;
        let mut reals: Vec<f64> = (1..=n).map(f64::from).collect();
        let mut imags: Vec<f64> = (1..=n).map(f64::from).collect();

        let now = std::time::Instant::now();
        fft_dif(&mut reals, &mut imags);
        let elapsed = now.elapsed().as_micros();
        println!("time elapsed: {elapsed} us\n----------------------------");
    }
}

fn main() {
    bm_fft(31);
}
