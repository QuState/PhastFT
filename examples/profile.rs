use phastft::fft_dif;

fn bm_fft(num_qubits: usize) {
    let n = 1 << num_qubits;
    let mut reals: Vec<f64> = (1..=n).map(f64::from).collect();
    let mut imags: Vec<f64> = (1..=n).map(f64::from).collect();

    fft_dif(&mut reals, &mut imags);
}

fn main() {
    bm_fft(25);
}
