use phastft::fft_dif;
use utilities::gen_random_signal;

fn bm_fft(num_qubits: usize) {
    for i in 4..num_qubits {
        println!("run PhastFT with {i} qubits");
        let n = 1 << i;

        let mut reals = vec![0.0; n];
        let mut imags = vec![0.0; n];

        gen_random_signal(&mut reals, &mut imags);

        let now = std::time::Instant::now();
        fft_dif(&mut reals, &mut imags);
        let elapsed = now.elapsed().as_micros();
        println!("time elapsed: {elapsed} us\n----------------------------");
    }
}

fn main() {
    bm_fft(31);
}
