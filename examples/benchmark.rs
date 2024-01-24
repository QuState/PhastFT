use spinoza::utils::pretty_print_int;
use spinoza::{core::State, math::Float};

use phastft::fft_dif;

fn bm_fft(num_qubits: usize) {
    for i in 4..num_qubits {
        println!("run PhastFT with {i} qubits");
        let n = 1 << i;
        let x_re: Vec<Float> = (1..=n).map(|i| i as Float).collect();
        let x_im: Vec<Float> = (1..=n).map(|i| i as Float).collect();
        let mut state = State {
            reals: x_re,
            imags: x_im,
            n: i as u8,
        };

        let now = std::time::Instant::now();
        fft_dif(&mut state);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("time elapsed: {elapsed} us\n----------------------------");
    }
}

fn main() {
    bm_fft(31);
}
