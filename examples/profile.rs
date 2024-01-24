use spinoza::{core::State, math::Float};

use phastft::fft_dif;

fn bm_fft(num_qubits: usize) {
    let n = 1 << num_qubits;
    let x_re: Vec<Float> = (1..=n).map(|i| i as Float).collect();
    let x_im: Vec<Float> = (1..=n).map(|i| i as Float).collect();
    let mut state = State {
        reals: x_re,
        imags: x_im,
        n: num_qubits as u8,
    };

    fft_dif(&mut state);
}

fn main() {
    bm_fft(25);
}
