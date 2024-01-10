use rustfft::num_complex::Complex;
use rustfft::{num_complex::Complex64, FftPlanner};
use spinoza::utils::pretty_print_int;

// Perform the combine step in-place
// This saves on use having to allocate more memory
// Takes in pre-calculated z values
fn combine(zs: &[Complex<f64>], input: &mut [Complex<f64>]) {
    let n = input.len();
    let z_step = zs.len() / n;

    for k in 0..n / 2 {
        let z = zs[k * z_step];

        let e = input[k];
        let o = input[k + n / 2];
        input[k] = e + z * o;
        input[k + n / 2] = e - z * o;
    }
}

pub fn fft_c(mut input: Vec<Complex<f64>>) {
    let n = input.len();

    // For 64bit usize and n = 16 (4 bits),
    // Since reverse_bits reverses the entire value,
    // we need to shift back 60 spaces
    // If n = 16, n-1 = 15, which has 60 leading zeros.
    let shift = (n - 1).leading_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            input.swap(i, j);
        }
    }

    let step = -2.0 * std::f64::consts::PI / (n as f64);
    let zs: Vec<_> = (0..n / 2)
        .map(|k| Complex::from_polar(1.0, (k as f64) * step))
        .collect();

    let mut m = 2;
    while m <= n {
        for s in (0..n).step_by(m) {
            combine(&zs, &mut input[s..s + m]);
        }
        m *= 2;
    }
}

fn main() {
    let N = 30;

    for i in 2..N {
        println!("run RustFFT with {i} qubits");
        let n = 1 << i;
        let now = std::time::Instant::now();
        let mut buffer: Vec<Complex64> = (1..n + 1)
            .map(|i| Complex64::new(i as f64, i as f64))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        eprintln!("time elapsed: {elapsed} us");
        // println!("{:?}", buffer);

        eprintln!("run C's FFT with {i} qubits");
        let n = 1 << i;
        let now = std::time::Instant::now();
        let buffer: Vec<Complex64> = (1..n + 1)
            .map(|i| Complex64::new(i as f64, i as f64))
            .collect();

        fft_c(buffer);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        eprintln!("time elapsed: {elapsed} us\n----------------------------");
    }
}
