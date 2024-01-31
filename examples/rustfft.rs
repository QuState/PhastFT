use utilities::{
    gen_random_signal,
    rustfft::{num_complex::Complex64, FftPlanner},
};

fn main() {
    let big_n = 31;

    for i in 4..big_n {
        println!("run RustFFT with {i} qubits");
        let n = 1 << i;

        let mut reals = vec![0.0; n];
        let mut imags = vec![0.0; n];

        gen_random_signal(&mut reals, &mut imags);
        let mut signal = vec![Complex64::default(); n];
        reals
            .drain(..)
            .zip(imags.drain(..))
            .zip(signal.iter_mut())
            .for_each(|((re, im), z)| {
                z.re = re;
                z.im = im;
            });

        let now = std::time::Instant::now();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(signal.len());
        fft.process(&mut signal);
        let elapsed = now.elapsed().as_micros();
        println!("time elapsed: {elapsed} us\n----------------------------");
    }
}
