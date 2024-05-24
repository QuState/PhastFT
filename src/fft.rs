//! Implementation of Real valued FFT
use std::f64::consts::PI;

use crate::{compute_twiddle_factors, Direction};
use crate::fft_64;

fn precompute_chirp(n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut chirp_re = vec![0.0; m];
    let mut chirp_im = vec![0.0; m];
    for k in 0..n {
        let angle = PI * (k * k) as f64 / n as f64;
        chirp_re[k] = angle.cos();
        chirp_im[k] = -angle.sin();
    }
    for k in n..m {
        chirp_re[k] = 0.0;
        chirp_im[k] = 0.0;
    }
    (chirp_re, chirp_im)
}

pub fn real_fft(input_re: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = input_re.len();

    // Splitting odd and even
    let (mut z_even, mut z_odd): (Vec<_>, Vec<_>) =
        input_re.chunks_exact(2).map(|c| (c[0], c[1])).unzip();

    // Z = np.fft.fft(z)
    fft_64(&mut z_even, &mut z_odd, Direction::Forward);

    // take care of the np.flip()
    let mut z_even_min_conj: Vec<_> = z_even.iter().copied().rev().collect();

    let mut z_odd_min_conj: Vec<_> = z_odd.iter().copied().rev().collect();

    // Zminconj = np.roll(np.flip(Z), 1).conj()
    // now roll both by 1
    z_even_min_conj.rotate_right(1);
    z_odd_min_conj.rotate_right(1);

    // the conj() call can be resolved by negating every imaginary component
    for zo in z_odd_min_conj.iter_mut() {
        *zo = -(*zo);
    }

    // Zx = 0.5  * (Z + Zminconj)
    let (z_x_re, z_x_im): (Vec<_>, Vec<_>) = z_even
        .iter()
        .zip(z_odd.iter())
        .zip(z_even_min_conj.iter())
        .zip(z_odd_min_conj.iter())
        .map(|(((ze, zo), ze_mc), zo_mc)| {
            let a = 0.5 * (ze + ze_mc);
            let b = 0.5 * (zo + zo_mc);
            (a, b)
        })
        .unzip();

    // Zy = -0.5j * (Z - Zminconj)
    let (z_y_re, z_y_im): (Vec<_>, Vec<_>) = z_even
        .iter()
        .zip(z_odd.iter())
        .zip(z_even_min_conj.iter())
        .zip(z_odd_min_conj.iter())
        .map(|(((ze, zo), ze_mc), zo_mc)| {
            let a = ze - ze_mc;
            let b = zo - zo_mc;

            // 0.5i (a + ib) = 0.5i * a - 0.5 * b
            (-0.5 * b, 0.5 * a)
        })
        .unzip();

    let (twiddle_re, twiddle_im) = compute_twiddle_factors(n); // np.exp(-1j * 2 * math.pi * np.arange(N//2) / N)

    // Zall = np.concatenate([Zx + W*Zy, Zx - W*Zy])
    let mut z_all_re = Vec::new();
    let mut z_all_im = Vec::new();

    for i in 0..n / 2 {
        let zx_re = z_x_re[i];
        let zx_im = z_x_im[i];
        let zy_re = z_y_re[i];
        let zy_im = z_y_im[i];
        let w_re = twiddle_re[i];
        let w_im = twiddle_im[i];

        let wz_re = w_re * zy_re - w_im * zy_im;
        let wz_im = w_re * zy_im + w_im * zy_re;

        // Zx + W * Zy
        z_all_re.push(zx_re + wz_re);
        z_all_im.push(zx_im + wz_im);
    }

    for i in 0..n / 2 {
        let zx_re = z_x_re[i];
        let zx_im = z_x_im[i];
        let zy_re = z_y_re[i];
        let zy_im = z_y_im[i];
        let w_re = twiddle_re[i];
        let w_im = twiddle_im[i];

        let wz_re = w_re * zy_re - w_im * zy_im;
        let wz_im = w_re * zy_im + w_im * zy_re;

        // Zx - W * Zy
        z_all_re.push(zx_re - wz_re);
        z_all_im.push(zx_im - wz_im);
    }

    // return Zall
    (z_all_re, z_all_im)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2c_vs_c2c() {
        let mut input_re: Vec<_> = (1..=16).map(|i| i as f64).collect(); // Length is 7, which is a prime number
        let mut input_im = vec![0.0; input_re.len()]; // Assume the imaginary part is zero for this example

        let (re, im) = real_fft(&mut input_re);
        println!("actual:\n{:?}\n{:?}\n", re, im);

        input_re = (1..=16).map(|i| i as f64).collect();
        input_im = vec![0.0; input_re.len()]; // Assume the imaginary part is zero for this example
        fft_64(&mut input_re, &mut input_im, Direction::Forward);
        println!("expected:\n{:?}\n{:?}\n", input_re, input_im);
    }
}
