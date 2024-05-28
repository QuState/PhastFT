//! Implementation of Real valued FFT
use std::f64::consts::PI;

use crate::fft_64;
use crate::{compute_twiddle_factors, Direction};

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

/// Implementation of Real-Valued FFT
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

    println!("Z even: {z_even:?}\nZ odd: {z_odd:?}\n");
    println!("Z even min conj : {z_even_min_conj:?}\nZ odd min conj: {z_odd_min_conj:?}");

    /*
    [ 64.       +72.j        -27.3137085+11.3137085j -16.        +0.j
    -11.3137085 -4.6862915j  -8.        -8.j         -4.6862915-11.3137085j
       0.       -16.j         11.3137085-27.3137085j]

    [ 64.       -72.j         11.3137085+27.3137085j   0.       +16.j
      -4.6862915+11.3137085j  -8.        +8.j        -11.3137085 +4.6862915j
     -16.        -0.j        -27.3137085-11.3137085j]

     */

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

    println!("Zx reals: {z_x_re:?}\nZx imags: {z_x_im:?}\n");

    // Zy = -0.5j * (Z - Zminconj)
    let (z_y_re, z_y_im): (Vec<_>, Vec<_>) = z_even
        .iter()
        .zip(z_odd.iter())
        .zip(z_even_min_conj.iter())
        .zip(z_odd_min_conj.iter())
        .map(|(((ze, zo), ze_mc), zo_mc)| {
            let a = ze - ze_mc;
            let b = zo - zo_mc;

            // -0.5i (a + ib) = -0.5i * a + 0.5 * b
            (0.5 * b, -0.5 * a)
        })
        .unzip();

    /*
    Zx:

    [64. +0.j        -8.+19.3137085j -8. +8.j        -8. +3.3137085j
     -8. +0.j        -8. -3.3137085j -8. -8.j        -8.-19.3137085j]

    Zy:

    [72. -0.j        -8.+19.3137085j -8. +8.j        -8. +3.3137085j
     -8. +0.j        -8. -3.3137085j -8. -8.j        -8.-19.3137085j]
     */

    println!("Zy reals: {z_y_re:?}\nZy imags: {z_y_im:?}");

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
    use utilities::assert_float_closeness;

    use super::*;

    #[test]
    fn r2c_vs_c2c() {
        let n = 4;
        let big_n = 1 << 4;
        let mut input_re: Vec<_> = (1..=big_n).map(|i| i as f64).collect(); // Length is 7, which is a prime number

        let (r2c_res_reals, r2c_res_imags) = real_fft(&mut input_re);

        input_re = (1..=big_n).map(|i| i as f64).collect();
        let mut input_im = vec![0.0; input_re.len()]; // Assume the imaginary part is zero for this example
        fft_64(&mut input_re, &mut input_im, Direction::Forward);

        r2c_res_reals
            .iter()
            .zip(r2c_res_imags.iter())
            .zip(input_re.iter())
            .zip(input_im.iter())
            .for_each(|(((a_re, a_im), e_re), e_im)| {
                assert_float_closeness(*a_re, *e_re, 1e-6);
                assert_float_closeness(*a_im, *e_im, 1e-6);
            });
    }
}
