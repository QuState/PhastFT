use spinoza::utils::pretty_print_int;
use spinoza::{core::State, math::Float};

use crate::bravo::bravo;
use crate::{bit_rev, bit_reverse_permutation, fft_dif};

pub(crate) fn bm_fft(num_qubits: usize) {
    for i in 4..num_qubits {
        println!("run chunk_n with {i} qubits");
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

fn bm_brp(num_qubits: usize) {
    for i in (2..num_qubits).map(|i| 1 << i) {
        let mut buf0: Vec<Float> = (0..i).map(|i| i as Float).collect();
        let now = std::time::Instant::now();
        bravo(&mut buf0);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("BRAVO: {buf0:?}");
        println!("time elapsed: {elapsed}");

        let mut buf1: Vec<Float> = (0..i).map(|i| i as Float).collect();
        let now = std::time::Instant::now();
        bit_reverse_permutation(&mut buf1);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("naive BR: {buf1:?}");
        println!("time elapsed: {elapsed}\n-----------------------");
        assert_eq!(buf0, buf1);
    }
}

fn benchmark_bit_reversal_permutation() {
    for n in 4..25 {
        let N: usize = 1 << n;
        let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();
        let now = std::time::Instant::now();
        bit_rev(&mut buf, n);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        eprintln!("time elapsed: {elapsed} us");

        let mut buf1: Vec<Float> = (0..N).map(|i| i as Float).collect();

        let now = std::time::Instant::now();
        bit_reverse_permutation(&mut buf1);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        eprintln!("time elapsed: {elapsed} us\n---------------------------");

        // for i in 0..N {
        //     println!("{} {}", buf[i], buf1[i]);
        // }
        assert_eq!(buf, buf1);
    }
}
