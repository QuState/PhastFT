//! This module provides several implementations of the bit reverse permutation, which is
//! essential for algorithms like FFT.
//!
//! In practice, most FFT implementations avoid bit reversals; however this comes at a computational
//! cost as well. For example, Bailey's 4 step FFT algorithm is O(N * lg(N) * lg(lg(N))).
//! The original Cooley-Tukey implementation is O(N * lg(N)). The extra term in the 4-step algorithm
//! comes from incorporating the bit reversals into each level of the recursion. By utilizing a
//! cache-optimal bit reversal, we are able to avoid this extra cost [1].
//!
//! # References
//!
//! [1] L. Carter and K. S. Gatlin, "Towards an optimal bit-reversal permutation program," Proceedings 39th Annual
//! Symposium on Foundations of Computer Science (Cat. No.98CB36280), Palo Alto, CA, USA, 1998, pp. 544-553, doi:
//! 10.1109/SFCS.1998.743505.
//! keywords: {Read-write memory;Costs;Computer science;Drives;Random access memory;Argon;Registers;Read only memory;Computational modeling;Libraries}

use num_traits::Float;

const BLOCK_WIDTH: usize = 128;
// size of the cacheline
const LOG_BLOCK_WIDTH: usize = 7; // log2 of cacheline

/// In-place bit reversal on a single buffer. Also referred to as "Jennifer's method" [1].
///
/// ## References
/// [1] <https://www.katjaas.nl/bitreversal/bitreversal.html>
#[inline]
pub(crate) fn bit_rev<T>(buf: &mut [T], log_n: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let big_n = 1 << log_n;
    let halfn = big_n >> 1; // frequently used 'constants'
    let quartn = big_n >> 2;
    let nmin1 = big_n - 1;

    let mut forward = halfn; // variable initialisations
    let mut rev = 1;

    let mut i = quartn;
    while i > 0 {
        // start of bit reversed permutation loop, N/4 iterations

        // Gray code generator for even values:

        nodd = !i; // counting ones is easier

        let mut zeros = 0;
        while (nodd & 1) == 1 {
            nodd >>= 1;
            zeros += 1;
        }

        forward ^= 2 << zeros; // toggle one bit of forward
        rev ^= quartn >> zeros; // toggle one bit of rev

        // swap even and ~even conditionally
        if forward < rev {
            buf.swap(forward, rev);
            nodd = nmin1 ^ forward; // compute the bitwise negations
            noddrev = nmin1 ^ rev;
            buf.swap(nodd, noddrev); // swap bitwise-negated pairs
        }

        nodd = forward ^ 1; // compute the odd values from the even
        noddrev = rev ^ halfn;

        // swap odd unconditionally
        buf.swap(nodd, noddrev);
        i -= 1;
    }
}

/// Run in-place bit reversal on the entire state (i.e., the reals and imags buffers)
///
/// ## References
/// [1] <https://www.katjaas.nl/bitreversal/bitreversal.html>
#[allow(unused)]
#[deprecated(
    since = "0.1.0",
    note = "Please use COBRA for a cache-optimal bit reverse permutation."
)]
fn complex_bit_rev<T: Float>(reals: &mut [T], imags: &mut [T], log_n: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let big_n = 1 << log_n;
    let halfn = big_n >> 1; // frequently used 'constants'
    let quartn = big_n >> 2;
    let nmin1 = big_n - 1;

    let mut forward = halfn; // variable initialisations
    let mut rev = 1;

    let mut i = quartn;
    while i > 0 {
        // start of bit-reversed permutation loop, N/4 iterations

        // Gray code generator for even values:

        nodd = !i; // counting ones is easier

        let mut zeros = 0;
        while (nodd & 1) == 1 {
            nodd >>= 1;
            zeros += 1;
        }

        forward ^= 2 << zeros; // toggle one bit of forward
        rev ^= quartn >> zeros; // toggle one bit of rev

        // swap even and ~even conditionally
        if forward < rev {
            reals.swap(forward, rev);
            imags.swap(forward, rev);
            nodd = nmin1 ^ forward; // compute the bitwise negations
            noddrev = nmin1 ^ rev;

            // swap bitwise-negated pairs
            reals.swap(nodd, noddrev);
            imags.swap(nodd, noddrev);
        }

        nodd = forward ^ 1; // compute the odd values from the even
        noddrev = rev ^ halfn;

        // swap odd unconditionally
        reals.swap(nodd, noddrev);
        imags.swap(nodd, noddrev);
        i -= 1;
    }
}

#[allow(dead_code)]
#[deprecated(
    since = "0.1.0",
    note = "Naive bit reverse permutation is slow and not cache friendly. COBRA should be used instead."
)]
pub(crate) fn bit_reverse_permutation<T>(buf: &mut [T]) {
    let n = buf.len();
    let mut j = 0;

    for i in 1..n {
        let mut bit = n >> 1;

        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            buf.swap(i, j);
        }
    }
}

/// Pure Rust implementation of Cache Optimal Bit-Reverse Algorithm (COBRA).
/// Rewritten from a C++ implementation [3].
///
/// ## References
/// [1] L. Carter and K. S. Gatlin, "Towards an optimal bit-reversal permutation program," Proceedings 39th Annual
/// Symposium on Foundations of Computer Science (Cat. No.98CB36280), Palo Alto, CA, USA, 1998, pp. 544-553, doi:
/// 10.1109/SFCS.1998.743505.
/// [2] Christian Knauth, Boran Adas, Daniel Whitfield, Xuesong Wang, Lydia Ickler, Tim Conrad, Oliver Serang:
/// Practically efficient methods for performing bit-reversed permutation in C++11 on the x86-64 architecture
/// [3] <https://bitbucket.org/orserang/bit-reversal-methods/src/master/src_and_bin/src/algorithms/COBRAShuffle.hpp>
#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
))]
pub fn cobra_apply<T: Default + Copy + Clone>(v: &mut [T], log_n: usize) {
    if log_n <= 2 * LOG_BLOCK_WIDTH {
        bit_rev(v, log_n);
        return;
    }
    let num_b_bits = log_n - 2 * LOG_BLOCK_WIDTH;
    let b_size: usize = 1 << num_b_bits;
    let block_width: usize = 1 << LOG_BLOCK_WIDTH;

    let mut buffer = [T::default(); BLOCK_WIDTH * BLOCK_WIDTH];

    for b in 0..b_size {
        let b_rev = b.reverse_bits() >> ((b_size - 1).leading_zeros());

        // Copy block to buffer
        for a in 0..block_width {
            let a_rev = a.reverse_bits() >> ((block_width - 1).leading_zeros());
            for c in 0..BLOCK_WIDTH {
                buffer[(a_rev << LOG_BLOCK_WIDTH) | c] =
                    v[(a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c];
            }
        }

        for c in 0..BLOCK_WIDTH {
            // NOTE: Typo in original pseudocode by Carter and Gatlin at the following line:
            let c_rev = c.reverse_bits() >> ((block_width - 1).leading_zeros());

            for a_rev in 0..BLOCK_WIDTH {
                let a = a_rev.reverse_bits() >> ((block_width - 1).leading_zeros());

                // To guarantee each value is swapped only one time:
                // index < reversed_index <-->
                // a b c < c' b' a' <-->
                // a < c' ||
                // a <= c' && b < b' ||
                // a <= c' && b <= b' && a' < c
                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    let v_idx = (c_rev << num_b_bits << LOG_BLOCK_WIDTH)
                        | (b_rev << LOG_BLOCK_WIDTH)
                        | a_rev;
                    let b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;
                    std::mem::swap(&mut v[v_idx], &mut buffer[b_idx]);
                }
            }
        }

        // Copy changes that were swapped into buffer above:
        for a in 0..BLOCK_WIDTH {
            let a_rev = a.reverse_bits() >> ((block_width - 1).leading_zeros());
            for c in 0..BLOCK_WIDTH {
                let c_rev = c.reverse_bits() >> ((block_width - 1).leading_zeros());
                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    let v_idx = (a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c;
                    let b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;
                    std::mem::swap(&mut v[v_idx], &mut buffer[b_idx]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Top down bit reverse interleaving. This is a very simple and well known approach that we only use for testing
    /// COBRA and any other bit reverse algorithms.
    fn top_down_bit_reverse_permutation<T: Copy + Clone>(x: &[T]) -> Vec<T> {
        if x.len() == 1 {
            return x.to_vec();
        }

        let mut y = Vec::with_capacity(x.len());
        let mut evens = Vec::with_capacity(x.len() >> 1);
        let mut odds = Vec::with_capacity(x.len() >> 1);

        let mut i = 1;
        while i < x.len() {
            evens.push(x[i - 1]);
            odds.push(x[i]);
            i += 2;
        }

        y.extend_from_slice(&top_down_bit_reverse_permutation(&evens));
        y.extend_from_slice(&top_down_bit_reverse_permutation(&odds));
        y
    }

    #[test]
    fn cobra() {
        for n in 4..23 {
            let big_n = 1 << n;
            let mut v: Vec<_> = (0..big_n).collect();
            cobra_apply(&mut v, n);

            let x: Vec<_> = (0..big_n).collect();
            assert_eq!(v, top_down_bit_reverse_permutation(&x));
        }
    }

    #[test]
    fn bit_reversal() {
        let n = 3;
        let big_n = 1 << n;
        let mut buf: Vec<f64> = (0..big_n).map(f64::from).collect();
        bit_rev(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(buf, vec![0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]);

        let n = 4;
        let big_n = 1 << n;
        let mut buf: Vec<f64> = (0..big_n).map(f64::from).collect();
        bit_rev(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(
            buf,
            vec![
                0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0,
                15.0,
            ]
        );
    }

    #[test]
    fn jennifer_method() {
        for n in 2..24 {
            let big_n = 1 << n;
            let mut actual_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let mut actual_im: Vec<f64> = (0..big_n).map(f64::from).collect();

            #[allow(deprecated)]
            complex_bit_rev(&mut actual_re, &mut actual_im, n);

            let input_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_re = top_down_bit_reverse_permutation(&input_re);
            assert_eq!(actual_re, expected_re);

            let input_im: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_im = top_down_bit_reverse_permutation(&input_im);
            assert_eq!(actual_im, expected_im);
        }
    }

    #[test]
    fn naive_bit_reverse_permutation() {
        for n in 2..24 {
            let big_n = 1 << n;
            let mut actual_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let mut actual_im: Vec<f64> = (0..big_n).map(f64::from).collect();

            #[allow(deprecated)]
            bit_reverse_permutation(&mut actual_re);

            #[allow(deprecated)]
            bit_reverse_permutation(&mut actual_im);

            let input_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_re = top_down_bit_reverse_permutation(&input_re);
            assert_eq!(actual_re, expected_re);

            let input_im: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_im = top_down_bit_reverse_permutation(&input_im);
            assert_eq!(actual_im, expected_im);
        }
    }
}
