//! Deprecated/legacy bit-reversal implementations, recovered from the (now removed)
//! `src/algorithms/cobra.rs` (`4d5c6d6`), and live here purely as benchmark reference points for
//! `benches/bit_reversal.rs`.
//!
//! These are neither part of PhastFT's public API, nor its default algorithm for bit reversal.
//! Rather, they exist only so the benchmark of the state-of-the-art algorithm can be compared
//! against them. The bodies are recovered verbatim from `4d5c6d6`.

const BLOCK_WIDTH: usize = 128; // size of the cache-resident tile (elements)
const LOG_BLOCK_WIDTH: usize = 7; // log2 of BLOCK_WIDTH

/// In-place, single-buffer bit reversal invented by Jennifer E. Elaan [1]: a Gray-code index
/// generator (advancing an index and its bit-reversed mirror in lockstep, so reversed indices are
/// never recomputed) combined with a 4-way symmetry fold — complement `reverse(!x) == !reverse(x)`
/// and odd/even `reverse(x | 1) == reverse(x) | N/2` — permuting the whole array in N/4 iterations.
///
/// ## References
/// [1] <https://www.katjaas.nl/bitreversal/bitreversal.html>
#[inline]
pub fn bit_rev<T>(buf: &mut [T], log_n: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let big_n = 1 << log_n;
    let halfn = big_n >> 1;
    let quartn = big_n >> 2;
    let nmin1 = big_n - 1;

    let mut forward = halfn;
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

/// Naive textbook bit-reversal permutation: walk indices while maintaining the
/// bit-reversed counter `j`, swapping `i`/`j` when `i < j`. Not cache friendly;
/// kept only as a benchmark baseline.
pub fn bit_reverse_permutation<T>(buf: &mut [T]) {
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

/// Pure Rust implementation of the Cache Optimal Bit-Reverse Algorithm (COBRA).
/// Rewritten from a C++ implementation [3].
///
/// Falls back to [`bit_rev`] (Elaan's method) when the whole array fits in a
/// single cache-resident tile (`log_n <= 2 * LOG_BLOCK_WIDTH`), where the
/// blocking machinery would only add overhead.
///
/// ## References
/// [1] L. Carter and K. S. Gatlin, "Towards an optimal bit-reversal permutation program," Proceedings 39th Annual
/// Symposium on Foundations of Computer Science (Cat. No.98CB36280), Palo Alto, CA, USA, 1998, pp. 544-553, doi:
/// 10.1109/SFCS.1998.743505.
/// [2] Christian Knauth, Boran Adas, Daniel Whitfield, Xuesong Wang, Lydia Ickler, Tim Conrad, Oliver Serang:
/// Practically efficient methods for performing bit-reversed permutation in C++11 on the x86-64 architecture
/// [3] <https://bitbucket.org/orserang/bit-reversal-methods/src/master/src_and_bin/src/algorithms/COBRAShuffle.hpp>
#[allow(clippy::needless_range_loop)]
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
