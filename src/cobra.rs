use spinoza::core::State;

const BLOCK_WIDTH: usize = 128;
// size of the cacheline
const LOG_BLOCK_WIDTH: usize = 7; // log2 of cacheline

/// Run in-place bit reversal on a single buffer
/// Source: https://www.katjaas.nl/bitreversal/bitreversal.html
pub(crate) fn bit_rev<T>(buf: &mut [T], log_n: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let N = 1 << log_n;
    let halfn = N >> 1; // frequently used 'constants'
    let quartn = N >> 2;
    let nmin1 = N - 1;

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
/// Source: https://www.katjaas.nl/bitreversal/bitreversal.html
fn complex_bit_rev(state: &mut State, log_n: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let N = 1 << log_n;
    let halfn = N >> 1; // frequently used 'constants'
    let quartn = N >> 2;
    let nmin1 = N - 1;

    let mut forward = halfn; // variable initialisations
    let mut rev = 1;

    let mut i = quartn;
    while i > 0 {
        // start of bitreversed permutation loop, N/4 iterations

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
            state.reals.swap(forward, rev);
            state.imags.swap(forward, rev);
            nodd = nmin1 ^ forward; // compute the bitwise negations
            noddrev = nmin1 ^ rev;

            // swap bitwise-negated pairs
            state.reals.swap(nodd, noddrev);
            state.imags.swap(nodd, noddrev);
        }

        nodd = forward ^ 1; // compute the odd values from the even
        noddrev = rev ^ halfn;

        // swap odd unconditionally
        state.reals.swap(nodd, noddrev);
        state.imags.swap(nodd, noddrev);
        i -= 1;
    }
}

/// Run in-place bit reversal on the entire state, in parallel.
/// This function uses 2 threads to run bit reverse on the reals buffer on one thread, and the other thread handles bit
/// reversing the imaginaries buffer
pub(crate) fn bit_reverse_permute_state_par(state: &mut State) {
    std::thread::scope(|s| {
        s.spawn(|| bit_rev(&mut state.reals, state.n as usize));
        s.spawn(|| bit_rev(&mut state.imags, state.n as usize));
    });
}

/// Run in-place bit reversal on the entire state, serially.
pub(crate) fn bit_reverse_permute_state_seq(state: &mut State) {
    complex_bit_rev(state, state.n as usize);
}

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

/// Pure Rust implementation of Cache Optimal BitReverse Algorithm (COBRA).
/// Rewritten from a C++ implementation, which is available here:
/// https://bitbucket.org/orserang/bit-reversal-methods/src/master/src_and_bin/src/algorithms/COBRAShuffle.hpp
///
/// References
/// [1] L. Carter and K. S. Gatlin, "Towards an optimal bit-reversal permutation program," Proceedings 39th Annual
/// Symposium on Foundations of Computer Science (Cat. No.98CB36280), Palo Alto, CA, USA, 1998, pp. 544-553, doi:
/// 10.1109/SFCS.1998.743505.
/// [2] Christian Knauth, Boran Adas, Daniel Whitfield, Xuesong Wang, Lydia Ickler, Tim Conrad, Oliver Serang:
/// Practically efficient methods for performing bit-reversed permutation in C++11 on the x86-64 architecture
pub(crate) fn cobra_apply<T: Default + Copy + Clone>(v: &mut [T], log_n: usize) {
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
    use spinoza::math::Float;

    use super::*;

    /// Top down bit reverse interleaving. This is a very simple and well known approach that we only use for testing
    /// COBRA and any other bit reverse algorithms.
    fn top_down_bril<T: Copy + Clone>(x: &[T]) -> Vec<T> {
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

        y.extend_from_slice(&top_down_bril(&evens));
        y.extend_from_slice(&top_down_bril(&odds));
        y
    }

    #[test]
    fn cobra() {
        for n in 4..23 {
            let N = 1 << n;
            let mut v: Vec<_> = (0..N).collect();
            cobra_apply(&mut v, n);

            let x: Vec<_> = (0..N).collect();
            assert_eq!(v, top_down_bril(&x));
        }
    }

    #[test]
    fn bit_reversal() {
        let n = 3;
        let N = 1 << n;
        let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();
        bit_rev(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(buf, vec![0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]);

        let n = 4;
        let N = 1 << n;
        let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();
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
}