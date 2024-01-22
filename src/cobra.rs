pub(crate) const BLOCK_WIDTH: usize = 64;
pub(crate) const LOG_BLOCK_WIDTH: usize = 6;

pub(crate) fn cobra_apply<T: Default + Copy + Clone>(v: &mut [T], log_n: usize) {
    assert!(log_n > 2 * LOG_BLOCK_WIDTH);
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
        // Swap v[rev_index] with buffer:
        for c in 0..BLOCK_WIDTH {
            // Note: Typo in original pseudocode by Carter and Gatlin at
            // the following line:
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
            let a_rev = a.reverse_bits() >> ((block_width - 1).leading_zeros()); // BitReversal::reverse_bytewise(a as u64, LOG_BLOCK_WIDTH);
            for c in 0..BLOCK_WIDTH {
                let c_rev = c.reverse_bits() >> ((block_width - 1).leading_zeros()); // BitReversal::reverse_bytewise(c as u64, LOG_BLOCK_WIDTH);
                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    // std::swap(v[ ], buffer[ ]);
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
    use crate::cobra::cobra_apply;

    fn top_down_bril<T: Clone>(x: &[T]) -> Vec<T> {
        if x.len() == 1 {
            return x.to_vec();
        }

        let mut y = Vec::new();
        let evens: Vec<T> = x
            .iter()
            .enumerate()
            .filter(|(i, _)| i & 1 == 0)
            .map(|(_, val)| val.clone())
            .collect();
        let odds: Vec<T> = x
            .iter()
            .enumerate()
            .filter(|(i, _)| i & 1 == 1)
            .map(|(_, val)| val.clone())
            .collect();

        y.extend_from_slice(&top_down_bril(&evens));
        y.extend_from_slice(&top_down_bril(&odds));
        y
    }

    #[test]
    fn cobra() {
        for n in 13..23 {
            let N = 1 << n;
            let mut v: Vec<_> = (0..N).collect();
            cobra_apply(&mut v, n);
            //println!("{v:?}");

            let x: Vec<_> = (0..N).collect();
            let y = top_down_bril(&x);
            // println!("{y:?}");

            assert_eq!(v, y);
        }
    }
}
