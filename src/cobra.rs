fn reverse(x: usize, lgn: usize) -> usize {
    if lgn == 0 {
        return x;
    }
    let shift = std::mem::size_of::<usize>() * 8 - lgn;
    x.reverse_bits() >> shift
}

fn COBRAShuffle(v: &mut [f64]) {
    let n = v.len();
    let LOG_N = n.ilog2() as usize;
    const CACHELINE_SIZE: usize = 64;
    let LOG_BLOCK_WIDTH = CACHELINE_SIZE.ilog2() as usize;
    let NUM_B_BITS: usize = LOG_N - 2 * LOG_BLOCK_WIDTH;
    let B_SIZE: usize = 1 << NUM_B_BITS;
    let BLOCK_WIDTH: usize = 1 << LOG_BLOCK_WIDTH;

    let mut buffer = vec![0.0; BLOCK_WIDTH * BLOCK_WIDTH];

    for b in 0..B_SIZE {
        let shift = std::mem::size_of::<usize>() * 8 - NUM_B_BITS;
        let b_rev = reverse(b, NUM_B_BITS); // BitReversal::<NUM_B_BITS>::reverse_bytewise(b);

        // Copy block to buffer:
        for a in 0..BLOCK_WIDTH {
            let a_rev = reverse(a, LOG_BLOCK_WIDTH); // BitReversal::<LOG_BLOCK_WIDTH>::reverse_bytewise(a);

            for c in 0..BLOCK_WIDTH {
                buffer[a_rev << LOG_BLOCK_WIDTH | c] =
                    v[a << NUM_B_BITS << LOG_BLOCK_WIDTH | b << LOG_BLOCK_WIDTH | c];
            }
        }

        // Swap v[rev_index] with buffer:
        for c in 0..BLOCK_WIDTH {
            let c_rev = reverse(c, LOG_N); // BitReversal::<LOG_BLOCK_WIDTH>::reverse_bytewise(c);

            for a_rev in 0..BLOCK_WIDTH {
                let a = reverse(a_rev, LOG_BLOCK_WIDTH); // BitReversal::<LOG_BLOCK_WIDTH>::reverse_bytewise(a_rev);

                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    v.swap(
                        (c_rev << NUM_B_BITS << LOG_BLOCK_WIDTH)
                            | (b_rev << LOG_BLOCK_WIDTH)
                            | a_rev,
                        (a_rev << LOG_BLOCK_WIDTH) | c,
                    );
                }
            }
        }

        // Copy changes that were swapped into buffer above:
        for a in 0..BLOCK_WIDTH {
            let a_rev = reverse(a, shift); // BitReversal::<LOG_BLOCK_WIDTH>::reverse_bytewise(a);

            for c in 0..BLOCK_WIDTH {
                let c_rev = reverse(c, shift); // BitReversal::<LOG_BLOCK_WIDTH>::reverse_bytewise(c);

                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    v.swap(
                        (a << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c,
                        (a_rev << LOG_BLOCK_WIDTH) | c,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cobra::COBRAShuffle;
    use spinoza::math::Float;

    #[test]
    fn cobra() {
        let n = 16;
        let N = 1 << n;
        let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();

        COBRAShuffle(&mut buf);
        println!("{:?}", buf);
    }
}
