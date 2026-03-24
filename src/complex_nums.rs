//! Functions for complex numbers such as interleave/deinterleave
//!
//! They are not part of the public API because the module they're in is private.
//! They can be accessed with `--cfg phastft_bench` for benchmarking only.

use bytemuck::cast_slice;
use num_complex::Complex;
use num_traits::Float;

/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length
pub fn deinterleave<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    // Despite relying on autovectorization, this is the fastest approach
    // because we don't need to initialize the output Vecs.
    // The Vecs are also allocated up front without intermediate reallocations.
    // This is faster than any implementation we could write without `unsafe`.
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}

/// Utility function to separate a slice of [`Complex64``]
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn deinterleave_complex64(signal: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
    let complex_t: &[f64] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Utility function to separate a slice of [`Complex32``]
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn deinterleave_complex32(signal: &[Complex<f32>]) -> (Vec<f32>, Vec<f32>) {
    let complex_t: &[f32] = cast_slice(signal);
    deinterleave(complex_t)
}

/// Utility function to combine separate vectors of real and imaginary components
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
pub fn combine_re_im<T: Float>(reals: &[T], imags: &[T]) -> Vec<Complex<T>> {
    assert_eq!(reals.len(), imags.len());

    reals
        .iter()
        .zip(imags.iter())
        .map(|(z_re, z_im)| Complex::new(*z_re, *z_im))
        .collect()
}

// FANCY IN-PLACE INTERLEAVING/DEINTERLEAVING
// SPECIALIZED FOR BUFFERS WITH POWER OF 2 SIZE

// Overview of the algorithm and possible other approaches:
// https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221mYtdKnDYXiB5yj7GvxSgrc-McoGZfKoL%22%5D,%22action%22:%22open%22,%22userId%22:%22100773621717907681037%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

// OUT-OF-PLACE FUNCTIONS FOR SMALL BLOCKS

/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length,
/// writing to existing output slices instead of allocating new Vecs.
pub fn deinterleave_into<T: Copy>(input: &[T], output_a: &mut [T], output_b: &mut [T]) {
    let half = input.len() / 2;
    assert_eq!(output_a.len(), half);
    assert_eq!(output_b.len(), half);

    for (i, c) in input.chunks_exact(2).enumerate() {
        output_a[i] = c[0];
        output_b[i] = c[1];
    }
}

/// Combines parallel reals and imags arrays into an interleaved output.
pub fn combine_re_im_into<T: Copy>(reals: &[T], imags: &[T], output: &mut [T]) {
    assert_eq!(reals.len(), imags.len());
    assert_eq!(output.len(), reals.len() * 2);

    for ((out, z_re), z_im) in output
        .chunks_exact_mut(2)
        .zip(reals.iter())
        .zip(imags.iter())
    {
        out[0] = *z_re;
        out[1] = *z_im;
    }
}

// CYCLE MATH (BIT ROTATIONS)

/// Computes a circular left bit-shift of `val` for a `k`-bit integer.
#[inline(always)]
fn left_rotate(val: usize, k: u32) -> usize {
    if k == 0 {
        return val;
    }
    let mask = (1_usize << k) - 1;
    ((val << 1) & mask) | (val >> (k - 1))
}

/// Computes a circular right bit-shift of `val` for a `k`-bit integer.
#[inline(always)]
fn right_rotate(val: usize, k: u32) -> usize {
    if k == 0 {
        return val;
    }
    (val >> 1) | ((val & 1) << (k - 1))
}

/// A block index is the "Cycle Leader" if it is the mathematical minimum
/// of all its bit-rotations. This guarantees we only process a cycle once.
fn is_cycle_leader(val: usize, k: u32) -> bool {
    let mask = (1_usize << k) - 1;
    if val == mask {
        return true;
    } // All 1s is always its own leader

    let mut current = val;
    for _ in 0..k {
        current = left_rotate(current, k);
        if current < val {
            return false;
        }
    }
    true
}

// MAIN IN-PLACE ALGORITHMS

/// Performs an in-place interleaving of `[Reals | Imags]` into `[Re, Im, Re, Im...]`.
/// `data.len()` and `block_size` must both be powers of 2.
pub fn interleave_inplace<T: Copy + Default>(data: &mut [T], block_size: usize) {
    let len = data.len();
    assert!(len.is_power_of_two(), "Data length must be a power of 2");
    assert!(
        block_size.is_power_of_two(),
        "Block size must be a power of 2"
    );
    assert!(len >= block_size * 2, "Data must contain at least 2 blocks");

    let num_blocks = len / block_size;
    let k = num_blocks.trailing_zeros(); // 2^k blocks

    // Phase 1: Block Perfect Shuffle (Forward)
    // Destination function: Left Rotate. Source function: Right Rotate.
    let mut temp_block = vec![T::default(); block_size];

    for i in 0..num_blocks {
        if is_cycle_leader(i, k) {
            // Pick up the leader block
            temp_block.copy_from_slice(&data[i * block_size..(i + 1) * block_size]);

            let mut hole = i;
            loop {
                // The block that *belongs* in the current hole
                let source = right_rotate(hole, k);

                if source == i {
                    // Loop closed, fill the final hole with the initial temp block
                    data[hole * block_size..(hole + 1) * block_size].copy_from_slice(&temp_block);
                    break;
                }

                // Move the source block into the hole safely
                data.copy_within(
                    source * block_size..(source + 1) * block_size,
                    hole * block_size,
                );
                hole = source;
            }
        }
    }

    // Phase 2: Local SIMD Interleave
    let mut temp_pair = vec![T::default(); 2 * block_size];
    for i in 0..(num_blocks / 2) {
        let window = &mut data[2 * i * block_size..2 * (i + 1) * block_size];
        let (reals, imags) = window.split_at(block_size);
        combine_re_im_into(reals, imags, &mut temp_pair);
        window.copy_from_slice(&temp_pair);
    }
}

/// Reverses the in-place interleaving, returning `[Re, Im, ...]` back to `[Reals | Imags]`.
pub fn deinterleave_inplace<T: Copy + Default>(data: &mut [T], block_size: usize) {
    let len = data.len();
    assert!(len.is_power_of_two(), "Data length must be a power of 2");
    assert!(
        block_size.is_power_of_two(),
        "Block size must be a power of 2"
    );
    assert!(len >= block_size * 2, "Data must contain at least 2 blocks");

    let num_blocks = len / block_size;
    let k = num_blocks.trailing_zeros();

    // Phase 1: Local SIMD Deinterleave
    let mut temp_re = vec![T::default(); block_size];
    let mut temp_im = vec![T::default(); block_size];

    for i in 0..(num_blocks / 2) {
        let window = &mut data[2 * i * block_size..2 * (i + 1) * block_size];
        deinterleave_into(window, &mut temp_re, &mut temp_im);

        // Write the newly split arrays back over the combined window
        let (reals, imags) = window.split_at_mut(block_size);
        reals.copy_from_slice(&temp_re);
        imags.copy_from_slice(&temp_im);
    }

    // Phase 2: Block Perfect Unshuffle (Reverse)
    // Destination function: Right Rotate. Source function: Left Rotate.
    let mut temp_block = vec![T::default(); block_size];

    for i in 0..num_blocks {
        // We use the EXACT same cycle leader logic. The sets are identical!
        if is_cycle_leader(i, k) {
            temp_block.copy_from_slice(&data[i * block_size..(i + 1) * block_size]);

            let mut hole = i;
            loop {
                // The ONLY DIFFERENCE from Phase 1: Source is now computed via Left Rotate
                let source = left_rotate(hole, k);

                if source == i {
                    data[hole * block_size..(hole + 1) * block_size].copy_from_slice(&temp_block);
                    break;
                }

                data.copy_within(
                    source * block_size..(source + 1) * block_size,
                    hole * block_size,
                );
                hole = source;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::distr::StandardUniform;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    fn gen_test_vec(len: usize) -> Vec<usize> {
        (0..len).collect()
    }

    /// Slow but obviously correct implementation of deinterleaving,
    /// to be used in tests
    fn deinterleave_naive<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
        input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
    }

    #[test]
    fn deinterleaving_correctness() {
        for len in [0, 1, 2, 3, 15, 16, 17, 127, 128, 129, 130, 135, 100500] {
            let input = gen_test_vec(len);
            let (naive_a, naive_b) = deinterleave_naive(&input);
            let (opt_a, opt_b) = deinterleave(&input);
            assert_eq!(naive_a, opt_a);
            assert_eq!(naive_b, opt_b);
        }
    }

    #[test]
    fn test_separate_and_combine_re_im() {
        let mut rng = SmallRng::from_os_rng();
        let complex_vec: Vec<f32> = (&mut rng)
            .sample_iter(StandardUniform)
            .take(16384)
            .collect();

        let (reals, imags) = deinterleave(&complex_vec);

        let recombined_vec = combine_re_im(&reals, &imags);

        let recombined_flat: &[f32] = cast_slice(recombined_vec.as_slice());
        assert_eq!(complex_vec, recombined_flat);
    }

    // Naive out-of-place implementations to test against
    fn interleave_out_of_place(reals: &[f32], imags: &[f32]) -> Vec<f32> {
        reals
            .iter()
            .zip(imags.iter())
            .flat_map(|(r, i)| vec![*r, *i])
            .collect()
    }

    #[test]
    fn test_perfect_round_trip() {
        let n = 2048;
        let block_size = 32; // Try tweaking this block size! (Must be a power of 2)
        let half = n / 2;

        let mut original = vec![0.0; n];
        for i in 0..n {
            original[i] = i as f32;
        }

        let mut data = original.clone();

        // 1. Generate the expected "truth" array using out-of-place allocation
        let expected_interleaved = interleave_out_of_place(&original[..half], &original[half..]);

        // 2. Run our in-place forward algorithm
        interleave_inplace(&mut data, block_size);

        // Verify it matches the standard out-of-place interleaving
        assert_eq!(
            data, expected_interleaved,
            "Forward inplace interleave failed!"
        );

        // 3. Run our in-place reverse algorithm
        deinterleave_inplace(&mut data, block_size);

        // Verify it correctly mapped back to[Reals | Imags]
        assert_eq!(data, original, "Reverse inplace deinterleave failed!");
    }

    #[test]
    fn test_tiny_blocks() {
        // Edge Case: Block size is 1. All work is handled by Phase 1 swapping.
        // Phase 2 will just do a 1-to-1 copy in place.
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 8 elements
        let original = data.clone();

        let expected = interleave_out_of_place(&original[..4], &original[4..]);

        interleave_inplace(&mut data, 1);
        assert_eq!(data, expected);

        deinterleave_inplace(&mut data, 1);
        assert_eq!(data, original);
    }

    #[test]
    fn deinterleave_into_correctness() {
        for len in [0, 1, 2, 3, 15, 16, 17, 127, 128, 129, 130, 135, 100500] {
            let input = gen_test_vec(len);
            let half = len / 2;
            let mut out_a = vec![0usize; half];
            let mut out_b = vec![0usize; half];
            deinterleave_into(&input, &mut out_a, &mut out_b);
            let (expected_a, expected_b) = deinterleave(&input);
            assert_eq!(out_a, expected_a);
            assert_eq!(out_b, expected_b);
        }
    }

    #[test]
    fn test_separate_and_combine_re_im_into() {
        let mut rng = SmallRng::from_os_rng();
        let complex_vec: Vec<f32> = (&mut rng)
            .sample_iter(StandardUniform)
            .take(16384)
            .collect();

        let (reals, imags) = deinterleave(&complex_vec);

        let mut output = vec![0.0f32; reals.len() * 2];
        combine_re_im_into(&reals, &imags, &mut output);

        let output_flat: &[f32] = cast_slice(output.as_slice());
        assert_eq!(complex_vec, output_flat);
    }
}
