//! Benchmarks various function implementations and returns the fastest one on the current hardware

use std::time::Instant;

use crate::algorithms::cobra::*;

// Type alias for the function signature
pub type BitRevFunc = fn(&mut [u8], usize);

/// Benchmarks multiple implementations and returns the fastest one
fn find_fastest_implementation(
    implementations: &[BitRevFunc],
    test_data: &[u8],
    iterations: usize,
) -> BitRevFunc {
    let mut results = Vec::new();

    for func in implementations.iter() {
        // Create a fresh copy of test data for each implementation
        let mut data = test_data.to_vec();
        let log_n = test_data.len().ilog2() as usize;

        // Warm-up run
        func(&mut data, log_n);
        // Reset data between iterations
        data.copy_from_slice(test_data);

        // Actual benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            // Reset data between iterations
            data.copy_from_slice(test_data);
            func(&mut data, log_n);
        }
        let elapsed = start.elapsed().as_nanos();

        results.push((*func, elapsed));
        // println!(
        //     "Implementation {}: {} ns total ({} ns/iter)",
        //     idx,
        //     elapsed,
        //     elapsed / iterations as u128
        // );
    }

    // Find the fastest
    let fastest = results.iter().min_by_key(|(_, time)| time).unwrap();

    fastest.0
}

pub fn measure_fastest_bit_reversal_impl(log_n: usize) -> BitRevFunc {
    let implementations: &[BitRevFunc] = &[bit_rev_cobra, bit_rev_gray, bit_rev_naive];

    let test_data: Vec<u8> = (0..=255).cycle().take(1 << log_n).collect();
    let iterations = 10;

    let fastest = find_fastest_implementation(implementations, &test_data, iterations);

    fastest
}

pub fn guess_fastest_bit_reversal_impl(log_n: usize) -> BitRevFunc {
    if log_n <= 10 {
        bit_rev_gray
    } else {
        bit_rev_cobra
    }
}
