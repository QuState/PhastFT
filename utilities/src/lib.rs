pub extern crate rustfft;

use std::fmt::Display;

use rand::distr::Uniform;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rustfft::num_traits::Float;

/// Asserts that two floating-point numbers are approximately equal.
///
/// # Panics
///
/// Panics if `actual` and `expected` are too far from each other
#[allow(dead_code)]
#[track_caller]
pub fn assert_float_closeness<T: Float + Display>(actual: T, expected: T, epsilon: T) {
    if (actual - expected).abs() >= epsilon {
        panic!(
            "Assertion failed: {actual} too far from expected value {expected} (with epsilon {epsilon})",
        );
    }
}

pub fn gen_random_signal_f32(reals: &mut [f32], imags: &mut [f32]) {
    assert!(reals.len() == imags.len() && !reals.is_empty());
    let mut rng = SmallRng::from_os_rng();
    let dist = Uniform::try_from(-1.0f32..1.0f32).unwrap();

    for (re, im) in reals.iter_mut().zip(imags.iter_mut()) {
        *re = dist.sample(&mut rng);
        *im = dist.sample(&mut rng);
    }

    let magnitude_sq: f32 = reals
        .iter()
        .zip(imags.iter())
        .map(|(re, im)| re * re + im * im)
        .sum();
    let scale = magnitude_sq.sqrt().recip();

    for (re, im) in reals.iter_mut().zip(imags.iter_mut()) {
        *re *= scale;
        *im *= scale;
    }
}

/// Generate a random, complex, signal in the provided buffers
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
pub fn gen_random_signal_f64(reals: &mut [f64], imags: &mut [f64]) {
    assert!(reals.len() == imags.len() && !reals.is_empty());
    let mut rng = SmallRng::from_os_rng();
    let dist = Uniform::try_from(-1.0f64..1.0f64).unwrap();

    for (re, im) in reals.iter_mut().zip(imags.iter_mut()) {
        *re = dist.sample(&mut rng);
        *im = dist.sample(&mut rng);
    }

    let magnitude_sq: f64 = reals
        .iter()
        .zip(imags.iter())
        .map(|(re, im)| re * re + im * im)
        .sum();
    let scale = magnitude_sq.sqrt().recip();

    for (re, im) in reals.iter_mut().zip(imags.iter_mut()) {
        *re *= scale;
        *im *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_random_signal() {
        let big_n = 1 << 25;
        let mut reals = vec![0.0; big_n];
        let mut imags = vec![0.0; big_n];

        gen_random_signal_f64(&mut reals, &mut imags);

        let sum: f64 = reals
            .iter()
            .zip(imags.iter())
            .map(|(re, im)| re.powi(2) + im.powi(2))
            .sum();

        assert_float_closeness(sum, 1.0, 1e-6);
    }
}
