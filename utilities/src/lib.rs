pub extern crate rustfft;

// export rustfft to phastft
use rand::{distributions::Uniform, prelude::*};
use rustfft::num_traits::Float;

/// Asserts that two fp numbers are approximately equal.
///
/// # Panics
///
/// Panics if `actual` and `expected` are too far from each other
#[allow(dead_code)]
#[track_caller]
pub fn assert_float_closeness<T: Float + std::fmt::Display>(actual: T, expected: T, epsilon: T) {
    if (actual - expected).abs() >= epsilon {
        panic!(
            "Assertion failed: {actual} too far from expected value {expected} (with epsilon {epsilon})",
        );
    }
}

/// Generate a random, complex, signal in the provided buffers
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
pub fn gen_random_signal<T>(reals: &mut [T], imags: &mut [T])
where
    T: Float + rand::distributions::uniform::SampleUniform,
{
    assert_eq!(
        reals.len(),
        imags.len(),
        "Real and imaginary slices must be of equal length"
    );

    let mut rng = thread_rng();

    let uniform_dist = Uniform::new(T::from(-1.0).unwrap(), T::from(1.0).unwrap());
    for (real, imag) in reals.iter_mut().zip(imags.iter_mut()) {
        *real = uniform_dist.sample(&mut rng);
        *imag = uniform_dist.sample(&mut rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_random_signal() {
        let big_n = 1 << 25;
        let mut reals: Vec<_> = vec![0.0; big_n];
        let mut imags: Vec<_> = vec![0.0; big_n];

        gen_random_signal::<f64>(&mut reals, &mut imags);

        let sum = reals
            .iter()
            .zip(imags.iter())
            .map(|(re, im)| re.powi(2) + im.powi(2))
            .sum();

        assert_float_closeness(sum, 1.0, 1e-6);
    }
}
