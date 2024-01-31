use rand::distributions::Uniform;
use rand::prelude::*;
use std::f64::consts::PI;
pub extern crate rustfft;

/// Asserts that two f64 numbers are approximately equal.
#[allow(dead_code)]
#[track_caller]
pub fn assert_f64_closeness(actual: f64, expected: f64, epsilon: f64) {
    if (actual - expected).abs() >= epsilon {
        panic!(
            "Assertion failed: {} too far from expected value {} (with epsilon {})",
            actual, expected, epsilon
        );
    }
}

/// Asserts that two f32 numbers are approximately equal.
#[allow(dead_code)]
#[track_caller]
pub fn assert_f32_closeness(actual: f32, expected: f32, epsilon: f32) {
    if (actual - expected).abs() >= epsilon {
        panic!(
            "Assertion failed: {} too far from expected value {} (with epsilon {})",
            actual, expected, epsilon
        );
    }
}

pub fn gen_random_signal(reals: &mut [f64], imags: &mut [f64]) {
    assert!(reals.len() == imags.len() && !reals.is_empty());
    let mut rng = thread_rng();
    let between = Uniform::from(0.0..1.0);
    let angle_dist = Uniform::from(0.0..2.0 * PI);
    let num_amps = reals.len();

    let mut probs: Vec<_> = (0..num_amps).map(|_| between.sample(&mut rng)).collect();

    let total: f64 = probs.iter().sum();
    let total_recip = total.recip();

    probs.iter_mut().for_each(|p| *p *= total_recip);

    let angles = (0..num_amps).map(|_| angle_dist.sample(&mut rng));

    probs
        .iter()
        .zip(angles)
        .enumerate()
        .for_each(|(i, (p, a))| {
            let p_sqrt = p.sqrt();
            let (sin_a, cos_a) = a.sin_cos();
            let re = p_sqrt * cos_a;
            let im = p_sqrt * sin_a;
            reals[i] = re;
            imags[i] = im;
        });
}
