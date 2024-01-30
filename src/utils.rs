/// Asserts that two f64 numbers are approximately equal.
#[allow(dead_code)]
#[track_caller]
pub(crate) fn assert_f64_closeness(actual: f64, expected: f64, epsilon: f64) {
    if !((actual - expected).abs() < epsilon) {
        panic!(
            "Assertion failed: {} too far from expected value {} (with epsilon {})",
            actual, expected, epsilon
        );
    }
}

/// Asserts that two f32 numbers are approximately equal.
#[allow(dead_code)]
#[track_caller]
pub(crate) fn assert_f32_closeness(actual: f32, expected: f32, epsilon: f32) {
    if !((actual - expected).abs() < epsilon) {
        panic!(
            "Assertion failed: {} too far from expected value {} (with epsilon {})",
            actual, expected, epsilon
        );
    }
}
