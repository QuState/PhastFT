//! Options to tune to improve performance depending on the hardware and input size.

/// Calling FFT routines without specifying options will automatically select reasonable defaults
/// depending on the input size and other factors.
///
/// You only need to tune these options if you are trying to squeeze maximum performance
/// out of a known hardware platform that you can benchmark at varying input sizes.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Whether to run the bit reversal step in 2 threads instead of one.
    /// This is beneficial only at large input sizes (i.e. gigabytes of data).
    /// The exact threshold where it starts being beneficial varies depending on the hardware.
    pub multithreaded_bit_reversal: bool,
}

impl Options {
    /// Attempt to guess the best settings to use for optimal FFT
    pub fn guess_options(input_size: usize) -> Options {
        let mut options = Options::default();
        let n: usize = input_size.ilog2() as usize;
        options.multithreaded_bit_reversal = n >= 22;
        options
    }
}
