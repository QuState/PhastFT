//! Options to tune to improve performance depending on the hardware and input size.

/// Calling FFT routines without specifying options will automatically select reasonable defaults
/// depending on the input size and other factors.
///
/// You only need to tune these options if you are trying to squeeze maximum performance
/// out of a known hardware platform that you can benchmark at varying input sizes.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Options {
    /// Whether to run the bit reversal step in 2 threads instead of one.
    /// This is beneficial only at large input sizes (i.e. gigabytes of data).
    /// The exact threshold where it starts being beneficial varies depending on the hardware.
    pub multithreaded_bit_reversal: bool,

    /// Controls bit reversal behavior for DIF FFT algorithms.
    ///
    /// **This option only affects DIF (Decimation-in-Frequency) algorithms** (`fft_32`, `fft_64`).
    /// **DIT algorithms ignore this setting** as they always require bit reversal for correctness.
    ///
    /// For DIF FFT:
    /// - `true` (default): Output is bit-reversed (standard FFT output)
    /// - `false`: Output remains in decimated order (useful when chaining operations or
    ///   when you need the output in the natural DIF order)
    ///
    /// For DIT FFT:
    /// - This option is ignored. DIT always performs bit reversal on input.
    ///
    /// # Example
    /// ```ignore
    /// let mut opts = Options::default();
    /// opts.dif_perform_bit_reversal = false;  // Keep DIF output in decimated order
    /// fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &planner);
    /// ```
    pub dif_perform_bit_reversal: bool,

    /// Do not split the input any further to run in parallel below this size
    ///
    /// Set to `usize::MAX` to disable parallelism in the recursive FFT step.
    pub smallest_parallel_chunk_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            multithreaded_bit_reversal: false,
            dif_perform_bit_reversal: true, // Default to standard FFT behavior
            smallest_parallel_chunk_size: usize::MAX,
        }
    }
}

impl Options {
    /// Attempt to guess the best settings to use for optimal FFT
    pub fn guess_options(input_size: usize) -> Options {
        let mut options = Options::default();
        let n: usize = input_size.ilog2() as usize;
        options.multithreaded_bit_reversal = n >= 16;
        options.smallest_parallel_chunk_size = 16384;
        options
    }
}
