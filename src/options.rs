/// Options to tune to improve performance depending on the hardware and input size.
/// 
/// Calling FFT routines without specifying options will automatically select reasonable defaults
/// depending on the input size and other factors.
/// 
/// You only need to tune these options if you are trying to squeeze maximum performance
/// out of a known hardware platform that you can bechmark at varying input sizes.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub bit_reverse: BitReverseAlgorithm,
}

impl Options {
    pub(crate) fn guess_options(input_size: usize) -> Options {
        let mut options = Options::default();
        let n: usize = input_size.ilog2() as usize;
        if n < 22 {
            options.bit_reverse = BitReverseAlgorithm::Cobra;
        } else {
            options.bit_reverse = BitReverseAlgorithm::MultiThreadedCobra;
        }
        options
    }
}

/// The algorithm to use for bit reversal.
/// Different algorithms perform best on different input sizes.
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum BitReverseAlgorithm {
    #[default]
    /// Straightforward algorithm that performs best at smaller sizes
    Plain,
    /// Cache-Optimal Bit Reversal Algorithm
    /// 
    /// This is faster at larger datasets that do not fit into the cache.
    /// The exact threshold where it starts being beneficial varies depending on the hardware.
    Cobra,
    /// COBRA but run on two threads instead of one.
    /// Typically beneficial at even larger sizes than single-threaded COBRA, and slower otherwise.
    /// The exact threshold where it starts being beneficial varies depending on the hardware.
    MultiThreadedCobra,
}