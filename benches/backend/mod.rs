//! FFT backend implementations used by the benchmark suite.
//!
//! Each backend module contains the benchmark functions and backend-specific
//! setup for a particular FFT library.
//!
//! The modules share the benchmark utilities in [`crate::common`] while
//! keeping library-specific planners, buffers, and execution details local
//! to each backend.
//!
//! Backend benchmark functions are registered by the individual binaries
//! under `benches/` rather than by this module.

#![allow(
    dead_code,
    reason = "Functions are shared across targets, but not every target uses all of them."
)]

pub mod fftw_lib;
pub mod phastft_lib;
pub mod realfft_lib;
pub mod rustfft_lib;
