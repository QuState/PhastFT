//! FFT Butterfly Kernels
//!
//! This module contains the FFT butterfly kernels optimized for various chunk sizes and precision
//! level (i.e., `f32`/`f64`). The kernels are automatically selected at runtime based on available
//! CPU features.
//!
//! ## Organization
//!
//! - `dit`: Decimation-in-Time kernels
//! - `common`: Shared utilities and simple kernels

pub mod common;
pub mod dit;
