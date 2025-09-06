//! FFT Algorithm Implementations
//!
//! This module contains the core FFT algorithm implementations.
//!
//! ## Available Algorithms
//!
//! - **DIF (Decimation-in-Frequency)**: The default algorithm that processes data from
//!   large butterflies to small. Input is in natural order, output is bit-reversed.
//!
//! - **DIT (Decimation-in-Time)**: Alternative algorithm that processes data from
//!   small butterflies to large. Input is bit-reversed, output is in natural order.
//!
//! ## Algorithm Selection
//!
//! - Use DIF if you don't want to or don't need to apply a bit reversal on the input.
//! - Use DIT for slightly better performance.

pub mod dif;
pub mod dit;
