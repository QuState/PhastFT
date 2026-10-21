//! Experimental full-width SIMD transposes, with packed or direct loads.
//!
//! The runtime-selected AVX-512 versions use f64x8 and f32x16 registers.
//! Keep the whole SIMD call chain force-inlined into dispatch!'s wrapper.

use fearless_simd::{dispatch, f32x16, f64x8, Level, Simd, SimdBase};

#[derive(Clone, Copy)]
pub struct Columns {
    pub rows: usize,
    pub cols: usize,
    pub first_col: usize,
}

macro_rules! implementation {
    ($name:ident, $inner:ident, $float:ty, $vector:ident, $lanes:literal, $stages:literal) => {
        pub fn $name<const WIDTH: usize, const PACKED: bool>(
            level: Level,
            re: &[$float],
            im: &[$float],
            dr: &mut [$float],
            di: &mut [$float],
            columns: Columns,
        ) {
            dispatch!(level, simd => $inner::<_, WIDTH, PACKED>(simd, re, im, dr, di, columns));
        }

        #[inline(always)] // Keep all SIMD operations inside dispatch!'s target-feature boundary.
        fn $inner<S: Simd, const WIDTH: usize, const PACKED: bool>(
            simd: S,
            re: &[$float],
            im: &[$float],
            dr: &mut [$float],
            di: &mut [$float],
            Columns { rows, cols, first_col }: Columns,
        ) {
            let width = dr.len() / rows;
            let mut tile = [[0.0; WIDTH]; super::TILE_ROWS];
            for (src, dst) in [(re, dr), (im, di)] {
                for r0 in (0..rows).step_by(super::TILE_ROWS) {
                    let height = (rows - r0).min(super::TILE_ROWS);
                    if PACKED {
                        for (r, tile_row) in tile.iter_mut().take(height).enumerate() {
                            let source = (r0 + r) * cols + first_col;
                            tile_row[..width].copy_from_slice(&src[source..source + width]);
                        }
                    }
                    // Power-of-two matrix dimensions only need scalar handling
                    // when an entire dimension is smaller than one vector.
                    if width < $lanes || height < $lanes {
                        for c in 0..width {
                            for r in 0..height {
                                dst[c * rows + r0 + r] = if PACKED { tile[r][c] } else { src[(r0 + r) * cols + first_col + c] };
                            }
                        }
                        continue;
                    }
                    for c in (0..width).step_by($lanes) {
                        for r in (0..height).step_by($lanes) {
                            let mut v = [$vector::splat(simd, 0.0); $lanes];
                            for j in 0..$lanes {
                                v[j] = if PACKED {
                                    $vector::from_slice(simd, &tile[r + j][c..c + $lanes])
                                } else {
                                    let source = (r0 + r + j) * cols + first_col + c;
                                    $vector::from_slice(simd, &src[source..source + $lanes])
                                };
                            }
                            // Each stage rotates the combined row/column index
                            // by one bit; log2(lanes) stages exchange the indices.
                            for _ in 0..$stages {
                                let mut next = v;
                                for j in 0..$lanes / 2 {
                                    (next[2 * j], next[2 * j + 1]) = v[j].interleave(v[j + $lanes / 2]);
                                }
                                v = next;
                            }
                            for j in 0..$lanes {
                                let dest = (c + j) * rows + r0 + r;
                                v[j].store_slice(&mut dst[dest..dest + $lanes]);
                            }
                        }
                    }
                }
            }
        }
    };
}

implementation!(columns_f32, columns_f32_impl, f32, f32x16, 16, 4);
implementation!(columns_f64, columns_f64_impl, f64, f64x8, 8, 3);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! permutation {
        ($name:ident, $float:ty, $bits:ty, $kernel:ident) => {
            #[test]
            fn $name() {
                fn check<const WIDTH: usize, const PACKED: bool>() {
                    // Include partial tiles, multiple row tiles and nonzero
                    // source column offsets, on baseline and detected ISAs.
                    for level in [Level::baseline(), Level::new()] {
                        for rows in [1, 2, 4, 8, 16, 32, 64] {
                            for cols in [1, 2, 4, 8, 16, 32, 128, 256] {
                                let re: Vec<$float> = (0..rows * cols)
                                    .map(|i| {
                                        // Preserve arbitrary bits, including NaNs and signed zero.
                                        let special = [
                                            0,
                                            <$float>::NAN.to_bits(),
                                            (-0.0 as $float).to_bits(),
                                            <$float>::INFINITY.to_bits(),
                                        ];
                                        <$float>::from_bits(if i < 4 {
                                            special[i]
                                        } else {
                                            (i as $bits).wrapping_mul(0x9e3779b9 as $bits)
                                        })
                                    })
                                    .collect();
                                let im: Vec<_> = re.iter().copied().rev().collect();
                                let mut dr = vec![0.0; rows * cols];
                                let mut di = vec![0.0; rows * cols];
                                for (batch, (r, i)) in dr
                                    .chunks_mut(WIDTH * rows)
                                    .zip(di.chunks_mut(WIDTH * rows))
                                    .enumerate()
                                {
                                    $kernel::<WIDTH, PACKED>(
                                        level,
                                        &re,
                                        &im,
                                        r,
                                        i,
                                        Columns {
                                            rows,
                                            cols,
                                            first_col: batch * WIDTH,
                                        },
                                    );
                                }
                                for r in 0..rows {
                                    for c in 0..cols {
                                        assert_eq!(
                                            dr[c * rows + r].to_bits(),
                                            re[r * cols + c].to_bits()
                                        );
                                        assert_eq!(
                                            di[c * rows + r].to_bits(),
                                            im[r * cols + c].to_bits()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                check::<16, true>();
                check::<128, true>();
                check::<16, false>();
                check::<128, false>();
            }
        };
    }
    permutation!(preserves_bits_f32, f32, u32, columns_f32);
    permutation!(preserves_bits_f64, f64, u64, columns_f64);
}
