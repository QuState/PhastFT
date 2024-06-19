use std::simd::{prelude::Simd, simd_swizzle, SimdElement};

#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
            "x86_64+avx2+fma", // x86_64-v3
            "x86_64+sse4.2", // x86_64-v2
            "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
            "x86+avx2+fma",
            "x86+sse4.2",
            "x86+sse2",
))]
pub(crate) fn deinterleave<T: Copy + Default + SimdElement>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;

    let out_len = input.len() / 2;
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(CHUNK_SIZE * 2)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            let in_first: Simd<T, CHUNK_SIZE> = Simd::from_array(in_chunk[..CHUNK_SIZE].try_into().unwrap());
            let in_second: Simd<T, CHUNK_SIZE> = Simd::from_array(in_chunk[CHUNK_SIZE..].try_into().unwrap());
            let result = simd_swizzle!(in_first, in_second, [0, 2, 4, 6, 1, 3, 5, 7]);
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    // TODO: handle remainder

    (out_odd, out_even)
}

/// Slow but obviously correct implementation of deinterleaving,
/// to be used in tests
fn deinterleave_naive<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}
