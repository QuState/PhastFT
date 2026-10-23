// Appended to a temporary copy of small.rs by run.py, so the probes use the
// production helpers without exposing them in PhastFT's public API.

#[inline(always)]
fn transpose8_generic_neon(_simd: Neon, v: [f32x8<Neon>; 8]) -> [f32x8<Neon>; 8] {
    let (a0, a1) = v[0].interleave(v[4]);
    let (b0, b1) = v[1].interleave(v[5]);
    let (c0, c1) = v[2].interleave(v[6]);
    let (d0, d1) = v[3].interleave(v[7]);
    let lo = transpose4([a0, b0, c0, d0]);
    let hi = transpose4([a1, b1, c1, d1]);
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}
#[inline(never)]
pub(crate) fn probe_transpose(simd: Neon, input: &[f32; 64], output: &mut [f32; 64]) {
    let v = std::array::from_fn(|row| f32x8::from_slice(simd, &input[row * 8..row * 8 + 8]));
    let v = TRANSPOSE_IMPL(simd, v);
    for row in 0..8 {
        v[row].store_slice(&mut output[row * 8..row * 8 + 8]);
    }
}
#[inline(never)]
pub(crate) fn probe_fft64(simd: Neon, re: &mut [f32; 64], im: &mut [f32; 64], plan: &Plan<f32>) {
    single::fft(simd, re, im, 1.0, plan);
}
