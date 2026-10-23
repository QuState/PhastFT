# NEON 8×8 transpose

The dedicated f32 transpose splits each eight-lane row into two native NEON
registers, transposes four 4×4 quadrants, and combines their columns. It uses
32 vector ZIP instructions instead of the generic network's 48. Split/combine
requires no instructions. Computing the two left quadrants together also
reduces spills when the transpose is inlined into the 64-point FFT.

This path is used by the f32 64-point leaves and the final f32 512-point
transpose. NEON f64 uses four-lane tiles and does not call this eight-lane path.
The x86 implementations are unchanged.

## Reproduce

From the repository root, after a Cargo build/check has resolved `Cargo.lock`:

```sh
python3 benches/neon_transpose/run.py \
  --llvm-mca /path/to/llvm-mca
```

Requires Python 3, `qemu-aarch64`, the installed Rust target
`aarch64-unknown-linux-musl`, and cached Cargo dependencies. The script uses
Rust's bundled `rust-lld` to link a static executable, so a cross GCC/sysroot
is unnecessary. Override `--qemu` or `--out-dir` as needed. `--check-only`
runs just the cross-build and QEMU correctness checks without requiring MCA.

The script copies the current production module into a temporary probe crate,
appends `probes.rs`, and substitutes only the transpose dispatch in the generic
baseline. Both implementations are built in the same binary. The correctness
harness also links the actual PhastFT library through its public API.
Generated assembly, complete MCA reports, tool versions and `summary.json`
are saved under `target/neon-transpose/results` by default.

## Model results

These are **llvm-mca simulations, not ARM hardware timings**. This run used
Rust 1.99.1 / LLVM 23.1.1 for code generation, resolved `fearless_simd` 1.1.0,
and llvm-mca 22.1.0. Code generation uses `target-cpu=generic`, matching a
portable AArch64 binary rather than specializing code for each modeled CPU.

The CPU names map to the dedicated `CortexA55Model`, `NeoverseN1Model`, and
`NeoverseN2Model` in LLVM 22's `AArch64Processors.td`. A55 is an in-order model;
N1 and N2 are out-of-order models. We do not use `cortex-a76` or `apple-m1` as
accurate models of those CPUs: this LLVM version maps them to the older A57
and Cyclone scheduling models respectively.

Long-run cycles per operation, using `Total Cycles / 10000`:

| Model | Transpose, generic → dedicated | Change | FFT64 body, generic → dedicated | Change |
|-------|-------------------------------|--------|---------------------------------|--------|
| Cortex-A55 | 95 → 79 | −16.8% | 738 → 747 | +1.2% |
| Neoverse N1 | 43 → 38 | −11.6% | 449 → 427 | −4.9% |
| Neoverse N2 | 32 → 24 | −25.0% | 273 → 252 | −7.7% |

The transpose region includes eight paired vector loads and eight paired
vector stores: 256 bytes read and 256 bytes written. It has no spills. It
excludes the function return and external call overhead. Total instruction
count falls from 64 to 48.

The FFT64 region includes both complex components, butterflies, twiddle loads,
constant setup, intermediate memory accesses, register spills/reloads, and
output stores. It excludes the length checks, panic path, callee-save
prologue/epilogue, return, and public API dispatch. It is the forward kernel
with scale 1. Its instruction count falls from 639 to 588; instructions
accessing the stack fall from 186 to 170. The different instruction mix still
increases A55's modeled load-pipeline pressure, despite the lower count.

### Throughput sanity checks

Counts and bounds below are per operation, generic → dedicated. Micro-op
counts are those reported by the scheduling model. Dispatch widths are
A55=2, N1=3, N2=5.

| Region / model | Micro-ops | Dispatch lower bound | Dominant overall lower bound | MCA Block RThroughput |
|----------------|-----------|----------------------|------------------------------|----------------------|
| Transpose / A55 | 72 → 56 | 36 → 28 | 48 → 48, load pipe | 48 → 48 |
| Transpose / N1 | 120 → 104 | 40 → 34.67 | 40 → 34.67, dispatch | 40 → 34.7 |
| Transpose / N2 | 96 → 80 | 19.2 → 16 | 32 → 24, vector pipes | 32 → 24 |
| FFT64 / A55 | 692 → 649 | 346 → 324.5 | 396 → 426, load pipe | 396 → 426 |
| FFT64 / N1 | 1187 → 1122 | 395.67 → 374 | 395.67 → 374, dispatch | 395.7 → 374 |
| FFT64 / N2 | 893 → 851 | 178.6 → 170.2 | 273 → 252, vector pipes | 273 → 252 |

The bounds include overlapping resource groups. For example, the N1 FFT's
combined vector-pipe demand is 562/2 → 517/2 cycles, and combined load/store
pipe demand is 480/2 → 464/2; neither exceeds its dispatch bound. N2's combined
vector-pipe demand is 546/2 → 504/2, which dominates. For the transpose, N1/N2
vector demand is 64/2 → 48/2. A55's single load pipe dominates both A55 bounds.
The remaining individual-port and resource-group pressures are below the
reported dominant bounds. No impossible static headline was found, but the
in-order A55 simulation takes much longer than its throughput lower bound.

Registers holding input addresses and stack bases are not overwritten in
the extracted regions. Loads and constant setup define the working values
anew on each iteration; output registers do not feed the next iteration.
The 1,000- and 10,000-iteration runs agree within 0.02 cycles per operation
for the default no-alias results.

### Memory limitations

The primary run uses MCA's `-noalias=true`. Its loads and stores consume
modeled resources, but it does not resolve the FFT's actual stack addresses,
model cache misses, or faithfully predict store-to-load forwarding.
As a sensitivity check, the script also runs the FFT body with
`-noalias=false`, which conservatively makes loads wait for earlier stores:

| Model | FFT64 cycles, generic → dedicated, conservative aliasing |
|-------|----------------------------------------------------------|
| Cortex-A55 | 862 → 848 |
| Neoverse N1 | 688 → 624 |
| Neoverse N2 | 630 → 565 |

This is not an exact memory model or a confidence interval. N1/N2 favor the
dedicated version in both simulations; A55's small full-kernel difference
changes sign. The isolated transpose improvement must not be presented as
an equivalent FFT speedup. Real ARM hardware benchmarks are still needed.

## Correctness

QEMU passes bitwise transpose comparisons, including signed zero, subnormals,
infinities and payload NaNs. It also runs the actual public f32 FFT API against
a scalar f64 DFT for every power of two from 1 through 512, forward and
normalized inverse, with zero, impulse, deterministic complex and tone inputs.
Input slices have independent unaligned offsets and sentinel-protected ends.
QEMU execution times are not used as performance measurements.
