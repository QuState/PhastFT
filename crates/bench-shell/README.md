# Shell-Driven Cross-Library Pipeline

big-N comparisons against RustFFT and FFTW3.

`scripts/benchmark.sh` builds the example binaries in `src/`
(`benchmark`, `rustfft`, `fftwrb`) and drives them through a power-of-2
size sweep, randomizing per-size invocation order so no single library
always runs first. Output is a timestamped `benchmark-data.<ts>/`
directory with one newline-separated ns-per-iter file per
(library, size).

### Run

Run from the **this directory**:

```bash
./scripts/benchmark.sh <n-lower-bound> <n-upper-bound>
```

or run from the **repo root**:

```bash
./crates/bench-shell/scripts/benchmark.sh <n-lower-bound> <n-upper-bound>
```

Each size's iteration count is derived from an N·log2(N) cost model
targeting `BUDGET_NS` (default 2 s) of wall clock. Override with
environment variables:

| Variable      | Default      | Controls                                   |
| ------------- | ------------ | ------------------------------------------ |
| `PRECISION`   | `32`         | `32` or `64` — single or double precision. |
| `BUDGET_NS`   | `2000000000` | Target wall-clock ns per size.             |
| `OVERHEAD_NS` | `200`        | Modeled fixed cost per iteration.          |
| `MIN_ITERS`   | `100`        | Floor on per-size iteration count.         |
| `MAX_ITERS`   | `10000000`   | Cap on per-size iteration count.           |

Output layout:

```
benchmark-data.YYYY.MM.DD.HH-MM-SS/
├── phastft/size_{n}    # newline-separated ns-per-iter floats
├── rustfft/size_{n}
└── fftwrb/size_{n}
```

### Plot

run from the **repo root**:

```bash
uv run ./crates/bench-shell/scripts/benchmark_plots.py
```

`benchmark_plots.py` uses PEP 723 inline script metadata, so `uv run`
fetches matplotlib / numpy / pandas on demand — no venv needed. It
auto-discovers the **latest** `benchmark-data.*` directory; to plot an
older run, move newer directories out of the way (there is no CLI
flag).

The plot is a grouped bar chart per size: bars are each library's
median, normalized against RustFFT's median; whiskers are the IQR in
that normalized space; the dashed `y = 1.0` line is RustFFT by
construction. PNGs land in the current working directory.
