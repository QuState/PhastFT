#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <n-lower-bound> <n-upper-bound>"
  echo ""
  echo "Environment overrides:"
  echo "  PRECISION=64              (32 or 64)"
  echo "  BUDGET_NS=2000000000      (target wall-clock ns per size)"
  echo "  OVERHEAD_NS=200           (modeled fixed cost per iteration)"
  echo "  MIN_ITERS=100             (floor on per-size iteration count)"
  echo "  MAX_ITERS=10000000        (cap on per-size iteration count)"
  exit 1
fi

start=$1
end=$2

precision=${PRECISION:-32}
budget_ns=${BUDGET_NS:-2000000000}
overhead_ns=${OVERHEAD_NS:-200}
min_iters=${MIN_ITERS:-100}
max_iters=${MAX_ITERS:-10000000}

case "$precision" in
  32|64) ;;
  *) echo "PRECISION must be 32 or 64 (got: $precision)" >&2; exit 1 ;;
esac

OUTPUT_DIR=benchmark-data.$(date +"%Y.%m.%d.%H-%M-%S")
mkdir -p \
  "$OUTPUT_DIR/rustfft" \
  "$OUTPUT_DIR/phastft" \
  "$OUTPUT_DIR/fftwrb"

echo "[build] cargo build --release --examples"
(cd .. && cargo build --release --examples)

PHASTFT_BIN=../target/release/examples/benchmark
RUSTFFT_BIN=../target/release/examples/rustfft
FFTWRB_BIN=../target/release/examples/fftwrb

echo "[run] precision=f${precision} budget_ns=${budget_ns}"

for n in $(seq "$start" "$end"); do
  # Per-iteration cost model (ns): fixed overhead + ~N*log2(N) butterflies
  # at one abstract ns each. Pick iters to fit the per-size wall-clock
  # budget. At small n the overhead dominates, so iters stays large
  # (millions of samples); at large n the N*log(N) term dominates and
  # iters shrinks naturally, floored at min_iters and capped at max_iters.
  log2_n=$(( n == 0 ? 1 : n ))
  cost_ns=$(( overhead_ns + (1 << n) * log2_n ))
  iters=$(( budget_ns / cost_ns ))
  if (( iters < min_iters )); then iters=$min_iters; fi
  if (( iters > max_iters )); then iters=$max_iters; fi

  echo "[size] n=2^${n} iters=${iters}"
  # Randomize invocation order per size so no single library always benefits
  # from (or pays for) fresh turbo headroom at the start of the per-size slot.
  printf '%s\n' \
    "phastft:$PHASTFT_BIN" \
    "rustfft:$RUSTFFT_BIN" \
    "fftwrb:$FFTWRB_BIN" \
    | awk 'BEGIN{srand()} {print rand()"\t"$0}' | sort -n | cut -f2- \
    | while IFS=: read -r lib bin; do
        "$bin" "$precision" "$n" "$iters" > "${OUTPUT_DIR}/${lib}/size_${n}"
      done
done

echo "[done] output=$OUTPUT_DIR"
