#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 1 ]]
then
    echo "Usage: $0 <n>"
    exit 1
fi

RUSTFLAGS='-Ctarget-cpu=native' cargo +nightly build --release --examples

sudo perf record --call-graph=dwarf ./target/release/examples/profile $1 && sudo perf script -f -F +pid > processed_result.perf

echo "done! results in process_result.perf"
