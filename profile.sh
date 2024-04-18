#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 1 ]]
then
    echo "Usage: $0 <n>"
    exit 1
fi

cargo +nightly build --profile profiling --example profile

sudo perf record --call-graph=dwarf ./target/profiling/examples/profile $1 && sudo perf script -f -F +pid > processed_result.perf

echo "done! results in process_result.perf"
