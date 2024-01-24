#!/usr/bin/env bash

RUSTFLAGS='-Ctarget-cpu=native' cargo +nightly build --release --examples

sudo perf record --call-graph=dwarf ./target/release/examples/profile && sudo perf script -f -F +pid > processed_result.perf

echo "done! results in process_result.perf"
