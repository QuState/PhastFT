#!/usr/bin/env bash

cargo build --release --bin phastft && sudo perf record --call-graph=dwarf ./target/release/phastft && sudo perf script -f -F +pid > processed_result.perf 

echo "done! results in process_result.perf"
