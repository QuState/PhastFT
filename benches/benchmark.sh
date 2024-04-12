#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 2 ]]
then
    echo "Usage: $0 <n-lower-bound> <n-upper-bound>"
    exit 1
fi

start=$1
end=$2
max_iters=1000  # Set your desired maximum number of iterations

OUTPUT_DIR=benchmark-data.$(date +"%Y.%m.%d.%H-%M-%S")
mkdir -p "$OUTPUT_DIR"/fftw3 && mkdir "$OUTPUT_DIR"/rustfft && mkdir "$OUTPUT_DIR"/phastft && mkdir "$OUTPUT_DIR"/fftwrb

benchmark_fftw3() {
    make clean && make

    for n in $(seq "$start" "$end"); do
        iters=$((2**($end - $n)))
        iters=$((iters > max_iters ? max_iters : iters)) # clamp to `max_iters`
        echo "Running FFTW3 benchmark for N = 2^${n} for ${iters} iterations..."

        for _ in $(seq 1 "$iters"); do
            ./bench_fftw "${n}" >> "${OUTPUT_DIR}/fftw3/size_${n}"
        done
    done
}

benchmark_phastft() {
    cargo clean && cargo build --release --examples

    for n in $(seq "$start" "$end"); do
        iters=$((2**($end - $n)))
        iters=$((iters > max_iters ? max_iters : iters))
        echo "Running PhastFT benchmark for N = 2^${n}..."

        for _ in $(seq 1 "$iters"); do
            ../target/release/examples/benchmark "${n}" >> "${OUTPUT_DIR}"/phastft/size_"${n}"
        done
    done
}

benchmark_rustfft() {
    cargo clean && cargo build --release --examples

    for n in $(seq "$start" "$end"); do
        iters=$((2**($end - $n)))
        iters=$((iters > max_iters ? max_iters : iters))
        echo "Running RustFFT benchmark for N = 2^${n}..."

        for _ in $(seq 1 "$iters"); do
            ../target/release/examples/rustfft "${n}" >> "${OUTPUT_DIR}"/rustfft/size_"${n}"
        done
    done
}

benchmark_rs_fftw3() {
    cargo clean && cargo build --release --examples

    for n in $(seq "$start" "$end"); do
        iters=$((2**($end - $n)))
        iters=$((iters > max_iters ? max_iters : iters))
        echo "Running FFTW3 Rust bindings benchmark for N = 2^${n}..."

        for _ in $(seq 1 "$iters"); do
            ../target/release/examples/fftwrb "${n}" >> "${OUTPUT_DIR}"/fftwrb/size_"${n}"
        done
    done
}

benchmark_rs_fftw3
benchmark_fftw3
benchmark_phastft
benchmark_rustfft
