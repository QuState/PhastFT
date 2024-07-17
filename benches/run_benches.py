import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import logging
import numpy as np

# Configuration
OUTPUT_DIR = "benchmark_output"
HISTORY_DIR = "benchmark_history"
LOG_DIR = "benchmark_logs"
MAX_ITERS = 1 << 10
START = 6
END = 20
STD_THRESHOLD = 0.05  # 5% standard deviation threshold

# Ensure log directory exists
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=Path(LOG_DIR) / "benchmark.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


def run_command(command, cwd=None):
    result = subprocess.run(
        command, shell=True, text=True, capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        logging.error(f"Error running command: {command}\n{result.stderr}")
        sys.exit(result.returncode)
    return result.stdout.strip()


def clean_build_rust():
    logging.info("Cleaning and building Rust project...")
    run_command("cargo clean")
    run_command("cargo build --release --examples")


def benchmark_with_stabilization(executable_name, n, max_iters, std_threshold):
    times = []
    for i in range(max_iters):
        result = run_command(f"../target/release/examples/{executable_name} {n}")
        times.append(int(result))
        if len(times) > 10:  # Start evaluating after a minimum number of runs
            current_std = np.std(times) / np.mean(times)
            if current_std < std_threshold:
                break
    return times


def benchmark(
    benchmark_name, output_subdir, start, end, max_iters, std_threshold, executable_name
):
    output_dir_path = Path(OUTPUT_DIR) / output_subdir
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for n in range(start, end + 1):
        logging.info(
            f"Running {benchmark_name} benchmark for N = 2^{n} with a standard deviation threshold of {std_threshold * 100}%..."
        )
        times = benchmark_with_stabilization(
            executable_name, n, max_iters, std_threshold
        )
        output_file = output_dir_path / f"size_{n}"
        with open(output_file, "w") as f:
            for time in times:
                f.write(f"{time}\n")
        logging.info(
            f"Completed N = 2^{n} in {len(times)} iterations with a final standard deviation of {np.std(times) / np.mean(times):.2%}"
        )


def read_benchmark_results(output_dir, start, end):
    sizes = []
    times = []

    for n in range(start, end + 1):
        size_file = Path(output_dir) / f"size_{n}"
        if size_file.exists():
            with open(size_file, "r") as f:
                data = f.readlines()
                data = [int(line.strip()) for line in data]
                if data:
                    min_time_ns = min(data)
                    sizes.append(2**n)
                    times.append(min_time_ns)
                else:
                    logging.warning(f"No data found in file: {size_file}")
        else:
            logging.warning(f"File does not exist: {size_file}")

    return sizes, times


def plot_benchmark_results(output_subdirs, start, end, history_dirs=[]):
    plt.figure(figsize=(10, 6))
    has_data = False

    # Plot current results
    for subdir in output_subdirs:
        sizes, times = read_benchmark_results(Path(OUTPUT_DIR) / subdir, start, end)
        if sizes and times:
            has_data = True
            plt.plot(sizes, times, marker="o", label=f"current {subdir}")

    # Plot previous results from history for PhastFT
    for history_dir in history_dirs:
        sizes, times = read_benchmark_results(
            Path(history_dir) / "benchmark_output" / "phastft", start, end
        )
        if sizes and times:
            has_data = True
            timestamp = Path(history_dir).stem
            plt.plot(
                sizes, times, marker="x", linestyle="--", label=f"{timestamp} phastft"
            )

    if has_data:
        plt.title("Benchmark Results")
        plt.xlabel("FFT Size (N)")
        plt.ylabel("Minimum Time (ns)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/benchmark_results.png", dpi=600)
        # plt.show()
    else:
        logging.warning("No data available to plot.")


def compare_results(current_dir, previous_dir, start, end):
    changes = {}
    for n in range(start, end + 1):
        current_file = Path(current_dir) / f"size_{n}"
        previous_file = (
            Path(previous_dir) / "benchmark_output" / "phastft" / f"size_{n}"
        )

        if current_file.exists() and previous_file.exists():
            with open(current_file, "r") as cf, open(previous_file, "r") as pf:
                current_data = [int(line.strip()) for line in cf.readlines()]
                previous_data = [int(line.strip()) for line in pf.readlines()]

                if current_data and previous_data:
                    current_min = min(current_data)
                    previous_min = min(previous_data)

                    if current_min != previous_min:
                        change = ((current_min - previous_min) / previous_min) * 100
                        changes[n] = change
                else:
                    logging.warning(
                        f"Data missing in files for size 2^{n}: Current data length: {len(current_data)}, Previous data length: {len(previous_data)}"
                    )
        else:
            logging.warning(
                f"Missing files for size 2^{n}: Current file exists: {current_file.exists()}, Previous file exists: {previous_file.exists()}"
            )

    return changes


def archive_current_results():
    if Path(OUTPUT_DIR).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = Path(HISTORY_DIR) / timestamp
        history_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(OUTPUT_DIR, history_dir)
        logging.info(f"Archived current results to: {history_dir}")
    else:
        logging.warning(
            f"Output directory '{OUTPUT_DIR}' does not exist and cannot be archived."
        )


def main():
    clean_build_rust()

    # Check if there are previous results for comparison
    history_dirs = (
        sorted(Path(HISTORY_DIR).iterdir(), key=os.path.getmtime)
        if Path(HISTORY_DIR).exists()
        else []
    )
    latest_previous_dir = history_dirs[-1] if history_dirs else None

    # Run new benchmarks for PhastFT, RustFFT, and FFTW3
    benchmark("PhastFT", "phastft", START, END, MAX_ITERS, STD_THRESHOLD, "benchmark")
    benchmark("RustFFT", "rustfft", START, END, MAX_ITERS, STD_THRESHOLD, "rustfft")
    benchmark(
        "FFTW3 Rust bindings", "fftwrb", START, END, MAX_ITERS, STD_THRESHOLD, "fftwrb"
    )

    # Compare new PhastFT benchmarks against previous results
    if latest_previous_dir:
        logging.info(f"Comparing with previous results from: {latest_previous_dir}")
        changes = compare_results(
            Path(OUTPUT_DIR) / "phastft", latest_previous_dir, START, END
        )
        for n, change in changes.items():
            status = "improvement" if change < 0 else "regression"
            logging.info(f"N = 2^{n}: {abs(change):.2f}% {status}")
    else:
        logging.info("No previous results found for comparison.")

    # Plot benchmark results
    plot_benchmark_results(["phastft", "rustfft", "fftwrb"], START, END, history_dirs)

    # Archive current results
    archive_current_results()


if __name__ == "__main__":
    main()
