import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfftw

pyfftw.interfaces.cache.enable()

from pyphastft import fft

from utils import bytes2human

plt.style.use("fivethirtyeight")


def gen_random_signal(dim: int) -> np.ndarray:
    """Generate a random, complex 1D signal"""
    return np.ascontiguousarray(
        np.random.randn(dim) + 1j * np.random.randn(dim),
        dtype="complex128",
    )


def main() -> None:
    with open("elapsed_times.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["n", "phastft_time", "numpy_fft_time", "pyfftw_fft_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for n in range(4, 29):
            print(f"n = {n}")
            big_n = 1 << n
            s = gen_random_signal(big_n)

            a_re = np.ascontiguousarray(s.real, dtype=np.float64)
            a_im = np.ascontiguousarray(s.imag, dtype=np.float64)

            start = time.time()
            fft(a_re, a_im, "f")
            phastft_elapsed = round((time.time() - start) * 10**6)
            print(f"PhastFT completed in {phastft_elapsed} us")

            a = s.copy()

            start = time.time()
            expected = np.fft.fft(a)
            numpy_elapsed = round((time.time() - start) * 10**6)
            print(f"NumPy fft completed in {numpy_elapsed} us")

            actual = np.asarray(
                [
                    complex(z_re, z_im)
                    for (z_re, z_im) in zip(
                        a_re,
                        a_im,
                    )
                ]
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)

            arr = s.copy()
            a = pyfftw.empty_aligned(big_n, dtype="complex128")
            a[:] = arr
            start = time.time()
            a = pyfftw.interfaces.numpy_fft.fft(a)
            pyfftw_elapsed = round((time.time() - start) * 10**6)
            print(f"pyFFTW completed in {pyfftw_elapsed} us")

            np.testing.assert_allclose(a, actual, rtol=1e-3, atol=0)

            writer.writerow(
                {
                    "n": n,
                    "phastft_time": phastft_elapsed,
                    "numpy_fft_time": numpy_elapsed,
                    "pyfftw_fft_time": pyfftw_elapsed,
                }
            )

    file_path = "elapsed_times.csv"
    loaded_data = read_csv_to_dict(file_path)
    grouped_bar_plot(loaded_data, start=0, end=9)
    grouped_bar_plot(loaded_data, start=9, end=29)


def read_csv_to_dict(file_path: str) -> dict:
    """Read the benchmark results from the csv file and convert it to a dict"""
    data: dict[str, list] = {
        "n": [],
        "phastft_time": [],
        "numpy_fft_time": [],
        "pyfftw_fft_time": [],
    }
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data["n"].append(int(row["n"]))
            data["phastft_time"].append(
                int(row["phastft_time"]) if row["phastft_time"] else None
            )
            data["numpy_fft_time"].append(
                int(row["numpy_fft_time"]) if row["numpy_fft_time"] else None
            )
            data["pyfftw_fft_time"].append(
                int(row["pyfftw_fft_time"]) if row["pyfftw_fft_time"] else None
            )
    return data


def plot_elapsed_times(data: dict) -> None:
    """Plot the timings for all libs using line plots"""
    index = [bytes2human(2**n * (128 / 8)) for n in data["n"]]
    np_fft_timings = np.asarray(data["numpy_fft_time"])
    pyfftw_timings = np.asarray(data["pyfftw_fft_time"])
    phastft_timings = np.asarray(data["phastft_time"])

    plt.plot(index, np_fft_timings, label="NumPy FFT", lw=0.8)
    plt.plot(index, pyfftw_timings, label="pyFFTW", lw=0.8)
    plt.plot(index, phastft_timings, label="pyPhastFT", lw=0.8)

    plt.title("pyPhastFT vs. pyFFTW vs. NumPy FFT")
    plt.xticks(fontsize=8, rotation=-45)
    plt.xlabel("size of input")
    plt.ylabel("time (us)")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("py_benchmarks.png", dpi=600)


def grouped_bar_plot(data: dict, start=0, end=1):
    """Plot the timings for all libs using a grouped bar chart"""
    index = data["n"]
    index = [bytes2human(2**n * (128 / 8)) for n in index]
    np_fft_timings = np.asarray(data["numpy_fft_time"])
    pyfftw_timings = np.asarray(data["pyfftw_fft_time"])  # / np_fft_timings
    phastft_timings = np.asarray(data["phastft_time"])  # / np_fft_timings

    df = pd.DataFrame(
        {
            "NumPy fft": np.ones(len(index)),
            "pyFFTW": pyfftw_timings / np_fft_timings,
            "pyPhastFT": phastft_timings / np_fft_timings,
        },
        index=index,
    )

    title = "pyPhastFT vs. pyFFTW vs. NumPy FFT"
    df[start:end].plot(kind="bar", linewidth=2, rot=0, title=title)
    plt.xticks(fontsize=8, rotation=-45)
    plt.xlabel("size of input")
    plt.ylabel("Execution Time Ratio\n(relative to NumPy FFT)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"py_benchmarks_bar_plot_{start}_{end-1}.png", dpi=600)


if __name__ == "__main__":
    main()
