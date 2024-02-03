import numpy as np
import pandas as pd
import pyfftw
import time
import csv
import matplotlib.pyplot as plt

from utils import bytes2human
from pybindings import fft

plt.style.use("fivethirtyeight")


def gen_random_signal(dim: int) -> np.ndarray:
    return np.asarray(
        np.random.randn(dim) + 1j * np.random.randn(dim),
        dtype="complex128",
    )


def main() -> None:
    with open("elapsed_times.csv", "w", newline="") as csvfile:
        fieldnames = ["n", "phastft_time", "numpy_fft_time", "pyfftw_fft_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for n in range(12, 29):
            print(f"n = {n}")
            big_n = 1 << n
            s = gen_random_signal(big_n)

            a_re = [None] * len(s)
            a_im = [None] * len(s)

            for i, val in enumerate(s):
                a_re[i] = val.real
                a_im[i] = val.imag

            a_re = np.asarray(a_re, dtype=np.float64)
            a_im = np.asarray(a_im, dtype=np.float64)

            start = time.time()
            fft(a_re, a_im)
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
            b = pyfftw.interfaces.numpy_fft.fft(a)
            pyfftw_elapsed = round((time.time() - start) * 10**6)
            print(f"pyFFTW completed in {pyfftw_elapsed} us")

            np.testing.assert_allclose(b, actual, rtol=1e-3, atol=0)

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
    plot_elapsed_times(loaded_data)
    grouped_bar_plot(loaded_data)


def read_csv_to_dict(file_path: str) -> dict:
    data = {"n": [], "phastft_time": [], "numpy_fft_time": [], "pyfftw_fft_time": []}
    with open(file_path, newline="") as csvfile:
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
    index = [bytes2human(2**n * (128 / 8)) for n in data["n"]]
    np_fft_timings = np.asarray(data["numpy_fft_time"])
    pyfftw_timings = np.asarray(data["pyfftw_fft_time"])
    phastft_timings = np.asarray(data["phastft_time"])

    plt.plot(index, np_fft_timings, label="NumPy FFT", lw=0.8)
    plt.plot(index, pyfftw_timings,  label="PyFFTW FFT", lw=0.8)
    plt.plot(index, phastft_timings, label="PhastFT", lw=0.98)

    plt.title("FFT Elapsed Times Comparison")
    plt.xticks(fontsize=9, rotation=-45)
    plt.yticks(fontsize=9)
    plt.xlabel("size of input")
    plt.ylabel("time (us)")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("py_benchmarks.png", dpi=600)


def grouped_bar_plot(data: dict):
    index = data["n"]
    index = [bytes2human(2**n * (128 / 8)) for n in index]
    np_fft_timings = np.asarray(data["numpy_fft_time"])
    pyfftw_timings = np.asarray(data["pyfftw_fft_time"])  # / np_fft_timings
    phastft_timings = np.asarray(data["phastft_time"])  # / np_fft_timings

    plt.figure()
    df = pd.DataFrame(
        {
            "NumPy fft": np.ones(len(index)),
            "pyFFTW": pyfftw_timings / np_fft_timings,
            "PhastFT": phastft_timings / np_fft_timings,
        },
        index=index,
    )

    ax = df.plot(kind="bar", linewidth=3, rot=0)
    plt.title("FFT Elapsed Times Comparison")
    plt.xticks(fontsize=9, rotation=-45)
    plt.yticks(fontsize=9)
    plt.xlabel("size of input")
    plt.ylabel("time taken (relative to NumPy FFT)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("py_benchmarks_bar_plot.png", dpi=600)


if __name__ == "__main__":
    main()
