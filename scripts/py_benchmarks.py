import numpy as np
import pyfftw
import time
import csv
from scipy.stats import unitary_group
import matplotlib.pyplot as plt

from pybindings import fft


def main() -> None:
    with open("elapsed_times.csv", "w", newline="") as csvfile:
        fieldnames = ["n", "phastft_time", "numpy_fft_time", "pyfftw_fft_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for n in range(4, 29):
            print(f"n = {n}")
            big_n = 1 << n
            x = unitary_group.rvs(big_n)

            a_re = x[:, 0].copy().real # np.asarray([float(i) for i in range(big_n)])
            a_im = x[:, 0].copy().imag # np.asarray([float(i) for i in range(big_n)])

            start = time.time()
            fft(a_re, a_im)
            phastft_elapsed = round((time.time() - start) * 10**6)

            a = x[:, 0].copy()

            start = time.time()
            expected = np.fft.fft(a)
            numpy_elapsed = round((time.time() - start) * 10**6)

            actual = np.asarray(
                [
                    complex(z_re, z_im)
                    for (z_re, z_im) in zip(
                        a_re,
                        a_im,
                    )
                ]
            )
            np.testing.assert_allclose(actual, expected)

            arr = x[:, 0].copy()
            a = pyfftw.empty_aligned(big_n, dtype="complex128")
            a[:] = arr
            start = time.time()
            b = pyfftw.interfaces.numpy_fft.fft(a)
            pyfftw_elapsed = round((time.time() - start) * 10**6)

            np.testing.assert_allclose(b, actual)

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
    print(loaded_data)
    plot_elapsed_times(loaded_data)


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
    plt.figure(figsize=(10, 6))
    plt.plot(data["n"], data["phastft_time"], label="PhastFT")
    plt.plot(data["n"], data["numpy_fft_time"], label="NumPy FFT")
    plt.plot(data["n"], data["pyfftw_fft_time"], label="PyFFTW FFT")

    plt.title("FFT Elapsed Times Comparison")
    plt.xlabel("n")
    plt.ylabel("Elapsed Time (microseconds)")
    plt.yscale("log")
    plt.legend()
    plt.savefig("py_benchmarks.png", dpi=600)


if __name__ == "__main__":
    main()
