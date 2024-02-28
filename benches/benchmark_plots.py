"""
Plot benchmark results for FFTW3, RustFFT, and PhastFT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import bytes2human, find_directory

plt.style.use("fivethirtyeight")


def read_file(filepath: str) -> list[float]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            y.append(float(line))

    return y


def get_figure_of_interest(vals: list[float]) -> float:
    return np.min(vals)


def build_and_clean_data(
    root_benchmark_dir: str, n_range: range, lib_name: str
) -> list[float]:
    data = []

    for n in n_range:
        y = read_file(f"{root_benchmark_dir}/{lib_name}/size_{n}")
        y_k = get_figure_of_interest(y)
        data.append(y_k)

    return data


def plot(data: dict[str, list], n_range: range) -> None:
    index = [bytes2human(2**n * (128 / 8)) for n in n_range]

    y0 = np.asarray(data["fftw3"])
    y1 = np.asarray(data["phastft"])
    y2 = np.asarray(data["rustfft"])
    y3 = np.asarray(data["fftwrb"])

    y0 /= y2
    y1 /= y2
    y3 /= y2

    df = pd.DataFrame(
        {
            "RustFFT": np.ones(len(index)),
            "FFTW3": y0,
            "PhastFT": y1,
            "FFTW3-RB": y3,
        },
        index=index,
    )

    df.plot(kind="bar", linewidth=2, rot=0)
    plt.title("PhastFT vs. FFTW3 vs. FFTW3 RB vs. RustFFT", fontsize=14)
    plt.xticks(fontsize=8, rotation=-45)
    plt.xlabel("size of input", fontsize=10)
    plt.ylabel("Execution Time Ratio\n(relative to RustFFT)", fontsize=11)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"benchmarks_bar_plot_{n_range.start}_{n_range.stop -1}.png", dpi=600)
    plt.show()


def main():
    """Entry point... yay"""
    lib_names = ("rustfft", "phastft", "fftw3", "fftwrb")
    ranges = (range(4, 13), range(13, 30))

    for n_range in ranges:
        all_data = {}

        for lib in lib_names:
            root_folder = find_directory()
            if root_folder is None:
                raise FileNotFoundError("unable to find the benchmark data directory")

            data = build_and_clean_data(root_folder, n_range, lib)
            all_data[lib] = data

        assert (
            len(all_data["rustfft"])
            == len(all_data["fftw3"])
            == len(all_data["phastft"])
            == len(all_data["fftwrb"])
        )
        plot(all_data, n_range)


if __name__ == "__main__":
    main()
