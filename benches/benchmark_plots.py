"""
Plot benchmark results for FFTW3, RustFFT, and PhastFT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import bytes2human

plt.style.use("fivethirtyeight")


def read_file(filepath: str) -> list[float]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            y.append(float(line))

    return y


def get_figure_of_interest(vals: list[int]) -> float:
    return np.median(vals)


def build_and_clean_data(
    root_benchmark_dir: str, n_range: range, lib_name: str
) -> list[float]:
    data = []

    for n in n_range:
        y = read_file(f"{root_benchmark_dir}/{lib_name}/size_{n}")
        y_k = get_figure_of_interest(y)
        data.append(y_k)

    return data


def plot_lines(data: dict[str, list], n_range: range) -> None:
    index = [bytes2human(2**n * (128 / 8)) for n in n_range]
    plt.figure()

    y0 = np.asarray(data["fftw3"])
    y1 = np.asarray(data["phastft"])
    y2 = np.asarray(data["rustfft"])

    y0 /= y2
    y1 /= y2

    df = pd.DataFrame(
        {
            "FFTW3": y0,
            "PhastFT": y1,
            "RustFFT": np.ones(len(index)),
        },
        index=index,
    )

    df.plot(kind="bar", linewidth=3, rot=0)

    plt.xticks(fontsize=9, rotation=-45)
    plt.yticks(fontsize=9)
    plt.xlabel("size of input")
    plt.ylabel("time taken (relative to RustFFT)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("benchmarks_bar_plot.png", dpi=600)
    # plt.show()


def main():
    lib_names = ("rustfft", "phastft", "fftw3")
    n_range = range(12, 30)

    all_data = {}

    for lib in lib_names:
        data = build_and_clean_data("benchmark-data.2024.02.02.16-45-50", n_range, lib)
        all_data[lib] = data

    assert (
        len(all_data["rustfft"]) == len(all_data["fftw3"]) == len(all_data["phastft"])
    )
    plot_lines(all_data, n_range)


if __name__ == "__main__":
    main()
