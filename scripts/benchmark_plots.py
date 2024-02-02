"""
Plot benchmark results for FFTW3, RustFFT, and PhastFT
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("fivethirtyeight")

def read_file(filepath: str) -> list[int]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            y.append(int(line))

    return y


def get_figure_of_interest(vals: list[int]) -> float:
    return np.mean(vals)


def build_and_clean_data(root_benchmark_dir: str, *names) -> defaultdict[str, list]:
    libs = ("rustfft", "phastft")
    n_range = range(4, 27)

    data = defaultdict(list)

    for lib in libs:
        for n in n_range:
            y = read_file(f"{root_benchmark_dir}/{lib}/size_{n}")
            y_k = get_figure_of_interest(y)
            data[lib].append(y_k)

    return data

def plot_lines(data: defaultdict[str, list]) -> None:
    index = list(range(4, 27))
    plt.figure()

    print(len(data["phastft"]))

    df = pd.DataFrame(
        {
            "PhastFT": data["phastft"],
            "RustFFT": data["rustfft"],
        },
        index=index,
    )

    df.plot(kind='bar', linewidth=3, rot=0)
    plt.xticks(fontsize=8)
    plt.xlabel("size")
    plt.ylabel("time (us)")
    plt.yscale("log")
    plt.show()
    # plt.tight_layout(pad=0.0)
    # plt.savefig("benchmarks_bar_plot.png", dpi=600)


if __name__ == "__main__":
    # y = read_file("benchmark-data.2024.02.02.11-02-07/phastft/size_16")
    data = build_and_clean_data("benchmark-data.2024.02.02.11-27-33")
    print(data)
    # plot_lines(data)