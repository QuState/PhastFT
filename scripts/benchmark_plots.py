"""
Plot benchmark results for FFTW3, RustFFT, and PhastFT
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("fivethirtyeight")


SYMBOLS = {
    'customary'     : ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
    'customary_ext' : ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa',
                       'zetta', 'iotta'),
    'iec'           : ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
    'iec_ext'       : ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi',
                       'zebi', 'yobi'),
}


def read_file(filepath: str) -> list[int]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            y.append(int(line))

    return y


def get_figure_of_interest(vals: list[int]) -> float:
    return np.median(vals)


def build_and_clean_data(root_benchmark_dir: str, *names) -> defaultdict[str, list]:
    libs = ("rustfft", "phastft")
    n_range = range(12, 30)

    data = defaultdict(list)

    for lib in libs:
        for n in n_range:
            y = read_file(f"{root_benchmark_dir}/{lib}/size_{n}")
            y_k = get_figure_of_interest(y)
            data[lib].append(y_k)

    return data

def plot_lines(data: defaultdict[str, list]) -> None:
    index = [bytes2human(2**n * (128 / 8)) for n in range(12, 30)]
    plt.figure()

    y0 = np.asarray(data["phastft"])
    y1 = np.asarray(data["rustfft"])
    y0 = y1/y0

    df = pd.DataFrame(
        {
            "PhastFT": y0,
            "RustFFT": np.ones(len(index)),
        },
        index=index,
    )

    df.plot(kind='bar', linewidth=3, rot=0)

    plt.xticks(fontsize=9, rotation=-45)
    plt.yticks(fontsize=9)
    plt.xlabel("size of input")
    plt.ylabel("speedup (relative to RustFFT)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("benchmarks_bar_plot.png", dpi=600)


# Source: https://stackoverflow.com/a/1094933
def bytes2human(n, format='%(value).1f %(symbol)s', symbols='customary'):
    """
    Convert n bytes into a human-readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: https://goo.gl/kTQMs
    """
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)



if __name__ == "__main__":
    # y = read_file("benchmark-data.2024.02.02.11-02-07/phastft/size_16")
    data = build_and_clean_data("benchmark-data.2024.02.02.11-43-10")
    # print(data)
    plot_lines(data)