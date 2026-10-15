# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
# ]
# ///
"""
Plot benchmark results for FFTW3, RustFFT, and PhastFT

Run with: `uv run benchmark_plots.py`
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import bytes2human, discover_sizes, find_directory

# Okabe-Ito colorblind-safe palette. 
PALETTE = {
    "RustFFT":  "#0072B2",
    "PhastFT":  "#D55E00",
    "FFTW3-RB": "#009E73",
}


def _configure_style() -> None:
    """Apply a clean whitegrid theme."""
    mpl.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           11,
        "axes.titlesize":      14,
        "axes.titleweight":    "semibold",
        "axes.titlepad":       14,
        "axes.labelsize":      11,
        "axes.labelweight":    "medium",
        "axes.labelpad":       8,
        "axes.labelcolor":     "#2F2F2F",
        "axes.edgecolor":      "#C9C9C9",
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.axisbelow":      True,
        "xtick.color":         "#4A4A4A",
        "ytick.color":         "#4A4A4A",
        "xtick.major.size":    0,
        "ytick.major.size":    3,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.frameon":      False,
        "legend.fontsize":     10,
        "legend.handlelength": 1.3,
        "grid.linestyle":      "-",
        "grid.color":          "#ECECEC",
        "grid.linewidth":      0.8,
        "savefig.facecolor":   "white",
        "savefig.bbox":        "tight",
    })


def read_file(filepath: str) -> list[float]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            y.append(float(line))

    return y


def summarize(vals: list[float]) -> tuple[float, float, float]:
    """Compute and return (median, q1, q3) so plots can show runtime dispersion"""
    arr = np.asarray(vals, dtype=float)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return float(median), float(q1), float(q3)


def build_and_clean_data(
    root_benchmark_dir: str, ns: list[int], lib_name: str
) -> list[tuple[float, float, float]]:
    data = []

    for n in ns:
        y = read_file(f"{root_benchmark_dir}/{lib_name}/size_{n}")
        data.append(summarize(y))

    return data


def _split_stats(
    stats: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(stats, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def plot(data: dict[str, list], ns: list[int]) -> None:
    index = [bytes2human(2**n * (128 / 8)) for n in ns]

    rust_med, rust_q1, rust_q3     = _split_stats(data["rustfft"])
    phast_med, phast_q1, phast_q3  = _split_stats(data["phastft"])
    fftw_med, fftw_q1, fftw_q3     = _split_stats(data["fftwrb"])

    # Normalize every statistic against RustFFT's median at each size so bars
    # and error whiskers live in the same relative space.
    def norm(med, q1, q3):
        m = med / rust_med
        low = m - q1 / rust_med
        high = q3 / rust_med - m
        return m, low, high

    rust_m, rust_low, rust_high    = norm(rust_med, rust_q1, rust_q3)
    phast_m, phast_low, phast_high = norm(phast_med, phast_q1, phast_q3)
    fftw_m, fftw_low, fftw_high    = norm(fftw_med, fftw_q1, fftw_q3)

    df = pd.DataFrame(
        {
            "RustFFT":  rust_m,
            "PhastFT":  phast_m,
            "FFTW3-RB": fftw_m,
        },
        index=index,
    )

    yerr = {
        "RustFFT":  np.vstack([rust_low, rust_high]),
        "PhastFT":  np.vstack([phast_low, phast_high]),
        "FFTW3-RB": np.vstack([fftw_low, fftw_high]),
    }

    _configure_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    df.plot(
        kind="bar",
        ax=ax,
        color=[PALETTE[c] for c in df.columns],
        width=0.78,
        edgecolor="white",
        linewidth=0.8,
        rot=0,
        zorder=3,
        legend=False,
        yerr=yerr,
        error_kw={"elinewidth": 0.9, "ecolor": "#2F2F2F", "capsize": 2.5},
    )

    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Dashed reference line at y=1.0 to establish RustFFT as the baseline
    ax.axhline(
        1.0,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        alpha=0.6,
        zorder=2,
        label="_nolegend_",
    )

    for container in ax.containers:
        # Skip the invisible errorbar line containers that bar_label can't format.
        if not hasattr(container, "patches"):
            continue
        ax.bar_label(
            container,
            fmt="%.2f",
            padding=3,
            fontsize=8,
            color="#5A5A5A",
        )

    # Headroom above the tallest whisker so value labels don't clip the top spine.
    top = float(max(
        (df.values + np.column_stack([rust_high, phast_high, fftw_high])).max(),
        1.0,
    ))
    ax.set_ylim(0, top * 1.16)

    ax.set_title("PhastFT vs. FFTW3-RB vs. RustFFT")
    ax.set_xlabel("Input size")
    ax.set_ylabel("Runtime relative to RustFFT median\n(median ± IQR, lower is faster)")

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.legend(
        loc="upper right",
        ncol=3,
        columnspacing=1.6,
        handlelength=1.2,
        handleheight=1.0,
    )

    fig.tight_layout()
    fig.savefig(f"benchmarks_bar_plot_{ns[0]}_{ns[-1]}.png", dpi=600)


def main():
    """Entry point... yay"""
    lib_names = ("rustfft", "phastft", "fftwrb")

    root_folder = find_directory()
    if root_folder is None:
        raise FileNotFoundError("unable to find the benchmark data directory")

    sizes = discover_sizes(root_folder, lib_names)
    # Split at the midpoint so small- and large-N results each get their own
    # chart (their absolute ratios live on different visual scales).
    mid = len(sizes) // 2
    groups = [g for g in (sizes[:mid], sizes[mid:]) if g]

    for ns in groups:
        all_data = {
            lib: build_and_clean_data(root_folder, ns, lib) for lib in lib_names
        }
        assert (
            len(all_data["rustfft"])
            == len(all_data["phastft"])
            == len(all_data["fftwrb"])
        )
        plot(all_data, ns)


if __name__ == "__main__":
    main()
