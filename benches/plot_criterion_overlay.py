# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
# ]
# ///
"""
Render one bar chart per criterion group aggregating every series on disk.

Parallels benchmark_plots.py's output style: grouped bars per input size, each
bar is a series' median runtime divided by the baseline's median at the same
size, whiskers are the IQR in the same normalized space, and a dashed y=1.0
line anchors the baseline. Sizes are split into a small-N and large-N half so
the bar layout stays readable when many series are plotted together.

This script exists because criterion regenerates `<group>/report/{lines,violin}.svg`
at the end of each `criterion_main!` using only the IDs registered in that
process. The PhastFT / RustFFT / FFTW modes live in separate `[[bench]]`
binaries so the per-group report is clobbered on each run. Here we walk the
per-ID `sample.json` files criterion keeps on disk across runs and emit a
proper cross-binary comparison.

Run from the repo root (default) or pass --criterion-dir:

    uv run benches/plot_criterion_overlay.py
    uv run benches/plot_criterion_overlay.py --groups "Forward f32,Forward f64"
    uv run benches/plot_criterion_overlay.py --baseline "PhastFT DIT"

Outputs: `criterion_overlay_<group>_<log2_lo>_<log2_hi>.{png,svg}` in the
current working directory, two files per group (small/large halves).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import bytes2human


# Okabe-Ito colorblind-safe palette. PhastFT/RustFFT assignments match
# benchmark_plots.py; the three FFTW modes get three more Okabe-Ito entries
# so all five series stay distinguishable when plotted together.
PALETTE = {
    "RustFFT":       "#0072B2",
    "PhastFT DIT":   "#D55E00",
    "FFTW Estimate": "#009E73",
    "FFTW Measure":  "#E69F00",
    "FFTW Conserve": "#CC79A7",
}
_FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#000000"]

# Left-to-right bar order. The baseline floats to index 0 regardless so the
# y=1.0 reference bar anchors the visual comparison.
SERIES_ORDER = [
    "RustFFT",
    "PhastFT DIT",
    "FFTW Estimate",
    "FFTW Measure",
    "FFTW Conserve",
]


_PARAM_DIR_RE = re.compile(r"^\d+$")


def _configure_style() -> None:
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


def discover_groups(criterion_dir: Path) -> list[Path]:
    return sorted(
        p for p in criterion_dir.iterdir()
        if p.is_dir() and p.name != "report"
    )


def discover_series(group_dir: Path) -> list[Path]:
    return sorted(
        p for p in group_dir.iterdir()
        if p.is_dir() and p.name != "report"
    )


def load_sample(param_dir: Path) -> tuple[int, int, float, float, float] | None:
    """Return (elements, bytes, median_ns_per_iter, q1, q3) or None.

    Median and IQR are percentiles of per-iter times (`times[i]/iters[i]`)
    read from criterion's `sample.json`, matching benchmark_plots.py's
    median+IQR convention.
    """
    sample_path = param_dir / "new" / "sample.json"
    if not sample_path.is_file():
        return None

    try:
        sample = json.loads(sample_path.read_text())
    except json.JSONDecodeError as e:
        print(f"warn: {sample_path}: {e}", file=sys.stderr)
        return None

    iters = np.asarray(sample.get("iters", []), dtype=float)
    times = np.asarray(sample.get("times", []), dtype=float)
    if iters.size == 0 or iters.size != times.size:
        return None

    per_iter = times / iters
    q1, median, q3 = np.percentile(per_iter, [25, 50, 75])

    elements: int | None = None
    byte_count: int | None = None
    bench_path = param_dir / "new" / "benchmark.json"
    if bench_path.is_file():
        try:
            bench = json.loads(bench_path.read_text())
            eb = bench.get("throughput", {}).get("ElementsAndBytes", {})
            elements = eb.get("elements")
            byte_count = eb.get("bytes")
        except json.JSONDecodeError:
            pass

    if elements is None:
        if _PARAM_DIR_RE.match(param_dir.name):
            elements = int(param_dir.name)
        else:
            print(f"warn: skipping non-integer param '{param_dir.name}'", file=sys.stderr)
            return None

    if byte_count is None:
        # Fallback assumes complex<f64> (16 bytes/elem); real data comes from
        # benchmark.json in practice, since every bench in benches/ sets
        # `Throughput::ElementsAndBytes`.
        byte_count = elements * 16

    return elements, byte_count, float(median), float(q1), float(q3)


def load_series(series_dir: Path) -> dict[int, tuple[int, float, float, float]]:
    """Return {elements: (bytes, median, q1, q3)} for a single series."""
    out: dict[int, tuple[int, float, float, float]] = {}
    for param_dir in series_dir.iterdir():
        if not param_dir.is_dir():
            continue
        row = load_sample(param_dir)
        if row is not None:
            elements, byte_count, med, q1, q3 = row
            out[elements] = (byte_count, med, q1, q3)
    return out


def order_series(available: list[str], baseline: str) -> list[str]:
    """Baseline first, then known series in SERIES_ORDER, then unknown
    series in alphabetical order."""
    known = [s for s in SERIES_ORDER if s in available]
    unknown = sorted(set(available) - set(SERIES_ORDER))
    ordered = known + unknown
    if baseline in ordered:
        ordered.remove(baseline)
        ordered.insert(0, baseline)
    return ordered


def _color_for(series_name: str, fallback_idx: int) -> str:
    return PALETTE.get(series_name, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])


def plot_half(
    group_name: str,
    baseline: str,
    series_names: list[str],
    group_data: dict[str, dict[int, tuple[int, float, float, float]]],
    sizes: list[int],
    out_stem: Path,
) -> None:
    baseline_data = group_data[baseline]

    index_labels: list[str] = []
    medians: dict[str, list[float]] = {s: [] for s in series_names}
    err_low: dict[str, list[float]] = {s: [] for s in series_names}
    err_high: dict[str, list[float]] = {s: [] for s in series_names}

    for size in sizes:
        b, r_med, _, _ = baseline_data[size]
        index_labels.append(bytes2human(b))
        for s in series_names:
            point = group_data[s].get(size)
            if point is None:
                medians[s].append(np.nan)
                err_low[s].append(0.0)
                err_high[s].append(0.0)
                continue
            _, med, q1, q3 = point
            medians[s].append(med / r_med)
            err_low[s].append((med - q1) / r_med)
            err_high[s].append((q3 - med) / r_med)

    df = pd.DataFrame(medians, index=index_labels)
    yerr = {s: np.vstack([err_low[s], err_high[s]]) for s in series_names}
    fallback_idx = 0
    colors: list[str] = []
    for s in series_names:
        if s in PALETTE:
            colors.append(PALETTE[s])
        else:
            colors.append(_color_for(s, fallback_idx))
            fallback_idx += 1

    fig, ax = plt.subplots(figsize=(11, 5.6))

    df.plot(
        kind="bar",
        ax=ax,
        color=colors,
        width=0.82,
        edgecolor="white",
        linewidth=0.8,
        rot=0,
        zorder=3,
        legend=False,
        yerr=yerr,
        error_kw={"elinewidth": 0.9, "ecolor": "#2F2F2F", "capsize": 2.0},
    )

    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Dashed reference line — baseline bars land here by construction.
    ax.axhline(
        1.0,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        alpha=0.6,
        zorder=2,
        label="_nolegend_",
    )

    # Value labels above each bar. Read ratios from the DataFrame instead of
    # patch heights because pandas silently fills NaN with 0.0 in the plotted
    # patches, which would otherwise show "0.00" over missing-data slots.
    bar_containers = [c for c in ax.containers if hasattr(c, "patches") and isinstance(c, mpl.container.BarContainer)]
    for i, container in enumerate(bar_containers):
        col_values = df.iloc[:, i].to_numpy()
        labels = [f"{v:.2f}" if np.isfinite(v) else "" for v in col_values]
        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=7,
            color="#5A5A5A",
        )

    high_err_stack = np.vstack([err_high[s] for s in series_names]).T
    top = float(np.nanmax(df.values + high_err_stack))
    ax.set_ylim(0, max(top * 1.16, 1.16))

    ax.set_title(f"{group_name}: Runtime relative to {baseline}")
    ax.set_xlabel("Input size")
    ax.set_ylabel(
        f"Runtime relative to {baseline} median\n(median ± IQR, lower is faster)"
    )

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.legend(
        loc="upper left",
        ncol=len(series_names),
        columnspacing=1.4,
        handlelength=1.2,
        handleheight=1.0,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_group(
    group_name: str,
    group_data: dict[str, dict[int, tuple[int, float, float, float]]],
    baseline: str,
    out_dir: Path,
) -> None:
    if baseline not in group_data or not group_data[baseline]:
        print(
            f"warn: group '{group_name}' has no '{baseline}' data — skipping "
            f"(nothing to normalize against; add --baseline <series> to pick a different anchor)",
            file=sys.stderr,
        )
        return

    series_names = order_series(list(group_data.keys()), baseline)
    sizes = sorted(group_data[baseline].keys())

    mid = len(sizes) // 2
    halves: list[list[int]] = []
    if mid > 0:
        halves.append(sizes[:mid])
    halves.append(sizes[mid:])

    group_slug = _sanitize(group_name)
    for half in halves:
        if not half:
            continue
        n_lo = int(math.log2(half[0]))
        n_hi = int(math.log2(half[-1]))
        out_stem = out_dir / f"criterion_overlay_{group_slug}_{n_lo}_{n_hi}"
        plot_half(group_name, baseline, series_names, group_data, half, out_stem)
        active = sum(1 for s in series_names if any(sz in group_data[s] for sz in half))
        print(f"wrote {out_stem.with_suffix('.png')} ({len(half)} sizes, {active} series active)")


def _sanitize(group_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", group_name).strip("_").lower()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Root of criterion output (default: target/criterion)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group names to plot (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for output PNG/SVG files (default: cwd)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="RustFFT",
        help="Series whose median normalizes every other series (default: RustFFT).",
    )
    args = parser.parse_args(argv)

    if not args.criterion_dir.is_dir():
        print(f"error: {args.criterion_dir} is not a directory", file=sys.stderr)
        return 1

    wanted: set[str] | None = None
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",") if g.strip()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    any_plotted = False
    for group_dir in discover_groups(args.criterion_dir):
        group_name = group_dir.name
        if wanted is not None and group_name not in wanted:
            continue

        group_data: dict[str, dict[int, tuple[int, float, float, float]]] = {}
        for series_dir in discover_series(group_dir):
            points = load_series(series_dir)
            if points:
                group_data[series_dir.name] = points

        if not group_data:
            print(f"warn: no measurements under {group_dir}", file=sys.stderr)
            continue

        plot_group(group_name, group_data, args.baseline, args.out_dir)
        any_plotted = True

    if wanted and not any_plotted:
        print(f"error: none of the requested groups matched: {sorted(wanted)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
