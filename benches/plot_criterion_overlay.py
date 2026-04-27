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

Each plot is a grouped bar chart: bar height = series median runtime divided
by the group's baseline median at that size, whiskers = IQR in the same
normalized space, dashed `y=1.0` reference line. Sizes are split into a
small-N and large-N half so the bar layout stays readable.

This script exists because criterion regenerates `<group>/report/{lines,
violin}.svg` at the end of each `criterion_main!` using only the IDs
registered in that process. The PhastFT / RustFFT / FFTW modes live in
separate `[[bench]]` binaries so the per-group report is clobbered on each
run. Here we walk the per-ID `sample.json` files criterion keeps on disk
across runs and emit a proper cross-binary comparison.

## Group naming convention

Bench source files in `benches/common/mod.rs::groups` use snake_case
(`c2c_forward_f32`, `r2c_f32`, `kernel_bit_reversal_f32`, …) so the names
round-trip cleanly through criterion's filename sanitizer, shell args, and
tab-completion. This script humanizes them at plot time via `GROUPS` —
chart titles read "C2C Forward (f32)".

## Per-group baseline auto-selection

Without `--baseline`, each group uses its registered baseline in `GROUPS`
(`RustFFT` for `c2c_*`, `realfft` for `r2c_*` / `c2r_*`, etc.).
`--baseline X` overrides for every group. Groups not in the registry are
skipped with a warning so missing entries surface immediately.

## Usage

    uv run benches/plot_criterion_overlay.py
    uv run benches/plot_criterion_overlay.py --groups c2c_forward_f32,r2c_f32
    uv run benches/plot_criterion_overlay.py --baseline "PhastFT DIT"
    uv run benches/plot_criterion_overlay.py --out-dir target/overlays

Outputs: `criterion_overlay_<group>_<log2_lo>_<log2_hi>.{png,svg}` in the
output directory, two files per group (small/large halves).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import bytes2human


@dataclass(frozen=True)
class GroupSpec:
    """Per-group plot config. `title` is the humanized chart title;
    `baseline` is the series whose median normalizes the others (`None`
    means single-series — skip ratio plot)."""
    title: str
    baseline: str | None


# Single source of truth for criterion group metadata. Keys are the
# snake_case group names declared in `benches/common/mod.rs::groups`.
# Groups discovered on disk but absent from this table render a warning
# and are skipped — the registry doubles as a "what groups are known"
# list so unregistered output surfaces immediately.
GROUPS: dict[str, GroupSpec] = {
    "c2c_forward_f32":          GroupSpec("C2C Forward (f32)",   "RustFFT"),
    "c2c_forward_f64":          GroupSpec("C2C Forward (f64)",   "RustFFT"),
    "c2c_inverse_f32":          GroupSpec("C2C Inverse (f32)",   "RustFFT"),
    "c2c_inverse_f64":          GroupSpec("C2C Inverse (f64)",   "RustFFT"),
    "r2c_f32":                  GroupSpec("R2C (f32)",           "realfft"),
    "r2c_f64":                  GroupSpec("R2C (f64)",           "realfft"),
    "c2r_f32":                  GroupSpec("C2R (f32)",           "realfft"),
    "c2r_f64":                  GroupSpec("C2R (f64)",           "realfft"),
    "planner_f32":              GroupSpec("Planner (f32)",       "RustFFT"),
    "planner_f64":              GroupSpec("Planner (f64)",       "RustFFT"),
    "planner_mode_f32":         GroupSpec("Planner Mode (f32)",  "Heuristic"),
    "planner_mode_f64":         GroupSpec("Planner Mode (f64)",  "Heuristic"),
    "kernel_bit_reversal_f32":  GroupSpec("Bit Reversal (f32)",  "BRAVO"),
    "kernel_bit_reversal_f64":  GroupSpec("Bit Reversal (f64)",  "BRAVO"),
    "kernel_deinterleave_f32":  GroupSpec("Deinterleave (f32)",  None),
    "kernel_deinterleave_f64":  GroupSpec("Deinterleave (f64)",  None),
    "kernel_combine_re_im_f32": GroupSpec("Combine Re/Im (f32)", None),
    "kernel_combine_re_im_f64": GroupSpec("Combine Re/Im (f64)", None),
}

# Series styling. Color is the canonical Okabe-Ito assignment per series;
# sort_index pins left-to-right bar order (lower = leftmost). PhastFT R2C/C2R
# reuse the PhastFT DIT orange and `realfft` reuses the RustFFT blue — they
# live in disjoint groups so the colors never collide on a single chart.
SERIES_REGISTRY: dict[str, tuple[str, int]] = {
    "RustFFT":       ("#0072B2", 0),
    "realfft":       ("#0072B2", 0),
    "PhastFT DIT":   ("#D55E00", 1),
    "PhastFT R2C":   ("#D55E00", 1),
    "PhastFT C2R":   ("#D55E00", 1),
    "FFTW Estimate": ("#009E73", 2),
    "FFTW Measure":  ("#E69F00", 3),
    "FFTW Conserve": ("#CC79A7", 4),
    "Heuristic":     ("#0072B2", 0),
    "Tune":          ("#D55E00", 1),
    "BRAVO":         ("#0072B2", 0),
    "COBRAVO":       ("#D55E00", 1),
    "deinterleave":  ("#D55E00", 0),
    "combine_re_im": ("#D55E00", 0),
}
_UNKNOWN_COLOR = "#7A7A7A"

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


def humanize(group_name: str) -> str:
    spec = GROUPS.get(group_name)
    if spec is None:
        return group_name
    return spec.title


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
    """Baseline first, then series sorted by SERIES_REGISTRY index, then
    unknown series alphabetically."""
    def key(s: str) -> tuple[int, str]:
        entry = SERIES_REGISTRY.get(s)
        # Unknown series sort after all known ones.
        return (entry[1] if entry is not None else 999, s)

    ordered = sorted(available, key=key)
    if baseline in ordered:
        ordered.remove(baseline)
        ordered.insert(0, baseline)
    return ordered


def _color_for(series_name: str) -> str:
    entry = SERIES_REGISTRY.get(series_name)
    if entry is not None:
        return entry[0]
    return _UNKNOWN_COLOR


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
    colors = [_color_for(s) for s in series_names]

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
    # patch heights because pandas silently fills NaN with 0.0 in the
    # plotted patches, which would otherwise show "0.00" over missing-data
    # slots.
    bar_containers = [
        c for c in ax.containers
        if hasattr(c, "patches") and isinstance(c, mpl.container.BarContainer)
    ]
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

    ax.set_title(f"{humanize(group_name)}: Runtime relative to {baseline}")
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
            f"warn: group '{group_name}' has no '{baseline}' data — "
            f"skipping (nothing to normalize against; pass --baseline <s> "
            f"to pick a different anchor)",
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

    for half in halves:
        if not half:
            continue
        n_lo = int(math.log2(half[0]))
        n_hi = int(math.log2(half[-1]))
        out_stem = out_dir / f"criterion_overlay_{group_name}_{n_lo}_{n_hi}"
        plot_half(group_name, baseline, series_names, group_data, half, out_stem)
        active = sum(
            1 for s in series_names if any(sz in group_data[s] for sz in half)
        )
        print(
            f"wrote {out_stem.with_suffix('.png')} "
            f"({len(half)} sizes, {active} series active)"
        )


def resolve_baseline(group_name: str, override: str | None) -> str | None:
    """`override` (the CLI `--baseline`) wins. Otherwise consult `GROUPS`;
    unknown groups return `None`, which signals "skip" upstream — better
    to fail loud than to silently normalize against the wrong baseline."""
    if override is not None:
        return override
    spec = GROUPS.get(group_name)
    if spec is None:
        return None
    return spec.baseline


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
        help="Comma-separated snake_case group names to plot (default: all)",
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
        default=None,
        help=(
            "Series whose median normalizes every other series. If unset, "
            "each group uses its GROUP_DEFAULTS entry (RustFFT for complex, "
            "realfft for R2C/C2R, etc.); explicit value applies to every "
            "group."
        ),
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

        baseline = resolve_baseline(group_name, args.baseline)
        if baseline is None:
            if group_name in GROUPS:
                # Registered single-series group — quiet skip.
                print(
                    f"info: group '{group_name}' is single-series — skipping "
                    f"(ratio plot has no anchor)",
                    file=sys.stderr,
                )
            else:
                # Unregistered group on disk — louder warning so new benches
                # without a registry entry get noticed.
                print(
                    f"warn: group '{group_name}' is not registered in "
                    f"GROUPS — skipping (add a GroupSpec to "
                    f"plot_criterion_overlay.py to plot it)",
                    file=sys.stderr,
                )
            continue

        group_data: dict[str, dict[int, tuple[int, float, float, float]]] = {}
        for series_dir in discover_series(group_dir):
            points = load_series(series_dir)
            if points:
                group_data[series_dir.name] = points

        if not group_data:
            print(f"warn: no measurements under {group_dir}", file=sys.stderr)
            continue

        plot_group(group_name, group_data, baseline, args.out_dir)
        any_plotted = True

    if wanted and not any_plotted:
        print(
            f"error: none of the requested groups matched: {sorted(wanted)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
