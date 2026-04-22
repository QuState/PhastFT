"""
Utility functions for plotting that are common to all scripts
"""
import os
import re
from collections.abc import Iterable
from datetime import datetime

_SIZE_FILE_RE = re.compile(r"^size_(\d+)$")

SYMBOLS = {
    "customary": ("B", "K", "M", "G", "T", "P", "E", "Z", "Y"),
    "customary_ext": (
        "byte",
        "kilo",
        "mega",
        "giga",
        "tera",
        "peta",
        "exa",
        "zetta",
        "iotta",
    ),
    "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
    "iec_ext": ("byte", "kibi", "mebi", "gibi", "tebi", "pebi", "exbi", "zebi", "yobi"),
}


def bytes2human(n, format="%(value).0f %(symbol)s", symbols="customary"):
    """
    Convert n bytes into a human-readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: https://goo.gl/kTQMs

    Source: https://stackoverflow.com/a/1094933
    """
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def find_directory(pattern="benchmark-data"):
    current_dir = os.getcwd()

    # List all directories in the current directory
    all_dirs = [
        d
        for d in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, d))
    ]

    # Define the regex pattern for matching
    date_pattern = re.compile(r"\d{4}\.\d{2}\.\d{2}\.\d{2}-\d{2}-\d{2}")

    # Iterate through directories and check if they match the pattern
    matching_dirs = [d for d in all_dirs if pattern in d and date_pattern.search(d)]

    if matching_dirs:
        # Sort directories based on the date in the directory name
        matching_dirs.sort(
            key=lambda x: datetime.strptime(
                date_pattern.search(x).group(), "%Y.%m.%d.%H-%M-%S"
            )
        )
        return os.path.join(
            current_dir, matching_dirs[-1]
        )  # Return the latest matching directory
    else:
        return None  # No matching directory found


def discover_sizes(root_dir: str, lib_names: Iterable[str]) -> list[int]:
    """
    Return sorted n values with data under every `<root_dir>/<lib>/size_<n>`
    """
    common: set[int] | None = None
    for lib in lib_names:
        lib_dir = os.path.join(root_dir, lib)
        if not os.path.isdir(lib_dir):
            raise FileNotFoundError(f"missing library subdirectory: {lib_dir}")
        sizes = {
            int(m.group(1))
            for name in os.listdir(lib_dir)
            if (m := _SIZE_FILE_RE.match(name))
        }
        if not sizes:
            raise FileNotFoundError(f"no size_<n> files under {lib_dir}")
        common = sizes if common is None else common & sizes
    if not common:
        raise RuntimeError(
            f"no sizes common to all libraries in {root_dir}: {list(lib_names)}"
        )
    return sorted(common)
