"""Shared helpers of the image analysis scripts (eda/i5_*.py).

Loaded by the scripts ``eda/i5_*.py`` by file path (``importlib``), because ``python -I`` does not put the script
directory on ``sys.path``. Always start from the repository root: ``python -I eda/<script>.py``.

Rules enforced here technically:

* Test images are read only by scripts that output aggregates or row-level statistics. There is no function here that
  displays an image or writes image content as an image file; ``assert_not_test`` protects every place where files are
  selected for viewing.
* Row-level statistics (with trace_id) live only under ``artifacts/eda/private/`` (gitignored).
* Verbatim data contents (transcriptions, question texts) go only to a private, gitignored notes folder.
* EXIF: only tag NAMES and groups are counted, no location or device values.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
IMG = DATA / "competition_data" / "images"
ART = COMP / "artifacts" / "eda"
PRIV = ART / "private"
FIG = ART / "fig"
JSON_PATH = ART / "i5_images.json"
STATS_PATH = PRIV / "i5_image_stats.parquet"
PARTS_DIR = PRIV / "i5_stats_parts"

FOLDERS = ["train", "test", "unlabelled", "questions"]
PHOTO_FOLDERS = ["train", "test", "unlabelled"]

# Documented random sample of the unlabelled pool for the expensive statistics.
UNL_SAMPLE_SEED = 20261010
UNL_SAMPLE_N = 10000


def load_table(name: str) -> pd.DataFrame:
    """Root CSV (English topics) as strings; empty cells stay ''."""
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)


def unl_sample_files() -> list[str]:
    """The fixed random sample of the unlabelled files (sorted file list, fixed seed)."""
    files = sorted(os.listdir(IMG / "unlabelled"))
    rng = np.random.default_rng(UNL_SAMPLE_SEED)
    idx = np.sort(rng.choice(len(files), size=min(UNL_SAMPLE_N, len(files)), replace=False))
    return [files[i] for i in idx]


def assert_not_test(path: str | os.PathLike) -> None:
    """Raises if a path lies in images/test/; call before every selection for viewing."""
    p = str(path).replace("\\", "/")
    if "/images/test/" in p or p.startswith("images/test/"):
        raise PermissionError(f"a test image must not be selected for viewing: {p}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


def _clean(obj):
    """NaN/Inf -> None, so that the JSON file stays valid."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    return obj


def update_json(block: str, payload: dict) -> None:
    """Write/replace a block in artifacts/eda/i5_images.json (atomically)."""
    ART.mkdir(parents=True, exist_ok=True)
    data = {}
    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    data[block] = _clean(payload)
    tmp = JSON_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(JSON_PATH)


def update_private_md(path: Path, section: str, markdown: str, head: str) -> None:
    """Replace a section in a private Markdown file (gitignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else head
    begin, end = f"<!-- BEGIN {section} -->", f"<!-- END {section} -->"
    block = f"{begin}\n{markdown.rstrip()}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), flags=re.S)
    if pattern.search(text):
        text = pattern.sub(lambda _m: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def quantiles(x, qs=(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    out = {f"q{int(round(q * 100)):03d}": float(np.quantile(x, q)) for q in qs}
    out |= {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}
    return out


def counts_dict(series: pd.Series, top: int | None = None) -> dict:
    vc = series.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    return {str(k): int(v) for k, v in vc.items()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, mid - half)), float(min(1.0, mid + half)))


def hr(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100, flush=True)


# ----------------------------------------------------------------------------- Figures
# Only distribution plots, never image contents. Colours: the first three categorical slots of the reference palette
# (blue, orange, aqua), checked for three series in all pairings (aqua is below 3:1 contrast, so the series also carry
# different line styles and there is always a legend plus a table in the report).
SERIES = {"train": "#2a78d6", "test": "#eb6834", "unlabelled": "#1baf7a"}
SERIES_LS = {"train": "-", "test": (0, (5, 2)), "unlabelled": (0, (1.5, 1.5))}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "xtick.major.size": 0, "ytick.major.size": 0, "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    })
    return plt


def density_lines(ax, groups: dict, bins, title: str, xlabel: str, logx: bool = False, cumulative: bool = False):
    """One density or distribution line per series (step line from the histogram, normalised to area 1)."""
    for name, vals in groups.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        h, edges = np.histogram(v, bins=bins)
        if cumulative:
            y = np.concatenate([[0], np.cumsum(h) / max(1, h.sum())])
            ax.plot(edges, y, color=SERIES.get(name, INK2), linestyle=SERIES_LS.get(name, "-"), label=f"{name} (n={len(v)})")
        else:
            width = np.diff(np.log10(edges)) if logx else np.diff(edges)
            y = h / max(1, h.sum()) / width
            ax.step(edges, np.concatenate([y, y[-1:]]), where="post", color=SERIES.get(name, INK2),
                    linestyle=SERIES_LS.get(name, "-"), label=f"{name} (n={len(v)})")
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative share" if cumulative else "density")


def load_stats(columns: list[str] | None = None) -> pd.DataFrame:
    """The statistics cache (one row per image file)."""
    return pd.read_parquet(STATS_PATH, columns=columns)
