"""Tables, folds and images for the vision-language model runs.

Reads the competition data identically everywhere (dtype=str, line endings normalised), finds the data root
locally, on Kaggle and on rented machines, and loads images at a fixed size. No data access at module level.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

KAGGLE_SLUG = "misconception-detection-in-k-12-student-math-workings"
# Checksums of the trace_id columns (row order of the root CSVs): make sure that folds and stored
# predictions match the loaded table. No row identifiers in the code.
ID_SHA256 = {
    "train": "4c2d012e8aa02f97c3a985f0d85e93eb5f42995286f048ea1df71f249f61cfcb",
    "test": "06674052818c256cbd31427bc408a230be647815897341cb55f70e55c893efe2",
    "unlabelled": "7887b624444e23c63a089efa553d5658faa14bfb3f2e07f8ca87df11c36104da",
}
FOLDS_HASH = "07ec381cc58b2a80"
TEXT_COLS = ("question_text", "correct_answer")


def comp_root() -> Path:
    """Competition root: the folder that contains `src/` and `configs/` (locally and in the Kaggle dataset)."""
    return Path(__file__).resolve().parents[2]


def find_data_root(hint: str | os.PathLike | None = None) -> Path:
    """Find the folder with train.csv and competition_data/: hint, environment, Kaggle, local."""
    candidates = [
        hint,
        os.environ.get("K12_DATA"),
        f"/kaggle/input/competitions/{KAGGLE_SLUG}",
        f"/kaggle/input/{KAGGLE_SLUG}",
        comp_root() / "data",
    ]
    for c in candidates:
        if c and (Path(c) / "train.csv").is_file():
            return Path(c)
    raise FileNotFoundError(f"data root not found; checked: {[str(c) for c in candidates if c]}")


def norm_text(s: str) -> str:
    """Normalise line endings to LF; the carriage-return signature of part of train.csv is deliberately not used."""
    return s.replace("\r\n", "\n").replace("\r", "\n")


def load_table(data_root: Path, name: str) -> pd.DataFrame:
    """train / test / unlabelled from the root CSVs, with a checked row order."""
    df = pd.read_csv(Path(data_root) / f"{name}.csv", dtype=str, keep_default_na=False)
    digest = hashlib.sha256("\n".join(df["trace_id"]).encode()).hexdigest()
    if digest != ID_SHA256[name]:
        raise ValueError(f"{name}.csv has a different row order or different rows than expected ({digest[:12]}…)")
    for c in TEXT_COLS:
        df[c] = df[c].map(norm_text)
    return df


def load_teacher_rows(data_root: Path, name, teacher_dir: Path | None = None, min_conf: float = 0.0) -> pd.DataFrame:
    """Teacher-labelled pool images as rows in the format of train.csv.

    Reads data/derived/teacher/<name>/labels.jsonl (frozen outputs of tools/teacher_api.py; `name` may be a list of
    plans) and joins them with unlabelled.csv through the row index. There is no path to a test row here: the file
    only names positions in the pool. Column `_source` = "teacher"; the images live under images/unlabelled.
    Extra columns: `_conf` (teacher confidence for its root code, NaN if not given), `_answer` (the student's final
    answer verbatim, otherwise ""). Rows the teacher reports as unreadable and rows below `min_conf` are dropped; a
    pool row appears at most once (the later plan wins).
    """
    import json

    names = [name] if isinstance(name, str) else list(name)
    if teacher_dir is not None and len(names) != 1:
        raise ValueError("teacher_dir applies to exactly one plan")
    pool = load_table(data_root, "unlabelled")
    rows: dict[int, dict] = {}
    for nm in names:
        d = Path(teacher_dir) if teacher_dir is not None else comp_root() / "data" / "derived" / "teacher" / nm
        src = d / "labels.jsonl"
        if teacher_dir is None and not src.is_file():
            # On Kaggle the competition folder holds only the code; frozen teacher outputs come from the private
            # dataset of derived files there, under the same relative path.
            import glob

            hits = sorted(glob.glob(f"/kaggle/input/**/derived/teacher/{nm}/labels.jsonl", recursive=True))
            if hits:
                src = Path(hits[0])
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["row"])
            if not 0 <= i < len(pool):
                raise ValueError(f"teacher file {nm}: row {i} lies outside the pool")
            conf = r.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else float("nan")
            if str(r.get("readable", "yes")).strip().lower() == "no" or (min_conf > 0 and not conf >= min_conf):
                rows.pop(i, None)
                continue
            base = pool.iloc[i].to_dict()
            finds = [{"step": f.get("step", ""), "span": "", "category": f.get("code", ""), "severity": f.get("severity", ""),
                      "root": bool(f.get("root")), "text": f.get("text", "")} for f in (r.get("findings") or [])]
            base.update({
                "root_code": str(r["root_code"]),
                "final_answer_correct": "True" if r.get("final_answer_correct") else "False",
                "clean": "True" if not finds else "False",
                "transcription_steps": "\n".join(str(x) for x in (r.get("transcription") or [])),
                "misconceptions": json.dumps(finds, ensure_ascii=False),
                "_source": "teacher",
                "_conf": conf,
                "_answer": str(r.get("student_final_answer") or "").strip(),
                # teacher ranking and probabilities as JSON text (for soft training targets)
                "_top3": json.dumps([str(c) for c in (r.get("top3") or [])[:3]]),
                "_probs": json.dumps([float(v) for v in (r.get("probabilities") or [])[:3]]),
            })
            rows[i] = base
    cols = list(pool.columns) + ["root_code", "final_answer_correct", "clean", "transcription_steps", "misconceptions", "_source", "_conf", "_answer", "_top3", "_probs"]
    return pd.DataFrame([rows[i] for i in sorted(rows)], columns=cols)


def load_taxonomy(data_root: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_root) / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)


def load_folds(n_train: int = 1500) -> np.ndarray:
    """Shared folds, bound to the row positions of train.csv; the hash is checked."""
    df = pd.read_parquet(comp_root() / "configs" / "folds.parquet")
    folds = df.sort_values("row")["fold"].to_numpy().astype(np.int64)
    digest = hashlib.sha256(np.ascontiguousarray(folds).tobytes()).hexdigest()[:16]
    if len(folds) != n_train or digest != FOLDS_HASH:
        raise ValueError(f"folds do not match: {len(folds)} rows, hash {digest}")
    return folds


def image_path(data_root: Path, split: str, file_name: str) -> Path:
    """split: train | test | unlabelled | questions."""
    return Path(data_root) / "competition_data" / "images" / split / file_name


def open_image(path: Path, max_side: int = 1024, min_side: int | None = None) -> Image.Image:
    """RGB image with EXIF orientation applied; long side at most max_side (and at least min_side)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    long_side = max(w, h)
    scale = 1.0
    if long_side > max_side:
        scale = max_side / long_side
    elif min_side and long_side < min_side:
        scale = min_side / long_side
    if scale != 1.0:
        im = im.resize((max(32, round(w * scale)), max(32, round(h * scale))), Image.LANCZOS if scale < 1 else Image.BICUBIC)
    return im
