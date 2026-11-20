"""Qwen3.5-397B-A17B as the SIXTH channel in the full primary-line recipe — evaluation of a rule fixed IN ADVANCE.

    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -I eda/c107_q397_full.py           # decision (full coverage)
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -I eda/c107_q397_full.py --dry     # code path only, no decision number

THE RULE (fixed before the first run with full coverage; see RULE_TEXT):
  Channel from the three named codes with their probabilities, floor 0.05, as the sixth channel in the primary-line recipe
  (v1r, zs27q, think18nb ×1.4, pr27xm, m3x2_full_cv01; β 0.2; transductive margin alignment 0.75 for classes with at least
  3 rows, per fold; weights cross-fitted on the other four folds) on all 1,500 rows without sample weights. PASSED only if
  all hold: Δ MAP@3 ≥ +0.006 with z ≥ 1.5 (σ over tasks), log-loss gain z ≥ 3, at least 4/5 folds with Δ ≥ 0, and the
  variant with fixed levels 0.6/0.25/0.1 instead of the stated probabilities Δ ≥ +0.003. For information only: alignment as
  one set, without the last step, inductive line. Passed → compute the channel's test side; failed → the channel is dropped.

READING where the wording is open (fixed before the first run with full coverage):
  1. All five criteria apply to the predictions of the FULL recipe (with margin alignment per fold), paired against the
     same recipe without the channel — which reproduces the earlier primary-line anchor exactly (0.8198889 = 7,379/9,000).
  2. σ over tasks = paired bootstrap over question_id (ratio estimator, 10,000 draws, seed 20261030, the same draws for all
     quantities) — for Δ MAP@3 AND for the log-loss gain.
  3. Log-loss gain = mean row log loss without the channel minus with the channel (positive = better; clipping 1e-12).
  4. The thresholds +0.006 / +0.003 and "Δ ≥ 0 per fold" are checked EXACTLY in units of 1/6 per row (MAP@3 row values are
     multiples of 1/6) — no floating-point noise at the boundary; a fold with Δ = 0 counts as "≥ 0".
  Alternative readings of criterion 3 (log loss without the last step; row bootstrap) are reported as a sensitivity. If one
  of them changes the criterion, the script says so explicitly; the decision follows the main reading.

CHANNEL (as in eda/a8_proto_oof.py and the test path tools/make_submission.py --proto): per row the LAST valid record counts
(ok and root_code in the class list); named codes with their stated probabilities, remaining mass spread evenly over the
classes with a training example; rows without a valid answer: uniform over these classes (no effect in the log-linear
part). Variant FIXED LEVELS: stated probabilities discarded, 0.6 / 0.25 / 0.1 for the three named codes. The own conversion is
checked at runtime to be bit-identical to vlm.proto.labels_to_probs (both variants, the same snapshot of the file). Floor
0.05 WITHOUT prior masking, as in the build and test paths (c99, make_submission); the "mask first" arithmetic is reported as a
sensitivity.

RECIPE (constants checked against the blend files): primary line = artifacts/stack_cv01_best_l2.json (channels, floors,
factor, β, last step, checksums); inductive line = artifacts/stack_v1r_zs27t_think18sb_pr27xm_m3x2_ar.json with the M3 channel
over all 1,500 rows (m3x2_full_cv01 instead of the 600-row version m3x2). Per fold k: weights a2_probe_stack.fit_mix (log loss)
only from the other four folds, then think18nb × 1.4; blend make_submission.mix_channels (log-linear + arithmetic share β);
last step transductive (best) = c21.fit_weights on the 300 rows of k with target = frequencies of the other 1,200 rows
(classes with at least 3 rows), strength 0.75; inductive alignment (inductive line) = class weights from the blend of the 1,200 fit rows (mask
first, as in make_submission.oof_mixture), only applied to k. Cross-checked against an independent re-implementation of the
cross-fit (bit-identical Q and θ for both lines).

COVERAGE (without --dry abort with code 3): every one of the 1,500 training rows has a record; no row without a valid answer
is still waiting for the finishing pass (tools/open_teacher.py finish); valid answers at least --min-valid (0.98); the file
ends with a complete record; the records of the earlier stratified measurement stand unchanged at the start of the file
(SHA-256 of the allowlist) and cover exactly the 860 rows of the sample design_stage2.json. The records come from
tools/open_teacher.py, which only reads the table train; row positions and a possible field `table` are checked.

CHECKS (abort with code 4): checksums of the base OOFs and of the blend files; recipe constants = blend files; anchors
without the channel exact (best: 7,379/9,000, per fold and log loss as in stack_cv01_best_l2.json; inductive: 7,308/9,000);
own channel conversion = vlm.proto.labels_to_probs; all predictions finite, normalised, without mass on classes without a
training example.

--dry: the same code path on a partial state, but NO decision number — neither on the console nor in a file; the verdict
is "NOT DECIDABLE (partial coverage)" or, with full coverage already, "NOT EVALUATED (--dry)". The anchors (without the
channel) are checked and shown here as well; the result file is built and serialised, but not written.

OUTPUT (training rows only, only aggregates on the console): artifacts/eda/private/q397_eval/c107_result.json (decision),
c107_rows.npz (row vectors, private); with --dry only c107_dry.json (checks, no number about the channel).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import proto as PR  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A2 = _load("c107_a2", COMP / "eda" / "a2_probe_stack.py")  # floor_probs, fit_mix
C21 = _load("c107_c21", COMP / "eda" / "c21_margin_align.py")  # fit_weights, apply_weights
MS = _load("c107_ms", COMP / "tools" / "make_submission.py")  # mix_channels (log-linear + arithmetic share)

# ------------------------------------------------------------------------------------------------ constants

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "q397_eval"
LABELS_REL = "derived/open_teacher/qwen35-397b/labels.jsonl"  # relative to the data root
DESIGN_REL = "derived/open_teacher/design_stage2.json"
REST_REL = "derived/open_teacher/rows_rest_397b.npy"
LEGACY_SHA256 = "a6f532a88f1a7f17dc2d434743473dd5d3f201bb13200435710ae0d601bcd3f6"  # records of the earlier stratified measurement (allowlist)
LEGACY_ROWS = 860

BOOT, SEED = 10_000, 20261030
EPS = 1e-9  # clipping of the channel logs as in the build and test paths
LL_CLIP = 1e-12  # clipping of the row log loss
ALIGN, ALIGN_MIN = 0.75, 3
N_FOLDS = 5
NEW, NEW_FIXED, FLOOR_NEW = "q397", "q397f", 0.05
LEVELS = (0.6, 0.25, 0.1)
MIN_VALID = 0.98
GROUPS = ("NONE", "seen", "never seen", "new question")
PART_A_ROWS = 358  # part A of train.csv (rows 0–357)

OOF_SHA16 = {"v1r": "e1729c1fddad7ebb", "zs27q": "7a11b94751f6d458", "zs27t": "54e045dfe32c7be1",
             "think18nb": "9c3403b57d07a956", "think18sb": "275cace8a36ca757", "pr27xm": "bf3d1b58abfc57eb",
             "m3x2_full_cv01": "e08f2deefe90a9b3"}
M3_SHA256 = "e08f2deefe90a9b3b903668222cb51aa3a07f6dd80705e8f2c078ff3b2302ef1"
STACK_BEST = ("stack_cv01_best_l2.json", "b2835875298a89247fbc8a4eeaf1cc4177de17af66ef395ffa636b586d1add43")
STACK_SAFE = ("stack_v1r_zs27t_think18sb_pr27xm_m3x2_ar.json", "fbb5c2ba87eec7a77f7537a139eba86e27d0890dc0f6992183b49524906ab1eb")

# line labels (recorded in the reports derived from these recipes)
LINE_BEST = {"label": "primary line (earlier recipe)", "names": ["v1r", "zs27q", "think18nb", "pr27xm", "m3x2_full_cv01"],
             "floors": {"think18nb": 0.01, "pr27xm": 0.05, "m3x2_full_cv01": 0.05}, "think": "think18nb", "factor": 1.4,
             "beta": 0.2, "last": "trans"}
LINE_SAFE = {"label": "inductive line (earlier recipe)", "names": ["v1r", "zs27t", "think18sb", "pr27xm", "m3x2_full_cv01"],
             "floors": {"think18sb": 0.01, "pr27xm": 0.05, "m3x2_full_cv01": 0.05}, "think": None, "factor": 1.0,
             "beta": 0.4, "last": "ind"}
# anchors as sums of the row values in units of 1/6: 0.8198889 = 7,379/9,000 (best), 0.8120 = 7,308/9,000 (inductive)
ANCHOR_UNITS = {"best": 7379, "safe": 7308}

CRITERIA = ("Δ MAP@3 ≥ +0.006", "z ≥ 1.5 (σ over tasks)", "log-loss gain z ≥ 3", "at least 4/5 folds Δ ≥ 0",
            "fixed levels Δ ≥ +0.003")
RULE_TEXT = ("Channel from the three named codes with their probabilities, floor 0.05, as the sixth channel in the primary "
             "recipe (v1r, zs27q, think18nb ×1.4, pr27xm, m3x2_full_cv01; β 0.2; transductive margin alignment 0.75 for classes with "
             "at least 3 rows, per fold; weights cross-fitted on the other four folds) on all 1,500 rows without sample weights. RULE — "
             "PASSED only if all hold: Δ MAP@3 ≥ +0.006 with z ≥ 1.5 (σ over tasks), log-loss gain z ≥ 3, at least 4/5 folds Δ ≥ 0, "
             "variant with fixed levels 0.6/0.25/0.1 instead of the stated probabilities Δ ≥ +0.003.")
READING = ("criteria on the predictions of the full recipe (with margin alignment per fold), paired against the same recipe without "
           "the channel (primary-line anchor)", "σ over tasks: paired bootstrap over question_id, 10,000 draws, seed 20261030, for MAP@3 "
                                  "and log loss", "log-loss gain = log loss without the channel − with the channel (clipping 1e-12)",
           "thresholds and folds exactly in units of 1/6 per row; Δ = 0 counts as ≥ 0")
CONSEQUENCE = {True: "passed: compute the channel's test side (stage 2)",
               False: "failed: the channel is not pursued further"}
DRY_VERDICT = {True: "NOT DECIDABLE (partial coverage)",
               False: "NOT EVALUATED (--dry; full coverage — decide with a run without --dry)"}


def abort(msg: str, code: int) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------------ context and existing state

def groups_of(y: np.ndarray, q_code: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Groups as in eda/c99_m3_deploy.context: NONE; code seen for the task in the other folds; task seen, code never; new
    task."""
    same_q, other = q_code[:, None] == q_code[None, :], folds[:, None] != folds[None, :]
    q_seen = (same_q & other).any(1)
    code_seen = (same_q & other & (y[:, None] == y[None, :])).any(1)
    return np.where(y == 0, "NONE", np.where(code_seen, "seen", np.where(q_seen, "never seen", "new question")))


def context(root: Path) -> dict:
    train = C.load_table(root, "train")  # training table only; row order checked
    n = len(train)
    folds = C.load_folds(n)
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    seen = labels.seen_mask(train["root_code"], classes)
    q_code = np.unique(train["question_id"].to_numpy(), return_inverse=True)[1]
    if sorted(set(folds.tolist())) != list(range(N_FOLDS)):
        abort("folds are not 0…4", 4)
    return {"n": n, "K": len(classes), "classes": classes, "y": y, "seen": seen, "folds": folds, "q_code": q_code,
            "group": groups_of(y, q_code, folds), "part_a": np.arange(n) < PART_A_ROWS}


def check_stacks(art: Path = ART) -> dict:
    """Recipe constants against the blend files of the submitted lines (primary and inductive line)."""
    out = {}
    for key, (fname, sha), line, m3 in (("best", STACK_BEST, LINE_BEST, "m3x2_full_cv01"), ("safe", STACK_SAFE, LINE_SAFE, "m3x2")):
        f = art / fname
        if sha256_file(f) != sha:
            abort(f"{fname}: checksum differs", 4)
        st = json.loads(f.read_text())
        names = [m3 if nm == "m3x2_full_cv01" else nm for nm in line["names"]]
        floors = {(m3 if k == "m3x2_full_cv01" else k): float(v) for k, v in line["floors"].items()}
        scaled = {line["think"]: float(line["factor"])} if line["think"] else {}
        last = str(st.get("last_step", ""))
        want = "transductive" if line["last"] == "trans" else "inductive alignment"
        bad = []
        if st.get("channels") != names:
            bad.append(f"channels {st.get('channels')}")
        if {k: float(v) for k, v in (st.get("floor") or {}).items()} != floors:
            bad.append(f"floors {st.get('floor')}")
        if {k: float(v) for k, v in (st.get("scaled") or {}).items()} != scaled:
            bad.append(f"factors {st.get('scaled')}")
        if float(st.get("arith_beta") or 0.0) != float(line["beta"]):
            bad.append(f"arith_beta {st.get('arith_beta')}")
        if want not in last or "0.75" not in last or "at least 3" not in last:
            bad.append(f"last step {last!r}")
        for nm in names:
            if nm == "m3x2":  # the inductive-line blend file carries the 600-row version; the cross-fit uses m3x2_full_cv01
                continue
            if (st.get("oof_sha256") or {}).get(nm) != OOF_SHA16[nm]:
                bad.append(f"checksum {nm}")
        if bad:
            abort(f"{fname} does not match the recipe of this script: {'; '.join(bad)}", 4)
        out[key] = {"file": fname, "sha256": sha}
        if key == "best":
            cr = st["cross"]
            if int(cr["n"]) != 1500:
                abort(f"{fname}: cross value not on 1,500 rows", 4)
            out[key].update({"map3": float(cr["map3"]), "logloss": float(cr["logloss"]),
                             "per_fold": [float(cr["per_fold"][str(k)]) for k in range(N_FOLDS)]})
    return out


def load_base(n: int, K: int, art: Path = ART) -> tuple[dict, dict]:
    mats, sha = {}, {}
    for nm, want in OOF_SHA16.items():
        f = art / f"oof_{nm}.npy"
        full = sha256_file(f)
        if full[:16] != want or (nm == "m3x2_full_cv01" and full != M3_SHA256):
            abort(f"oof_{nm}.npy: checksum {full[:16]} instead of {want}", 4)
        P = np.load(f)
        if P.shape != (n, K) or not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-6):
            abort(f"oof_{nm}.npy: shape {P.shape}, not finite or not normalised", 4)
        mats[nm], sha[nm] = P, full[:16]
    return mats, sha


# ------------------------------------------------------------------------------------------------ answer file

def parse_snapshot(raw: bytes) -> tuple[bytes, list[dict], dict]:
    """Snapshot of the answer file -> (complete lines as bytes, records, statistics). An incomplete last line (the run is
    writing) is cut off and reported; an unreadable complete line is an error."""
    cut = raw.rfind(b"\n") + 1
    complete, tail = raw[:cut], raw[cut:]
    records = []
    for k, line in enumerate(complete.decode("utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"line {k + 1} of the answer file is not JSON ({e.msg})") from None
    info = {"bytes": len(raw), "bytes_complete": len(complete), "partial_tail": bool(tail.strip()), "records": len(records),
            "sha256_complete": sha256_bytes(complete)}
    return complete, records, info


def validate_records(records: list[dict], n: int) -> None:
    """Training positions only: row an integer in [0, n); a possible field table must be 'train'."""
    for r in records:
        row = r.get("row")
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < n:
            raise ValueError(f"record with an invalid row position {row!r}")
        if "table" in r and r["table"] != "train":
            raise ValueError(f"record with table={r['table']!r} — only training rows are allowed")


def legacy_prefix(complete: bytes, sha: str = LEGACY_SHA256) -> dict | None:
    """Shortest prefix of the file (at line boundaries) whose SHA-256 equals the earlier records -> {bytes, lines} or None."""
    h = hashlib.sha256()
    pos = 0
    for k, line in enumerate(complete.splitlines(keepends=True), 1):
        h.update(line)
        pos += len(line)
        if h.copy().hexdigest() == sha:
            return {"bytes": pos, "lines": k}
    return None


def coverage(records: list[dict], n: int, classes: list[str]) -> dict:
    """Coverage per training row. valid = ok and root_code in the class list (as in labels_to_probs); waiting = records exist,
    but no valid answer and no finishing pass yet (phase finish)."""
    index = set(classes)
    have, valid, finish = set(), set(), set()
    for r in records:
        i = int(r["row"])
        have.add(i)
        if r.get("ok", True) and r.get("root_code") in index:
            valid.add(i)
        if r.get("phase") == "finish":
            finish.add(i)
    return {"rows": n, "rows_with_record": len(have), "rows_missing": n - len(have), "rows_valid": len(valid),
            "rows_pending_finish": len(have - valid - finish), "rows_finish_invalid": len((have & finish) - valid),
            "valid_share": len(valid) / n if n else 0.0}


def coverage_problems(cov: dict, partial_tail: bool, min_valid: float, legacy_ok: bool) -> list[str]:
    """Reasons why no decision is possible (empty = full coverage)."""
    out = []
    if partial_tail:
        out.append("the last line of the answer file is incomplete (the run is still writing)")
    if cov["rows_missing"]:
        out.append(f"{cov['rows_missing']} of {cov['rows']} training rows without any record (run pass not finished)")
    if cov["rows_pending_finish"]:
        out.append(f"{cov['rows_pending_finish']} rows without a valid answer are waiting for the finishing pass (tools/open_teacher.py finish)")
    if cov["valid_share"] < min_valid:
        out.append(f"valid answers {cov['rows_valid']}/{cov['rows']} ({cov['valid_share']:.1%}) below {min_valid:.0%}")
    if not legacy_ok:
        out.append(f"earlier records ({LEGACY_ROWS} rows) not unchanged at the start of the file or not exactly "
                   "the sample design_stage2.json")
    return out


def require_full(problems: list[str], dry: bool) -> None:
    if problems and not dry:
        abort("no full coverage — no decision possible:\n  - " + "\n  - ".join(problems)
              + "\n  (restart after both the run and the finish passes have ended; code path without a decision: --dry)", 3)


def proto_channel(records: list[dict], n: int, classes: list[str], seen: np.ndarray, mode: str = "stated",
                  levels: tuple = LEVELS) -> tuple[np.ndarray, np.ndarray]:
    """Channel from answers in the teacher protocol — step by step as in vlm.proto.labels_to_probs: named codes with their
    stated probabilities, fixed levels without a usable number, remaining mass spread evenly over the classes with a training
    example, per row the last valid record. mode 'levels': stated numbers discarded, always fixed levels.
    -> (P [n, K], valid [n]); rows without a valid answer stay zero."""
    if mode not in ("stated", "levels"):
        raise ValueError(f"unknown variant {mode!r}")
    index = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    seen_cls = np.asarray(seen, dtype=bool)
    probs, has = np.zeros((n, K)), np.zeros(n, bool)
    for r in records:
        if not r.get("ok", True) or r.get("root_code") not in index:
            continue
        p = np.zeros(K)
        if mode == "stated":
            for c, w in zip(r.get("top3") or [], r.get("probabilities") or []):
                if c in index and PR._is_number(w) and np.isfinite(float(w)):
                    p[index[c]] += max(0.0, float(w))
        if p.sum() <= 0:
            for c, w in zip(r.get("top3") or [r["root_code"]], levels):
                if c in index:
                    p[index[c]] += w
        if p.sum() > 1:
            p = p / p.sum()
        p = p + max(0.0, 1.0 - p.sum()) * seen_cls / seen_cls.sum()  # the same order of operations as labels_to_probs (bit-identical)
        probs[int(r["row"])], has[int(r["row"])] = p / p.sum(), True
    return probs, has


def fill_missing(P: np.ndarray, has: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Rows without a valid answer: uniform over the classes with a training example (a8.mean_of_draws, test path)."""
    seen_cls = np.asarray(seen, dtype=bool)
    out = np.array(P, dtype=np.float64, copy=True)
    out[~np.asarray(has, dtype=bool)] = seen_cls / seen_cls.sum()
    return out


def verify_channel(complete: bytes, records: list[dict], n: int, classes: list[str], seen: np.ndarray, P_stated: np.ndarray,
                   P_levels: np.ndarray, has: np.ndarray, tmp_root: Path) -> dict:
    """Own conversion against vlm.proto.labels_to_probs on the same snapshot — bit-identical, both variants (fixed levels:
    probabilities = None in every record). Intermediate files only in the private output folder, deleted immediately."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix="c107_") as td:
        f1, f2 = Path(td) / "snapshot.jsonl", Path(td) / "snapshot_levels.jsonl"
        f1.write_bytes(complete)
        with f2.open("w", encoding="utf-8") as fh:
            for r in records:
                r2 = dict(r)
                r2["probabilities"] = None
                fh.write(json.dumps(r2) + "\n")
        try:
            Pp, hp = PR.labels_to_probs(f1, n, classes, seen)
            Pl, hl = PR.labels_to_probs(f2, n, classes, seen)
        except (ValueError, KeyError, TypeError) as e:
            return {"ok": False, "error": f"labels_to_probs fails on the file: {type(e).__name__}"}
    ok = bool(np.array_equal(hp, has) and np.array_equal(hl, has) and np.array_equal(Pp, P_stated) and np.array_equal(Pl, P_levels))
    return {"ok": ok, "rows_valid": int(has.sum()), "max_abs_stated": float(np.abs(Pp - P_stated).max()),
            "max_abs_levels": float(np.abs(Pl - P_levels).max())}


# ------------------------------------------------------------------------------------------------ recipe

def logs_of(mats: dict, names: list[str], floors: dict, seen: np.ndarray, mask_first: bool = False) -> list[np.ndarray]:
    """Channels as log probabilities with floor. mask_first=False as in the build path (c99) and the test path
    (make_submission); True as in make_submission.oof_mixture (basis of the inductive class weights)."""
    out = []
    for nm in names:
        P = mats[nm]
        if mask_first:
            P = np.where(seen[None, :], P, 0.0)
        out.append(np.log(np.clip(A2.floor_probs(P, float(floors.get(nm, 0.0)), seen), EPS, 1.0)))
    return out


def target_of(y: np.ndarray, rows: np.ndarray, K: int, min_rows: int = ALIGN_MIN) -> np.ndarray:
    """Target of the alignment: class frequencies of the rows `rows`, classes below min_rows rows without a target (0)."""
    c = np.bincount(y[rows], minlength=K)
    return np.where(c >= min_rows, c, 0) / len(rows)


def apply_last(P: np.ndarray, w: np.ndarray, seen: np.ndarray, s: float = ALIGN) -> np.ndarray:
    Q = C21.apply_weights(P, w, s)
    Q = np.where(seen[None, :], Q, 0.0)
    return Q / Q.sum(1, keepdims=True)


def with_channel(spec: dict, name: str, floor: float = FLOOR_NEW) -> dict:
    """Recipe plus one channel (appended last, own floor); the original recipe stays unchanged."""
    return {**spec, "names": list(spec["names"]) + [name], "floors": {**spec["floors"], name: float(floor)}}


def cross_fit(y: np.ndarray, folds: np.ndarray, seen: np.ndarray, mats: dict, spec: dict, last: str | None = None,
              mask_first: bool = False) -> dict:
    """Cross-fit per fold k: weights (a2.fit_mix, log loss) only from the other four folds F, then the factor on the reasoning
    channel; blend make_submission.mix_channels (β) on fold k; last step
      trans  margin alignment 0.75 of the rows of k (weights on the evaluated set), target = frequencies of F (at least 3 rows);
      ind    inductive: class weights from the blend for F (mask first, as in oof_mixture), target as above, applied to k;
      none   no last step.
    -> {Q (after the last step), P (before), theta [folds, channels] (deployed, after the factor), names}"""
    names, floors = list(spec["names"]), dict(spec["floors"])
    last = spec["last"] if last is None else last
    if last not in ("trans", "ind", "none"):
        raise ValueError(f"unknown last step {last!r}")
    n, K = len(y), len(seen)
    L = logs_of(mats, names, floors, seen, mask_first)
    Lm = logs_of(mats, names, floors, seen, True) if last == "ind" else None
    think = spec.get("think")
    ti = names.index(think) if think in names else None
    Q, P = np.zeros((n, K)), np.zeros((n, K))
    thetas = []
    for k in range(N_FOLDS):
        F, E = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        th = np.asarray(A2.fit_mix([lg[F] for lg in L], None, y[F]), dtype=float).copy()
        if ti is not None:
            th[ti] *= float(spec["factor"])
        st = {"theta": [float(v) for v in th], "arith_beta": float(spec["beta"])}
        PE = MS.mix_channels(st, [lg[E] for lg in L], None, seen)
        P[E] = PE
        tgt = target_of(y, F, K)
        if last == "none":
            Q[E] = PE
        elif last == "trans":
            Q[E] = apply_last(PE, C21.fit_weights(PE, tgt), seen)
        else:
            PF = MS.mix_channels(st, [lg[F] for lg in Lm], None, seen)
            Q[E] = apply_last(PE, C21.fit_weights(PF, tgt), seen)
        thetas.append(th)
    return {"Q": Q, "P": P, "theta": np.array(thetas), "names": names}


def set_align(P: np.ndarray, y: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Alignment as ONE set over all rows (target = frequencies of all rows, at least 3) on the blend before the last step —
    closer to the test side (one set of 3,498 rows), slightly optimistic (target = own shares)."""
    return apply_last(P, C21.fit_weights(P, target_of(y, np.arange(len(y)), len(seen))), seen)


def rows_map(y: np.ndarray, Q: np.ndarray, seen: np.ndarray) -> np.ndarray:
    return metric.map3_rows_from_proba(y, Q, seen)


def rows_ll(y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(Q[np.arange(len(y)), y], LL_CLIP, 1.0))


def sane(Q: np.ndarray, seen: np.ndarray) -> bool:
    """Predictions finite, non-negative, normalised per row, without mass on classes without a training example."""
    return bool(np.isfinite(Q).all() and (Q >= 0).all() and np.allclose(Q.sum(1), 1.0, atol=1e-9) and not Q[:, ~seen].any())


# ------------------------------------------------------------------------------------------------ statistics

def units(r) -> np.ndarray:
    """MAP@3 row values in whole units of 1/6 (1 → 6, 1/2 → 3, 1/3 → 2, 0 → 0); anything else is an error."""
    r = np.asarray(r, dtype=np.float64)
    u = np.rint(r * 6.0)
    if r.ndim != 1 or not np.allclose(u / 6.0, r, rtol=0.0, atol=1e-9) or (u < 0).any() or (u > 6).any():
        raise ValueError("row values are not MAP@3 values (multiples of 1/6 between 0 and 1)")
    return u.astype(np.int64)


def zval(mean: float, se: float) -> float:
    if not np.isfinite(se):
        return float("nan")
    if se > 0:
        return float(mean / se)
    return float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0


def question_draws(q_code: np.ndarray, B: int = BOOT, seed: int = SEED) -> np.ndarray:
    """Bootstrap draws over tasks: [B, number of tasks] indices (with replacement); the same for all quantities."""
    nq = int(np.max(q_code)) + 1
    return np.random.default_rng(seed).integers(0, nq, size=(B, nq))


def cluster_boot(v: np.ndarray, q_code: np.ndarray, draws: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Mean of v (over mask) with σ over tasks: per draw the drawn tasks with all their rows (ratio estimator
    Σ sums / Σ rows). Subgroups: the same draws, only the rows of the group count."""
    v = np.asarray(v, dtype=np.float64)
    m = np.ones(len(v), bool) if mask is None else np.asarray(mask, dtype=bool)
    nq = draws.shape[1]
    s = np.bincount(q_code[m], weights=v[m], minlength=nq)
    c = np.bincount(q_code[m], minlength=nq).astype(np.float64)
    S, N = s[draws].sum(1), c[draws].sum(1)
    ok = N > 0
    vals = S[ok] / N[ok]
    mean = float(v[m].mean()) if m.any() else float("nan")
    se = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    q = np.quantile(vals, [0.025, 0.975]) if len(vals) else [float("nan")] * 2
    return {"mean": mean, "se_q": se, "z_q": zval(mean, se), "ci95_q": [float(q[0]), float(q[1])],
            "p_le_0": float((vals <= 0).mean()) if len(vals) else float("nan"),
            "p_below_m0002": float((vals < -0.002).mean()) if len(vals) else float("nan"), "n_rows": int(m.sum()),
            "draws_used": int(ok.sum())}


def compare(new_rows: np.ndarray, base_rows: np.ndarray, ctx: dict, draws: np.ndarray, new_ll: np.ndarray | None = None,
            base_ll: np.ndarray | None = None) -> dict:
    """Paired comparison (with channel − without channel) on the same rows: Δ MAP@3 with σ over tasks and over rows, per fold
    (exact), groups, data parts; with log-loss rows additionally the log-loss gain (without − with)."""
    folds, n = ctx["folds"], len(new_rows)
    d = np.asarray(new_rows, float) - np.asarray(base_rows, float)
    du = units(new_rows) - units(base_rows)
    cb = cluster_boot(d, ctx["q_code"], draws)
    pb = metric.paired_bootstrap(new_rows, base_rows)
    fu = [int(du[folds == k].sum()) for k in range(N_FOLDS)]
    out = {"map_new": float(np.mean(new_rows)), "map_base": float(np.mean(base_rows)), "delta": cb["mean"],
           "delta_exact": Fraction(int(du.sum()), 6 * n), "se_q": cb["se_q"], "z_q": cb["z_q"], "ci95_q": cb["ci95_q"],
           "p_le_0": cb["p_le_0"], "p_below_m0002": cb["p_below_m0002"], "se_rows": float(pb["sigma"]), "z_rows": float(pb["z"]),
           "per_fold": [float(d[folds == k].mean()) for k in range(N_FOLDS)], "per_fold_units": fu,
           "folds_ge0": int(sum(u >= 0 for u in fu)), "folds_gt0": int(sum(u > 0 for u in fu)),
           "rows_better": int((du > 0).sum()), "rows_worse": int((du < 0).sum()), "groups": {}, "parts": {}}
    for g in GROUPS:
        m = ctx["group"] == g
        gb = cluster_boot(d, ctx["q_code"], draws, m)
        out["groups"][g] = {"n": int(m.sum()), "delta": gb["mean"], "se_q": gb["se_q"], "contrib": float(d[m].sum() / n)}
    for p, m in (("A", ctx["part_a"]), ("B", ~ctx["part_a"])):
        pbq = cluster_boot(d, ctx["q_code"], draws, m)
        out["parts"][p] = {"n": int(m.sum()), "delta": pbq["mean"], "se_q": pbq["se_q"]}
    if new_ll is not None:
        g = np.asarray(base_ll, float) - np.asarray(new_ll, float)
        cl = cluster_boot(g, ctx["q_code"], draws)
        pl = metric.paired_bootstrap(base_ll, new_ll)
        out.update({"ll_new": float(np.mean(new_ll)), "ll_base": float(np.mean(base_ll)), "ll_gain": cl["mean"],
                    "ll_se_q": cl["se_q"], "ll_z_q": cl["z_q"], "ll_se_rows": float(pl["sigma"]), "ll_z_rows": float(pl["z"])})
    return out


def evaluate_rule(new_rows, base_rows, fixed_rows, folds: np.ndarray, z_map: float, z_ll: float) -> dict:
    """The pre-registered rule, exactly: Δ and Δ(fixed levels) as fractions in units of 1/6, folds as integer sums."""
    du = units(new_rows) - units(base_rows)
    dfu = units(fixed_rows) - units(base_rows)
    n = len(du)
    delta, delta_fix = Fraction(int(du.sum()), 6 * n), Fraction(int(dfu.sum()), 6 * n)
    fold_units = [int(du[folds == k].sum()) for k in range(N_FOLDS)]
    n_ge0 = int(sum(u >= 0 for u in fold_units))
    checks = [
        {"criterion": CRITERIA[0], "value": float(delta), "exact": delta, "ok": delta >= Fraction(6, 1000)},
        {"criterion": CRITERIA[1], "value": float(z_map), "ok": bool(z_map >= 1.5)},
        {"criterion": CRITERIA[2], "value": float(z_ll), "ok": bool(z_ll >= 3.0)},
        {"criterion": CRITERIA[3], "value": n_ge0, "per_fold_units": fold_units, "ok": n_ge0 >= 4},
        {"criterion": CRITERIA[4], "value": float(delta_fix), "exact": delta_fix, "ok": delta_fix >= Fraction(3, 1000)},
    ]
    passed = all(c["ok"] for c in checks)
    return {"checks": checks, "passed": passed, "verdict": "PASSED" if passed else "FAILED"}


def ll_readings(final_base: np.ndarray, final_new: np.ndarray, pre_base: np.ndarray, pre_new: np.ndarray, q_code: np.ndarray,
                draws: np.ndarray, main_ok: bool) -> dict:
    """Alternative readings of criterion 3 (log-loss gain z ≥ 3): full recipe / without last step, σ over tasks / rows."""
    vals = {}
    for key, (lb, ln) in {"full recipe": (final_base, final_new), "without the last step": (pre_base, pre_new)}.items():
        cb = cluster_boot(lb - ln, q_code, draws)
        pb = metric.paired_bootstrap(lb, ln)
        vals[f"{key}, σ over tasks"] = {"gain": cb["mean"], "z": cb["z_q"]}
        vals[f"{key}, row bootstrap"] = {"gain": float(pb["delta"]), "z": float(pb["z"])}
    flips = [k for k, v in vals.items() if bool(v["z"] >= 3.0) != bool(main_ok)]
    return {"readings": vals, "main": "full recipe, σ over tasks", "flips": flips}


# ------------------------------------------------------------------------------------------------ anchors

def check_anchor(rows: np.ndarray, expected_units: int, label: str, per_fold: list[float] | None = None,
                 folds: np.ndarray | None = None) -> dict:
    """Anchors exact: sum of the row values in units of 1/6 (and per fold against the blend file) — otherwise abort."""
    u = units(rows)
    got = int(u.sum())
    if got != expected_units:
        abort(f"anchor {label} missed: {got}/{6 * len(u)} = {got / (6 * len(u)):.7f} instead of {expected_units}/{6 * len(u)} "
              f"= {expected_units / (6 * len(u)):.7f} — check the re-implementation of the recipe", 4)
    out = {"map3": got / (6 * len(u)), "units": got, "expected_units": expected_units}
    if per_fold is not None:
        fold_units = []
        for k, ref in enumerate(per_fold):
            nk = int((folds == k).sum())
            want = ref * 6 * nk
            if abs(want - round(want)) > 1e-6 or int(u[folds == k].sum()) != int(round(want)):
                abort(f"anchor {label}, fold {k}: {int(u[folds == k].sum())}/{6 * nk} instead of {ref:.7f}", 4)
            fold_units.append(int(round(want)))
        out["per_fold_units"] = fold_units
    return out


def anchors(ctx: dict, mats: dict, stacks: dict) -> tuple[dict, dict, dict]:
    """Both lines without the new channel: primary line (transductive alignment, β 0.2) and inductive line (inductive alignment, β 0.4) — exactly as submitted."""
    y, folds, seen = ctx["y"], ctx["folds"], ctx["seen"]
    fb = cross_fit(y, folds, seen, mats, LINE_BEST)
    fs = cross_fit(y, folds, seen, mats, LINE_SAFE)
    rb, rs = rows_map(y, fb["Q"], seen), rows_map(y, fs["Q"], seen)
    ab = check_anchor(rb, ANCHOR_UNITS["best"], LINE_BEST["label"], stacks["best"]["per_fold"], folds)
    llb = float(rows_ll(y, fb["Q"]).mean())
    if abs(llb - stacks["best"]["logloss"]) > 1e-9:
        abort(f"anchor {LINE_BEST['label']}: log loss {llb:.10f} instead of {stacks['best']['logloss']:.10f}", 4)
    ab["logloss"] = llb
    as_ = check_anchor(rs, ANCHOR_UNITS["safe"], LINE_SAFE["label"])
    as_["logloss"] = float(rows_ll(y, fs["Q"]).mean())
    return {"best": ab, "safe": as_}, fb, fs


# ------------------------------------------------------------------------------------------------ output

def fmt_cmp(c: dict, with_groups: bool = True) -> str:
    pf = " ".join(f"{x:+.4f}" for x in c["per_fold"])
    s = (f"MAP@3 {c['map_base']:.5f} → {c['map_new']:.5f}: Δ {c['delta']:+.5f} (exact {c['delta_exact'].numerator}/"
         f"{c['delta_exact'].denominator}) ± {c['se_q']:.5f} over tasks (z {c['z_q']:+.2f}; 95 % {c['ci95_q'][0]:+.4f} … "
         f"{c['ci95_q'][1]:+.4f}; P(Δ ≤ 0) {c['p_le_0']:.3f}; P(Δ < −0.002) {c['p_below_m0002']:.3f}) | rows ± {c['se_rows']:.5f}\n"
         f"      per fold {pf} ({c['folds_ge0']}/5 with Δ ≥ 0, {c['folds_gt0']}/5 > 0) | rows better/worse "
         f"{c['rows_better']}/{c['rows_worse']}")
    if "ll_gain" in c:
        s += (f"\n      log loss {c['ll_base']:.4f} → {c['ll_new']:.4f}: gain {c['ll_gain']:+.4f} ± {c['ll_se_q']:.4f} over "
              f"tasks (z {c['ll_z_q']:+.2f}) | rows ± {c['ll_se_rows']:.4f} (z {c['ll_z_rows']:+.2f})")
    if with_groups:
        s += "\n      groups: " + " | ".join(f"{g} ({v['n']}) {v['delta']:+.4f} ± {v['se_q']:.4f} (contribution {v['contrib']:+.4f})"
                                         for g, v in c["groups"].items())
        s += "\n      parts: " + " | ".join(f"{p} ({v['n']}) {v['delta']:+.4f} ± {v['se_q']:.4f}" for p, v in c["parts"].items())
    return s


def weights_report(fit: dict, name: str) -> dict:
    th = np.asarray(fit["theta"], dtype=float)
    j = fit["names"].index(name)
    share = th[:, j] / th.sum(1)
    return {"names": fit["names"], "theta_per_fold": th.tolist(), "theta_new": th[:, j].tolist(), "share_new": share.tolist(),
            "theta_mean": th.mean(0).tolist()}


def protocol_stats(records: list[dict], n_legacy: int) -> dict:
    """Statistics of the run per part (earlier records / newly appended): records, rows, valid rows, share of first passes at
    the token limit, output tokens (median) — aggregates only."""
    out = {}
    for key, part in (("earlier", records[:n_legacy]), ("new", records[n_legacy:])):
        first = [r for r in part if r.get("phase") == "first"]
        out[key] = {"records": len(part), "rows": len({int(r["row"]) for r in part}),
                    "rows_valid": len({int(r["row"]) for r in part if r.get("ok")}),
                    "first_records": len(first), "first_stop_length": float(np.mean(["length" in str(r.get("stop", "")).lower() for r in first])) if first else float("nan"),
                    "out_tokens_median": float(np.median([r.get("out_tokens", 0) for r in part])) if part else float("nan")}
    return out


def report_decision(alone: dict, main_cmp: dict, fix_cmp: dict, rule: dict, ll_reading: dict, info: dict, wts: dict) -> None:
    """Console output of the decision (aggregates only; only without --dry)."""
    print(f"\nCHANNEL {NEW} alone (valid answers, descriptive): MAP@3 {alone['map3_valid']:.4f}, top correct {alone['top1_valid']:.3f}, "
          f"NONE at the top {alone['none_top_valid']:.3f} (true {alone['none_true_valid']:.3f}) | "
          + " | ".join(f"{g} {v:.3f}" for g, v in alone["groups_valid"].items()))
    print(f"\nMAIN COMPUTATION — primary-line recipe + {NEW} (floor {FLOOR_NEW}), 1,500 rows, weights cross-fitted, margin alignment per fold:")
    print("   " + fmt_cmp(main_cmp))
    w = wts["primary"]
    print("      weight of the channel per fold: θ " + " ".join(f"{x:.3f}" for x in w["theta_new"])
          + " | share of Σθ " + " ".join(f"{x:.3f}" for x in w["share_new"]))
    print(f"\nFIXED LEVELS {LEVELS[0]}/{LEVELS[1]}/{LEVELS[2]} instead of the stated probabilities:")
    print("   " + fmt_cmp(fix_cmp, with_groups=False))
    print("\nPRE-REGISTERED RULE: " + " | ".join(f"{c['criterion']}: {'yes' if c['ok'] else 'NO'}" for c in rule["checks"])
          + f"  =>  {rule['verdict']}")
    print(f"   consequence: {CONSEQUENCE[rule['passed']]}")
    print("   reading of criterion 3: " + " | ".join(f"{k} z {v['z']:+.2f}" for k, v in ll_reading["readings"].items())
          + ("  — changes the criterion: " + ", ".join(ll_reading["flips"]) if ll_reading["flips"] else "  — does not change the criterion"))
    print("\nFOR INFORMATION ONLY (no decision):")
    for label, c in info.items():
        print(f"   {label}:\n      " + fmt_cmp(c, with_groups=label.startswith("inductive line")))
    ws = wts["inductive"]
    print("      inductive line, weight of the channel per fold: θ " + " ".join(f"{x:.3f}" for x in ws["theta_new"]))


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=json_default) + "\n"


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(COMP), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------------------------------------ main program

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="only check the code path on the partial state; no decision number")
    ap.add_argument("--min-valid", type=float, default=MIN_VALID, help="minimum share of rows with a valid answer (0.98)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.min_valid != MIN_VALID:
        print(f"WARNING: --min-valid {a.min_valid} differs from the default {MIN_VALID}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    root = C.find_data_root()
    ctx = context(root)
    n, K, y, seen, folds = ctx["n"], ctx["K"], ctx["y"], ctx["seen"], ctx["folds"]
    stacks = check_stacks()
    mats, oof_sha = load_base(n, K)
    print(f"q397 sixth channel / c107 — {'CODE PATH ONLY (--dry), no decision number' if a.dry else 'DECISION by the pre-registered rule'}")
    print(f"recipe checked against {STACK_BEST[0]} (best) and {STACK_SAFE[0]} (inductive); base OOF checksums {len(oof_sha)}/{len(OOF_SHA16)}")

    # answer file: one snapshot, checks, coverage
    path = root / LABELS_REL
    complete, records, sinfo = parse_snapshot(path.read_bytes())
    validate_records(records, n)
    leg = legacy_prefix(complete)
    n_leg_records, legacy_rows_ok = 0, False
    if leg is not None:
        pre = [ln for ln in complete[: leg["bytes"]].decode("utf-8").split("\n") if ln.strip()]
        n_leg_records = len(pre)
        design = {int(k) for k in json.loads((root / DESIGN_REL).read_text())["weight"]}
        legacy_rows_ok = {int(json.loads(ln)["row"]) for ln in pre} == design and len(design) == LEGACY_ROWS
    rest = set(np.load(root / REST_REL).astype(int).tolist()) if (root / REST_REL).is_file() else set()
    cov = coverage(records, n, ctx["classes"])
    problems = coverage_problems(cov, sinfo["partial_tail"], a.min_valid, leg is not None and legacy_rows_ok)
    proto = protocol_stats(records, n_leg_records)
    print(f"answer file {LABELS_REL}: {sinfo['records']} records ({sinfo['bytes_complete']} bytes complete"
          f"{', last line incomplete' if sinfo['partial_tail'] else ''}); earlier records "
          + (f"unchanged ({leg['lines']} lines, {n_leg_records} records, sample {'matched exactly' if legacy_rows_ok else 'NOT matched'})"
             if leg else "NOT found"))
    print(f"coverage: {cov['rows_with_record']}/{n} rows with a record, valid {cov['rows_valid']} ({cov['valid_share']:.1%}), "
          f"waiting for finish {cov['rows_pending_finish']}, invalid after finish {cov['rows_finish_invalid']}; plan "
          f"(sample + {REST_REL.rsplit('/', 1)[-1]}) covers {len(design | rest) if leg else '?'} rows")
    print("run per part: " + " | ".join(f"{k}: {v['records']} records, {v['rows']} rows, valid {v['rows_valid']}, first passes at the "
                                        f"token limit {v['first_stop_length']:.2f}, output tokens median {v['out_tokens_median']:.0f}"
                                       for k, v in proto.items()))
    require_full(problems, a.dry)
    if problems:
        print("PARTIAL COVERAGE — code path only, no decision: " + "; ".join(problems))

    # channel (named / fixed levels), bit-identical against labels_to_probs
    Ps, has = proto_channel(records, n, ctx["classes"], seen, "stated")
    Pl, has_l = proto_channel(records, n, ctx["classes"], seen, "levels")
    ver = verify_channel(complete, records, n, ctx["classes"], seen, Ps, Pl, has, OUT)
    if not ver["ok"] or not np.array_equal(has, has_l):
        abort(f"channel conversion differs from vlm.proto.labels_to_probs: {ver}", 4)
    unseen_named = int((Ps[:, ~seen].sum(1) > 0).sum())
    mats[NEW], mats[NEW_FIXED] = fill_missing(Ps, has, seen), fill_missing(Pl, has, seen)
    print(f"channel {NEW}: conversion bit-identical to vlm.proto.labels_to_probs (named and fixed levels); {int(has.sum())} rows with "
          f"a valid answer, the rest uniform; {unseen_named} rows name a code without a training example (mass kept until the "
          "mask after blending, as in the test path)")

    # anchors without the channel
    anc, fit_b, fit_s = anchors(ctx, mats, stacks)
    print(f"ANCHOR primary (without channel): {anc['best']['map3']:.7f} = {anc['best']['units']}/{6 * n} exact, per fold "
          f"{anc['best']['per_fold_units']} as in the blend file, log loss {anc['best']['logloss']:.6f} as there — matched")
    print(f"ANCHOR inductive (without channel, 1,500-row cross-fit): {anc['safe']['map3']:.7f} = {anc['safe']['units']}/{6 * n} exact — matched")

    # with the channel: main computation, fixed levels, inductive line, sensitivity "mask first"
    spec_new, spec_fix = with_channel(LINE_BEST, NEW), with_channel(LINE_BEST, NEW_FIXED)
    fit_n = cross_fit(y, folds, seen, mats, spec_new)
    fit_f = cross_fit(y, folds, seen, mats, spec_fix)
    fit_sn = cross_fit(y, folds, seen, mats, with_channel(LINE_SAFE, NEW))
    fit_mb = cross_fit(y, folds, seen, mats, LINE_BEST, mask_first=True)
    fit_mn = cross_fit(y, folds, seen, mats, spec_new, mask_first=True)
    fits = {"base_best": fit_b, "new_best": fit_n, "fixed_best": fit_f, "base_safe": fit_s, "new_safe": fit_sn,
            "maskfirst_base": fit_mb, "maskfirst_new": fit_mn}
    R = {k: rows_map(y, f["Q"], seen) for k, f in fits.items()}
    LL = {k: rows_ll(y, f["Q"]) for k, f in fits.items()}
    for k in ("base_best", "new_best"):  # without the last step and as one set (information only)
        R[f"{k}_pre"], LL[f"{k}_pre"] = rows_map(y, fits[k]["P"], seen), rows_ll(y, fits[k]["P"])
        Qs = set_align(fits[k]["P"], y, seen)
        R[f"{k}_set"], LL[f"{k}_set"] = rows_map(y, Qs, seen), rows_ll(y, Qs)
    draws = question_draws(ctx["q_code"])
    main_cmp = compare(R["new_best"], R["base_best"], ctx, draws, LL["new_best"], LL["base_best"])
    fix_cmp = compare(R["fixed_best"], R["base_best"], ctx, draws, LL["fixed_best"], LL["base_best"])
    info = {"one set (alignment over all 1,500 rows)": compare(R["new_best_set"], R["base_best_set"], ctx, draws,
                                                              LL["new_best_set"], LL["base_best_set"]),
            "without the last step": compare(R["new_best_pre"], R["base_best_pre"], ctx, draws, LL["new_best_pre"], LL["base_best_pre"]),
            "inductive line, earlier recipe (inductive, β 0.4)": compare(R["new_safe"], R["base_safe"], ctx, draws, LL["new_safe"], LL["base_safe"]),
            "sensitivity: mask before the floor": compare(R["maskfirst_new"], R["maskfirst_base"], ctx, draws,
                                                          LL["maskfirst_new"], LL["maskfirst_base"])}
    rule = evaluate_rule(R["new_best"], R["base_best"], R["fixed_best"], folds, main_cmp["z_q"], main_cmp["ll_z_q"])
    ll_reading = ll_readings(LL["base_best"], LL["new_best"], LL["base_best_pre"], LL["new_best_pre"], ctx["q_code"], draws,
                         rule["checks"][2]["ok"])
    wts = {"primary": weights_report(fit_n, NEW), "fixed levels": weights_report(fit_f, NEW_FIXED),
           "inductive": weights_report(fit_sn, NEW)}
    rr = metric.map3_rows_from_proba(y, mats[NEW], seen)
    top = metric.top3_indices(mats[NEW], seen)[:, 0]
    alone = {"rows_valid": int(has.sum()), "map3_valid": float(rr[has].mean()) if has.any() else float("nan"),
             "top1_valid": float((top[has] == y[has]).mean()) if has.any() else float("nan"),
             "none_top_valid": float((top[has] == 0).mean()) if has.any() else float("nan"),
             "none_true_valid": float((y[has] == 0).mean()) if has.any() else float("nan"),
             "groups_valid": {g: float(rr[has & (ctx["group"] == g)].mean()) if (has & (ctx["group"] == g)).any() else float("nan")
                              for g in GROUPS}}

    # code path: all predictions clean, all quantities finite
    path_ok = all(sane(f["Q"], seen) and sane(f["P"], seen) and np.isfinite(f["theta"]).all() for f in fits.values())
    nums = [main_cmp["delta"], main_cmp["se_q"], main_cmp["ll_gain"], main_cmp["ll_se_q"], fix_cmp["delta"]]
    nums += [v["delta"] for v in info.values()] + [v["se_q"] for v in info.values()]
    nums += [x for w in wts.values() for x in w["theta_new"]]
    path_ok = path_ok and bool(np.isfinite(nums).all())
    if not path_ok:
        abort("code path: non-finite or non-normalised quantities", 4)
    integrity = {"labels_sha256_complete": sinfo["sha256_complete"], "labels_records": sinfo["records"],
                 "labels_partial_tail": sinfo["partial_tail"], "legacy_prefix": leg, "legacy_records": n_leg_records,
                 "legacy_rows_equal_design": legacy_rows_ok, "plan_rows": len(design | rest) if leg else None,
                 "channel_vs_labels_to_probs": ver, "oof_sha16": oof_sha, "stacks": stacks,
                 "script_sha256": sha256_file(Path(__file__)), "git_head": git_head()}

    result = {"experiment": "q397-sixth-channel", "script": "eda/c107_q397_full.py", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "rule_text": RULE_TEXT, "reading": READING, "coverage": cov, "protocol": proto, "integrity": integrity,
              "anchors": anc, "channel_alone": alone, "main": main_cmp, "fixed_levels": fix_cmp, "rule": rule,
              "verdict": rule["verdict"], "consequence": CONSEQUENCE[rule["passed"]], "ll_readings": ll_reading, "weights": wts,
              "info": info, "boot": {"draws": BOOT, "seed": SEED, "unit": "question_id"}, "seconds": round(time.time() - t0, 1)}
    result_json = to_json(result)  # also built and serialised in the dry run (code path), but never written there
    rows_npz = io.BytesIO()
    np.savez_compressed(rows_npz, **{f"map_{k}": v for k, v in R.items()}, **{f"ll_{k}": v for k, v in LL.items()}, has=has, folds=folds)

    if a.dry:
        del result, result_json, rows_npz  # partial state: no channel number leaves memory
        print("CODE PATH: all computations ran (main computation, fixed levels, one set, without last step, inductive line, "
              "mask first, readings, weights, groups; result file built and serialised, not written) — all "
              "predictions normalised and without mass on classes without a training example, all quantities finite")
        partial = bool(problems)
        print("PRE-REGISTERED RULE: " + " | ".join(f"{c}: {'not decidable' if partial else 'not evaluated'}" for c in CRITERIA)
              + f"  =>  {DRY_VERDICT[partial]}")
        dry = {"mode": "dry", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decision": DRY_VERDICT[partial],
               "note": "--dry: no number about the channel (neither decision nor information) — checks only",
               "coverage": cov, "problems": problems, "protocol": proto, "integrity": integrity,
               "anchors": anc, "codepath_ok": path_ok, "computed": ["main computation", "fixed levels", "one set", "without last step",
                                                                   "inductive line", "mask first", "readings", "weights", "groups"],
               "seconds": round(time.time() - t0, 1)}
        (OUT / "c107_dry.json").write_text(to_json(dry))
        print(f"duration {time.time() - t0:.0f} s -> {OUT / 'c107_dry.json'} (without channel numbers)")
        return 0

    # decision: output
    report_decision(alone, main_cmp, fix_cmp, rule, ll_reading, info, wts)
    (OUT / "c107_result.json").write_text(result_json)
    (OUT / "c107_rows.npz").write_bytes(rows_npz.getvalue())
    print(f"\nduration {time.time() - t0:.0f} s -> {OUT / 'c107_result.json'}, {OUT / 'c107_rows.npz'} (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
