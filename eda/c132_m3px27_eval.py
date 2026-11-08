"""Channel m3px27: M3 scores all codes behind the notes of the FIRST pr27 pass (the same notes behind which the untrained 27B
computed px27) — evaluation under the binding PRE-REGISTERED m3px27 rule in the final primary form of the primary line
(expected-value rule, ½(F+M)).

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c132_m3px27_eval.py --preflight      # BEFORE the run: configurations as the runner reads them, note file, check_recipe, prompt fidelity
    python -I eda/c132_m3px27_eval.py --controls       # BEFORE the run: controls of the same role (original + 20 splits) = positive control
    python -I eda/c132_m3px27_eval.py --status         # state of the five shards (exit 0 = complete, 3 = not)
    python -I eda/c132_m3px27_eval.py --early          # early-stop check on the finished shards (≥ 300 rows)
    python -I eda/c132_m3px27_eval.py                  # evaluation and verdict (exit 3 while a shard is missing)
    python -I eda/c132_m3px27_eval.py --dry [--splits N]   # CODE-PATH CHECK: m3px27 := px27 (copy) — NO verdict
    python -I eda/c132_m3px27_eval.py --oof-save | --oof-verify   # artifact artifacts/oof_m3px27.npy + meta_m3px27.json

CHANNEL m3px27 (training rows): five shards runs/m3px27_oof_rp/m3px27_rank_f{k}/bench (configs/vlm/m3px27_oof_rp.yaml) — per fold k the
exact ranking of the M3 adapter that did not learn fold k, behind the pr27 note of the row (derived/sft_m3px27/m3px27_train_notes.jsonl,
tools/make_m3px27_notes.py; character-identical with derived/proto_notes/train_pr27.jsonl, the input of px27); M3 writes the finding itself.
Conversion as for q397m3 (eda/c115_q397m3_eval.assemble -> a8.load_rank, vlm.engine.probs_from_logp, absent = classes without a training
example); every shard complete and exact and covering EXACTLY the rows of its fold.
TECHNICAL CHECKS (m3px27 rule; abort with exit 4, no verdict): 1,500/1,500 rows exact; note file = meta checksum, 1,500 rows, no finding
given, character-identical with the px27 notes; sft_from of every shard = the note file and ranking settings equal to d3 (as for d1–d4);
FOLD INTEGRITY (c108.fold_check: run configuration -> merge job -> sft_recipe.json holds out exactly this fold, gold-only, local adapter =
merged one; without findings and without unknowns); log: in EVERY ranking job (pre-run and five folds) the line "sft_from: notes for
1500 rows of table train, of which without a finding 1500" (log line of vlm/runner.py; LOG_LINE also accepts the earlier wording of the
recorded worker logs); pre-run computed without failure; NONE at the top (rank 1 only over classes with a training example, tie ->
smaller index) 0.30–0.38 (exactly as a fraction).
EARLY STOP (only to save compute; --early): same top m3px27/px27 ≥ 0.95 on COMPLETE shards with together ≥ 300 rows -> stop the GPU
machine, STOP (copy). Partial states do not count (the probe computes the longest prompts first).
EXPECTED-VALUE RULE (recipe of the final primary form = c124 run_mm "real", feature set main_guard_i1_mm, NONE guard; forms F = per fold,
M = one set; Δ = variant − final primary form in units of 1/6 per row, ½(F+M) = (u_F + u_M)/18,000 exactly):
  PRIMARY ROLE pr27xm' = ½(pr27 + m3px27) instead of ½(pr27 + px27); the channel name stays, blend, β, margin alignment and second
               stage are refitted nested (swap the matrix, rebuild Static, c121.prepare, c124.run_mm).
  NULL POINT   m3px27 := px27 ⇒ final primary form bit-identical (matrix ½(pr27 + px27) = oof_pr27xm bit-identical; base = row values of
               c124 oof bit-identical).
  CONTROLS     of the same role, pre-registered: copy px27 (0), ½(px27 + d4) (−0.0009), d4 (−0.0017), copy m3x3_full (0.0000); d4 = M3
               draw runs/m3d4_oof_rp/m3d4_rank_f*/bench (not in the recipe). On the original split they must hit their pre-registered
               values to four decimals (POSITIVE CONTROL of the measurement path; otherwise a technical finding).
  SPLITS       StratifiedKFold(5, shuffle, 1000 + s) by root_code, s = 0 … 19 (c121.split_folds, as for the splits of c124); channel
               OOFs fixed.
PRE-REGISTERED m3px27 RULE (wording): "DEPLOY only if ½(F+M) on the original split lies above max(0, controls of the same role) […] AND
  the mean over 20 meta-splits is > 0 and above the control mean; secondary roles only nested; the expected-value rule additionally in
  the recipe without the 397B (report, no condition)."
READING (fixed before the run): (1) "above" = strictly greater, exactly in units; (2) control mean = mean of the four controls, each
  control its mean over THE SAME 20 splits; (3) the decision is taken only with exactly 20 splits and only in the final primary form;
  (4) the secondary roles (full replacement, seventh channel, fourth M3 draw) are NOT computed by this script — their deployment would
  need a nested choice of the role and is thus excluded; (5) a same top m3px27/px27 over all rows ≥ 0.95 is no condition, but is
  reported as a WARNING (suspected copy).
RECIPE WITHOUT THE 397B (report): the final primary form without the channel q397m3 (channels v1r, zs27q, think18nb, pr27xm, m3x3_full;
  everything else unchanged) — the same computation including the controls; the result is a report only.
CONTROLS CACHE (--controls): per recipe artifacts/eda/private/m3px27_eval/c132_controls_<recipe>.json with the row-value checksums of the
  base per split and the checksums of the control matrices; the evaluation recomputes the base of every split and takes a control over
  only with the same base checksum and the same inputs (otherwise it is recomputed).
OUTPUT (only aggregates on the console): artifacts/eda/private/m3px27_eval/c132_m3px27_eval[_dry].{json,txt} and …_rows.npz (private);
  c132_early.json; c132_controls_<recipe>.json; c132_preflight.json. --oof-save writes artifacts/oof_m3px27.npy / meta_m3px27.json
  (kaggle_core.artifacts.save_oof, reproducible byte for byte), independently of the verdict (OOFs are blend currency), but only with
  passed technical checks without the NONE band.
LIMITS: the same 1,500 labels as all measurements (no new holdout); base OOFs not nested in the meta fold; the 20 splits share the same
  rows (their σ is descriptive). Test side: M3 of the full fit behind the pr27 test notes (a different fit than the fold adapters, the
  same note source).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("RAYON_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import metric  # noqa: E402

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "m3px27_eval"
NAME = "c132_m3px27_eval"
CH = "m3px27"
RUN, TEST_RUN = "m3px27_oof_rp", "m3px27te_rp"
SHARDS = tuple(f"runs/{RUN}/m3px27_rank_f{k}/bench" for k in range(5))
SOURCE = "+".join(SHARDS)
SMOKE = "m3px27_smoke"
RANK_JOBS = (SMOKE,) + tuple(f"m3px27_rank_f{k}" for k in (1, 2, 0, 3, 4))  # order of the run configuration
REF_SHARDS = tuple(f"runs/m3d3_oof_rp/m3d3_rank_f{k}/bench" for k in range(5))  # d3: ranking settings for comparison
NOTES_REL = "derived/sft_m3px27/m3px27_train_notes.jsonl"
TEST_NOTES_REL = "derived/sft_m3px27te/m3px27_test_notes.jsonl"
Q397M3_NOTES_REL = "derived/sft/q397m3_train_notes.jsonl"
PR27, PX27, SLOT, M3, Q3, D1 = "pr27", "pr27x", "pr27xm", "m3x3_full", "q397m3", "m3p5"
D4_SPEC = "+".join(f"runs/m3d4_oof_rp/m3d4_rank_f{k}/bench" for k in range(5))
STACK = "s032_stage2"
FORMS = ("fold", "set")
SPLIT_SEED0, N_SPLITS = 1000, 20
UNITS_ROWS = 9000  # 1,500 rows × 6 units
NONE_BAND = (Fraction(30, 100), Fraction(38, 100))
EARLY_SAME_TOP, EARLY_MIN_ROWS = Fraction(95, 100), 300
COPY_WARN = Fraction(95, 100)
CONTROLS = ("copy_px27", "half_px27_d4", "d4", "copy_m3x3_full")
CONTROL_LABEL = {"copy_px27": "copy px27", "half_px27_d4": "½(px27 + d4)", "d4": "d4", "copy_m3x3_full": "copy m3x3_full"}
PREREG_ORIG = {"copy_px27": 0.0, "half_px27_d4": -0.0009, "d4": -0.0017, "copy_m3x3_full": 0.0}  # pre-registered (m3px27 rule), four decimals
PREREG_TOL = 0.00005
RECIPES = ("primary_final", "no397")
RECIPE_LABEL = {"primary_final": "final primary form (decides)", "no397": "final primary form without q397m3 (report only)"}
MM_ROWS = ART / "eda" / "private" / "stage2_mm" / "c124_mm_oof_rows.npz"
# Log lines of vlm/runner.py (sft_from line and job header). The worker logs of the recorded runs were written with an earlier, German
# wording of the same lines ("sft_from: Ausgaben zu N Zeilen der Tabelle T, davon ohne Befund M", "=== Auftrag <name> (<kind>) ===");
# both wordings are accepted, the groups (rows, table, without a finding; name, kind) are the same in both.
LOG_LINE = re.compile(r"sft_from: (?:notes for|Ausgaben zu) (\d+) (?:rows of table|Zeilen der Tabelle) (\w+), "
                      r"(?:of which without a finding|davon ohne Befund) (\d+)")
JOB_LINE = re.compile(r"=== (?:job|Auftrag) (\S+) \((\w+)\) ===")
HF_27B = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.8-27B" / "snapshots" / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECT_ADAPTERS = {1: "../m3_pilot_rp/m3_train_f1", 2: "../m3_pilot_rp/m3_train_f2", 0: "../m3_f034_rp/m3_train_f0",
                   3: "../m3_f034_rp/m3_train_f3", 4: "../m3_f034_rp/m3_train_f4"}
FULL_ADAPTER = "../m3_fullte_rp/m3_train_full"
RANK_FREE = {"name", "model", "sft_from", "rows"}
RULE_TEXT = ("DEPLOY only if ½(F+M) on the original split lies above max(0, controls of the same role) (copy px27, ½(px27 + d4), "
             "d4, copy m3x3_full) AND the mean over 20 meta-splits is > 0 and above the control mean; secondary roles only nested; "
             "the expected-value rule additionally in the recipe without the 397B (report).")
CONSEQUENCE = {"DEPLOY": "test side configs/vlm/m3px27te_rp.yaml, then the build script tools/builds/primary_m3px27.sh with the copy "
                         "null point (m3px27 test := n1_rp rank_test ⇒ CSV of the final primary form byte-identical); candidate of the primary line",
               "NO DEPLOYMENT": "no test side; m3px27 rejected in the primary role; the artifact oof_m3px27 stays blend currency"}

_MODS: dict = {}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c115():
    """eda/c115_q397m3_eval.py (assemble, check_shard_settings, top1, same_top) — underneath C107, C108, A9, A8."""
    if "c115" not in _MODS:
        _MODS["c115"] = _load("c132_c115", COMP / "eda" / "c115_q397m3_eval.py")
    return _MODS["c115"]


def c124():
    """eda/c124_stage2_mm.py (recipe of the final primary form: load_context, run_mm; underneath c121, c118, stage2) — unchanged, only
    as a module."""
    if "c124" not in _MODS:
        _MODS["c124"] = _load("c132_c124", COMP / "eda" / "c124_stage2_mm.py")
    return _MODS["c124"]


def mp():
    """tools/make_m3px27_notes.py (read_jsonl, check_px27, constants)."""
    if "mp" not in _MODS:
        _MODS["mp"] = _load("c132_mp", COMP / "tools" / "make_m3px27_notes.py")
    return _MODS["mp"]


class TechnicalError(RuntimeError):
    """Technical finding: abort with exit 4, no verdict, nothing written."""


class Missing(TechnicalError):
    """Input still missing (run not finished or not fetched): exit 3, nothing written."""


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
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=json_default) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def npy_sha256(P: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(P, dtype=np.float64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ rules (pure, tested)

def half_of(u_fold: int, u_set: int) -> Fraction:
    """½(F+M) exactly: (u_F + u_M) / (2 · 9,000) — u = sum of the row changes in units of 1/6."""
    return Fraction(int(u_fold) + int(u_set), 2 * UNITS_ROWS)


def decide(cand_orig: Fraction, ctrl_orig: dict, cand_splits: list, ctrl_splits: dict, n_splits: int = N_SPLITS) -> dict:
    """Pre-registered m3px27 rule (reading in the module docstring). Values as Fraction (½(F+M)); ctrl_*: per control (all four)
    the original or the list over the same n_splits splits."""
    if set(ctrl_orig) != set(CONTROLS) or set(ctrl_splits) != set(CONTROLS):
        raise ValueError(f"controls {sorted(ctrl_orig)} / {sorted(ctrl_splits)} instead of {list(CONTROLS)}")
    if len(cand_splits) != n_splits or any(len(v) != n_splits for v in ctrl_splits.values()):
        raise ValueError(f"exactly {n_splits} splits required for the candidate and the controls")
    cand_orig = Fraction(cand_orig)
    bar = max([Fraction(0)] + [Fraction(v) for v in ctrl_orig.values()])
    c_mean = sum(Fraction(v) for v in cand_splits) / n_splits
    ctrl_means = {k: sum(Fraction(x) for x in v) / n_splits for k, v in ctrl_splits.items()}
    ctrl_mean = sum(ctrl_means.values()) / len(ctrl_means)
    checks = {"original_above_max_0_controls": bool(cand_orig > bar), "splits_mean_above_0": bool(c_mean > 0),
              "splits_mean_above_control_mean": bool(c_mean > ctrl_mean)}
    verdict = "DEPLOY" if all(checks.values()) else "NO DEPLOYMENT"
    reasons = []
    if not checks["original_above_max_0_controls"]:
        reasons.append(f"original ½(F+M) {float(cand_orig):+.5f} not above max(0, controls) = {float(bar):+.5f}")
    if not checks["splits_mean_above_0"]:
        reasons.append(f"mean over {n_splits} splits {float(c_mean):+.5f} not > 0")
    if not checks["splits_mean_above_control_mean"]:
        reasons.append(f"mean over {n_splits} splits {float(c_mean):+.5f} not above the control mean {float(ctrl_mean):+.5f}")
    strict = max(ctrl_means.values())
    return {"verdict": verdict, "checks": checks, "reasons": reasons, "rule": RULE_TEXT,
            "original": {"candidate": cand_orig, "bar": bar, "controls": {k: Fraction(v) for k, v in ctrl_orig.items()}},
            "splits": {"n": n_splits, "candidate_mean": c_mean, "control_means": ctrl_means, "control_mean": ctrl_mean,
                       "candidate_share_pos": float(np.mean([Fraction(v) > 0 for v in cand_splits])),
                       "sensitivity_strict_max_control_mean": strict, "strict_would_pass": bool(c_mean > 0 and c_mean > strict)}}


def decide_early(n_same: int, n_rows: int, n_cover: int | None = None) -> dict:
    """Early stop (m3px27 rule): same top m3px27/px27 ≥ 0.95 (n_same of n_rows) on complete shards with together ≥ 300 rows (n_cover;
    default n_rows) -> STOP; otherwise keep computing."""
    n_cover = int(n_rows if n_cover is None else n_cover)
    if n_cover < EARLY_MIN_ROWS or n_rows <= 0:
        return {"verdict": "TOO EARLY", "n_same": int(n_same), "n_rows": int(n_rows), "n_cover": n_cover}
    stop = Fraction(int(n_same), int(n_rows)) >= EARLY_SAME_TOP
    return {"verdict": "STOP (early stop: copy)" if stop else "KEEP COMPUTING", "n_same": int(n_same), "n_rows": int(n_rows),
            "n_cover": n_cover, "share": n_same / n_rows}


def none_band_ok(n_none: int, n_rows: int) -> bool:
    """NONE at the top within the band 0.30–0.38 (both bounds included, exactly)."""
    if n_rows <= 0:
        return False
    s = Fraction(int(n_none), int(n_rows))
    return NONE_BAND[0] <= s <= NONE_BAND[1]


def log_check(text: str, jobs=RANK_JOBS, n: int = 1500, table: str = "train") -> dict:
    """Log of the worker: per ranking job (header line "=== job <name> (bench) ===") the line "sft_from: notes for n rows of table
    <table>, of which without a finding n" before the next job (both wordings, see LOG_LINE). -> {per_job, problems}."""
    current, seen = None, {}
    for line in str(text).split("\n"):
        m = JOB_LINE.search(line)
        if m:
            current = m.group(1)
            continue
        m = LOG_LINE.search(line)
        if m and current is not None:
            seen.setdefault(current, []).append((int(m.group(1)), m.group(2), int(m.group(3))))
    per, problems = {}, []
    for j in jobs:
        got = seen.get(j, [])
        per[j] = [{"rows": a, "table": b, "without_finding": c} for a, b, c in got]
        if len(got) != 1:
            problems.append(f"{j}: {len(got)} sft_from lines in the log (exactly one required)")
        elif got[0] != (n, table, n):
            problems.append(f"{j}: \"notes for {got[0][0]} rows of table {got[0][1]}, of which without a finding {got[0][2]}\" instead of {n}/{table}/{n}")
    return {"per_job": per, "problems": problems}


def top1(P: np.ndarray, seen: np.ndarray) -> np.ndarray:
    return metric.top3_indices(P, seen)[:, 0]


def same_top(Pa: np.ndarray, Pb: np.ndarray, seen: np.ndarray, rows=None) -> tuple[int, int]:
    """(rows with the same top, rows) on `rows` (mask or indices; default all)."""
    ta, tb = top1(Pa, seen), top1(Pb, seen)
    if rows is None:
        m = np.ones(len(ta), bool)
    else:
        r = np.asarray(rows)
        m = r if r.dtype == bool else np.isin(np.arange(len(ta)), r)
    return int((ta[m] == tb[m]).sum()), int(m.sum())


def units_of(d: np.ndarray) -> int:
    """Sum of a row change in units of 1/6 (exact; otherwise an error)."""
    u = np.rint(6.0 * np.asarray(d, dtype=np.float64))
    if not np.allclose(u / 6.0, d, rtol=0.0, atol=1e-9):
        raise TechnicalError("row values are not multiples of 1/6")
    return int(u.sum())


def rows_hash(r: dict) -> str:
    """Checksum of the row values of a recipe run (forms F and M: forced, order, candidates)."""
    h = hashlib.sha256()
    for f in FORMS:
        for k in ("forced", "top", "cand"):
            a = np.ascontiguousarray(r[f][k])
            h.update(f"{f}/{k}/{a.dtype.str}/{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ notes and shards

def check_notes(root: Path, n: int) -> dict:
    """Note file = checksum of the meta file; table train with all n rows; no finding given; character-identical with the px27
    notes (tools/make_m3px27_notes.check_px27). -> checksum, states per row, texts."""
    from vlm import sft as S

    f = Path(root) / NOTES_REL
    mf = f.with_suffix(".meta.json")
    if not f.is_file() or not mf.is_file():
        raise Missing(f"{NOTES_REL} or meta file missing — python -I tools/make_m3px27_notes.py")
    meta = json.loads(mf.read_text())
    sha = sha256_file(f)
    if meta.get("sha256") != sha:
        raise TechnicalError(f"{NOTES_REL}: checksum does not match the meta file")
    extra = sorted(p.name for p in f.parent.iterdir() if p.name not in (f.name, mf.name))
    if extra:
        raise TechnicalError(f"upload folder {rel(f.parent)} contains foreign entries {extra}")
    table, notes = S.load_notes(f)
    if table != "train" or sorted(notes) != list(range(n)):
        raise TechnicalError(f"{NOTES_REL}: table {table}, {len(notes)} rows (required train, all {n})")
    if any(fd is not None for _, fd in notes.values()):
        raise TechnicalError(f"{NOTES_REL}: findings given — M3 is supposed to write the finding itself")
    recs = mp().read_jsonl(f)
    try:
        px = mp().check_px27(recs, root, "train")
    except mp().ConvertError as exc:
        raise TechnicalError(str(exc)) from None
    states = {int(r["row"]): str(r.get("pr27_state") or "") for r in recs}
    return {"sha256": sha, "rows": len(notes), "pr27_states": meta.get("pr27_states"), "px27_notes": px, "states_by_row": states,
            "texts": {i: t for i, (t, _) in notes.items()}}


def shard_dir(shard: str) -> tuple[Path, str]:
    return c115().shard_dir(shard)


def assemble(n: int, seen: np.ndarray, folds: np.ndarray, require_all: bool = True, shards=SHARDS):
    C = c115()
    try:
        return C.assemble(shards, n, seen, folds, require_all=require_all)
    except C.Missing as exc:
        raise Missing(str(exc)) from None
    except C.TechnicalError as exc:
        raise TechnicalError(str(exc)) from None


def run_root() -> Path:
    return ART / "runs" / RUN / RUN


def technical_checks(ctx: dict, P: np.ndarray, info: list, require_none_band: bool = True) -> dict:
    """Technical checks of the m3px27 rule on the complete channel. Violated -> TechnicalError."""
    C = c115()
    n, seen, folds = ctx["n"], ctx["seen"], ctx["folds"]
    out: dict = {"shards": info}
    try:
        st = C.check_shard_settings(SHARDS, REF_SHARDS, NOTES_REL)
    except C.TechnicalError as exc:
        raise TechnicalError(str(exc)) from None
    if not st["same_as_d3"]:
        raise TechnicalError("settings of the d3 ranking not readable — the comparison \"ranking exactly as d1–d4\" is not possible")
    out["settings"] = st
    recs = C.C108.fold_check(SOURCE, folds, {})
    bad = [f"{r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"] + r["unknown"]]
    if bad:
        raise TechnicalError("fold integrity violated or not determinable: " + "; ".join(bad))
    out["fold_check"] = [{k: x.get(k) for k in ("run", "job", "model", "merge_job", "train", "train_fold", "gold_only", "adapter_matches_merge",
                                                 "rows", "row_folds")} for x in recs]
    logf = run_root() / "worker0.log"
    if not logf.is_file():
        raise Missing(f"{rel(logf)} missing (run not completely fetched)")
    lc = log_check(logf.read_text(errors="replace"), RANK_JOBS, n, "train")
    if lc["problems"]:
        raise TechnicalError("log: " + "; ".join(lc["problems"]))
    out["log"] = lc
    sm = run_root() / SMOKE / "summary.json"
    if not sm.is_file():
        raise Missing(f"{rel(sm)} missing (pre-run not fetched)")
    timing = (json.loads(sm.read_text()).get("timing") or {})
    fails = [v.get("failed") for v in timing.values() if isinstance(v, dict) and "failed" in v]
    if not fails or any(int(x) != 0 for x in fails):
        raise TechnicalError(f"pre-run {SMOKE}: failures {fails} (required: computed without failure)")
    out["smoke"] = {"timing": timing}
    t = top1(P, seen)
    n_none = int((t == 0).sum())
    out["none_top"] = {"n": n_none, "rows": n, "share": n_none / n, "band": [float(x) for x in NONE_BAND], "ok": none_band_ok(n_none, n)}
    if require_none_band and not out["none_top"]["ok"]:
        raise TechnicalError(f"NONE at the top {n_none}/{n} = {n_none / n:.3f} outside 0.30–0.38")
    if not c115().C107.sane(P, seen):
        raise TechnicalError("channel not finite/normalised or with mass on classes without a training example")
    f = ART / f"oof_{CH}.npy"
    if f.is_file():
        if not np.array_equal(np.load(f), P):
            raise TechnicalError(f"{f.name} exists, but is not the channel from the five shards")
        out["oof_artifact_bit_identical"] = True
    out["channel_sha256"] = npy_sha256(P)
    return out


# ------------------------------------------------------------------------------------------------ recipe of the final primary form and swap

def load_ctx() -> dict:
    """Context of the primary line (c124.load_context: c118 + mech27), plus pr27 / px27 and the check of the role:
    oof_pr27xm = ½(oof_pr27 + oof_pr27x) bit-identical (otherwise the copy null point is no copy)."""
    M = c124()
    try:
        ctx = M.load_context(STACK)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 4
        raise (Missing if code == 3 else TechnicalError)(f"c124.load_context({STACK}) aborted (exit {code})") from None
    for nm in (PR27, PX27):
        f = ART / f"oof_{nm}.npy"
        if not f.is_file():
            raise Missing(f"{rel(f)} missing")
        ctx[nm] = np.load(f)
    if not np.array_equal(0.5 * (ctx[PR27] + ctx[PX27]), ctx["mats"][SLOT]):
        raise TechnicalError("oof_pr27xm is not bit-identical to ½(oof_pr27 + oof_pr27x) — the primary role is not defined this way")
    if list(ctx["spec"]["names"]) != ["v1r", "zs27q", "think18nb", SLOT, M3, Q3]:
        raise TechnicalError(f"recipe channels {ctx['spec']['names']} are not those of the final primary form")
    return ctx


def ctx_for(ctx0: dict, folds=None, slot=None, no397: bool = False) -> dict:
    """Variant of the context: another meta-split (folds), another matrix in the slot pr27xm (slot), without q397m3 (no397). Static is
    rebuilt as soon as channels or matrices change."""
    M = c124()
    S2 = M.S2
    ctx = dict(ctx0)
    ctx["mats"] = dict(ctx0["mats"])
    if folds is not None:
        ctx["folds"] = np.asarray(folds)
    if no397:
        sp = dict(ctx0["spec"])
        sp["names"] = [nm for nm in ctx0["spec"]["names"] if nm != Q3]
        sp["floors"] = {k: v for k, v in ctx0["spec"]["floors"].items() if k != Q3}
        ctx["spec"] = sp
    if slot is not None:
        ctx["mats"][SLOT] = np.asarray(slot, dtype=np.float64)
    if no397 or slot is not None:
        st = S2.Static(ctx["spec"]["names"], ctx["mats"], ctx["draws"], ctx["grade"], ctx["seen"], ctx["classes"], ctx["notes"])
        S2.attach_mech(st, ctx["mech27"], ctx["minor"])
        ctx["static"] = st
    return ctx


def run_recipe(ctx: dict) -> dict:
    """Recipe of the final primary form (nested): c121.prepare, c124.run_mm "real" for the forms F and M."""
    M = c124()
    M121 = M.c121()
    std, betas, nested = M121.prepare(ctx)
    out = {"betas": [float(b) for b in betas]}
    for f in FORMS:
        r = M.run_mm(ctx, std, nested, f, "real")
        out[f] = {k: r[k] for k in ("forced", "abstain", "top", "cand")}
    return out


def same_run(a: dict, b: dict) -> bool:
    return bool(all(np.array_equal(a[f][k], b[f][k]) for f in FORMS for k in ("forced", "top", "cand")))


def split_folds(y: np.ndarray, seed: int) -> np.ndarray:
    return c124().c121().split_folds(y, seed)


def control_matrices(ctx: dict) -> tuple[dict, dict]:
    """Controls of the same role (pre-registered) -> ({name: matrix X}, checksums of the inputs); the slot then holds ½(pr27 + X)."""
    A9 = c115().A9
    n, seen = ctx["n"], ctx["seen"]
    try:
        P4, h4, _ = A9.load_draw(D4_SPEC, n, seen)
    except SystemExit as exc:
        raise Missing(f"d4 ({D4_SPEC}): {exc}") from None
    if not h4.all():
        raise TechnicalError("d4 does not cover all training rows")
    px = ctx[PX27]
    X = {"copy_px27": px, "half_px27_d4": 0.5 * (px + P4), "d4": P4, "copy_m3x3_full": ctx["mats"][M3]}
    inputs = {"pr27": npy_sha256(ctx[PR27]), **{k: npy_sha256(v) for k, v in X.items()}}
    return X, inputs


def boot_entry(ctx: dict, rv: dict, rb: dict, draws) -> dict:
    """Original split: Δ per form with σ over questions (c118.boot), ½(F+M) row by row with σ, groups."""
    C = c124().c118()
    C107 = C.C107
    out = {}
    dd = {}
    for f in FORMS:
        d = rv[f]["forced"] - rb[f]["forced"]
        dd[f] = d
        b = C.boot(d, ctx, draws)
        out[f] = {"delta": float(b["delta"]), "se_q": float(b["se_q"]), "units": int(b["units"]), "fold_units": b["fold_units"],
                  "folds_pos": int(b["folds_pos"]), "other_top1": int((rv[f]["top"][:, 0] != rb[f]["top"][:, 0]).sum())}
    hb = C107.cluster_boot(0.5 * (dd["fold"] + dd["set"]), ctx["q_code"], draws)
    out["half_se_q"] = float(hb["se_q"])
    groups = {}
    for g in ("NONE", "seen", "never seen", "new question"):
        m = ctx["group"] == g
        if m.any():
            groups[g] = {"n": int(m.sum()), **{f: float(dd[f][m].sum() / ctx["n"]) for f in FORMS}}
    out["groups_contrib"] = groups
    return out


def measure(ctx0: dict, recipe: str, variants: dict, n_splits: int, cached: dict | None = None, ctrl_inputs: dict | None = None,
            log=say, keep_rows: bool = False) -> dict:
    """Expected-value computation of a recipe: original split and n_splits new splits; per split the base (final primary form, or
    without q397m3) and every variant X in the slot pr27xm = ½(pr27 + X). Controls from `cached` (same base checksum and inputs) are
    taken over. -> {orig: {...}, splits: [...], rows (optional)}."""
    no397 = recipe == "no397"
    draws = c124().c118().C107.question_draws(ctx0["q_code"])
    t0 = time.time()
    cache_ok = bool(cached) and ctrl_inputs is not None and cached.get("control_inputs") == ctrl_inputs
    by_seed = {}
    if cache_ok:
        by_seed = {str(r.get("seed")): r for r in [cached.get("orig", {})] + list(cached.get("splits", []))}
    res: dict = {"recipe": recipe, "orig": None, "splits": [], "reused": 0, "computed": 0}
    rows_out = {}
    for s in range(-1, int(n_splits)):
        seed = None if s < 0 else SPLIT_SEED0 + s
        folds = ctx0["folds"] if s < 0 else split_folds(ctx0["y"], seed)
        cb = ctx_for(ctx0, folds=folds, no397=no397)
        base = run_recipe(cb)
        h = rows_hash(base)
        if s < 0 and recipe == "primary_final":
            if not MM_ROWS.is_file():
                raise Missing(f"{rel(MM_ROWS)} missing (row values of the final primary form, c124 oof)")
            ref = np.load(MM_ROWS)
            if not all(np.array_equal(base[f][k], ref[f"mm_{f}_{k}"]) for f in FORMS for k in ("forced", "abstain", "top", "cand")):
                raise TechnicalError("the base of the original split does not hit the row values of the final primary form (c124 oof) "
                                     "bit-identically — recipe changed?")
        rec = {"seed": seed, "base_hash": h, "base": {f: float(base[f]["forced"].mean()) for f in FORMS}, "betas": base["betas"], "variants": {}}
        prev = by_seed.get(str(seed))
        for name, X in variants.items():
            if name in CONTROLS and prev is not None and prev.get("base_hash") == h and name in (prev.get("variants") or {}):
                rec["variants"][name] = prev["variants"][name]
                res["reused"] += 1
                continue
            cv = ctx_for(ctx0, folds=folds, slot=0.5 * (ctx0[PR27] + X), no397=no397)
            rv = run_recipe(cv)
            uF, uM = units_of(rv["fold"]["forced"] - base["fold"]["forced"]), units_of(rv["set"]["forced"] - base["set"]["forced"])
            ent = {"uF": uF, "uM": uM, "half": half_of(uF, uM), "bit_identical": same_run(rv, base)}
            if s < 0:
                ent.update(boot_entry(cb, rv, base, draws))
                if keep_rows:
                    rows_out.update({f"{recipe}_{name}_{f}": rv[f]["forced"] for f in FORMS})
            rec["variants"][name] = ent
            res["computed"] += 1
        if s < 0:
            res["orig"] = rec
            if keep_rows:
                rows_out.update({f"{recipe}_base_{f}": base[f]["forced"] for f in FORMS})
        else:
            res["splits"].append(rec)
        vs = " | ".join(f"{nm} {float(e['half']):+.5f}" for nm, e in rec["variants"].items())
        log(f"  [{recipe}] {'original' if s < 0 else f'split {seed}'}: base F {rec['base']['fold']:.5f} M {rec['base']['set']:.5f} | {vs} "
            f"({time.time() - t0:.0f} s)")
    if keep_rows:
        res["rows"] = rows_out
    return res


def summarize(res: dict, names) -> dict:
    """Per variant: original (units, ½(F+M)) and splits (mean, σ, share > 0)."""
    out = {}
    for nm in names:
        o = res["orig"]["variants"][nm]
        vals = [Fraction(r["variants"][nm]["half"]) for r in res["splits"]]
        a = np.array([float(v) for v in vals]) if vals else np.zeros(0)
        out[nm] = {"orig_half": Fraction(o["half"]), "orig_uF": int(o["uF"]), "orig_uM": int(o["uM"]), "orig_bit_identical": bool(o.get("bit_identical")),
                   "splits_n": len(vals), "splits_mean": (sum(vals) / len(vals)) if vals else None,
                   "splits_se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
                   "splits_share_pos": float(np.mean(a > 0)) if len(a) else None,
                   "splits_F_mean": float(np.mean([r["variants"][nm]["uF"] for r in res["splits"]]) / UNITS_ROWS) if vals else None,
                   "splits_M_mean": float(np.mean([r["variants"][nm]["uM"] for r in res["splits"]]) / UNITS_ROWS) if vals else None}
    return out


def positive_control(summ: dict, res: dict, recipe: str) -> dict:
    """Control values against the pre-registered values of the m3px27 rule (four decimals, original split; only in the final
    primary form) and the copy px27 bit-identical."""
    out = {"preregistered": {}, "ok": True}
    if recipe != "primary_final":
        out["note"] = "pre-registered values exist only for the final primary form"
        return out
    for nm in CONTROLS:
        v = float(summ[nm]["orig_half"])
        hit = abs(v - PREREG_ORIG[nm]) <= PREREG_TOL
        out["preregistered"][nm] = {"got": v, "preregistered": PREREG_ORIG[nm], "ok": bool(hit)}
        out["ok"] &= bool(hit)
    if summ["copy_px27"]["orig_uF"] != 0 or summ["copy_px27"]["orig_uM"] != 0 or not summ["copy_px27"]["orig_bit_identical"]:
        out["ok"] = False
        out["copy_px27_bit_identical"] = False
    else:
        out["copy_px27_bit_identical"] = True
    return out


def controls_path(recipe: str) -> Path:
    return OUT / f"c132_controls_{recipe}.json"


def load_cached_controls(recipe: str) -> dict | None:
    f = controls_path(recipe)
    if not f.is_file():
        return None
    js = json.loads(f.read_text())
    for blk in [js.get("orig") or {}] + list(js.get("splits") or []):
        for e in (blk.get("variants") or {}).values():
            if isinstance(e.get("half"), str):
                a, b = e["half"].split("/")
                e["half"] = Fraction(int(a), int(b))
    return js


def run_controls(ctx: dict, recipes, n_splits: int, log=say) -> dict:
    X, inputs = control_matrices(ctx)
    out = {}
    for recipe in recipes:
        t0 = time.time()
        res = measure(ctx, recipe, X, n_splits, log=log)
        summ = summarize(res, CONTROLS)
        pc = positive_control(summ, res, recipe)
        js = {"kind": f"m3px27 rule: controls of the same role in the slot pr27xm ({RECIPE_LABEL[recipe]})", "recipe": recipe, "n_splits": n_splits,
              "split_seeds": [SPLIT_SEED0 + s for s in range(n_splits)], "control_inputs": inputs, "orig": res["orig"], "splits": res["splits"],
              "summary": summ, "positive_control": pc, "seconds": round(time.time() - t0, 1), "created_utc": now_utc()}
        OUT.mkdir(parents=True, exist_ok=True)
        controls_path(recipe).write_text(to_json(js))
        out[recipe] = js
        log(f"[{recipe}] controls (original | mean over {n_splits} splits, share > 0): " + " ; ".join(
            f"{CONTROL_LABEL[nm]} {float(summ[nm]['orig_half']):+.5f} | {float(summ[nm]['splits_mean'] or 0):+.5f} ({summ[nm]['splits_share_pos']})"
            for nm in CONTROLS))
        if recipe == "primary_final":
            log(f"[{recipe}] POSITIVE CONTROL against the pre-registered values of the m3px27 rule (four decimals): {'PASSED' if pc['ok'] else 'FAILED'}")
        log(f"   written: {rel(controls_path(recipe))} ({js['seconds']} s)")
    return out


# ------------------------------------------------------------------------------------------------ channel statistics

def channel_stats(ctx: dict, P: np.ndarray) -> dict:
    C107 = c115().C107
    y, seen = ctx["y"], ctx["seen"]
    r = C107.rows_map(y, P, seen)
    draws = C107.question_draws(ctx["q_code"])
    cb = C107.cluster_boot(r, ctx["q_code"], draws)
    t = top1(P, seen)
    out = {"map3": float(r.mean()), "se_q": float(cb["se_q"]), "top1": float((t == y).mean()), "none_top": float((t == 0).mean()),
           "logloss": float(C107.rows_ll(y, P).mean()), "groups": {}}
    for g in ("NONE", "seen", "never seen", "new question"):
        m = ctx["group"] == g
        if m.any():
            out["groups"][g] = {"n": int(m.sum()), "map3": float(r[m].mean()), "top1": float((t[m] == y[m]).mean())}
    return out


def same_top_table(ctx: dict, P: np.ndarray, others: dict, states: dict) -> dict:
    seen = ctx["seen"]
    st = np.array([states.get(i, "") for i in range(ctx["n"])])
    out = {}
    for lab, B in others.items():
        s, k = same_top(P, B, seen)
        rec = {"n_same": s, "n_rows": k, "share": s / max(1, k), "by_state": {}, "by_group": {}}
        for name, label in mp().RECORDED_STATE.items():  # pr27 state: name -> label recorded in the note file
            sx, kx = same_top(P, B, seen, st == label)
            rec["by_state"][name] = {"n_same": sx, "n_rows": kx, "share": sx / max(1, kx)}
        for g in ("NONE", "seen", "never seen", "new question"):
            sg, kg = same_top(P, B, seen, ctx["group"] == g)
            rec["by_group"][g] = {"n_same": sg, "n_rows": kg, "share": sg / max(1, kg)}
        out[lab] = rec
    return out


def paired(ctx: dict, Pa: np.ndarray, Pb: np.ndarray) -> dict:
    C107 = c115().C107
    y, seen = ctx["y"], ctx["seen"]
    d = C107.rows_map(y, Pa, seen) - C107.rows_map(y, Pb, seen)
    cb = C107.cluster_boot(d, ctx["q_code"], C107.question_draws(ctx["q_code"]))
    return {"delta": float(d.mean()), "se_q": float(cb["se_q"]),
            "ll_delta": float(C107.rows_ll(y, Pa).mean() - C107.rows_ll(y, Pb).mean())}


def comparison_mats(ctx: dict) -> dict:
    mats = {PX27: ctx[PX27], PR27: ctx[PR27], SLOT: ctx["mats"][SLOT], M3: ctx["mats"][M3], Q3: ctx["mats"][Q3]}
    if D1 in ctx["mats"]:
        mats["d1"] = ctx["mats"][D1]
    return mats


# ------------------------------------------------------------------------------------------------ commands

def ctx_light() -> dict:
    """Light context (labels, folds, classes) for --status / --early."""
    from vlm import common as VC
    import labels

    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    return {"root": root, "train": train, "classes": classes, "y": labels.encode(train["root_code"], classes),
            "seen": labels.seen_mask(train["root_code"], classes), "folds": VC.load_folds(len(train)), "n": len(train)}


def cmd_status() -> int:
    C = c115()
    ctx = ctx_light()
    n, seen, folds = ctx["n"], ctx["seen"], ctx["folds"]
    ok_all = True
    for k, shard in enumerate(SHARDS):
        st = C.C108.shard_status(shard, n, seen)
        if st["missing"]:
            print(f"  fold {k} {shard}: MISSING")
            ok_all = False
            continue
        if st.get("error"):
            print(f"  fold {k} {shard}: ERROR {st['error']}")
            ok_all = False
            continue
        d, tag = shard_dir(shard)
        idx = np.load(d / f"{tag}_idx.npy")
        fold_ok = bool(len(idx) and set(folds[idx].tolist()) == {k} and len(idx) == int((folds == k).sum()))
        print(f"  fold {k} {shard}: {st['done']}/{st['rows']} computed, exact {st['exact']}, mass median {st['mass_median']:.3f}, "
              f"fold rows {'ok' if fold_ok else 'WRONG'} -> {'COMPLETE' if st['ok'] and fold_ok else 'INCOMPLETE'}")
        ok_all &= bool(st["ok"] and fold_ok)
    stat = run_root() / "status.json"
    print(f"  end of the run: {'status.json ' + stat.read_text().strip() if stat.is_file() else 'status.json missing (run not finished or not fetched)'}")
    print(f"-> {'COMPLETE' if ok_all else 'INCOMPLETE'}")
    return 0 if ok_all else 3


def cmd_early() -> int:
    ctx = ctx_light()
    n, seen, y = ctx["n"], ctx["seen"], ctx["y"]
    P, have, info = assemble(n, seen, ctx["folds"], require_all=False)
    px = np.load(ART / f"oof_{PX27}.npy")
    s, k = same_top(P, px, seen, have)
    dec = decide_early(s, k, int(have.sum()))
    rec = {"shards_complete": [x["shard"] for x in info], "rows": int(have.sum()), "same_top_px27": {"n_same": s, "n_rows": k}, "decision": dec,
           "rule": "early stop of the m3px27 rule: same top m3px27/px27 ≥ 0.95 on complete shards with together ≥ 300 rows -> stop the GPU machine, STOP"}
    if have.any():
        C107 = c115().C107
        others = {PX27: px, PR27: np.load(ART / f"oof_{PR27}.npy"), SLOT: np.load(ART / f"oof_{SLOT}.npy"), M3: np.load(ART / f"oof_{M3}.npy")}
        rec["channel_on_rows"] = {CH: float(C107.rows_map(y[have], P[have], seen).mean()),
                                  **{lab: float(C107.rows_map(y[have], A[have], seen).mean()) for lab, A in others.items()}}
        rec["same_top_on_rows"] = {lab: dict(zip(("n_same", "n_rows"), same_top(P, A, seen, have))) for lab, A in others.items()}
        rec["none_top_on_rows"] = float((top1(P, seen)[have] == 0).mean())
    logf = run_root() / "worker0.log"  # hint only (technical problems visible early): log lines of the jobs run so far
    if logf.is_file():
        lc = log_check(logf.read_text(errors="replace"), RANK_JOBS, n, "train")
        rec["log_so_far"] = {j: v for j, v in lc["per_job"].items() if v}
        rec["log_problems_so_far"] = [p for p in lc["problems"] if not p.endswith("0 sft_from lines in the log (exactly one required)")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c132_early.json").write_text(to_json({**rec, "created_utc": now_utc()}))
    print(f"early-stop check: {len(info)} complete shards, {int(have.sum())} rows; same top m3px27/px27 {s}/{k}"
          + (f" = {s / k:.3f}" if k else "") + f" -> {dec['verdict']}")
    if "channel_on_rows" in rec:
        print("  MAP@3 on these rows: " + ", ".join(f"{a} {b:.4f}" for a, b in rec["channel_on_rows"].items())
              + f"; NONE at the top {rec['none_top_on_rows']:.3f}")
        print("  same top m3px27 with: " + ", ".join(f"{a} {v['n_same']}/{v['n_rows']}" for a, v in rec["same_top_on_rows"].items()))
    if "log_so_far" in rec:
        print("  log so far: " + ("; ".join(f"{j} {v}" for j, v in rec["log_so_far"].items()) or "no sft_from line")
              + ("" if not rec["log_problems_so_far"] else " — PROBLEM: " + "; ".join(rec["log_problems_so_far"])))
    if dec["verdict"].startswith("STOP"):
        print("  -> stop the GPU machine; STOP under the m3px27 rule (copy)")
    return 0


def rule_block(ctx: dict, recipe: str, cand: np.ndarray, n_splits: int, log=say, keep_rows: bool = False) -> dict:
    """Expected-value rule of a recipe: candidate (slot pr27xm = ½(pr27 + cand)) and the four controls on the original split and
    n_splits splits; controls from the cache where valid. -> {res, summary, positive_control, decision (only with n_splits = 20)}."""
    X, inputs = control_matrices(ctx)
    variants = {CH: cand, **X}
    cached = load_cached_controls(recipe)
    res = measure(ctx, recipe, variants, n_splits, cached=cached, ctrl_inputs=inputs, log=log, keep_rows=keep_rows)
    summ = summarize(res, list(variants))
    pc = positive_control(summ, res, recipe)
    out = {"recipe": recipe, "label": RECIPE_LABEL[recipe], "n_splits": n_splits, "control_inputs": inputs, "reused_controls": res["reused"],
           "computed": res["computed"], "orig": res["orig"], "splits": res["splits"], "summary": summ, "positive_control": pc}
    if n_splits == N_SPLITS:
        out["decision"] = decide(summ[CH]["orig_half"], {k: summ[k]["orig_half"] for k in CONTROLS},
                                 [r["variants"][CH]["half"] for r in res["splits"]], {k: [r["variants"][k]["half"] for r in res["splits"]] for k in CONTROLS})
    if keep_rows:
        out["rows"] = res.get("rows", {})
    return out


def report_lines(rep: dict) -> list[str]:
    tag = "CHECK (--dry: m3px27 := px27) — " if rep["dry"] else ""
    L = [f"{tag}c132 m3px27 — notes {rep['technical'].get('notes', {}).get('sha256', '')[:16]}… (equal to px27: "
         f"{rep['technical'].get('notes', {}).get('px27_notes', {}).get('identical_notes')}/1500)"]
    for lab, c in rep["channel"].items():
        gs = " | ".join(f"{g} {v['map3']:.3f}" for g, v in c["groups"].items())
        L.append(f"  single channel {lab:9s} MAP@3 {c['map3']:.4f} ± {c['se_q']:.4f}, top correct {c['top1']:.3f}, NONE on top {c['none_top']:.3f}, "
                 f"log loss {c['logloss']:.3f} | {gs}")
    for lab, c in rep["channel_paired"].items():
        L.append(f"  paired {lab}: Δ {c['delta']:+.4f} ± {c['se_q']:.4f}, log loss {c['ll_delta']:+.3f}")
    for lab, s in rep["same_top"].items():
        bs = " ".join(f"{k} {v['share']:.3f} ({v['n_rows']})" for k, v in s["by_state"].items())
        L.append(f"  same top m3px27/{lab}: {s['n_same']}/{s['n_rows']} = {s['share']:.3f} | per pr27 state {bs}")
    for recipe in RECIPES:
        b = rep["rule_eval"].get(recipe)
        if not b:
            continue
        L.append(f"  expected-value rule, {b['label']} (controls taken over {b['reused_controls']}, computed {b['computed']}):")
        for nm, s in b["summary"].items():
            o = b["orig"]["variants"][nm]
            lab = "m3px27" if nm == CH else CONTROL_LABEL[nm]
            extra = (f" | F {o['fold']['delta']:+.5f} ± {o['fold']['se_q']:.5f} {o['fold']['fold_units']} M {o['set']['delta']:+.5f} ± "
                     f"{o['set']['se_q']:.5f} {o['set']['fold_units']}") if "fold" in o else ""
            sm = s["splits_mean"]
            L.append(f"    {lab:16s} original ½(F+M) {float(s['orig_half']):+.5f} (u_F {s['orig_uF']:+d}, u_M {s['orig_uM']:+d}{', bit-identical' if s['orig_bit_identical'] else ''})"
                     + (f" | {s['splits_n']} splits {float(sm):+.5f} ± {s['splits_se']:.5f}, > 0 in {s['splits_share_pos']:.2f}" if sm is not None else "")
                     + extra)
        pc = b["positive_control"]
        if recipe == "primary_final":
            L.append(f"    positive control (controls = pre-registered values of the m3px27 rule to four decimals): {'PASSED' if pc['ok'] else 'FAILED'}")
        d = b.get("decision")
        if d:
            L.append(f"    m3px27 rule in {recipe}: {d['verdict']} — " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in d["checks"].items())
                     + f"; control mean {float(d['splits']['control_mean']):+.5f}, strictest reading (above the largest control mean "
                       f"{float(d['splits']['sensitivity_strict_max_control_mean']):+.5f}) {'would pass' if d['splits']['strict_would_pass'] else 'would NOT pass'}")
    d = rep["decision"]
    L.append(f"VERDICT: {d['verdict']}" + (f" — reasons: {'; '.join(d['reasons'])}" if d.get("reasons") else ""))
    for w in d.get("warnings", []):
        L.append(f"  WARNING: {w}")
    if d.get("consequence"):
        L.append(f"  consequence: {d['consequence']}")
    return L


def cmd_eval(dry: bool, n_splits: int, recipes=RECIPES) -> int:
    t0 = time.time()
    ctx = load_ctx()
    n, seen = ctx["n"], ctx["seen"]
    nt = check_notes(ctx["root"], n)
    tech: dict = {"notes": {k: v for k, v in nt.items() if k not in ("texts", "states_by_row")}}
    if dry:
        P = ctx[PX27].copy()
        tech["channel"] = "CHECK: m3px27 := copy of px27"
    else:
        P, _, info = assemble(n, seen, ctx["folds"], require_all=True)
        tech.update(technical_checks(ctx, P, info))
    rep: dict = {"dry": dry, "technical": tech, "rule": RULE_TEXT}
    comp = comparison_mats(ctx)
    rep["channel"] = {CH: channel_stats(ctx, P), f"pr27xm'": channel_stats(ctx, 0.5 * (ctx[PR27] + P)),
                      **{lab: channel_stats(ctx, A) for lab, A in comp.items()}}
    rep["channel_paired"] = {"m3px27 − px27": paired(ctx, P, ctx[PX27]), "pr27xm' − pr27xm": paired(ctx, 0.5 * (ctx[PR27] + P), ctx["mats"][SLOT]),
                             "m3px27 − q397m3": paired(ctx, P, ctx["mats"][Q3])}
    rep["same_top"] = same_top_table(ctx, P, comp, nt["states_by_row"])
    rep["rule_eval"] = {}
    rows = {}
    for recipe in recipes:
        b = rule_block(ctx, recipe, P, n_splits, keep_rows=True)
        rows.update(b.pop("rows", {}))
        rep["rule_eval"][recipe] = b
        pc = b["positive_control"]
        if recipe == "primary_final" and not pc["ok"]:
            raise TechnicalError("positive control failed: the controls do not hit their pre-registered values (m3px27 rule) — "
                                 "measurement path changed")
        if recipe == "primary_final" and dry:
            bad = [nm for nm in [CH] if not (b["summary"][nm]["orig_bit_identical"] and all(r["variants"][nm]["bit_identical"] for r in b["splits"]))]
            if bad:
                raise TechnicalError("check --dry: m3px27 := px27 does not hit the final primary form bit-identically (copy null point violated)")
    warn = []
    st = rep["same_top"][PX27]
    if Fraction(st["n_same"], max(1, st["n_rows"])) >= COPY_WARN:
        warn.append(f"same top m3px27/px27 over all rows {st['n_same']}/{st['n_rows']} ≥ 0.95 (suspected copy; no condition of the rule)")
    b = rep["rule_eval"].get("primary_final")
    if b is None or "decision" not in b:
        dec = {"verdict": f"NO VERDICT (decision only in the final primary form and with exactly {N_SPLITS} splits)", "reasons": [], "checks": {}}
    else:
        dec = dict(b["decision"])
    if dry:
        dec = {**dec, "verdict": f"CHECK (--dry) — the rule would give {dec['verdict']}"}
    dec["warnings"] = warn
    dec["consequence"] = None if dry else CONSEQUENCE.get(dec["verdict"])
    no = rep["rule_eval"].get("no397", {}).get("decision")
    if no:
        dec["without_397b_report"] = no["verdict"]
    rep["decision"] = dec
    rep["seconds"] = round(time.time() - t0, 1)
    lines = report_lines(rep)
    for line in lines:
        print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = NAME + ("_dry" if dry else "")
    (OUT / f"{stem}.json").write_text(to_json({**rep, "created_utc": now_utc()}))
    (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    np.savez_compressed(OUT / f"{stem}_rows.npz", channel=c115().C107.rows_map(ctx["y"], P, seen), **rows)
    print(f"written: {rel(OUT / stem)}.json / .txt / _rows.npz ({rep['seconds']} s)")
    return 0


def cmd_controls(n_splits: int, recipes) -> int:
    ctx = load_ctx()
    run_controls(ctx, recipes, n_splits)
    return 0


# ------------------------------------------------------------------------------------------------ artifact

DEFINITION = ("pure ranking: per fold k the exact distribution over all codes (probe proto_rank, mode sft) of the M3 adapter without fold k "
              "behind the note of the first pr27 pass of the row (derived/sft_m3px27/m3px27_train_notes.jsonl = the px27 notes; M3 writes the "
              "finding itself); a8.load_rank / vlm.engine.probs_from_logp with absent = classes without a training example; no mean with named "
              "codes, no debiasing, no floor. Deployment role (m3px27 rule): pr27xm' = ½(pr27 + m3px27)")


def build_oof(ctx: dict) -> tuple[np.ndarray, dict]:
    n, seen, folds, y = ctx["n"], ctx["seen"], ctx["folds"], ctx["y"]
    nt = check_notes(ctx["root"], n)
    P, _, info = assemble(n, seen, folds, require_all=True)
    tech = technical_checks(ctx, P, info, require_none_band=False)
    from kaggle_core.folds import folds_hash

    r = metric.map3_rows_from_proba(y, P)
    meta = {"folds_hash": folds_hash(folds), "model_type": "vlm-sft-rank", "sources": [SOURCE], "cv_score": float(r.mean()),
            "cv_sigma": float(metric.bootstrap_sigma(r)),
            "feature_set": "image+task, without task knowledge (trained model M3, per fold the adapter without that fold; ranking behind the "
                           "notes of the first pr27 pass = the notes of px27)",
            "rows_without_answer": 0, "neutral_rows": None, "rows_neutral": 0,
            "rank": {"source": SOURCE, "rows": int(n), "rows_named_only": 0, "partial": False, "shards": info,
                     "settings": tech["settings"]["settings"], "settings_same_as_d3": True, "fold_check": tech["fold_check"],
                     "none_top": tech["none_top"],
                     "notes": {"train": {"sft_from": NOTES_REL, "sha256": nt["sha256"], "upload": NOTES_REL, "source_kind": "pr27-d1",
                                         "px27_notes": nt["px27_notes"]["file"], "px27_notes_sha256": nt["px27_notes"]["sha256"],
                                         "pr27_states": nt.get("pr27_states")},
                               "test": {"sft_from": TEST_NOTES_REL, "upload": TEST_NOTES_REL, "source_runs": ["runs/pr27te_rp/proto_te/proto"],
                                        "model": mp().MODEL},
                               "note": "note guard as for q397m3 (tools/make_submission.py rank_only_notes_guard); test notes with python -I "
                                       "tools/make_m3px27_notes.py --table test"},
                     "definition": DEFINITION, "function": "eda/c132_m3px27_eval.py --oof-save (c115.assemble / a8.load_rank)"},
            "checks": {"fold_check_clean": True, "log_without_finding": True, "settings_same_as_d3": True}}
    return P, meta


def meta_text(P: np.ndarray, meta: dict, K: int) -> str:
    from kaggle_core import artifacts as KA

    full = {"name": CH, "oof_shape": list(P.shape), "test_shape": [0, K], "oof_hash": KA._array_hash(P), **meta}
    return json.dumps(full, indent=2, ensure_ascii=False, default=str)


def cmd_oof(save: bool) -> int:
    ctx = load_ctx()
    P, meta = build_oof(ctx)
    K = ctx["K"]
    f, mf = ART / f"oof_{CH}.npy", ART / f"meta_{CH}.json"
    text = meta_text(P, meta, K)
    print(f"{CH}: ranking from {len(SHARDS)} shards ({ctx['n']} rows exact, fold integrity without findings, settings = d3); MAP@3 alone "
          f"{meta['cv_score']:.4f} ± {meta['cv_sigma']:.4f}; NONE at the top {meta['rank']['none_top']['share']:.3f}; SHA-256 of the values "
          f"{npy_sha256(P)[:16]}…")
    if not save:
        if not f.is_file() or not mf.is_file():
            raise Missing(f"{rel(f)} or {rel(mf)} missing — run --oof-save first")
        same = np.load(f).dtype == np.float64 and np.array_equal(np.load(f), P)
        same_meta = mf.read_text(encoding="utf-8") == text
        if not (same and same_meta):
            raise TechnicalError(f"{rel(f)} / {rel(mf)} is not the recomputation (values {'equal' if same else 'DIFFERENT'}, meta {'equal' if same_meta else 'DIFFERENT'})")
        print(f"   VERIFIED: {rel(f)} (SHA-256 {sha256_file(f)[:16]}…) and {rel(mf)} = recomputation")
        return 0
    if f.is_file() and not (np.load(f).dtype == np.float64 and np.array_equal(np.load(f), P)):
        raise TechnicalError(f"{rel(f)} exists with other values — not overwritten")
    if mf.is_file() and mf.read_text(encoding="utf-8") != text:
        raise TechnicalError(f"{rel(mf)} exists with other content — not overwritten")
    from kaggle_core import artifacts as KA

    KA.save_oof(CH, P, np.zeros((0, K)), meta, ART)
    if mf.read_text(encoding="utf-8") != text:
        raise TechnicalError("meta file not as expected after writing (save_oof changed?)")
    print(f"   saved: {rel(f)} (SHA-256 {sha256_file(f)}), {rel(mf)} (SHA-256 {sha256_file(mf)})")
    return 0


# ------------------------------------------------------------------------------------------------ preflight (local, without GPU)

def load_yaml(name: str) -> tuple[dict, dict, list]:
    import yaml

    cfg = yaml.safe_load((COMP / "configs" / "vlm" / f"{name}.yaml").read_text())
    jobs = [j for w in cfg["workers"] for j in w["jobs"]]
    return cfg, {j["name"]: j for j in jobs}, [j["name"] for j in jobs]


def config_check() -> dict:
    """Configurations as the runner reads them: jobs, models, merge jobs -> fold adapters, fold rule, note source, settings equal to d3."""
    from vlm import sft as S

    problems = []
    cfg, jobs, order = load_yaml(RUN)
    ref_cfg, ref, _ = load_yaml("m3d3_oof_rp")
    want = ["m3px27_merge_f1", SMOKE, "m3px27_rank_f1"] + [x for k in (2, 0, 3, 4) for x in (f"m3px27_merge_f{k}", f"m3px27_rank_f{k}")]
    if cfg.get("run") != RUN or order != want or len(cfg["workers"]) != 1 or cfg["workers"][0].get("gpu") != 0:
        problems.append(f"{RUN}: run/order/worker {cfg.get('run')} {order}")
    if cfg["models"].get("q38_27b") != ref_cfg["models"].get("q38_27b"):
        problems.append(f"{RUN}: base model differs from m3d3_oof_rp")
    merged_fold, adapters = {}, {}
    for nm in order:
        j = jobs[nm]
        spec = cfg["models"].get(j["model"]) or {}
        if j["kind"] == "merge":
            k = int(nm.rsplit("_f", 1)[1])
            if j["model"] != "q38_27b" or j["adapter_from"] != EXPECT_ADAPTERS[k] or j["adapter_from"] != ref[f"m3d3_merge_f{k}"]["adapter_from"]:
                problems.append(f"{nm}: adapter {j.get('adapter_from')} instead of {EXPECT_ADAPTERS[k]}")
            outm = [m for m, s in cfg["models"].items() if s.get("merged") == j["out_model"]]
            if len(outm) != 1:
                problems.append(f"{nm}: out_model {j['out_model']} not assigned to exactly one model")
                continue
            merged_fold[outm[0]], adapters[outm[0]] = k, j["adapter_from"]
            continue
        if j["kind"] != "bench" or j["model"] not in merged_fold:
            problems.append(f"{nm}: kind {j['kind']} / model {j['model']} used before its merge")
            continue
        k = merged_fold[j["model"]]
        if (j.get("rows") or {}).get("folds") != [k]:
            problems.append(f"{nm}: rows.folds {(j.get('rows') or {}).get('folds')} instead of [{k}] (fold rule)")
        if j.get("sft_from") != NOTES_REL or j.get("probes") != ["proto_rank"] or j.get("split", "train") != "train":
            problems.append(f"{nm}: sft_from/probes/split {j.get('sft_from')} {j.get('probes')} {j.get('split')}")
        if any(j.get(x) for x in S.NO_TASK_KEYS) or "sft_examples" in j:
            problems.append(f"{nm}: task knowledge in the job")
        if nm == SMOKE:
            if not (j.get("must_pass") is True and j["rows"].get("limit") == 6 and j.get("drop_model") is False and k == 1):
                problems.append(f"{nm}: pre-run not must_pass/limit 6/drop_model false on fold 1")
            main = jobs["m3px27_rank_f1"]
            free = ("name", "must_pass", "rows", "save_every", "drop_model")
            if {x: v for x, v in j.items() if x not in free} != {x: v for x, v in main.items() if x not in free}:
                problems.append(f"{nm}: pre-run computes differently from m3px27_rank_f1")
            continue
        rk = ref[f"m3d3_rank_f{k}"]
        if {x: v for x, v in j.items() if x not in RANK_FREE} != {x: v for x, v in rk.items() if x not in RANK_FREE} or j["rows"] != rk["rows"]:
            problems.append(f"{nm}: settings differ from m3d3_rank_f{k}")
    if sorted(merged_fold.values()) != [0, 1, 2, 3, 4]:
        problems.append(f"{RUN}: fold adapters {sorted(merged_fold.values())}")
    if "test" in json.dumps(cfg):
        problems.append(f"{RUN}: the OOF configuration names a test file or a test job")
    # test side
    tcfg, tjobs, torder = load_yaml(TEST_RUN)
    tref_cfg, tref, _ = load_yaml("m3d3te_rp")
    if tcfg.get("run") != TEST_RUN or torder != ["m3px27_merge_full", "m3px27_test_smoke", "m3px27_test_rank"]:
        problems.append(f"{TEST_RUN}: run/order {torder}")
    if tcfg["models"].get("q38_27b") != tref_cfg["models"].get("q38_27b"):
        problems.append(f"{TEST_RUN}: base model differs")
    mj = tjobs.get("m3px27_merge_full", {})
    if mj.get("adapter_from") != FULL_ADAPTER or mj.get("adapter_from") != tref["m3d3_merge"]["adapter_from"]:
        problems.append(f"{TEST_RUN}: full fit {mj.get('adapter_from')} instead of {FULL_ADAPTER}")
    if (tcfg["models"].get("m3px27_full") or {}).get("merged") != mj.get("out_model"):
        problems.append(f"{TEST_RUN}: model m3px27_full is not the merged one")
    for nm in ("m3px27_test_smoke", "m3px27_test_rank"):
        j = tjobs.get(nm, {})
        if j.get("kind") != "predict" or j.get("fold") is not None or j.get("model") != "m3px27_full" or j.get("sft_from") != TEST_NOTES_REL:
            problems.append(f"{TEST_RUN}/{nm}: kind/fold/model/sft_from")
        if [p.get("split") for p in j.get("predict", [])] != ["test"] or "rows" in j:
            problems.append(f"{TEST_RUN}/{nm}: predict not only test or with rows")
    rk, rref = tjobs.get("m3px27_test_rank", {}), tref["m3d3_test_rank"]
    if {x: v for x, v in rk.items() if x not in RANK_FREE} != {x: v for x, v in rref.items() if x not in RANK_FREE}:
        problems.append(f"{TEST_RUN}: settings of the test ranking differ from m3d3_test_rank")
    sm = tjobs.get("m3px27_test_smoke", {})
    if not (sm.get("must_pass") is True and (sm.get("predict") or [{}])[0].get("rows") == {"limit": 6} and sm["predict"][0].get("tag") == "smoke"):
        problems.append(f"{TEST_RUN}: pre-run not must_pass/limit 6/tag smoke")
    return {"problems": problems, "adapters": adapters, "merged_fold": merged_fold}


def runner_emulation(cc: dict) -> dict:
    """What the runner does before computing, without a model: find the note file (find_file), read it (sft.load_notes, log line),
    row selection (select_rows), notes for every selected row, sft.check_recipe with the recipe of the adapter (as it is in the model
    after the merge) — plus the counter-check: the recipe of a FOREIGN fold must be rejected."""
    from vlm import common as VC
    from vlm import runner as R
    from vlm import sft as S

    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    tax = VC.load_taxonomy(root)
    folds = VC.load_folds(len(train))
    system = S.system_text(tax, "full")
    sys_sha = S.sha256_text(system)
    _, jobs, order = load_yaml(RUN)
    out, problems = {"system_sha256": sys_sha, "jobs": {}}, []
    recipes = {}
    for model, src in cc["adapters"].items():
        run, job = src[3:].split("/")
        recipes[model] = S.read_adapter_recipe(ART / "runs" / run / run / job)
    for nm in order:
        j = jobs[nm]
        if j["kind"] != "bench":
            continue
        f = R.find_file(j["sft_from"], root)
        table, notes = S.load_notes(f)
        n_open = sum(fd is None for _, fd in notes.values())
        line = f"sft_from: notes for {len(notes)} rows of table {table}, of which without a finding {n_open} (findings are generated before scoring)"
        idx = R.select_rows(j.get("rows", {}), len(train), folds)
        have = np.isin(idx, np.fromiter(notes, dtype=np.int64, count=len(notes)))
        rec = recipes.get(j["model"])
        try:
            S.check_recipe(rec, system, "train", idx, folds)
            ok = True
        except ValueError as exc:
            ok, problems = False, problems + [f"{nm}: check_recipe: {exc}"]
        k = cc["merged_fold"][j["model"]]
        other = next(m for m, kk in cc["merged_fold"].items() if kk != k)
        try:
            S.check_recipe(recipes[other], system, "train", idx, folds)
            neg = "NOT rejected"
            problems.append(f"{nm}: counter-check — the recipe of fold {cc['merged_fold'][other]} was not rejected")
        except ValueError:
            neg = "rejected"
        if Path(f).resolve() != (root / NOTES_REL).resolve():
            problems.append(f"{nm}: find_file found {f}, not {NOTES_REL}")
        if not LOG_LINE.search(line) or LOG_LINE.search(line).groups() != (str(len(train)), "train", str(len(train))):
            problems.append(f"{nm}: the log line would be \"{line}\"")
        if not have.all():
            problems.append(f"{nm}: {int((~have).sum())} selected rows without a note")
        out["jobs"][nm] = {"file": rel(Path(f)), "log_line": line, "rows": int(len(idx)), "fold": k, "rows_in_fold": bool(np.all(folds[idx] == k)),
                           "check_recipe": "passed" if ok else "FAILED", "foreign_recipe": neg,
                           "recipe_fold": (rec or {}).get("fold"), "recipe_system_sha_ok": (rec or {}).get("system_sha256") == sys_sha}
    # test side: notes of all 3,498 rows, recipe of the full fit (fold null) with the same system text
    tf = R.find_file(TEST_NOTES_REL, root)
    ttable, tnotes = S.load_notes(tf)
    n_test = len(VC.load_table(root, "test")) if (root / "test.csv").is_file() else None
    full = S.read_adapter_recipe(ART / "runs" / "m3_fullte_rp" / "m3_fullte_rp" / "m3_train_full")
    try:
        S.check_recipe(full, system, "test", [], None)
        tok = "passed"
    except ValueError as exc:
        tok, problems = "FAILED", problems + [f"test side check_recipe: {exc}"]
    if ttable != "test" or n_test is None or len(tnotes) != n_test or any(fd is not None for _, fd in tnotes.values()):
        problems.append(f"test notes: table {ttable}, {len(tnotes)} rows (test table {n_test}), findings given?")
    out["test"] = {"file": rel(Path(tf)), "table": ttable, "rows": len(tnotes), "test_rows": n_test,
                   "without_finding": int(sum(fd is None for _, fd in tnotes.values())), "check_recipe_full": tok,
                   "full_recipe_fold": (full or {}).get("fold"), "full_rows_used": (full or {}).get("rows_used")}
    out["problems"] = problems
    return out


class _Shim:
    """Shim for vlm.engine.Engine.chat_text without a model (only the processor of the 27B for the chat template)."""

    def __init__(self, proc):
        self.proc = proc


IMG = object()  # placeholder: chat_text only sets {"type": "image"} for an image; no image is opened


def prompt_fidelity(limit: int | None = None) -> dict:
    """Prompt fidelity against q397m3 (the same structure, only other notes), against the M3 training (the same prompt as a training
    target row without loss on the notes) and against px27 (the same note, another system text and ending) — chat template and
    tokenizer of Qwen3.8-27B from the Hugging Face cache (revision of the run). Images as placeholders as in chat_text (image tokens are
    expanded when computing and are the same for both note sources of a row). Aggregates only."""
    from transformers import AutoProcessor

    from vlm import common as VC
    from vlm import engine as E
    from vlm import proto as PR
    from vlm import sft as S

    if not (HF_27B / "tokenizer.json").is_file():
        return {"skipped": f"{HF_27B} missing"}
    proc = AutoProcessor.from_pretrained(str(HF_27B))
    tok = proc.tokenizer
    shim = _Shim(proc)
    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    tax = VC.load_taxonomy(root)
    system, rank_sys = S.system_text(tax, "full"), PR.rank_system_text(tax)
    _, nm = S.load_notes(root / NOTES_REL)
    qfile = root / Q397M3_NOTES_REL
    nq = S.load_notes(qfile)[1] if qfile.is_file() else None
    rows = list(range(len(train)))[: limit or None]
    tail = S.CLOSE + S.HEAD
    t_tail = tok(tail, add_special_tokens=False)["input_ids"]
    cnt = {"rows": 0, "head_same_q397m3": 0, "user_same_px27": 0, "prefix_canonical": 0, "head_tokens_prefix_m": 0, "head_tokens_prefix_q": 0,
           "tail_tokens_m": 0, "tail_tokens_q": 0, "same_as_training_prompt": 0, "training_cut_rows": 0, "notes_in_px27_prompt": 0}
    ntok_m, ntok_q, ptok_m, ptok_q, diff_cut, htok = [], [], [], [], [], []
    rec_tr = {"finding": "none", "code": "NONE", "notes_loss": False}
    for i in rows:
        row = train.iloc[i]
        qimg = IMG if row["question_image_file"] else None
        notes_m = nm[i][0]
        sm = S.rank_sample(row, IMG, qimg, system, notes_m, None)
        text_m = E.Engine.chat_text(shim, sm)
        head = text_m[: len(text_m) - len(sm.prefix)]
        cnt["rows"] += 1
        cnt["prefix_canonical"] += int(sm.prefix == notes_m.strip() + tail and S.needs_finding(sm))
        ids_m = tok(text_m, add_special_tokens=False)["input_ids"]
        ids_h = tok(head, add_special_tokens=False)["input_ids"]
        htok.append(len(ids_h))
        cnt["head_tokens_prefix_m"] += int(ids_m[: len(ids_h)] == ids_h)
        cnt["tail_tokens_m"] += int(ids_m[-len(t_tail):] == t_tail)
        ntok_m.append(len(ids_m) - len(ids_h) - len(t_tail))
        ptok_m.append(len(ids_m))
        # M3 training: the same row as a training target without loss on the notes (notes cut to 2,500 tokens as in the runner)
        cut, _, was_cut = S.cut_notes(notes_m, tok, 2500)
        st = S.train_sample(row, IMG, qimg, system, rec_tr, cut)
        same_tr = E.Engine.chat_text(shim, st) == text_m
        cnt["same_as_training_prompt"] += int(same_tr)
        cnt["training_cut_rows"] += int(was_cut)
        if was_cut:
            diff_cut.append(len(ids_m) - len(tok(E.Engine.chat_text(shim, st), add_special_tokens=False)["input_ids"]))
        if not same_tr and not was_cut:
            cnt.setdefault("training_mismatch_uncut", 0)
            cnt["training_mismatch_uncut"] += 1
        # px27: the same note under the system text of the ranking, ending "Root:"
        sp = PR.rank_sample(row, IMG, qimg, rank_sys, notes_m)
        text_p = E.Engine.chat_text(shim, sp)
        cnt["notes_in_px27_prompt"] += int(sp.prefix == notes_m.rstrip() + "\n</think>\n\nRoot:")
        user = lambda t: t.split("<|im_start|>user\n", 1)[1].split("<|im_end|>\n<|im_start|>assistant", 1)[0]  # noqa: E731
        cnt["user_same_px27"] += int(user(text_p) == user(text_m))
        if nq is not None:
            sq = S.rank_sample(row, IMG, qimg, system, nq[i][0], None)
            text_q = E.Engine.chat_text(shim, sq)
            cnt["head_same_q397m3"] += int(text_q[: len(text_q) - len(sq.prefix)] == head)
            ids_q = tok(text_q, add_special_tokens=False)["input_ids"]
            cnt["head_tokens_prefix_q"] += int(ids_q[: len(ids_h)] == ids_h)
            cnt["tail_tokens_q"] += int(ids_q[-len(t_tail):] == t_tail)
            ntok_q.append(len(ids_q) - len(ids_h) - len(t_tail))
            ptok_q.append(len(ids_q))
    q = lambda a: {str(p): int(np.percentile(a, p)) for p in (5, 50, 95, 100)} if len(a) else {}  # noqa: E731
    return {"tokenizer": f"Qwen3.8-27B ({HF_27B.name[:8]}, HF cache)", "system_sha256": S.sha256_text(system),
            "rank_system_sha256_px27": S.sha256_text(rank_sys), "head_tokens": q(htok), "tail": repr(tail),
            "tail_tokens": t_tail, "counts": cnt, "notes_tokens_m3px27": q(ntok_m), "notes_tokens_q397m3": q(ntok_q),
            "prompt_tokens_text_m3px27": q(ptok_m), "prompt_tokens_text_q397m3": q(ptok_q),
            "training_cut_extra_tokens": q(diff_cut), "note": "prompt tokens without image expansion (one placeholder token per image)"}


def cmd_preflight(limit: int | None) -> int:
    t0 = time.time()
    from vlm import common as VC

    rep: dict = {}
    cc = config_check()
    rep["config"] = {"problems": cc["problems"], "adapters": cc["adapters"]}
    rep["runner"] = runner_emulation(cc)
    root = VC.find_data_root()
    nt = check_notes(root, 1500)
    rep["notes"] = {k: v for k, v in nt.items() if k not in ("texts", "states_by_row")}
    tm = json.loads((root / TEST_NOTES_REL).with_suffix(".meta.json").read_text())
    rep["test_notes"] = {"sha256": tm.get("sha256"), "sha_ok": tm.get("sha256") == sha256_file(root / TEST_NOTES_REL), "rows": tm.get("rows"),
                         "states": tm.get("states"), "px27_notes": tm.get("px27_notes"), "model": tm.get("model"), "source": tm.get("source"),
                         "tokens": tm.get("tokens")}
    rep["fidelity"] = prompt_fidelity(limit)
    probs = cc["problems"] + rep["runner"]["problems"]
    fc = rep["fidelity"].get("counts") or {}
    n = fc.get("rows", 0)
    if fc:
        for k in ("prefix_canonical", "head_tokens_prefix_m", "user_same_px27", "notes_in_px27_prompt"):
            if fc.get(k) != n:
                probs.append(f"prompt fidelity: {k} {fc.get(k)}/{n}")
        if fc.get("head_same_q397m3") not in (n, 0) or (fc.get("head_same_q397m3") == n and fc.get("head_tokens_prefix_q") != n):
            probs.append(f"prompt fidelity against q397m3: head equal {fc.get('head_same_q397m3')}/{n}, head tokens {fc.get('head_tokens_prefix_q')}/{n}")
        if fc.get("same_as_training_prompt", 0) + fc.get("training_cut_rows", 0) != n or fc.get("training_mismatch_uncut"):
            probs.append(f"prompt fidelity against the M3 training: equal {fc.get('same_as_training_prompt')}, cut {fc.get('training_cut_rows')}, "
                         f"uncut different {fc.get('training_mismatch_uncut', 0)} of {n}")
    if not rep["test_notes"]["sha_ok"] or rep["test_notes"]["rows"] != 3498:
        probs.append("test notes: checksum or number of rows")
    rep["problems"] = probs
    rep["ok"] = not probs
    rep["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c132_preflight.json").write_text(to_json({**rep, "created_utc": now_utc()}))
    print(f"preflight {'PASSED' if rep['ok'] else 'FAILED'} ({rep['seconds']} s); written {rel(OUT / 'c132_preflight.json')}")
    for nm, j in rep["runner"]["jobs"].items():
        print(f"  {nm}: {j['log_line'].split(' (')[0]} | {j['rows']} rows in fold {j['fold']} ({'ok' if j['rows_in_fold'] else 'WRONG'}) | "
              f"check_recipe {j['check_recipe']}, foreign recipe {j['foreign_recipe']}")
    t = rep["runner"]["test"]
    print(f"  test side: {t['rows']}/{t['test_rows']} notes, without a finding {t['without_finding']}, check_recipe (full fit, fold {t['full_recipe_fold']}) {t['check_recipe_full']}")
    if fc:
        print(f"  prompt fidelity ({n} rows): answer start canonical {fc['prefix_canonical']}, head tokens prefix {fc['head_tokens_prefix_m']}, tail tokens "
              f"{fc['tail_tokens_m']}; against q397m3 head equal {fc['head_same_q397m3']}, head tokens {fc['head_tokens_prefix_q']}, tail {fc['tail_tokens_q']}; "
              f"equal to the M3 training prompt {fc['same_as_training_prompt']} (+ {fc['training_cut_rows']} cut to 2,500 tokens in training); "
              f"user part equal to px27 {fc['user_same_px27']}, note in the px27 prompt {fc['notes_in_px27_prompt']}")
        fi = rep["fidelity"]
        print(f"  tokens of the note in the prompt: m3px27 {fi['notes_tokens_m3px27']} against q397m3 {fi['notes_tokens_q397m3']}; prompt without images "
              f"{fi['prompt_tokens_text_m3px27']} against {fi['prompt_tokens_text_q397m3']}; head (chat template up to \"<think>\") {fi['head_tokens']} tokens, "
              f"system text {fi['system_sha256'][:16]}…")
    for p in probs:
        print(f"  PROBLEM: {p}")
    return 0 if rep["ok"] else 4


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true")
    g.add_argument("--early", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--controls", action="store_true")
    g.add_argument("--dry", action="store_true")
    g.add_argument("--oof-save", action="store_true")
    g.add_argument("--oof-verify", action="store_true")
    ap.add_argument("--splits", type=int, default=N_SPLITS, help=f"new meta-splits (decision only with {N_SPLITS})")
    ap.add_argument("--recipe", choices=("both",) + RECIPES, default="both")
    ap.add_argument("--limit", type=int, help="--preflight: only the first N rows for the prompt fidelity")
    a = ap.parse_args(argv)
    recipes = RECIPES if a.recipe == "both" else (a.recipe,)
    try:
        if a.status:
            return cmd_status()
        if a.early:
            return cmd_early()
        if a.preflight:
            return cmd_preflight(a.limit)
        if a.controls:
            return cmd_controls(a.splits, recipes)
        if a.oof_save or a.oof_verify:
            return cmd_oof(save=a.oof_save)
        return cmd_eval(a.dry, a.splits, recipes)
    except Missing as exc:
        print(f"NOT YET EVALUABLE: {exc}", file=sys.stderr)
        return 3
    except TechnicalError as exc:
        print(f"ABORT (technical): {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
