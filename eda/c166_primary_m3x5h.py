"""DEPLOYMENT PATH OF PRIMARY SUBMISSION A primary_m3x5h: the intermediate build primary_cols (tools/builds/primary_cols.sh: primary_m3x4
with the extra columns qfp at the end of the second stage) with ONE change: the M3 channel m3x5h is per row the mean of the draws that cover
the row — d1, d2, d3, d4 on all rows, h (M3 with the photo at long side 2,048 px, seed 5) ONLY on the rows of the list h20 (character height
≤ 20 px; tools/make_charsize_rows.py). An internal evaluation measured this role ("add5_sm": h as fifth member of the M3 mean, only on the
rows with small script) over 20 untouched meta-splits; its sixths on the original split are pinned in this module (UNITS), and `measure`
recomputes the comparison over meta-splits. Build script tools/builds/primary_m3x5h.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c166_primary_m3x5h.py oof (--save | --verify)        # artifacts/oof_m3x5h.npy + meta_m3x5h.json (channel, row list, reference of the guard)
    python -I eda/c166_primary_m3x5h.py stack (--save | --verify)      # artifacts/stack_primary_m3x5h.json (blend as primary_m3x4, channel m3x5h)
    python -I eda/c166_primary_m3x5h.py fit-base (--save | --verify)   # artifacts/stage2_primary_m3x5h_base.json (chain s032_stage2 -> … -> primary_final as primary_m3x4)
    python -I eda/c166_primary_m3x5h.py fit (--save | --verify)        # artifacts/stage2_primary_m3x5h.json (columns qfp as primary_cols)
    python -I eda/c166_primary_m3x5h.py copy [--keep]                  # COPY NULL POINT "empty list": channel = m3x4_full -> blend, chain, qfp = primary_m3x4/primary_cols
    python -I eda/c166_primary_m3x5h.py clean-copy                     # remove the temporary copy files under artifacts/
    python -I eda/c166_primary_m3x5h.py measure [--seeds orig,11000-11019]   # comparison over meta-splits (sixths per split, base and add5_sm)
    python -I eda/c166_primary_m3x5h.py null --source F --out G        # null model (weight 0) for the deployment file
    python -I eda/c166_primary_m3x5h.py h-guard [--control none|h_shift|h_uniform|notes_cut]   # large-photo draw guard of the test draw h (exit 0/3/5)
    python -I eda/c166_primary_m3x5h.py h-controls                     # positive controls of the guard (each must fire, otherwise exit 6)
    python -I eda/c166_primary_m3x5h.py digest --side train|test       # one checksum over the files of one side of the draw h (pins of the build script)

CHANNEL m3x5h (training rows): eda/a9_m3_draws_oof.mean_of_draws over [(d1, all), (d2, all), (d3, all), (d4, all), (h, list)] — per row
Σ P_j / number of draws of the row (sum in this order); d1–d3 = meta_m3x3_full.json, d4 = runs/m3d4_oof_rp, h = runs/m3ha_oof_rp (folds 0–2)
+ runs/m3hb_oof_rp (folds 3, 4), shards m3h_rank_f<k>/bench; list data/derived/charsize/train_h20_rows.npy (1,001 rows, SHA-256 pinned).
Checks: all five draws complete and exact; mean d1–d4 = oof_m3x4_full bit-identical; outside the list = oof_m3x4_full bit-identical;
= channel add5_sm of the evaluation (np.where(list, mean(d1…d4, h), mean(d1…d4))) bit-identical. Meta: rank.row_list_draws (draw 5 with
training and test list including SHA-256, reference and rule of the large-photo draw guard) — tools/make_submission.py reads the field (path
"draw with row list").
BLEND (`stack`): eda/c135_primary_m3x4.build_stack with m3x5h in place of m3x3_full (θ on log loss from all 1,500 rows, β rule; as
primary_m3x4 with m3x4_full); only the texts (replaces, note, cv_note) rewritten for primary_m3x5h.
CHAIN (`fit-base`): as eda/c135_primary_m3x4.py fit (c135.swap_ctx, c133.oof_chain, c133.compose under c135._DrawsChannelAs) with m3x5h and
stack_primary_m3x5h.json; MM rows F and M = those of the bank variant (c135.Bank.variant with the channel m3x5h, features d1–d3; = the frame
of the evaluation) bit-identical and = sixths of the evaluation (arm add5_sm: line F / line M; pinned). Deployment file = fields as
stage2_primary_m3x4.json.
COLUMNS (`fit`): as eda/c151_primary_cols.py fit (c151.train_inputs, run_x, deploy_extra, fit_main, extra_field) on the base
stage2_primary_m3x5h_base.json; nested MM and MM + columns in F, oL, M; sixths = evaluation (arm add5_sm: F/M with the columns qfp; pinned);
null point (e).
COPY NULL POINT (`copy`): the same path with an EMPTY list (channel m3x5hk = oof_m3x4_full bit-identical): blend numbers =
stack_primary_m3x4.json, chain (MM rows = bank R4_0; files a–e = those of primary_m3x4 except provenance and channel names), qfp file =
stage2_primary_cols.json (except provenance and channel names); --keep leaves the copy files for the copy run of the test side (build script:
CSV and test matrix byte-identical to primary_cols).
LARGE-PHOTO DRAW GUARD (`h-guard`, fixed in advance, before any output of the test draw h was read): statistics of the test draw h on its
rows against the OOF draw h on the training list rows (reference in meta_m3x5h.json); hard conditions coverage = list, settings, training
regime. Aggregates only.
Only aggregates and checksums are printed. Exit 3 = input missing; 4 = checksum/recomputation/null point; 5 = test guard violated;
6 = positive control does not fire.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import stage2 as S2M  # noqa: E402  (the same module as in c118/c124: sys.modules['stage2'])
import stage2_sources as XS  # noqa: E402

ART = COMP / "artifacts"
PRIV = ART / "eda" / "private" / "primary_m3x5h"
VERSION = "c166-m3x5h-1"
CH, CH_COPY = "m3x5h", "m3x5hk"
STEM, STEM_COPY = "primary_m3x5h", "primary_m3x5h_copy"
TAG, TAG_COPY = "m3x5h", "m3x5hk"
COLS = "qfp"
STAGE2_BASE = ART / f"stage2_{STEM}_base.json"
STAGE2 = ART / f"stage2_{STEM}.json"
M3, M3X4 = "m3x3_full", "m3x4_full"
M3X4_OOF_SHA = "96ad8725bbaa9d274376a0bb68549abc4051878171ebb572f7707962600399a6"
M3X4_META_SHA = ""  # meta_m3x4_full.json is written by eda/c135_primary_m3x4.py oof: pinned after the first build (not used here)
STACK_M3X4 = ART / "stack_primary_m3x4.json"
STACK_M3X4_SHA = ""  # written by eda/c135_primary_m3x4.py stack: pinned after the first build (not used here)
STAGE2_M3X4 = ART / "stage2_primary_m3x4.json"
STAGE2_M3X4_SHA = ""  # written by eda/c135_primary_m3x4.py fit: likewise
STAGE2_COLS = ART / "stage2_primary_cols.json"
STAGE2_COLS_SHA = ""  # written by eda/c151_primary_cols.py fit: likewise
CHAIN_M3X4 = ART / "eda" / "private" / "primary_m3x4" / "chain"  # chain files a–d of primary_m3x4 (c135 fit)
H_SHARDS = {0: "m3ha_oof_rp", 1: "m3ha_oof_rp", 2: "m3ha_oof_rp", 3: "m3hb_oof_rp", 4: "m3hb_oof_rp"}
H_SPEC = "+".join(f"runs/{H_SHARDS[k]}/m3h_rank_f{k}/bench" for k in range(5))  # = the OOF draw h of the internal evaluation
H_TEST_RUNS = ("m3hte_a_rp", "m3hte_b_rp")
H_TEST_SPEC = "+".join(f"runs/{r}/m3h_test_rank/test" for r in H_TEST_RUNS)  # two shards, each half of the list
TEST_RANKS4 = ("runs/m3_fullte_rp/m3_test_rank/test", "runs/m3d2te_rp/m3d2_test_rank/test", "runs/m3d3te_rp/m3d3_test_rank/test",
               "runs/m3d4te_rp/m3d4_test_rank/test")
LISTS = {"train": ("derived/charsize/train_h20_rows.npy", "11afce9af96fca09032153b566ee81725f466ab927e782fcdad4324b55d18e77", 1001),
         "test": ("derived/charsize/test_h20_rows.npy", "9b1a1837bc2cf399e2e221c49def4b4ae11d666b5346ecb1eddc076622c8340a", 2459)}
LIST_META = ("derived/charsize/charsize_h20.meta.json", "4a48a02b00a53aaf022a48d3031ef2ad7cd8f1fc10052139cbb875c3431007df")
ARM = "add5_sm"
UNITS = {"base": {"line": [7592, 7542], "with": [7597, 7566]}, ARM: {"line": [7584, 7542], "with": [7583, 7564]}}  # internal evaluation, original split (pinned)
FULL_TRAIN = ("m3_fullte_rp", "m3_train_full")
FORMS3 = ("fold", "pre", "set")
UNITS_ROWS = 9000
RZH = {"rule": "large-photo draw guard", "rel": 0.10, "z_max": 3.0, "floors": {"share_answered": 0.97},
       "shares": ("share_answered", "share_complete", "share_none_top", "share_general_top", "share_same_top_mean4"),
       "means": ("notes_chars_mean", "mean_top_prob"),
       "text": ("test guard of the draw h (fixed in advance, before any output of the test draw h was read): statistic within the band ±10 % "
                "relative around the training value OR |z| ≤ 3 (shares: two-sample binomial test, stage2.share_z; means: Welch z); "
                "share_answered additionally ≥ 0.97; hard: coverage of the exact ranking = exactly the list, settings = OOF draw h, training "
                "regime = full fit")}
CONTROLS = ("h_shift", "h_uniform", "notes_cut")
NONE_BAND_H = (0.27, 0.42)  # malfunction band of NONE at the top of the test draw h on the test list (make_submission, row_list_draws[].none_band):
# OOF share on the training list 0.3457 ± 0.07, rounded outwards to 0.01 — fixed in advance, before any output of the test draw h was read:
# ±0.04 of the general band (whole test draw, 3,498 rows) scaled to 2,459 rows (±0.048) plus the shift of the subset (d4: test list 0.365
# against all test rows 0.344). The statistical check is done by the large-photo draw guard (tighter).
DEFINITION5 = ("mean of the probabilities of the M3 draws that cover the row (eda/a9_m3_draws_oof.mean_of_draws, order d1, d2, d3, d4, h): "
               "d1–d4 (photo 1,024 px; as m3x4_full) on all rows, h (photo 2,048 px, seed 5) only on the rows of the list h20 (character "
               "height ≤ 20 px, tools/make_charsize_rows.py); OOF per fold the adapter without that fold, test rows M3 of the full fit; before "
               "the floor; floor in the blend 0.05; features of the second stage still from d1–d3 (draws_first)")
ROWLIST_DEFINITION = ("draw with row list: the draw counts only on the rows of the list (training: OOF draw on train; test: test draw whose "
                      "exact ranking covers EXACTLY the test list — tools/make_submission.py, field rank.row_list_draws); all other draws "
                      "cover all rows; the blend is fitted on all training rows (the same structure on both sides)")
DECISION = "five-draw rule: large-photo draw on rows with small script (expected-value rule)"

_MODS: dict = {}


class TechnicalError(RuntimeError):
    """Checksum, recomputation or null point violated: exit 4."""


class Missing(TechnicalError):
    """Input missing: exit 3."""


class GuardFail(RuntimeError):
    """Test guard violated: exit 5."""


class ControlFail(RuntimeError):
    """Positive control does not fire: exit 6."""


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c151():
    if "c151" not in _MODS:
        _MODS["c151"] = _load("c166_c151", COMP / "eda" / "c151_primary_cols.py")
    return _MODS["c151"]


def c135():
    return c151().c135()  # the same module as in c151 (one primary_m3px27 context per process)


def c133():
    return c135().c133()


def c118():
    return c133().c118()


def c121():
    return c133().c121()


def c124():
    return c133().c124()


def S2():
    return S2M  # = c124().S2 (the same module); directly, so that the large-photo draw guard loads without the chain


def A9():
    return c118().A9


def ms():
    return c133().ms()


def say(*a) -> None:
    print(*a, flush=True)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def to_json(obj) -> str:
    return c133().to_json(obj)


def units_of(rows: np.ndarray) -> int:
    u = np.rint(6.0 * np.asarray(rows, dtype=np.float64))
    if not np.allclose(u / 6.0, rows, rtol=0.0, atol=1e-9):
        raise TechnicalError("row values are not multiples of 1/6")
    return int(u.sum())


def write_or_verify(path: Path, text: str, save: bool) -> str:
    return c133().write_or_verify_text(path, text, save)


# ------------------------------------------------------------------------------------------------ row lists and channel (pure functions)

def rows_sha256(rows) -> str:
    """Checksum of a row list as eda/a9_m3_draws_oof.rows_sha256 (int64 little-endian)."""
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).astype("<i8").tobytes()).hexdigest()


def resolve(root: Path, f: str) -> Path:
    return Path(f) if Path(f).is_absolute() else Path(root) / f


def load_list(root: Path, side: str, override: tuple[str, str, int] | None = None) -> tuple[np.ndarray, dict]:
    """Row list (pinned): file = SHA-256, ascending unique int64 positions, count as pinned. -> (positions, statistics)."""
    f, want, n_want = override or LISTS[side]
    p = resolve(root, f)
    if not p.is_file():
        raise Missing(f"row list {f} missing — tools/make_charsize_rows.py --max-height 20 --name h20")
    sha = sha256_file(p)
    if sha != want:
        raise TechnicalError(f"row list {f}: SHA-256 {sha[:16]}… instead of {want[:16]}… (pinned)")
    rows = np.load(p)
    if rows.dtype != np.int64 or rows.ndim != 1 or (len(rows) and (np.any(np.diff(rows) <= 0) or rows.min() < 0)) or len(rows) != n_want:
        raise TechnicalError(f"row list {f}: not an ascending list of unique positions with {n_want} entries")
    return rows, {"file": f, "sha256": sha, "rows": int(len(rows)), "rows_sha256": rows_sha256(rows)}


def mask_of(rows: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    m[np.asarray(rows, dtype=np.int64)] = True
    return m


def combine(draws: list[tuple[np.ndarray, np.ndarray]], restricted: dict[int, np.ndarray], n: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Channel from draws with a row list: draw j (0-based) in `restricted` counts only on the rows of its mask (∧ its coverage); then
    a9.mean_of_draws in the given order (sum in order, division by the number of draws of the row)."""
    use = [(P, (h & restricted[j]) if j in restricted else h) for j, (P, h) in enumerate(draws)]
    return A9().mean_of_draws(use, n, K)


# ------------------------------------------------------------------------------------------------ large-photo draw guard (statistics and rule)

def rzh_stats(P: np.ndarray, mean4: np.ndarray, rows: np.ndarray, notes: dict, general: np.ndarray) -> dict:
    """Statistics of the large-photo draw guard for the draw P on the rows `rows` (table positions). notes: {position: (state, text)} of the
    generation behind the ranking; general: mask of the five generic codes. Aggregates only."""
    rows = np.asarray(rows, dtype=np.int64)
    n = int(len(rows))
    if n == 0:
        return {"rows": 0}
    top, top4, pmax = P[rows].argmax(1), mean4[rows].argmax(1), P[rows].max(1)
    have = np.array([int(r) in notes for r in rows])
    comp = np.array([notes.get(int(r), ("", ""))[0] == "complete" for r in rows])
    chars = np.array([len(notes[int(r)][1]) for r in rows if int(r) in notes], dtype=np.float64)
    return {"rows": n, "share_answered": float(have.mean()), "share_complete": float(comp.mean()),
            "notes_chars_mean": float(chars.mean()) if len(chars) else 0.0, "notes_chars_sd": float(chars.std(ddof=1)) if len(chars) > 1 else 0.0,
            "notes_rows": int(len(chars)), "share_none_top": float(np.mean(top == 0)), "share_general_top": float(np.mean(general[top])),
            "mean_top_prob": float(pmax.mean()), "mean_top_prob_sd": float(pmax.std(ddof=1)) if n > 1 else 0.0,
            "share_same_top_mean4": float(np.mean(top == top4))}


def welch_z(m_ref: float, s_ref: float, n_ref: int, m_test: float, s_test: float, n_test: int) -> float:
    se = math.sqrt((s_ref ** 2) / max(n_ref, 1) + (s_test ** 2) / max(n_test, 1))
    d = float(m_test) - float(m_ref)
    if se == 0.0:
        return 0.0 if d == 0.0 else math.copysign(math.inf, d)
    return d / se


def rzh_check(ref: dict, test: dict, rule: dict = RZH) -> dict:
    """Rule of the large-photo draw guard: band |test − training| ≤ rel·|training| OR |z| ≤ z_max; floors additionally. -> {checks, ok, failed}."""
    checks = []
    n_r, n_t = int(ref["rows"]), int(test.get("rows") or 0)
    if n_t == 0:
        return {"checks": [], "ok": False, "failed": [{"stat": "rows", "train": n_r, "test": 0, "z": float("inf"), "ok": False}], "rule": rule["rule"]}
    for st in rule["shares"] + rule["means"]:
        r, t = float(ref[st]), float(test[st])
        if st in rule["shares"]:
            z = S2().share_z(r, n_r, t, n_t)
        else:
            sd = "notes_chars_sd" if st == "notes_chars_mean" else "mean_top_prob_sd"
            nr = int(ref["notes_rows"]) if st == "notes_chars_mean" else n_r
            nt = int(test["notes_rows"]) if st == "notes_chars_mean" else n_t
            z = welch_z(r, float(ref[sd]), nr, t, float(test[sd]), nt)
        ok_band = bool(abs(t - r) <= rule["rel"] * abs(r)) if r != 0.0 else bool(t == 0.0)
        ok_z = bool(abs(z) <= rule["z_max"])
        fl = rule["floors"].get(st)
        ok_floor = None if fl is None else bool(t >= fl)
        ok = bool((ok_band or ok_z) and ok_floor is not False)
        checks.append({"stat": st, "train": r, "test": t, "z": float(z), "ok_band": ok_band, "ok_z": ok_z, "floor": fl, "ok_floor": ok_floor, "ok": ok,
                       "passed_by": ("band" if ok_band else ("z" if ok_z else None)) if ok else None})
    failed = [c for c in checks if not c["ok"]]
    return {"checks": checks, "ok": not failed, "failed": failed, "rule": rule["rule"]}


def apply_control(control: str, P: np.ndarray, rows: np.ndarray, notes: dict, seen: np.ndarray) -> tuple[np.ndarray, dict]:
    """Positive controls of the test guard (in memory only): h_shift (draw shifted by one row within the list), h_uniform (draw = uniform
    distribution over the classes with a training example), notes_cut (notes cut to a quarter, states unchanged)."""
    rows = np.asarray(rows, dtype=np.int64)
    if control == "h_shift":
        Q = P.copy()
        Q[rows] = P[np.roll(rows, 1)]
        return Q, notes
    if control == "h_uniform":
        Q = P.copy()
        Q[rows] = (seen / seen.sum())[None, :]
        return Q, notes
    if control == "notes_cut":
        return P, {r: (s, t[: len(t) // 4]) for r, (s, t) in notes.items()}
    raise ValueError(f"unknown control {control!r}")


def read_gen_notes(run: str, rank_meta: dict, table: str) -> dict:
    """Generation behind a ranking (sft.sft_from = <job>/<tag> in the same run) -> {position: (state, text)} from <tag>_notes.jsonl (exactly
    one table, no position twice). Only state and length are used; nothing is printed."""
    src = str((rank_meta.get("sft") or {}).get("sft_from") or "")
    if "/" not in src:
        raise TechnicalError(f"run {run}: ranking without sft_from")
    job, tag = src.split("/", 1)
    f = A9().run_dir(run) / job / f"{tag}_notes.jsonl"
    if not f.is_file():
        raise Missing(f"artifacts/runs/{run}/{run}/{job}/{tag}_notes.jsonl missing")
    out = {}
    for line in f.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("table")) != table:
            raise TechnicalError(f"{f.name} ({run}): table {r.get('table')!r} instead of {table}")
        i = int(r["row"])
        if i in out:
            raise TechnicalError(f"{f.name} ({run}): a position twice")
        out[i] = (str(r.get("state") or ""), str(r.get("text") or ""))
    return out


def notes_of_spec(spec: str, table: str) -> dict:
    """Notes (state, text) of all shards of a draw (+ joined) — per shard the generation behind its ranking."""
    out: dict = {}
    for shard in str(spec).split("+"):
        run, job, tag = A9().parse_source(shard)
        meta = A9().read_json(A9().run_dir(run) / job / f"{tag}_meta.json")
        if meta is None:
            raise Missing(f"{shard}: {tag}_meta.json missing")
        for i, v in read_gen_notes(run, meta, table).items():
            if i in out:
                raise TechnicalError(f"{spec}: notes of the same position from two shards")
            out[i] = v
    return out


# ------------------------------------------------------------------------------------------------ OOF artifact m3x5h

def light() -> dict:
    return c135().light()


def train_reference(lc: dict, Ph: np.ndarray, rows: np.ndarray) -> dict:
    """Reference of the large-photo draw guard: OOF draw h on the training list rows; mean d1–d4 = oof_m3x4_full."""
    m4 = np.load(ART / f"oof_{M3X4}.npy")
    general = np.isin(np.array(lc["classes"]), S2().GENERAL_CODES)
    notes = notes_of_spec(H_SPEC, "train")
    return rzh_stats(Ph, m4, rows, notes, general)


def build_channel(lc: dict, rows_train: np.ndarray, list_info: dict, test_info: dict, name: str = CH) -> tuple[np.ndarray, dict, dict]:
    """Channel m3x5h (or the copy with an empty list) and its meta fields (notation as eda/a9_m3_draws_oof.py main / c135.build_m3x4)."""
    import metric
    from kaggle_core.folds import folds_hash

    n, K, y, seen, folds = lc["n"], lc["K"], lc["y"], lc["seen"], lc["folds"]
    specs = c135().four_specs() + [H_SPEC]
    _, _, draws, infos = A9().build(specs, n, seen)  # loads all five draws complete and exact; checks shards/jobs without repetition
    for j, (_, hj) in enumerate(draws):
        if not hj.all():
            raise TechnicalError(f"draw {j + 1} ({specs[j]}) covers {int(hj.sum())} instead of {n} training rows")
    m4 = np.load(ART / f"oof_{M3X4}.npy")
    if sha256_file(ART / f"oof_{M3X4}.npy") != M3X4_OOF_SHA:
        raise TechnicalError(f"oof_{M3X4}.npy is not the M3 channel of primary_m3x4 (checksum)")
    if not np.array_equal(A9().mean_of_draws(draws[:4], n, K)[0], m4):
        raise TechnicalError(f"mean of the draws d1–d4 != oof_{M3X4}.npy (bit-identity required)")
    inlist = mask_of(rows_train, n)
    P, cnt = combine(draws, {4: inlist}, n, K)
    if not np.array_equal(P[~inlist], m4[~inlist]) or not np.array_equal(cnt, 4 + inlist.astype(np.int64)):
        raise TechnicalError("outside the list != oof_m3x4_full bit-identical, or number of draws per row not 4 + list")
    Ph = draws[4][0]
    infos[4] = {**infos[4], "rows_shards": infos[4]["rows"], "rows_sha256_shards": infos[4]["rows_sha256"], "rows": int(inlist.sum()),
                "rows_sha256": rows_sha256(rows_train), "restricted_to": list_info["file"]}
    for j, (Pj, hj) in enumerate(draws):
        hj2 = hj & inlist if j == 4 else hj
        infos[j]["cv_alone_own_rows"] = float(metric.map3_rows_from_proba(y[hj2], Pj[hj2]).mean()) if hj2.any() else None
        infos[j]["none_top_own_rows"] = float(np.mean(Pj[hj2].argmax(1) == 0)) if hj2.any() else None
    r_all = metric.map3_rows_from_proba(y, P)
    on_list = {"rows": int(inlist.sum()), "mean": float(r_all[inlist].mean()) if inlist.any() else None,
               "mean_m3x4_full": float(metric.map3_rows_from_proba(y[inlist], m4[inlist]).mean()) if inlist.any() else None,
               **{f"draw{j + 1}": (float(metric.map3_rows_from_proba(y[inlist], Pj[inlist]).mean()) if inlist.any() else None)
                  for j, (Pj, _) in enumerate(draws)}}
    ref = train_reference(lc, Ph, rows_train) if len(rows_train) else {"rows": 0}
    rk = A9().rank_record(specs, cnt, infos, list_info["file"], np.asarray(rows_train, dtype=np.int64))
    rk["note"] = ("draw 5 (h) with a row list (rank.row_list_draws): blend weights fitted on ALL 1,500 training rows — on the test rows h covers "
                  "exactly the test list, the same structure as here (tools/make_submission.py, path \"draw with row list\")")
    rk["row_list_draws"] = [{"draw": 5, "source": H_SPEC, "definition": ROWLIST_DEFINITION, "train": list_info, "test": test_info,
                             "list_meta": {"file": LIST_META[0], "sha256": LIST_META[1]}, "none_band": list(NONE_BAND_H),
                             "guard": {**{k: v for k, v in RZH.items() if k != "text"}, "shares": list(RZH["shares"]), "means": list(RZH["means"]),
                                       "text": RZH["text"], "reference": ref}}]
    checks = {"m3x4_full": {"first_four_mean_bit_identical": True, "outside_list_bit_identical": True, "oof_sha256": M3X4_OOF_SHA},
              "measurement": "channel = add5_sm of the internal evaluation (np.where(list, mean(d1…d4, h), mean(d1…d4))) bit-identical (c166 oof)"}
    meta = {"folds_hash": folds_hash(folds), "model_type": "vlm-sft-rank", "sources": specs, "cv_score": float(r_all.mean()),
            "cv_sigma": float(metric.bootstrap_sigma(r_all)),
            "feature_set": "image+task, without task knowledge (trained model, ranking behind its own text)",
            "rows_without_answer": 0, "neutral_rows": None, "rows_neutral": 0, "rank": rk, "cv_rows_list": on_list, "checks": checks,
            "cv_note": (f"{DEFINITION5}; channel {'copy null point (empty list) — temporary, never submit' if name == CH_COPY else 'deployment primary_m3x5h'}")}
    return P, meta, {"draws": draws, "inlist": inlist, "Ph": Ph, "m4": m4}


def measurement_channel_check(P: np.ndarray, parts: dict) -> bool:
    """Cross-check against the computation of the internal evaluation (add5_sm): np.where(list, mean(d1…d4, h), mean(d1…d4)) with
    c135.mean_draws."""
    n, K = P.shape
    D = [Pj for Pj, _ in parts["draws"][:4]]
    add5 = c135().mean_draws(D + [parts["Ph"]], n, K)
    ch4 = c135().mean_draws(D, n, K)
    return bool(np.array_equal(P, np.where(parts["inlist"][:, None], add5, ch4)))


def cmd_oof(save: bool) -> int:
    lc = light()
    rows, li = load_list(lc["root"], "train")
    _, ti = load_list(lc["root"], "test")
    P, meta, parts = build_channel(lc, rows, li, ti)
    if not measurement_channel_check(P, parts):
        raise TechnicalError("channel != add5_sm of the internal evaluation — not bit-identical")
    info = c133().save_or_verify(CH, P, meta, lc["K"], save)
    ref = meta["rank"]["row_list_draws"][0]["guard"]["reference"]
    say(f"{CH}: mean of d1–d4 (all rows) and h (list {li['file']}, {li['rows']} rows) — MAP@3 alone {meta['cv_score']:.4f} ± {meta['cv_sigma']:.4f}; "
        f"on the list {meta['cv_rows_list']['mean']:.4f} (m3x4_full {meta['cv_rows_list']['mean_m3x4_full']:.4f}); rows by draws "
        f"{meta['rank']['rows_by_draws']}; = add5_sm of the evaluation bit-identical; outside the list = m3x4_full bit-identical")
    say(f"   reference of the guard (OOF h on {ref['rows']} list rows): answered {ref['share_answered']:.4f}, complete {ref['share_complete']:.4f}, notes "
        f"{ref['notes_chars_mean']:.0f} characters, NONE at the top {ref['share_none_top']:.4f}, generic {ref['share_general_top']:.4f}, top "
        f"{ref['mean_top_prob']:.4f}, = mean d1–d4 {ref['share_same_top_mean4']:.4f}")
    say(f"   {'stored' if save else 'CHECKED'}: {info['oof']} ({info['oof_sha256'][:16]}…), {info['meta']} ({info['meta_sha256'][:16]}…)")
    return 0


# ------------------------------------------------------------------------------------------------ blend

def build_stack5(name: str, P: np.ndarray) -> tuple[dict, dict]:
    """c135.build_stack (blend of primary_m3px27 with `name` in place of m3x3_full, θ on log loss from all rows, β rule) — texts for
    primary_m3x5h."""
    stack, rep = c135().build_stack(name, P)
    stack["replaces"] = {**stack["replaces"], "definition": DEFINITION5 if name == CH else f"copy null point \"empty list\" (= {M3X4} bit-identical)",
                         "decision": DECISION}
    stack["cv_note"] = stack["cv_note"].replace("decision on the channel and the second stage: four-draw rule (c135 rule)",
                                                "decision on the channel and the second stage: five-draw rule (internal evaluation, role add5_sm)")
    stack["note"] = (f"eda/c166_primary_m3x5h.py stack: recipe primary_m3px27 ({stack['replaces']['stack_primary_m3px27']}) with {name} instead of {M3} "
                     f"(same position, floor 0.05; as primary_m3x4 with {M3X4}); weights of ALL channels on log loss from all 1,500 rows, β by "
                     "arith_beta_rule; build script tools/builds/primary_m3x5h.sh")
    return stack, rep


def stack_numbers_equal_m3x4(a: dict, b: dict, name: str) -> list[str]:
    """Copy null point of the blend: numbers of a (channel `name`) = those of stack_primary_m3x4.json (channel m3x4_full) — θ, base θ, floors,
    β, cv, cross-fit, in-sample and β-rule numbers, rows, folds, fit rows, last step; rank_only except the M3 entry (source, draws)."""
    ren = lambda k: M3X4 if k == name else k  # noqa: E731
    bad = [k for k in ("theta", "arith_beta", "cv") if a[k] != b[k]]
    if a["base"]["theta"] != b["base"]["theta"] or [ren(c) for c in a["channels"]] != b["channels"]:
        bad.append("channels/base.theta")
    if {ren(k): v for k, v in a["floor"].items()} != b["floor"] or a["scaled"] != b["scaled"]:
        bad.append("floor/scaled")
    ra, rb = {ren(k): v for k, v in a["rank_only"].items()}, b["rank_only"]
    if set(ra) != set(rb) or any(ra[k] != rb[k] for k in ra if k != M3X4) or ra[M3X4]["floor"] != rb[M3X4]["floor"] or ra[M3X4]["rows"] != rb[M3X4]["rows"]:
        bad.append("rank_only")
    for k in ("map3", "logloss", "per_fold", "map3_pre_last_step", "map3_one_set"):
        if a["cross"][k] != b["cross"][k]:
            bad.append(f"cross.{k}")
    if a["insample"] != b["insample"]:
        bad.append("insample")
    for k in ("logloss_pre_last_step", "chosen", "grid"):
        if a["arith_beta_rule"][k] != b["arith_beta_rule"][k]:
            bad.append(f"arith_beta_rule.{k}")
    for k in ("rows", "folds", "fit_rows", "fit_rows_sha256", "none", "probes", "last_step"):
        if a[k] != b[k]:
            bad.append(k)
    return bad


def cmd_stack(save: bool) -> int:
    f = ART / f"oof_{CH}.npy"
    if not f.is_file():
        raise Missing(f"{rel(f)} missing — run oof --save first")
    stack, rep = build_stack5(CH, np.load(f))
    sha = write_or_verify(ART / f"stack_{STEM}.json", to_json(stack), save)
    ref = json.loads(STACK_M3X4.read_text())
    say(f"stack_{STEM}.json: channels {stack['channels']}; β {rep['beta']:g} (primary_m3x4 {ref['arith_beta']:g}); θ "
        + " · ".join(f"{k} {v:.4f}" for k, v in rep["theta"].items()))
    say(f"   cross-fit {rep['cross']['map3']:.5f} ({rep['cross']['units']}/9000; primary_m3x4 {float(ref['cv']):.5f}); without the last step "
        f"{rep['cross']['map3_pre_last_step']:.5f}, one set {rep['cross']['map3_one_set']:.5f}; oof_mixture cross-check "
        f"{rep['crosscheck_oof_mixture']['max_abs_mask_first']:.1e}/{rep['crosscheck_oof_mixture']['max_abs_test_way']:.1e}; "
        f"{'stored' if save else 'CHECKED'} (SHA-256 {sha})")
    return 0


# ------------------------------------------------------------------------------------------------ chain of the second stage (as primary_m3x4)

def mm_rows_check(rows: dict, ctx_m3px27: dict, P: np.ndarray, kind: str) -> dict:
    """DEPLOYMENT: MM rows F and M of the chain = bank variant (c135.Bank.variant with the channel P, features d1–d3; the frame of the
    evaluation) bit-identical, sixths = evaluation (arm add5_sm or — copy — base; pinned)."""
    B = c135()
    P4, _ = B.load_d4_matrix(ctx_m3px27)
    bank = B.Bank(ctx_m3px27, P4, None)
    u, brows, betas = B.run_units(bank.variant(B.TRIPLES[0], P, np.asarray(ctx_m3px27["folds"])))
    if not (np.array_equal(rows["mm_fold_forced"], brows["fold"]) and np.array_equal(rows["mm_set_forced"], brows["set"])):
        raise TechnicalError("MM rows of the chain != bank variant with the channel — not bit-identical")
    arm = ARM if kind == "deploy" else "base"
    want = UNITS[arm]["line"]
    if list(u) != want:
        raise TechnicalError(f"MM sixths of the chain {list(u)} != evaluation {arm} {want} (pinned)")
    say(f"   MM rows of the chain = bank variant bit-identical; sixths {list(u)} = evaluation arm {arm} (pinned)")
    return {"label": f"bank variant (c135.Bank.variant, triple d1d2d3, channel {CH if kind == 'deploy' else CH_COPY}) = evaluation arm {arm}",
            "bit_identical": True, "units": list(u), "betas": betas}


def base_field(kind: str, name: str, check: dict, res: dict, list_info: dict) -> dict:
    return {"decision": DECISION, "kind": kind, "candidate": "primary_m3x5h" if kind == "deploy" else "copy null point (empty list)",
            "channel": {"slot": M3, "by": name, "m3x4_channel": M3X4}, "definition": DEFINITION5, "draws_first": len(S2().DRAWS),
            "row_list": list_info, "test_side": f"make_submission --rank-only {name}=<d1>,<d2>,<d3>,<d4>,<h shards>@<test list> --stage2 (draws_first: "
                                                "features from the first three; draw with row list: rank.row_list_draws)",
            "oof_rows": check, "beta_outer": res["beta_outer"], "null_point_i": res["null_point_i"],
            "rule": "pre-registered rule (passed): best role over 20 untouched meta-splits > 0 in M and ½(F+M) and above the copy null point",
            "copy_null_point": "eda/c166_primary_m3x5h.py copy: the same path with an empty list hits primary_m3x4/primary_cols (blend, chain, qfp; test "
                               "side byte-identical)"}


def run_chain5(name: str, P: np.ndarray, stem: str, tag: str, kind: str, list_info: dict, log=say) -> tuple[list, dict, dict, Path, dict, dict]:
    B, X = c135(), c133()
    ctx_m3px27 = B.m3px27_context()
    ctx = B.swap_ctx(ctx_m3px27, name, P, stem)
    res, rows, std = X.oof_chain(ctx, log=log)
    check = mm_rows_check(rows, ctx_m3px27, P, kind)
    res["reproduction"] = {"kind": "MM rows = bank variant bit-identical, sixths = internal evaluation", **check}
    m3px27_src = json.loads(B.STAGE2_M3PX27.read_text()).get("m3px27_build", {}).get("rank_artifact")
    s2b = X.m3px27_build_field(res, None, m3px27_src, "deploy")
    out_dir = PRIV / ("chain" if kind == "deploy" else "copy")
    oof_file = PRIV / f"c166_oof_{tag}.json"
    with B._DrawsChannelAs(name):
        steps = X.compose(ctx, res, std, oof_file, out_dir, tag, s2b)
    steps[-1][2]["m3x4_build"] = base_field(kind, name, check, res, list_info)
    return steps, res, rows, oof_file, ctx, ctx_m3px27


def cmd_fit_base(save: bool) -> int:
    t0 = time.time()
    if sha256_file(c135().STAGE2_M3PX27) != c135().STAGE2_M3PX27_SHA:
        raise TechnicalError("stage2_primary_m3px27.json is not the deployment file of primary_m3px27")
    f = ART / f"oof_{CH}.npy"
    if not f.is_file():
        raise Missing(f"{rel(f)} missing — run oof --save first")
    li = (json.loads((ART / f"meta_{CH}.json").read_text())["rank"]["row_list_draws"][0])["train"]
    steps, res, rows, oof_file, ctx, _ = run_chain5(CH, np.load(f), STEM, TAG, "deploy", li)
    texts = [(s, (STAGE2_BASE if p is None else p), to_json(d)) for s, p, d in steps]
    write_or_verify(oof_file, to_json(res), save)
    if save:
        np.savez_compressed(PRIV / f"c166_oof_{TAG}_rows.npz", **rows)
    shas = {s: write_or_verify(p, t, save) for s, p, t in texts}
    e = steps[-1][2]
    m, fb, df = e["models"]["main"], e["models"]["fallback"], e["m3px27_build"]["deploy_forms"]
    say(f"stage2_{STEM}_base.json: main_guard_i1_mm ({len(m['features'])} features, guard on; gate {m['gate']['inner']:.5f} against "
        f"{m['gate']['inner_base']:.5f} -> {'ABSTAIN' if m['abstain'] else 'apply'}; OOF new top code {m['oof_changed_top_share']:.4f}; in-sample "
        f"{m['insample']['map_base']:.5f} → {m['insample']['map_reranked']:.5f}); fallback {len(fb['features'])} features; β {e['arith_beta']:g}; "
        f"draws_channel {e['draws_channel']} (draws_first {e['draws_first']}); β per fold {df['beta_per_fold']}")
    say(f"   chain a–e {'stored' if save else 'CHECKED'}: " + "; ".join(f"{s} {sha[:16]}…" for s, sha in shas.items()) + f" ({time.time() - t0:.0f} s)")
    return 0


def check_base_file(e0: dict, name: str, stem: str) -> None:
    """Fields of the base deployment file as stage2_primary_m3x4.json (order, channel name, draws_first, blend file)."""
    a = json.loads(STAGE2_M3X4.read_text())
    if list(e0) != list(a):
        raise TechnicalError(f"fields of the base {list(e0)[-4:]} != those of stage2_primary_m3x4.json {list(a)[-4:]}")
    st = ART / f"stack_{stem}.json"
    if e0["stack"] != rel(st) or e0["stack_sha256"] != sha256_file(st):
        raise TechnicalError(f"base: blend file {e0['stack']} != {rel(st)} (checksum)")
    if e0["draws_channel"] != name or e0["draws_first"] != 3 or name not in e0["channels"]:
        raise TechnicalError(f"base: draws_channel {e0['draws_channel']} / draws_first {e0['draws_first']} / channels {e0['channels']}")


# ------------------------------------------------------------------------------------------------ columns qfp (as primary_cols)

def chain_qfp(ctx: dict, e0: dict, kind: str, zero: bool = False, log=say) -> dict:
    """As c151.chain(base, "qfp") with the context `ctx` (channel m3x5h or the copy) and the base deployment file e0 (dict). zero=True:
    columns := 0."""
    t0 = time.time()
    X1 = c151()
    C, S, M121, M124 = c118(), S2(), c121(), c124()
    sources = list(XS.VARIANTS[COLS])
    inp, info, codes = X1.train_inputs(ctx, sources)
    cols_real = XS.build_columns(sources, inp)
    cols = cols_real.zero() if zero else cols_real
    std, betas, nested = M121.prepare(ctx)
    forms, betas2 = C.standard_forms(ctx, "select")
    if [float(v) for v in betas] != [float(v) for v in betas2] or not all(np.array_equal(forms[f], std[f]) for f in FORMS3):
        raise TechnicalError("standard cross-fit recomputed != the one of the nested reproduction")
    df = (e0.get("m3px27_build") or {}).get("deploy_forms") or {}
    if [float(v) for v in betas] != [float(v) for v in df.get("beta_per_fold") or []]:
        raise TechnicalError(f"β per fold {betas} != base {df.get('beta_per_fold')}")
    X_mm, cand, names = C.deploy_features(ctx, forms, S.MM_SET)  # (1) regression in the process: main model MM of the base
    m0 = e0["models"]["main"]
    if names != m0["features"] or X1.sha256_array(X_mm) != m0["features_sha256"]:
        raise TechnicalError("deployment features MM != models.main of the base (names or checksum)")
    mod_mm, g_mm, ins_mm = X1.fit_main(ctx, X_mm, cand)
    reg = {"w": [float(v) for v in mod_mm["w"]] == m0["w"], "mu": [float(v) for v in mod_mm["mu"]] == m0["mu"],
           "sd": [float(v) for v in mod_mm["sd"]] == m0["sd"], "gate": g_mm == m0["gate"], "insample": ins_mm == m0["insample"]}
    if not all(reg.values()):
        raise TechnicalError(f"regression: main model MM in the process != models.main of the base ({[k for k, v in reg.items() if not v]})")
    y, seen = ctx["y"], ctx["seen"]
    draws_q = C.C107.question_draws(ctx["q_code"])
    rec = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS3}
    r_mm, r_x, oof = {}, {}, {}
    for f in FORMS3:  # (2) nested: MM and MM + columns
        r_mm[f] = M124.run_mm(ctx, std, nested, f, "real")
        r_x[f] = X1.run_x(ctx, std, nested, f, cols)
        oof[f] = X1.entry(r_x[f], r_mm[f], rec[f], ctx, draws_q)
        log(f"   [{f}] MM {units_of(r_mm[f]['forced'])}/9000, MM + columns {oof[f]['units_forced']}/9000 (Δ {oof[f]['vs_mm']['units']:+d} sixths, "
            f"other top code {oof[f]['rows_other_top1_vs_mm']}) ({time.time() - t0:.0f} s)")
    u_mm = [units_of(r_mm["fold"]["forced"]), units_of(r_mm["set"]["forced"])]
    u_x = [oof["fold"]["units_forced"], oof["set"]["units_forced"]]
    arm = ARM if kind == "deploy" else "base"
    want_mm = ((e0.get("m3x4_build") or {}).get("oof_rows") or {}).get("units")
    if u_mm != UNITS[arm]["line"] or (want_mm is not None and list(want_mm) != u_mm):
        raise TechnicalError(f"MM sixths {u_mm} != evaluation {arm} {UNITS[arm]['line']} / base {want_mm}")
    if not zero and u_x != UNITS[arm]["with"]:
        raise TechnicalError(f"sixths with columns {u_x} != evaluation {arm} {UNITS[arm]['with']} (pinned)")
    null_i = {f: X1.same_rows(r_x[f], r_mm[f]) for f in FORMS3} if zero else None
    if zero and not all(null_i.values()):
        raise TechnicalError(f"copy null point (i) violated: columns := 0 does not hit MM bit-identically ({null_i})")
    E_dep = X1.deploy_extra(ctx, cand, cols)  # (3) deployment form, null point (e)
    Xf = np.concatenate([X_mm, E_dep], 1)
    if X1.sha256_array(Xf) != X1.sha256_array(C.eval_rows_in_table_order(ctx, r_x["set"]["X_eval"])):
        raise TechnicalError("null point (e) violated: deployment features (MM + columns) != evaluated OOF features of form M")
    model, g, ins = X1.fit_main(ctx, Xf, cand)  # (4) main model; reference of the test guard of the columns (real columns, test form of the count)
    main = X1.main_entry(names, cols.features, model, g, ins, oof, Xf, X_mm)
    E_guard = X1.deploy_extra(ctx, cand, cols_real, guard=True)
    guard_ref = XS.guard_stats(cols_real, E_guard, cand, inp, np.arange(ctx["n"]), np.arange(ctx["n"]))
    mats = XS.matrices_sha256(cols_real, ctx["n"], ctx["K"])
    return {"sources": sources, "ctx": ctx, "e0": e0, "betas": [float(v) for v in betas], "main": main, "gate": g, "insample": ins, "oof": oof,
            "r_mm": r_mm, "r_x": r_x, "null_i": null_i, "u_mm": u_mm, "u_x": u_x, "guard_ref": guard_ref, "mats_sha": mats, "info": info,
            "minor_codes": codes, "cols_obj": cols_real, "reg": reg, "seconds": round(time.time() - t0, 1)}


def qfp_field(res: dict, kind: str, base_file: Path) -> dict:
    o = res["oof"]
    return {"decision": DECISION, "kind": kind, "candidate": "primary_m3x5h" if kind == "deploy" else "copy null point (empty list)", "variant": COLS,
            "base": {"label": "primary_m3x5h (base: recipe of primary_m3x4 with m3x5h)", "file": rel(base_file), "sha256": sha256_file(base_file),
                     "stack": res["e0"]["stack"], "stack_sha256": res["e0"]["stack_sha256"]},
            "feature_set": "main_guard_i1_mm (64, unchanged) + extra columns extra_columns.features at the end (src/stage2_sources.py), NONE guard",
            "regression_base_main": {"equal": True, "fields": sorted(res["reg"])},
            "oof_units": {f: {"mm": units_of(res["r_mm"][f]["forced"]), "with_columns": o[f]["units_forced"]} for f in FORMS3},
            "oof_vs_mm": {f: {k: o[f]["vs_mm"][k] for k in ("delta", "se_q", "z", "units", "fold_units", "folds_pos")} for f in FORMS3},
            "measurement": {"arm": ARM if kind == "deploy" else "base", "units_line": res["u_mm"], "units_with": res["u_x"],
                            "source": "internal evaluation of the large-photo draw (original split), pinned in eda/c166_primary_m3x5h.py (UNITS)"},
            "null_point_e": True, "null_point_i": None if res["null_i"] is None else {"zero_equals_mm": res["null_i"]}, "beta_outer": res["betas"],
            "test_side": "make_submission --stage2 <deployment file>: columns of the test rows from extra_columns (checksums required), test guard per "
                         "column and source (extra_columns.guard_rule), abort without a file if violated; M3 channel with the draw h only on the test list",
            "rule": "five-draw rule: add5_sm over 20 untouched meta-splits (11000–11019) ½(F+M) +0.00083 ± 0.00017, M +0.00108 ± 0.00030 against "
                    "primary_cols"}


def compose_qfp(res: dict, kind: str, base_file: Path) -> dict:
    X1 = c151()
    out = {}
    for k, v in res["e0"].items():
        out[k] = {"main": res["main"], "fallback": v["fallback"]} if k == "models" else v
    out["extra_columns"] = X1.extra_field(res["ctx"], res["cols_obj"], res["info"], res["minor_codes"], res["guard_ref"], res["mats_sha"])
    out["m3x5h_build"] = qfp_field(res, kind, base_file)
    return out


def ctx_of(name: str, stem: str) -> dict:
    """Training context of primary_m3px27 with `name` (oof_<name>.npy) in place of m3x3_full and stack_<stem>.json (c135.swap_ctx)."""
    f = ART / f"oof_{name}.npy"
    if not f.is_file():
        raise Missing(f"{rel(f)} missing")
    return c135().swap_ctx(c151().m3px27(), name, np.load(f), stem)


def cmd_fit(save: bool) -> int:
    if not STAGE2_BASE.is_file():
        raise Missing(f"{rel(STAGE2_BASE)} missing — run fit-base --save first")
    e0 = json.loads(STAGE2_BASE.read_text())
    check_base_file(e0, CH, STEM)
    ctx = ctx_of(CH, STEM)
    res = chain_qfp(ctx, e0, "deploy")
    out = compose_qfp(res, "deploy", STAGE2_BASE)
    sha = write_or_verify(STAGE2, to_json(out), save)
    if save:
        PRIV.mkdir(parents=True, exist_ok=True)
        rows = {f"{nm}_{f}_{k}": d[f][k] for nm, d in (("mm", res["r_mm"]), ("x", res["r_x"])) for f in FORMS3 for k in c151().ROW_KEYS}
        np.savez_compressed(PRIV / f"c166_rows_{TAG}{COLS}.npz", **rows)
        (PRIV / f"c166_fit_{TAG}{COLS}.json").write_text(to_json({"out": rel(STAGE2), "out_sha256": sha, "oof": res["oof"], "u_mm": res["u_mm"],
                                                                  "u_x": res["u_x"], "betas": res["betas"], "gate": res["gate"], "insample": res["insample"],
                                                                  "regression": res["reg"]}))
    m, o = out["models"]["main"], res["oof"]
    say(f"stage2_{STEM}.json: main_guard_i1_mm + {len(m['extra_columns'])} columns ({len(m['features'])} features, guard on; gate {m['gate']['inner']:.5f} "
        f"against {m['gate']['inner_base']:.5f} -> {'ABSTAIN' if m['abstain'] else 'apply'}; OOF new top code {m['oof_changed_top_share']:.4f}; in-sample "
        f"{m['insample']['map_base']:.5f} → {m['insample']['map_reranked']:.5f}); fallback unchanged")
    say(f"   nested: MM {res['u_mm']}, with columns {res['u_x']} = evaluation arm {ARM} (½(F+M) against primary_cols "
        f"{float(Fraction(sum(res['u_x']) - sum(UNITS['base']['with']), 2 * UNITS_ROWS)):+.5f}); Δ against MM F {o['fold']['vs_mm']['units']:+d}, oL "
        f"{o['pre']['vs_mm']['units']:+d}, M {o['set']['vs_mm']['units']:+d}")
    say(f"   {'stored' if save else 'CHECKED'}: {rel(STAGE2)} (SHA-256 {sha}) ({res['seconds']:.0f} s)")
    return 0


# ------------------------------------------------------------------------------------------------ copy null point "empty list"

COPY_FILES = (f"oof_{CH_COPY}.npy", f"test_{CH_COPY}.npy", f"meta_{CH_COPY}.json", f"stack_{STEM_COPY}.json")
COPY_DIR = PRIV / "copy"
COPY_STAGE2 = COPY_DIR / f"stage2_{TAG_COPY}_cols.json"
COPY_LISTS = {"train": COPY_DIR / "train_empty_rows.npy", "test": COPY_DIR / "test_empty_rows.npy"}
PROV_TOP = ("stack", "stack_sha256", "inputs_sha256", "oof_reproduction", "derived_from", "m3px27_build", "m3x4_build", "cols_build", "m3x5h_build",
            "draws_first")
PROV_BUILD = ("oof", "splits", "rule_oof", "fallback")


def clean_copy() -> list[str]:
    gone = []
    for f in COPY_FILES:
        p = ART / f
        if p.is_file():
            p.unlink()
            gone.append(f)
    return gone


def normalized5(d: dict, name: str) -> dict:
    """Comparison form: channel name -> m3x3_full (channels, features, draws_channel, extra_columns.channels); provenance fields removed."""
    out = copy.deepcopy({k: v for k, v in d.items() if k not in PROV_TOP})
    ren = lambda s: s.replace(f"{name}_", f"{M3}_", 1) if isinstance(s, str) and s.startswith(f"{name}_") else (M3 if s == name else s)  # noqa: E731
    for key in ("channels", "draws_channel"):
        if key in out:
            out[key] = [ren(c) for c in out[key]] if isinstance(out[key], list) else ren(out[key])
    for m in (out.get("models") or {}).values():
        m["features"] = [ren(f) for f in m["features"]]
    if "extra_columns" in out:
        out["extra_columns"]["channels"] = [ren(c) for c in out["extra_columns"]["channels"]]
    for b in ("guard_build", "i1_build", "mm_build"):
        if b in out:
            out[b] = {k: v for k, v in out[b].items() if k not in PROV_BUILD}
    return out


def cmd_copy(keep: bool) -> int:
    t0 = time.time()
    X = c133()
    lc = light()
    clean_copy()
    COPY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for side, f in COPY_LISTS.items():
            np.save(f, np.zeros(0, dtype=np.int64))
        ov = {s: (str(COPY_LISTS[s]), sha256_file(COPY_LISTS[s]), 0) for s in COPY_LISTS}
        rows, li = load_list(lc["root"], "train", ov["train"])
        _, ti = load_list(lc["root"], "test", ov["test"])
        P, meta, parts = build_channel(lc, rows, li, ti, name=CH_COPY)
        if not np.array_equal(P, np.load(ART / f"oof_{M3X4}.npy")):
            raise TechnicalError("copy channel (empty list) != oof_m3x4_full bit-identical")
        X.save_or_verify(CH_COPY, P, meta, lc["K"], True)
        stack, _ = build_stack5(CH_COPY, P)
        bad = stack_numbers_equal_m3x4(stack, json.loads(STACK_M3X4.read_text()), CH_COPY)
        if bad:
            raise TechnicalError(f"copy null point of the blend violated: {bad}")
        (ART / f"stack_{STEM_COPY}.json").write_text(to_json(stack))
        say(f"copy null point (blend): stack_{STEM_COPY}.json hits θ, base θ, floors, β, cv, cross-fit, in-sample and β-rule numbers, rows and "
            f"rank_only (except the M3 entry) of stack_primary_m3x4.json exactly")
        steps, res, rows_c, oof_file, ctx, _ = run_chain5(CH_COPY, P, STEM_COPY, TAG_COPY, "copy", li)
        oof_file.write_text(to_json(res))
        refs = {s: CHAIN_M3X4 / f"stage2_m3x4_{s}.json" for s in "abcd"}
        refs["e"] = STAGE2_M3X4
        comp, e_copy = {}, None
        for s, p, d in steps:
            if not refs[s].is_file():
                raise Missing(f"{rel(refs[s])} missing — run bash tools/builds/primary_m3x4.sh first")
            dk = X.diff_keys(normalized5(d, CH_COPY), normalized5(json.loads(refs[s].read_text()), M3X4))
            comp[s] = {"reference": rel(refs[s]), "equal": not dk, "diff": dk}
            path = (COPY_DIR / f"stage2_{TAG_COPY}_e.json") if p is None else p
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(to_json(d))
            if s == "e":
                e_copy, e_path = d, path
        bad = {s: v["diff"] for s, v in comp.items() if not v["equal"]}
        if bad:
            raise TechnicalError(f"copy null point of the chain violated: {bad}")
        say(f"copy null point (chain): MM rows = bank R4_0 (= primary_m3x4) bit-identical; files a–e = those of primary_m3x4 (except provenance, "
            f"channel names) ({time.time() - t0:.0f} s)")
        resq = chain_qfp(ctx, e_copy, "copy")
        outq = compose_qfp(resq, "copy", e_path)
        dq = X.diff_keys(normalized5(outq, CH_COPY), normalized5(json.loads(STAGE2_COLS.read_text()), M3X4))
        COPY_STAGE2.write_text(to_json(outq))
        rep = {"kind": "primary_m3x5h: copy null point \"empty list\" of the training side", "stack": f"stack_{STEM_COPY}.json", "chain": comp,
               "qfp_diff": dq, "qfp_units": {"mm": resq["u_mm"], "with": resq["u_x"]}, "copy_stage2": rel(COPY_STAGE2), "seconds": round(time.time() - t0, 1)}
        (PRIV / "c166_copy.json").write_text(to_json(rep))
        if dq:
            raise TechnicalError(f"copy null point of the columns qfp violated: fields {dq[:8]} != stage2_primary_cols.json")
        say(f"copy null point (qfp): sixths MM {resq['u_mm']}, with columns {resq['u_x']} = primary_cols; deployment file = stage2_primary_cols.json in "
            f"all fields except provenance and channel names -> {rel(COPY_STAGE2)} ({time.time() - t0:.0f} s)")
    except Exception:
        if not keep:
            clean_copy()
        raise
    if not keep:
        say(f"   temporary copy files removed: {', '.join(clean_copy())}")
    else:
        say(f"   copy files for the copy run of the test side lie under artifacts/: {', '.join(COPY_FILES)} (then clean-copy)")
    return 0


# ------------------------------------------------------------------------------------------------ comparison over meta-splits

def parse_seeds(text: str) -> list:
    return c151().parse_seeds(text)


def cmd_measure(seeds: list) -> int:
    """Per meta-split (original, c121.split_folds(y, seed)): primary_cols (channel m3x4_full) and primary_m3x5h (channel m3x5h), MM (c124.run_mm
    "real") and MM + columns qfp (c151.run_x without gate) in F and M — a self-contained recomputation of the comparison of the internal
    evaluation (arms base and add5_sm). Mean over the splits ≠ orig: ½(F+M) and M primary_m3x5h − primary_cols. Training rows only."""
    t0 = time.time()
    X1, M121, M124 = c151(), c121(), c124()
    ctxs = {"base": X1.base_context("m3x4"), ARM: ctx_of(CH, STEM)}
    sources = list(XS.VARIANTS[COLS])
    inp, _, _ = X1.train_inputs(ctxs["base"], sources)
    cols = XS.build_columns(sources, inp)
    recs = {}
    for seed in seeds:
        lab = "orig" if seed is None else str(int(seed))
        folds = X1.split_folds(ctxs["base"]["y"], seed)
        u = {}
        for arm, c0 in ctxs.items():
            c = dict(c0)
            c["folds"] = folds
            std, _b, nested = M121.prepare(c)
            u[arm] = {"line_F": units_of(M124.run_mm(c, std, nested, "fold", "real")["forced"]),
                      "line_M": units_of(M124.run_mm(c, std, nested, "set", "real")["forced"]),
                      "F": units_of(X1.run_x(c, std, nested, "fold", cols, gate=False)["forced"]),
                      "M": units_of(X1.run_x(c, std, nested, "set", cols, gate=False)["forced"])}
        recs[lab] = {"units": u}
        d = {k: u[ARM][k] - u["base"][k] for k in ("F", "M")}
        say(f"[{lab}] base {u['base']} | {ARM} {u[ARM]} | Δ F {d['F']:+d}, M {d['M']:+d} sixths ({time.time() - t0:.0f} s)")
    sp = [v["units"] for k, v in recs.items() if k != "orig"]
    summ = None
    if sp:
        h = np.array([(v[ARM]["F"] - v["base"]["F"] + v[ARM]["M"] - v["base"]["M"]) / (2 * UNITS_ROWS) for v in sp])
        M = np.array([(v[ARM]["M"] - v["base"]["M"]) / UNITS_ROWS for v in sp])
        F = np.array([(v[ARM]["F"] - v["base"]["F"]) / UNITS_ROWS for v in sp])
        se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None  # noqa: E731
        summ = {"n_splits": len(sp), "half_mean": float(h.mean()), "half_se": se(h), "share_pos": float((h > 0).mean()), "M_mean": float(M.mean()),
                "M_se": se(M), "M_share_pos": float((M > 0).mean()), "F_mean": float(F.mean())}
        say(f"   {len(sp)} meta-splits: ½(F+M) {summ['half_mean']:+.6f} ± {summ['half_se']:.6f} (> 0 in {summ['share_pos']:.0%}), M {summ['M_mean']:+.6f} ± "
            f"{summ['M_se']:.6f} (> 0 in {summ['M_share_pos']:.0%}), F {summ['F_mean']:+.6f}")
    PRIV.mkdir(parents=True, exist_ok=True)
    full = {str(s) for s in range(11000, 11020)} <= set(recs)  # the evidence over all 20 untouched meta-splits stays a file of its own
    (PRIV / ("c166_measure.json" if full else "c166_measure_short.json")).write_text(to_json({"seeds": recs, "summary": summ}))
    return 0


# ------------------------------------------------------------------------------------------------ null model

def cmd_null(source: Path, out: Path, stem: str) -> int:
    ref = json.loads(Path(source).read_text())
    res = c118().null_stage2(stem, Path(source))
    res["mech27"] = ref["mech27"]
    if "draws_first" in ref:
        res = {k2: v2 for k, v in res.items() for k2, v2 in ([(k, v), ("draws_first", ref["draws_first"])] if k == "draws_channel" else [(k, v)])}
    if "extra_columns" in ref:
        res["extra_columns"] = ref["extra_columns"]
    Path(out).write_text(to_json(res))
    say(f"null model for {rel(Path(source))} -> {out}")
    return 0


# ------------------------------------------------------------------------------------------------ large-photo draw guard of the test draw h

def test_side_h(spec: str, list_rows: np.ndarray, n_test: int, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Test draw h from its shards (each complete and exact, without overlap; a8.load_rank table test) -> (P, coverage, findings, statistics)."""
    M = ms()
    A8 = M.a8
    problems, info = [], {"shards": []}
    P, cov = np.zeros((n_test, len(seen))), np.zeros(n_test, bool)
    for shard in spec.split("+"):
        missing = M.c95.rank_files_missing(shard)
        if missing:
            raise Missing(f"test ranking {shard}: missing ({', '.join(missing)})")
        R, h, st = A8.load_rank(shard, n_test, seen, table="test", allow_partial=True)
        hr = h & M.c95.exact_rank_rows(shard, n_test, seen)
        complete = bool(st["rows_probe"] == st["rows_done"] == st["rows_exact"] == int(hr.sum()))
        if not complete:
            problems.append(f"shard {shard}: probe {st['rows_probe']}, computed {st['rows_done']}, exact {st['rows_exact']} — not completely exact")
        if (cov & hr).any():
            problems.append(f"shard {shard}: {int((cov & hr).sum())} rows already covered by another shard")
        P[hr] = R[hr]
        cov |= hr
        info["shards"].append({"spec": shard, "rows_probe": st["rows_probe"], "rows_done": st["rows_done"], "rows_exact": st["rows_exact"],
                               "complete": complete, "mass_median": st["mass_median"], "mass_p05": st["mass_p05"], "idx_sha256": st["idx_sha256"]})
    want = mask_of(list_rows, n_test)
    info["coverage"] = {"rows": int(cov.sum()), "list_rows": int(want.sum()), "outside_list": int((cov & ~want).sum()), "missing_in_list": int((want & ~cov).sum()),
                        "equals_list": bool(np.array_equal(cov, want))}
    if not info["coverage"]["equals_list"]:
        problems.append(f"coverage of the test draw ({int(cov.sum())} rows; outside the list {info['coverage']['outside_list']}, missing in the list "
                        f"{info['coverage']['missing_in_list']}) != test list ({int(want.sum())})")
    return P, cov, problems, info


def test_mean4(n_test: int, seen: np.ndarray) -> np.ndarray:
    """Mean of the four test draws d1–d4 (sum in order / number of draws with an exact ranking; as make_submission)."""
    M = ms()
    S, cnt = np.zeros((n_test, len(seen))), np.zeros(n_test, dtype=np.int64)
    for spec in TEST_RANKS4:
        R, h, _ = M.a8.load_rank(spec, n_test, seen, table="test", allow_partial=True)
        hr = h & M.c95.exact_rank_rows(spec, n_test, seen)
        S[hr] += R[hr]
        cnt[hr] += 1
    P = np.tile(seen / seen.sum(), (n_test, 1)).astype(float)
    have = cnt > 0
    P[have] = S[have] / cnt[have, None]
    return P


def training_regime_h(spec: str, n_train: int, folds: np.ndarray) -> tuple[list[str], dict]:
    """Test model of every shard = merge of the full fit (as d1–d4); recipe equal to the fold adapters of the OOF draw h (as c135.test_guard)."""
    X = c133()
    A9_, C108 = A9(), X.c132().c115().C108
    problems: list[str] = []
    recs = C108.fold_check(H_SPEC, folds, {})
    problems += [f"OOF h {r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"] + r["unknown"]]
    fold_recipes = {}
    for r in recs:
        if r.get("train"):
            r2, j2 = str(r["train"]).split("/", 1)
            fold_recipes[r["train"]] = A9_.read_json(A9_.run_dir(r2) / j2 / "sft_recipe.json") or {}
    if len(fold_recipes) != 5:
        problems.append(f"recipes of the fold adapters of h: {len(fold_recipes)} instead of 5")
    out = {"fold_adapters": [{k: r.get(k) for k in ("job", "train", "train_fold", "gold_only")} for r in recs], "test_models": []}
    for shard in spec.split("+"):
        run, job, tag = A9_.parse_source(shard)
        t_meta = A9_.read_json(A9_.run_dir(run) / job / f"{tag}_meta.json") or {}
        if t_meta.get("table") != "test":
            problems.append(f"{shard}: table {t_meta.get('table')!r}")
        rec, where, p = X.adapter_recipe(run, str((t_meta.get("recipe") or {}).get("model")))
        problems += [f"test {run}/{job}: {x}" for x in p]
        if where.get("train") != "/".join(FULL_TRAIN):
            problems.append(f"{shard}: test model from {where.get('train')} instead of {'/'.join(FULL_TRAIN)} (full fit)")
        if rec is not None:
            if rec.get("fold") is not None:
                problems.append(f"{shard}: test model holds out fold {rec.get('fold')} — not a full fit")
            if rec.get("teacher_derived") or rec.get("pool"):
                problems.append(f"{shard}: test model with teacher targets (not gold-only)")
            if int(rec.get("rows_used") or -1) != int(n_train):
                problems.append(f"{shard}: test model learned {rec.get('rows_used')} instead of {n_train} training rows")
            if rec.get("steps_done") != rec.get("steps_total"):
                problems.append(f"{shard}: training of the test model not finished")
            for nm, fr in fold_recipes.items():
                dd = {k: (rec.get(k), fr.get(k)) for k in X.RECIPE_SAME if rec.get(k) != fr.get(k)}
                if dd:
                    problems.append(f"{shard}: test model fitted differently from the fold adapter {nm}: {dd}")
        out["test_models"].append({"shard": shard, "train": where.get("train"), "adapter_sha256": where.get("adapter_sha256"),
                                   "fold": (rec or {}).get("fold"), "rows_used": (rec or {}).get("rows_used")})
    return problems, out


def h_guard(spec: str, control: str | None = None, meta_name: str = CH) -> tuple[int, list[str], dict]:
    """Large-photo draw guard (fixed in advance). -> (exit code 0 / 3 / 5, lines, statistics; aggregates only)."""
    from vlm import common as VC

    lc = light()
    root, seen, n_train = lc["root"], lc["seen"], lc["n"]
    n_test = len(VC.load_table(root, "test"))
    meta = json.loads((ART / f"meta_{meta_name}.json").read_text())
    rl = (meta.get("rank") or {}).get("row_list_draws") or []
    if len(rl) != 1 or rl[0]["source"] != H_SPEC:
        raise TechnicalError(f"meta_{meta_name}.json: rank.row_list_draws does not name the draw h")
    g = rl[0]["guard"]
    if g.get("rule") != RZH["rule"] or g.get("rel") != RZH["rel"] or g.get("z_max") != RZH["z_max"] or g.get("floors") != RZH["floors"]:
        raise TechnicalError("rule of the test guard in the meta file != the rule of the large-photo draw guard fixed in advance")
    t = rl[0]["test"]
    rows_t, _ = load_list(root, "test", (t["file"], t["sha256"], int(t["rows"])))
    M = ms()
    for shard in spec.split("+"):
        if M.c95.rank_files_missing(shard):
            return 3, [f"test ranking {shard}: MISSING ({', '.join(M.c95.rank_files_missing(shard))})"], {}
    P, cov, problems, info = test_side_h(spec, rows_t, n_test, seen)
    if not all(s["complete"] for s in info["shards"]):  # run not finished or not completely fetched: no verdict (as c135.test_guard)
        return 3, [f"test draw h {spec}: INCOMPLETE (" + ", ".join(f"{s['rows_exact']}/{s['rows_probe']} exact" for s in info["shards"])
                   + ") — run not finished or not completely fetched; no verdict"], info
    diff = {}
    for shard in spec.split("+"):
        diff.update({f"{k} <{shard}>": v for k, v in M.rank_only_draw_differences(shard, H_SPEC).items()})
    if diff:
        problems.append(f"settings of the test draw differ from those of the OOF draw h: {diff}")
    p, binfo = training_regime_h(spec, n_train, lc["folds"])
    problems += p
    notes = notes_of_spec(spec, "test")
    m4 = test_mean4(n_test, seen)
    if control is not None:
        P, notes = apply_control(control, P, rows_t, notes, seen)
    general = np.isin(np.array(lc["classes"]), S2().GENERAL_CODES)
    st = rzh_stats(P, m4, rows_t, notes, general)
    chk = rzh_check(g["reference"], st)
    info.update(settings_equal=not diff, training_regime=binfo, test=st, reference=g["reference"], guard=chk, control=control)
    lines = [f"test draw h {spec}: coverage {info['coverage']['rows']} rows = list {info['coverage']['equals_list']} ({info['coverage']['list_rows']}); shards "
             + ", ".join(f"{s['rows_exact']}/{s['rows_probe']} exact" for s in info["shards"]) + f"; settings against the OOF draw h "
             f"{'equal' if not diff else 'DIFFERENT'}; training regime {len(binfo['test_models'])} test models"]
    for c in chk["checks"]:
        lines.append(f"  {c['stat']:22s} training {c['train']:10.4f}  test {c['test']:10.4f}  z {c['z']:+7.2f}  "
                     + (f"ok ({c['passed_by']})" if c["ok"] else "VIOLATED") + (f"  floor {c['floor']:g}" if c.get("floor") is not None else ""))
    lines += [f"PROBLEM: {x}" for x in problems]
    code = 5 if (problems or not chk["ok"]) else 0
    return code, lines, info


def cmd_h_guard(spec: str, control: str | None, meta_name: str) -> int:
    code, lines, info = h_guard(spec, None if control in (None, "none") else control, meta_name)
    for line in lines:
        say(line)
    PRIV.mkdir(parents=True, exist_ok=True)
    out = PRIV / (f"c166_h_guard{'' if control in (None, 'none') else '_' + control}.json")
    out.write_text(to_json({"code": code, "lines": lines, **info}))
    say(f"-> large-photo draw guard {'PASSED' if code == 0 else ('MISSING' if code == 3 else 'VIOLATED')} (exit {code}); {rel(out)}")
    if code == 5 and control in (None, "none"):
        raise GuardFail("large-photo draw guard of the draw h violated — no file")
    return code


def cmd_h_controls(spec: str, meta_name: str) -> int:
    res, bad = {}, []
    for c in CONTROLS:
        code, _, info = h_guard(spec, c, meta_name)
        if code == 3:
            raise Missing("test draw h missing — controls cannot be computed")
        fired = code == 5 and not info["guard"]["ok"]
        res[c] = {"exit": code, "fired": bool(fired), "failed": [f"{x['stat']} {x['train']:.4f} -> {x['test']:.4f}" for x in info["guard"]["failed"]]}
        say(f"  control {c}: {'fires' if fired else 'DOES NOT FIRE'} — {', '.join(res[c]['failed'][:4]) or 'no violated statistic'}")
        if not fired:
            bad.append(c)
    PRIV.mkdir(parents=True, exist_ok=True)
    (PRIV / "c166_h_controls.json").write_text(to_json(res))
    if bad:
        raise ControlFail(f"positive controls without effect: {bad}")
    say(f"-> positive controls of the large-photo draw guard PASSED: all {len(res)} trigger the test guard")
    return 0


# ------------------------------------------------------------------------------------------------ checksums of the inputs (build script)

def side_files(side: str) -> list[Path]:
    """Files of one side of the draw h (for the pins of the build script): per shard meta, row index, ranking, completeness mask, output of the
    ranking; generation behind it (recipe, meta, notes); row list; test side additionally summary.json of the merge jobs (adapter checksum)."""
    root = light()["root"]
    spec = H_SPEC if side == "train" else H_TEST_SPEC
    files: list[Path] = []
    for shard in spec.split("+"):
        run, job, t = A9().parse_source(shard)
        d = A9().run_dir(run) / job
        files += [d / f"{t}_meta.json", d / f"{t}_idx.npy", d / f"{t}_proto_rank.npy", d / f"{t}_proto_rank.done.npy", d / f"{t}_proto_rank.jsonl"]
        meta = A9().read_json(d / f"{t}_meta.json") or {}
        src = str((meta.get("sft") or {}).get("sft_from") or "")
        if "/" in src:
            gj, gt = src.split("/", 1)
            g = A9().run_dir(run) / gj
            files += [g / f"{gt}_generation.json", g / f"{gt}_meta.json", g / f"{gt}_notes.jsonl"]
        if side == "test":
            files += sorted((A9().run_dir(run)).glob("m3h_merge*/summary.json"))
    files.append(resolve(root, LISTS[side][0]))
    if side == "train":
        files.append(resolve(root, LIST_META[0]))
    return files


def cmd_digest(side: str) -> int:
    """One checksum over path and SHA-256 of all files of one side (sorted); output "<number> <checksum>" or "MISSING <path>"."""
    files = side_files(side)
    lines = []
    for f in sorted(set(files), key=lambda p: rel(p)):
        if not f.is_file():
            print(f"MISSING {rel(f)}")
            return 3
        lines.append(f"{rel(f)}\t{sha256_file(f)}\n")
    print(f"{len(lines)} {hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()}")
    return 0


# ------------------------------------------------------------------------------------------------ entry point

def exit_code_of(exc: BaseException) -> int | None:
    names = {k.__name__ for k in type(exc).__mro__}
    if "GuardFail" in names:
        return 5
    if "ControlFail" in names:
        return 6
    if "SourceError" in names:
        return 4
    return 3 if "Missing" in names else (4 if "TechnicalError" in names else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("oof", "stack", "fit-base", "fit"):
        p = sub.add_parser(nm)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--save", action="store_true", help="writes missing files, requires the same content for existing ones (never overwrites)")
        g.add_argument("--verify", action="store_true", help="check only (requires all files)")
    p = sub.add_parser("copy")
    p.add_argument("--keep", action="store_true")
    sub.add_parser("clean-copy")
    p = sub.add_parser("measure")
    p.add_argument("--seeds", default="orig," + ",".join(str(s) for s in range(11000, 11020)))
    p = sub.add_parser("null")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stack", default=STEM)
    p = sub.add_parser("h-guard")
    p.add_argument("--spec", default=H_TEST_SPEC)
    p.add_argument("--control", choices=("none", *CONTROLS), default="none")
    p.add_argument("--meta", default=CH)
    p = sub.add_parser("h-controls")
    p.add_argument("--spec", default=H_TEST_SPEC)
    p.add_argument("--meta", default=CH)
    p = sub.add_parser("digest")
    p.add_argument("--side", choices=("train", "test"), required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "oof":
            return cmd_oof(a.save)
        if a.cmd == "stack":
            return cmd_stack(a.save)
        if a.cmd == "fit-base":
            return cmd_fit_base(a.save)
        if a.cmd == "fit":
            return cmd_fit(a.save)
        if a.cmd == "copy":
            return cmd_copy(a.keep)
        if a.cmd == "clean-copy":
            say(f"removed: {', '.join(clean_copy()) or 'nothing'}")
            return 0
        if a.cmd == "measure":
            return cmd_measure(parse_seeds(a.seeds))
        if a.cmd == "null":
            return cmd_null(Path(a.source), Path(a.out), a.stack)
        if a.cmd == "h-guard":
            return cmd_h_guard(a.spec, a.control, a.meta)
        if a.cmd == "digest":
            return cmd_digest(a.side)
        return cmd_h_controls(a.spec, a.meta)
    except Exception as exc:  # noqa: BLE001 — exit code per exception type of the loaded modules
        code = exit_code_of(exc)
        if code is None:
            raise
        label = {3: "NOT YET EVALUABLE", 4: "ABORT (technical)", 5: "TEST GUARD VIOLATED", 6: "POSITIVE CONTROL WITHOUT EFFECT"}[code]
        print(f"{label}: {exc}", file=sys.stderr)
        return code


if __name__ == "__main__":
    sys.exit(main())
