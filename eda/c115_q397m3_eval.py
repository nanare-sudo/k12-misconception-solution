""""The 397B finds, M3 names": channel q397m3 — M3 scores all codes behind the notes of Qwen3.5-397B — evaluation according to
a RULE FIXED IN ADVANCE in the six-channel recipe of the primary line.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c115_q397m3_eval.py --notes-check   # BEFORE the run: candidates in the converted notes (training rows)
    python -I eda/c115_q397m3_eval.py --status        # state of the five shards (exit 0 = complete, 3 = not)
    python -I eda/c115_q397m3_eval.py --early         # early-stop check on the finished shards (top-code takeover only)
    python -I eda/c115_q397m3_eval.py                 # evaluation and verdict (exit 3 while a shard is missing)
    python -I eda/c115_q397m3_eval.py --dry           # CODE-PATH CHECK: q397m3 := copy of q397 — NO verdict

CHANNEL q397m3 (training rows): five shards runs/q397m3_oof_rp/q397m3_rank_f{k}/bench (configs/vlm/q397m3_oof_rp.yaml) — per fold
k the exact ranking of the M3 adapter that did not learn fold k, behind the 397B note of the row
(derived/sft/q397m3_train_notes.jsonl, tools/make_q397m3_notes.py); M3 writes the finding itself. Conversion as in a8 --rank / a9
(a8.load_rank: vlm.engine.probs_from_logp, absent = classes without a training example); every shard complete and exact
(completeness mask) and covering EXACTLY the rows of its fold, together all 1,500 rows. If artifacts/oof_q397m3.npy already exists
(deployment path: python -m vlm.sft combine + eda/a8_proto_oof.py --rank), it must be bit-identical.
TECHNICAL CHECKS (abort with exit 4, nothing written): note file = meta checksum = copy in the upload folder, 1,500 training rows,
no finding given; per shard sft_from = derived/sft/q397m3_train_notes.jsonl and ranking settings equal to d3 (m3d3_rank_f*:
a9.rank_settings — beams, finish, image, number format, probe); FOLD INTEGRITY per shard (c108.fold_check: run configuration ->
merge job -> sft_recipe.json holds out exactly this fold, gold-only, local adapter = merged one); ANCHOR of the six-channel recipe
(the primary-line recipe with m3x3_full plus q397 as the sixth channel; c110.line_specs, c107.cross_fit) 7,463/9,000 and log loss
0.8670138 exactly;
MANDATORY CHECK "replacement q397 := copy of q397" reproduces the six-channel recipe bit-identically.

QUANTITIES (1,500 rows; σ over tasks: paired bootstrap over question_id, 10,000 draws, seed 20261030 as in c107):
  single channel  q397m3 against q397 (Tinker, named codes), m3x3_full and d1 (m3p5, one M3 draw): MAP@3, top correct,
               log loss, groups; SAME TOP q397m3/q397 (takeover test; rows with a valid q397 answer), q397m3/m3x3, q397m3/d1,
               reference q397/m3x3 — per 397B state of the note (finished / draft / open) and group.
  ADDITION     six-channel recipe plus q397m3 as a seventh channel (floor 0.05), paired against the six-channel recipe (weights
               per fold from the other four, a2.fit_mix on log loss, reasoning factor 1.4, β 0.2, margin alignment 0.75 per fold
               — c107.cross_fit).
  REPLACEMENT  six-channel recipe with q397m3 instead of q397 (floor 0.05), paired against the six-channel recipe.
  NULL POINT   ADDITION with q397m3 := copy of q397 (a seventh channel without new information) — report only.
  Report only: MEAN (M3 = (3·m3x3 + q397m3)/4 instead of m3x3 — q397m3 as a fourth "draw"; deployment would need an extension
  of a9 / make_submission for notes from a file), without the last step, alignment as one set, weight shares, choice among
  candidates (top correct on error rows whose 397B note names the gold code), ranking of the single models.
RULE (fixed before any measurement; see RULE_TEXT):
  "Continue to the test side only if ADDITION or REPLACEMENT in the six-channel recipe gives Δ ≥ +0.002 with z ≥ 1.0 AND same
  top q397m3/q397 ≤ 0.90 (otherwise mere takeover) AND log loss not worse; otherwise STOP."
READING (fixed before the first run with the channel):
  1. Δ, z and log loss refer to THE SAME form: a form (addition or replacement) passes if Δ ≥ +0.002 (exactly in units of 1/6
     per row: sum ≥ 18 of 9,000), z ≥ 1.0 (σ over tasks) AND its log loss (full recipe, clipping 1e-12) is not above that of
     the six-channel recipe. The same top is a property of the channel and applies to both forms.
  2. Same top = share of rows with a valid q397 answer on which rank 1 of q397m3 = rank 1 of q397 (only classes with a training
     example; ties -> smaller class index as in metric.top3_indices), exactly as a fraction against 9/10.
  3. The copy null point is not part of the rule; if a passing form is not above it, the script says so explicitly.
EARLY STOP (only to save compute, fixed in advance; --early): same top q397m3/q397 ≥ 0.95 (rows with a valid q397 answer) on
  COMPLETE shards with together at least 300 rows -> stop the machine, STOP (the main rule requires ≤ 0.90 over all rows).
  Partial states do not count (the probe computes the longest prompts first — not a random sample).
--notes-check (before the run, training rows only): gold code mentioned (whole word of the class list) in the converted notes
  against the raw field text of the Tinker file and against M3's own notes (d1 ∪ d3); error rows whose note names the gold code
  and where the six-channel recipe does not have it at rank 1, with the oracle gain — an upper bound for a perfect namer behind
  these notes, not a marginal gain.
--dry: q397m3 := copy of q397. Expectation: replacement Δ exactly 0 (bit-identical to the six-channel recipe), addition = copy
  null point exactly, same top 1.0 -> the rule would give STOP (takeover). Everything marked as a CHECK, no verdict, separate
  output files.
OUTPUT (only aggregates on the console): artifacts/eda/private/q397m3_eval/c115_q397m3_eval[_dry].{json,txt} and …_rows.npz
  (private); --notes-check -> c115_notes_check.json, --early -> c115_early.json.
LIMITS: the same 1,500 labels as all measurements (no new holdout); base OOFs not nested in the meta fold; OOF notes from Tinker,
  test notes from the own compute path (fidelity check: same top 0.781, no note identical) — the test-side channel is a different
  draw of the 397B behind a differently fitted M3 (full fit instead of fold adapters).
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

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import metric  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C110 = _load("c115_c110", COMP / "eda" / "c110_q397_deploy.py")  # six-channel recipe (line_specs, load_mats, tinker_channel)
C107 = C110.C107  # cross_fit, compare, rows_map, rows_ll, units, question_draws, cluster_boot, with_channel
C108 = _load("c115_c108", COMP / "eda" / "c108_m3_three_draws.py")  # fold_check
A9 = C108.A9
A8 = A9.A8  # load_rank
MQ = _load("c115_mq", COMP / "tools" / "make_q397m3_notes.py")  # sources of the notes (read_jsonl, M3_NOTES, TINKER_DIR)

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "q397m3_eval"
NAME = "c115_q397m3_eval"
CH, FLOOR = "q397m3", 0.05
RUN = "q397m3_oof_rp"
SHARDS = tuple(f"runs/{RUN}/q397m3_rank_f{k}/bench" for k in range(5))
REF_SHARDS = tuple(f"runs/m3d3_oof_rp/m3d3_rank_f{k}/bench" for k in range(5))  # d3: ranking settings for comparison
NOTES_REL = "derived/sft/q397m3_train_notes.jsonl"
UPLOAD_REL = "derived/sft_q397m3/q397m3_train_notes.jsonl"
M3 = "m3x3_full"
D1 = "m3p5"  # first M3 draw alone (runs/m3_5f/m3_rank/bench)
ANCHOR_UNITS = 7463  # six-channel recipe (c110 stack --line best --m3 m3x3_full): 0.8292222 = 7,463/9,000
ANCHOR_LL = 0.8670138272421626
ANCHOR_JSON = ART / "eda" / "private" / "q397_deploy" / "c110_stack_s029_best_m3x3.json"
DELTA_MIN, Z_MIN, SAME_TOP_MAX = Fraction(2, 1000), 1.0, Fraction(9, 10)
EARLY_SAME_TOP, EARLY_MIN_ROWS = Fraction(95, 100), 300
RULE_TEXT = ("Continue to the test side only if ADDITION or REPLACEMENT in the six-channel recipe gives Δ ≥ +0.002 with z ≥ 1.0 AND same top "
             "q397m3/q397 ≤ 0.90 (otherwise mere takeover) AND log loss not worse; otherwise STOP.")
# verdict values ("CONTINUE", "STOP") and form names ("addition", "replacement") are values of the report; other scripts
# compare against them
CONSEQUENCE = {"CONTINUE": "test side: tools/make_q397m3_notes.py --test-from runs/q397te_rp/proto_te/proto, then configs/vlm/q397m3te_rp.yaml "
                         "(≈ 9–12 USD of GPU time); then the deployment path (vlm.sft combine, a8 --rank, blend file with q397m3, make_submission --rank-only)",
               "STOP": "no test side; the approach \"M3 names behind foreign notes\" is closed in this form (cost: the pilot only)"}
GROUPS = C107.GROUPS
# note states of the 397B notes: name -> label recorded in the note files (tools/make_q397m3_notes.RECORDED_STATE)
STATES = {name: MQ.RECORDED_STATE[name] for name in ("complete", "draft", "open")}


class Technik(RuntimeError):
    """Technical finding: abort with exit 4, no verdict, nothing written."""


class Missing(Technik):
    """Input still missing (run not finished or not fetched): exit 3, nothing written."""


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=C107.json_default) + "\n"


def sha256_file(path: Path) -> str:
    return C107.sha256_file(Path(path))


def npy_sha256(P: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(P, dtype=np.float64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ------------------------------------------------------------------------------------------------ rule (pure, tested)

def form_check(c: dict) -> dict:
    """One form (addition or replacement) according to reading 1: Δ exactly ≥ +0.002, z ≥ 1.0, log loss not worse. c needs
    delta_exact (Fraction), z_q, ll_new, ll_base."""
    d = c["delta_exact"] if isinstance(c["delta_exact"], Fraction) else Fraction(c["delta_exact"])
    checks = {"delta": bool(d >= DELTA_MIN), "z": bool(float(c["z_q"]) >= Z_MIN), "logloss": bool(float(c["ll_new"]) <= float(c["ll_base"]))}
    return {"checks": checks, "ok": all(checks.values()), "delta": float(d), "z": float(c["z_q"]), "ll_new": float(c["ll_new"]),
            "ll_base": float(c["ll_base"])}


def same_top_ok(n_same: int, n_rows: int) -> bool:
    return n_rows > 0 and Fraction(int(n_same), int(n_rows)) <= SAME_TOP_MAX


def decide(add: dict, rep: dict, n_same: int, n_rows: int) -> dict:
    """The rule fixed in advance (reading above). add/rep: comparisons with delta_exact, z_q, ll_new, ll_base; n_same/n_rows: same top."""
    fa, fr = form_check(add), form_check(rep)
    take = same_top_ok(n_same, n_rows)
    weiter = (fa["ok"] or fr["ok"]) and take
    reason = []
    if not take:
        reason.append(f"same top q397m3/q397 {n_same}/{n_rows} = {n_same / max(1, n_rows):.3f} > 0.90 (takeover)")
    for lab, f in (("addition", fa), ("replacement", fr)):
        if not f["ok"]:
            reason.append(f"{lab}: " + ", ".join(k for k, v in f["checks"].items() if not v) + " missed")
    return {"verdict": "CONTINUE" if weiter else "STOP", "addition": fa, "replacement": fr, "same_top": {"n_same": int(n_same), "n_rows": int(n_rows),
            "share": n_same / max(1, n_rows), "ok": take}, "passed_forms": [lab for lab, f in (("addition", fa), ("replacement", fr)) if f["ok"]],
            "reasons": reason, "rule": RULE_TEXT}


def decide_early(n_same: int, n_rows: int, n_cover: int | None = None) -> dict:
    """Early stop (fixed in advance): same top ≥ 0.95 (n_same of n_rows rows with a valid q397 answer) on complete shards with
    together ≥ 300 rows (n_cover; default n_rows) -> STOP; otherwise keep computing."""
    n_cover = int(n_rows if n_cover is None else n_cover)
    if n_cover < EARLY_MIN_ROWS or n_rows <= 0:
        return {"verdict": "TOO EARLY", "n_same": int(n_same), "n_rows": int(n_rows), "n_cover": n_cover}
    stop = Fraction(int(n_same), int(n_rows)) >= EARLY_SAME_TOP
    return {"verdict": "STOP (early stop: takeover)" if stop else "KEEP COMPUTING", "n_same": int(n_same), "n_rows": int(n_rows),
            "n_cover": n_cover, "share": n_same / n_rows}


def top1(P: np.ndarray, seen: np.ndarray) -> np.ndarray:
    return metric.top3_indices(P, seen)[:, 0]


def same_top(Pa: np.ndarray, Pb: np.ndarray, seen: np.ndarray, rows: np.ndarray | None = None) -> tuple[int, int]:
    """(rows with the same top, rows) on `rows` (mask or indices; default all)."""
    ta, tb = top1(Pa, seen), top1(Pb, seen)
    m = np.ones(len(ta), bool) if rows is None else (np.asarray(rows) if np.asarray(rows).dtype == bool else np.isin(np.arange(len(ta)), rows))
    return int((ta[m] == tb[m]).sum()), int(m.sum())


def mention_pattern(classes: list[str]) -> re.Pattern:
    """Codes of the class list as whole words, longer ones first (as in vlm/sft.lead_code); without NONE."""
    codes = sorted({str(c) for c in classes if c != "NONE"}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(c) for c in codes) + r")(?![A-Za-z0-9])")


def mentioned(texts: dict[int, str], pat: re.Pattern, n: int) -> list[set]:
    return [set(pat.findall(texts.get(i, "") or "")) for i in range(n)]


def topic_never(y: np.ndarray, folds: np.ndarray, topic: np.ndarray) -> np.ndarray:
    """Cell "never for the topic": error row whose gold code was the root of NO row of the same topic (topic_name) in the other
    folds."""
    topic = np.asarray(topic).astype(str)
    same = (folds[:, None] != folds[None, :]) & (topic[:, None] == topic[None, :]) & (y[:, None] == y[None, :])
    return (y != 0) & ~same.any(1)


# ------------------------------------------------------------------------------------------------ channel from shards

def shard_dir(shard: str) -> tuple[Path, str]:
    run, job, tag = A9.parse_source(shard)
    return A9.run_dir(run) / job, tag


def assemble(shards, n: int, seen: np.ndarray, folds: np.ndarray, require_all: bool = True) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Channel from fold shards (shard k = fold k): each complete and exact (a8.load_rank without partial state, completeness mask),
    covering EXACTLY the rows of its fold, no overlap. require_all: all shards must exist (otherwise a technical finding); otherwise
    only the complete ones (early stop). -> (P [n, K], covered [n], statistics per shard)."""
    P, have, info = np.zeros((n, len(seen))), np.zeros(n, bool), []
    for k, shard in enumerate(shards):
        d, tag = shard_dir(shard)
        need = [f"{tag}_meta.json", f"{tag}_idx.npy", f"{tag}_proto_rank.npy", f"{tag}_proto_rank.done.npy"]
        if not all((d / f).is_file() for f in need):
            if require_all:
                raise Missing(f"{shard}: files missing ({', '.join(f for f in need if not (d / f).is_file())})")
            continue
        try:
            R, h, st = A8.load_rank(shard, n, seen, table="train", allow_partial=False, require_full_table=False)
        except SystemExit as exc:
            if require_all:
                raise Technik(str(exc)) from None
            continue
        if not st["completion_manifest"]:
            raise Technik(f"{shard}: completeness mask missing")
        rows = np.flatnonzero(h)
        want = np.flatnonzero(folds == k)
        if not np.array_equal(rows, want):
            raise Technik(f"{shard}: covers {len(rows)} rows, exactly the {len(want)} rows of fold {k} are required (fold assignment violated)")
        if (have & h).any():
            raise Technik(f"{shard}: overlaps with an earlier shard")
        P[h], have[h] = R[h], True
        info.append({"shard": shard, "fold": k, **{x: st[x] for x in ("rows_probe", "rows_done", "rows_exact", "mass_median", "mass_p05",
                                                                        "mass_below_half", "idx_sha256")}})
    if require_all and not have.all():
        raise Technik(f"channel covers {int(have.sum())} of {n} rows")
    return P, have, info


def shard_meta(shard: str) -> dict:
    d, tag = shard_dir(shard)
    return json.loads((d / f"{tag}_meta.json").read_text())


def check_shard_settings(shards, ref_shards, notes_rel: str = NOTES_REL) -> dict:
    """sft_from = note file; ranking settings (a9.rank_settings) equal in all shards and equal to d3."""
    got = [A9.rank_settings(shard_meta(s)) for s in shards]
    src = [(shard_meta(s).get("sft") or {}).get("sft_from") for s in shards]
    if any(x != notes_rel for x in src):
        raise Technik(f"sft_from of the shards {sorted(set(map(str, src)))} instead of {notes_rel}")
    if len({json.dumps(g, sort_keys=True, default=str) for g in got}) != 1:
        raise Technik("the shards were computed differently (a9.rank_settings)")
    ref = None
    try:
        refs = [A9.rank_settings(shard_meta(s)) for s in ref_shards]
        ref = refs[0] if len({json.dumps(r, sort_keys=True, default=str) for r in refs}) == 1 else None
    except (OSError, json.JSONDecodeError):
        ref = None
    if ref is not None and json.dumps(ref, sort_keys=True, default=str) != json.dumps(got[0], sort_keys=True, default=str):
        diff = {k: (got[0].get(k), ref.get(k)) for k in set(got[0]) | set(ref) if got[0].get(k) != ref.get(k)}
        raise Technik(f"ranking settings differ from d3: {diff}")
    return {"settings": got[0], "same_as_d3": ref is not None, "sft_from": notes_rel}


def check_notes(root: Path, n: int) -> dict:
    """Note file = checksum in the meta file = copy in the upload folder; 1,500 training rows, no finding given."""
    from vlm import sft as S

    f, up = root / NOTES_REL, root / UPLOAD_REL
    meta_f = f.with_suffix(".meta.json")
    if not f.is_file() or not meta_f.is_file():
        raise Missing(f"{NOTES_REL} or meta file missing — tools/make_q397m3_notes.py")
    meta = json.loads(meta_f.read_text())
    sha = sha256_file(f)
    if meta.get("sha256") != sha:
        raise Technik(f"{NOTES_REL}: checksum does not match the meta file")
    if not up.is_file() or sha256_file(up) != sha:
        raise Technik(f"{UPLOAD_REL}: copy in the upload folder missing or different (the remote machine computed with this file)")
    table, notes = S.load_notes(f)
    if table != "train" or sorted(notes) != list(range(n)):
        raise Technik(f"{NOTES_REL}: table {table}, {len(notes)} rows (required train, all {n})")
    if any(fd is not None for _, fd in notes.values()):
        raise Technik(f"{NOTES_REL}: findings given — M3 is supposed to write the finding itself")
    states = {}
    for rec in MQ.read_jsonl(f):
        states[int(rec["row"])] = str(rec.get("q397_state") or "")
    return {"sha256": sha, "rows": len(notes), "q397_states": meta.get("q397_states"), "states_by_row": states,
            "texts": {i: t for i, (t, _) in notes.items()}, "source_sha256": meta.get("source_sha256"), "labels_sha256": meta.get("labels_sha256")}


# ------------------------------------------------------------------------------------------------ context and recipe

def context() -> tuple[dict, dict, dict]:
    ctx = C110.setup()
    mats, sha16, metas = C110.load_mats(ctx, M3)
    d1 = ART / f"oof_{D1}.npy"
    if d1.is_file():
        mats[D1] = np.load(d1)
    _, has, _, _ = C110.tinker_channel(ctx)
    ctx["q397_valid"] = has
    ctx["topic_never"] = topic_never(ctx["y"], ctx["folds"], ctx["topic"])
    ctx["draws"] = C107.question_draws(ctx["q_code"])
    return ctx, mats, sha16


def spec_base() -> dict:
    return C110.line_specs("best", M3)[1]


def replaced(spec: dict, old: str, new: str) -> dict:
    names = [new if nm == old else nm for nm in spec["names"]]
    floors = {(new if k == old else k): float(v) for k, v in spec["floors"].items()}
    if names == list(spec["names"]):
        raise ValueError(f"{old} is not part of the recipe")
    return {**spec, "names": names, "floors": floors}


def run_recipe(ctx: dict, mats: dict, spec: dict) -> dict:
    fit = C107.cross_fit(ctx["y"], ctx["folds"], ctx["seen"], mats, spec)
    y, seen = ctx["y"], ctx["seen"]
    if not C107.sane(fit["Q"], seen):
        raise Technik(f"predictions of the recipe {spec['names']} not finite/normalised")
    return {"fit": fit, "rows": C107.rows_map(y, fit["Q"], seen), "ll": C107.rows_ll(y, fit["Q"]),
            "rows_pre": C107.rows_map(y, fit["P"], seen), "rows_set": C107.rows_map(y, C107.set_align(fit["P"], y, seen), seen)}


def anchor(ctx: dict, base: dict) -> dict:
    units = int(C107.units(base["rows"]).sum())
    ll = float(base["ll"].mean())
    if units != ANCHOR_UNITS or abs(ll - ANCHOR_LL) > 1e-9:
        raise Technik(f"anchor of the six-channel recipe missed: {units}/9000, log loss {ll:.10f} (required {ANCHOR_UNITS}, {ANCHOR_LL:.10f})")
    out = {"map3": units / 9000, "units": units, "logloss": ll}
    if ANCHOR_JSON.is_file():
        js = json.loads(ANCHOR_JSON.read_text())
        if int(js["cross"]["units"]) != units:
            raise Technik(f"anchor differs from {ANCHOR_JSON.name}")
        out["c110_json"] = str(ANCHOR_JSON.relative_to(COMP))
    return out


def channel_stats(ctx: dict, P: np.ndarray) -> dict:
    y, seen = ctx["y"], ctx["seen"]
    r = C107.rows_map(y, P, seen)
    cb = C107.cluster_boot(r, ctx["q_code"], ctx["draws"])
    t = top1(P, seen)
    out = {"map3": float(r.mean()), "se_q": cb["se_q"], "top1": float((t == y).mean()), "none_top": float((t == 0).mean()),
           "logloss": float(C107.rows_ll(y, P).mean()), "groups": {}}
    for g in GROUPS:
        m = ctx["group"] == g
        out["groups"][g] = {"n": int(m.sum()), "map3": float(r[m].mean()), "top1": float((t[m] == y[m]).mean())}
    m = ctx["topic_never"]
    out["groups"]["never for the topic"] = {"n": int(m.sum()), "map3": float(r[m].mean()), "top1": float((t[m] == y[m]).mean())}
    return out


def paired_channels(ctx: dict, Pa: np.ndarray, Pb: np.ndarray) -> dict:
    y, seen = ctx["y"], ctx["seen"]
    ra, rb = C107.rows_map(y, Pa, seen), C107.rows_map(y, Pb, seen)
    return C107.compare(ra, rb, ctx, ctx["draws"], C107.rows_ll(y, Pa), C107.rows_ll(y, Pb))


def same_top_table(ctx: dict, P: np.ndarray, mats: dict, states: dict[int, str]) -> dict:
    """Same top of the channel with q397 (valid rows), m3x3, d1; reference q397/m3x3; per 397B state and group."""
    seen, valid = ctx["seen"], ctx["q397_valid"]
    st = np.array([states.get(i, "") for i in range(ctx["n"])])
    out = {}
    pairs = [("q397m3/q397", P, mats["q397"], valid), ("q397m3/m3x3", P, mats[M3], None), ("q397/m3x3", mats["q397"], mats[M3], valid)]
    if D1 in mats:
        pairs.append(("q397m3/d1", P, mats[D1], None))
    for lab, A, B, base_mask in pairs:
        m0 = np.ones(ctx["n"], bool) if base_mask is None else base_mask
        s, k = same_top(A, B, seen, m0)
        rec = {"n_same": s, "n_rows": k, "share": s / max(1, k), "by_state": {}, "by_group": {}}
        for name, label in STATES.items():
            sx, kx = same_top(A, B, seen, m0 & (st == label))
            rec["by_state"][name] = {"n_same": sx, "n_rows": kx, "share": sx / max(1, kx)}
        for g in GROUPS:
            sg, kg = same_top(A, B, seen, m0 & (ctx["group"] == g))
            rec["by_group"][g] = {"n_same": sg, "n_rows": kg, "share": sg / max(1, kg)}
        out[lab] = rec
    return out


def choice_among_candidates(ctx: dict, P: np.ndarray, mats: dict, texts: dict[int, str], base_rows: np.ndarray, base_Q: np.ndarray) -> dict:
    """Choice among candidates: error rows whose 397B note names the gold code — top correct for q397m3, q397, m3x3 and the
    six-channel recipe."""
    y, seen, n = ctx["y"], ctx["seen"], ctx["n"]
    pat = mention_pattern(ctx["classes"])
    men = mentioned(texts, pat, n)
    cls = ctx["classes"]
    err = y != 0
    hit = np.array([cls[y[i]] in men[i] for i in range(n)]) & err
    out = {"error_rows": int(err.sum()), "gold_in_notes": int(hit.sum())}
    for lab, A in (("q397m3", P), ("q397", mats["q397"]), ("m3x3", mats[M3]), ("six_channel", base_Q)):
        t = top1(A, seen)
        out[lab] = {"top1_gold_in_notes": float((t[hit] == y[hit]).mean()) if hit.any() else None,
                    "top1_gold_not_in_notes": float((t[err & ~hit] == y[err & ~hit]).mean()) if (err & ~hit).any() else None}
    miss = hit & (top1(base_Q, seen) != y)
    out["six_channel_misses_with_gold_in_notes"] = int(miss.sum())
    out["oracle_gain_six_channel"] = float((1.0 - base_rows[miss]).sum() / n)
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def evaluate(ctx: dict, mats: dict, P: np.ndarray, states: dict[int, str], texts: dict[int, str], dry: bool) -> tuple[dict, dict]:
    y, seen, n = ctx["y"], ctx["seen"], ctx["n"]
    mats = {**mats, CH: P}
    sb = spec_base()
    base = run_recipe(ctx, mats, sb)
    rep: dict = {"dry": dry, "anchor": anchor(ctx, base)}
    # mandatory check: replacing q397 by a copy of q397 reproduces the six-channel recipe bit-identically
    probe = run_recipe(ctx, {**mats, "q397copy": mats["q397"]}, replaced(sb, "q397", "q397copy"))
    if not np.array_equal(probe["fit"]["Q"], base["fit"]["Q"]):
        raise Technik("mandatory check: replacement q397 := copy of q397 does not reproduce the six-channel recipe bit-identically")
    rep["mandatory_check"] = "replacement q397 := copy of q397 bit-identical to the six-channel recipe"

    spec_add, spec_rep = C107.with_channel(sb, CH, FLOOR), replaced(sb, "q397", CH)
    add, rpl = run_recipe(ctx, mats, spec_add), run_recipe(ctx, mats, spec_rep)
    null = run_recipe(ctx, {**mats, CH: mats["q397"]}, spec_add)
    mean = run_recipe(ctx, {**mats, "m3x4q": (3.0 * mats[M3] + P) / 4.0}, replaced(sb, M3, "m3x4q"))
    mean0 = run_recipe(ctx, {**mats, "m3x4q": (3.0 * mats[M3] + mats["q397"]) / 4.0}, replaced(sb, M3, "m3x4q"))
    draws = ctx["draws"]

    def cmp(x: dict) -> dict:
        c = C107.compare(x["rows"], base["rows"], ctx, draws, x["ll"], base["ll"])
        tn = C107.cluster_boot(np.asarray(x["rows"], float) - np.asarray(base["rows"], float), ctx["q_code"], draws, ctx["topic_never"])
        c["groups"]["never for the topic"] = {"n": tn["n_rows"], "delta": tn["mean"], "se_q": tn["se_q"],
                                         "contrib": float((x["rows"] - base["rows"])[ctx["topic_never"]].sum() / n)}
        return c

    c_add, c_rep, c_null, c_mean, c_mean0 = cmp(add), cmp(rpl), cmp(null), cmp(mean), cmp(mean0)
    rep["recipes"] = {
        "six_channel": {"map3": float(base["rows"].mean()), "logloss": float(base["ll"].mean()), "pre_last_step": float(base["rows_pre"].mean()),
                        "one_set": float(base["rows_set"].mean())},
        "addition": {**C110.summarize_cross(c_add), "delta_exact": c_add["delta_exact"], "pre_last_step": float(add["rows_pre"].mean()),
                     "one_set": float(add["rows_set"].mean()), "weights": C107.weights_report(add["fit"], CH)},
        "replacement": {**C110.summarize_cross(c_rep), "delta_exact": c_rep["delta_exact"], "pre_last_step": float(rpl["rows_pre"].mean()),
                        "one_set": float(rpl["rows_set"].mean()), "weights": C107.weights_report(rpl["fit"], CH)},
        "null_addition_copy_q397": {**C110.summarize_cross(c_null), "delta_exact": c_null["delta_exact"]},
        "mean_info": {**C110.summarize_cross(c_mean), "delta_exact": c_mean["delta_exact"], "definition": "M3 = (3·m3x3_full + q397m3)/4"},
        "mean_null_copy_q397": {**C110.summarize_cross(c_mean0), "delta_exact": c_mean0["delta_exact"],
                                "definition": "M3 = (3·m3x3_full + q397)/4 — arithmetic share of q397 without new information"},
    }
    rep["channel"] = {"q397m3": channel_stats(ctx, P), "q397": channel_stats(ctx, mats["q397"]), "m3x3": channel_stats(ctx, mats[M3])}
    if D1 in mats:
        rep["channel"]["d1"] = channel_stats(ctx, mats[D1])
    rep["channel_paired"] = {"q397m3-q397": C110.summarize_cross(paired_channels(ctx, P, mats["q397"])),
                             "q397m3-m3x3": C110.summarize_cross(paired_channels(ctx, P, mats[M3]))}
    if D1 in mats:
        rep["channel_paired"]["q397m3-d1"] = C110.summarize_cross(paired_channels(ctx, P, mats[D1]))
    rep["same_top"] = same_top_table(ctx, P, mats, states)
    rep["candidates"] = choice_among_candidates(ctx, P, mats, texts, base["rows"], base["fit"]["Q"])
    st = rep["same_top"]["q397m3/q397"]
    dec = decide(c_add, c_rep, st["n_same"], st["n_rows"])
    warn = []
    for lab, c in (("addition", c_add), ("replacement", c_rep)):
        if lab in dec["passed_forms"] and c["delta_exact"] <= c_null["delta_exact"]:
            warn.append(f"{lab} passes, but does not lie above the copy null point (Δ {float(c['delta_exact']):+.5f} ≤ {float(c_null['delta_exact']):+.5f})")
    dec["warnings"] = warn
    if dry:
        dec = {**dec, "verdict": f"CHECK (--dry) — the rule would give {dec['verdict']}"}
    dec["consequence"] = None if dry else CONSEQUENCE[dec["verdict"]]
    rep["decision"] = dec
    rows = {"base": base["rows"], "addition": add["rows"], "replacement": rpl["rows"], "null": null["rows"], "mean": mean["rows"], "mean0": mean0["rows"],
            "ll_base": base["ll"], "ll_addition": add["ll"], "ll_replacement": rpl["ll"], "channel": C107.rows_map(y, P, seen)}
    return rep, rows


def fmt_cmp(c: dict) -> str:
    pf = " ".join(f"{x:+.4f}" for x in c["per_fold"])
    g = " | ".join(f"{k} ({v['n']}) {v['delta']:+.4f}±{v['se_q']:.4f}" for k, v in c["groups"].items())
    return (f"{c['map_base']:.5f} -> {c['map_new']:.5f}: Δ {c['delta']:+.5f} ± {c['se_q']:.5f} (z {c['z_q']:+.2f}; 95 % {c['ci95_q'][0]:+.4f} … "
            f"{c['ci95_q'][1]:+.4f}); per fold {pf} ({c['folds_gt0']}/5 > 0); log loss {c['ll_base']:.4f} -> {c['ll_new']:.4f} "
            f"(gain {c['ll_gain']:+.4f}, z {c['ll_z_q']:+.2f})\n      groups: {g}")


def report_lines(rep: dict) -> list[str]:
    tag = "CHECK (--dry: q397m3 := copy of q397) — " if rep["dry"] else ""
    a = rep["anchor"]
    L = [f"{tag}c115 q397m3 — anchor of the six-channel recipe {a['units']}/9000 = {a['map3']:.7f}, log loss {a['logloss']:.7f}; {rep['mandatory_check']}"]
    for lab in ("q397m3", "q397", "m3x3", "d1"):
        c = rep["channel"].get(lab)
        if c:
            gs = " | ".join(f"{g} {v['map3']:.3f}" for g, v in c["groups"].items())
            L.append(f"  single channel {lab:7s} MAP@3 {c['map3']:.4f} ± {c['se_q']:.4f}, top correct {c['top1']:.3f}, NONE at the top {c['none_top']:.3f}, "
                     f"log loss {c['logloss']:.3f} | {gs}")
    for lab, c in rep["channel_paired"].items():
        L.append(f"  paired {lab}: Δ {c['delta']:+.4f} ± {c['se_q']:.4f} (z {c['z_q']:+.2f})")
    for lab, s in rep["same_top"].items():
        bs = " ".join(f"{k} {v['share']:.3f} ({v['n_rows']})" for k, v in s["by_state"].items())
        L.append(f"  same top {lab}: {s['n_same']}/{s['n_rows']} = {s['share']:.3f} | per 397B state {bs}")
    cand = rep["candidates"]
    L.append(f"  choice among candidates: {cand['gold_in_notes']} of {cand['error_rows']} error rows with the gold code in the 397B note; top correct there "
             + ", ".join(f"{k} {cand[k]['top1_gold_in_notes']:.3f}" for k in ("q397m3", "q397", "m3x3", "six_channel") if cand[k]['top1_gold_in_notes'] is not None)
             + f"; the six-channel recipe misses there {cand['six_channel_misses_with_gold_in_notes']} (oracle {cand['oracle_gain_six_channel']:+.4f})")
    for lab in ("addition", "replacement", "null_addition_copy_q397", "mean_info", "mean_null_copy_q397"):
        c = rep["recipes"][lab]
        extra = f" | share q397m3 {np.mean(c['weights']['share_new']):.3f}" if "weights" in c else ""
        L.append(f"  {lab}: " + fmt_cmp({**c, "map_base": c["map_base"], "map_new": c["map_new"]}) + extra)
    d = rep["decision"]
    L.append(f"VERDICT: {d['verdict']} — passed: {', '.join(d['passed_forms']) or 'no form'}; same top {d['same_top']['share']:.3f}"
             + (f"; reasons: {'; '.join(d['reasons'])}" if d["reasons"] else ""))
    for w in d.get("warnings", []):
        L.append(f"  WARNING: {w}")
    if d.get("consequence"):
        L.append(f"  consequence: {d['consequence']}")
    return L


# ------------------------------------------------------------------------------------------------ commands

def cmd_status() -> int:
    from vlm import common as C

    root = C.find_data_root()
    import labels

    train = C.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    seen, folds, n = labels.seen_mask(train["root_code"], classes), C.load_folds(len(train)), len(train)
    ok_all = True
    for k, shard in enumerate(SHARDS):
        st = C108.shard_status(shard, n, seen)
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
    print(f"-> {'COMPLETE' if ok_all else 'INCOMPLETE'}")
    return 0 if ok_all else 3


def cmd_early() -> int:
    ctx, mats, _ = context()
    P, have, info = assemble(SHARDS, ctx["n"], ctx["seen"], ctx["folds"], require_all=False)
    m = have & ctx["q397_valid"]
    s, k = same_top(P, mats["q397"], ctx["seen"], m)
    dec = decide_early(s, k, int(have.sum()))
    rec = {"shards_complete": [x["shard"] for x in info], "rows": int(have.sum()), "same_top_q397": {"n_same": s, "n_rows": k}, "decision": dec}
    if have.any():
        y, seen = ctx["y"], ctx["seen"]
        rec["channel_on_rows"] = {lab: float(C107.rows_map(y[have], A[have], seen).mean()) for lab, A in (("q397m3", P), ("q397", mats["q397"]), ("m3x3", mats[M3]))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c115_early.json").write_text(to_json({**rec, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(f"early-stop check: {len(info)} complete shards, {int(have.sum())} rows; same top q397m3/q397 {s}/{k}"
          + (f" = {s / k:.3f}" if k else "") + f" -> {dec['verdict']}")
    if "channel_on_rows" in rec:
        print("  MAP@3 on these rows: " + ", ".join(f"{a} {b:.4f}" for a, b in rec["channel_on_rows"].items()))
    return 0


def cmd_notes_check() -> int:
    ctx, mats, _ = context()
    root, n, y, cls = ctx["root"], ctx["n"], ctx["y"], ctx["classes"]
    nt = check_notes(root, n)
    pat = mention_pattern(cls)
    raw = MQ.read_jsonl(root / MQ.TINKER_DIR / "raw.jsonl")
    raw_text = {int(r["row"]): str(r.get("text") or "") for r in raw if r.get("phase") == "first"}
    m3_draws: dict[str, dict[int, str]] = {}  # M3's own notes per draw (d1: greedy 3,200, d3: sampled 5,200)
    for draw, parts in MQ.M3_NOTES.items():
        files = [ART / "runs" / run / run / job / "gen_notes.jsonl" for run, job in parts]
        if all(f.is_file() for f in files):
            m3_draws[draw] = {int(r["row"]): str(r.get("text") or "") for f in files for r in MQ.read_jsonl(f)}
    men_notes, men_raw = mentioned(nt["texts"], pat, n), mentioned(raw_text, pat, n)
    err = y != 0
    gold = [cls[v] for v in y]
    in_notes = np.array([gold[i] in men_notes[i] for i in range(n)]) & err
    in_raw = np.array([gold[i] in men_raw[i] for i in range(n)]) & err
    sb = spec_base()
    base = run_recipe(ctx, mats, sb)
    anc = anchor(ctx, base)
    t_b = top1(base["fit"]["Q"], ctx["seen"])
    miss = in_notes & (t_b != y)
    rec = {"anchor": anc, "notes_sha256": nt["sha256"], "error_rows": int(err.sum()),
           "gold_in_notes": int(in_notes.sum()), "gold_in_raw_text": int(in_raw.sum()),
           "share_notes": float(in_notes.sum() / err.sum()), "share_raw_text": float(in_raw.sum() / err.sum()),
           "lost_by_cleaning": int((in_raw & ~in_notes).sum()), "gained_by_cleaning": int((in_notes & ~in_raw).sum()),
           "by_group": {}, "by_state": {},
           "six_channel_misses_gold_in_notes": int(miss.sum()), "oracle_gain": float((1.0 - base["rows"][miss]).sum() / n)}
    st = np.array([nt["states_by_row"].get(i, "") for i in range(n)])
    for g in GROUPS:
        m = err & (ctx["group"] == g)
        if m.any():
            rec["by_group"][g] = {"n": int(m.sum()), "notes": float(in_notes[m].mean()), "raw_text": float(in_raw[m].mean())}
    m = ctx["topic_never"]
    rec["by_group"]["never for the topic"] = {"n": int(m.sum()), "notes": float(in_notes[m].mean()), "raw_text": float(in_raw[m].mean())}
    for name, label in STATES.items():
        m = err & (st == label)
        if m.any():
            rec["by_state"][name] = {"n": int(m.sum()), "notes": float(in_notes[m].mean()), "raw_text": float(in_raw[m].mean())}
    m3_in = {d: np.array([gold[i] in men[i] for i in range(n)]) & err for d, men in ((d, mentioned(t, pat, n)) for d, t in m3_draws.items())}
    if len(m3_in) == len(MQ.M3_NOTES):
        m3_in["d1∪d3"] = np.logical_or.reduce(list(m3_in.values()))
    # choice among candidates: top correct if the OWN notes name the gold code (q397 behind the 397B notes, d1/d3 behind theirs)
    draw_P = {"d1": mats.get(D1)}
    try:
        draw_P["d3"] = A9.load_draw("+".join(REF_SHARDS), n, ctx["seen"])[0]
    except SystemExit:
        draw_P["d3"] = None
    tq = top1(mats["q397"], ctx["seen"])
    rec["selection"] = {"q397_given_gold_in_397b_notes": float((tq[in_notes] == y[in_notes]).mean())}
    for d in ("d1", "d3"):
        if draw_P.get(d) is not None and d in m3_in:
            td = top1(draw_P[d], ctx["seen"])
            rec["selection"][f"{d}_given_gold_in_own_notes"] = float((td[m3_in[d]] == y[m3_in[d]]).mean())
            rec["selection"][f"{d}_on_397b_candidate_rows"] = float((td[in_notes] == y[in_notes]).mean())
    rec["selection"]["six_channel_on_397b_candidate_rows"] = float((t_b[in_notes] == y[in_notes]).mean())
    rec["m3_notes"] = {}
    for d, in_m3 in m3_in.items():
        only = in_notes & ~in_m3
        miss_only = only & (t_b != y)
        rec["m3_notes"][d] = {"gold_in_m3": int(in_m3.sum()), "share": float(in_m3.sum() / err.sum()), "only_397b": int(only.sum()),
                              "only_m3": int((in_m3 & ~in_notes).sum()), "only_397b_six_channel_misses": int(miss_only.sum()),
                              "oracle_gain_only_397b": float((1.0 - base["rows"][miss_only]).sum() / n),
                              "by_group": {**{g: float(in_m3[err & (ctx["group"] == g)].mean()) for g in GROUPS if (err & (ctx["group"] == g)).any()},
                                           "never for the topic": float(in_m3[ctx["topic_never"]].mean())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c115_notes_check.json").write_text(to_json({**rec, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(f"notes {NOTES_REL} (SHA-256 {nt['sha256'][:16]}…); anchor of the six-channel recipe {anc['units']}/9000")
    print(f"  gold code mentioned (error rows {rec['error_rows']}): converted notes {rec['gold_in_notes']} ({rec['share_notes']:.3f}), raw field text "
          f"{rec['gold_in_raw_text']} ({rec['share_raw_text']:.3f}); lost through cleaning {rec['lost_by_cleaning']}, gained {rec['gained_by_cleaning']}")
    print("  per group: " + " | ".join(f"{g} ({v['n']}) {v['notes']:.3f}/{v['raw_text']:.3f}" for g, v in rec["by_group"].items()))
    print("  per 397B state: " + " | ".join(f"{g} ({v['n']}) {v['notes']:.3f}/{v['raw_text']:.3f}" for g, v in rec["by_state"].items()))
    print(f"  the six-channel recipe does not have the gold code at rank 1 although the note names it: {rec['six_channel_misses_gold_in_notes']} rows (oracle {rec['oracle_gain']:+.4f})")
    print("  top correct if the own notes name the gold code: " + ", ".join(f"{k} {v:.3f}" for k, v in rec["selection"].items()))
    for d, v in rec["m3_notes"].items():
        print(f"  gold code in M3 notes {d}: {v['gold_in_m3']} ({v['share']:.3f}; per group "
              + " | ".join(f"{g} {x:.3f}" for g, x in v["by_group"].items()) + f"); only in the 397B note {v['only_397b']} (six-channel recipe not at "
                                                                               f"rank 1 there: {v['only_397b_six_channel_misses']}, oracle {v['oracle_gain_only_397b']:+.4f}), only for M3 {v['only_m3']}")
    return 0


def cmd_eval(dry: bool) -> int:
    t0 = time.time()
    ctx, mats, sha16 = context()
    nt = check_notes(ctx["root"], ctx["n"])
    tech: dict = {"notes": {k: v for k, v in nt.items() if k not in ("texts", "states_by_row")}}
    if dry:
        P = mats["q397"].copy()
        tech["channel"] = "CHECK: q397m3 := copy of q397"
    else:
        P, _, info = assemble(SHARDS, ctx["n"], ctx["seen"], ctx["folds"], require_all=True)
        tech["shards"] = info
        tech["settings"] = check_shard_settings(SHARDS, REF_SHARDS)
        recs = C108.fold_check("+".join(SHARDS), ctx["folds"], {})
        bad = [f"{r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"]]
        if bad:
            raise Technik("fold integrity violated: " + "; ".join(bad))
        tech["fold_check"] = recs
        tech["fold_check_unknown"] = [f"{r['run']}/{r['job']}: {u}" for r in recs for u in r["unknown"]]
        f = ART / f"oof_{CH}.npy"
        if f.is_file():
            if not np.array_equal(np.load(f), P):
                raise Technik(f"{f.name} exists but is not the channel from the five shards")
            tech["oof_artifact_bit_identical"] = True
        tech["channel_sha256"] = npy_sha256(P)
    rep, rows = evaluate(ctx, mats, P, nt["states_by_row"], nt["texts"], dry)
    rep["technical"] = tech
    rep["oof_sha16"] = sha16
    rep["seconds"] = round(time.time() - t0, 1)
    lines = report_lines(rep)
    for line in lines:
        print(line)
    for u in tech.get("fold_check_unknown", []):
        print(f"  NOTE fold integrity cannot be determined: {u}")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = NAME + ("_dry" if dry else "")
    (OUT / f"{stem}.json").write_text(to_json({**rep, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    np.savez_compressed(OUT / f"{stem}_rows.npz", **rows)
    print(f"written: {(OUT / stem).relative_to(COMP)}.json / .txt / _rows.npz")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true")
    g.add_argument("--early", action="store_true")
    g.add_argument("--notes-check", action="store_true")
    g.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.status:
            return cmd_status()
        if a.early:
            return cmd_early()
        if a.notes_check:
            return cmd_notes_check()
        return cmd_eval(a.dry)
    except Missing as exc:
        print(f"NOT YET EVALUABLE: {exc}", file=sys.stderr)
        return 3
    except Technik as exc:
        print(f"ABORT (technical): {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
