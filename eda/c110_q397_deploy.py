"""DEPLOYMENT PATH of Qwen3.5-397B as the SIXTH channel `q397` — OOF artifact from the Tinker answers, blend files for
tools/make_submission.py, check of the test side and two switches that need no new model runs. Build script:
tools/builds/s029_q397.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c110_q397_deploy.py source                      # source folder runs/q397_tinker/proto_t/proto (copy of the Tinker file)
    python -I eda/a8_proto_oof.py q397 runs/q397_tinker/proto_t/proto   # OOF artifact artifacts/oof_q397.npy, meta_q397.json
    python -I eda/c110_q397_deploy.py verify-oof                  # oof_q397.npy bit-identical to the channel of eda/c107_q397_full.py
    python -I eda/c110_q397_deploy.py stack --line best|safe [--m3 m3x2_full_cv01|m3x3_full] [--beta-novel] (--save | --verify)
    python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 (--save | --verify)
    python -I eda/c110_q397_deploy.py test-status runs/<run>/proto_te/proto [--placeholder]
    python -I eda/c110_q397_deploy.py readable [--save]           # REL-01 list: blank images ∪ pr27 "readable = no"

SOURCE: data/derived/open_teacher/qwen35-397b/labels.jsonl — Qwen3.5-397B-A17B via the Tinker API in the teacher protocol
(greedy, thinking mode up to 2,500 tokens, finishing pass up to 1,500 without thinking), training rows ONLY, 1,496 of 1,500
valid. `source` creates a byte-identical copy as job runs/q397_tinker/proto_t/proto (proto_labels.jsonl + proto_meta.json with
provenance and checksum), so that eda/a8_proto_oof.py builds the artifact the same way as for pr27 (labels_to_probs,
kaggle_core.artifacts). `verify-oof` checks: oof_q397.npy is bit-identical to the channel that c107 evaluated.

BLEND FILES (`stack`; format of tools/make_submission.py, fields as in stack_cv01_best_l2.json):
  best  primary-line recipe + q397: v1r, zs27q, think18nb (floor 0.01; weight × 1.4), pr27xm (0.05), M3 (0.05), q397 (0.05); β 0.2;
        last step margin alignment 0.75 for classes with at least 3 rows (transductive)  -> artifacts/stack_s029_best[_m3x3][_bn].json
  safe  inductive-line recipe + q397: v1r, zs27t, think18sb (0.01), pr27xm (0.05), M3 (0.05), q397 (0.05); β 0.4; inductive alignment 0.75
        (class weights from make_submission.oof_mixture of the 1,500 fit rows, checksum inductive_check)
                                                                                         -> artifacts/stack_s029_safe[_m3x3][_bn].json
  M3 = m3x2_full_cv01 (default, two draws on all 1,500 rows) or m3x3_full (three draws; artifact from tools/builds/s029_m3x3.sh /
  eda/a9_m3_draws_oof.py). Like c107, the inductive-line version fits on 1,500 rows with m3x2_full_cv01 (the earlier inductive-line blend file used
  600 rows with m3x2) — this alone already changes some top codes.
  WEIGHTS on log loss from ALL 1,500 rows (a2.fit_mix as in eda/c99_m3_deploy.fit_theta; then think18nb × 1.4), arithmetic of the
  build and test paths (floor without prior masking). Field cv = CROSS VALUE of the recipe with q397 (weights per fold from the other
  four, last step per fold; c107.cross_fit) — for M3 = m3x2_full_cv01 exactly the value of the c107 decision (primary
  7,453/9,000 = 0.82811, inductive 7,386/9,000 = 0.82067; otherwise abort).
  CHECKS: checksums of the base OOFs and blend files (c107.check_stacks, load_base); anchors without q397 exact (best 7,379/9,000,
  inductive 7,308/9,000); the same fitting path without q397 reproduces θ of the earlier blend files (stack_cv01_best_l2.json, 1,500
  rows; inductive: 600 rows with m3x2) BIT-IDENTICALLY; rank_only entry of the M3 channel as in stack_cv01_best_l2.json;
  make_submission.oof_mixture of the saved blend = own blend (mask first bit-identical, test-path arithmetic ≤ 1e-9); in-sample
  with the last step.
  --verify recomputes everything and requires the saved file to be identical character for character (it contains no
  timestamps).

SWITCHES (default OFF, each with a cross value):
  --beta-novel  arithmetic share by code NOVELTY: per (row, class) β_new instead of β if the code was never the root among the
                reference rows of the same task (question_id, same topic_name) (make_submission.code_novelty; field
                arith_beta_novel). ONE fixed β_new from all 1,500 rows by the rule "smallest log loss of the blend before the last
                step" (grid 0.2…0.8; novelty of the training rows against the other folds). Cross values: nested (anchor exact,
                otherwise abort) and with the fixed β_new in all folds.
  --replace-q397 q397m3  REPLACEMENT instead of the sixth channel q397 (only --line best --m3 m3x3_full, not with --beta-novel):
                q397m3 = M3 behind the notes of the 397B, pure ranking of the five fold shards (eda/c117_q397m3_deploy.py oof;
                bit-identical to the channel of the c115 evaluation) in place of q397, floor 0.05; rank_only entries for M3 AND
                q397m3 (make_submission --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test), no proto entry. Field cv =
                cross value, required to equal the replacement value of c115 exactly (7,494/9,000), the six-channel recipe with
                q397 in the same run exactly the anchor (7,463/9,000). CHECK: the same path with q397 (without the switch)
                reproduces stack_s029_best_m3x3.json character for character. -> artifacts/stack_s029_best_q397m3.json
  --arith-beta <β> | --arith-beta-select  [--stem <name>]  (only with --replace-q397): arithmetic share β instead of 0.2 — fixed
                or by the rule argmin log loss of the blend BEFORE the last step on all 1,500 rows (θ from the same rows, grid
                0.1…0.8; the same rule as the nested choice in the model sweep). Check: the path with β 0.2 reproduces
                stack_s029_best_q397m3.json character for character (with --arith-beta 0.2 it is exactly that path). The fields
                arith_beta, cv, cv_note, cross, insample, note and arith_beta_rule change; θ, floors, rank_only and checksums stay.
                    python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 --arith-beta-select --stem s032_stage2 --save
                -> artifacts/stack_s032_stage2.json (build script tools/builds/s032_stage2.sh)
  readable     REL-01 at rank 1 also for test rows for which pr27 (runs/pr27te_rp/proto_te/proto, last valid record) reports
                "readable = no" — list derived/blank/test_rows_readable.npy = blank images ∪ these rows (make_submission
                --blank-rows, no code change); cross value on the training rows with the pr27 answers d1.

TEST SIDE (`test-status`; aggregates only): answer file and meta file present, table test, all 3,498 rows; valid answers
(labels_to_probs) ≥ 97 % and equal to the field valid; NONE at the top (valid rows) within the band fixed in advance, training ±
0.06 (training = Tinker answers, valid rows; a malfunction guard, never used to choose between variants); model Qwen3.5-397B
(summary.json) and settings of the configuration (proto_gen, split test, all rows, greedy, 2,500/1,500 tokens, images
1,024/448) as in the Tinker measurement — with --placeholder (code check with a foreign test side) model and settings are not
checked.

Output only aggregates; logs under artifacts/eda/private/q397_deploy/c110_*.json. Test files are only counted, never inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import proto as PR  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C107 = _load("c110_c107", COMP / "eda" / "c107_q397_full.py")  # recipe, anchors, channel and comparison of the c107 decision
MS, A2, C21 = C107.MS, C107.A2, C107.C21  # tools/make_submission.py, eda/a2_probe_stack.py, eda/c21_margin_align.py

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "q397_deploy"
CH, FLOOR = "q397", 0.05
TINKER_REL = C107.LABELS_REL  # derived/open_teacher/qwen35-397b/labels.jsonl (relative to the data root)
TINKER_SHA256 = "6935614248a22da250054dd14b180fc5875832ddba7afc49c4b3da0337503280"  # state of the c107 decision (c107_result.json)
SRC_RUN, SRC_JOB, SRC_TAG = "q397_tinker", "proto_t", "proto"
SRC_SPEC = f"runs/{SRC_RUN}/{SRC_JOB}/{SRC_TAG}"
M3_DEFAULT, M3_CHOICES = "m3x2_full_cv01", ("m3x2_full_cv01", "m3x3_full")
TEST_RUNS = ("runs/q397te_rp/proto_te/proto", "runs/q397te8_rp/proto_te/proto")  # 4×H200 (TP 4) or 8×H100 (TP 8)
TEST_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
# cross values of the c107 decision in units of 1/6 per row
CROSS_UNITS = {"best": 7453, "safe": 7386}  # 0.8281111 / 0.8206667 on 9,000
# anchors of the nested β_new choice: 0.8302222 / 0.8218889
NESTED_BN_UNITS = {"best": 7472, "safe": 7397}
GRID_BN = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
NONE_BAND = 0.06
MIN_VALID = 0.97
# switch --replace-q397: replacement channel q397m3 = M3 behind the 397B notes (pure ranking of the five fold shards,
# eda/c117_q397m3_deploy.py oof), primary line with m3x3_full only (six-channel recipe); cross values of the c115 evaluation in 1/6
REPLACE_CHOICES = ("q397m3",)
REPLACE_SOURCE = {"q397m3": "+".join(f"runs/q397m3_oof_rp/q397m3_rank_f{k}/bench" for k in range(5))}
REPLACE_C115 = ART / "eda" / "private" / "q397m3_eval" / "c115_q397m3_eval.json"
REPLACE_UNITS = {"six_channel": 7463, "q397m3": 7494}  # 0.8292222 -> 0.8326667
LINES = {
    "best": {"spec": C107.LINE_BEST, "src_stack": C107.STACK_BEST, "anchor": "earlier primary-line recipe",
             "ref_sub": ("artifacts/submissions/sub_cv01_best_l2.csv", "1c8c4bb12122e3feeef07d4488e79339635078ab667c29a57192560dcba4e346")},
    "safe": {"spec": C107.LINE_SAFE, "src_stack": C107.STACK_SAFE, "anchor": "earlier inductive-line recipe",
             "ref_sub": ("artifacts/submissions/sub_v1r_zs27t_think18sb_pr27xm_m3x2_ar_ia_t4.csv",
                         "80ea14fe28449a4b7be66bb0eeb8622f4870a4629105ebe6dc7a3859c36d5557")},
}
PR27_TEST = "artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_labels.jsonl"
PR27_TRAIN = ("artifacts/runs/pr27_rp/pr27_rp/proto_a/proto_labels.jsonl", "artifacts/runs/pr27_rp/pr27_rp/proto_b/proto_labels.jsonl")
BLANK_TEST, BLANK_TRAIN, READABLE_TEST = "derived/blank/test_rows.npy", "derived/blank/train_rows.npy", "derived/blank/test_rows_readable.npy"


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    return C107.sha256_file(Path(path))


def rel(path: Path) -> str:
    """Path relative to the competition folder (absolute if outside — e.g. a check folder)."""
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).astype("<i8").tobytes()).hexdigest()


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=C107.json_default) + "\n"


def stem_of(line: str, m3: str, beta_novel: bool, replace: str | None = None) -> str:
    if replace:  # switch --replace-q397: own name, the versions with q397 stay untouched
        return f"s029_{line}_{replace}"
    return f"s029_{line}" + ("_m3x3" if m3 == "m3x3_full" else "") + ("_bn" if beta_novel else "")


# ------------------------------------------------------------------------------------------------ context

def setup() -> dict:
    root = C.find_data_root()
    ctx = C107.context(root)  # training table, folds, classes, groups
    train = C.load_table(root, "train")
    ctx.update(root=root, train=train, qid=train["question_id"].to_numpy(), topic=train["topic_name"].to_numpy())
    return ctx


def tinker_channel(ctx: dict) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Channel q397 from the Tinker file exactly as in c107 (own conversion = labels_to_probs, rows without an answer uniform)."""
    raw = (ctx["root"] / TINKER_REL).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TINKER_SHA256:
        abort(f"{TINKER_REL}: checksum differs from the state of the c107 decision")
    complete, records, info = C107.parse_snapshot(raw)
    if info["partial_tail"]:
        abort(f"{TINKER_REL}: last line incomplete")
    C107.validate_records(records, ctx["n"])
    Ps, has = C107.proto_channel(records, ctx["n"], ctx["classes"], ctx["seen"], "stated")
    return C107.fill_missing(Ps, has, ctx["seen"]), has, records, info


# ------------------------------------------------------------------------------------------------ source / verify-oof

def cmd_source(a) -> int:
    ctx = setup()
    n, classes, seen = ctx["n"], ctx["classes"], ctx["seen"]
    _, has, records, info = tinker_channel(ctx)
    cov = C107.coverage(records, n, classes)
    if cov["rows_missing"] or cov["rows_pending_finish"]:
        abort(f"Tinker file incomplete: {cov}")
    d = ART / "runs" / SRC_RUN / SRC_RUN / SRC_JOB
    d.mkdir(parents=True, exist_ok=True)
    dst, meta_dst = d / f"{SRC_TAG}_labels.jsonl", d / f"{SRC_TAG}_meta.json"
    if dst.exists():
        if sha256_file(dst) != TINKER_SHA256:
            abort(f"{rel(dst)} exists and is NOT the Tinker file — not overwritten")
    else:
        shutil.copyfile(ctx["root"] / TINKER_REL, dst)
        if sha256_file(dst) != TINKER_SHA256:
            abort("copy differs from the source")
    P, h = PR.labels_to_probs(dst, n, classes, seen)
    if not np.array_equal(h, has):
        abort("labels_to_probs and c107 count different valid rows")
    first = [r for r in records if r.get("phase") == "first"]
    last_valid = {}
    for r in records:
        if r.get("ok", True) and r.get("root_code") in set(classes):
            last_valid[int(r["row"])] = r
    meta = {"table": "train", "rows": n, "answered": cov["rows_with_record"], "valid": int(h.sum()),
            "records": info["records"], "records_first": len(first), "records_finish": sum(r.get("phase") == "finish" for r in records),
            "valid_first": sum(r.get("phase") == "first" for r in last_valid.values()),
            "ends_first": dict(Counter(str(r.get("stop")) for r in first)),
            "out_tokens_first_median": float(np.median([r.get("out_tokens", 0) for r in first])),
            "in_tokens_first_median": float(np.median([r.get("in_tokens", 0) for r in first])),
            "share_none_top": float(np.mean(P[h].argmax(1) == 0)), "distinct_top": int(len(set(P[h].argmax(1).tolist()))),
            "source": f"data/{TINKER_REL}", "source_sha256": TINKER_SHA256,
            "model": "Qwen3.5-397B-A17B (Tinker-API, tools/open_teacher.py)",
            "origin": "860 rows of the stratified sample (design_stage2) plus the remaining 640 rows; teacher protocol without a labelled "
                      "row, greedy, thinking mode up to 2,500 tokens, finishing pass up to 1,500 without thinking, images 1,024/448 px (JPEG q75)",
            "created_by": "eda/c110_q397_deploy.py source",
            "note": "byte-identical copy of the Tinker answers as the source job for eda/a8_proto_oof.py (runs/q397_tinker/proto_t/proto); "
                    "training rows only"}
    text = to_json(meta)
    if meta_dst.exists() and meta_dst.read_text() != text:
        abort(f"{rel(meta_dst)} exists with different content — not overwritten")
    meta_dst.write_text(text)
    print(f"source job {SRC_SPEC}: {rel(dst)} (SHA-256 {TINKER_SHA256[:16]}…, {info['records']} records) and "
          f"{meta_dst.name}: {n} rows, valid {int(h.sum())}, NONE at the top {meta['share_none_top']:.3f}")
    print(f"next: python -I eda/a8_proto_oof.py {CH} {SRC_SPEC}")
    return 0


def verify_oof(ctx: dict, quiet: bool = False) -> dict:
    """oof_q397.npy (a8 over the source job) bit-identical to the channel of c107; meta file matches (source, no debiasing, no mean channel)."""
    ref, has, _, _ = tinker_channel(ctx)
    d = ART / "runs" / SRC_RUN / SRC_RUN / SRC_JOB
    if sha256_file(d / f"{SRC_TAG}_labels.jsonl") != TINKER_SHA256:
        abort(f"source job {SRC_SPEC}: the answer file is not the Tinker file")
    oof = np.load(ART / f"oof_{CH}.npy")
    meta = json.loads((ART / f"meta_{CH}.json").read_text())
    bad = []
    if oof.shape != ref.shape or not np.array_equal(oof, ref):
        bad.append(f"values differ (largest deviation {float(np.abs(oof - ref).max()) if oof.shape == ref.shape else 'Form'})")
    if meta.get("sources") != [SRC_SPEC]:
        bad.append(f"sources {meta.get('sources')}")
    if meta.get("rank_mean") or meta.get("class_weights") is not None or meta.get("rank"):
        bad.append("meta file describes a mean channel, debiasing or a ranking")
    if int(meta.get("rows_without_answer", -1)) != int((~has).sum()) or meta.get("neutral_rows") is not None:
        bad.append(f"rows without an answer {meta.get('rows_without_answer')} / gate {meta.get('neutral_rows')}")
    if meta.get("folds_hash") != C.FOLDS_HASH:
        bad.append(f"folds_hash {meta.get('folds_hash')}")
    if bad:
        abort("oof_q397 does not match the channel of the c107 decision: " + "; ".join(bad))
    out = {"oof_sha256": sha256_file(ART / f"oof_{CH}.npy"), "meta_sha256": sha256_file(ART / f"meta_{CH}.json"), "rows_valid": int(has.sum()),
           "cv_score": meta.get("cv_score"), "bit_identical_c107": True}
    if not quiet:
        print(f"oof_{CH}.npy bit-identical to the channel of c107 (Tinker file {TINKER_SHA256[:16]}…; {int(has.sum())} valid rows, the rest uniform); "
              f"channel alone MAP@3 {meta.get('cv_score'):.4f}; SHA-256 {out['oof_sha256'][:16]}…")
    return out


def cmd_verify_oof(a) -> int:
    verify_oof(setup())
    return 0


# ------------------------------------------------------------------------------------------------ recipe and novelty

def line_specs(line: str, m3: str) -> tuple[dict, dict]:
    """(recipe of the line with the chosen M3 channel, the same plus q397 appended with floor 0.05)."""
    base = LINES[line]["spec"]

    def ren(nm: str) -> str:
        return m3 if nm == M3_DEFAULT else nm

    spec = {**base, "names": [ren(nm) for nm in base["names"]], "floors": {ren(k): float(v) for k, v in base["floors"].items()}}
    return spec, C107.with_channel(spec, CH, FLOOR)


def load_mats(ctx: dict, m3: str) -> tuple[dict, dict, dict]:
    """Base OOFs (checksums c107), q397 (bit-identical c107) and, if applicable, m3x3_full (three draws on all rows)."""
    mats, sha16 = C107.load_base(ctx["n"], ctx["K"])
    verify_oof(ctx, quiet=True)
    mats[CH] = np.load(ART / f"oof_{CH}.npy")
    sha16[CH] = sha256_file(ART / f"oof_{CH}.npy")[:16]
    metas = {M3_DEFAULT: json.loads((ART / f"meta_{M3_DEFAULT}.json").read_text())}
    if m3 != M3_DEFAULT:
        f, mf = ART / f"oof_{m3}.npy", ART / f"meta_{m3}.json"
        if not f.is_file() or not mf.is_file():
            abort(f"{f.name} or {mf.name} missing — run tools/builds/s029_m3x3.sh or eda/a9_m3_draws_oof.py first", 3)
        P, meta = np.load(f), json.loads(mf.read_text())
        rk = meta.get("rank") or {}
        if P.shape != (ctx["n"], ctx["K"]) or not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-6):
            abort(f"{f.name}: shape {P.shape}, not finite or not normalised")
        if len(rk.get("draws") or []) != 3 or rk.get("rows_by_draws") != {"3": ctx["n"]}:
            abort(f"{mf.name}: not three draws on all {ctx['n']} rows ({rk.get('rows_by_draws')})")
        mats[m3], sha16[m3], metas[m3] = P, sha256_file(f)[:16], meta
    return mats, sha16, metas


def novelty(ctx: dict, rows: np.ndarray, exclude: set | tuple = ()) -> np.ndarray:
    """Novelty per (row, class) for the training rows `rows`: reference = rows outside the fold of the row and outside `exclude`
    (make_submission.code_novelty)."""
    rows = np.asarray(rows)
    folds, y, K = ctx["folds"], ctx["y"], ctx["K"]
    out = np.zeros((len(rows), K), dtype=bool)
    fr = folds[rows]
    for f in np.unique(fr):
        sel = fr == f
        ref = (folds != f) & ~np.isin(folds, list(exclude))
        out[sel] = MS.code_novelty(ctx["qid"][ref], ctx["topic"][ref], y[ref], ctx["qid"][rows[sel]], ctx["topic"][rows[sel]], K)
    return out


def mixture(th, beta: float, logs: list[np.ndarray], rows: np.ndarray, seen: np.ndarray, bn: float | None = None,
            nov: np.ndarray | None = None) -> np.ndarray:
    """make_submission.mix_channels on `rows` (with β_new if bn is set)."""
    st = {"theta": [float(v) for v in th], "arith_beta": float(beta)}
    if bn is not None:
        st["arith_beta_novel"] = float(bn)
    return MS.mix_channels(st, [lg[rows] for lg in logs], None, seen, novel=nov)


def mean_ll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def choose_bn(th, beta: float, logs, rows, y, seen, nov, grid=GRID_BN) -> tuple[float, dict]:
    """Rule for β_new: the value with the smallest log loss of the blend before the last step on the fit rows."""
    best, bb, lls = None, None, {}
    for bn in grid:
        ll = mean_ll(mixture(th, beta, logs, rows, seen, bn, nov), y[rows])
        lls[f"{bn:g}"] = ll
        if best is None or ll < best - 1e-12:
            best, bb = ll, bn
    return float(bb), lls


def cross_bn(ctx: dict, mats: dict, spec: dict, fixed: float | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Cross-fit as in c107.cross_fit, but with β_new per fold: chosen nested (fixed None; novelty of the fit rows without the
    evaluated fold) or fixed. With fixed = β this is c107.cross_fit itself (positive control).
    -> (after the last step, before it, chosen β_new per fold)"""
    y, folds, seen, n, K = ctx["y"], ctx["folds"], ctx["seen"], ctx["n"], ctx["K"]
    names, floors = spec["names"], spec["floors"]
    L = C107.logs_of(mats, names, floors, seen, False)
    Lm = C107.logs_of(mats, names, floors, seen, True)
    ti = names.index(spec["think"]) if spec.get("think") in names else None
    beta = float(spec["beta"])
    Q, P, chosen = np.zeros((n, K)), np.zeros((n, K)), []
    for k in range(C107.N_FOLDS):
        F, E = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        th = np.asarray(A2.fit_mix([lg[F] for lg in L], None, y[F]), dtype=float).copy()
        if ti is not None:
            th[ti] *= float(spec["factor"])
        nF, nE = novelty(ctx, F, {k}), novelty(ctx, E)
        bb = choose_bn(th, beta, L, F, y, seen, nF)[0] if fixed is None else float(fixed)
        chosen.append(bb)
        PE = mixture(th, beta, L, E, seen, bb, nE)
        P[E] = PE
        tgt = C107.target_of(y, F, K)
        if spec["last"] == "trans":
            Q[E] = C107.apply_last(PE, C21.fit_weights(PE, tgt), seen)
        else:
            PF = mixture(th, beta, Lm, F, seen, bb, nF)
            Q[E] = C107.apply_last(PE, C21.fit_weights(PF, tgt), seen)
    return Q, P, chosen


# ------------------------------------------------------------------------------------------------ stack

def fit_all(mats: dict, spec: dict, y: np.ndarray, seen: np.ndarray, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list]:
    """Weights on log loss from `rows` (default all) as in c99.fit_theta -> (deployed, before the factor, logs without masking)."""
    L = C107.logs_of(mats, spec["names"], spec["floors"], seen, False)
    rows = np.arange(len(y)) if rows is None else np.asarray(rows)
    th0 = np.asarray(A2.fit_mix([lg[rows] for lg in L], None, y[rows]), dtype=float)
    th = th0.copy()
    if spec.get("think") in spec["names"]:
        th[spec["names"].index(spec["think"])] *= float(spec["factor"])
    return th, th0, L


def rank_only_entry(m3: str, meta: dict, sha16: str, n: int) -> dict:
    rk = meta.get("rank") or {}
    ent = {"floor": FLOOR, "rank_source": rk.get("source"), "oof_sha256": sha16, "rows": int(n)}
    if rk.get("draws"):
        ent.update(draws=len(rk["draws"]), rows_all_draws_sha256=rk.get("rows_all_draws_sha256"))
    return ent


def summarize_cross(c: dict) -> dict:
    keep = ("map_new", "map_base", "delta", "delta_exact", "se_q", "z_q", "ci95_q", "p_le_0", "p_below_m0002", "se_rows", "per_fold",
            "per_fold_units", "folds_ge0", "folds_gt0", "rows_better", "rows_worse", "ll_new", "ll_base", "ll_gain", "ll_se_q", "ll_z_q")
    out = {k: c[k] for k in keep if k in c}
    out["groups"] = {g: {kk: round(float(v), 6) if isinstance(v, float) else v for kk, v in d.items()} for g, d in c["groups"].items()}
    return out


def build(line: str, m3: str, beta_novel: bool) -> tuple[dict, dict]:
    """Blend file (without timestamps) and log of one line."""
    t0 = time.time()
    ctx = setup()
    y, folds, seen, n, K = ctx["y"], ctx["folds"], ctx["seen"], ctx["n"], ctx["K"]
    stacks = C107.check_stacks()  # checksums and recipe constants of the earlier blend files of the primary and inductive lines
    mats, sha16, metas = load_mats(ctx, m3)
    spec5, spec6 = line_specs(line, m3)
    rep: dict = {"line": line, "m3": m3, "switch_beta_novel": beta_novel, "names": spec6["names"]}
    draws = C107.question_draws(ctx["q_code"])

    # anchors without q397 (with m3x2_full_cv01, as submitted) exact
    anc, fit_b, fit_s = C107.anchors(ctx, mats, stacks)
    rep["anchors"] = anc
    # check of the fitting path: without q397 it reproduces θ of the submitted blend file bit-identically
    if line == "best":
        ref = json.loads((ART / C107.STACK_BEST[0]).read_text())
        th, th0, _ = fit_all(mats, line_specs("best", M3_DEFAULT)[0], y, seen)
        ok = np.array_equal(th, np.asarray(ref["theta"], dtype=float)) and np.array_equal(th0, np.asarray(ref["base"]["theta"], dtype=float))
        label = "the earlier primary-line blend file (stack_cv01_best_l2.json, 1,500 rows)"
    else:
        ref = json.loads((ART / C107.STACK_SAFE[0]).read_text())
        m3x2 = ART / "oof_m3x2.npy"
        if sha256_file(m3x2)[:16] != ref["oof_sha256"]["m3x2"]:
            abort("oof_m3x2.npy differs from the earlier inductive-line blend file")
        mats_s = {**mats, "m3x2": np.load(m3x2)}
        spec_s = {**LINES["safe"]["spec"], "names": ref["channels"], "floors": {k: float(v) for k, v in ref["floor"].items()}}
        th, th0, _ = fit_all(mats_s, spec_s, y, seen, np.asarray(ref["fit_rows"]))
        ok = np.array_equal(th, np.asarray(ref["theta"], dtype=float))
        label = "the earlier inductive-line blend file (stack_v1r_zs27t_think18sb_pr27xm_m3x2_ar.json, 600 rows with m3x2)"
    if not ok:
        abort(f"check: the same fitting path without q397 does not reproduce θ of {label} bit-identically")
    rep["check_theta"] = f"bit-identical to {label}"

    # cross value with q397 (and without, with the same procedure and the same M3 channel)
    fit_n = C107.cross_fit(y, folds, seen, mats, spec6)
    fit_0 = C107.cross_fit(y, folds, seen, mats, spec5)
    r_n, r_0 = C107.rows_map(y, fit_n["Q"], seen), C107.rows_map(y, fit_0["Q"], seen)
    ll_n, ll_0 = C107.rows_ll(y, fit_n["Q"]), C107.rows_ll(y, fit_0["Q"])
    units = int(C107.units(r_n).sum())
    if m3 == M3_DEFAULT and units != CROSS_UNITS[line]:
        abort(f"cross value {units}/{6 * n} differs from the value of the c107 decision ({CROSS_UNITS[line]}/{6 * n})")
    cmp6 = C107.compare(r_n, r_0, ctx, draws, ll_n, ll_0)
    th_f = np.asarray(fit_n["theta"])
    # comparison values for the in-sample check: cross WITHOUT the last step and (primary) with the alignment as ONE set of 1,500
    # rows — the alignment per fold on 300 rows contributes something itself; the in-sample procedure aligns as one set
    cross_pre = float(C107.rows_map(y, fit_n["P"], seen).mean())
    cross_set = float(C107.rows_map(y, C107.set_align(fit_n["P"], y, seen), seen).mean()) if line == "best" else None
    rep["cross"] = {"map3": float(r_n.mean()), "units": units, "logloss": float(ll_n.mean()),
                    "map3_pre_last_step": cross_pre, "map3_one_set": cross_set, "logloss_pre_last_step": float(C107.rows_ll(y, fit_n["P"]).mean()),
                    "per_fold": [float(r_n[folds == k].mean()) for k in range(C107.N_FOLDS)],
                    "theta_per_fold": th_f.tolist(), "theta_mean": th_f.mean(0).tolist(), "share_q397": (th_f[:, -1] / th_f.sum(1)).tolist(),
                    "without_q397": {"map3": float(r_0.mean()), "logloss": float(ll_0.mean())}, "delta_q397": summarize_cross(cmp6)}
    if m3 != M3_DEFAULT:  # against the same line with m3x2_full_cv01 + q397 (same rows, same procedure)
        spec6_ref = line_specs(line, M3_DEFAULT)[1]
        fit_r = C107.cross_fit(y, folds, seen, mats, spec6_ref)
        r_r, ll_r = C107.rows_map(y, fit_r["Q"], seen), C107.rows_ll(y, fit_r["Q"])
        rep["cross"]["delta_vs_m3x2_q397"] = summarize_cross(C107.compare(r_n, r_r, ctx, draws, ll_n, ll_r))

    # deployed weights (all 1,500 rows) and blend file
    th, th0, L = fit_all(mats, spec6, y, seen)
    allr = np.arange(n)
    floors = {k: float(v) for k, v in spec6["floors"].items() if v}
    src = json.loads((ART / LINES[line]["src_stack"][0]).read_text())
    ro = rank_only_entry(m3, metas[m3], sha16[m3], n)
    if line == "best" and m3 == M3_DEFAULT and ro != src["rank_only"][M3_DEFAULT]:
        abort(f"rank_only entry {ro} differs from stack_cv01_best_l2.json")
    stack = {"channels": list(spec6["names"]), "none": [], "probes": {}, "theta": [float(v) for v in th], "floor": floors,
             "rows": int(n), "folds": list(range(C107.N_FOLDS)), "fit_rows": [int(i) for i in allr], "fit_rows_sha256": rows_sha256(allr),
             "base": {"channels": list(spec6["names"]), "theta": [float(v) for v in th0]},
             "scaled": {spec6["think"]: float(spec6["factor"])} if spec6.get("think") else {},
             "rank_only": {m3: ro},
             "proto": {CH: {"floor": FLOOR, "source": SRC_SPEC, "tinker_sha256": TINKER_SHA256, "oof_sha256": sha16[CH],
                            "rows_valid": int(verify_oof(ctx, quiet=True)["rows_valid"]), "test_runs": list(TEST_RUNS),
                            # text fields of the blend file (definition, notes, rules): `stack --verify` compares the recomputed file character
                            # for character with the stored one
                            "definition": "named codes with their stated probabilities, the rest spread evenly over the classes with a training "
                                          "example, rows without a valid answer uniform (vlm.proto.labels_to_probs, eda/a8_proto_oof.py; "
                                          "test rows: make_submission --proto q397=…); no gate, no debiasing"}},
             "arith_beta": float(spec6["beta"])}
    nov_all = None
    if beta_novel:
        nov_all = novelty(ctx, allr)
        if not np.array_equal(nov_all, MS.train_novelty(ctx["train"], folds, ctx["classes"])):
            abort("novelty of the training rows: c110.novelty and make_submission.train_novelty differ")
        bn, lls = choose_bn(th, float(spec6["beta"]), L, allr, y, seen, nov_all)
        stack["arith_beta_novel"] = bn
        stack["arith_beta_novel_rule"] = {"grid": list(GRID_BN), "chosen": bn, "logloss_pre_last_step": lls,
                                          "rule": "smallest log loss of the blend before the last step on all 1,500 rows "
                                                  "(weights of this file; novelty against the other folds)",
                                          "novelty": "make_submission.code_novelty: the code was never the root among the reference rows of the "
                                                     "same task (question_id, same topic_name); NONE is never novel"}
        Qn, _, ch_n = cross_bn(ctx, mats, spec6)
        rn = C107.rows_map(y, Qn, seen)
        if m3 == M3_DEFAULT and int(C107.units(rn).sum()) != NESTED_BN_UNITS[line]:
            abort(f"nested β_new choice {int(C107.units(rn).sum())}/{6 * n} does not reproduce the anchor ({NESTED_BN_UNITS[line]})")
        Qc, _, _ = cross_bn(ctx, mats, spec6, fixed=float(spec6["beta"]))
        if not np.array_equal(C107.units(C107.rows_map(y, Qc, seen)), C107.units(r_n)):
            abort("positive control: β_new = β does not reproduce c107.cross_fit row by row")
        Qf, Pf_cross, _ = cross_bn(ctx, mats, spec6, fixed=bn)
        rf = C107.rows_map(y, Qf, seen)
        rep["beta_novel"] = {"chosen_all_rows": bn, "logloss_grid_all_rows": lls,
                             "cross_nested": {"map3": float(rn.mean()), "chosen_per_fold": ch_n,
                                              "delta": summarize_cross(C107.compare(rn, r_n, ctx, draws, C107.rows_ll(y, Qn), ll_n))},
                             "cross_fixed": {"map3": float(rf.mean()), "beta_novel": bn,
                                             "map3_pre_last_step": float(C107.rows_map(y, Pf_cross, seen).mean()),
                                             "logloss_pre_last_step": float(C107.rows_ll(y, Pf_cross).mean()),
                                             "map3_one_set": float(C107.rows_map(y, C107.set_align(Pf_cross, y, seen), seen).mean()) if line == "best" else None,
                                             "delta": summarize_cross(C107.compare(rf, r_n, ctx, draws, C107.rows_ll(y, Qf), ll_n))},
                             "share_novel_pairs_train": float(nov_all[:, seen].mean())}
    cv = rep["beta_novel"]["cross_fixed"]["map3"] if beta_novel else rep["cross"]["map3"]
    stack.update({"cv": cv, "cv_rows": int(n),
                  "cv_note": ("cross-fit per fold on 1,500 rows WITH the last step (weights per fold from the other four, eda/c107_q397_full.py "
                              "cross_fit" + (f"; β_new {bn:g} fixed in all folds" if beta_novel else "") + f"); without q397 {rep['cross']['without_q397']['map3']:.7f}; "
                              "base OOFs not fully nested in the meta fold; weights θ from the Tinker OOF of the 397B"),
                  "cross": {"map3": rep["cross"]["map3"], "logloss": rep["cross"]["logloss"], "per_fold": rep["cross"]["per_fold"],
                            "map3_pre_last_step": rep["cross"]["map3_pre_last_step"], "map3_one_set": rep["cross"]["map3_one_set"],
                            "without_q397": rep["cross"]["without_q397"]["map3"],
                            "delta_q397": {k: rep["cross"]["delta_q397"][k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")},
                            **({"beta_novel_nested": rep["beta_novel"]["cross_nested"]["map3"], "beta_novel_fixed": rep["beta_novel"]["cross_fixed"]["map3"]}
                               if beta_novel else {})},
                  "oof_sha256": {nm: sha16[nm] for nm in spec6["names"]},
                  "last_step": src["last_step"],
                  "note": (f"eda/c110_q397_deploy.py stack --line {line} --m3 {m3}" + (" --beta-novel" if beta_novel else "")
                           + f": {LINES[line]['anchor']} + q397 (sixth channel, floor {FLOOR:g}); weights of ALL channels "
                           f"on log loss from all {n} rows (a2.fit_mix" + (f", then {spec6['think']} × {spec6['factor']:g}" if spec6.get("think") else "")
                           + "); build script tools/builds/s029_q397.sh")})
    if line == "safe":  # inductive class weights as for the earlier inductive-line blend file (make_submission recomputes them)
        counts = np.bincount(y, minlength=K)
        target = np.where(counts >= C107.ALIGN_MIN, counts, 0) / n
        Pf = MS.oof_mixture(stack, seen, ART, n_rows=n, novel=nov_all)
        w = C21.fit_weights(Pf[MS.stack_fit_rows(stack, n)], target)
        used = target > 0
        stack["inductive_check"] = {"strength": C107.ALIGN, "min_rows": C107.ALIGN_MIN,
                                    "weights_sha256": hashlib.sha256(np.ascontiguousarray(w, dtype=np.float64).tobytes()).hexdigest()[:16],
                                    "fit_rows": int(n), "target_rows": int(n), "classes": int(used.sum()),
                                    "weights_q10_q50_q90": [round(float(q), 3) for q in np.quantile(w[used], [0.1, 0.5, 0.9])],
                                    "note": "class weights = c21.fit_weights(make_submission.oof_mixture(blend)[fit_rows], frequencies of all "
                                            f"training rows, classes with at least 3); blend with q397 and the arithmetic share β = {spec6['beta']:g}"
                                            + (f", β_new {stack['arith_beta_novel']:g}" if beta_novel else "")}

    # check make_submission.oof_mixture against the own blend; in-sample with the last step
    P_ms = MS.oof_mixture(stack, seen, ART, n_rows=n, novel=nov_all)
    Lm = C107.logs_of(mats, spec6["names"], spec6["floors"], seen, True)
    P_mask = MS.mix_channels(stack, Lm, None, seen, novel=nov_all)
    P_test_way = MS.mix_channels(stack, L, None, seen, novel=nov_all)
    d_mask, d_way = float(np.abs(P_ms - P_mask).max()), float(np.abs(P_ms - P_test_way).max())
    if d_mask != 0.0 or d_way > 1e-9:
        abort(f"check oof_mixture: deviation {d_mask:.1e} (mask first) / {d_way:.1e} (test path)")
    target_all = C107.target_of(y, allr, K)
    if line == "best":
        Q_in = C107.apply_last(P_test_way, C21.fit_weights(P_test_way, target_all), seen)
    else:
        Q_in = C107.apply_last(P_test_way, C21.fit_weights(P_ms[MS.stack_fit_rows(stack, n)], target_all), seen)
    r_in = C107.rows_map(y, Q_in, seen)
    pre_in = float(C107.rows_map(y, P_test_way, seen).mean())
    cref = rep["beta_novel"]["cross_fixed"] if beta_novel else rep["cross"]  # cross value of the same recipe (with β_new, if set)
    like = cref["map3_one_set"] if line == "best" else cref["map3"]  # same procedure for the last step
    ll_pre_in = float(C107.rows_ll(y, P_test_way).mean())
    rep["insample"] = {"map3": float(r_in.mean()), "logloss": float(C107.rows_ll(y, Q_in).mean()), "map3_pre_last_step": pre_in,
                       "logloss_pre_last_step": ll_pre_in, "vs_cross_same_last_step": float(r_in.mean()) - like,
                       "vs_cross_pre_last_step": pre_in - cref["map3_pre_last_step"],
                       "vs_cross_logloss_pre_last_step": ll_pre_in - cref["logloss_pre_last_step"],
                       # the fit is on log loss before the last step: there, in-sample must be better; MAP@3 (not fitted) may lie a few rows'
                       # noise below — gross-error guard: at most 0.003 below the cross value of the same form
                       "plausible": bool(ll_pre_in <= cref["logloss_pre_last_step"] and pre_in >= cref["map3_pre_last_step"] - 0.003
                                         and float(r_in.mean()) >= like - 0.003),
                       "note": "OPTIMISTIC: weights and last step from the same 1,500 rows; primary line aligned as ONE set — "
                               "compared with the cross-fit of the same form (one set) or without the last step"}
    stack["insample"] = {"map3": rep["insample"]["map3"], "map3_pre_last_step": rep["insample"]["map3_pre_last_step"],
                         "logloss": rep["insample"]["logloss"]}
    rep["check_oof_mixture"] = {"max_abs_mask_first": d_mask, "max_abs_test_way": d_way}
    rep["theta"] = {nm: float(v) for nm, v in zip(spec6["names"], th)}
    rep["theta_before_factor"] = {nm: float(v) for nm, v in zip(spec6["names"], th0)}
    rep["share_theta"] = {nm: float(v / th.sum()) for nm, v in zip(spec6["names"], th)}
    rep["seconds"] = round(time.time() - t0, 1)
    if not C107.sane(Q_in, seen) or not C107.sane(P_ms, seen):
        abort("predictions not normalised or with mass on classes without a training example")
    return stack, rep


def cmd_stack(a) -> int:
    if getattr(a, "replace_q397", None):  # switch --replace-q397 (default off): own path, the path without the switch stays unchanged
        return cmd_stack_replaced(a)
    if getattr(a, "arith_beta", None) is not None or getattr(a, "arith_beta_select", False) or getattr(a, "stem", None):
        abort("--arith-beta / --arith-beta-select / --stem only with --replace-q397", 2)
    stem = stem_of(a.line, a.m3, a.beta_novel)
    path = ART / f"stack_{stem}.json"
    stack, rep = build(a.line, a.m3, a.beta_novel)
    text = to_json(stack)
    c = rep["cross"]
    print(f"{stem}: channels {stack['channels']}")
    print("   θ (all 1,500 rows" + (f", {rep['names'][2]} × 1.4" if a.line == "best" else "") + "): "
          + " · ".join(f"{nm} {v:.4f}" for nm, v in rep["theta"].items()) + f" | share q397 {rep['share_theta'][CH]:.3f}")
    print(f"   {rep['check_theta']}; anchors without q397: best {rep['anchors']['best']['map3']:.7f}, inductive {rep['anchors']['safe']['map3']:.7f}")
    d = c["delta_q397"]
    print(f"   CROSS with q397 {c['map3']:.5f} ({c['units']}/{6 * len(stack['fit_rows'])}; without q397 {c['without_q397']['map3']:.5f}): "
          f"Δ {d['delta']:+.5f} ± {d['se_q']:.5f} over tasks (z {d['z_q']:+.2f}; {d['folds_gt0']}/5 folds > 0), log loss "
          f"{c['without_q397']['logloss']:.4f} → {c['logloss']:.4f} (z {d['ll_z_q']:+.2f}); θ q397 per fold "
          + " ".join(f"{x[-1]:.3f}" for x in c["theta_per_fold"]))
    if "delta_vs_m3x2_q397" in c:
        dv = c["delta_vs_m3x2_q397"]
        print(f"   against m3x2_full_cv01 + q397 with the same procedure: Δ {dv['delta']:+.5f} ± {dv['se_q']:.5f} (z {dv['z_q']:+.2f})")
    if a.beta_novel:
        b = rep["beta_novel"]
        print(f"   β_new from all rows {b['chosen_all_rows']:g} (log loss per grid value "
              + " ".join(f"{k} {v:.4f}" for k, v in b["logloss_grid_all_rows"].items()) + ")")
        for lab, key in (("nested", "cross_nested"), (f"fixed {b['chosen_all_rows']:g}", "cross_fixed")):
            dd = b[key]["delta"]
            print(f"   CROSS β_new {lab}: {b[key]['map3']:.5f}, against q397 without β_new Δ {dd['delta']:+.5f} ± {dd['se_q']:.5f} (z {dd['z_q']:+.2f}; "
                  f"{dd['folds_gt0']}/5 > 0; log-loss gain {dd['ll_gain']:+.4f}, z {dd['ll_z_q']:+.2f})"
                  + (f"; chosen per fold {b[key]['chosen_per_fold']}" if key == "cross_nested" else ""))
    ins = rep["insample"]
    like = "cross aligned as one set" if a.line == "best" else "cross"
    print(f"   IN-SAMPLE with the last step {ins['map3']:.5f} (OPTIMISTIC; {like} {ins['map3'] - ins['vs_cross_same_last_step']:.5f}, "
          f"difference {ins['vs_cross_same_last_step']:+.5f}); without the last step {ins['map3_pre_last_step']:.5f} (cross "
          f"{ins['map3_pre_last_step'] - ins['vs_cross_pre_last_step']:.5f}, {ins['vs_cross_pre_last_step']:+.5f}; log loss {ins['logloss_pre_last_step']:.4f} "
          f"against {ins['logloss_pre_last_step'] - ins['vs_cross_logloss_pre_last_step']:.4f}) -> "
          f"{'plausible' if ins['plausible'] else 'CONSPICUOUS (in-sample log loss above the cross value or MAP@3 more than 0.003 below)'}")
    print(f"   oof_mixture against own blend: mask first {rep['check_oof_mixture']['max_abs_mask_first']:.1e}, "
          f"test path {rep['check_oof_mixture']['max_abs_test_way']:.1e}")
    if "inductive_check" in stack:
        ic = stack["inductive_check"]
        print(f"   inductive class weights: {ic['classes']} classes, 10/50/90 % {ic['weights_q10_q50_q90']}, checksum {ic['weights_sha256']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"c110_stack_{stem}.json").write_text(to_json({**rep, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                           "stack_file": str(rel(path)), "stack_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    if a.verify:
        if not path.is_file():
            abort(f"{rel(path)} missing — build it with --save first", 3)
        if path.read_text() != text:
            abort(f"{rel(path)} is not the recomputed blend (channels, weights, cross value or checksums changed)")
        print(f"   VERIFIED: {rel(path)} = recomputation (SHA-256 {sha256_file(path)[:16]}…)")
    if a.save:
        if path.is_file() and path.read_text() != text:
            abort(f"{rel(path)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        path.write_text(text)
        print(f"   saved: {rel(path)} (SHA-256 {sha256_file(path)}) -> {rel(OUT / f'c110_stack_{stem}.json')}")
    return 0


# ------------------------------------------------------------------------------------------------ stack --replace-q397

def replaced_spec(spec: dict, old: str, new: str) -> dict:
    """Recipe with `new` in place of `old` (same order, same floor) — as in eda/c115_q397m3_eval.replaced."""
    if old not in spec["names"]:
        abort(f"{old} is not part of the recipe {spec['names']}")
    return {**spec, "names": [new if nm == old else nm for nm in spec["names"]],
            "floors": {(new if k == old else k): float(v) for k, v in spec["floors"].items()}}


def npy_values_sha256(P: np.ndarray) -> str:
    """SHA-256 of the values as in eda/c115_q397m3_eval.npy_sha256 (np.save as float64 in memory)."""
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(P, dtype=np.float64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_replacement(ctx: dict, name: str) -> tuple[np.ndarray, str, dict, dict]:
    """Replacement channel (eda/c117_q397m3_deploy.py oof): shape, finite, normalised; pure, complete ranking of the five shards
    (rank.source, no named codes, no debiasing, no mean channel) with a note guard (rank.notes); bit-identical to the channel that
    eda/c115_q397m3_eval.py evaluated (real run, verdict CONTINUE for the replacement). -> (P, SHA-256[:16], meta, c115)"""
    f, mf = ART / f"oof_{name}.npy", ART / f"meta_{name}.json"
    if not f.is_file() or not mf.is_file():
        abort(f"{f.name} or {mf.name} missing — run python -I eda/c117_q397m3_deploy.py oof --save first", 3)
    P, meta = np.load(f), json.loads(mf.read_text())
    rk = meta.get("rank") or {}
    bad = []
    if P.shape != (ctx["n"], ctx["K"]) or not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-6):
        bad.append(f"shape {P.shape}, not finite or not normalised")
    if rk.get("source") != REPLACE_SOURCE[name] or int(rk.get("rows_named_only", -1)) != 0 or rk.get("partial"):
        bad.append(f"not a pure, complete ranking of the five shards (rank.source {rk.get('source')!r})")
    if meta.get("rank_mean") or meta.get("class_weights") is not None or not rk.get("notes"):
        bad.append("mean channel, debiasing or without a note guard (rank.notes)")
    if meta.get("folds_hash") != C.FOLDS_HASH:
        bad.append(f"folds_hash {meta.get('folds_hash')}")
    if not REPLACE_C115.is_file():
        abort(f"{rel(REPLACE_C115)} missing — run the evaluation python -I eda/c115_q397m3_eval.py first (verdict of the pre-registered rule)", 3)
    c115 = json.loads(REPLACE_C115.read_text())
    dec = c115.get("decision") or {}
    if c115.get("dry") is not False or dec.get("verdict") != "CONTINUE" or "replacement" not in (dec.get("passed_forms") or []):
        bad.append(f"c115 is not a real evaluation with the verdict CONTINUE for the replacement (dry {c115.get('dry')}, {dec.get('verdict')}, "
                   f"{dec.get('passed_forms')})")
    elif npy_values_sha256(P) != (c115.get("technical") or {}).get("channel_sha256"):
        bad.append("values not bit-identical to the channel that c115 evaluated")
    if bad:
        abort(f"{f.name}: " + "; ".join(bad))
    return P, sha256_file(f)[:16], meta, c115


def build_replaced(line: str, m3: str, replace: str) -> tuple[dict, dict]:
    """Blend file of the six-channel recipe with `replace` instead of q397 at the same position (floor 0.05) and log — procedure as
    in build: weights of ALL channels on log loss from all 1,500 rows (a2.fit_mix, then reasoning factor 1.4), β 0.2, field cv =
    cross value per fold with the last step (c107.cross_fit), required to equal the replacement value of the c115 evaluation
    exactly. CHECK first: the same path with q397 (build without the switch) reproduces stack_s029_best_m3x3.json character for
    character."""
    if line != "best" or m3 != "m3x3_full":
        abort(f"--replace-q397 {replace}: only with --line best --m3 m3x3_full (the six-channel recipe of the c115 evaluation; primary line only)", 2)
    t0 = time.time()
    ref_path = ART / f"stack_{stem_of(line, m3, False)}.json"
    ref_stack, _ = build(line, m3, False)
    if not ref_path.is_file() or ref_path.read_text() != to_json(ref_stack):
        abort(f"check: the same path with q397 does not reproduce {rel(ref_path)} character for character")
    ctx = setup()
    y, folds, seen, n, K = ctx["y"], ctx["folds"], ctx["seen"], ctx["n"], ctx["K"]
    mats, sha16, metas = load_mats(ctx, m3)
    P, sha_r, meta_r, c115 = load_replacement(ctx, replace)
    mats[replace], sha16[replace] = P, sha_r
    spec5, spec_b = line_specs(line, m3)
    spec_r = replaced_spec(spec_b, CH, replace)
    draws = C107.question_draws(ctx["q_code"])
    fit_r = C107.cross_fit(y, folds, seen, mats, spec_r)
    fit_b = C107.cross_fit(y, folds, seen, mats, spec_b)
    fit_0 = C107.cross_fit(y, folds, seen, mats, spec5)
    r_r, r_b, r_0 = (C107.rows_map(y, fit["Q"], seen) for fit in (fit_r, fit_b, fit_0))
    ll_r, ll_b, ll_0 = (C107.rows_ll(y, fit["Q"]) for fit in (fit_r, fit_b, fit_0))
    u_r, u_b = int(C107.units(r_r).sum()), int(C107.units(r_b).sum())
    er, anc = c115["recipes"]["replacement"], c115["anchor"]
    if u_b != REPLACE_UNITS["six_channel"] or u_b != int(anc["units"]) or abs(float(ll_b.mean()) - float(anc["logloss"])) > 1e-12:
        abort(f"the six-channel recipe in the cross-fit {u_b}/{6 * n} (log loss {ll_b.mean():.12f}) does not reproduce the anchor of the c115 evaluation ({anc['units']}, {anc['logloss']})")
    if (u_r != REPLACE_UNITS[replace] or u_r != int(round(float(er["map_new"]) * 6 * n))
            or abs(float(ll_r.mean()) - float(er["ll_new"])) > 1e-12):
        abort(f"cross value with {replace} {u_r}/{6 * n} (log loss {ll_r.mean():.12f}) differs from the replacement value of the c115 evaluation "
              f"({er['map_new']}, {er['ll_new']})")
    cmp_b = C107.compare(r_r, r_b, ctx, draws, ll_r, ll_b)
    if str(cmp_b["delta_exact"]) != str(er["delta_exact"]):
        abort(f"Δ against the six-channel recipe {cmp_b['delta_exact']} instead of {er['delta_exact']} (c115)")
    cmp_0 = C107.compare(r_r, r_0, ctx, draws, ll_r, ll_0)
    th_f = np.asarray(fit_r["theta"])
    j = spec_r["names"].index(replace)
    rep: dict = {"line": line, "m3": m3, "replace": {"channel": CH, "by": replace}, "names": spec_r["names"],
                 "check_stack_six_channel": f"bit-identical to {ref_path.name} (SHA-256 {sha256_file(ref_path)[:16]}…)"}
    rep["cross"] = {"map3": float(r_r.mean()), "units": u_r, "logloss": float(ll_r.mean()),
                    "map3_pre_last_step": float(C107.rows_map(y, fit_r["P"], seen).mean()),
                    "map3_one_set": float(C107.rows_map(y, C107.set_align(fit_r["P"], y, seen), seen).mean()),
                    "logloss_pre_last_step": float(C107.rows_ll(y, fit_r["P"]).mean()),
                    "per_fold": [float(r_r[folds == k].mean()) for k in range(C107.N_FOLDS)],
                    "theta_per_fold": th_f.tolist(), "theta_mean": th_f.mean(0).tolist(), f"share_{replace}": (th_f[:, j] / th_f.sum(1)).tolist(),
                    "six_channel": {"map3": float(r_b.mean()), "units": u_b, "logloss": float(ll_b.mean())},
                    "delta_vs_six_channel": summarize_cross(cmp_b),
                    "without_sixth": {"map3": float(r_0.mean()), "logloss": float(ll_0.mean())},
                    "delta_vs_without_sixth": summarize_cross(cmp_0)}

    # deployed weights (all 1,500 rows) and blend file
    th, th0, L = fit_all(mats, spec_r, y, seen)
    allr = np.arange(n)
    floors = {k: float(v) for k, v in spec_r["floors"].items() if v}
    src = json.loads((ART / LINES[line]["src_stack"][0]).read_text())
    ro_m3 = rank_only_entry(m3, metas[m3], sha16[m3], n)
    if ro_m3 != ref_stack["rank_only"][m3]:
        abort(f"rank_only entry {m3} differs from {ref_path.name}")
    nt = meta_r["rank"]["notes"]
    ro_r = {"floor": FLOOR, "rank_source": meta_r["rank"]["source"], "oof_sha256": sha_r, "rows": int(n),
            "notes_train_sha256": nt["train"]["sha256"], "notes_test": nt["test"]["sft_from"],
            "c115_channel_sha256": c115["technical"]["channel_sha256"],
            "definition": "pure ranking (probe proto_rank, mode sft): M3 behind the notes of Qwen3.5-397B — OOF per fold the adapter without "
                          "that fold behind the Tinker notes, test rows M3 of the final fit behind the notes of our own 397B run "
                          "(make_submission --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test; note guard rank.notes in meta_q397m3.json)"}
    c = rep["cross"]
    d = c["delta_vs_six_channel"]
    stack = {"channels": list(spec_r["names"]), "none": [], "probes": {}, "theta": [float(v) for v in th], "floor": floors,
             "rows": int(n), "folds": list(range(C107.N_FOLDS)), "fit_rows": [int(i) for i in allr], "fit_rows_sha256": rows_sha256(allr),
             "base": {"channels": list(spec_r["names"]), "theta": [float(v) for v in th0]},
             "scaled": {spec_r["think"]: float(spec_r["factor"])} if spec_r.get("think") else {},
             "rank_only": {m3: ro_m3, replace: ro_r},
             "arith_beta": float(spec_r["beta"]),
             "cv": c["map3"], "cv_rows": int(n),
             "cv_note": (f"cross-fit per fold on 1,500 rows WITH the last step (weights per fold from the other four, eda/c107_q397_full.py "
                         f"cross_fit) = replacement value of the evaluation eda/c115_q397m3_eval.py ({u_r}/{6 * n}); six-channel recipe with q397 "
                         f"{c['six_channel']['map3']:.7f}, Δ {d['delta']:+.5f} ± {d['se_q']:.5f} over tasks (z {d['z_q']:+.2f}); without a sixth channel "
                         f"{c['without_sixth']['map3']:.7f}; base OOFs not fully nested in the meta fold; weights θ from the OOF of the channel q397m3 "
                         "(fold adapters behind the Tinker notes of the 397B) — the test side is M3 of the final fit behind a different draw of the "
                         "397B notes (own run, checked by the fidelity check)"),
             "cross": {"map3": c["map3"], "logloss": c["logloss"], "per_fold": c["per_fold"], "map3_pre_last_step": c["map3_pre_last_step"],
                       "map3_one_set": c["map3_one_set"], "six_channel": c["six_channel"]["map3"], "without_sixth": c["without_sixth"]["map3"],
                       "delta_vs_six_channel": {k: d[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}},
             "oof_sha256": {nm: sha16[nm] for nm in spec_r["names"]},
             "last_step": src["last_step"],
             "replaces": {"channel": CH, "by": replace, "stack_six_channel": ref_path.name, "stack_six_channel_sha256": sha256_file(ref_path),
                          "c115": rel(REPLACE_C115),
                          "c115_replacement": {k: er[k] for k in ("map_new", "map_base", "delta_exact", "delta", "se_q", "z_q", "ll_new", "ll_base")}},
             "note": (f"eda/c110_q397_deploy.py stack --line {line} --m3 {m3} --replace-q397 {replace}: six-channel recipe with {replace} instead of "
                      f"q397 (floor {FLOOR:g}); weights of ALL channels on log loss from all {n} rows (a2.fit_mix, then "
                      f"{spec_r['think']} × {spec_r['factor']:g}); build script tools/builds/s029_q397m3.sh")}

    # check make_submission.oof_mixture against the own blend; in-sample with the last step (one set, as in build)
    P_ms = MS.oof_mixture(stack, seen, ART, n_rows=n)
    Lm = C107.logs_of(mats, spec_r["names"], spec_r["floors"], seen, True)
    P_mask = MS.mix_channels(stack, Lm, None, seen)
    P_test_way = MS.mix_channels(stack, L, None, seen)
    d_mask, d_way = float(np.abs(P_ms - P_mask).max()), float(np.abs(P_ms - P_test_way).max())
    if d_mask != 0.0 or d_way > 1e-9:
        abort(f"check oof_mixture: deviation {d_mask:.1e} (mask first) / {d_way:.1e} (test path)")
    Q_in = C107.apply_last(P_test_way, C21.fit_weights(P_test_way, C107.target_of(y, allr, K)), seen)
    r_in = C107.rows_map(y, Q_in, seen)
    pre_in = float(C107.rows_map(y, P_test_way, seen).mean())
    ll_pre_in = float(C107.rows_ll(y, P_test_way).mean())
    rep["insample"] = {"map3": float(r_in.mean()), "logloss": float(C107.rows_ll(y, Q_in).mean()), "map3_pre_last_step": pre_in,
                       "logloss_pre_last_step": ll_pre_in, "vs_cross_same_last_step": float(r_in.mean()) - c["map3_one_set"],
                       "vs_cross_pre_last_step": pre_in - c["map3_pre_last_step"],
                       "vs_cross_logloss_pre_last_step": ll_pre_in - c["logloss_pre_last_step"],
                       "plausible": bool(ll_pre_in <= c["logloss_pre_last_step"] and pre_in >= c["map3_pre_last_step"] - 0.003
                                         and float(r_in.mean()) >= c["map3_one_set"] - 0.003),
                       "note": "OPTIMISTIC: weights and last step from the same 1,500 rows, aligned as ONE set — compared with "
                               "the cross-fit of the same form (one set) or without the last step"}
    stack["insample"] = {"map3": rep["insample"]["map3"], "map3_pre_last_step": pre_in, "logloss": rep["insample"]["logloss"]}
    rep["check_oof_mixture"] = {"max_abs_mask_first": d_mask, "max_abs_test_way": d_way}
    rep["theta"] = {nm: float(v) for nm, v in zip(spec_r["names"], th)}
    rep["theta_before_factor"] = {nm: float(v) for nm, v in zip(spec_r["names"], th0)}
    rep["share_theta"] = {nm: float(v / th.sum()) for nm, v in zip(spec_r["names"], th)}
    rep["seconds"] = round(time.time() - t0, 1)
    if not C107.sane(Q_in, seen) or not C107.sane(P_ms, seen):
        abort("predictions not normalised or with mass on classes without a training example")
    return stack, rep


BETA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)  # grid of the β rule (the same as the nested choice in the model sweep)


def choose_beta(th, logs, y: np.ndarray, seen: np.ndarray, grid=BETA_GRID) -> tuple[float, dict]:
    """β rule: argmin log loss of the blend BEFORE the last step on all fit rows (θ from the same rows, after the reasoning
    factor), grid 0.1…0.8; on a tie (≤ 1e-12) the smaller value. -> (β, log loss per grid value)."""
    rows = np.arange(len(y))
    best, bb, lls = None, None, {}
    for b in grid:
        ll = mean_ll(mixture(th, b, logs, rows, seen), y)
        lls[f"{b:g}"] = ll
        if best is None or ll < best - 1e-12:
            best, bb = ll, b
    return float(bb), lls


def build_replaced_beta(line: str, m3: str, replace: str, beta_fixed: float | None, beta_select: bool) -> tuple[dict, dict]:
    """Switch --arith-beta / --arith-beta-select (default off): blend file of the replacement recipe with a different arithmetic
    share β. First the unchanged path with β 0.2 (build_replaced) — it must reproduce stack_<s029_best_q397m3>.json character for
    character (check); then the same file with the new β: fields arith_beta, cv (cross value per fold with the last step at the new
    β), cv_note, cross, insample, note and arith_beta_rule (rule, grid, log loss per grid value, cross value at β 0.2). θ, floors,
    channels, rank_only and checksums stay — θ does not depend on β (a2.fit_mix on the log-linear blend)."""
    stack0, rep0 = build_replaced(line, m3, replace)
    ref_path = ART / f"stack_{stem_of(line, m3, False, replace)}.json"
    if not ref_path.is_file() or ref_path.read_text() != to_json(stack0):
        abort(f"check β 0.2: the unchanged path does not reproduce {rel(ref_path)} character for character")
    ctx = setup()
    y, folds, seen, n, K = ctx["y"], ctx["folds"], ctx["seen"], ctx["n"], ctx["K"]
    mats, sha16, _ = load_mats(ctx, m3)
    P, sha_r, _, _ = load_replacement(ctx, replace)
    mats[replace], sha16[replace] = P, sha_r
    _, spec_b = line_specs(line, m3)
    spec_r = replaced_spec(spec_b, CH, replace)
    th, _, L = fit_all(mats, spec_r, y, seen)
    if [float(v) for v in th] != stack0["theta"]:
        abort("θ of the β version differs from the version with β 0.2")
    _, lls = choose_beta(th, L, y, seen)
    beta = choose_beta(th, L, y, seen)[0] if beta_select else float(beta_fixed)
    if not 0.0 <= beta <= 1.0:
        abort(f"β {beta} is not in [0, 1]", 2)
    spec_s = {**spec_r, "beta": beta}
    draws = C107.question_draws(ctx["q_code"])
    fit_s = C107.cross_fit(y, folds, seen, mats, spec_s)
    fit_0 = C107.cross_fit(y, folds, seen, mats, spec_r)
    fit_b = C107.cross_fit(y, folds, seen, mats, spec_b)
    r_s, r_0, r_b = (C107.rows_map(y, f["Q"], seen) for f in (fit_s, fit_0, fit_b))
    ll_s, ll_0, ll_b = (C107.rows_ll(y, f["Q"]) for f in (fit_s, fit_0, fit_b))
    if int(C107.units(r_0).sum()) != REPLACE_UNITS[replace] or int(C107.units(r_b).sum()) != REPLACE_UNITS["six_channel"]:
        abort("cross value at β 0.2 or of the six-channel recipe does not reproduce the values of the c115 evaluation")
    cmp_b, cmp_0 = C107.compare(r_s, r_b, ctx, draws, ll_s, ll_b), C107.compare(r_s, r_0, ctx, draws, ll_s, ll_0)
    pre_s, set_s = C107.rows_map(y, fit_s["P"], seen), C107.rows_map(y, C107.set_align(fit_s["P"], y, seen), seen)
    pre_0, set_0 = C107.rows_map(y, fit_0["P"], seen), C107.rows_map(y, C107.set_align(fit_0["P"], y, seen), seen)
    pre_b, set_b = C107.rows_map(y, fit_b["P"], seen), C107.rows_map(y, C107.set_align(fit_b["P"], y, seen), seen)
    forms = {"pre_last_step": {"vs_beta_0_2": C107.compare(pre_s, pre_0, ctx, draws), "vs_six_channel": C107.compare(pre_s, pre_b, ctx, draws)},
             "one_set": {"vs_beta_0_2": C107.compare(set_s, set_0, ctx, draws), "vs_six_channel": C107.compare(set_s, set_b, ctx, draws)}}
    th_f = np.asarray(fit_s["theta"])
    j = spec_s["names"].index(replace)
    c = {"map3": float(r_s.mean()), "units": int(C107.units(r_s).sum()), "logloss": float(ll_s.mean()),
         "map3_pre_last_step": float(pre_s.mean()), "map3_one_set": float(set_s.mean()),
         "logloss_pre_last_step": float(C107.rows_ll(y, fit_s["P"]).mean()), "per_fold": [float(r_s[folds == k].mean()) for k in range(C107.N_FOLDS)],
         "theta_per_fold": th_f.tolist(), f"share_{replace}": (th_f[:, j] / th_f.sum(1)).tolist(),
         "six_channel": {"map3": float(r_b.mean()), "units": int(C107.units(r_b).sum()), "logloss": float(ll_b.mean())},
         "delta_vs_six_channel": summarize_cross(cmp_b),
         "beta_0_2": {"map3": float(r_0.mean()), "units": int(C107.units(r_0).sum()), "logloss": float(ll_0.mean())},
         "delta_vs_beta_0_2": summarize_cross(cmp_0),
         "forms": {k: {kk: {x: vv[x] for x in ("delta", "se_q", "z_q", "folds_gt0") if x in vv} for kk, vv in v.items()} for k, v in forms.items()}}
    d, d0 = c["delta_vs_six_channel"], c["delta_vs_beta_0_2"]
    stack = json.loads(json.dumps(stack0))  # deep copy, the field order stays
    rule = {"rule": ("argmin log loss of the blend before the last step on all 1,500 fit rows (θ from the same rows, after the "
                     "reasoning factor), grid 0.1…0.8 — the same rule as the nested choice in the model sweep") if beta_select
            else f"fixed {beta:g} (switch --arith-beta)",
            "grid": list(BETA_GRID), "logloss_pre_last_step": lls, "chosen": beta, "selected": bool(beta_select),
            "stack_beta_0_2": ref_path.name, "stack_beta_0_2_sha256": sha256_file(ref_path),
            "cross_beta_0_2": c["beta_0_2"], "delta_vs_beta_0_2": {k: d0[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}}
    stack["arith_beta"] = beta
    stack["cv"] = c["map3"]
    stack["cv_note"] = (f"cross-fit per fold on 1,500 rows WITH the last step at the arithmetic share β {beta:g} (weights per fold from the other "
                        f"four, eda/c107_q397_full.py cross_fit; β by arith_beta_rule from all 1,500 rows); at β 0.2 {c['beta_0_2']['map3']:.7f} "
                        f"(replacement value of c115), Δ {d0['delta']:+.5f} ± {d0['se_q']:.5f}; against the six-channel recipe with q397 "
                        f"{c['six_channel']['map3']:.7f} Δ {d['delta']:+.5f} ± {d['se_q']:.5f} over tasks (z {d['z_q']:+.2f}); base OOFs not fully nested in "
                        "the meta fold")
    stack["cross"] = {"map3": c["map3"], "logloss": c["logloss"], "per_fold": c["per_fold"], "map3_pre_last_step": c["map3_pre_last_step"],
                      "map3_one_set": c["map3_one_set"], "six_channel": c["six_channel"]["map3"], "beta_0_2": c["beta_0_2"]["map3"],
                      "delta_vs_six_channel": {k: d[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}}
    stack["arith_beta_rule"] = rule
    stack["note"] = (f"eda/c110_q397_deploy.py stack --line {line} --m3 {m3} --replace-q397 {replace} "
                     + ("--arith-beta-select" if beta_select else f"--arith-beta {beta:g}")
                     + f": six-channel recipe with {replace} instead of q397 (floor {FLOOR:g}) at β {beta:g} instead of 0.2; weights of ALL channels "
                     f"on log loss from all {n} rows (a2.fit_mix, then {spec_r['think']} × {spec_r['factor']:g}); build script tools/builds/s032_stage2.sh")
    allr = np.arange(n)
    P_ms = MS.oof_mixture(stack, seen, ART, n_rows=n)
    Lm = C107.logs_of(mats, spec_s["names"], spec_s["floors"], seen, True)
    P_mask, P_way = MS.mix_channels(stack, Lm, None, seen), MS.mix_channels(stack, L, None, seen)
    d_mask, d_way = float(np.abs(P_ms - P_mask).max()), float(np.abs(P_ms - P_way).max())
    if d_mask != 0.0 or d_way > 1e-9:
        abort(f"check oof_mixture (β {beta:g}): deviation {d_mask:.1e} / {d_way:.1e}")
    Q_in = C107.apply_last(P_way, C21.fit_weights(P_way, C107.target_of(y, allr, K)), seen)
    stack["insample"] = {"map3": float(C107.rows_map(y, Q_in, seen).mean()), "map3_pre_last_step": float(C107.rows_map(y, P_way, seen).mean()),
                         "logloss": float(C107.rows_ll(y, Q_in).mean())}
    if not C107.sane(Q_in, seen) or not C107.sane(P_ms, seen):
        abort("predictions not normalised or with mass on classes without a training example")
    rep = {**rep0, "cross_beta": c, "arith_beta_rule": rule, "insample_beta": stack["insample"],
           "check_beta_0_2": f"character for character {ref_path.name} (SHA-256 {sha256_file(ref_path)[:16]}…)",
           "check_oof_mixture_beta": {"max_abs_mask_first": d_mask, "max_abs_test_way": d_way}}
    return stack, rep


def cmd_stack_replaced(a) -> int:
    if a.beta_novel:
        abort("--replace-q397 together with --beta-novel was not measured (c115 evaluated the six-channel recipe without β_new)", 2)
    stem = a.stem or stem_of(a.line, a.m3, False, a.replace_q397)
    path = ART / f"stack_{stem}.json"
    default_beta = float(C107.LINE_BEST["beta"])
    with_beta = a.arith_beta_select or (a.arith_beta is not None and float(a.arith_beta) != default_beta)
    if with_beta:  # switch for a different β (the version with β 0.2 is the check inside it)
        stack, rep = build_replaced_beta(a.line, a.m3, a.replace_q397, a.arith_beta, a.arith_beta_select)
        cb, rule = rep["cross_beta"], rep["arith_beta_rule"]
        print(f"{stem}: arithmetic share β {rule['chosen']:g} ({'rule: argmin log loss before the last step, all 1,500 rows' if rule['selected'] else 'fixed'}); "
              "log loss per grid value " + " ".join(f"{k} {v:.4f}" for k, v in rule["logloss_pre_last_step"].items()))
        d0, db = cb["delta_vs_beta_0_2"], cb["delta_vs_six_channel"]
        print(f"   CROSS β {rule['chosen']:g}: {cb['map3']:.5f} ({cb['units']}/9000) | against β 0.2 ({cb['beta_0_2']['map3']:.5f}) Δ {d0['delta']:+.5f} ± "
              f"{d0['se_q']:.5f} (z {d0['z_q']:+.2f}, {d0['folds_gt0']}/5 > 0; log-loss gain {d0['ll_gain']:+.4f}) | against the six-channel recipe Δ {db['delta']:+.5f} ± {db['se_q']:.5f}")
        for fm, lab in (("pre_last_step", "without the last step"), ("one_set", "as one set")):
            v0, vb = cb["forms"][fm]["vs_beta_0_2"], cb["forms"][fm]["vs_six_channel"]
            print(f"   {lab}: against β 0.2 Δ {v0['delta']:+.5f} ± {v0['se_q']:.5f} | against the six-channel recipe Δ {vb['delta']:+.5f} ± {vb['se_q']:.5f}")
        print(f"   {rep['check_beta_0_2']} (path without the switch)")
        text = to_json(stack)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"c110_stack_{stem}.json").write_text(to_json({**rep, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                               "stack_file": str(rel(path)), "stack_sha256": hashlib.sha256(text.encode()).hexdigest()}))
        if a.verify:
            if not path.is_file():
                abort(f"{rel(path)} missing — build it with --save first", 3)
            if path.read_text() != text:
                abort(f"{rel(path)} is not the recomputed blend")
            print(f"   VERIFIED: {rel(path)} = recomputation (SHA-256 {sha256_file(path)[:16]}…)")
        if a.save:
            if path.is_file() and path.read_text() != text:
                abort(f"{rel(path)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
            path.write_text(text)
            print(f"   saved: {rel(path)} (SHA-256 {sha256_file(path)})")
        return 0
    stack, rep = build_replaced(a.line, a.m3, a.replace_q397)
    text = to_json(stack)
    c, r = rep["cross"], a.replace_q397
    d, d0 = c["delta_vs_six_channel"], c["delta_vs_without_sixth"]
    print(f"{stem}: channels {stack['channels']} ({r} instead of q397)")
    print("   θ (all 1,500 rows, think18nb × 1.4): " + " · ".join(f"{nm} {v:.4f}" for nm, v in rep["theta"].items())
          + f" | share {r} {rep['share_theta'][r]:.3f}")
    print(f"   {rep['check_stack_six_channel']} (the same path with q397)")
    print(f"   CROSS with {r} {c['map3']:.5f} ({c['units']}/{6 * len(stack['fit_rows'])} = replacement value of c115; six-channel recipe {c['six_channel']['map3']:.5f}, "
          f"{c['six_channel']['units']}/{6 * len(stack['fit_rows'])}): Δ {d['delta']:+.5f} ± {d['se_q']:.5f} over tasks (z {d['z_q']:+.2f}; "
          f"{d['folds_gt0']}/5 folds > 0), log loss {c['six_channel']['logloss']:.4f} → {c['logloss']:.4f} (z {d['ll_z_q']:+.2f}); without a sixth channel "
          f"{c['without_sixth']['map3']:.5f} (Δ {d0['delta']:+.5f} ± {d0['se_q']:.5f}); θ {r} per fold "
          + " ".join(f"{x[rep['names'].index(r)]:.3f}" for x in c["theta_per_fold"]))
    ins = rep["insample"]
    print(f"   IN-SAMPLE with the last step {ins['map3']:.5f} (OPTIMISTIC; cross aligned as one set {c['map3_one_set']:.5f}, difference "
          f"{ins['vs_cross_same_last_step']:+.5f}); without the last step {ins['map3_pre_last_step']:.5f} (cross {c['map3_pre_last_step']:.5f}; log loss "
          f"{ins['logloss_pre_last_step']:.4f} against {c['logloss_pre_last_step']:.4f}) -> "
          f"{'plausible' if ins['plausible'] else 'CONSPICUOUS (in-sample log loss above the cross value or MAP@3 more than 0.003 below)'}")
    print(f"   oof_mixture against own blend: mask first {rep['check_oof_mixture']['max_abs_mask_first']:.1e}, "
          f"test path {rep['check_oof_mixture']['max_abs_test_way']:.1e}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"c110_stack_{stem}.json").write_text(to_json({**rep, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                           "stack_file": str(rel(path)), "stack_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    if a.verify:
        if not path.is_file():
            abort(f"{rel(path)} missing — build it with --save first", 3)
        if path.read_text() != text:
            abort(f"{rel(path)} is not the recomputed blend (channels, weights, cross value or checksums changed)")
        print(f"   VERIFIED: {rel(path)} = recomputation (SHA-256 {sha256_file(path)[:16]}…)")
    if a.save:
        if path.is_file() and path.read_text() != text:
            abort(f"{rel(path)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        path.write_text(text)
        print(f"   saved: {rel(path)} (SHA-256 {sha256_file(path)}) -> {rel(OUT / f'c110_stack_{stem}.json')}")
    return 0


# ------------------------------------------------------------------------------------------------ test-status

def cmd_test_status(a) -> int:
    bits = a.run.split("/")
    if len(bits) != 4 or bits[0] != "runs":
        abort(f"{a.run}: expected runs/<run>/<job>/<tag>", 2)
    _, run, job, tag = bits
    d = ART / "runs" / run / run / job
    meta_f, lab_f = d / f"{tag}_meta.json", d / f"{tag}_labels.jsonl"
    if not meta_f.is_file() or not lab_f.is_file():
        abort(f"{a.run}: {meta_f.name} or {lab_f.name} missing (run not finished or not fetched: run {run})", 3)
    root = C.find_data_root()
    train, test = C.load_table(root, "train"), C.load_table(root, "test")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    seen = labels.seen_mask(train["root_code"], classes)
    n_test = len(test)
    meta = json.loads(meta_f.read_text())
    problems, hard = [], []
    if meta.get("table") != "test" or int(meta.get("rows", -1)) != n_test:
        hard.append(f"table {meta.get('table')!r}, {meta.get('rows')} rows (required test, {n_test})")
    P, has = PR.labels_to_probs(lab_f, n_test, classes, seen)
    valid = int(has.sum())
    if int(meta.get("valid", -1)) != valid:
        hard.append(f"answer file {valid} valid rows, meta file {meta.get('valid')}")
    if valid < MIN_VALID * n_test:
        problems.append(f"valid {valid}/{n_test} ({valid / n_test:.1%}) below {MIN_VALID:.0%}")
    none_top = float(np.mean(P[has].argmax(1) == 0)) if valid else float("nan")
    Pt, ht = PR.labels_to_probs(root / TINKER_REL, len(train), classes, seen)
    none_train = float(np.mean(Pt[ht].argmax(1) == 0))
    if not abs(none_top - none_train) <= NONE_BAND:
        hard.append(f"NONE at the top {none_top:.3f} outside the band fixed in advance, training {none_train:.3f} ± {NONE_BAND:g}")
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").is_file() else {}
    model = (summ.get("model") or {}).get("repo")
    settings = {}
    cfg_f = COMP / "configs" / "vlm" / f"{run}.yaml"
    if cfg_f.is_file():
        import yaml

        cfg = yaml.safe_load(cfg_f.read_text()) or {}
        jobs = [j for w in cfg.get("workers") or [] for j in w.get("jobs") or [] if j.get("name") == job]
        settings = jobs[0] if len(jobs) == 1 else {}
    if not a.placeholder:
        if model != TEST_MODEL:
            hard.append(f"model {model!r} instead of {TEST_MODEL}")
        want = {"kind": "proto_gen", "split": "test", "max_tokens": 2500, "finish_tokens": 1500, "tag": tag}
        diff = {k: settings.get(k) for k, v in want.items() if settings.get(k) != v}
        img = settings.get("image") or {}
        if img.get("max_side") != 1024 or img.get("question_max_side") != 448:
            diff["image"] = img
        if settings.get("rows") or settings.get("temperature") or settings.get("sampling"):
            diff["rows/temperature/sampling"] = (settings.get("rows"), settings.get("temperature"), settings.get("sampling"))
        if not settings:
            diff["config"] = f"{rel(cfg_f)} without a unique job {job}"
        if diff:
            hard.append(f"settings differ from the Tinker measurement: {diff}")
    rec = {"run": a.run, "placeholder": bool(a.placeholder), "rows": n_test, "answered": meta.get("answered"), "failed": meta.get("failed"),
           "valid": valid, "valid_share": valid / n_test, "none_top_valid": none_top, "none_top_train_tinker": none_train,
           "none_band": NONE_BAND, "model": model, "labels_sha256": sha256_file(lab_f), "meta_sha256": sha256_file(meta_f),
           "ends_first": meta.get("ends_first"), "out_tokens_first_median": meta.get("out_tokens_first_median"),
           "in_tokens_first_median": meta.get("in_tokens_first_median"), "problems": problems, "hard": hard}
    print(f"test side {a.run}{' (PLACEHOLDER — code check only)' if a.placeholder else ''}: {n_test} rows, answered {meta.get('answered')}, "
          f"failures {meta.get('failed')}, valid {valid} ({valid / n_test:.1%}); NONE at the top {none_top:.3f} (training/Tinker {none_train:.3f}, "
          f"band ± {NONE_BAND:g}); model {model}; first passes {meta.get('ends_first')}; answer file SHA-256 {rec['labels_sha256'][:16]}…")
    if not a.placeholder:  # placeholder (foreign test side, code check only): no log that could be mistaken for a 397B check
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"c110_test_status_{run}.json").write_text(to_json({**rec, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    if hard:
        abort("test side not deployable: " + "; ".join(hard), 5)
    if problems:
        abort("test side incomplete: " + "; ".join(problems), 3)
    print("   test side deployable (coverage, NONE band" + ("" if a.placeholder else ", model, settings") + ")")
    return 0


# ------------------------------------------------------------------------------------------------ readable

def readable_no(paths: list[Path], n: int, classes: list[str]) -> tuple[np.ndarray, Counter]:
    """Rows whose last valid record (ok, root_code in the class list — as in labels_to_probs) reports "readable = no".
    -> (mask [n], counts of the values over the valid rows)."""
    idx = set(classes)
    last = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ok", True) or r.get("root_code") not in idx:
                continue
            last[int(r["row"])] = r
    vals = {i: str(r.get("readable", "")).strip().lower() for i, r in last.items()}
    m = np.zeros(n, dtype=bool)
    for i, v in vals.items():
        if v.startswith("no"):
            m[i] = True
    return m, Counter(v[:6] for v in vals.values())


def force_code(Q: np.ndarray, rows: np.ndarray, j: int) -> np.ndarray:
    """Code j at rank 1 of the rows `rows`, the rest of the order stays — as in make_submission --blank-rows."""
    P = Q.copy()
    if len(rows):
        P[rows, j] = P[rows].max(1) + 1.0
        P[rows] = P[rows] / P[rows].sum(1, keepdims=True)
    return P


def cmd_readable(a) -> int:
    ctx = setup()
    root, n, y, seen, classes = ctx["root"], ctx["n"], ctx["y"], ctx["seen"], ctx["classes"]
    j = classes.index("REL-01")
    # training rows: cross value of the rule in the recipes of both lines (with q397, M3 = m3x2_full_cv01)
    no_tr, vals_tr = readable_no([COMP / p for p in PR27_TRAIN], n, classes)
    blank_tr = np.load(root / BLANK_TRAIN).astype(np.int64)
    union_tr = np.union1d(blank_tr, np.flatnonzero(no_tr))
    C107.check_stacks()  # checksums and recipe constants of the earlier blend files of the primary and inductive lines
    mats, _, _ = load_mats(ctx, M3_DEFAULT)
    draws = C107.question_draws(ctx["q_code"])
    res = {"train": {"rows_readable_no": int(no_tr.sum()), "blank_rows": int(len(blank_tr)), "union": int(len(union_tr)),
                     "new_rows": int(len(union_tr) - len(blank_tr)), "gold_rel01_in_readable_no": int((y[no_tr] == j).sum()),
                     "values": dict(vals_tr)}, "lines": {}}
    for line in ("best", "safe"):
        Q = C107.cross_fit(ctx["y"], ctx["folds"], seen, mats, line_specs(line, M3_DEFAULT)[1])["Q"]
        r_b = C107.rows_map(y, force_code(Q, blank_tr, j), seen)
        r_u = C107.rows_map(y, force_code(Q, union_tr, j), seen)
        c = C107.compare(r_u, r_b, ctx, draws)
        res["lines"][line] = {"cross_blank": float(r_b.mean()), "cross_blank_and_readable": float(r_u.mean()),
                              "cross_without_rel01": float(C107.rows_map(y, Q, seen).mean()), "delta": c["delta"], "delta_exact": c["delta_exact"],
                              "se_q": c["se_q"], "rows_better": c["rows_better"], "rows_worse": c["rows_worse"]}
    # test rows: counts only
    test = C.load_table(root, "test")
    no_te, vals_te = readable_no([COMP / PR27_TEST], len(test), classes)
    blank_te = np.load(root / BLANK_TEST).astype(np.int64)
    union_te = np.union1d(blank_te, np.flatnonzero(no_te)).astype(np.int64)
    res["test"] = {"rows_readable_no": int(no_te.sum()), "blank_rows": int(len(blank_te)),
                   "overlap": int(np.isin(blank_te, np.flatnonzero(no_te)).sum()), "union": int(len(union_te)),
                   "new_rows": int(len(union_te) - len(blank_te)), "values": dict(vals_te), "pr27_test_sha256": sha256_file(COMP / PR27_TEST),
                   "rows_sha256": rows_sha256(union_te)}
    t, tr = res["test"], res["train"]
    print(f"REL-01 for \"readable = no\" (pr27): training {tr['rows_readable_no']} rows (gold REL-01: {tr['gold_rel01_in_readable_no']}), blank images "
          f"{tr['blank_rows']}, new {tr['new_rows']}; values {tr['values']}")
    for line, v in res["lines"].items():
        print(f"   CROSS {line} (+ q397): without REL-01 {v['cross_without_rel01']:.5f}, blank images {v['cross_blank']:.5f}, + readable = no "
              f"{v['cross_blank_and_readable']:.5f} (Δ {v['delta']:+.5f} = {v['delta_exact']}; better/worse {v['rows_better']}/{v['rows_worse']})")
    print(f"   TEST (counts only): readable = no {t['rows_readable_no']}, of which blank images {t['overlap']}; list {t['union']} rows = "
          f"{t['blank_rows']} blank + {t['new_rows']} new; values {t['values']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c110_readable.json").write_text(to_json({**res, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    if a.save:
        dst = root / READABLE_TEST
        if dst.is_file() and not np.array_equal(np.load(dst), union_te):
            abort(f"data/{READABLE_TEST} exists with different content — not overwritten")
        np.save(dst, union_te)
        meta = {"rule": "blank photos (tools/make_blank_list.py) ∪ test rows whose last valid pr27 record "
                        "(runs/pr27te_rp/proto_te/proto) reports readable = no",
                "rows": t["union"], "blank_rows": t["blank_rows"], "new_rows": t["new_rows"], "rows_sha256": t["rows_sha256"],
                "pr27_test_sha256": t["pr27_test_sha256"], "blank_sha256": sha256_file(root / BLANK_TEST),
                "train_cross": {line: {k: v[k] for k in ("cross_blank", "cross_blank_and_readable", "delta")} for line, v in res["lines"].items()},
                "created_by": "eda/c110_q397_deploy.py readable --save"}
        (dst.parent / "test_rows_readable.meta.json").write_text(to_json(meta))
        print(f"   saved: data/{READABLE_TEST} ({t['union']} rows; SHA-256 of the positions {t['rows_sha256'][:16]}…)")
    return 0


# ------------------------------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("source", help="create (or check) the source job runs/q397_tinker/proto_t/proto from the Tinker file")
    sub.add_parser("verify-oof", help="oof_q397.npy bit-identical to the channel of the c107 decision")
    s = sub.add_parser("stack", help="build (--save) or recompute (--verify) the blend file of one line")
    s.add_argument("--line", required=True, choices=sorted(LINES))
    s.add_argument("--m3", default=M3_DEFAULT, choices=M3_CHOICES)
    s.add_argument("--beta-novel", action="store_true", help="switch: arithmetic share by code novelty")
    s.add_argument("--replace-q397", choices=REPLACE_CHOICES, help="switch (default off): replacement channel instead of q397 — q397m3 (only --line best "
                                                                   "--m3 m3x3_full) -> artifacts/stack_s029_best_q397m3.json")
    bg = s.add_mutually_exclusive_group()
    bg.add_argument("--arith-beta", type=float, help="switch (only with --replace-q397): fixed arithmetic share β instead of 0.2")
    bg.add_argument("--arith-beta-select", action="store_true", help="switch (only with --replace-q397): β = argmin log loss before the last "
                                                                     "step on all 1,500 rows, grid 0.1…0.8")
    s.add_argument("--stem", help="name of the blend file artifacts/stack_<stem>.json (only with --replace-q397; default s029_best_<replacement>)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    t = sub.add_parser("test-status", help="check the test side of the 397B (aggregates only)")
    t.add_argument("run", help="runs/<run>/proto_te/proto")
    t.add_argument("--placeholder", action="store_true", help="foreign test side as a placeholder (code check only): model and settings are not checked")
    r = sub.add_parser("readable", help="REL-01 list from pr27 \"readable = no\" (test rows only counted) and cross value on the training rows")
    r.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    return {"source": cmd_source, "verify-oof": cmd_verify_oof, "stack": cmd_stack, "test-status": cmd_test_status, "readable": cmd_readable}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
