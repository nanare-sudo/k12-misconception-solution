"""DEPLOYMENT PATH OF THE PRIMARY CANDIDATE primary_m3px27: the final primary form (primary_final: second stage main_guard_i1_mm with the
NONE guard under the widened rule) with the mean channel m3px27xm = ½(pr27 + m3px27) in place of pr27xm = ½(pr27 + px27). Build script
tools/builds/primary_m3px27.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c133_primary_m3px27.py oof (--save | --verify)      # artifacts/oof_m3px27xm.npy + meta_m3px27xm.json (mean channel)
    python -I eda/c133_primary_m3px27.py stack (--save | --verify)    # artifacts/stack_primary_m3px27.json (blend)
    python -I eda/c133_primary_m3px27.py fit (--save | --verify)      # artifacts/stage2_primary_m3px27.json (chain main form -> widened rule -> guard form -> I1 form -> final form)
    python -I eda/c133_primary_m3px27.py copy [--keep]                # COPY NULL POINT: m3px27 := px27 -> blend and chain = final primary form (fields)
    python -I eda/c133_primary_m3px27.py clean-copy                   # remove the temporary copy files under artifacts/
    python -I eda/c133_primary_m3px27.py test-guard [--rank SPEC]     # technical checks of the test side (aggregates only; exit 0 / 3 missing / 5 violated)
    python -I eda/c133_primary_m3px27.py null --source F --out G      # null model (weight 0) for the deployment file, with mech27

CHANNEL m3px27xm (training rows): rank_mean as in eda/c95_rank_mean.channel — 0.5 · (named pr27 codes + M3 ranking behind the same notes)
before the floor; named = a8.mean_of_draws of the pr27 answers (= oof_pr27 bit-identical), ranking = oof_m3px27 (eda/c132_m3px27_eval.py
--oof-save: five fold shards, fold fidelity checked, = the channel evaluated by c132). So the matrix is bit-identical to the one of the
expected-value rule computation of the m3px27 rule (c132: ½(oof_pr27 + m3px27)). Meta file: field rank_mean with rank_source = the five
shards, rank_compare "sft" and the note guard notes (tools/make_submission.py: rank_mean_notes_guard, comparison of the settings against
every shard).
BLEND (`stack`): stack_s032_stage2.json (final primary form) with m3px27xm instead of pr27xm (same position, floor 0.05); θ on log loss from
all 1,500 rows (c110.fit_all, reasoning factor 1.4), β by the same rule (c110.choose_beta), field cv = cross-fit per fold with the last step
(c107.cross_fit), in-sample fields and the check make_submission.oof_mixture as in c110; rank_only entries, rows, folds, last step
unchanged. CHECK (`copy`): the same path with the copy (m3px27xmc := oof_pr27xm bit-identical) hits θ, base θ, β, cv, cross-fit, in-sample
and β-rule numbers of stack_s032_stage2.json exactly.
CHAIN OF THE SECOND STAGE (`fit`): the context of the final primary form (c124.load_context, unchanged) with the swapped matrix and the new
channel name (Static rebuilt as in c132); nested (c121.prepare): main form and fallback (c118.run_nested), main_guard (with copy null point
(i)), I1 (c121.run_i1, null point (i)), MM (c124.run_mm, null point (i)) in the forms F, oL, M — the MM row values F and M must hit those of
the expected-value rule computation of c132 (c132_m3px27_eval_rows.npz) BIT FOR BIT. Deployment: c118.build_stage2 (main form + fallback +
note guard) -> widened rule -> main_guard -> main_guard_i1 -> main_guard_i1_mm, each step exactly the computation of c118 guard-fit /
c121 fit / c124 fit (null point (e): deployment features = evaluated OOF features; the first features = those of the predecessor
bit-identical). DEPLOYMENT FORM of the training rows = the standard cross-fit of the nested reproduction (c118.standard_forms with β
"select" per fold; deploy_forms_of): for the final primary form every fold chose β 0.6 = β of the blend file, there it is bit-identical
to c118.deploy_forms (the copy null point checks this); with m3px27xm fold 1 chooses β 0.7, and only the standard cross-fit satisfies
null point (e) (c118.build_stage2 with the additive arguments forms/cv_units_want). Sensitivity (report only): the same main model on
c118.deploy_forms -> artifacts/eda/private/primary_m3px27/sens/. The intermediate files lie under artifacts/eda/private/primary_m3px27/chain/
with provenance fields (derived_from); result artifacts/stage2_primary_m3px27.json with the field m3px27_build.
COPY NULL POINT (`copy`): the same chain with the copy hits the stored row values of stage2/stage2_guard/stage2_i1/stage2_mm bit for bit
and the files stage2_s032/s034/s035/primary_i1/primary_final.json in all fields except provenance (stack, stack_sha256, inputs_sha256,
oof_reproduction, derived_from, oof/splits/rule_oof/fallback of the *_build fields, m3px27_build) and the channel name. With --keep
oof/meta/test_m3px27xmc and stack_primary_m3px27_copy.json stay (only for the copy run of the test side in the build script); clean-copy
removes them.
TEST GUARD (`test-guard`): test ranking m3px27te_rp complete and exact (3,498/3,498), log "of which without a finding 3498", NONE at the
top in the band 0.30–0.38 (as on the OOF side, m3px27 rule), settings equal to every OOF shard (only model, fold and note source free),
note guard (make_submission.rank_mean_notes_guard), training regime (test model = merge of the full fit m3_fullte_rp/m3_train_full: no
held-out fold, all 1,500 rows, gold-only, training finished, task/system text/taxonomy/note limit/image/base model as all five fold
adapters, local adapter = merged one); plus as an aggregate the share of the same top code against the px27 test ranking (n1_rp/rank_test).
Test files are only counted and hashed.
Only aggregates and checksums are printed. Exit 3 = input missing; 4 = checksum/recomputation/null point; 5 = test guard violated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import metric  # noqa: E402

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "primary_m3px27"
CH_RANK = "m3px27"  # pure ranking (c132 --oof-save)
CH = "m3px27xm"  # mean channel in the recipe primary_m3px27
CH_COPY = "m3px27xmc"  # copy null point: the same path with px27 instead of m3px27 (= oof_pr27xm bit-identical)
SLOT = "pr27xm"
STEM, STEM_COPY, STEM_REF = "primary_m3px27", "primary_m3px27_copy", "s032_stage2"
STAGE2 = ART / "stage2_primary_m3px27.json"
PROTO_TRAIN = "runs/pr27_rp/proto_a/proto+runs/pr27_rp/proto_b/proto"
PROTO_TEST = "runs/pr27te_rp/proto_te/proto"
PX27_RANK = "runs/px27_rp/rank_tr/bench"
PX27_TEST = "runs/n1_rp/rank_test/test"
TEST_RUN = "m3px27te_rp"
TEST_RANK = f"runs/{TEST_RUN}/m3px27_test_rank/test"
TEST_MODEL = "Qwen/Qwen3.8-27B"
FULL_TRAIN = ("m3_fullte_rp", "m3_train_full")
FLOOR = 0.05
NONE_BAND = (0.30, 0.38)  # technical check of the m3px27 rule — the same band on the test side
FORMS3 = ("fold", "pre", "set")
C132_JSON = ART / "eda" / "private" / "m3px27_eval" / "c132_m3px27_eval.json"
C132_ROWS = ART / "eda" / "private" / "m3px27_eval" / "c132_m3px27_eval_rows.npz"
REF_CHAIN = {"a": "stage2_s032.json", "b": "stage2_s034.json", "c": "stage2_s035.json", "d": "stage2_primary_i1.json", "e": "stage2_primary_final.json"}
REF_ROWS = {"main": ("stage2", "c118_oof_rows.npz", "main_{f}_forced"), "fallback": ("stage2", "c118_oof_rows.npz", "fallback_{f}_forced"),
            "main_guard": ("stage2_guard", "c118_guard_oof_rows.npz", "main_guard_{f}_forced"), "i1": ("stage2_i1", "c121_i1_oof_rows.npz", "i1_{f}_forced"),
            "mm": ("stage2_mm", "c124_mm_oof_rows.npz", "mm_{f}_forced")}
PROVENANCE_TOP = ("stack", "stack_sha256", "inputs_sha256", "oof_reproduction", "derived_from", "m3px27_build")
PROVENANCE_BUILD = ("oof", "splits", "rule_oof", "fallback")
RECIPE_SAME = ("task", "system_sha256", "taxonomy", "notes_max_tokens", "image", "model")  # full fit equal to the fold adapters (as c117)
# Log line of vlm/runner.py (sft_from). The worker logs of the recorded test run were written with an earlier, German wording of the same
# line; both wordings are accepted (as in eda/c117_q397m3_deploy.py).
LOG_PREFIXES = ("sft_from: notes for", "sft_from: Ausgaben zu")
LOG_LINE = re.compile(r"sft_from: (?:notes for (\d+) rows of table (\w+), of which without a finding (\d+)"
                      r"|Ausgaben zu (\d+) Zeilen der Tabelle (\w+), davon ohne Befund (\d+))")
DEFINITION = ("0.5 · (named pr27 codes + exact ranking of the fine-tuned M3 behind the same notes of the first pr27 pass); "
              "OOF per fold the adapter without this fold, test rows M3 of the full fit; before the floor; floor in the blend 0.05 (m3px27 rule; "
              "function eda/c95_rank_mean.rank_mean)")

_MODS: dict = {}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c132():
    if "c132" not in _MODS:
        _MODS["c132"] = _load("c133_c132", COMP / "eda" / "c132_m3px27_eval.py")
    return _MODS["c132"]


def c124():
    return c132().c124()


def c118():
    return c124().c118()


def c121():
    return c124().c121()


def c110():
    if "c110" not in _MODS:
        _MODS["c110"] = _load("c133_c110", COMP / "eda" / "c110_q397_deploy.py")
    return _MODS["c110"]


def c95():
    if "c95" not in _MODS:
        _MODS["c95"] = _load("c133_c95", COMP / "eda" / "c95_rank_mean.py")
    return _MODS["c95"]


def ms():
    if "ms" not in _MODS:
        _MODS["ms"] = _load("c133_ms", COMP / "tools" / "make_submission.py")
    return _MODS["ms"]


def mp():
    return c132().mp()


class TechnicalError(RuntimeError):
    """Checksum, recomputation or null point violated: exit 4."""


class Missing(TechnicalError):
    """Input missing: exit 3."""


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def npy_sha256(P: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(P, dtype=np.float64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def to_json(obj) -> str:
    return c118().to_json(obj)


def light() -> dict:
    """Labels, classes, folds (without the recipe)."""
    from vlm import common as VC
    import labels

    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    return {"root": root, "train": train, "classes": classes, "y": labels.encode(train["root_code"], classes),
            "seen": labels.seen_mask(train["root_code"], classes), "folds": VC.load_folds(len(train)), "n": len(train), "K": len(classes)}


# ------------------------------------------------------------------------------------------------ mean channel (training rows)

def rank_artifact() -> tuple[np.ndarray, dict]:
    """oof_m3px27 (c132 --oof-save) = the channel evaluated by c132 (checksum of the values) from the five shards; verdict DEPLOY."""
    f, mf = ART / f"oof_{CH_RANK}.npy", ART / f"meta_{CH_RANK}.json"
    if not f.is_file() or not mf.is_file():
        raise Missing(f"{rel(f)} or {rel(mf)} missing — run python -I eda/c132_m3px27_eval.py --oof-save first")
    if not C132_JSON.is_file():
        raise Missing(f"{rel(C132_JSON)} missing — run python -I eda/c132_m3px27_eval.py first")
    ev = json.loads(C132_JSON.read_text())
    if ev.get("dry") is not False or (ev.get("decision") or {}).get("verdict") != "DEPLOY":
        raise TechnicalError(f"c132 gave {(ev.get('decision') or {}).get('verdict')!r} (dry {ev.get('dry')}) — no deployment under the m3px27 rule")
    meta = json.loads(mf.read_text())
    rk = meta.get("rank") or {}
    if rk.get("source") != c132().SOURCE or int(rk.get("rows") or 0) != 1500 or rk.get("rows_named_only"):
        raise TechnicalError(f"{rel(mf)}: rank.source/rows are not the five m3px27 shards")
    R = np.load(f)
    sha = npy_sha256(R)
    if sha != (ev.get("technical") or {}).get("channel_sha256"):
        raise TechnicalError(f"{rel(f)} ({sha[:16]}…) is not the channel evaluated by c132 ({str((ev.get('technical') or {}).get('channel_sha256'))[:16]}…)")
    return R, {"name": CH_RANK, "oof_file": rel(f), "oof_sha256": sha256_file(f), "meta_sha256": sha256_file(mf), "values_sha256": sha,
               "rank_source": rk.get("source"), "c132": rel(C132_JSON), "c132_sha256": sha256_file(C132_JSON),
               "c132_verdict": ev["decision"]["verdict"]}


def mean_channel(lc: dict, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """0.5 · (named + ranking) as c95.rank_mean; named = a8.mean_of_draws of the pr27 answers = oof_pr27 bit-identical, matrix =
    0.5 · (oof_pr27 + R) bit-identical (the form of the expected-value rule computation). -> (P, number of valid answers per row,
    statistics)."""
    C95 = c95()
    n, classes, seen, y = lc["n"], lc["classes"], lc["seen"], lc["y"]
    named, cnt = C95.A8.mean_of_draws([PROTO_TRAIN], n, classes, seen, table="train")
    pr27 = np.load(ART / "oof_pr27.npy")
    if not np.array_equal(named, pr27):
        raise TechnicalError("named pr27 codes (a8.mean_of_draws) != oof_pr27.npy")
    P = C95.rank_mean(named, R, np.ones(n, bool))
    if not np.array_equal(P, 0.5 * (pr27 + R)):
        raise TechnicalError("mean channel != 0.5 · (oof_pr27 + ranking) — not the matrix of the expected-value rule computation")
    if not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-9):
        raise TechnicalError("mean channel not finite or not normalised")
    rows = metric.map3_rows_from_proba(y, P)
    st = {"cv_named": float(metric.map3_rows_from_proba(y, named).mean()), "cv_rank": float(metric.map3_rows_from_proba(y, R).mean()),
          "none_true": float(np.mean(y == 0)), "none_top_rank": C95.top_none_share(R, np.ones(n, bool)),
          "none_top_named": C95.top_none_share(named, cnt > 0), "none_top_mean": C95.top_none_share(P, cnt >= 0),
          "same_top_named_rank": float(np.mean(named[cnt > 0].argmax(1) == R[cnt > 0].argmax(1))), "rows_table": int(n), "rows_rank": int(n),
          "rows_without_answer": int((cnt == 0).sum()), "rows_rank_without_answer": int((cnt == 0).sum()), "share_rank": 1.0, "partial": False}
    return P, cnt, {"rows": rows, **st}


def mean_meta(lc: dict, P: np.ndarray, cnt: np.ndarray, st: dict, src: dict) -> dict:
    from kaggle_core.folds import folds_hash

    rows = st["rows"]
    tn = mp().PX27_NOTES["test"]
    notes_tr = COMP / "data" / c132().NOTES_REL
    return {"folds_hash": folds_hash(lc["folds"]), "model_type": "vlm-protocol", "sources": [PROTO_TRAIN], "cv_score": float(rows.mean()),
            "cv_sigma": float(metric.bootstrap_sigma(rows)),
            "feature_set": "image+task, without task knowledge (mean of the named pr27 codes and the M3 ranking behind the same notes)",
            "rows_without_answer": int((cnt == 0).sum()), "neutral_rows": None, "rows_neutral": 0,
            "rank_mean": {"definition": DEFINITION, "proto_sources": [PROTO_TRAIN], "rank_source": c132().SOURCE, "floor": FLOOR,
                          "rank_compare": "sft",
                          "notes": {"train": {"sft_from": c132().NOTES_REL, "sha256": sha256_file(notes_tr), "upload": c132().NOTES_REL},
                                    "test": {"sft_from": c132().TEST_NOTES_REL, "upload": c132().TEST_NOTES_REL, "source_runs": [PROTO_TEST],
                                             "model": TEST_MODEL, "px27_file": tn[0], "px27_sha256": tn[1]},
                                    "note": "note guard tools/make_submission.py rank_mean_notes_guard; test notes with python -I tools/make_m3px27_notes.py "
                                            "--table test (character-identical with the input of the px27 test ranking)"},
                          "rank_artifact": src, "origin": "m3px27 rule: expected-value rule DEPLOY (eda/c132_m3px27_eval.py); deployment form primary_m3px27",
                          "function": "eda/c133_primary_m3px27.py oof: c95.rank_mean(a8.mean_of_draws(pr27), oof_m3px27)",
                          **{k: v for k, v in st.items() if k != "rows"}}}


def copy_meta() -> dict:
    """Meta fields of the copy channel: those of meta_pr27xm.json (without name/oof_shape/test_shape/oof_hash) and a copy mark."""
    m = json.loads((ART / f"meta_{SLOT}.json").read_text())
    out = {k: v for k, v in m.items() if k not in ("name", "oof_shape", "test_shape", "oof_hash")}
    out["copy_null_point"] = {"of": SLOT, "note": "copy null point (eda/c133_primary_m3px27.py copy): the path of primary_m3px27 with px27 instead of "
                                                  "m3px27 — temporary, never to be submitted"}
    return out


def meta_text(name: str, P: np.ndarray, meta: dict, K: int) -> str:
    from kaggle_core import artifacts as KA

    full = {"name": name, "oof_shape": list(P.shape), "test_shape": [0, K], "oof_hash": KA._array_hash(P), **meta}
    return json.dumps(full, indent=2, ensure_ascii=False, default=str)


def save_or_verify(name: str, P: np.ndarray, meta: dict, K: int, save: bool) -> dict:
    from kaggle_core import artifacts as KA

    f, mf = ART / f"oof_{name}.npy", ART / f"meta_{name}.json"
    text = meta_text(name, P, meta, K)
    same = f.is_file() and np.load(f).dtype == np.float64 and np.array_equal(np.load(f), P)
    same_meta = mf.is_file() and mf.read_text(encoding="utf-8") == text
    if not save:
        if not f.is_file() or not mf.is_file():
            raise Missing(f"{rel(f)} or {rel(mf)} missing — build them with --save first")
        if not (same and same_meta):
            raise TechnicalError(f"{rel(f)} / {rel(mf)} is not the recomputation (values {'equal' if same else 'DIFFERENT'}, meta {'equal' if same_meta else 'DIFFERENT'})")
    else:
        if f.is_file() and not same:
            raise TechnicalError(f"{rel(f)} exists with different values — not overwritten (delete it deliberately, then rebuild)")
        if mf.is_file() and not same_meta:
            raise TechnicalError(f"{rel(mf)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        KA.save_oof(name, P, np.zeros((0, K)), meta, ART)
        if mf.read_text(encoding="utf-8") != text:
            raise TechnicalError("meta file not as expected after writing (save_oof changed?)")
    return {"oof": rel(f), "oof_sha256": sha256_file(f), "meta": rel(mf), "meta_sha256": sha256_file(mf)}


def cmd_oof(save: bool) -> int:
    lc = light()
    R, src = rank_artifact()
    P, cnt, st = mean_channel(lc, R)
    meta = mean_meta(lc, P, cnt, st, src)
    info = save_or_verify(CH, P, meta, lc["K"], save)
    # check in memory: the same path with px27 hits oof_pr27xm bit for bit
    Pk, _, _ = mean_channel(lc, np.load(ART / "oof_pr27x.npy"))
    if not np.array_equal(Pk, np.load(ART / f"oof_{SLOT}.npy")):
        raise TechnicalError("check: the same path with px27 does not hit oof_pr27xm.npy bit for bit")
    say(f"{CH}: 0.5 · (pr27 named + m3px27) — MAP@3 alone {meta['cv_score']:.4f} ± {meta['cv_sigma']:.4f} (named {st['cv_named']:.4f}, ranking "
        f"{st['cv_rank']:.4f}); NONE at the top {st['none_top_mean']:.3f}; rows without a valid answer {st['rows_without_answer']}; check with px27 = "
        f"oof_pr27xm bit-identical; {'saved' if save else 'VERIFIED'}: {info['oof']} ({info['oof_sha256'][:16]}…), {info['meta']} ({info['meta_sha256'][:16]}…)")
    return 0


# ------------------------------------------------------------------------------------------------ blend

def build_stack(name: str, P: np.ndarray) -> tuple[dict, dict]:
    """Blend file of the final primary form with `name` instead of pr27xm (procedure as c110 build_replaced_beta). Requires
    oof_<name>.npy = P."""
    X = c110()
    C107 = X.C107
    f = ART / f"oof_{name}.npy"
    if not f.is_file() or not np.array_equal(np.load(f), P):
        raise Missing(f"{rel(f)} missing or not the matrix of the channel — run oof --save or copy first")
    ref_file = ART / f"stack_{STEM_REF}.json"
    ref = json.loads(ref_file.read_text())
    ctx = X.setup()
    y, folds, seen, n, K = ctx["y"], ctx["folds"], ctx["seen"], ctx["n"], ctx["K"]
    mats, sha16, metas = X.load_mats(ctx, "m3x3_full")
    Pq, sha_q, _, _ = X.load_replacement(ctx, "q397m3")
    mats["q397m3"], sha16["q397m3"] = Pq, sha_q
    mats[name], sha16[name] = P, sha256_file(f)[:16]
    _, spec_b = X.line_specs("best", "m3x3_full")
    spec_r = X.replaced_spec(spec_b, X.CH, "q397m3")  # recipe of stack_s032_stage2.json (final primary form)
    if spec_r["names"] != list(ref["channels"]):
        raise TechnicalError(f"recipe {spec_r['names']} != channels of {ref_file.name}")
    spec_n = X.replaced_spec(spec_r, SLOT, name)
    th, th0, L = X.fit_all(mats, spec_n, y, seen)
    beta, lls = X.choose_beta(th, L, y, seen)
    spec_s = {**spec_n, "beta": beta}
    draws = C107.question_draws(ctx["q_code"])
    fit_s = C107.cross_fit(y, folds, seen, mats, spec_s)
    fit_02 = C107.cross_fit(y, folds, seen, mats, spec_n)
    fit_ref = C107.cross_fit(y, folds, seen, mats, {**spec_r, "beta": float(ref["arith_beta"])})
    r_s, r_02, r_ref = (C107.rows_map(y, fx["Q"], seen) for fx in (fit_s, fit_02, fit_ref))
    ll_s, ll_02, ll_ref = (C107.rows_ll(y, fx["Q"]) for fx in (fit_s, fit_02, fit_ref))
    if int(C107.units(r_ref).sum()) != int(round(float(ref["cv"]) * 6 * n)):
        raise TechnicalError(f"cross-fit of the final primary blend {C107.units(r_ref).sum()}/{6 * n} does not hit the field cv of {ref_file.name}")
    cmp_ref, cmp_02 = C107.compare(r_s, r_ref, ctx, draws, ll_s, ll_ref), C107.compare(r_s, r_02, ctx, draws, ll_s, ll_02)
    pre_s, set_s = C107.rows_map(y, fit_s["P"], seen), C107.rows_map(y, C107.set_align(fit_s["P"], y, seen), seen)
    c = {"map3": float(r_s.mean()), "units": int(C107.units(r_s).sum()), "logloss": float(ll_s.mean()), "map3_pre_last_step": float(pre_s.mean()),
         "map3_one_set": float(set_s.mean()), "per_fold": [float(r_s[folds == k].mean()) for k in range(C107.N_FOLDS)]}
    d, d0 = X.summarize_cross(cmp_ref), X.summarize_cross(cmp_02)
    ren = lambda k: name if k == SLOT else k  # noqa: E731
    stack = copy.deepcopy(ref)
    stack["channels"] = list(spec_n["names"])
    stack["theta"] = [float(v) for v in th]
    stack["floor"] = {ren(k): float(v) for k, v in ref["floor"].items()}
    stack["base"] = {"channels": list(spec_n["names"]), "theta": [float(v) for v in th0]}
    if X.rank_only_entry("m3x3_full", metas["m3x3_full"], sha16["m3x3_full"], n) != ref["rank_only"]["m3x3_full"]:
        raise TechnicalError(f"rank_only entry m3x3_full differs from {ref_file.name}")
    if (ref["rank_only"].get("q397m3") or {}).get("oof_sha256") != sha_q:
        raise TechnicalError(f"rank_only entry q397m3 of {ref_file.name} names a different artifact")
    stack["arith_beta"] = float(beta)
    stack["cv"] = c["map3"]
    stack["cv_note"] = (f"cross-fit per fold on 1,500 rows WITH the last step at the arithmetic share β {beta:g} (weights per fold from the other "
                        f"four, eda/c107_q397_full.py cross_fit; β by arith_beta_rule from all 1,500 rows) with {name} instead of pr27xm; final "
                        f"primary blend {float(ref['cv']):.7f}, Δ {d['delta']:+.5f} ± {d['se_q']:.5f} over tasks (z {d['z_q']:+.2f}); at β 0.2 "
                        f"{float(r_02.mean()):.7f}; base OOFs not fully nested in the meta fold; decision on the second stage: m3px27 rule (c132)")
    stack["cross"] = {"map3": c["map3"], "logloss": c["logloss"], "per_fold": c["per_fold"], "map3_pre_last_step": c["map3_pre_last_step"],
                      "map3_one_set": c["map3_one_set"], "primary_final": float(ref["cv"]), "beta_0_2": float(r_02.mean()),
                      "delta_vs_primary_final": {k: d[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}}
    stack["oof_sha256"] = {nm: sha16[nm] for nm in spec_n["names"]}
    prev = ref.get("replaces")
    stack["replaces"] = {"channel": SLOT, "by": name, "definition": DEFINITION if name == CH else f"copy of {SLOT} (copy null point)",
                         "stack_primary_final": ref_file.name, "stack_primary_final_sha256": sha256_file(ref_file),
                         "decision": "m3px27 rule (expected-value rule: DEPLOY)",
                         "c132": rel(C132_JSON) if C132_JSON.is_file() else None, "previous": prev}
    stack["note"] = (f"eda/c133_primary_m3px27.py stack: recipe of the final primary form ({ref_file.name}) with {name} instead of pr27xm (same "
                     f"position, floor {FLOOR:g}); weights of ALL channels on log loss from all {n} rows (c110.fit_all, then {spec_n['think']} × "
                     f"{spec_n['factor']:g}); β by arith_beta_rule; build script tools/builds/primary_m3px27.sh")
    allr = np.arange(n)
    P_ms = X.MS.oof_mixture(stack, seen, ART, n_rows=n)
    Lm = C107.logs_of(mats, spec_s["names"], spec_s["floors"], seen, True)
    P_mask, P_way = X.MS.mix_channels(stack, Lm, None, seen), X.MS.mix_channels(stack, L, None, seen)
    d_mask, d_way = float(np.abs(P_ms - P_mask).max()), float(np.abs(P_ms - P_way).max())
    if d_mask != 0.0 or d_way > 1e-9:
        raise TechnicalError(f"check oof_mixture: deviation {d_mask:.1e} / {d_way:.1e}")
    Q_in = C107.apply_last(P_way, X.C21.fit_weights(P_way, C107.target_of(y, allr, K)), seen)
    stack["insample"] = {"map3": float(C107.rows_map(y, Q_in, seen).mean()), "map3_pre_last_step": float(C107.rows_map(y, P_way, seen).mean()),
                         "logloss": float(C107.rows_ll(y, Q_in).mean())}
    rule = dict(ref["arith_beta_rule"])
    rule.update({"grid": list(X.BETA_GRID), "logloss_pre_last_step": lls, "chosen": float(beta), "selected": True,
                 "stack_beta_0_2": None, "stack_beta_0_2_sha256": None,
                 "cross_beta_0_2": {"map3": float(r_02.mean()), "units": int(C107.units(r_02).sum()), "logloss": float(ll_02.mean())},
                 "delta_vs_beta_0_2": {k: d0[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}})
    stack["arith_beta_rule"] = rule
    if not C107.sane(Q_in, seen) or not C107.sane(P_ms, seen):
        raise TechnicalError("predictions not normalised or with mass on classes without a training example")
    rep = {"names": spec_n["names"], "beta": float(beta), "cross": c, "delta_vs_primary_final": d, "delta_vs_beta_0_2": d0,
           "theta": dict(zip(spec_n["names"], map(float, th))), "insample": stack["insample"],
           "check_oof_mixture": {"max_abs_mask_first": d_mask, "max_abs_test_way": d_way}}
    return stack, rep


STACK_NUMBERS = ("theta", "arith_beta", "cv")


def stack_numbers_equal(a: dict, b: dict) -> list[str]:
    """Numbers of the blend (copy null point): θ, base θ, floor values, β, cv, cross-fit, in-sample and β-rule numbers."""
    bad = [k for k in STACK_NUMBERS if a[k] != b[k]]
    if a["base"]["theta"] != b["base"]["theta"]:
        bad.append("base.theta")
    if list(a["floor"].values()) != list(b["floor"].values()) or a["scaled"] != b["scaled"] or a["rank_only"] != b["rank_only"]:
        bad.append("floor/scaled/rank_only")
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


def write_or_verify_text(path: Path, text: str, save: bool) -> str:
    if save:
        if path.is_file() and path.read_text() != text:
            raise TechnicalError(f"{rel(path)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        if not path.is_file():
            raise Missing(f"{rel(path)} missing — build it with --save first")
        if path.read_text() != text:
            raise TechnicalError(f"{rel(path)} is not the recomputation")
    return sha256_file(path)


def cmd_stack(save: bool) -> int:
    P = np.load(ART / f"oof_{CH}.npy") if (ART / f"oof_{CH}.npy").is_file() else None
    if P is None:
        raise Missing(f"artifacts/oof_{CH}.npy missing — run oof --save first")
    stack, rep = build_stack(CH, P)
    sha = write_or_verify_text(ART / f"stack_{STEM}.json", to_json(stack), save)
    d = rep["delta_vs_primary_final"]
    say(f"stack_{STEM}.json: channels {stack['channels']}; β {rep['beta']:g}; θ " + " · ".join(f"{k} {v:.4f}" for k, v in rep["theta"].items()))
    say(f"   cross-fit {rep['cross']['map3']:.5f} ({rep['cross']['units']}/9000) against the final primary blend Δ {d['delta']:+.5f} ± {d['se_q']:.5f} "
        f"(z {d['z_q']:+.2f}); without the last step {rep['cross']['map3_pre_last_step']:.5f}, one set {rep['cross']['map3_one_set']:.5f}; oof_mixture check "
        f"{rep['check_oof_mixture']['max_abs_mask_first']:.1e}/{rep['check_oof_mixture']['max_abs_test_way']:.1e}; "
        f"{'saved' if save else 'VERIFIED'} (SHA-256 {sha})")
    return 0


# ------------------------------------------------------------------------------------------------ chain of the second stage

def chain_ctx(ctx0: dict, name: str, P: np.ndarray, stem: str) -> dict:
    """Context of the final primary form (c124.load_context) with `name` (matrix P) in place of pr27xm and the blend file
    stack_<stem>.json."""
    C, S2 = c118(), c124().S2
    stack_file = ART / f"stack_{stem}.json"
    if not stack_file.is_file():
        raise Missing(f"{rel(stack_file)} missing — run stack --save (or copy) first")
    stack = json.loads(stack_file.read_text())
    spec = C.C115.replaced(ctx0["spec"], SLOT, name)
    floors = {k: float(v) for k, v in spec["floors"].items() if v}
    if stack.get("channels") != list(spec["names"]) or {k: float(v) for k, v in (stack.get("floor") or {}).items()} != floors:
        raise TechnicalError(f"{stack_file.name}: channels/floors {stack.get('channels')} / {stack.get('floor')} do not match the recipe {spec['names']}")
    if (stack.get("scaled") or {}) != {spec["think"]: float(spec["factor"])}:
        raise TechnicalError(f"{stack_file.name}: reasoning factor {stack.get('scaled')}")
    if not np.array_equal(np.load(ART / f"oof_{name}.npy"), P):
        raise TechnicalError(f"artifacts/oof_{name}.npy is not the matrix of the channel")
    ctx = dict(ctx0)
    ctx["mats"] = {**ctx0["mats"], name: np.asarray(P, dtype=np.float64)}
    ctx.update(spec=spec, stack=stack, stack_file=stack_file, stack_stem=stem, beta=float(stack.get("arith_beta") or 0.0))
    st = S2.Static(spec["names"], ctx["mats"], ctx["draws"], ctx["grade"], ctx["seen"], ctx["classes"], ctx["notes"])
    S2.attach_mech(st, ctx["mech27"], ctx["minor"])
    ctx["static"] = st
    return ctx


def common_entry(r: dict) -> dict:
    return {"map_forced": float(r["forced"].mean()), "map_abstain": float(r["abstain"].mean()),
            "abstain_folds": int(sum(c["abstain"] for c in r["choice"].values())), "gate": {str(k): v for k, v in r["choice"].items()},
            "n_features": len(r["features"]), "changed_top_share": float(r["changed_top"].mean()),
            "changed_top_share_forced": float(r["changed_top_forced"].mean()), "changed_top3_share_forced": float(r["changed_top3_forced"].mean())}


def oof_chain(ctx: dict, log=say) -> tuple[dict, dict, dict]:
    """Nested reproduction of the whole chain (as c118 oof, c118 guard-oof, c121 oof, c124 oof; without placebos and meta-splits).
    -> (result as the OOF files of the chain, row values, standard cross-fit of the evaluated rows (β per outer fold by log loss))."""
    C, M121, M124, S2 = c118(), c121(), c124(), c124().S2
    C107 = C.C107
    t0 = time.time()
    y, seen = ctx["y"], ctx["seen"]
    draws_q = C107.question_draws(ctx["q_code"])
    base_b = C.C115.run_recipe(ctx, ctx["mats"], C.C115.spec_base())
    anchor = {"fold": base_b["rows"], "pre": base_b["rows_pre"], "set": base_b["rows_set"]}
    for f in FORMS3:
        if int(C107.units(anchor[f]).sum()) != C.ANCHOR_UNITS[f]:
            raise TechnicalError(f"anchor B+ [{f}] does not hit {C.ANCHOR_UNITS[f]}/9000")
    own = C.C115.run_recipe(ctx, ctx["mats"], ctx["spec"])
    own_rows = {"fold": own["rows"], "pre": own["rows_pre"], "set": own["rows_set"]}
    std, betas, nested = M121.prepare(ctx)
    rec_rows = {f: C107.rows_map(y, std[f], seen) for f in FORMS3}
    res = {"kind": "nested reproduction of the chain main form -> final primary form with the swapped channel (c118/c121/c124, without placebos)",
           "stack": rel(ctx["stack_file"]), "stack_sha256": sha256_file(ctx["stack_file"]), "channels": list(ctx["spec"]["names"]), "beta_outer": betas,
           "beta_stack": ctx["beta"], "K": S2.K_CAND, "lambda": S2.LAMBDA, "anchor_map": {f: float(anchor[f].mean()) for f in FORMS3},
           "own_recipe_map": {f: float(own_rows[f].mean()) for f in FORMS3}, "recipe_beta_map": {f: float(rec_rows[f].mean()) for f in FORMS3},
           "sets": {}, "inputs": C.inputs_sha256(ctx)}
    rows, keep = {}, {}

    def put(fs: str, form: str, r: dict, ent: dict) -> None:
        if form == "set":
            ent["eval_features_sha256"] = C.sha256_array(C.eval_rows_in_table_order(ctx, r["X_eval"]))
            ent["features"] = r["features"]
        res["sets"].setdefault(fs, {})[form] = ent
        keep.setdefault(fs, {})[form] = r
        for key in ("forced", "abstain", "top", "cand", "protected"):
            if key in r:
                rows[f"{fs}_{form}_{key}"] = r[key]

    for fs in S2.FEATURE_SETS:  # main form and fallback (c118 oof)
        for form in FORMS3:
            r = C.run_nested(ctx, std, nested, form, fs)
            if not np.allclose(r["base"], rec_rows[form]):
                raise TechnicalError(f"base of the candidate order [{fs}|{form}] != recipe")
            ent = common_entry(r)
            for mode in ("forced", "abstain"):
                ent[mode] = {"vs_anchor": C.boot(r[mode] - anchor[form], ctx, draws_q), "vs_own_recipe": C.boot(r[mode] - own_rows[form], ctx, draws_q),
                             "vs_recipe_beta": C.boot(r[mode] - rec_rows[form], ctx, draws_q)}
            put(fs, form, r, ent)
        log(f"   [{fs}] MAP F {res['sets'][fs]['fold']['map_forced']:.6f} oL {res['sets'][fs]['pre']['map_forced']:.6f} M {res['sets'][fs]['set']['map_forced']:.6f} "
            f"({time.time() - t0:.0f} s)")
    G, null = S2.GUARD_SET, {}
    for form in FORMS3:  # main_guard (c118 guard-oof) with copy null point (i)
        r, m = C.run_nested(ctx, std, nested, form, G), keep["main"][form]
        z = C.run_nested(ctx, std, nested, form, G, ids_zero=True)
        ok = bool(all(np.array_equal(z[k], m[k]) for k in ("forced", "abstain", "top", "cand")))
        null.setdefault("main_guard", {})[form] = ok
        ent = {**common_entry(r), "vs_recipe_beta": C.boot(r["forced"] - rec_rows[form], ctx, draws_q),
               "vs_main": C.boot(r["forced"] - m["forced"], ctx, draws_q), "vs_main_with_abstention": C.boot(r["abstain"] - m["abstain"], ctx, draws_q),
               "rows_protected": int(r["protected"].sum()), "rows_other_top1_vs_main": int((r["top"][:, 0] != m["top"][:, 0]).sum()),
               "null_ids_zero_equals_main": ok}
        put(G, form, r, ent)
    for form in FORMS3:  # I1 (c121 oof) with copy null point (i)
        r, s = M121.run_i1(ctx, std, nested, form, "real"), keep[G][form]
        ok = M121.same_rows(M121.run_i1(ctx, std, nested, form, "zero"), s)
        null.setdefault("i1", {})[form] = ok
        d = r["forced"] - s["forced"]
        ent = {**common_entry(r), "vs_guard": C.boot(d, ctx, draws_q), "vs_guard_with_abstention": C.boot(r["abstain"] - s["abstain"], ctx, draws_q),
               "vs_recipe_beta": C.boot(r["forced"] - rec_rows[form], ctx, draws_q), "rows_protected": int(r["protected"].sum()),
               "rows_other_top1_vs_guard": int((r["top"][:, 0] != s["top"][:, 0]).sum()),
               "rows_other_top3_vs_guard": int((r["top"][:, :3] != s["top"][:, :3]).any(1).sum()), "rows_score_changed_vs_guard": int((d != 0).sum()),
               "null_i1_zero_equals_guard": ok}
        put("i1", form, r, ent)
    for form in FORMS3:  # MM (c124 oof) with copy null point (i)
        r, s = M124.run_mm(ctx, std, nested, form, "real"), keep["i1"][form]
        ok = M124.same_rows(M124.run_mm(ctx, std, nested, form, "zero"), s)
        null.setdefault("mm", {})[form] = ok
        d = r["forced"] - s["forced"]
        ent = {**common_entry(r), "vs_i1": C.boot(d, ctx, draws_q), "vs_i1_with_abstention": C.boot(r["abstain"] - s["abstain"], ctx, draws_q),
               "vs_recipe_beta": C.boot(r["forced"] - rec_rows[form], ctx, draws_q), "rows_protected": int(r["protected"].sum()),
               "rows_other_top1_vs_i1": int((r["top"][:, 0] != s["top"][:, 0]).sum()),
               "rows_other_top3_vs_i1": int((r["top"][:, :3] != s["top"][:, :3]).any(1).sum()), "null_zero_equals_i1": ok}
        put("mm", form, r, ent)
    res["null_point_i"] = {k: {"ok": bool(all(v.values())), "forms": v} for k, v in null.items()}
    if not all(v["ok"] for v in res["null_point_i"].values()):
        raise TechnicalError(f"copy null point (i) of the chain violated: {res['null_point_i']}")
    res["summary"] = {fs: {f: res["sets"][fs][f]["map_forced"] for f in FORMS3} for fs in res["sets"]}
    log(f"   chain nested: MM F {res['summary']['mm']['fold']:.6f} M {res['summary']['mm']['set']:.6f}; null points (i) bit-identical; β per outer "
        f"fold {betas} (blend file {ctx['beta']:g}) ({time.time() - t0:.1f} s; result without run time, so that --verify checks byte for byte)")
    return res, rows, std


def check_rows(rows: dict, ref: dict[str, np.ndarray], label: str) -> dict:
    """Row values of the chain against a reference {key: rows} (bit-identical required)."""
    bad = [k for k, v in ref.items() if k not in rows or not np.array_equal(rows[k], v)]
    if bad:
        raise TechnicalError(f"{label}: row values {bad} not bit-identical")
    return {"label": label, "keys": sorted(ref), "bit_identical": True}


def model_main(ctx: dict, forms: dict, feature_set: str, prev: dict, extra_feats: tuple, oof_set: dict, vs_key: str, label: str) -> dict:
    """Main model of one step exactly as c118 guard-fit / c121 fit / c124 fit: deployment features = evaluated OOF features (null point
    (e)), first features = features of the predecessor bit-identical, gate with the guard, conditional logit, NONE guard."""
    C, S2 = c118(), c124().S2
    y, Kc = ctx["y"], S2.K_CAND
    X, cand, names = C.deploy_features(ctx, forms, feature_set)
    got = C.sha256_array(X)
    want = (oof_set.get("set") or {}).get("eval_features_sha256")
    if got != want:
        raise TechnicalError(f"null point (e) violated [{feature_set}]: deployment features ({got[:16]}…) != evaluated OOF features ({str(want)[:16]}…)")
    nm = len(prev["features"])
    if names[:nm] != prev["features"] or names[nm:] != list(extra_feats) or C.sha256_array(X[:, :nm]) != prev["features_sha256"]:
        raise TechnicalError(f"the first features of {feature_set} are not the features of the predecessor (bit-identical)")
    lab = S2.labels_of(cand, y)
    g = S2.gate(C.fold_blocks(ctx, X, cand, lab), protect=True)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top0 = S2.reorder(cand, S2.scores(model, X, Kc))
    top = S2.protect_none(top0, cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    return {"features": names, "feature_set": feature_set, "protect_none": True, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
            "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
            "oof_changed_top_share": float(oof_set["set"]["changed_top_share_forced"]),
            "oof_changed_top_share_with_abstention": float(oof_set["set"]["changed_top_share"]),
            "oof_changed_top3_share": float(oof_set["set"]["changed_top3_share_forced"]),
            vs_key: {f: oof_set[f][vs_key.replace("oof_", "")]["delta"] for f in FORMS3},
            "oof_vs_recipe_beta": {f: oof_set[f]["vs_recipe_beta"]["delta"] for f in FORMS3}, "features_sha256": got, "insample": ins}


def deploy_forms_of(ctx: dict, res: dict, std: dict) -> tuple[dict, dict]:
    """DEPLOYMENT FORM OF THE TRAINING ROWS of this chain = the standard cross-fit of the nested reproduction (c118.standard_forms with β
    "select" per fold: θ and β from the other four folds), recomputed and required bit-identical to that of the reproduction. For the final
    primary form every outer fold chose β 0.6 = β of the blend file; there this form is bit-identical to c118.deploy_forms (cross-fit with the
    fixed β of the blend file) — the copy null point checks this. With m3px27xm fold 1 chooses β 0.7: then the standard cross-fit is the
    deployment form under which null point (e) (deployment features = evaluated OOF features) holds and the expected-value rule computation
    (c132: training rows per fold with β "select") is the honest estimate. -> (forms, statistics)."""
    C = c118()
    C107 = C.C107
    y, seen, n = ctx["y"], ctx["seen"], ctx["n"]
    forms, betas = C.standard_forms(ctx, "select")
    if [float(b) for b in betas] != [float(b) for b in res["beta_outer"]] or not all(np.array_equal(forms[f], std[f]) for f in FORMS3):
        raise TechnicalError("standard cross-fit recomputed != that of the nested reproduction")
    fixed = C.deploy_forms(ctx)
    same = {f: bool(np.array_equal(forms[f], fixed[f])) for f in FORMS3}
    cand_s, cand_f = (c124().S2.candidates(F["set"], seen) for F in (forms, fixed))
    info = {"kind": "standard_forms_select", "beta_per_fold": [float(b) for b in betas], "beta_stack": float(ctx["beta"]),
            "folds_other_beta": [k for k, b in enumerate(betas) if float(b) != float(ctx["beta"])], "equals_fixed_beta_forms": same,
            "rows_other_candidates_set_vs_fixed": int((cand_s != cand_f).any(1).sum()),
            "rows_other_top1_set_vs_fixed": int((cand_s[:, 0] != cand_f[:, 0]).sum()),
            "units_fold": int(C107.units(C107.rows_map(y, forms["fold"], seen)).sum()),
            "units_fold_fixed_beta": int(C107.units(C107.rows_map(y, fixed["fold"], seen)).sum()), "rows": int(n),
            "note": "deployment form = standard cross-fit of the reproduction (β per fold by log loss from the other four folds); with the same β "
                    "in all folds bit-identical to c118.deploy_forms (main form … final primary form)"}
    return forms, {**info, "fixed": fixed}


def compose(ctx: dict, res: dict, std: dict, oof_file: Path, chain_dir: Path, tag: str, m3px27_build: dict) -> list[tuple[str, Path, dict]]:
    """Deployment files of the chain a (main form + fallback) -> b (widened rule) -> c (main_guard) -> d (main_guard_i1) -> e
    (main_guard_i1_mm), each step as the tool of the stored chain (c118 fit / c118 guard-fit / c121 fit / c124 fit, same fields and texts),
    with the deployment form deploy_forms_of. -> [(step, path, content)] (not yet written; provenance per step = path and checksum of the
    predecessor)."""
    C, M124, S2 = c118(), c124(), c124().S2
    n = ctx["n"]
    forms, dinfo = deploy_forms_of(ctx, res, std)
    m3px27_build["deploy_forms"] = {k: v for k, v in dinfo.items() if k != "fixed"}
    out_a, _ = C.build_stage2(ctx, {"sets": {fs: res["sets"][fs] for fs in S2.FEATURE_SETS}, "inputs": res["inputs"],
                                    "reproduction": res.get("reproduction"), "beta_outer": res["beta_outer"]},
                              forms=forms, cv_units_want=int(round(float(res["recipe_beta_map"]["fold"]) * 6 * n)))
    out_a["oof_reproduction"] = {"file": rel(oof_file), "reproduction": res.get("reproduction"), "beta_outer": res["beta_outer"]}
    paths = {s: chain_dir / f"stage2_{tag}_{s}.json" for s in "abcd"}
    texts = {"a": to_json(out_a)}
    out_b = {k: ({**v, "rule": "widened", "share_z_max": C.WIDENED_SHARE_Z_MAX, "n_ref": int(n), "rule_text": C.WIDENED_RULE_TEXT} if k == "guard_rule" else v)
             for k, v in out_a.items()}
    out_b["derived_from"] = {"file": rel(paths["a"]), "sha256": sha256_text(texts["a"]), "differs_only_in": ["guard_rule", "derived_from"],
                             "build": "tools/builds/s034_stage2_main.sh"}
    texts["b"] = to_json(out_b)
    G = S2.GUARD_SET
    gm = model_main(ctx, forms, G, out_b["models"]["main"], S2.GENERAL_ID_FEATS, res["sets"][G], "oof_vs_main", "c")
    out_c = {k: ({"main": gm, "fallback": v["fallback"]} if k == "models" else v) for k, v in out_b.items() if k != "derived_from"}
    out_c["guard_build"] = {"feature_set": G, "protect_none": True, "id_features": list(S2.GENERAL_ID_FEATS),
                            "oof": rel(oof_file), "null_point_i": res["null_point_i"]["main_guard"],
                            "fallback": f"models.fallback unchanged from {paths['b'].name} (fallback of this chain)"}
    out_c["derived_from"] = {"file": rel(paths["b"]), "sha256": sha256_text(texts["b"]), "differs_only_in": ["models.main", "guard_build", "derived_from"],
                             "build": "tools/builds/s035_stage2_guard.sh", "previous": out_b["derived_from"]}
    texts["c"] = to_json(out_c)
    im = model_main(ctx, forms, S2.I1_SET, out_c["models"]["main"], S2.I1_FEATS, res["sets"]["i1"], "oof_vs_guard", "d")
    out_d = {k: ({"main": im, "fallback": v["fallback"]} if k == "models" else v) for k, v in out_c.items() if k != "derived_from"}
    out_d["i1_build"] = {"rule": "expected-value rule", "feature_set": S2.I1_SET, "protect_none": True,
                         "interactions": {"row_features": list(S2.I1_ROW_FEATS), "candidate_features": list(S2.I1_CAND_FEATS), "features": list(S2.I1_FEATS),
                                          "definition": "product of the raw values X[r]·X[c] (row feature outer, candidate feature inner), at the end of "
                                                        "main_guard, standardised like all features"},
                         "oof": rel(oof_file), "splits": None,
                         "rule_oof": {"note": "recipe step of the I1 form (decided on the guard form) taken over unchanged; deployment of this chain "
                                              "under the m3px27 rule (m3px27_build)", "null_point_i": res["null_point_i"]["i1"]},
                         "reference_rows_test": "knowledge features of the test side (task_rows and others) from all 1,500 training rows as in the guard form, not rescaled",
                         "test_side": "eda/c121_stage2_i1.py submit admits the set only for the main model (checks as for main_guard) and calls "
                                      "make_submission.main() with the same arguments",
                         "fallback": f"models.fallback unchanged from {paths['c'].name} (fallback of this chain)"}
    out_d["derived_from"] = {"file": rel(paths["c"]), "sha256": sha256_text(texts["c"]), "differs_only_in": ["models.main", "i1_build", "derived_from"],
                             "build": "tools/builds/primary_i1.sh", "previous": out_c.get("derived_from")}
    texts["d"] = to_json(out_d)
    mmm = model_main(ctx, forms, S2.MM_SET, out_d["models"]["main"], S2.MM_FEATS, res["sets"]["mm"], "oof_vs_i1", "e")
    out_e = {k: ({"main": mmm, "fallback": v["fallback"]} if k == "models" else v) for k, v in out_d.items() if k != "derived_from"}
    out_e["mm_build"] = {"rule": "expected-value rule", "feature_set": S2.MM_SET, "protect_none": True, "feature": list(S2.MM_FEATS),
                         "definition": "1[default_severity of the candidate = minor] · 1[final answer read by the pr27 channel matches the answer key "
                                       "mechanically] (src/answer_match.py without eval), at the end of main_guard_i1, standardised like all features",
                         "oof": rel(oof_file), "splits": None,
                         "rule_oof": {"note": "recipe step of the final primary form (decided on the I1 form) taken over unchanged; deployment of this "
                                              "chain under the m3px27 rule (m3px27_build)", "null_point_i": res["null_point_i"]["mm"]},
                         "fallback": f"models.fallback unchanged from {paths['d'].name} (fallback of this chain)"}
    out_e["mech27"] = {"test": {"file": M124.TEST_ANSWERS, "sha256": sha256_file(COMP / M124.TEST_ANSWERS)},
                       "train": [{"file": f, "sha256": sha256_file(COMP / f)} for f in M124.TRAIN_ANSWERS],
                       "minor_codes": [c for c, v in zip(ctx["classes"], ctx["minor"]) if v], "train_share_mech": float(ctx["mech27"].mean()),
                       "reader": "src/answer_match.py: last_answers (last readable JSON answer per row), mech_rows (numeric value rel. 1e-6 or text)"}
    out_e["derived_from"] = {"file": rel(paths["d"]), "sha256": sha256_text(texts["d"]), "differs_only_in": ["models.main", "mm_build", "mech27", "derived_from"],
                             "build": "tools/builds/primary_final.sh", "previous": out_d.get("derived_from")}
    out_e["m3px27_build"] = m3px27_build
    return [(s, paths[s], d) for s, d in zip("abcd", (out_a, out_b, out_c, out_d))] + [("e", None, out_e)]


def normalized(d: dict, name: str) -> dict:
    """Comparison form for the copy null point: channel name -> pr27xm in channels and feature lists; provenance fields removed."""
    out = copy.deepcopy({k: v for k, v in d.items() if k not in PROVENANCE_TOP})
    ren = lambda s: s.replace(f"{name}_", f"{SLOT}_") if isinstance(s, str) and s.startswith(f"{name}_") else (SLOT if s == name else s)  # noqa: E731
    if "channels" in out:
        out["channels"] = [ren(c) for c in out["channels"]]
    for m in (out.get("models") or {}).values():
        m["features"] = [ren(f) for f in m["features"]]
    for b in ("guard_build", "i1_build", "mm_build"):
        if b in out:
            out[b] = {k: v for k, v in out[b].items() if k not in PROVENANCE_BUILD}
    return out


def diff_keys(a, b, path: str = "") -> list[str]:
    """Paths at which two JSON values differ (at most 20)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(dict.fromkeys(list(a) + list(b))):
            if k not in a or k not in b:
                out.append(f"{path}.{k} (missing)")
            else:
                out += diff_keys(a[k], b[k], f"{path}.{k}")
        return out[:20]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and any(isinstance(x, (dict, list)) for x in a):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_keys(x, y, f"{path}[{i}]")
        return out[:20]
    return [] if a == b else [path or "."]


def m3px27_build_field(res: dict, rev: dict | None, src: dict | None, kind: str) -> dict:
    return {"decision": "m3px27 rule (expected-value rule; primary role pr27xm' = ½(pr27 + m3px27))", "kind": kind,
            "channel": {"slot": SLOT, "by": CH if kind == "deploy" else CH_COPY},
            "rank_artifact": src, "rule_rows": rev, "null_point_i": res["null_point_i"], "beta_outer": res["beta_outer"],
            "rule": "deploy under the m3px27 rule: ½(F+M) on the original meta-split above max(0, controls) and the mean over 20 meta-splits > 0 "
                    "and above the control mean (c132); the chain takes over the recipe of the final primary form unchanged (main form and fallback, "
                    "widened rule, guard form, I1 form, final form)",
            "copy_null_point": "eda/c133_primary_m3px27.py copy: the same path with px27 instead of m3px27 hits the final primary form (blend, chain, "
                               "test side byte-identical)",
            "test_side": f"make_submission --proto {CH}={PROTO_TEST} --rank {CH}={TEST_RANK} (rank_compare sft, note guard rank_mean.notes)"}


def run_chain(name: str, P: np.ndarray, stem: str, tag: str, kind: str, src: dict | None, log=say) -> tuple[list, dict, dict, Path, dict]:
    ctx0 = c124().load_context(STEM_REF)
    ctx = chain_ctx(ctx0, name, P, stem)
    res, rows, std = oof_chain(ctx, log=log)
    oof_file = OUT / f"c133_oof_{tag}.json"
    rev = None
    if kind == "deploy":  # the MM row values F and M = those of the expected-value rule computation of c132 (bit-identical)
        if not C132_ROWS.is_file():
            raise Missing(f"{rel(C132_ROWS)} missing — run python -I eda/c132_m3px27_eval.py first")
        z = np.load(C132_ROWS)
        rev = check_rows(rows, {"mm_fold_forced": z["primary_final_m3px27_fold"], "mm_set_forced": z["primary_final_m3px27_set"]},
                         "expected-value rule (c132)")
        rev.update(file=rel(C132_ROWS), sha256=sha256_file(C132_ROWS))
    else:  # copy: the row values of the chain = the stored ones of the reference runs
        refs = {}
        for fs, (d, f, key) in REF_ROWS.items():
            zz = np.load(ART / "eda" / "private" / d / f)
            for form in FORMS3:
                refs[f"{fs}_{form}_forced"] = zz[key.format(f=form)]
        rev = check_rows(rows, refs, "stored reference rows stage2/stage2_guard/stage2_i1/stage2_mm")
    res["reproduction"] = {"kind": "expected-value rule rows (c132) bit-identical" if kind == "deploy" else "row values of the stored reference runs bit-identical", **rev}
    s2b = m3px27_build_field(res, rev, src, kind)
    steps = compose(ctx, res, std, oof_file, OUT / ("chain" if kind == "deploy" else "copy"), tag, s2b)
    if kind != "deploy" and not all(s2b["deploy_forms"]["equals_fixed_beta_forms"].values()):
        raise TechnicalError("copy null point: standard cross-fit != cross-fit with the fixed β of the blend file (the final primary form chose β 0.6 per fold)")
    return steps, res, rows, oof_file, ctx


SENS_STAGE2 = OUT / "sens" / "stage2_primary_m3px27_fixbeta.json"


def sensitivity_fixed(ctx: dict, e: dict) -> tuple[dict, dict]:
    """SENSITIVITY (report only, never deployed): the main model main_guard_i1_mm on the deployment form of c118.deploy_forms (cross-fit
    with the FIXED β of the blend file in all folds) instead of the standard cross-fit; fallback, guards and reference unchanged from the
    deployment file e. Null point (e) deliberately does not hold here (fold 1 chooses β 0.7 in the reproduction). -> (probe file,
    statistics)."""
    C, S2 = c118(), c124().S2
    y, Kc = ctx["y"], S2.K_CAND
    fixed = C.deploy_forms(ctx)
    X, cand, names = C.deploy_features(ctx, fixed, S2.MM_SET)
    m0 = e["models"]["main"]
    if names != m0["features"]:
        raise TechnicalError("sensitivity: feature names != deployment file")
    lab = S2.labels_of(cand, y)
    g = S2.gate(C.fold_blocks(ctx, X, cand, lab), protect=True)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top0 = S2.reorder(cand, S2.scores(model, X, Kc))
    top = S2.protect_none(top0, cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    main = {**m0, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]], "sd": [float(v) for v in model["sd"]], "gate": g,
            "abstain": g["abstain"], "features_sha256": C.sha256_array(X), "insample": ins}
    out = copy.deepcopy(e)
    out["models"]["main"] = main
    out["probe"] = True
    out["probe_note"] = ("SENSITIVITY (eda/c133_primary_m3px27.py fit): main model on c118.deploy_forms (fixed β of the blend file) instead of the "
                         "standard cross-fit of the reproduction; report only (aggregates), never to be submitted")
    w1, w0 = np.asarray(main["w"]), np.asarray(m0["w"])
    rep = {"abstain": bool(g["abstain"]), "max_abs_dw": float(np.abs(w1 - w0).max()), "corr_w": float(np.corrcoef(w1, w0)[0, 1]),
           "insample": ins, "insample_deploy": m0["insample"], "gate": {k: g[k] for k in ("inner", "inner_base") if k in g},
           "gate_deploy": {k: m0["gate"][k] for k in ("inner", "inner_base") if k in m0["gate"]}}
    return out, rep


def cmd_fit(save: bool) -> int:
    t0 = time.time()
    _, src = rank_artifact()
    P = np.load(ART / f"oof_{CH}.npy")
    steps, res, rows, oof_file, ctx = run_chain(CH, P, STEM, "primary_m3px27", "deploy", src)
    texts = [(s, (STAGE2 if p is None else p), to_json(d)) for s, p, d in steps]
    write_or_verify_text(oof_file, to_json(res), save)
    if save:
        np.savez_compressed(OUT / "c133_oof_rows.npz", **rows)
    shas = {s: write_or_verify_text(p, t, save) for s, p, t in texts}
    e = steps[-1][2]
    m, fb, df = e["models"]["main"], e["models"]["fallback"], e["m3px27_build"]["deploy_forms"]
    say(f"stage2_primary_m3px27.json: main_guard_i1_mm ({len(m['features'])} features, guard on; gate {m['gate']['inner']:.5f} vs. {m['gate']['inner_base']:.5f} -> "
        f"{'ABSTAIN' if m['abstain'] else 'apply'}; OOF new top code {m['oof_changed_top_share']:.4f}; in-sample {m['insample']['map_base']:.5f} → "
        f"{m['insample']['map_reranked']:.5f}); fallback {len(fb['features'])} features; β {e['arith_beta']:g}; MM rows = expected-value rule rows (c132) bit-identical")
    say(f"   deployment form of the training rows: standard cross-fit of the reproduction, β per fold {df['beta_per_fold']} (blend file {df['beta_stack']:g}); "
        f"against the fixed β: other candidates in {df['rows_other_candidates_set_vs_fixed']} rows (other top code {df['rows_other_top1_set_vs_fixed']}), form F "
        f"{df['units_fold']}/9000 vs. {df['units_fold_fixed_beta']}/9000")
    sens, srep = sensitivity_fixed(ctx, e)
    write_or_verify_text(SENS_STAGE2, to_json(sens), save)
    write_or_verify_text(OUT / "sens" / "c133_sens_fixbeta.json", to_json(srep), save)
    say(f"   sensitivity (fixed β, report only): max |Δw| {srep['max_abs_dw']:.4f}, correlation of the weights {srep['corr_w']:.4f}; in-sample "
        f"{srep['insample']['map_reranked']:.5f} vs. {srep['insample_deploy']['map_reranked']:.5f} -> {rel(SENS_STAGE2)}")
    say(f"   chain a–e {'saved' if save else 'VERIFIED'}: " + "; ".join(f"{s} {sha[:16]}…" for s, sha in shas.items()) + f" ({time.time() - t0:.0f} s)")
    return 0


COPY_FILES = (f"oof_{CH_COPY}.npy", f"test_{CH_COPY}.npy", f"meta_{CH_COPY}.json", f"stack_{STEM_COPY}.json")


def clean_copy() -> list[str]:
    gone = []
    for f in COPY_FILES:
        p = ART / f
        if p.is_file():
            p.unlink()
            gone.append(f)
    return gone


def cmd_copy(keep: bool) -> int:
    """COPY NULL POINT of the training side: mean channel with px27 (= oof_pr27xm bit-identical), blend (numbers = stack_s032_stage2.json),
    chain (rows = stored reference runs, files = stage2_s032/s034/s035/primary_i1/primary_final.json except provenance and channel names).
    --keep: the copy files under artifacts/ stay for the copy run of the test side (build script)."""
    t0 = time.time()
    lc = light()
    clean_copy()
    try:
        P, _, _ = mean_channel(lc, np.load(ART / "oof_pr27x.npy"))
        if not np.array_equal(P, np.load(ART / f"oof_{SLOT}.npy")):
            raise TechnicalError("copy channel != oof_pr27xm.npy")
        save_or_verify(CH_COPY, P, copy_meta(), lc["K"], True)
        stack, _ = build_stack(CH_COPY, P)
        ref = json.loads((ART / f"stack_{STEM_REF}.json").read_text())
        bad = stack_numbers_equal(stack, ref)
        if bad:
            raise TechnicalError(f"copy null point of the blend violated: {bad}")
        (ART / f"stack_{STEM_COPY}.json").write_text(to_json(stack))
        say(f"copy null point (blend): stack_{STEM_COPY}.json hits θ, base θ, floors, β, cv, cross-fit, in-sample and β-rule numbers of "
            f"stack_{STEM_REF}.json exactly")
        steps, res, rows, oof_file, _ = run_chain(CH_COPY, P, STEM_COPY, "primary_m3px27_copy", "copy", None)
        OUT.mkdir(parents=True, exist_ok=True)
        oof_file.write_text(to_json(res))
        comp = {}
        for s, p, d in steps:
            refd = json.loads((ART / REF_CHAIN[s]).read_text())
            dk = diff_keys(normalized(d, CH_COPY), normalized(refd, SLOT))
            comp[s] = {"reference": REF_CHAIN[s], "equal": not dk, "diff": dk}
            path = (OUT / "copy" / f"stage2_primary_m3px27_copy_{s}.json") if p is None else p
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(to_json(d))
        bad = {s: v["diff"] for s, v in comp.items() if not v["equal"]}
        rep = {"kind": "copy null point of the training side (m3px27 := px27)", "stack": f"stack_{STEM_COPY}.json", "chain": comp,
               "rows": res["reproduction"], "seconds": round(time.time() - t0, 1)}
        (OUT / "c133_copy.json").write_text(to_json(rep))
        if bad:
            raise TechnicalError(f"copy null point of the chain violated: {bad}")
        say(f"copy null point (chain): row values = stage2/stage2_guard/stage2_i1/stage2_mm bit-identical; files a–e = {', '.join(REF_CHAIN.values())} "
            f"(except provenance and channel names) -> deployment file of the copy {rel(OUT / 'copy' / 'stage2_primary_m3px27_copy_e.json')} ({rep['seconds']} s)")
    except Exception:
        if not keep:
            clean_copy()
        raise
    if not keep:
        gone = clean_copy()
        say(f"   temporary copy files removed: {', '.join(gone)}")
    else:
        say(f"   copy files for the copy run of the test side lie under artifacts/: {', '.join(COPY_FILES)} (then clean-copy)")
    return 0


# ------------------------------------------------------------------------------------------------ test side

def adapter_recipe(run: str, model: str) -> tuple[dict | None, dict, list[str]]:
    """Model of a run -> merge job (run configuration) -> adapter_from -> sft_recipe.json; local adapter = merged one."""
    A9, C108 = c132().c115().A9, c132().c115().C108
    cfg, src = C108.run_config(run)
    where: dict = {"run": run, "model": model, "config": src}
    if cfg is None:
        return None, where, [f"run configuration configs/vlm/{run}.yaml missing"]
    merged = ((cfg.get("models") or {}).get(model) or {}).get("merged")
    jobs = [jb for w in cfg.get("workers") or [] for jb in w.get("jobs") or []]
    mj = [jb for jb in jobs if jb.get("kind") == "merge" and merged and jb.get("out_model") == merged]
    if len(mj) != 1:
        return None, where, [f"no unique merge job for model {model!r} in {src}"]
    ad = str(mj[0]["adapter_from"])
    r2, j2 = ad[3:].split("/", 1) if ad.startswith("../") else (run, ad)
    where.update(merge_job=mj[0]["name"], train=f"{r2}/{j2}")
    rec = A9.read_json(A9.run_dir(r2) / j2 / "sft_recipe.json")
    if rec is None:
        return None, where, [f"artifacts/runs/{r2}/{r2}/{j2}/sft_recipe.json missing"]
    problems = []
    summ = A9.read_json(A9.run_dir(run) / mj[0]["name"] / "summary.json")
    want = ((summ or {}).get("merge") or {}).get("adapter_sha256")
    local = A9.run_dir(r2) / j2 / "adapter" / "adapter_model.safetensors"
    if not want or not local.is_file():
        problems.append(f"checksum of the merged adapter cannot be determined ({mj[0]['name']}/summary.json or local adapter missing)")
    elif sha256_file(local) != want:
        problems.append(f"local adapter {r2}/{j2} is not the one merged in the test run ({mj[0]['name']})")
    else:
        where["adapter_sha256"] = want
    return rec, where, problems


def training_regime_of(spec: str, n_train: int) -> tuple[list[str], dict]:
    """Test model = merge of the full fit belonging to the five fold adapters of the OOF ranking (as c117.training_regime_of)."""
    A9, C108 = c132().c115().A9, c132().c115().C108
    lc = light()
    problems: list[str] = []
    recs = C108.fold_check(c132().SOURCE, lc["folds"], {})
    problems += [f"OOF {r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"] + r["unknown"]]
    fold_recipes = {}
    for r in recs:
        if r.get("train"):
            r2, j2 = str(r["train"]).split("/", 1)
            fold_recipes[r["train"]] = A9.read_json(A9.run_dir(r2) / j2 / "sft_recipe.json") or {}
    if len(fold_recipes) != 5:
        problems.append(f"recipes of the fold adapters: {len(fold_recipes)} instead of 5")
    run, job, tag = A9.parse_source(spec)
    meta = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json") or {}
    model = str((meta.get("recipe") or {}).get("model"))
    rec, where, p = adapter_recipe(run, model)
    problems += [f"Test {run}/{job}: {x}" for x in p]
    if where.get("train") != "/".join(FULL_TRAIN):
        problems.append(f"test model from {where.get('train')} instead of {'/'.join(FULL_TRAIN)} (full fit)")
    if rec is not None:
        if rec.get("fold") is not None:
            problems.append(f"test model holds out fold {rec.get('fold')} — not a full fit")
        if rec.get("teacher_derived") or rec.get("pool"):
            problems.append("test model with teacher targets (not gold-only)")
        if int(rec.get("rows_used") or -1) != int(n_train):
            problems.append(f"test model learned {rec.get('rows_used')} instead of {n_train} training rows")
        if rec.get("steps_done") != rec.get("steps_total"):
            problems.append(f"training of the test model not finished ({rec.get('steps_done')}/{rec.get('steps_total')})")
        for nm, fr in fold_recipes.items():
            diff = {k: (rec.get(k), fr.get(k)) for k in RECIPE_SAME if rec.get(k) != fr.get(k)}
            if diff:
                problems.append(f"test model fitted differently from fold adapter {nm}: {diff}")
    return problems, {"test_model": where, "test_recipe": {k: (rec or {}).get(k) for k in ("task", "fold", "rows_used", "steps_done", "steps_total",
                                                                                              "taxonomy", "notes_max_tokens", "pool", "teacher_derived")},
                      "fold_adapters": [{k: r.get(k) for k in ("job", "train", "train_fold", "gold_only", "adapter_matches_merge")} for r in recs]}


def run_log_check(spec: str, n_test: int) -> tuple[list[str], dict]:
    """Log of the test run: every line "sft_from: notes for N rows of table test, of which without a finding N" (log line of
    vlm/runner.py; recorded logs in the earlier wording are accepted, see LOG_LINE) with N = number of test rows."""
    A9 = c132().c115().A9
    run, _, _ = A9.parse_source(spec)
    logs = sorted(A9.run_dir(run).glob("worker*.log"))
    if not logs:
        return [f"log artifacts/runs/{run}/{run}/worker*.log missing"], {"lines": 0}
    found, bad = [], []
    for f in logs:
        for line in f.read_text(errors="replace").splitlines():
            if not any(p in line for p in LOG_PREFIXES):
                continue
            m = LOG_LINE.search(line)
            if not m:
                bad.append("line in an unknown format")
                continue
            g = [v for v in m.groups() if v is not None]
            rows, table, without = int(g[0]), g[1], int(g[2])
            found.append((rows, table, without))
            if table != "test" or rows != n_test or without != rows:
                bad.append(f"{rows} rows of table {table}, of which without a finding {without}")
    if len(found) < 2:  # preliminary run and ranking
        bad.append(f"{len(found)} lines \"sft_from: notes for …\" (preliminary run and ranking required)")
    return [f"log: {b}" for b in bad], {"logs": [rel(f) for f in logs], "lines": len(found)}


def test_guard(spec: str) -> tuple[int, list[str], dict]:
    from vlm import common as VC

    lc = light()
    root, seen, n_train = lc["root"], lc["seen"], lc["n"]
    n_test = len(VC.load_table(root, "test"))
    M = ms()
    missing = M.c95.rank_files_missing(spec)
    if missing:
        return 3, [f"test ranking {spec}: MISSING ({', '.join(missing)})"], {}
    A9, A8 = c132().c115().A9, c132().c115().A8
    run, job, tag = A9.parse_source(spec)
    # the run's status.json is not part of the private package; what counts is the completeness of the exact ranking below
    meta_f = ART / f"meta_{CH}.json"
    rm = (json.loads(meta_f.read_text()).get("rank_mean") if meta_f.is_file() else None) or {}
    if rm.get("rank_source") != c132().SOURCE or rm.get("rank_compare") != "sft" or not rm.get("notes"):
        return 5, [f"{rel(meta_f)} missing or not the artifact of this script — run oof --save first"], {}
    problems: list[str] = []
    t_meta = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json") or {}
    if t_meta.get("table") != "test":
        problems.append(f"test ranking computed table {t_meta.get('table')!r}")
    R, h, st = A8.load_rank(spec, n_test, seen, table="test", allow_partial=True)
    hr = h & M.c95.exact_rank_rows(spec, n_test, seen)
    exact = int(hr.sum())
    complete = bool(st["rows_probe"] == st["rows_done"] == st["rows_exact"] == exact == n_test)
    if not complete:
        problems.append(f"test ranking not completely exact: probe {st['rows_probe']}, computed {st['rows_done']}, exact {st['rows_exact']}, "
                        f"exact ranking {exact} of {n_test}")
    none_top = M.c95.top_none_share(R, hr)
    oof_none = float(rm.get("none_top_rank") or float("nan"))
    if not (NONE_BAND[0] <= none_top <= NONE_BAND[1]):
        problems.append(f"NONE at the top of the test ranking {none_top:.3f} outside {NONE_BAND[0]:.2f}–{NONE_BAND[1]:.2f} (OOF {oof_none:.3f})")
    diff = M.rank_only_source_differences(spec, c132().SOURCE)
    if diff:
        problems.append(f"settings of the test ranking differ from those of the OOF shards: {diff}")
    info: dict = {"spec": spec, "coverage": {"rows_probe": st["rows_probe"], "rows_done": st["rows_done"], "rows_exact": st["rows_exact"],
                                             "exact_rows": exact, "rows_table": n_test, "complete": complete, "mass_median": st["mass_median"],
                                             "mass_p05": st["mass_p05"]},
                  "none_top": {"test": none_top, "oof": oof_none, "band": list(NONE_BAND)}, "settings_equal": not diff}
    try:
        info["notes"] = M.rank_mean_notes_guard(CH, spec, root, n_test, n_train, rm)
    except SystemExit as exc:
        problems.append(str(exc))
    p, li = run_log_check(spec, n_test)
    problems += p
    info["log"] = li
    p, bi = training_regime_of(spec, n_train)
    problems += p
    info["training_regime"] = bi
    px_missing = M.c95.rank_files_missing(PX27_TEST)
    if not px_missing:  # same top code against the px27 test ranking (aggregate only)
        Rx, hx, _ = A8.load_rank(PX27_TEST, n_test, seen, table="test", allow_partial=True)
        both = hr & hx & M.c95.exact_rank_rows(PX27_TEST, n_test, seen)
        info["same_top_px27_test"] = {"rows": int(both.sum()),
                                      "share": float(np.mean(metric.top3_indices(R, seen)[both, 0] == metric.top3_indices(Rx, seen)[both, 0])) if both.any() else None,
                                      "oof_reference": "m3px27/px27 on the training rows 0.853 (c132)"}
    lines = [f"test ranking {spec}: {exact}/{n_test} exact, mass median {st['mass_median']:.3f}; NONE at the top {none_top:.3f} (OOF {oof_none:.3f}, "
             f"band {NONE_BAND[0]:.2f}–{NONE_BAND[1]:.2f}); settings against the five OOF shards {'equal' if not diff else 'DIFFERENT'}; log "
             f"{li.get('lines')} sft_from lines",
             f"training regime: test model {bi['test_model'].get('model')} <- {bi['test_model'].get('train')} (fold {bi['test_recipe'].get('fold')}, "
             f"{bi['test_recipe'].get('rows_used')} rows, {bi['test_recipe'].get('steps_done')}/{bi['test_recipe'].get('steps_total')} steps)"]
    if info.get("same_top_px27_test"):
        s = info["same_top_px27_test"]
        lines.append(f"same top code m3px27 test/px27 test {s['share']:.3f} on {s['rows']} rows (OOF 0.853)")
    lines += [f"PROBLEM: {x}" for x in problems]
    return (5 if problems else 0), lines, info


def cmd_test_guard(spec: str) -> int:
    code, lines, info = test_guard(spec)
    for line in lines:
        say(line)
    OUT.mkdir(parents=True, exist_ok=True)
    tagname = "c133_test_guard.json" if spec == TEST_RANK else f"c133_test_guard_{re.sub(r'[^A-Za-z0-9]+', '_', spec)}.json"
    (OUT / tagname).write_text(to_json({"code": code, "lines": lines, **info, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    say(f"-> {'PASSED' if code == 0 else ('MISSING' if code == 3 else 'VIOLATED')} (exit {code}); {rel(OUT / tagname)}")
    return code


def cmd_null(source: Path, out: Path, stem: str) -> int:
    C = c118()
    ref = json.loads(Path(source).read_text())
    res = C.null_stage2(stem, Path(source))
    res["mech27"] = ref["mech27"]
    Path(out).write_text(to_json(res))
    say(f"null model for {rel(Path(source))} -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("oof", "stack", "fit"):
        p = sub.add_parser(nm)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--save", action="store_true")
        g.add_argument("--verify", action="store_true")
    c = sub.add_parser("copy")
    c.add_argument("--keep", action="store_true")
    sub.add_parser("clean-copy")
    t = sub.add_parser("test-guard")
    t.add_argument("--rank", default=TEST_RANK)
    nl = sub.add_parser("null")
    nl.add_argument("--source", default=str(STAGE2))
    nl.add_argument("--out", required=True)
    nl.add_argument("--stack", default=STEM)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "oof":
            return cmd_oof(a.save)
        if a.cmd == "stack":
            return cmd_stack(a.save)
        if a.cmd == "fit":
            return cmd_fit(a.save)
        if a.cmd == "copy":
            return cmd_copy(a.keep)
        if a.cmd == "clean-copy":
            say(f"removed: {', '.join(clean_copy()) or 'nothing'}")
            return 0
        if a.cmd == "test-guard":
            return cmd_test_guard(a.rank)
        return cmd_null(Path(a.source), Path(a.out), a.stack)
    except Missing as exc:
        print(f"NOT YET EVALUABLE: {exc}", file=sys.stderr)
        return 3
    except TechnicalError as exc:
        print(f"ABORT (technical): {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
