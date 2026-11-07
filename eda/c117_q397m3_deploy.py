"""DEPLOYMENT PATH of the channel q397m3 — M3 names behind the notes of Qwen3.5-397B — as a REPLACEMENT of q397 in the
primary line (six-channel recipe; pilot: cross value 0.82922 -> 0.83267). Build script tools/builds/s029_q397m3.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c117_q397m3_deploy.py oof --save      # artifacts/oof_q397m3.npy + meta_q397m3.json from the five shards
    python -I eda/c117_q397m3_deploy.py oof --verify    # recompute (bit-identical, meta file character for character), writes nothing
    python -I eda/c117_q397m3_deploy.py test-guard runs/q397m3te_rp/q397m3_test_rank/test [--notes-source runs/q397te_rp/proto_te/proto]
    python -I eda/c117_q397m3_deploy.py test-guard runs/m3d3te_rp/m3d3_test_rank/test --placeholder   # code check only
    python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 (--save | --verify)   # blend file

OOF ARTIFACT (`oof`): PURE ranking — per fold k the exact distribution over all codes (probe proto_rank in sft mode) of the M3
adapter that did not learn fold k, behind the 397B note of the row; conversion a8.load_rank (vlm.engine.probs_from_logp,
absent = classes without a training example) per shard via eda/c115_q397m3_eval.assemble — THE SAME computation that c115
evaluated. No mean with named codes, no debiasing, no floor (0.05 only in the blend). Source (rank.source): the five shards
joined with +, like one draw in eda/a9_m3_draws_oof.py; tools/make_submission.py --rank-only compares the settings of the test
ranking with EVERY shard (rank_only_source_differences).
CHECKS before writing (abort with exit 4, nothing written; exit 3 while an input is missing):
  notes     derived/sft/q397m3_train_notes.jsonl = meta checksum = copy in the upload folder, 1,500 training rows, no finding
            given (c115.check_notes), the Tinker measurement (labels_sha256), the same file as in the c115 evaluation;
  shards    complete and exact, per shard EXACTLY the rows of its fold (c115.assemble); sft_from = note file, settings equal to
            d3 (c115.check_shard_settings); FOLD INTEGRITY and training regime per shard without findings and without unknowns
            (c108.fold_check: run configuration -> merge job -> sft_recipe.json holds out exactly this fold, gold-only, local
            adapter = merged one);
  c115      artifacts/eda/private/q397m3_eval/c115_q397m3_eval.json from a real run (not --dry), verdict CONTINUE for the
            replacement, replacement cross value 7,494/9,000; the channel is bit-identical to the one evaluated there (SHA-256 of
            the values);
  a8 path   recomputation with the computation of eda/a8_proto_oof.py --rank over a combined ranking (vlm/sft.combine_rank, in
            memory only) and with its fallback to the named pr27 codes: both bit-identical. A call
            `a8 q397m3 <pr27 answers> --rank <combined>` would therefore give the same numbers (the ranking covers all 1,500 rows
            exactly, the pr27 codes drop out completely), but would write a wrong provenance into the meta file (sources = pr27
            answers, cv_score_named from pr27, model type "protocol"), would need an additional combined run folder and would
            check neither fold integrity nor note source nor settings — hence this dedicated path.
META (kaggle_core.artifacts.save_oof, without timestamps, byte-identically reproducible): rank.source / rows / rows_named_only = 0 /
  shards / settings / fold_check; rank.notes = NOTE GUARD for tools/make_submission.py (rank_only_notes_guard: training and test
  notes, upload copies, source and model of the test notes); c115 (checksum, verdict, cross values).
TEST-SIDE GUARD (`test-guard`; aggregates only; exit 0 = deployable, 3 = missing or incomplete — all other checks still run and
are reported, so an intermediate state can already be checked —, 5 = violated):
  state          probe complete and exact on ≥ 98 % of the test rows (as in eda/c95_rank_mean.py --status); on the complete test
                 side the share of NONE at the top within the band fixed in advance NONE_BAND = 0.27–0.35 (OOF 0.310 ± 0.04; the
                 build script passes the same band to make_submission --rank-none-band-for, instead of the common M3 band
                 0.30–0.38);
  settings       make_submission.rank_only_source_differences against every shard empty (image, beams, finish, number format,
                 probe);
  notes          make_submission.rank_only_notes_guard: sft_from of the test ranking = derived/sft/q397m3_test_notes.jsonl, file =
                 meta checksum = copy in the upload folder, table test with all rows, each with a 397B note, source = the 397B test
                 run (checksums of <tag>_raw.jsonl and <tag>_meta.json; with --notes-source exactly this run), model
                 Qwen/Qwen3.5-397B-A17B-FP8; log of the run: every line "sft_from: notes for N rows of table test, of which
                 without a finding N" (log line of vlm/runner.py; LOG_LINE also accepts the wording of the recorded runs) with
                 N = number of test rows (the remote machine read notes without a given finding);
  regime         test model = merge of the FULL FIT (sft_recipe.json: fold null, all training rows, gold-only, training finished)
                 with task, system text, taxonomy, note limit, image size and base model of all five fold adapters
                 (c108.fold_check without findings); local adapter = merged one (summary.json of the test run's merge job);
                 eda/c99_m3_deploy.variant_check --require-gold-only.
  --placeholder  (code check with a foreign test ranking, e.g. runs/m3d3te_rp/m3d3_test_rank/test): notes and log are skipped,
                 settings and regime are not; no log written to disk.
Test files are only counted and checked via checksums, never inspected. Logs (aggregates) under
artifacts/eda/private/q397m3_deploy/.
"""
from __future__ import annotations

import argparse
import importlib.util
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C115 = _load("c117_c115", COMP / "eda" / "c115_q397m3_eval.py")  # assemble, check_notes, check_shard_settings, npy_sha256
C110 = C115.C110  # setup, TEST_RUNS, TEST_MODEL, TINKER_SHA256
C108 = C115.C108  # fold_check, run_config
C99 = C108.C99  # variant_check
A9, A8 = C115.A9, C115.A8
MS = _load("c117_ms", COMP / "tools" / "make_submission.py")  # rank_only_source_differences, rank_only_notes_guard, rank_only_guard

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "q397m3_deploy"
CH = C115.CH  # q397m3
SHARDS = tuple(C115.SHARDS)
SOURCE = "+".join(SHARDS)
C115_JSON = C115.OUT / f"{C115.NAME}.json"
NOTES_TRAIN, UPLOAD_TRAIN = C115.NOTES_REL, C115.UPLOAD_REL
NOTES_TEST, UPLOAD_TEST = "derived/sft/q397m3_test_notes.jsonl", "derived/sft_q397m3te/q397m3_test_notes.jsonl"
TEST_SOURCE_RUNS = tuple(C110.TEST_RUNS)  # 397B test side: q397te_rp (4×H200) or q397te8_rp (8×H100)
TEST_MODEL = C110.TEST_MODEL  # Qwen/Qwen3.5-397B-A17B-FP8
PR27_NAMED = "runs/pr27_rp/proto_a/proto+runs/pr27_rp/proto_b/proto"  # answers that a call to a8 with the pr27 answers would use
SIX_CHANNEL_UNITS, REPLACEMENT_UNITS = 7463, 7494  # six-channel recipe 0.8292222, with q397m3 instead of q397 0.8326667 (units of 1/6)
RECIPE_SAME = ("task", "system_sha256", "taxonomy", "notes_max_tokens", "image", "model")  # full fit equal to the fold adapters
# Malfunction band for the share of NONE at the top of the test ranking: OOF value of the channel 0.310 ± 0.04 (half-width as
# for the common M3 band 0.30–0.38 around 0.34, make_submission --rank-none-band). Fixed in advance, before the test side was
# complete; never used to choose between variants. tools/builds/s029_q397m3.sh passes the same band to make_submission
# (--rank-none-band-for q397m3=0.27,0.35).
NONE_BAND = (0.27, 0.35)
# Log line of vlm/runner.py (sft_from). The worker logs of the recorded test run were written with an earlier, German wording of
# the same line ("Ausgaben zu N Zeilen der Tabelle T, davon ohne Befund M"); both wordings are accepted.
LOG_PREFIXES = ("sft_from: notes for", "sft_from: Ausgaben zu")
LOG_LINE = re.compile(r"sft_from: (?:notes for (\d+) rows of table (\w+), of which without a finding (\d+)"
                      r"|Ausgaben zu (\d+) Zeilen der Tabelle (\w+), davon ohne Befund (\d+))")
# definition of the channel, recorded in meta_q397m3.json
DEFINITION = ("pure ranking: per fold k the exact distribution over all codes (probe proto_rank, mode sft) of the M3 adapter without fold k "
              "behind the 397B note of the row (derived/sft/q397m3_train_notes.jsonl; M3 writes the finding itself); a8.load_rank / "
              "vlm.engine.probs_from_logp with absent = classes without a training example; no mean with named codes, no debiasing; "
              "floor only in the blend (0.05; eda/c110_q397_deploy.py stack --replace-q397 q397m3)")


class Abort(SystemExit):
    pass


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise Abort(code)


def sha256_file(path: Path) -> str:
    return C115.sha256_file(Path(path))


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=C110.C107.json_default) + "\n"


# ------------------------------------------------------------------------------------------------ c115 and recomputation

def load_c115(notes_sha: str, path: Path = C115_JSON) -> dict:
    """c115 evaluation (real run): verdict CONTINUE for the replacement, anchor and replacement cross value as recorded, the same
    note file. -> {channel_sha256, replacement_units, replacement_logloss, replacement_delta_exact, …}"""
    if not path.is_file():
        abort(f"{rel(path)} missing — run python -I eda/c115_q397m3_eval.py first (verdict of the pre-registered rule)", 3)
    js = json.loads(path.read_text())
    if js.get("dry") is not False:
        abort(f"{rel(path)} comes from the --dry check, not from the real run")
    dec = js.get("decision") or {}
    if dec.get("verdict") != "CONTINUE" or "replacement" not in (dec.get("passed_forms") or []):
        abort(f"c115 gave {dec.get('verdict')!r} (passed: {dec.get('passed_forms')}) — deployment as a replacement of q397 is not covered")
    tech = js.get("technical") or {}
    if (tech.get("notes") or {}).get("sha256") != notes_sha:
        abort("c115 evaluated a different note file")
    if [s.get("shard") for s in tech.get("shards") or []] != list(SHARDS):
        abort("c115 evaluated different shards")
    er, anc = js["recipes"]["replacement"], js["anchor"]
    units = int(round(float(er["map_new"]) * 9000))
    if int(anc["units"]) != SIX_CHANNEL_UNITS or units != REPLACEMENT_UNITS:
        abort(f"c115: anchor {anc['units']} / replacement {units} instead of {SIX_CHANNEL_UNITS} / {REPLACEMENT_UNITS}")
    return {"json": rel(path), "channel_sha256": tech["channel_sha256"], "verdict": dec["verdict"], "passed_forms": list(dec["passed_forms"]),
            "anchor_units": int(anc["units"]), "anchor_logloss": float(anc["logloss"]), "replacement_units": units,
            "replacement_map3": float(er["map_new"]), "replacement_logloss": float(er["ll_new"]), "replacement_delta_exact": str(er["delta_exact"]),
            "replacement_delta": float(er["delta"]), "replacement_se_q": float(er["se_q"]), "replacement_z_q": float(er["z_q"])}


def a8_counter(shards, n: int, seen: np.ndarray, classes: list[str], named_source: str | None = PR27_NAMED) -> tuple[np.ndarray, np.ndarray | None]:
    """Recomputation with the computation of eda/a8_proto_oof.py --rank (in memory only): shards concatenated as in
    vlm/sft.combine_rank and sorted by table position, then a8.load_rank (complete, exact, all rows) with ONE call of
    probs_from_logp; with `named_source` additionally a8's fallback rule np.where(computed, ranking, named codes).
    -> (ranking, a8 with pr27 or None)."""
    from vlm.engine import probs_from_logp

    parts = []
    for s in shards:
        d, tag = C115.shard_dir(s)
        parts.append((np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy"), np.load(d / f"{tag}_proto_rank.done.npy")))
    idx = np.concatenate([p[0] for p in parts])
    lp = np.concatenate([p[1] for p in parts])
    done = np.concatenate([p[2] for p in parts])
    order = np.argsort(idx, kind="stable")
    idx, lp, done = idx[order], lp[order], done[order]
    if len(idx) != n or not np.array_equal(idx, np.arange(n)):
        raise C115.TechnicalError("combined ranking does not cover exactly all training rows")
    done = done & np.isfinite(lp).any(axis=1)
    if not (done & np.isfinite(lp[:, seen]).all(axis=1)).all():
        raise C115.TechnicalError("combined ranking not completely exact")
    R = np.zeros((n, len(seen)))
    R[idx[done]] = probs_from_logp(lp[done], absent=~seen)
    if named_source is None:
        return R, None
    try:
        named, _ = A8.mean_of_draws([named_source], n, classes, seen)
    except (SystemExit, OSError):
        return R, None
    hr = np.zeros(n, bool)
    hr[idx[done]] = True
    return R, np.where(hr[:, None], R, named)


# ------------------------------------------------------------------------------------------------ oof

def build_oof(ctx: dict) -> tuple[np.ndarray, dict, dict]:
    """Channel from the five shards with all checks. -> (P, meta fields for save_oof, log)."""
    root, n, seen, folds, y, classes = ctx["root"], ctx["n"], ctx["seen"], ctx["folds"], ctx["y"], ctx["classes"]
    try:
        nt = C115.check_notes(root, n)
        meta_n = json.loads((root / NOTES_TRAIN).with_suffix(".meta.json").read_text())
        if meta_n.get("labels_sha256") != C110.TINKER_SHA256 or meta_n.get("source_kind") != "tinker":
            raise C115.TechnicalError(f"{NOTES_TRAIN}: not from the Tinker measurement of the c107 decision (labels_sha256 {meta_n.get('labels_sha256')})")
        c115 = load_c115(nt["sha256"])
        P, have, info = C115.assemble(SHARDS, n, seen, folds, require_all=True)
        settings = C115.check_shard_settings(SHARDS, C115.REF_SHARDS)
        if not settings["same_as_d3"]:
            raise C115.TechnicalError("settings of the d3 ranking not readable — comparison with d3 not possible")
        recs = C108.fold_check(SOURCE, folds, {})
        bad = [f"{r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"] + r["unknown"]]
        if bad:
            raise C115.TechnicalError("fold integrity violated or cannot be determined: " + "; ".join(bad))
        sha = C115.npy_sha256(P)
        if sha != c115["channel_sha256"]:
            raise C115.TechnicalError(f"channel ({sha[:16]}…) is not the one evaluated by c115 ({c115['channel_sha256'][:16]}…)")
        R, Pa8 = a8_counter(SHARDS, n, seen, classes)
        if not np.array_equal(R, P):
            raise C115.TechnicalError("recomputation over the combined ranking (a8 path) not bit-identical")
        if Pa8 is not None and not np.array_equal(Pa8, P):
            raise C115.TechnicalError("a8 path with fallback to the pr27 codes not bit-identical (not every row computed exactly?)")
    except C115.Missing as exc:
        abort(str(exc), 3)
    except C115.TechnicalError as exc:
        abort(str(exc), 4)
    if not C110.C107.sane(P, seen):
        abort("channel not finite/normalised or with mass on classes without a training example")
    from kaggle_core.folds import folds_hash

    r = metric.map3_rows_from_proba(y, P)
    top = P.argmax(1)
    fold_rec = [{k: x.get(k) for k in ("run", "job", "model", "merge_job", "train", "train_fold", "gold_only", "adapter_matches_merge", "rows", "row_folds")}
                for x in recs]
    meta = {"folds_hash": folds_hash(folds), "model_type": "vlm-sft-rank", "sources": [SOURCE], "cv_score": float(r.mean()),
            "cv_sigma": float(metric.bootstrap_sigma(r)),
            "feature_set": "image+task, without task knowledge (trained model M3, per fold the adapter without that fold; ranking behind "
                           "the notes of Qwen3.5-397B from the Tinker measurement)",
            "rows_without_answer": 0, "neutral_rows": None, "rows_neutral": 0,
            "rank": {"source": SOURCE, "rows": int(n), "rows_named_only": 0, "partial": False, "shards": info,
                     "settings": settings["settings"], "settings_same_as_d3": True, "fold_check": fold_rec,
                     "notes": {"train": {"sft_from": NOTES_TRAIN, "sha256": nt["sha256"], "upload": UPLOAD_TRAIN, "source_kind": "tinker",
                                         "labels_sha256": C110.TINKER_SHA256, "q397_states": nt.get("q397_states")},
                               "test": {"sft_from": NOTES_TEST, "upload": UPLOAD_TEST, "source_runs": list(TEST_SOURCE_RUNS), "model": TEST_MODEL},
                               "note": "note guard: tools/make_submission.py (rank_only_notes_guard) requires these note sources; the test notes "
                                       "are created with python -I tools/make_q397m3_notes.py --test-from <397B test run>"},
                     "definition": DEFINITION, "function": "eda/c117_q397m3_deploy.py oof (c115.assemble / a8.load_rank)"},
            "c115": {k: c115[k] for k in ("json", "channel_sha256", "verdict", "passed_forms", "anchor_units", "replacement_units", "replacement_map3",
                                          "replacement_logloss", "replacement_delta_exact")},
            "checks": {"c115_bit_identical": True, "a8_combined_bit_identical": True, "a8_with_pr27_named_bit_identical": Pa8 is not None,
                       "fold_check_clean": True},
            "cv_note": "single channel over all 1,500 rows (per row the ranking of the adapter that did not learn the row's fold); "
                       "deployed only as a replacement of q397 in the primary line (cross-fit in the recipe: blend file stack_s029_best_q397m3.json)"}
    proto = {"cv_score": meta["cv_score"], "per_fold": [float(r[folds == k].mean()) for k in range(5)], "none_top": float(np.mean(top == 0)),
             "none_true": float(np.mean(y == 0)), "channel_sha256": sha, "notes_sha256": nt["sha256"], "fold_check": fold_rec}
    return P, meta, proto


def meta_text(P: np.ndarray, meta: dict, K: int) -> str:
    """Meta file exactly as kaggle_core.artifacts.save_oof writes it (for --verify)."""
    from kaggle_core import artifacts as KA

    full = {"name": CH, "oof_shape": list(P.shape), "test_shape": [0, K], "oof_hash": KA._array_hash(P), **meta}
    return json.dumps(full, indent=2, ensure_ascii=False, default=str)


def cmd_oof(a) -> int:
    ctx = C110.setup()
    P, meta, proto = build_oof(ctx)
    K = ctx["K"]
    f, mf = ART / f"oof_{CH}.npy", ART / f"meta_{CH}.json"
    text = meta_text(P, meta, K)
    print(f"{CH}: pure ranking from {len(SHARDS)} shards ({ctx['n']} rows, all exact, fold integrity without findings, settings = d3); "
          f"bit-identical to the channel of the c115 evaluation and to the a8 path (combined; with pr27 fallback {meta['checks']['a8_with_pr27_named_bit_identical']})")
    print(f"   MAP@3 alone {proto['cv_score']:.4f} ± {meta['cv_sigma']:.4f}; per fold {[round(v, 4) for v in proto['per_fold']]}; NONE at the top "
          f"{proto['none_top']:.3f} (true {proto['none_true']:.3f}); SHA-256 of the values {proto['channel_sha256'][:16]}…; notes {proto['notes_sha256'][:16]}…")
    if a.verify:
        if not f.is_file() or not mf.is_file():
            abort(f"{rel(f)} or {rel(mf)} missing — build it with --save first", 3)
        same = np.load(f).dtype == np.float64 and np.array_equal(np.load(f), P)
        same_meta = mf.read_text(encoding="utf-8") == text
        if not same or not same_meta:
            abort(f"{rel(f)} / {rel(mf)} is not the recomputation (values {'equal' if same else 'DIFFERENT'}, meta file {'equal' if same_meta else 'DIFFERENT'})")
        print(f"   VERIFIED: {rel(f)} (SHA-256 {sha256_file(f)[:16]}…) and {rel(mf)} = recomputation")
    if a.save:
        if f.is_file() and not (np.load(f).dtype == np.float64 and np.array_equal(np.load(f), P)):
            abort(f"{rel(f)} exists with different values — not overwritten (delete it deliberately, then rebuild)")
        if mf.is_file() and mf.read_text(encoding="utf-8") != text:
            abort(f"{rel(mf)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        from kaggle_core import artifacts as KA

        KA.save_oof(CH, P, np.zeros((0, K)), meta, ART)
        if mf.read_text(encoding="utf-8") != text:
            abort("meta file not as expected after writing (save_oof changed?)")
        print(f"   saved: {rel(f)} (SHA-256 {sha256_file(f)}), {rel(mf)} (SHA-256 {sha256_file(mf)})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c117_oof.json").write_text(to_json({**proto, "oof_sha256": sha256_file(f) if f.is_file() else None,
                                                 "meta_sha256": sha256_file(mf) if mf.is_file() else None,
                                                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return 0


# ------------------------------------------------------------------------------------------------ test-guard

def adapter_recipe(run: str, model: str) -> tuple[dict | None, dict, list[str]]:
    """Model `model` of a run -> merge job (run configuration) -> training job (adapter_from) -> sft_recipe.json.
    -> (recipe or None, provenance, findings)."""
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
        problems.append(f"checksum of the merged adapter cannot be determined (summary.json of merge job {mj[0]['name']} or local adapter missing)")
    elif sha256_file(local) != want:
        problems.append(f"local adapter {r2}/{j2} is not the one merged in the test run ({mj[0]['name']})")
    else:
        where["adapter_sha256"] = want
    return rec, where, problems


def training_regime_of(spec: str, folds: np.ndarray, n_train: int) -> tuple[list[str], dict]:
    """Test model = merge of the full fit belonging to the five fold adapters of the OOF ranking."""
    problems: list[str] = []
    recs = C108.fold_check(SOURCE, folds, {})
    problems += [f"OOF {r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"] + r["unknown"]]
    fold_recipes = {}
    for r in recs:
        if not r.get("train"):  # without a unique merge job — already reported as a finding above
            continue
        r2, j2 = str(r["train"]).split("/", 1)
        fold_recipes[r["train"]] = A9.read_json(A9.run_dir(r2) / j2 / "sft_recipe.json") or {}
    if len(fold_recipes) != len(SHARDS):
        problems.append(f"recipes of the fold adapters: {len(fold_recipes)} instead of {len(SHARDS)} determinable")
    run, job, tag = A9.parse_source(spec)
    meta = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json") or {}
    model = str((meta.get("recipe") or {}).get("model"))
    rec, where, p = adapter_recipe(run, model)
    problems += [f"Test {run}/{job}: {x}" for x in p]
    if rec is not None:
        if rec.get("fold") is not None:
            problems.append(f"test model holds out fold {rec.get('fold')} — not a full fit on all training rows")
        if rec.get("teacher_derived") or rec.get("pool"):
            problems.append("test model with teacher targets (not gold-only)")
        if int(rec.get("rows_used") or -1) != int(n_train):
            problems.append(f"test model learned {rec.get('rows_used')} instead of all {n_train} training rows")
        if rec.get("steps_done") != rec.get("steps_total"):
            problems.append(f"training of the test model not finished ({rec.get('steps_done')}/{rec.get('steps_total')} steps)")
        for name, fr in fold_recipes.items():
            diff = {k: (rec.get(k), fr.get(k)) for k in RECIPE_SAME if rec.get(k) != fr.get(k)}
            if diff:
                problems.append(f"test model fitted differently from fold adapter {name}: {diff}")
    code, lines, _ = C99.variant_check(spec, CH, require_gold_only=True)
    if code != 0:
        problems.append(f"c99.variant_check --require-gold-only: Exit {code} ({lines[-1]})")
    info = {"test_model": where, "test_recipe": {k: (rec or {}).get(k) for k in ("task", "fold", "rows_used", "steps_done", "steps_total", "taxonomy",
                                                                                   "notes_max_tokens", "pool", "teacher_derived", "targets_sha256")},
            "fold_adapters": [{k: r.get(k) for k in ("job", "train", "train_fold", "gold_only", "adapter_matches_merge")} for r in recs],
            "variant_code": code}
    return problems, info


def run_log_check(spec: str, n_test: int) -> tuple[list[str], dict]:
    """Log of the test run: every line "sft_from: notes for N rows of table T, of which without a finding M" (log line of
    vlm/runner.py; recorded logs in the earlier wording are accepted, see LOG_LINE) with T = test, N = M = number of test rows (the
    remote machine read a note file without a given finding — the q397m3 test notes); only these lines, only numbers."""
    run, _, _ = A9.parse_source(spec)
    logs = sorted(A9.run_dir(run).glob("worker*.log"))
    if not logs:
        return [f"log artifacts/runs/{run}/{run}/worker*.log missing — note source of the remote run not documented"], {"lines": 0}
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
    if not found:
        bad.append("no line \"sft_from: notes for …\" — note source of the remote run not documented")
    return ([f"log: {b}" for b in bad], {"logs": [rel(f) for f in logs], "lines": len(found)})


def test_guard(spec: str, placeholder: bool = False, notes_source: str | None = None) -> tuple[int, list[str], dict]:
    """Test-side guard (see the module docstring). -> (exit code 0/3/5, lines, statistics)."""
    from vlm import common as C

    root = C.find_data_root()
    n_train = len(C.load_table(root, "train"))
    n_test = len(C.load_table(root, "test"))
    folds = C.load_folds(n_train)
    lines: list[str] = []
    missing = MS.c95.rank_files_missing(spec)
    if missing:
        return 3, [f"test ranking {spec}: MISSING ({', '.join(missing)}) — wait for the run or fetch it"], {}
    meta_q = A9.read_json(ART / f"meta_{CH}.json")
    if meta_q is None or (meta_q.get("rank") or {}).get("source") != SOURCE or not (meta_q.get("rank") or {}).get("notes"):
        return 5, [f"artifacts/meta_{CH}.json missing or not the artifact from this script (rank.source, rank.notes) — run oof --save first"], {}
    problems: list[str] = []
    run, job, tag = A9.parse_source(spec)
    t_meta = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json") or {}
    if t_meta.get("table") != "test":
        problems.append(f"test ranking computed table {t_meta.get('table')!r}")
    import labels

    train = C.load_table(root, "train")
    seen = labels.seen_mask(train["root_code"], labels.load_classes(root / "competition_data" / "taxonomy_v1.csv"))
    R, h, st = A8.load_rank(spec, n_test, seen, table="test", allow_partial=True)  # state as in eda/c95_rank_mean.py --status (aggregates)
    hr = h & MS.c95.exact_rank_rows(spec, n_test, seen)
    exact = int(hr.sum())
    complete = bool(st["rows_done"] == st["rows_probe"] == st["rows_exact"] and exact >= 0.98 * n_test)
    none_top = MS.c95.top_none_share(R, hr)
    oof_f = ART / f"oof_{CH}.npy"
    none_oof = float(np.mean(np.load(oof_f).argmax(1) == 0)) if oof_f.is_file() else None
    if complete and not (NONE_BAND[0] <= none_top <= NONE_BAND[1]):  # intermediate states are not a random sample (longest prompts first)
        problems.append(f"share of NONE at the top of the test ranking {none_top:.3f} outside the band fixed in advance {NONE_BAND[0]:.2f}–{NONE_BAND[1]:.2f} "
                        f"(OOF {none_oof:.3f})" if none_oof is not None else f"share of NONE at the top {none_top:.3f} outside the band")
    diff = MS.rank_only_source_differences(spec, SOURCE)
    if diff:
        problems.append(f"settings of the test ranking differ from those of the OOF shards: {diff}")
    info: dict = {"spec": spec, "placeholder": bool(placeholder), "settings_equal": not diff,
                  "coverage": {"rows_probe": st["rows_probe"], "rows_done": st["rows_done"], "rows_exact": st["rows_exact"], "exact_rows": exact,
                               "rows_table": n_test, "complete": complete},
                  "none_top": {"test": none_top, "oof": none_oof, "band": list(NONE_BAND), "checked": complete}}
    if placeholder:
        lines.append(f"PLACEHOLDER: {spec} is NOT the q397m3 test side — notes and log not checked, only settings and training regime")
    else:
        try:
            ninfo = MS.rank_only_notes_guard(CH, spec, root, n_test, n_train)
        except SystemExit as exc:
            problems.append(str(exc))
            ninfo = None
        if ninfo is not None:
            info["notes"] = ninfo
            if notes_source is not None and ninfo.get("notes_test_source") != notes_source:
                problems.append(f"test notes come from {ninfo.get('notes_test_source')}, the deployed 397B test side is {notes_source}")
        p, li = run_log_check(spec, n_test)
        problems += p
        info["log"] = li
    p, bi = training_regime_of(spec, folds, n_train)
    problems += p
    info["training_regime"] = bi
    tr = bi["test_recipe"]
    lines.append(f"training regime: test model {bi['test_model'].get('model')} <- {bi['test_model'].get('train')} (fold {tr.get('fold')}, {tr.get('rows_used')} rows, "
                 f"{tr.get('steps_done')}/{tr.get('steps_total')} steps, teacher {bool(tr.get('pool') or tr.get('teacher_derived'))}); fold adapters "
                 + ", ".join(f"{x['train']} (Fold {x['train_fold']})" for x in bi["fold_adapters"]) + f"; c99 Exit {bi['variant_code']}")
    lines.append(f"settings against the five OOF shards: {'equal' if not diff else 'DIFFERENT'}")
    if "notes" in info:
        ni = info["notes"]
        lines.append(f"notes: training {ni.get('notes_train_sha256', '')[:16]}…, test {ni.get('notes_test_sha256', '')[:16]}… from {ni.get('notes_test_source')} "
                     f"(model {ni.get('notes_test_model')}); log: {info['log'].get('lines')} \"sft_from\" lines")
    cov = info["coverage"]
    lines.append(f"state of the test ranking: probe {cov['rows_probe']} rows, computed {cov['rows_done']}, exact {cov['rows_exact']}; exact ranking on "
                 f"{cov['exact_rows']} of {n_test} test rows -> {'COMPLETE' if complete else 'INCOMPLETE (wait for the run or fetch it completely)'}")
    nt_ = info["none_top"]
    lines.append(f"NONE at the top: test {nt_['test']:.3f}" + (f", OOF {nt_['oof']:.3f}" if nt_["oof"] is not None else "")
                 + f"; Band {NONE_BAND[0]:.2f}–{NONE_BAND[1]:.2f} " + ("checked" if complete else "only checked on the complete test side (an intermediate state is not a random sample)"))
    code = 5 if problems else (0 if complete else 3)
    for p in problems:
        lines.append(f"VIOLATED: {p}")
    verdict = {0: "DEPLOYABLE", 3: "GUARD HELD, BUT INCOMPLETE", 5: "NOT DEPLOYABLE"}[code]
    lines.append(f"-> {verdict}{' (PLACEHOLDER — code check only, never submit)' if placeholder else ''} (exit {code})")
    info["problems"] = problems
    return code, lines, info


def cmd_test_guard(a) -> int:
    code, lines, info = test_guard(a.run, a.placeholder, a.notes_source)
    for line in lines:
        print(line)
    if not a.placeholder and info:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "c117_test_guard.json").write_text(to_json({**info, "code": code, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof", help="build (--save) or recompute (--verify) the OOF artifact q397m3 from the five shards")
    g = o.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    t = sub.add_parser("test-guard", help="test-side guard: settings, notes (397B), training regime (full fit, gold-only)")
    t.add_argument("run", help="runs/<run>/<job>/<tag> of the test ranking")
    t.add_argument("--placeholder", action="store_true", help="foreign test ranking (code check only): notes and log are not checked")
    t.add_argument("--notes-source", help="runs/<run>/proto_te/proto: the deployed 397B test side — the test notes must come from exactly this run")
    a = ap.parse_args(argv)
    try:
        return {"oof": cmd_oof, "test-guard": cmd_test_guard}[a.cmd](a)
    except Abort as exc:
        return int(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
