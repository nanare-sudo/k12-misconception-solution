"""SAFETY LINE WITH NOTE FEATURES (safety_nd): second stage of the set-aligned safety line on the feature set fallback_i1_nd.

The safety line with set alignment (safety_final, variant b) with the feature set "fallback_i1_nd" in the second stage: fallback_i1 (55,
unchanged) + note mentions m3_men, m3_last (M3 notes d1–d3) + example similarity demo_sim (src/note_sim.py). eda/c123_safety_set.py stays
unchanged; this script only uses it as a module. Build script: tools/builds/safety_nd.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c129_safety_nd.py export-notes (--save | --verify)   # training notes d1–d3 frozen -> data/derived/sft/m3_d<k>_train_frozen_notes.jsonl
    python -I eda/c129_safety_nd.py oof                                # nested: ND against b, copy null point (ND columns 0 -> b), placebos
    python -I eda/c129_safety_nd.py splits [--n 20]                    # new meta-splits: ND − b (expected-value rule)
    python -I eda/c129_safety_nd.py rule                               # safety-nd rule (6) with the risk rule (τ_d = 0; sensitivity τ_d 0.001)
    python -I eda/c129_safety_nd.py fit (--save | --verify)            # deployment file artifacts/stage2_safety_nd.json
    python -I eda/c129_safety_nd.py null --out <file>                  # null model (weight 0) for the deployment file
    python -I eda/c129_safety_nd.py b-zero --out <file>                # copy null point: weights of safety_final (stage2_safety_final.json) + 0 for the ND columns

FEATURES (fixed before the evaluation): notes = field text of the frozen M3 notes per training row (c118.train_note_texts of the three draws
of m3x3_full, the same source as the note features of the primary line), frozen as files (export-notes) so that the test side reads the same
texts as the reference. m3_men/m3_last as in src/stage2.notes_features. demo_sim: TF-IDF per draw fitted on the training notes only, cosine
averaged over d1–d3, per candidate the mean of the three largest similarities to reference rows with gold = candidate, never the same
question. Reference NESTED like the knowledge: evaluated rows of fold k: rows of the folds ≠ k; training rows from fold j (outer fold k):
folds ∉ {j, k}; test rows: all 1,500 training rows (without the same question).
OOF (`oof`): as c123.run_nested variant b (gate over the four inner folds with guard, NONE guard), forms F (per fold) and M (one set);
copy null point: ND columns 0 -> result of c123 variant b bit-identical (row values, orders, guard); placebo: the three ND columns from another
row of the same block (notes and similarity of that row, at the candidates of the own row; default_rng(seed + R)), 15 seeds as in c123.
SAFETY-ND RULE (6) (fixed before the evaluation; `rule`): ½(F+M) ND − b (form "forced") > 0 and above the placebo mean (15 seeds) on the
original meta-split; mean over 20 new meta-splits > 0; risk rule (eda/risk_rule.py; τ_d = 0, the rule's default; class R, k 6, ρ 0.5,
spread 0.0015, seed 170007): P(θ < −0.002) ≤ 0.05 — sensitivity τ_d 0.001 reported only. The other conditions (guards, copy null points of
the file, regressions) are checked by the build script tools/builds/safety_nd.sh, the byte-identical rebuild without network separately.
DEPLOYMENT (`fit`): training rows = all 1,500 in form M, knowledge and reference of the similarity from the other folds, gate over the five
folds; features of the deployment form = evaluated rows of the OOF reproduction (null point (e)). Guards of the test side
(tools/make_submission.py, apply_stage2_safe): widened note guard against guard_reference, similarity guard against sim_reference (band
±10 % OR |z| ≤ 3), guard (c) — each ONLY as an abort without a file (safety-nd rule (4)), never as a switch.

Training rows only for statistics; texts are never printed. Exit 3 = input missing; 4 = checksum/recomputation/null point violated;
5 = pre-registered rule not met (no deployment).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import note_sim as NS  # noqa: E402
import stage2 as S2  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C123 = _load("c129_c123", COMP / "eda" / "c123_safety_set.py")
C118 = _load("c129_c118", COMP / "eda" / "c118_stage2.py")

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "safety_nd"
OOF_JSON = OUT / "c129_oof.json"
OOF_ROWS = OUT / "c129_oof_rows.npz"
SPLITS_JSON = OUT / "c129_splits.json"
RULE_JSON = OUT / "c129_rule.json"
STAGE2_FILE = ART / "stage2_safety_nd.json"
STAGE2_SAFETY_FINAL = ART / "stage2_safety_final.json"  # deployment file of safety_final (variant b, with the override of (ii))
STAGE2_SAFETY_FINAL_SHA = "9b6dfbed3ac97a6083e45bab68a52b55118c3b0e3a35dd58371bad47c6e8c81a"
FORMS = ("fold", "set")
FORM_SHORT = {"fold": "F", "set": "M"}
TRAIN_NOTES = tuple(f"derived/sft/m3_d{d}_train_frozen_notes.jsonl" for d in (1, 2, 3))
TEST_NOTES = tuple(C118.TEST_NOTES)
TEST_DRAWS = tuple(C123.c120().TEST_DRAWS)
PLACEBO_SEEDS = tuple(C123.PLACEBO_SEEDS)
SPLIT_SEED0 = 1000
SPLIT_PLACEBO_SEED0 = 7900
MIN_SPLITS = 20
RISK_SEED, RISK_K, RISK_RHO, RISK_SPREAD, RISK_PHARM = 170_007, 6, 0.5, 0.0015, 0.05
SHARE_Z_MAX, NOTES_REL = 3.0, 0.10


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ training notes (frozen)

def notes_payload(texts: list[str]) -> bytes:
    return "".join(json.dumps({"table": "train", "row": i, "text": t or ""}, ensure_ascii=False) + "\n" for i, t in enumerate(texts)).encode("utf-8")


def source_texts(ctx: dict) -> list[list[str]]:
    """Notes of the three draws of m3x3_full from the runs (definition; c118.train_note_texts with the original folds)."""
    try:
        return [C118.train_note_texts(sp, ctx["folds"]) for sp in ctx["draw_specs"]]
    except C118.Missing as e:
        abort(str(e), 3)


def read_train_notes(root: Path, rel: str, n: int, want_sha: str | None = None) -> tuple[list[str], str]:
    """Frozen training notes of one draw: file = checksum of the meta file (and want_sha), table train, every row exactly once."""
    f = Path(root) / rel
    mf = f.with_suffix(".meta.json")
    if not f.is_file() or not mf.is_file():
        abort(f"{rel} or {mf.name} missing — run c129 export-notes --save first", 3)
    raw = f.read_bytes()
    sha = sha256_bytes(raw)
    meta = json.loads(mf.read_text())
    if meta.get("sha256") != sha or (want_sha is not None and sha != want_sha):
        abort(f"{rel}: checksum {sha[:16]}… does not match the meta file or the deployment file")
    texts: dict[int, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            i = int(r["row"])
            if r.get("table") != "train" or i in texts or not 0 <= i < n:
                abort(f"{rel}: rows not unique or not table train")
            texts[i] = str(r.get("text") or "")
    if len(texts) != n or meta.get("table") != "train" or int(meta.get("rows", -1)) != n:
        abort(f"{rel}: {len(texts)} of {n} rows")
    return [texts[i] for i in range(n)], sha


def cmd_export_notes(a) -> int:
    ctx = C123.load_context()
    root, n = ctx["root"], ctx["n"]
    texts = source_texts(ctx)
    for rel, spec, tx in zip(TRAIN_NOTES, ctx["draw_specs"], texts):
        payload = notes_payload(tx)
        sha = sha256_bytes(payload)
        meta = {"table": "train", "rows": n, "sha256": sha, "source_draw": spec, "field": "text",
                "kind": "M3 notes per training row (eda/c118_stage2.train_note_texts), reference of the example similarity of the test side",
                "empty_rows": int(sum(1 for t in tx if not t))}
        f = root / rel
        mtext = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"
        if a.verify:
            mf = f.with_suffix(".meta.json")
            frozen = json.loads(mf.read_text()) if mf.is_file() else {}
            # every field of the meta file except the free-text description "kind" (fields, order and values; the payload byte for byte)
            if not f.is_file() or f.read_bytes() != payload or list(frozen) != list(meta) or any(frozen[k] != v for k, v in meta.items() if k != "kind"):
                abort(f"{rel} is not the recomputation from {spec}")
            print(f"  VERIFIED: {rel} (SHA-256 {sha[:16]}…, {n} rows, empty {meta['empty_rows']})")
        else:
            if f.is_file() and f.read_bytes() != payload:
                abort(f"{rel} exists with different content — not overwritten")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(payload)
            f.with_suffix(".meta.json").write_text(mtext)
            print(f"  saved: {rel} (SHA-256 {sha}, {n} rows, empty {meta['empty_rows']})")
    return 0


# ------------------------------------------------------------------------------------------------ context with notes and similarity

def nd_context(check_source: bool = True) -> dict:
    """c123 context + notes (frozen files, checked against the runs), Static with notes, similarity of the training rows among each other
    (same question and the row itself −∞)."""
    C = C123.c120()
    ctx = C123.load_context()
    n = ctx["n"]
    texts, shas = [], []
    for rel in TRAIN_NOTES:
        t, s = read_train_notes(ctx["root"], rel, n)
        texts.append(t)
        shas.append(s)
    if check_source and texts != source_texts(ctx):
        abort("frozen training notes != notes of the runs — rerun export-notes")
    notes = S2.notes_features(texts, ctx["classes"])
    ctx["note_texts"] = texts
    ctx["train_notes_sha256"] = shas
    ctx["notes"] = notes
    ctx["static_nd"] = S2.Static(ctx["spec"]["names"], ctx["mats"], ctx["draws"], ctx["grade"], ctx["seen"], ctx["classes"], notes)
    ctx["SIM"] = NS.mask_same_task(NS.similarity(texts, texts), ctx["key"], ctx["key"])
    ctx["inputs_nd"] = {**C.inputs_sha256(ctx), "train_notes": dict(zip(TRAIN_NOTES, shas))}
    return ctx


def block_nd(ctx: dict, rows: np.ndarray, FM: np.ndarray, PM: np.ndarray, exclude=(), placebo_seed: int | None = None):
    """Features "fallback_i1_nd" of the rows `rows` of ONE fold; knowledge and reference of the similarity from the other folds except
    `exclude`. placebo_seed: the three ND columns from another row of the same block (at the candidates of the own row)."""
    folds, y, K = ctx["folds"], ctx["y"], ctx["K"]
    fs = set(int(v) for v in folds[rows])
    if len(fs) != 1:
        abort(f"rows from several folds {sorted(fs)}")
    f = fs.pop()
    kn = S2.knowledge_other_folds(ctx["key"], ctx["topic_s"], y, folds, rows, K, exclude)
    pool = np.flatnonzero((folds != f) & ~np.isin(folds, list(exclude)))
    ds = NS.demo_sim(ctx["SIM"][np.ix_(rows, pool)], y[pool], K)
    kn["demo_sim"] = ds
    X, cand, names = S2.pair_features(ctx["static_nd"], rows, FM, PM, kn, S2.ND_SET)
    if placebo_seed is not None:
        R, Kc = len(rows), S2.K_CAND
        src = np.random.default_rng(placebo_seed + R).permutation(R)
        ii, cc = np.repeat(np.arange(R), Kc), cand.reshape(-1)
        X = X.copy()
        X[:, -3] = ctx["notes"]["m3_men"].astype(float)[rows[src]][ii, cc]
        X[:, -2] = ctx["notes"]["m3_last"].astype(float)[rows[src]][ii, cc]
        X[:, -1] = ds[src][ii, cc]
    return X, cand, names


def run_nested_nd(ctx: dict, std: dict, nested: dict, form: str, mode: str = "real", seed: int | None = None) -> dict:
    """As c123.run_nested (variant b) with the feature set fallback_i1_nd. mode: real, zero (ND columns 0 -> variant b bit-identical),
    placebo (seed)."""
    if mode not in ("real", "zero", "placebo") or (mode == "placebo") != (seed is not None):
        raise ValueError(f"mode {mode!r} / seed {seed!r}")
    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND
    nd = len(S2.ND_FEATS)

    def prep(X: np.ndarray) -> np.ndarray:
        if mode != "zero":
            return X
        X = X.copy()
        X[:, -nd:] = 0.0
        return X

    forced, gated, base, null = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    top_all, cand_all, protected = np.zeros((n, Kc), np.int64), np.zeros((n, Kc), np.int64), np.zeros(n, bool)
    gates, X_eval, names = {}, {}, None
    for k in range(5):
        forms_T = nested[k]
        E = np.flatnonzero(folds == k)
        XT, candT, rowsT = [], [], []
        for j in [j for j in range(5) if j != k]:
            Rj = np.flatnonzero(folds == j)
            X, cand, _ = block_nd(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], exclude=(k,), placebo_seed=seed)
            XT.append(prep(X))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = block_nd(ctx, E, std[form][E], std["pre"][E], placebo_seed=seed)
        XE = prep(XE)
        X_eval[k] = XE
        labT = [S2.labels_of(c, y[r]) for c, r in zip(candT, rowsT)]
        g = S2.gate([(XT[b], labT[b], candT[b], y[rowsT[b]]) for b in range(len(XT))], protect=True)
        model = S2.fit_clogit(np.concatenate(XT), np.concatenate(labT), Kc, S2.LAMBDA)
        top0 = S2.reorder(candE, S2.scores(model, XE, Kc))
        topE = S2.protect_none(top0, candE)
        zero = S2.protect_none(S2.reorder(candE, S2.scores(S2.zero_model(XE.shape[1]), XE, Kc)), candE)
        rr, bb = S2.map_rows(topE, y[E]), S2.map_rows(candE, y[E])
        forced[E], gated[E], base[E], null[E] = rr, (bb if g["abstain"] else rr), bb, S2.map_rows(zero, y[E])
        protected[E] = (top0 != topE).any(1)
        top_all[E], cand_all[E] = topE, candE
        gates[k] = g
    return {"forced": forced, "gated": gated, "base": base, "null": null, "top": top_all, "cand": cand_all, "protected": protected,
            "gates": gates, "X_eval": X_eval, "features": names}


def train_top_sim(ctx: dict) -> np.ndarray:
    """Top similarity per training row to all other training rows of other questions (reference of the similarity guard; the test side
    compares against all 1,500 training rows without the same question)."""
    return NS.top_sim(ctx["SIM"])


# ------------------------------------------------------------------------------------------------ nested reproduction

def cmd_oof(a) -> int:
    t0 = time.time()
    C = C123.c120()
    ctx = nd_context()
    y, seen = ctx["y"], ctx["seen"]
    std, nested = C123.prepare(ctx)
    rec_rows = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS}
    draws_q = C.C107.question_draws(ctx["q_code"])
    seeds = PLACEBO_SEEDS[: a.placebos]
    res = {"kind": "safety_nd: recipe of safety_final (variant b) with the feature set fallback_i1_nd (note mentions + example similarity) in the second stage",
           "stack": C.rel(C.STACK), "stack_sha256": C.sha256_file(C.STACK), "K": S2.K_CAND, "lambda": S2.LAMBDA, "feature_set": S2.ND_SET,
           "protect_none": True, "base_map": {f: float(rec_rows[f].mean()) for f in FORMS}, "sets": {"b": {}, "nd": {}},
           "placebo": {"seeds": list(seeds), "per_seed": []}, "inputs": ctx["inputs_nd"]}
    rows_out, null_ok, pl = {}, {}, {s: {} for s in seeds}
    for form in FORMS:
        rb = C123.run_nested(ctx, std, nested, form, "b")
        rc = run_nested_nd(ctx, std, nested, form, "real")
        rz = run_nested_nd(ctx, std, nested, form, "zero")
        null_ok[form] = C123.same_rows(rz, rb)
        for nm, rr in (("b", rb), ("nd", rc)):
            ent = {"map_forced": float(rr["forced"].mean()), "map_gated": float(rr["gated"].mean()),
                   "units_forced": int(C.C107.units(rr["forced"]).sum()), "units_gated": int(C.C107.units(rr["gated"]).sum()),
                   "abstain_folds": int(sum(g["abstain"] for g in rr["gates"].values())), "gates": {str(k): v for k, v in rr["gates"].items()},
                   "n_features": len(rr["features"]), "changed_top_share": float((rr["top"][:, 0] != rr["cand"][:, 0]).mean()),
                   "changed_top3_share": float((rr["top"][:, :3] != rr["cand"][:, :3]).any(1).mean()), "rows_protected": int(rr["protected"].sum()),
                   "vs_base": C123.boot(ctx, rr["gated"] - rec_rows[form], draws_q)}
            if nm == "nd":
                ent["vs_b"] = C123.boot(ctx, rr["forced"] - rb["forced"], draws_q)
                ent["vs_b_gated"] = C123.boot(ctx, rr["gated"] - rb["gated"], draws_q)
                ent["null_nd_zero_equals_b"] = null_ok[form]
            if form == "set":
                Xt = C123.eval_rows_in_table_order(ctx, rr["X_eval"])
                ent["eval_features_sha256"] = C.sha256_array(Xt)
                ent["features"] = rr["features"]
                ent["novel_task_share_top5"] = float(Xt[:, rr["features"].index("novel_task")].mean())
            res["sets"][nm][form] = ent
            for key in ("forced", "gated", "top", "cand", "protected"):
                rows_out[f"{nm}_{form}_{key}"] = rr[key]
        for sd in seeds:
            p = run_nested_nd(ctx, std, nested, form, "placebo", sd)
            pl[sd][form] = float((p["forced"] - rb["forced"]).mean())
        e = res["sets"]["nd"][form]
        print(f"  [{FORM_SHORT[form]}] base {res['base_map'][form]:.6f} | b {res['sets']['b'][form]['map_forced']:.6f} | ND {e['map_forced']:.6f} "
              f"(ND − b {e['vs_b']['delta']:+.5f} ± {e['vs_b']['se_q']:.5f} {e['vs_b']['fold_units']}; gate + guard {e['vs_b_gated']['delta']:+.5f}; "
              f"abstention {e['abstain_folds']}/5; new top code {e['changed_top_share']:.4f}; null point {'bit-identical' if null_ok[form] else 'VIOLATED'}; "
              f"placebo mean {np.mean([pl[s][form] for s in seeds]):+.5f}) ({time.time() - t0:.0f} s)", flush=True)
    res["null_point"] = {"nd_zero_equals_b": {"ok": bool(all(null_ok.values())), "forms": null_ok}}
    d = {f: res["sets"]["nd"][f]["vs_b"]["delta"] for f in FORMS}
    half_rows = 0.5 * ((rows_out["nd_fold_forced"] - rows_out["b_fold_forced"]) + (rows_out["nd_set_forced"] - rows_out["b_set_forced"]))
    res["nd_vs_b"] = {"F": d["fold"], "M": d["set"], "A": 0.5 * (d["fold"] + d["set"]), "A_boot": C123.boot(ctx, half_rows, draws_q)}
    per = [{"seed": s, "F": pl[s]["fold"], "M": pl[s]["set"], "A": 0.5 * (pl[s]["fold"] + pl[s]["set"])} for s in seeds]
    pa = np.array([p["A"] for p in per]) if per else np.zeros(0)
    res["placebo"].update({"per_seed": per, "n": len(per), "A_mean": float(pa.mean()) if len(pa) else None,
                           "A_sd": float(pa.std(ddof=1)) if len(pa) > 1 else None, "A_max": float(pa.max()) if len(pa) else None,
                           "share_ge_real": float((pa >= res["nd_vs_b"]["A"]).mean()) if len(pa) else None})
    OUT.mkdir(parents=True, exist_ok=True)
    OOF_JSON.write_text(C.to_json(res))
    np.savez_compressed(OOF_ROWS, **rows_out, **{f"base_{f}": rec_rows[f] for f in FORMS})
    r = res["nd_vs_b"]
    plt = (f"{res['placebo']['A_mean']:+.5f} (max {res['placebo']['A_max']:+.5f}, n {len(per)})" if per else "— (none)")
    print(f"ND − b: F {r['F']:+.5f}, M {r['M']:+.5f}, ½(F+M) {r['A']:+.5f} ± {r['A_boot']['se_q']:.5f} (over questions); placebo mean "
          f"{plt}; copy null point (ND 0 -> b) "
          f"{'bit-identical' if res['null_point']['nd_zero_equals_b']['ok'] else 'VIOLATED'}; written: {C.rel(OOF_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0 if res["null_point"]["nd_zero_equals_b"]["ok"] else 4


# ------------------------------------------------------------------------------------------------ new meta-splits

def cmd_splits(a) -> int:
    t0 = time.time()
    if a.n < MIN_SPLITS:
        abort(f"--n {a.n}: the pre-registered rule requires at least {MIN_SPLITS} meta-splits", 2)
    C = C123.c120()
    ctx0 = nd_context()
    y = ctx0["y"]
    res = {"kind": "safety-nd rule (6): ND − b over new meta-splits (channel OOFs and notes fixed, meta-folds new)", "stack": C.rel(C.STACK),
           "stack_sha256": C.sha256_file(C.STACK), "splits": [], "inputs": ctx0["inputs_nd"]}
    for s in range(a.n):
        ctx = dict(ctx0)
        ctx["folds"] = C123.split_folds(y, SPLIT_SEED0 + s)
        std, nested = C123.prepare(ctx)
        rec = {"seed": SPLIT_SEED0 + s, "placebo_seed": SPLIT_PLACEBO_SEED0 + s, "nd_minus_b": {}, "placebo_minus_b": {}, "b_map": {}}
        for form in FORMS:
            rb = C123.run_nested(ctx, std, nested, form, "b")
            rc = run_nested_nd(ctx, std, nested, form, "real")
            rp = run_nested_nd(ctx, std, nested, form, "placebo", SPLIT_PLACEBO_SEED0 + s)
            rec["b_map"][form] = float(rb["forced"].mean())
            rec["nd_minus_b"][form] = float((rc["forced"] - rb["forced"]).mean())
            rec["placebo_minus_b"][form] = float((rp["forced"] - rb["forced"]).mean())
        res["splits"].append(rec)
        print(f"  split {s:2d} (seed {rec['seed']}): ND − b F {rec['nd_minus_b']['fold']:+.4f} M {rec['nd_minus_b']['set']:+.4f} | placebo − b "
              f"F {rec['placebo_minus_b']['fold']:+.4f} M {rec['placebo_minus_b']['set']:+.4f} ({time.time() - t0:.0f} s)", flush=True)
    summ = {}
    for key in ("nd_minus_b", "placebo_minus_b"):
        F = np.array([r[key]["fold"] for r in res["splits"]])
        M = np.array([r[key]["set"] for r in res["splits"]])
        A = 0.5 * (F + M)
        summ[key] = {"F_mean": float(F.mean()), "M_mean": float(M.mean()), "A_mean": float(A.mean()), "A_se": float(A.std(ddof=1) / np.sqrt(len(A))),
                     "share_A_pos": float((A > 0).mean())}
    res["summary"] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(C.to_json(res))
    sb, sp = summ["nd_minus_b"], summ["placebo_minus_b"]
    print(f"ND − b over {a.n} meta-splits: ½(F+M) {sb['A_mean']:+.5f} ± {sb['A_se']:.5f} ({sb['share_A_pos']:.0%} > 0; F {sb['F_mean']:+.5f}, M "
          f"{sb['M_mean']:+.5f}); placebo − b {sp['A_mean']:+.5f}; written: {C.rel(SPLITS_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ pre-registered safety-nd rule (6)

def risk_rule(o: float, s: float) -> dict:
    """Risk rule (eda/risk_rule.py: NPMLE prior of class R, selection-corrected posterior): τ_d 0 (the rule's default, decides) and 0.001
    (sensitivity)."""
    RR = _load("c129_risk", COMP / "eda" / "risk_rule.py")
    wR = RR.class_r_prior()
    out = {}
    for tau in (0.0, 0.0010):
        rng = np.random.default_rng(RISK_SEED)
        tol = 0.0004
        p = RR.combined_posterior(o, s, wR, rng, k=RISK_K, rho=RISK_RHO, tau_d=tau, spread=RISK_SPREAD, n_sim=400_000, chunks=8, tol=tol)
        while (p.get("ess") or 0) < 2000 and tol < 0.0032:
            tol *= 2
            p = RR.combined_posterior(o, s, wR, rng, k=RISK_K, rho=RISK_RHO, tau_d=tau, spread=RISK_SPREAD, n_sim=400_000, chunks=8, tol=tol)
        out[f"tau_{tau:g}"] = {"tau_d": tau, "mean": float(p["mean"]), "p_lt_m002": float(p["p_lt_m002"]), "ess": p.get("ess"), "tol": tol}
    return out


def cmd_rule(a) -> int:
    C = C123.c120()
    if not OOF_JSON.is_file() or not SPLITS_JSON.is_file():
        abort(f"{C.rel(OOF_JSON)} or {C.rel(SPLITS_JSON)} missing — run c129 oof and splits first", 3)
    oof, sp = json.loads(OOF_JSON.read_text()), json.loads(SPLITS_JSON.read_text())
    if oof.get("inputs") != sp.get("inputs"):
        abort("oof and splits were computed with different inputs")
    r, pl, s = oof["nd_vs_b"], oof["placebo"], sp["summary"]["nd_minus_b"]
    ok_null = bool(oof["null_point"]["nd_zero_equals_b"]["ok"])
    ok_orig = bool(r["A"] > 0 and int(pl.get("n") or 0) >= 15 and r["A"] > float(pl["A_mean"]))
    ok_spl = bool(len(sp["splits"]) >= MIN_SPLITS and s["A_mean"] > 0)
    risk = risk_rule(float(r["A"]), float(r["A_boot"]["se_q"]))
    ok_risk = bool(risk["tau_0"]["p_lt_m002"] <= RISK_PHARM)
    ok = ok_null and ok_orig and ok_spl and ok_risk
    out = {"rule": "safety-nd rule (6)", "null_point_oof": ok_null, "original": {"A": r["A"], "F": r["F"], "M": r["M"], "se_q": r["A_boot"]["se_q"],
           "placebo_mean": pl["A_mean"], "placebo_max": pl["A_max"], "placebo_n": pl["n"], "ok": ok_orig},
           "splits": {"n": len(sp["splits"]), "A_mean": s["A_mean"], "A_se": s["A_se"], "share_pos": s["share_A_pos"], "ok": ok_spl},
           "risk_rule": {**risk, "p_harm_max": RISK_PHARM, "ok": ok_risk, "convention": "τ_d = 0 decides (the rule's default); τ_d 0.001 reported only"},
           "verdict": "deploy" if ok else "no deployment"}
    OUT.mkdir(parents=True, exist_ok=True)
    RULE_JSON.write_text(C.to_json(out))
    print(f"safety-nd rule (6): copy null point OOF {'bit-identical' if ok_null else 'VIOLATED'} | original ½(F+M) {r['A']:+.5f} ± {r['A_boot']['se_q']:.5f} vs. "
          f"placebo mean {pl['A_mean']:+.5f} -> {'yes' if ok_orig else 'NO'} | {len(sp['splits'])} meta-splits {s['A_mean']:+.5f} ± {s['A_se']:.5f} "
          f"-> {'yes' if ok_spl else 'NO'} | risk rule P(θ<−0.002) τ0 {risk['tau_0']['p_lt_m002']:.4f} (τ0.001 {risk['tau_0.001']['p_lt_m002']:.4f}), "
          f"posterior mean {risk['tau_0']['mean']:+.5f} -> {'yes' if ok_risk else 'NO'} => {out['verdict']}")
    return 0 if ok else 5


# ------------------------------------------------------------------------------------------------ deployment file

def guard_reference(ctx: dict, cand: np.ndarray) -> dict:
    st = S2.note_stats(ctx["notes"]["men_per_draw"], ctx["notes"]["m3_men"], cand, cand[:, 0])
    t = ctx["note_texts"]
    st["identical"] = {"d1-d2": S2.identical_share(t[0], t[1]), "d1-d3": S2.identical_share(t[0], t[2]), "d2-d3": S2.identical_share(t[1], t[2])}
    return st


def build_stage2(ctx: dict, oof: dict, rule: dict) -> dict:
    C = C123.c120()
    y, Kc, folds, n = ctx["y"], S2.K_CAND, ctx["folds"], ctx["n"]
    std = C123.standard_forms(ctx)
    X, cand, names = None, np.zeros((n, Kc), np.int64), None
    for k in range(5):
        E = np.flatnonzero(folds == k)
        Xk, ck, names = block_nd(ctx, E, std["set"][E], std["pre"][E])
        if X is None:
            X = np.zeros((n * Kc, Xk.shape[1]))
        X[(E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)] = Xk
        cand[E] = ck
    got = C.sha256_array(X)
    o = oof["sets"]["nd"]
    if got != o["set"].get("eval_features_sha256"):
        abort(f"null point (e) violated: features of the deployment form ({got[:16]}…) != evaluated rows of the OOF reproduction — rerun c129 oof")
    lab = S2.labels_of(cand, y)
    blocks = []
    for k in range(5):
        E = np.flatnonzero(folds == k)
        pos = (E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)
        blocks.append((X[pos], lab[pos], cand[E], y[E]))
    g = S2.gate(blocks, protect=True)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top0 = S2.reorder(cand, S2.scores(model, X, Kc))
    top = S2.protect_none(top0, cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked_protected": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    meta = json.loads((ART / f"meta_{C.M3}.json").read_text())
    for f_, r_ in zip(TEST_NOTES, TEST_DRAWS):  # the test notes must belong to the test rankings of the draws
        b = r_.split("/")
        tm = json.loads((ctx["root"] / f_).with_suffix(".meta.json").read_text())
        if tm.get("source_rank") != f"runs/{b[1]}/{b[1]}/{b[2]}/{b[3]}" or tm.get("table") != "test":
            abort(f"{f_}: source {tm.get('source_rank')} is not the test ranking {r_}")
    sim_ref = NS.sim_stats(train_top_sim(ctx))
    return {"kind": "second stage of the safety line with margin alignment as ONE set over the test rows: conditional logit over the first K "
                    "candidates (src/stage2.py, feature set fallback_i1_nd = fallback_i1 + note mentions + example similarity) + NONE guard "
                    "(eda/c129_safety_nd.py)",
            "stack": C.rel(C.STACK), "stack_sha256": C.sha256_file(C.STACK), "channels": list(ctx["stack"]["channels"]),
            "arith_beta": float(ctx["stack"]["arith_beta"]), "align_mode": "set", "align": C123.ALIGN, "align_min": C123.ALIGN_MIN,
            "K": Kc, "lambda": S2.LAMBDA, "null": False, "feature_set": S2.ND_SET, "protect_none": True, "variant": "nd",
            "draws_channel": C.M3, "draw_sources": ctx["draw_specs"], "draw_rank_source": meta["rank"]["source"], "test_draws": list(TEST_DRAWS),
            "test_notes": [{"file": f_, "rank": r_} for f_, r_ in zip(TEST_NOTES, TEST_DRAWS)],
            "train_notes": [{"file": f_, "sha256": s_} for f_, s_ in zip(TRAIN_NOTES, ctx["train_notes_sha256"])],
            "note_sim": {"tail_chars": NS.TAIL_CHARS, "top_m": NS.TOP_M, "tfidf": "word 1–2, sublinear, min_df 2, max_features 200000, training notes only",
                         "exclude": "same question (question_id|topic_name)"},
            "model": {"features": names, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
                      "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
                      "oof_changed_top_share": float(o["set"]["changed_top_share"]), "oof_changed_top3_share": float(o["set"]["changed_top3_share"]),
                      "oof_vs_base": {f: o[f]["vs_base"]["delta"] for f in FORMS}, "oof_vs_b": {f: o[f]["vs_b"]["delta"] for f in FORMS},
                      "features_sha256": got, "insample": ins},
            "novel_task_share_top5": float(o["set"]["novel_task_share_top5"]),
            "guard_reference": guard_reference(ctx, S2.candidates(std["set"], ctx["seen"])),
            "sim_reference": sim_ref,
            "guard_rule": {"changed_top_factor": list(C123.CHANGED_TOP_FACTOR), "class_mass_l1_max": C123.L1_MAX, "rule": "widened",
                           "share_z_max": SHARE_Z_MAX, "notes_rel_tolerance": NOTES_REL, "n_ref": n, "sim_rel": NS.SIM_REL, "sim_z_max": NS.SIM_Z_MAX,
                           "abort_only": True,
                           "text": "safety-nd rule (4): widened note guard, similarity guard (band ±10 % OR |z| ≤ 3) and guard (c) (½ to 2 times the OOF "
                                   "share of new top codes) each ONLY as an abort without a file, never as a switch; class masses L1 ≤ 0.035 (build script)"},
            "inputs_sha256": oof["inputs"], "rule_oof": rule,
            "based_on": {"stage2_safety_final.json": C.sha256_file(STAGE2_SAFETY_FINAL), "decision": "decisions/override_cv_condition_ii.json",
                         "build": "tools/builds/safety_nd.sh"},
            "order": "blend -> margin alignment (one set over the test rows) -> second stage over the first K -> NONE guard -> REL-01"}


def cmd_fit(a) -> int:
    C = C123.c120()
    target = Path(a.out) if a.out else STAGE2_FILE
    if target.resolve() != STAGE2_FILE.resolve() and not target.resolve().is_relative_to(OUT.resolve()):
        abort(f"fit only writes {C.rel(STAGE2_FILE)} or under {C.rel(OUT)}", 2)
    if not RULE_JSON.is_file():
        abort(f"{C.rel(RULE_JSON)} missing — run c129 rule first", 3)
    rule = json.loads(RULE_JSON.read_text())
    if rule.get("verdict") != "deploy":
        abort(f"pre-registered safety-nd rule (6) not met ({rule.get('verdict')}) — no deployment, no file", 5)
    ctx = nd_context()
    oof = json.loads(OOF_JSON.read_text())
    if oof.get("inputs") != ctx["inputs_nd"]:
        abort(f"{C.rel(OOF_JSON)} was computed with other inputs — recompute it")
    out = build_stage2(ctx, oof, rule)
    text = C.to_json(out)
    m = out["model"]
    print(f"  [safety_nd {S2.ND_SET}] {len(m['features'])} features | gate (guarded) inner {m['gate']['inner']:.5f} vs. base {m['gate']['inner_base']:.5f} -> "
          f"{'ABSTAIN (weight 0)' if m['abstain'] else 'apply'} | in-sample {m['insample']['map_base']:.5f} → {m['insample']['map_reranked_protected']:.5f} "
          f"(new top code {m['insample']['changed_top_share']:.4f}; OOF {m['oof_changed_top_share']:.4f}) | weights ND "
          f"{', '.join(f'{nm} {w:+.3f}' for nm, w in zip(m['features'][-3:], m['w'][-3:]))} | top similarity training {out['sim_reference']['mean']:.4f}")
    if a.verify:
        if not target.is_file() or target.read_text() != text:
            abort(f"{C.rel(target)} missing or not the recomputation")
        print(f"   VERIFIED: {C.rel(target)} = recomputation (SHA-256 {C.sha256_file(target)[:16]}…)")
    if a.save:
        if target.is_file() and target.read_text() != text:
            abort(f"{C.rel(target)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"   saved: {C.rel(target)} (SHA-256 {C.sha256_file(target)})")
    return 0


NULL_KEEP = ("kind", "stack", "stack_sha256", "channels", "arith_beta", "align_mode", "align", "align_min", "K", "lambda", "feature_set", "protect_none",
             "draws_channel", "draw_sources", "draw_rank_source", "test_draws", "test_notes", "train_notes", "note_sim", "novel_task_share_top5",
             "guard_reference", "sim_reference", "guard_rule")


def cmd_null(a) -> int:
    """Null model (weight 0) for the deployment file: features and guards are computed, score 0 -> base order (copy null point)."""
    ref = json.loads(Path(a.source).read_text())
    nf = len(ref["model"]["features"])
    model = {"features": ref["model"]["features"], "w": [0.0] * nf, "mu": [0.0] * nf, "sd": [1.0] * nf, "abstain": True, "oof_changed_top_share": 0.0}
    Path(a.out).write_text(C123.c120().to_json({**{k: ref[k] for k in NULL_KEEP}, "null": True, "model": model, "null_from": str(a.source)}))
    print(f"null model for {a.source} -> {a.out}")
    return 0


def cmd_b_zero(a) -> int:
    """Copy null point (safety-nd rule (5)): model of safety_final (stage2_safety_final.json, 55 features fallback_i1) + weight 0 for the
    three ND columns, otherwise the deployment file of safety_nd (notes, reference, guards) -> the test side must give the file of
    safety_final byte-identically."""
    final = json.loads(STAGE2_SAFETY_FINAL.read_text())
    if C123.c120().sha256_file(STAGE2_SAFETY_FINAL) != STAGE2_SAFETY_FINAL_SHA or final.get("feature_set") != S2.FB_I1_SET:
        abort(f"{STAGE2_SAFETY_FINAL.name} is not the deployment file of safety_final", 4)
    ref = json.loads(Path(a.source).read_text())
    m_final, mref = final["model"], ref["model"]
    nd = len(S2.ND_FEATS)
    if mref["features"][:-nd] != m_final["features"] or tuple(mref["features"][-nd:]) != S2.ND_FEATS:
        abort("features of safety_final and safety_nd do not match", 4)
    out = json.loads(json.dumps(ref))
    out["model"].update({"w": m_final["w"] + [0.0] * nd, "mu": m_final["mu"] + [0.0] * nd, "sd": m_final["sd"] + [1.0] * nd,
                         "oof_changed_top_share": m_final["oof_changed_top_share"], "oof_changed_top3_share": m_final["oof_changed_top3_share"]})
    out["b_zero_from"] = STAGE2_SAFETY_FINAL.name
    Path(a.out).write_text(C123.c120().to_json(out))
    print(f"copy null point (weights of safety_final + 0 for {', '.join(S2.ND_FEATS)}) -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-notes", help="freeze the training notes d1–d3 (data/derived/sft/)")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    o = sub.add_parser("oof", help="nested reproduction ND against b, null point, placebos -> artifacts/eda/private/safety_nd/")
    o.add_argument("--placebos", type=int, default=len(PLACEBO_SEEDS))
    s = sub.add_parser("splits", help="new meta-splits ND − b")
    s.add_argument("--n", type=int, default=MIN_SPLITS)
    sub.add_parser("rule", help="pre-registered safety-nd rule (6)")
    f = sub.add_parser("fit", help="deployment file artifacts/stage2_safety_nd.json")
    f.add_argument("--out")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    nl = sub.add_parser("null", help="null model for the deployment file")
    nl.add_argument("--out", required=True)
    nl.add_argument("--source", default=str(STAGE2_FILE))
    bz = sub.add_parser("b-zero", help="copy null point: weights of safety_final + 0 for the ND columns")
    bz.add_argument("--out", required=True)
    bz.add_argument("--source", default=str(STAGE2_FILE))
    a = ap.parse_args(argv)
    if a.cmd == "oof" and not 0 <= a.placebos <= len(PLACEBO_SEEDS):
        abort(f"--placebos {a.placebos}", 2)
    return {"export-notes": cmd_export_notes, "oof": cmd_oof, "splits": cmd_splits, "rule": cmd_rule, "fit": cmd_fit, "null": cmd_null,
            "b-zero": cmd_b_zero}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
