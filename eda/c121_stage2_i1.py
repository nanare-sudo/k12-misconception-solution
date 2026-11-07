"""SECOND STAGE WITH INTERACTIONS I1 (primary line): the guard form of the second stage (main_guard + NONE guard,
artifacts/stage2_s035.json) with the interactions I1 (feature set main_guard_i1, src/stage2.py). eda/c118_stage2.py stays unchanged;
this script only uses its building blocks as a module. Build script: tools/builds/primary_i1.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c121_stage2_i1.py oof [--stack s032_stage2]                 # nested: guard form bit-identical, I1, copy null point (i), placebos
    python -I eda/c121_stage2_i1.py splits [--stack s032_stage2] [--n 20]     # new meta-splits
    python -I eda/c121_stage2_i1.py rule                                      # pre-registered expected-value rule from oof + splits (read only)
    python -I eda/c121_stage2_i1.py fit --stack s032_stage2 (--save | --verify)   # deployment file artifacts/stage2_primary_i1.json
    python -I eda/c121_stage2_i1.py test-ref [--stack s032_stage2]            # aggregates only: reference rows of the count features, training/test
    python -I eda/c121_stage2_i1.py submit <arguments of tools/make_submission.py>   # test side with main_guard_i1 (see below)

FORM (fixed before the evaluation): I1 = {row_ent, row_max, row_d_ntop, row_ch_ntop, task_rows} × {f_logp, is_general, is_none},
product of the raw values at the end of main_guard, then standardised like all features (src/stage2.i1_interactions). Otherwise the
guard form unchanged: K 5, λ 0.1, β choice, folds, knowledge from other folds, NONE guard (also in the abstention gate), fallback
model and the note guard under the widened rule.

OOF (`oof`; as c118 guard-oof): per form (F = per fold, oL = without the last step, M = alignment as one set)
  * guard form recomputed (c118.run_nested with main_guard) — must hit the row values, orders and guard marks of
    artifacts/eda/private/stage2_guard/c118_guard_oof_rows.npz bit for bit (otherwise abort);
  * I1 (main_guard_i1 + guard); copy null point (i): I1 columns = 0 (their weights stay 0) -> guard form bit-identical (row values,
    orders, candidates, guard marks);
  * Δ against the guard form with σ over questions (c107.cluster_boot, 10,000 draws), sixths per fold, share of new top codes;
  * placebos: the row values of the five I1 row features are permuted row-wise within each block (fold) (random order
    default_rng(seed + rows of the block)), then multiplied with the unpermuted candidate features; seeds 5000 + 31·s (s 0–11), then
    900 + 17·s (s 0–2), as in the independent cross-check of the same form;
  * checksum of the evaluated features of form M (table row order) -> null point (e) in `fit`.
  -> artifacts/eda/private/stage2_i1/c121_i1_oof.json (without run time, so that it is repeatable) and c121_i1_oof_rows.npz.
SPLITS (`splits`): channel OOFs fixed, only the meta-folds are drawn anew (weights, β, alignment, knowledge, second stage):
  StratifiedKFold(5, shuffle, seed 1000 + s) by root_code; per split the guard form, I1 and a placebo of I1 (seed 7700 + s);
  Δ = mean of the row differences per form. -> artifacts/eda/private/stage2_i1/c121_i1_splits.json.
EXPECTED-VALUE RULE (fixed before the evaluation): deploy if (i) the copy null point is exact (OOF here; file part in the build script
  tools/builds/primary_i1.sh), (ii) ½(F+M) against the guard form on the original meta-split > 0 and above the mean of ≥ 5 placebos,
  (iii) the mean of ½(F+M) over ≥ 10 new meta-splits > 0. `fit` only builds the deployment file if (i)-OOF, (ii) and (iii) hold
  (otherwise exit 5, no file).
DEPLOYMENT (`fit`; as c118 guard-fit): first recomputes artifacts/stage2_s035.json character for character (c118 guard-fit --verify in
  this process), then replaces ONLY models.main by the main_guard_i1 model (features of the deployment form = evaluated rows of `oof`,
  bit-identical = null point (e); the first 48 features bit-identical those of the guard form) and appends i1_build and derived_from.
  Fallback model, note guard (reference, widened rule), β, K, λ and the checksums of the inputs stay those of stage2_s035.json ->
  artifacts/stage2_primary_i1.json.
TEST SIDE (`submit`): tools/make_submission.py admits main_guard_i1 for the main model itself. `submit` keeps the path of the original
  build: it loads make_submission.py as a module, admits the set main_guard_i1 ONLY for the main model (checks of the NONE guard as for
  main_guard; a fallback model with this set is rejected) and calls its main() with the same arguments. For all other models the path
  is exactly that of make_submission.py.

Training rows only for statistics; of the test side only aggregates (test-ref) and checksums. Notes are only counted by machine.
Exit 3 = input missing; 4 = checksum/recomputation/null point violated; 5 = pre-registered rule not met (no deployment).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import stage2 as S2  # noqa: E402

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "stage2_i1"
I1_OOF_JSON = OUT / "c121_i1_oof.json"
I1_OOF_ROWS = OUT / "c121_i1_oof_rows.npz"
I1_SPLITS_JSON = OUT / "c121_i1_splits.json"
TEST_REF_JSON = OUT / "c121_test_ref.json"
STAGE2_I1 = ART / "stage2_primary_i1.json"
STACK_DEFAULT = "s032_stage2"
FORMS = ("fold", "pre", "set")
FORM_SHORT = {"fold": "F", "pre": "oL", "set": "M"}
# placebo seeds in the order of the independent cross-check (first the 12 of its placebo run, then the 3 of its first run)
PLACEBO_SEEDS = tuple(5000 + 31 * s for s in range(12)) + tuple(900 + 17 * s for s in range(3))
SPLIT_SEED0 = 1000  # StratifiedKFold(5, shuffle=True, random_state=1000 + s)
SPLIT_PLACEBO_SEED0 = 7700  # per split s one placebo of I1 with seed 7700 + s
MIN_PLACEBOS, MIN_SPLITS = 5, 10  # expected-value rule

_C118 = None


def c118():
    """eda/c118_stage2.py as a module (loaded once; unchanged)."""
    global _C118
    if _C118 is None:
        spec = importlib.util.spec_from_file_location("c121_c118", COMP / "eda" / "c118_stage2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _C118 = mod
    return _C118


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def load_context(stack: str) -> dict:
    C = c118()
    try:
        return C.context(stack)
    except C.Missing as e:
        abort(str(e), 3)


# ------------------------------------------------------------------------------------------------ nested reproduction

def block_perm(R: int, Kc: int, seed: int) -> np.ndarray:
    """Placebo permutation: reorder the rows of a block with R rows at random (default_rng(seed + R)), the candidate slots per row
    stay -> pair indices [R·Kc]."""
    rng = np.random.default_rng(seed + R)
    return np.repeat(rng.permutation(R) * Kc, Kc) + np.tile(np.arange(Kc), R)


def run_i1(ctx: dict, std: dict, nested: dict, form: str, mode: str = "real", seed: int | None = None) -> dict:
    """As c118.run_nested for main_guard (same steps, same NONE guard also in the gate), but with the feature set main_guard_i1.
    mode: "real" (I1 as computed), "zero" (I1 columns = 0: copy null point (i), the weights of the columns stay 0), "placebo"
    (I1 from row values permuted within the block, seed `seed`). -> row values and orders as c118.run_nested."""
    if mode not in ("real", "zero", "placebo") or (mode == "placebo") != (seed is not None):
        raise ValueError(f"mode {mode!r} / seed {seed!r}")
    C = c118()
    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND
    fs = S2.I1_SET

    def prep(X: np.ndarray, names: list[str]) -> np.ndarray:
        if mode == "real":
            return X
        X = X.copy()
        cols = [names.index(nm) for nm in S2.I1_FEATS]
        if mode == "zero":
            X[:, cols] = 0.0
        else:
            X[:, cols] = S2.i1_interactions(X, names, block_perm(X.shape[0] // Kc, Kc, seed))
        return X

    forced, abst, base = np.zeros(n), np.zeros(n), np.zeros(n)
    changed, changed_forced, changed3 = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    top_all, cand_all, protected = np.zeros((n, Kc), np.int64), np.zeros((n, Kc), np.int64), np.zeros(n, bool)
    choice, X_eval, names = {}, {}, None
    for k in range(5):
        forms_T = nested[k]
        E = np.flatnonzero(folds == k)
        Tfolds = [j for j in range(5) if j != k]
        XT, candT, rowsT = [], [], []
        for j in Tfolds:
            Rj = np.flatnonzero(folds == j)
            X, cand, names_j = C.block(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], fs, exclude=(k,))
            XT.append(prep(X, names_j))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = C.block(ctx, E, std[form][E], std["pre"][E], fs)
        XE = prep(XE, names)
        X_eval[k] = XE
        labT = [S2.labels_of(c, y[r]) for c, r in zip(candT, rowsT)]
        g = S2.gate([(XT[b], labT[b], candT[b], y[rowsT[b]]) for b in range(len(Tfolds))], protect=True)
        model = S2.fit_clogit(np.concatenate(XT), np.concatenate(labT), Kc, S2.LAMBDA)
        top0 = S2.reorder(candE, S2.scores(model, XE, Kc))
        topE = S2.protect_none(top0, candE)
        protected[E] = (top0 != topE).any(1)
        rr, bb = S2.map_rows(topE, y[E]), S2.map_rows(candE, y[E])
        forced[E], abst[E], base[E] = rr, (bb if g["abstain"] else rr), bb
        changed_forced[E] = topE[:, 0] != candE[:, 0]
        changed3[E] = (topE[:, :3] != candE[:, :3]).any(1)
        changed[E] = changed_forced[E] & (not g["abstain"])
        top_all[E], cand_all[E] = topE, candE
        choice[k] = g
    return {"forced": forced, "abstain": abst, "base": base, "changed_top": changed, "changed_top_forced": changed_forced,
            "changed_top3_forced": changed3, "choice": choice, "features": names, "X_eval": X_eval, "top": top_all, "cand": cand_all,
            "protected": protected}


def prepare(ctx: dict) -> tuple[dict, list, dict]:
    """Base forms as in c118 guard-oof: standard cross-fit (evaluated rows, β per outer fold by log loss) and nested forms."""
    C = c118()
    std, betas = C.standard_forms(ctx, "select")
    L = C.C107.logs_of(ctx["mats"], ctx["spec"]["names"], ctx["spec"]["floors"], ctx["seen"], False)
    nested = {k: C.nested_forms(ctx, "select", k, L) for k in range(5)}
    return std, betas, nested


def same_rows(a: dict, b: dict, keys=("forced", "abstain", "top", "cand", "protected")) -> bool:
    return bool(all(np.array_equal(a[k], b[k]) for k in keys))


def half_fm(d: dict) -> float:
    return 0.5 * (float(d["fold"]) + float(d["set"]))


def cmd_oof(a) -> int:
    t0 = time.time()
    C = c118()
    ctx = load_context(a.stack)
    for f in (C.GUARD_OOF_JSON, C.GUARD_OOF_ROWS):
        if not f.is_file():
            abort(f"{C.rel(f)} missing — run python -I eda/c118_stage2.py guard-oof --stack {a.stack} first", 3)
    goof = json.loads(C.GUARD_OOF_JSON.read_text())
    inputs = C.inputs_sha256(ctx)
    if goof.get("stack_sha256") != C.sha256_file(ctx["stack_file"]) or goof.get("inputs") != inputs:
        abort(f"{C.rel(C.GUARD_OOF_JSON)} was computed with a different blend file or different inputs — rerun c118 guard-oof")
    ref = np.load(C.GUARD_OOF_ROWS)
    y, seen, G = ctx["y"], ctx["seen"], S2.GUARD_SET
    draws_q = C.C107.question_draws(ctx["q_code"])
    std, betas, nested = prepare(ctx)
    rec_rows = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS}
    seeds = PLACEBO_SEEDS[: a.placebos]
    res = {"kind": "guard form (main_guard + NONE guard) with the interactions I1 (main_guard_i1), nested as in c118 guard-oof",
           "stack": C.rel(ctx["stack_file"]), "stack_sha256": C.sha256_file(ctx["stack_file"]), "beta_outer": betas, "K": S2.K_CAND,
           "lambda": S2.LAMBDA, "feature_set": S2.I1_SET, "protect_none": True,
           "interactions": {"row_features": list(S2.I1_ROW_FEATS), "candidate_features": list(S2.I1_CAND_FEATS), "features": list(S2.I1_FEATS)},
           "reference_rows": C.rel(C.GUARD_OOF_ROWS), "sets": {"guard": {}, "i1": {}}, "placebo": {"seeds": list(seeds), "per_seed": []},
           "inputs": inputs}
    rows_out, guard, null_ok = {}, {}, {}
    pl_delta = {s: {} for s in seeds}
    for form in FORMS:
        s = C.run_nested(ctx, std, nested, form, G)  # guard form recomputed (main_guard + guard)
        want = {key: ref[f"{G}_{form}_{key}"] for key in ("forced", "abstain", "top", "cand", "protected")}
        if not same_rows(s, want):
            abort(f"guard form [{form}] does not hit the row values of {C.rel(C.GUARD_OOF_ROWS)} bit for bit")
        if not np.allclose(s["base"], rec_rows[form]):
            abort(f"base of the candidate order [guard|{form}] != recipe")
        guard[form] = s
        res["sets"]["guard"][form] = {"map_forced": float(s["forced"].mean()), "abstain_folds": int(sum(c["abstain"] for c in s["choice"].values())),
                                      "changed_top_share_forced": float(s["changed_top_forced"].mean()), "rows_equal_reference": True}
        r = run_i1(ctx, std, nested, form, "real")
        if not np.allclose(r["base"], rec_rows[form]):
            abort(f"base of the candidate order [I1|{form}] != recipe")
        z = run_i1(ctx, std, nested, form, "zero")
        null_ok[form] = same_rows(z, s)
        d = r["forced"] - s["forced"]
        ent = {"map_forced": float(r["forced"].mean()), "map_abstain": float(r["abstain"].mean()),
               "abstain_folds": int(sum(c["abstain"] for c in r["choice"].values())), "gate": {str(k): v for k, v in r["choice"].items()},
               "n_features": len(r["features"]), "changed_top_share": float(r["changed_top"].mean()),
               "changed_top_share_forced": float(r["changed_top_forced"].mean()),
               "changed_top3_share_forced": float(r["changed_top3_forced"].mean()),
               "vs_guard": C.boot(d, ctx, draws_q), "vs_guard_with_abstention": C.boot(r["abstain"] - s["abstain"], ctx, draws_q),
               "vs_recipe_beta": C.boot(r["forced"] - rec_rows[form], ctx, draws_q), "rows_protected": int(r["protected"].sum()),
               "rows_other_top1_vs_guard": int((r["top"][:, 0] != s["top"][:, 0]).sum()),
               "rows_other_top3_vs_guard": int((r["top"][:, :3] != s["top"][:, :3]).any(1).sum()),
               "rows_score_changed_vs_guard": int((d != 0).sum()), "null_i1_zero_equals_guard": null_ok[form]}
        if form == "set":
            ent["eval_features_sha256"] = C.sha256_array(C.eval_rows_in_table_order(ctx, r["X_eval"]))
            ent["features"] = r["features"]
        res["sets"]["i1"][form] = ent
        for key in ("forced", "abstain", "top", "cand", "protected"):
            rows_out[f"i1_{form}_{key}"] = r[key]
        for sd in seeds:
            p = run_i1(ctx, std, nested, form, "placebo", sd)
            pl_delta[sd][form] = float((p["forced"] - s["forced"]).mean())
            rows_out[f"placebo{sd}_{form}_forced"] = p["forced"]
        v = ent["vs_guard"]
        print(f"  [I1|{form:4s}] {ent['n_features']} features | MAP {ent['map_forced']:.6f} (guard form {res['sets']['guard'][form]['map_forced']:.6f}) | "
              f"vs. guard form {v['delta']:+.5f} ± {v['se_q']:.5f} {v['fold_units']} | null point (i) {'bit-identical' if null_ok[form] else 'VIOLATED'} | "
              f"abstention {ent['abstain_folds']}/5 | new top code {ent['changed_top_share_forced']:.4f} | other top code than the guard form "
              f"{ent['rows_other_top1_vs_guard']} | placebo mean {np.mean([pl_delta[q][form] for q in seeds]):+.5f}", flush=True)
    res["null_point_i"] = {"ok": bool(all(null_ok.values())), "forms": null_ok}
    summ = {FORM_SHORT[f]: res["sets"]["i1"][f]["vs_guard"]["delta"] for f in FORMS}
    summ["A"] = half_fm({f: res["sets"]["i1"][f]["vs_guard"]["delta"] for f in FORMS})
    summ["units"] = {FORM_SHORT[f]: res["sets"]["i1"][f]["vs_guard"]["units"] for f in FORMS}
    res["summary"] = summ
    per = [{"seed": sd, "F": pl_delta[sd]["fold"], "oL": pl_delta[sd]["pre"], "M": pl_delta[sd]["set"], "A": half_fm(pl_delta[sd])} for sd in seeds]
    pa = np.array([p["A"] for p in per])
    res["placebo"].update({"per_seed": per, "n": len(per), "A_mean": float(pa.mean()) if len(pa) else None,
                           "A_sd": float(pa.std(ddof=1)) if len(pa) > 1 else None, "A_max": float(pa.max()) if len(pa) else None,
                           "share_ge_real": float((pa >= summ["A"]).mean()) if len(pa) else None,
                           "F_mean": float(np.mean([p["F"] for p in per])) if per else None,
                           "M_mean": float(np.mean([p["M"] for p in per])) if per else None})
    OUT.mkdir(parents=True, exist_ok=True)
    I1_OOF_JSON.write_text(C.to_json(res))
    np.savez_compressed(I1_OOF_ROWS, **rows_out)
    pl = res["placebo"]
    print(f"I1 vs. guard form: F {summ['F']:+.5f} | oL {summ['oL']:+.5f} | M {summ['M']:+.5f} | ½(F+M) {summ['A']:+.5f}; placebos (n {pl['n']}) ½(F+M) mean "
          f"{pl['A_mean']:+.5f} sd {pl['A_sd']:.5f} max {pl['A_max']:+.5f}, share ≥ real {pl['share_ge_real']:.3f}; copy null point (i) "
          f"{'PASSED' if res['null_point_i']['ok'] else 'VIOLATED'}; written: {C.rel(I1_OOF_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0 if res["null_point_i"]["ok"] else 4


# ------------------------------------------------------------------------------------------------ new meta-splits

def split_folds(y: np.ndarray, seed: int) -> np.ndarray:
    """StratifiedKFold(5, shuffle=True, random_state=seed) by root_code -> fold per row."""
    from sklearn.model_selection import StratifiedKFold

    f = np.zeros(len(y), np.int64)
    with warnings.catch_warnings():  # some classes have fewer than 5 rows (warning of sklearn, suppressed)
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            f[te] = k
    return f


def cmd_splits(a) -> int:
    t0 = time.time()
    if a.n < MIN_SPLITS:
        abort(f"--n {a.n}: the pre-registered rule requires at least {MIN_SPLITS} meta-splits", 2)
    C = c118()
    ctx0 = load_context(a.stack)
    y = ctx0["y"]
    res = {"kind": "expected-value rule (iii): I1 over new meta-splits (channel OOFs fixed, meta-folds new)",
           "stack": C.rel(ctx0["stack_file"]), "stack_sha256": C.sha256_file(ctx0["stack_file"]), "split_seeds": [], "splits": [],
           "inputs": C.inputs_sha256(ctx0)}
    for s in range(a.n):
        seed = SPLIT_SEED0 + s
        ctx = dict(ctx0)
        ctx["folds"] = split_folds(y, seed)
        std, betas, nested = prepare(ctx)
        rec = {"seed": seed, "placebo_seed": SPLIT_PLACEBO_SEED0 + s, "beta_outer": betas, "base_map": {}, "i1": {}, "i1_placebo": {}}
        for form in FORMS:
            b = C.run_nested(ctx, std, nested, form, S2.GUARD_SET)
            r = run_i1(ctx, std, nested, form, "real")
            p = run_i1(ctx, std, nested, form, "placebo", SPLIT_PLACEBO_SEED0 + s)
            rec["base_map"][form] = float(b["forced"].mean())
            rec["i1"][form] = float((r["forced"] - b["forced"]).mean())
            rec["i1_placebo"][form] = float((p["forced"] - b["forced"]).mean())
        res["split_seeds"].append(seed)
        res["splits"].append(rec)
        print(f"  split {s:2d} (seed {seed}): guard form F {rec['base_map']['fold']:.4f} M {rec['base_map']['set']:.4f} | I1 F {rec['i1']['fold']:+.4f} "
              f"M {rec['i1']['set']:+.4f} | placebo F {rec['i1_placebo']['fold']:+.4f} M {rec['i1_placebo']['set']:+.4f} ({time.time() - t0:.0f} s)",
              flush=True)
    summ = {}
    for key in ("i1", "i1_placebo"):
        F = np.array([r[key]["fold"] for r in res["splits"]])
        oL = np.array([r[key]["pre"] for r in res["splits"]])
        M = np.array([r[key]["set"] for r in res["splits"]])
        A = 0.5 * (F + M)
        summ[key] = {"F_mean": float(F.mean()), "oL_mean": float(oL.mean()), "M_mean": float(M.mean()), "A_mean": float(A.mean()),
                     "A_sd": float(A.std(ddof=1)), "A_se": float(A.std(ddof=1) / np.sqrt(len(A))), "share_F_pos": float((F > 0).mean()),
                     "share_M_pos": float((M > 0).mean()), "share_A_pos": float((A > 0).mean())}
    Ad = np.array([0.5 * (r["i1"]["fold"] + r["i1"]["set"] - r["i1_placebo"]["fold"] - r["i1_placebo"]["set"]) for r in res["splits"]])
    summ["i1_minus_placebo"] = {"A_mean": float(Ad.mean()), "A_se": float(Ad.std(ddof=1) / np.sqrt(len(Ad))), "share_A_pos": float((Ad > 0).mean())}
    base_F = np.array([r["base_map"]["fold"] for r in res["splits"]])
    base_M = np.array([r["base_map"]["set"] for r in res["splits"]])
    summ["guard_map"] = {"F_mean": float(base_F.mean()), "F_sd": float(base_F.std(ddof=1)), "M_mean": float(base_M.mean()),
                         "M_sd": float(base_M.std(ddof=1))}
    res["summary"] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    I1_SPLITS_JSON.write_text(C.to_json(res))
    si, sp = summ["i1"], summ["i1_placebo"]
    print(f"I1 over {a.n} meta-splits: F {si['F_mean']:+.5f} ({si['share_F_pos']:.0%} > 0) | M {si['M_mean']:+.5f} ({si['share_M_pos']:.0%}) | ½(F+M) "
          f"{si['A_mean']:+.5f} ± {si['A_se']:.5f} ({si['share_A_pos']:.0%}); placebo ½(F+M) {sp['A_mean']:+.5f} ({sp['share_A_pos']:.0%}); I1 − placebo "
          f"{summ['i1_minus_placebo']['A_mean']:+.5f} ± {summ['i1_minus_placebo']['A_se']:.5f}; written: {C.rel(I1_SPLITS_JSON)} "
          f"({time.time() - t0:.1f} s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ expected-value rule

def rule_result(oof: dict, splits: dict) -> dict:
    """Pre-registered expected-value rule from the results of `oof` and `splits`. (i) only the OOF part here; the file part (I1 weights
    0 -> guard-form file byte-identical) is checked by the build script, without it there is no file."""
    A, pl = float(oof["summary"]["A"]), oof["placebo"]
    n_pl = int(pl.get("n") or 0)
    ok_ii = bool(A > 0 and n_pl >= MIN_PLACEBOS and A > float(pl["A_mean"]))
    si = splits["summary"]["i1"]
    n_sp = len(splits["splits"])
    ok_iii = bool(n_sp >= MIN_SPLITS and float(si["A_mean"]) > 0)
    ok_i = bool(oof["null_point_i"]["ok"])
    return {"rule": "expected-value rule",
            "i_oof": {"ok": ok_i, "forms": oof["null_point_i"]["forms"]},
            "i_file": "build script tools/builds/primary_i1.sh: I1 weights 0 -> guard-form file and test matrix byte-identical (otherwise abort, no file)",
            "ii": {"ok": ok_ii, "half_F_plus_M": A, "F": float(oof["summary"]["F"]), "M": float(oof["summary"]["M"]), "oL": float(oof["summary"]["oL"]),
                   "placebo_n": n_pl, "placebo_mean": float(pl["A_mean"]), "placebo_max": float(pl["A_max"]),
                   "placebo_share_ge_real": float(pl["share_ge_real"])},
            "iii": {"ok": ok_iii, "n_splits": n_sp, "half_F_plus_M_mean": float(si["A_mean"]), "se": float(si["A_se"]), "share_pos": float(si["share_A_pos"]),
                    "F_mean": float(si["F_mean"]), "M_mean": float(si["M_mean"])},
            "verdict_oof": "deploy" if (ok_i and ok_ii and ok_iii) else "no deployment"}


def load_results(ctx: dict | None) -> tuple[dict, dict]:
    C = c118()
    for f in (I1_OOF_JSON, I1_SPLITS_JSON):
        if not f.is_file():
            abort(f"{C.rel(f)} missing — run python -I eda/c121_stage2_i1.py oof and splits first", 3)
    oof, sp = json.loads(I1_OOF_JSON.read_text()), json.loads(I1_SPLITS_JSON.read_text())
    if ctx is not None:
        sha, inputs = C.sha256_file(ctx["stack_file"]), C.inputs_sha256(ctx)
        for nm, o in ((I1_OOF_JSON, oof), (I1_SPLITS_JSON, sp)):
            if o.get("stack_sha256") != sha or o.get("inputs") != inputs:
                abort(f"{C.rel(nm)} was computed with a different blend file or different inputs — recompute it")
    return oof, sp


def cmd_rule(a) -> int:
    oof, sp = load_results(None)
    r = rule_result(oof, sp)
    ii, iii = r["ii"], r["iii"]
    print(f"expected-value rule: (i) OOF null point {'bit-identical' if r['i_oof']['ok'] else 'VIOLATED'} | (ii) ½(F+M) {ii['half_F_plus_M']:+.5f} > 0 and > "
          f"placebo mean {ii['placebo_mean']:+.5f} (n {ii['placebo_n']}) -> {'yes' if ii['ok'] else 'NO'} | (iii) mean over {iii['n_splits']} "
          f"meta-splits {iii['half_F_plus_M_mean']:+.5f} ± {iii['se']:.5f} -> {'yes' if iii['ok'] else 'NO'} => {r['verdict_oof']} "
          "(file null point in the build script)")
    return 0 if r["verdict_oof"] == "deploy" else 5


# ------------------------------------------------------------------------------------------------ deployment file

def cmd_fit(a) -> int:
    C = c118()
    target = Path(a.out) if a.out else STAGE2_I1
    if target.resolve() in (C.STAGE2_FILE.resolve(), C.STAGE2_WIDENED.resolve(), C.STAGE2_GUARD.resolve()):
        abort(f"fit never writes stage2_s032/s034/s035.json — separate output (default {C.rel(STAGE2_I1)})", 2)
    # 1. recompute stage2_s035.json character for character (c118 guard-fit --verify, unchanged, in this process)
    print("  recomputation of stage2_s035.json (c118 guard-fit --verify):", flush=True)
    rc = C.cmd_guard_fit(argparse.Namespace(stack=a.stack, out=None, verify=True, save=False))
    if rc != 0:
        abort(f"c118 guard-fit --verify code {rc}")
    if not C.STAGE2_GUARD.is_file():
        abort(f"{C.rel(C.STAGE2_GUARD)} missing", 3)
    guard_text = C.STAGE2_GUARD.read_text()
    guard = json.loads(guard_text)
    ctx = load_context(a.stack)
    oof, sp = load_results(ctx)
    rule = rule_result(oof, sp)
    if rule["verdict_oof"] != "deploy":
        abort(f"expected-value rule not met ({rule['verdict_oof']}: (i) {rule['i_oof']['ok']}, (ii) {rule['ii']['ok']}, (iii) {rule['iii']['ok']}) "
              "— no deployment, no file", 5)
    y, Kc, I = ctx["y"], S2.K_CAND, S2.I1_SET
    forms = C.deploy_forms(ctx)
    X, cand, names = C.deploy_features(ctx, forms, I)
    got = C.sha256_array(X)
    want = (oof["sets"]["i1"].get("set") or {}).get("eval_features_sha256")
    if got != want:
        abort(f"null point (e) violated: features of the deployment form [{I}] ({got[:16]}…) != evaluated rows of c121 oof ({str(want)[:16]}…)")
    mm = guard["models"]["main"]
    nm = len(mm["features"])
    if mm.get("feature_set") != S2.GUARD_SET or names[:nm] != mm["features"] or names[nm:] != list(S2.I1_FEATS) \
            or C.sha256_array(X[:, :nm]) != mm["features_sha256"]:
        abort("the first features of main_guard_i1 are not the features of main_guard from stage2_s035.json (bit-identical)")
    lab = S2.labels_of(cand, y)
    g = S2.gate(C.fold_blocks(ctx, X, cand, lab), protect=True)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top0 = S2.reorder(cand, S2.scores(model, X, Kc))
    top = S2.protect_none(top0, cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    o = oof["sets"]["i1"]
    gm = {"features": names, "feature_set": I, "protect_none": True, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
          "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
          "oof_changed_top_share": float(o["set"]["changed_top_share_forced"]),
          "oof_changed_top_share_with_abstention": float(o["set"]["changed_top_share"]),
          "oof_changed_top3_share": float(o["set"]["changed_top3_share_forced"]),
          "oof_vs_guard": {f: o[f]["vs_guard"]["delta"] for f in FORMS}, "oof_vs_recipe_beta": {f: o[f]["vs_recipe_beta"]["delta"] for f in FORMS},
          "features_sha256": got, "insample": ins}
    new = {}
    for k, v in guard.items():
        if k != "derived_from":
            new[k] = {"main": gm, "fallback": v["fallback"]} if k == "models" else v
    new["i1_build"] = {"rule": "expected-value rule", "feature_set": I, "protect_none": True,
                       "interactions": {"row_features": list(S2.I1_ROW_FEATS), "candidate_features": list(S2.I1_CAND_FEATS), "features": list(S2.I1_FEATS),
                                        "definition": "product of the raw values X[r]·X[c] (row feature outer, candidate feature inner), at the end of "
                                                      "main_guard, standardised like all features"},
                       "oof": C.rel(I1_OOF_JSON), "splits": C.rel(I1_SPLITS_JSON), "rule_oof": rule,
                       "reference_rows_test": "knowledge features of the test side (task_rows and others) from all 1,500 training rows as in the guard form, not rescaled",
                       "test_side": "eda/c121_stage2_i1.py submit admits the set only for the main model (checks as for main_guard) and calls "
                                    "make_submission.main() with the same arguments",
                       "fallback": "models.fallback unchanged from stage2_s035.json (fallback = the fallback-form file)"}
    new["derived_from"] = {"file": C.rel(C.STAGE2_GUARD), "sha256": C.sha256_file(C.STAGE2_GUARD),
                           "differs_only_in": ["models.main", "i1_build", "derived_from"], "build": "tools/builds/primary_i1.sh",
                           "previous": guard.get("derived_from")}
    text = C.to_json(new)
    print(f"  [{I}] {len(names)} features (main_guard {nm} bit-identical + {len(S2.I1_FEATS)} interactions) | inner gate {g['inner']:.5f} vs. base "
          f"{g['inner_base']:.5f} -> {'ABSTAIN (weight 0)' if g['abstain'] else 'apply'} | in-sample {ins['map_base']:.5f} → {ins['map_reranked']:.5f} "
          f"(new top code {ins['changed_top_share']:.4f}, guarded {ins['rows_protected']}; OOF {gm['oof_changed_top_share']:.4f}) | I1 weights "
          + ", ".join(f"{f} {w:+.3f}" for f, w in zip(names[nm:], gm["w"][nm:])))
    print(f"  expected-value rule (OOF): {rule['verdict_oof']} | fallback, note guard (widened rule), β, K, λ, reference = {C.rel(C.STAGE2_GUARD)} (SHA-256 "
          f"{new['derived_from']['sha256'][:16]}…)")
    if a.verify:
        if not target.is_file():
            abort(f"{C.rel(target)} missing — build it with --save first", 3)
        if target.read_text() != text:
            abort(f"{C.rel(target)} is not the recomputation")
        print(f"   VERIFIED: {C.rel(target)} = recomputation (SHA-256 {C.sha256_file(target)[:16]}…)")
    if a.save:
        if target.is_file() and target.read_text() != text:
            abort(f"{C.rel(target)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        target.write_text(text)
        print(f"   saved: {C.rel(target)} (SHA-256 {C.sha256_file(target)})")
    return 0


# ------------------------------------------------------------------------------------------------ reference rows of the count features (aggregates)

def cmd_test_ref(a) -> int:
    """Aggregates only: number of reference rows per question (task_rows) and per topic (topic_rows) in the deployment form of the
    training rows (other folds, 1,200 rows) against the test side (all 1,500 training rows, as make_submission.apply_stage2). No
    individual test row."""
    C = c118()
    ctx = load_context(a.stack)
    from vlm import common as VC

    test = VC.load_table(ctx["root"], "test")
    K = ctx["K"]
    tr_task, tr_topic = np.zeros(ctx["n"]), np.zeros(ctx["n"])
    for k in range(5):
        E = np.flatnonzero(ctx["folds"] == k)
        kn = S2.knowledge_other_folds(ctx["key"], ctx["topic_s"], ctx["y"], ctx["folds"], E, K)
        tr_task[E], tr_topic[E] = kn["task_rows"], kn["topic_rows"]
    kt = S2.knowledge(ctx["key"], ctx["topic_s"], ctx["y"], S2.task_keys(test["question_id"].to_numpy(), test["topic_name"].to_numpy()),
                      np.asarray(test["topic_name"].to_numpy()).astype(str), K)

    def agg(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "q90": float(np.quantile(v, 0.9)), "share_zero": float((v == 0).mean())}

    res = {"kind": "reference rows of the count features (aggregates only)", "train_deploy": {"task_rows": agg(tr_task), "topic_rows": agg(tr_topic)},
           "test": {"task_rows": agg(kt["task_rows"]), "topic_rows": agg(kt["topic_rows"]), "n_rows": int(len(test))},
           "ref_rows": {"train_deploy": "other four folds (1,200 rows)", "test": f"all {ctx['n']} training rows"}}
    res["ratio_mean_task_rows"] = res["test"]["task_rows"]["mean"] / res["train_deploy"]["task_rows"]["mean"]
    res["ratio_mean_topic_rows"] = res["test"]["topic_rows"]["mean"] / res["train_deploy"]["topic_rows"]["mean"]
    OUT.mkdir(parents=True, exist_ok=True)
    TEST_REF_JSON.write_text(C.to_json(res))
    t, d = res["test"]["task_rows"], res["train_deploy"]["task_rows"]
    print(f"task_rows: deployment form training mean {d['mean']:.3f} (median {d['median']:.0f}, without reference {d['share_zero']:.3f}) | test side mean {t['mean']:.3f} "
          f"(median {t['median']:.0f}, without reference {t['share_zero']:.3f}) -> ratio {res['ratio_mean_task_rows']:.3f}; topic_rows ratio "
          f"{res['ratio_mean_topic_rows']:.3f}; written: {C.rel(TEST_REF_JSON)}")
    return 0


# ------------------------------------------------------------------------------------------------ test side: make_submission with main_guard_i1

def model_form_with_i1(orig):
    """Wrapper around make_submission._stage2_model_form: admits main_guard_i1 ONLY for the main model (checks of protect_none as for
    main_guard, through the original function); every other model goes unchanged through the original function."""
    def form(model: dict, fs: str, classes: list[str]) -> tuple[str, bool]:
        fs_feat = model.get("feature_set", fs)
        if fs_feat != S2.I1_SET:
            return orig(model, fs, classes)
        if fs != "main":
            raise SystemExit(f"--stage2: the fallback model names the feature set {fs_feat!r}, which reads the notes")
        _, protect = orig({**model, "feature_set": S2.GUARD_SET}, fs, classes)
        return fs_feat, protect
    return form


def load_make_submission(name: str = "make_submission_c121"):
    """Load tools/make_submission.py as a module (file unchanged) and admit main_guard_i1 in this process."""
    spec = importlib.util.spec_from_file_location(name, COMP / "tools" / "make_submission.py")
    MS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MS)
    if MS.S2 is not S2:
        raise RuntimeError("make_submission loads a different stage2 module")
    MS._stage2_model_form = model_form_with_i1(MS._stage2_model_form)
    return MS


def cmd_submit(argv: list[str]) -> int:
    MS = load_make_submission()
    sys.argv = ["make_submission.py", *argv]
    return MS.main()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "submit":  # all other arguments belong to make_submission.py
        return cmd_submit(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof", help="nested reproduction guard form + I1 + copy null point (i) + placebos -> artifacts/eda/private/stage2_i1/")
    o.add_argument("--stack", default=STACK_DEFAULT)
    o.add_argument("--placebos", type=int, default=len(PLACEBO_SEEDS), help=f"number of placebos (at most {len(PLACEBO_SEEDS)}, fixed seeds)")
    s = sub.add_parser("splits", help="new meta-splits -> artifacts/eda/private/stage2_i1/")
    s.add_argument("--stack", default=STACK_DEFAULT)
    s.add_argument("--n", type=int, default=20)
    sub.add_parser("rule", help="pre-registered expected-value rule from oof + splits")
    f = sub.add_parser("fit", help="build (--save) or recompute (--verify) the deployment file artifacts/stage2_primary_i1.json")
    f.add_argument("--stack", default=STACK_DEFAULT)
    f.add_argument("--out", help="output file (default artifacts/stage2_primary_i1.json; never stage2_s032/s034/s035.json)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    t = sub.add_parser("test-ref", help="aggregates only: reference rows of the count features, training/test")
    t.add_argument("--stack", default=STACK_DEFAULT)
    sub.add_parser("submit", help="tools/make_submission.py with main_guard_i1 (all further arguments go to make_submission)")
    a = ap.parse_args(argv)
    if a.cmd == "oof" and not 0 <= a.placebos <= len(PLACEBO_SEEDS):
        abort(f"--placebos {a.placebos}", 2)
    return {"oof": cmd_oof, "splits": cmd_splits, "rule": cmd_rule, "fit": cmd_fit, "test-ref": cmd_test_ref}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
