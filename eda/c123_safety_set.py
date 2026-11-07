"""SAFETY LINE WITH SET ALIGNMENT: the channels of the strictly inductive line (with second stage, artifacts/stage2_s036_safe.json) with
the MARGIN ALIGNMENT OVER THE TEST SET (as one set, as in the primary line) instead of the inductive one, and the second stage fitted
anew in this form. eda/c120_s036_safe_stage2.py stays unchanged; this script only uses its building blocks as a module. Build script:
tools/builds/safety_final.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c123_safety_set.py oof                          # nested: a (fallback), b (fallback + I1), null points, placebos
    python -I eda/c123_safety_set.py splits [--n 20]              # new meta-splits: b − a (expected-value rule)
    python -I eda/c123_safety_set.py rule [--test-report <file>]  # pre-registered safety-set rule (with a test report also the L1 part)
    python -I eda/c123_safety_set.py fit (--save | --verify)      # deployment file artifacts/stage2_safety_final.json (variant by the rule)
    python -I eda/c123_safety_set.py fit --probe --save           # PROBE despite a missed rule, private only (safety_set/probe/), never deployed
    python -I eda/c123_safety_set.py rule|fit … --override-ii override_cv_condition_ii   # deployment despite a missed CV condition (ii),
                                                                  #   only with the pinned decision file decisions/override_cv_condition_ii.json
    python -I eda/c123_safety_set.py null --out <file>            # null model (weight 0) for the deployment file
    python -I eda/c123_safety_set.py test-mass --name <blend> --align-mode set|inductive -- <sources>   # L1 of the test masses (aggregate)

RECIPE (fixed before the evaluation): channels of the strictly inductive line v1r, zs27t, think18sb, pr27xm, m3x3_full; floors and β 0.4
as in the strictly inductive line with second stage (blend file of that line, checked by c120.context); weights per fold by log loss
(a2.fit_mix). Last step margin alignment 0.75 for classes with at least 3 training rows:
  F  per fold (weights on the 300 evaluated rows, target from the fitting rows; = c107.cross_fit with last step "trans"),
  oL without the last step (reported only),
  M  as ONE set over 1,500 rows (c107.set_align; deployment form, as on the test side: make_submission --align 0.75 --align-min 3).
Second stage: conditional logit K 5, λ 0.1, feature set "fallback" (40 features, no notes), knowledge of the training rows only from
the other folds, NONE guard (also in the abstention gate, stage2.gate(protect=True)), then REL-01 (test side). Variant b: "fallback" + the
15 interactions I1 of the primary line (src/stage2.i1_interactions, appended at the end). No teacher channel, no 397B, no notes.

OOF (`oof`; nested as in c118/c120): outer fold k = evaluated rows (standard cross-fit, knowledge from the folds ≠ k); training rows T
(folds ≠ k) with base forms from the folds except {j, k}, set form over T with the target from T; gate over the four inner folds
(guarded). Checks: standard cross-fit = c107.cross_fit(last "trans") bit-identical, blend before the last step = that of the strictly
inductive line (c120) bit-identical, copy null point a (weight 0 -> base bit-identical), copy null point b (I1 columns 0 -> a
bit-identical). Comparison with the strictly inductive line in its deployment form (F, inductive, gate + guard; row values from
artifacts/eda/private/stage2_inductive/c120_oof_rows.npz, 7,410/9,000), paired, σ over questions. Placebos for b as in the primary line
(row values of the I1 factors permuted within the blocks, the same 15 seeds). Class masses of the forms after the alignment against the
target (L1). -> artifacts/eda/private/safety_set/c123_oof.json (without run time) and c123_oof_rows.npz.
SPLITS (`splits`): channel OOFs fixed, meta-folds new (StratifiedKFold(5, shuffle, seed 1000 + s) by root_code); per split a, b and a
placebo of b (seed 7700 + s). -> artifacts/eda/private/safety_set/c123_splits.json.
SAFETY-SET RULE (fixed before the evaluation): (i) copy null points (OOF here; file part in the build script tools/builds/safety_final.sh:
weight 0 -> file without second stage byte-identical; chain of the strictly inductive line unchanged byte-identical); (ii) a counts if
CV(a, M, gate + guard) ≥ CV(strictly inductive line, F inductive) − 0.002 AND the class masses of the test side after the alignment lie at
most L1 0.035 from the target (test report of make_submission, field class_mass); (iii) b instead of a only if ½(F+M) b − a > 0, above the
placebo mean (≥ 5) and in the mean over ≥ 10 new meta-splits > 0.
OVERRIDE (decision of the entrant of 2026-11-06, decisions/override_cv_condition_ii.json): with --override-ii override_cv_condition_ii
the missed CV condition of (ii) counts as overridden — only it; the test-mass condition, (i), (iii), null points and guards stay in force.
The decision file must hit the SHA-256 pinned here (DECISION_SHA), be active and name this rule; the result file of the rule
safety_set/c123_rule_override_cv_condition_ii.json (without the switch: safety_set/c123_rule.json) and the deployment file carry the field
override_ii. Without the switch everything as before (exit 5).
DEPLOYMENT (`fit`): variant by the rule; training rows = all 1,500 in form M (standard cross-fit + set over 1,500 rows), knowledge only from
the other folds; gate over the five folds (guarded); features of the deployment form = evaluated rows of the OOF reproduction (null point
(e)). Test side: tools/make_submission.py --align 0.75 --align-min 3 --stage2 artifacts/stage2_safety_final.json (align_mode "set" ->
apply_stage2_safe).

Training rows only for statistics; of the test side only aggregates (class masses from the report). Exit 3 = input missing; 4 = checksum/
recomputation/null point violated; 5 = pre-registered rule not met (no deployment).
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
OUT = ART / "eda" / "private" / "safety_set"
OOF_JSON = OUT / "c123_oof.json"
OOF_ROWS = OUT / "c123_oof_rows.npz"
SPLITS_JSON = OUT / "c123_splits.json"
STAGE2_FILE = ART / "stage2_safety_final.json"
PROBE_STAGE2 = OUT / "probe" / "stage2_safety_a_probe.json"  # probe (private, never deployed); variant b with --out …/stage2_safety_b_probe.json
FORMS = ("fold", "pre", "set")
FORM_SHORT = {"fold": "F", "pre": "oL", "set": "M"}
ALIGN, ALIGN_MIN = 0.75, 3
FEATURE_SET = "fallback"
VARIANT_SET = {"a": FEATURE_SET, "b": S2.FB_I1_SET}  # variant b = fallback + interactions I1
INDUCTIVE_UNITS = 7410  # strictly inductive line with second stage in its deployment form (F inductive, gate + guard; c120 oof)
CV_MARGIN = 0.002  # rule (ii)
L1_MAX = 0.035  # rule (ii): class masses of the test side after the alignment
PLACEBO_SEEDS = tuple(5000 + 31 * s for s in range(12)) + tuple(900 + 17 * s for s in range(3))  # as in eda/c121_stage2_i1.py
SPLIT_SEED0, SPLIT_PLACEBO_SEED0 = 1000, 7700
MIN_PLACEBOS, MIN_SPLITS = 5, 10
CHANGED_TOP_FACTOR = (0.5, 2.0)
OVERRIDE_DECISIONS = ("override_cv_condition_ii",)  # rule (ii): only this decision of the entrant may override the CV condition of (ii)
RULE_NAME = "safety-set rule"  # the rule the decision file must name
# versioned decision file, its checksum pinned here
DECISION_DIR = Path("decisions")
DECISION_SHA = {"override_cv_condition_ii": "80040bd638758b171b3b746fcc32051408d00358fa362f27e13f1c6a7ee21168"}
RULE_JSON = OUT / "c123_rule.json"  # result file of the rule without override; with --override-ii <id>: c123_rule_<id>.json (both stay)

_C120 = None


def c120():
    """eda/c120_s036_safe_stage2.py as a module (loaded once; unchanged)."""
    global _C120
    if _C120 is None:
        spec = importlib.util.spec_from_file_location("c123_c120", COMP / "eda" / "c120_s036_safe_stage2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _C120 = mod
    return _C120


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def load_context() -> dict:
    C = c120()
    try:
        return C.context()
    except C.Missing as e:
        abort(str(e), 3)


# ------------------------------------------------------------------------------------------------ recipe (transductive alignment)

def fit_apply_trans(ctx: dict, fit_idx: np.ndarray, ev_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights by log loss from fit_idx (a2.fit_mix), blend with β on ev_idx (logs without mask as in c120); margin alignment 0.75 of
    the rows ev_idx as a set, target = frequencies of fit_idx from 3 rows (= c107.cross_fit with last step "trans"). -> (P, Q)."""
    C = c120()
    y, seen, K, spec, L = ctx["y"], ctx["seen"], ctx["K"], ctx["spec"], ctx["L"]
    th = np.asarray(C.A2.fit_mix([lg[fit_idx] for lg in L], None, y[fit_idx]), dtype=float).copy()
    st = {"theta": [float(v) for v in th], "arith_beta": float(spec["beta"])}
    PE = C.MS.mix_channels(st, [lg[ev_idx] for lg in L], None, seen)
    QE = C.C107.apply_last(PE, C.C21.fit_weights(PE, C.C107.target_of(y, fit_idx, K)), seen)
    return PE, QE


def standard_forms(ctx: dict) -> dict:
    """Standard cross-fit per fold (weights from the folds ≠ k): F (aligned per fold), oL (before), M (one set over all rows)."""
    C = c120()
    n, K, folds = ctx["n"], ctx["K"], ctx["folds"]
    P, Q = np.zeros((n, K)), np.zeros((n, K))
    for k in range(5):
        F, E = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        P[E], Q[E] = fit_apply_trans(ctx, F, E)
    return {"fold": Q, "pre": P, "set": C.C107.set_align(P, ctx["y"], ctx["seen"])}


def nested_forms(ctx: dict, k: int) -> dict:
    """Base forms of the training rows T (folds ≠ k) without any label of fold k; set form over T with the target from T."""
    C = c120()
    y, seen, n, K, folds = ctx["y"], ctx["seen"], ctx["n"], ctx["K"], ctx["folds"]
    T = np.flatnonzero(folds != k)
    P, Q = np.zeros((n, K)), np.zeros((n, K))
    for j in range(5):
        if j == k:
            continue
        F = np.flatnonzero((folds != j) & (folds != k))
        E = np.flatnonzero(folds == j)
        P[E], Q[E] = fit_apply_trans(ctx, F, E)
    S = np.zeros((n, K))
    S[T] = C.C107.apply_last(P[T], C.C21.fit_weights(P[T], C.C107.target_of(y, T, K)), seen)
    return {"fold": Q, "pre": P, "set": S}


def block(ctx: dict, rows: np.ndarray, FM: np.ndarray, PM: np.ndarray, variant: str, exclude=()) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features of the rows `rows` of ONE fold (knowledge from the other folds except `exclude`, as c120.block): variant a "fallback",
    variant b "fallback_i1" (fallback + the 15 interactions I1 at the end)."""
    kn = S2.knowledge_other_folds(ctx["key"], ctx["topic_s"], ctx["y"], ctx["folds"], rows, ctx["K"], exclude)
    return S2.pair_features(ctx["static"], rows, FM, PM, kn, VARIANT_SET[variant])


def block_perm(R: int, Kc: int, seed: int) -> np.ndarray:
    """Placebo permutation as in the primary line: reorder the rows of a block (default_rng(seed + R)), the candidate slots stay."""
    rng = np.random.default_rng(seed + R)
    return np.repeat(rng.permutation(R) * Kc, Kc) + np.tile(np.arange(Kc), R)


def run_nested(ctx: dict, std: dict, nested: dict, form: str, variant: str = "a", mode: str = "real", seed: int | None = None) -> dict:
    """Per outer fold k: fit on T (nested base forms of the form `form`), gate over the four inner folds (with guard), application on k
    with the NONE guard. variant a/b; mode (only b): real, zero (I1 columns 0 -> a bit-identical), placebo (seed). Plus the row values of
    the null model (weight 0 -> base). -> row values, orders, guard marks, gate."""
    if variant not in ("a", "b") or mode not in ("real", "zero", "placebo") or (mode != "real" and variant != "b") \
            or (mode == "placebo") != (seed is not None):
        raise ValueError(f"variant {variant!r} / mode {mode!r} / seed {seed!r}")
    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND

    def prep(X: np.ndarray, names: list[str]) -> np.ndarray:
        if mode == "real":
            return X
        X = X.copy()
        cols = [names.index(nm) for nm in S2.I1_FEATS]
        X[:, cols] = 0.0 if mode == "zero" else S2.i1_interactions(X, names, block_perm(X.shape[0] // Kc, Kc, seed))
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
            X, cand, names_j = block(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], variant, exclude=(k,))
            XT.append(prep(X, names_j))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = block(ctx, E, std[form][E], std["pre"][E], variant)
        XE = prep(XE, names)
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


def eval_rows_in_table_order(ctx: dict, X_eval: dict) -> np.ndarray:
    return c120().eval_rows_in_table_order(ctx, X_eval)


def same_rows(a: dict, b: dict, keys=("forced", "gated", "top", "cand", "protected")) -> bool:
    return bool(all(np.array_equal(a[k], b[k]) for k in keys))


def boot(ctx: dict, d: np.ndarray, draws: np.ndarray) -> dict:
    C = c120()
    cb = C.C107.cluster_boot(np.asarray(d, float), ctx["q_code"], draws)
    fu = [int(np.rint(6 * d[ctx["folds"] == k].sum())) for k in range(5)]
    return {"delta": cb["mean"], "se_q": cb["se_q"], "z": cb["z_q"], "units": int(np.rint(6 * np.sum(d))), "fold_units": fu,
            "folds_pos": int(sum(u > 0 for u in fu))}


def class_mass_l1(Q: np.ndarray, y: np.ndarray, K: int, min_rows: int = ALIGN_MIN) -> dict:
    """L1 of the mean class mass against the target (frequencies of all rows with at least min_rows), aggregates only."""
    counts = np.bincount(y, minlength=K)
    target = np.where(counts >= min_rows, counts, 0) / len(y)
    used = target > 0
    m = np.asarray(Q, np.float64).mean(0)
    return {"l1": float(np.abs(m - target)[used].sum()), "classes": int(used.sum()), "none": float(m[0]), "none_target": float(target[0])}


def half_fm(d: dict) -> float:
    return 0.5 * (float(d["fold"]) + float(d["set"]))


def prepare(ctx: dict) -> tuple[dict, dict]:
    std = standard_forms(ctx)
    nested = {k: nested_forms(ctx, k) for k in range(5)}
    return std, nested


# ------------------------------------------------------------------------------------------------ nested reproduction

def cmd_oof(a) -> int:
    t0 = time.time()
    C = c120()
    ctx = load_context()
    y, seen, K = ctx["y"], ctx["seen"], ctx["K"]
    if not C.OOF_JSON.is_file() or not (C.OUT / "c120_oof_rows.npz").is_file():
        abort(f"{C.rel(C.OOF_JSON)} missing — run python -I eda/c120_s036_safe_stage2.py oof first (build of the strictly inductive line)", 3)
    o036 = json.loads(C.OOF_JSON.read_text())
    inputs = C.inputs_sha256(ctx)
    if o036.get("inputs") != inputs or int(o036.get("deploy_units", -1)) != INDUCTIVE_UNITS:
        abort(f"{C.rel(C.OOF_JSON)} was computed with other inputs or does not hit {INDUCTIVE_UNITS}/9000 — rerun c120 oof")
    inductive = np.load(C.OUT / "c120_oof_rows.npz")["gated_prot"]
    if int(C.C107.units(inductive).sum()) != INDUCTIVE_UNITS:
        abort("row values of the strictly inductive line (c120_oof_rows.npz) do not hit 7,410/9,000")
    std, nested = prepare(ctx)
    lib = C.C107.cross_fit(y, ctx["folds"], seen, ctx["mats"], ctx["spec"], last="trans")
    if not (np.array_equal(lib["Q"], std["fold"]) and np.array_equal(lib["P"], std["pre"])):
        abort("standard cross-fit differs from c107.cross_fit (last step \"trans\")")
    if not np.array_equal(C.standard_forms(ctx)["pre"], std["pre"]):
        abort("blend before the last step != that of the strictly inductive line (c120)")
    rec_rows = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS}
    draws_q = C.C107.question_draws(ctx["q_code"])
    seeds = PLACEBO_SEEDS[: a.placebos]
    res = {"kind": "safety line with set alignment: recipe of the strictly inductive line with margin alignment as one set (transductive), "
                   "second stage fallback + NONE guard fitted anew; variant b with the interactions I1",
           "stack": C.rel(C.STACK), "stack_sha256": C.sha256_file(C.STACK), "K": S2.K_CAND, "lambda": S2.LAMBDA, "feature_set": FEATURE_SET,
           "protect_none": True, "align": ALIGN, "align_min": ALIGN_MIN, "inductive_units": INDUCTIVE_UNITS, "inductive_map": INDUCTIVE_UNITS / 9000,
           "base_map": {f: float(rec_rows[f].mean()) for f in FORMS}, "base_units": {f: int(C.C107.units(rec_rows[f]).sum()) for f in FORMS},
           "class_mass": {"F_trans": class_mass_l1(std["fold"], y, K), "M_set": class_mass_l1(std["set"], y, K),
                          "oL": class_mass_l1(std["pre"], y, K)},
           "sets": {"a": {}, "b": {}}, "placebo": {"seeds": list(seeds), "per_seed": []}, "inputs": inputs}
    rows_out, ra = {}, {}
    null_a, null_b = {}, {}
    pl = {s: {} for s in seeds}
    for form in FORMS:
        r = run_nested(ctx, std, nested, form, "a")
        if not np.array_equal(r["base"], rec_rows[form]):
            abort(f"base of the candidate order [a|{form}] != recipe")
        null_a[form] = bool(np.array_equal(r["null"], rec_rows[form]))
        ra[form] = r
        b = run_nested(ctx, std, nested, form, "b")
        bz = run_nested(ctx, std, nested, form, "b", "zero")
        null_b[form] = same_rows(bz, r)
        for nm, rr in (("a", r), ("b", b)):
            ent = {"map_forced": float(rr["forced"].mean()), "map_gated": float(rr["gated"].mean()),
                   "units_gated": int(C.C107.units(rr["gated"]).sum()), "abstain_folds": int(sum(g["abstain"] for g in rr["gates"].values())),
                   "gates": {str(k): v for k, v in rr["gates"].items()}, "n_features": len(rr["features"]),
                   "changed_top_share": float((rr["top"][:, 0] != rr["cand"][:, 0]).mean()),
                   "changed_top3_share": float((rr["top"][:, :3] != rr["cand"][:, :3]).any(1).mean()),
                   "rows_protected": int(rr["protected"].sum()), "vs_base": boot(ctx, rr["gated"] - rec_rows[form], draws_q)}
            if nm == "a":
                ent["vs_inductive"] = boot(ctx, rr["gated"] - inductive, draws_q)
            else:
                ent["vs_a"] = boot(ctx, rr["forced"] - r["forced"], draws_q)
                ent["vs_a_gated"] = boot(ctx, rr["gated"] - r["gated"], draws_q)
                ent["null_i1_zero_equals_a"] = null_b[form]
            if form == "set":
                ent["eval_features_sha256"] = C.sha256_array(eval_rows_in_table_order(ctx, rr["X_eval"]))
                ent["features"] = rr["features"]
                ent["novel_task_share_top5"] = float(eval_rows_in_table_order(ctx, rr["X_eval"])[:, rr["features"].index("novel_task")].mean())
            res["sets"][nm][form] = ent
            for key in ("forced", "gated", "top", "cand", "protected"):
                rows_out[f"{nm}_{form}_{key}"] = rr[key]
        res["sets"]["a"][form]["null_zero_model_equals_base"] = null_a[form]
        for sd in seeds:
            p = run_nested(ctx, std, nested, form, "b", "placebo", sd)
            pl[sd][form] = float((p["forced"] - r["forced"]).mean())
        ea, eb = res["sets"]["a"][form], res["sets"]["b"][form]
        print(f"  [{FORM_SHORT[form]:2s}] base {res['base_map'][form]:.6f} | a {ea['map_gated']:.6f} ({ea['units_gated']}/9000; second stage "
              f"{ea['vs_base']['delta']:+.5f} ± {ea['vs_base']['se_q']:.5f}; vs. strictly inductive {ea['vs_inductive']['delta']:+.5f} ± {ea['vs_inductive']['se_q']:.5f}; "
              f"abstention {ea['abstain_folds']}/5; new top code {ea['changed_top_share']:.4f}; null point {'bit-identical' if null_a[form] else 'VIOLATED'}) | "
              f"b − a {eb['vs_a']['delta']:+.5f} ± {eb['vs_a']['se_q']:.5f} {eb['vs_a']['fold_units']} (null point b {'bit-identical' if null_b[form] else 'VIOLATED'}; "
              f"placebo mean {np.mean([pl[s][form] for s in seeds]):+.5f})", flush=True)
    res["null_point"] = {"a_zero_model": {"ok": bool(all(null_a.values())), "forms": null_a},
                         "b_i1_zero": {"ok": bool(all(null_b.values())), "forms": null_b}}
    da = {f: res["sets"]["b"][f]["vs_a"]["delta"] for f in FORMS}
    res["b_vs_a"] = {"F": da["fold"], "oL": da["pre"], "M": da["set"], "A": half_fm(da),
                     "units": {FORM_SHORT[f]: res["sets"]["b"][f]["vs_a"]["units"] for f in FORMS}}
    per = [{"seed": s, "F": pl[s]["fold"], "oL": pl[s]["pre"], "M": pl[s]["set"], "A": half_fm(pl[s])} for s in seeds]
    pa = np.array([p["A"] for p in per]) if per else np.zeros(0)
    res["placebo"].update({"per_seed": per, "n": len(per), "A_mean": float(pa.mean()) if len(pa) else None,
                           "A_sd": float(pa.std(ddof=1)) if len(pa) > 1 else None, "A_max": float(pa.max()) if len(pa) else None,
                           "share_ge_real": float((pa >= res["b_vs_a"]["A"]).mean()) if len(pa) else None})
    am = res["sets"]["a"]["set"]
    res["cv_rule"] = {"a_M_gated": am["map_gated"], "a_M_units": am["units_gated"], "inductive_F": INDUCTIVE_UNITS / 9000, "margin": CV_MARGIN,
                      "threshold": INDUCTIVE_UNITS / 9000 - CV_MARGIN, "ok": bool(am["map_gated"] >= INDUCTIVE_UNITS / 9000 - CV_MARGIN - 1e-12)}
    OUT.mkdir(parents=True, exist_ok=True)
    OOF_JSON.write_text(c120().to_json(res))
    np.savez_compressed(OOF_ROWS, **rows_out, inductive=inductive, **{f"base_{f}": rec_rows[f] for f in FORMS})
    cm = res["class_mass"]
    print(f"safety-set a (M, gate + guard) {am['map_gated']:.6f} vs. strictly inductive (F inductive) {INDUCTIVE_UNITS / 9000:.6f} − {CV_MARGIN} -> "
          f"{'yes' if res['cv_rule']['ok'] else 'NO'}; OOF class masses after the alignment L1: M {cm['M_set']['l1']:.4f}, F {cm['F_trans']['l1']:.4f} "
          f"(oL {cm['oL']['l1']:.4f}); b − a ½(F+M) {res['b_vs_a']['A']:+.5f}, placebo mean {res['placebo']['A_mean']:+.5f} (n {len(per)}); null points a "
          f"{'bit-identical' if res['null_point']['a_zero_model']['ok'] else 'VIOLATED'}, b {'bit-identical' if res['null_point']['b_i1_zero']['ok'] else 'VIOLATED'}; "
          f"written: {C.rel(OOF_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0 if res["null_point"]["a_zero_model"]["ok"] and res["null_point"]["b_i1_zero"]["ok"] else 4


# ------------------------------------------------------------------------------------------------ new meta-splits

def split_folds(y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    f = np.zeros(len(y), np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            f[te] = k
    return f


def cmd_splits(a) -> int:
    t0 = time.time()
    if a.n < MIN_SPLITS:
        abort(f"--n {a.n}: the pre-registered rule requires at least {MIN_SPLITS} meta-splits", 2)
    C = c120()
    ctx0 = load_context()
    y = ctx0["y"]
    res = {"kind": "safety-set rule (iii): b − a over new meta-splits (channel OOFs fixed, meta-folds new)", "stack": C.rel(C.STACK),
           "stack_sha256": C.sha256_file(C.STACK), "splits": [], "inputs": C.inputs_sha256(ctx0)}
    for s in range(a.n):
        ctx = dict(ctx0)
        ctx["folds"] = split_folds(y, SPLIT_SEED0 + s)
        std, nested = prepare(ctx)
        rec = {"seed": SPLIT_SEED0 + s, "placebo_seed": SPLIT_PLACEBO_SEED0 + s, "a_map": {}, "a_minus_base": {}, "b_minus_a": {}, "placebo_minus_a": {}}
        for form in FORMS:
            base_rows = C.C107.rows_map(y, std[form], ctx["seen"])
            ra = run_nested(ctx, std, nested, form, "a")
            rb = run_nested(ctx, std, nested, form, "b")
            rp = run_nested(ctx, std, nested, form, "b", "placebo", SPLIT_PLACEBO_SEED0 + s)
            rec["a_map"][form] = float(ra["gated"].mean())
            rec["a_minus_base"][form] = float((ra["gated"] - base_rows).mean())
            rec["b_minus_a"][form] = float((rb["forced"] - ra["forced"]).mean())
            rec["placebo_minus_a"][form] = float((rp["forced"] - ra["forced"]).mean())
        res["splits"].append(rec)
        print(f"  split {s:2d} (seed {rec['seed']}): a F {rec['a_map']['fold']:.4f} M {rec['a_map']['set']:.4f} | b − a F {rec['b_minus_a']['fold']:+.4f} "
              f"M {rec['b_minus_a']['set']:+.4f} | placebo − a F {rec['placebo_minus_a']['fold']:+.4f} M {rec['placebo_minus_a']['set']:+.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    summ = {}
    for key in ("b_minus_a", "placebo_minus_a", "a_minus_base"):
        F = np.array([r[key]["fold"] for r in res["splits"]])
        M = np.array([r[key]["set"] for r in res["splits"]])
        A = 0.5 * (F + M)
        summ[key] = {"F_mean": float(F.mean()), "M_mean": float(M.mean()), "A_mean": float(A.mean()), "A_se": float(A.std(ddof=1) / np.sqrt(len(A))),
                     "share_A_pos": float((A > 0).mean()), "share_F_pos": float((F > 0).mean()), "share_M_pos": float((M > 0).mean())}
    aF = np.array([r["a_map"]["fold"] for r in res["splits"]])
    aM = np.array([r["a_map"]["set"] for r in res["splits"]])
    summ["a_map"] = {"F_mean": float(aF.mean()), "F_sd": float(aF.std(ddof=1)), "M_mean": float(aM.mean()), "M_sd": float(aM.std(ddof=1))}
    res["summary"] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(C.to_json(res))
    sb, sp = summ["b_minus_a"], summ["placebo_minus_a"]
    print(f"b − a over {a.n} meta-splits: ½(F+M) {sb['A_mean']:+.5f} ± {sb['A_se']:.5f} ({sb['share_A_pos']:.0%} > 0); placebo − a {sp['A_mean']:+.5f}; "
          f"a over meta-splits F {summ['a_map']['F_mean']:.4f} M {summ['a_map']['M_mean']:.4f}; second stage (a − base) ½(F+M) "
          f"{summ['a_minus_base']['A_mean']:+.5f}; written: {C.rel(SPLITS_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0


def cmd_splits_inductive(a) -> int:
    """Additional evaluation (not part of the pre-registered rule): the strictly inductive line in its deployment form (F inductive, gate
    + guard; c120.run_nested) on THE SAME new meta-splits as `splits`, paired with a in M (gate + guard) from c123_splits.json.
    -> safety_set/c123_splits_inductive.json."""
    t0 = time.time()
    C = c120()
    if not SPLITS_JSON.is_file():
        abort(f"{C.rel(SPLITS_JSON)} missing — run c123 splits first", 3)
    sp = json.loads(SPLITS_JSON.read_text())
    ctx0 = load_context()
    if sp.get("inputs") != C.inputs_sha256(ctx0):
        abort(f"{C.rel(SPLITS_JSON)} was computed with other inputs", 4)
    y = ctx0["y"]
    rows = []
    for rec in sp["splits"]:
        ctx = dict(ctx0)
        ctx["folds"] = split_folds(y, int(rec["seed"]))
        r = C.run_nested(ctx, C.standard_forms(ctx))
        inductive = float(r["gated_prot"].mean())
        rows.append({"seed": rec["seed"], "inductive_F": inductive, "a_M": rec["a_map"]["set"], "a_F": rec["a_map"]["fold"],
                     "a_M_minus_inductive_F": rec["a_map"]["set"] - inductive,
                     "inductive_abstain_folds": int(sum(g["abstain"] for g in r["gates"].values()))})
        print(f"  seed {rec['seed']}: strictly inductive F {inductive:.4f} | a M {rec['a_map']['set']:.4f} -> {rec['a_map']['set'] - inductive:+.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    d = np.array([r["a_M_minus_inductive_F"] for r in rows])
    s36 = np.array([r["inductive_F"] for r in rows])
    res = {"kind": "additional evaluation of rule (ii): strictly inductive line (F inductive, gate + guard) against safety-set a (M, gate + guard) "
                   "on the same meta-splits",
           "splits": rows, "summary": {"inductive_F_mean": float(s36.mean()), "inductive_F_sd": float(s36.std(ddof=1)),
                                        "a_M_mean": float(np.mean([r["a_M"] for r in rows])), "diff_mean": float(d.mean()),
                                        "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))), "share_diff_ge_minus_margin": float((d >= -CV_MARGIN).mean()),
                                        "share_diff_pos": float((d > 0).mean())}, "inputs": sp["inputs"]}
    (OUT / "c123_splits_inductive.json").write_text(C.to_json(res))
    s = res["summary"]
    print(f"strictly inductive F over {len(rows)} meta-splits {s['inductive_F_mean']:.4f} (sd {s['inductive_F_sd']:.4f}); a M {s['a_M_mean']:.4f}; a M − strictly "
          f"inductive F {s['diff_mean']:+.5f} ± {s['diff_se']:.5f} (share ≥ −{CV_MARGIN}: {s['share_diff_ge_minus_margin']:.0%}, > 0: {s['share_diff_pos']:.0%}) "
          f"({time.time() - t0:.1f} s)")
    return 0


# ------------------------------------------------------------------------------------------------ pre-registered safety-set rule

def _file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Path relative to the repository root for messages (without module dependency; COMP may be redirected in tests)."""
    try:
        return str(path.relative_to(COMP))
    except ValueError:
        return str(path)


def override_record(decision: str) -> dict:
    """Override of the CV condition of rule (ii), bound to a decision of the entrant: only override_cv_condition_ii; the versioned
    decision file decisions/<id>.json must exist, hit the checksum pinned in DECISION_SHA, carry its id, be "active" and name this rule
    (RULE_NAME); otherwise abort. -> details for the result file and the deployment file."""
    if decision not in OVERRIDE_DECISIONS:
        abort(f"override of (ii) only with {', '.join(OVERRIDE_DECISIONS)}, not {decision!r}", 2)
    rf = COMP / DECISION_DIR / f"{decision}.json"
    if not rf.is_file():
        abort(f"{_rel(rf)} missing — override cannot be checked", 3)
    if _file_sha256(rf) != DECISION_SHA.get(decision):
        abort(f"{_rel(rf)} differs from the pinned checksum — no override", 5)
    rec = json.loads(rf.read_text(encoding="utf-8"))
    if rec.get("id") != decision:
        abort(f"{_rel(rf)} does not belong to {decision} — no override", 5)
    status, title, where = rec.get("status"), rec.get("title"), _rel(rf)
    if status != "active":
        abort(f"{where}: {decision} is not active (status {status}) — no override", 5)
    if rec.get("rule") != RULE_NAME:
        abort(f"{where}: {decision} does not name the {RULE_NAME} — no override", 5)
    return {"decision": decision, "status": status, "title": title,
            "scope": "only the CV condition of rule (ii); the test-mass condition (L1 ≤ 0.035), (i), (iii), null points and guards stay in force"}


def rule_result(oof: dict, splits: dict | None, test_report: dict | None, override: dict | None = None) -> dict:
    """Pre-registered safety-set rule. (i) OOF part here (file part in the build script); (ii) CV and — with a test report — class masses;
    (iii) b against a. override (only override_record): the CV condition of (ii) counts as overridden; without override the return value is
    unchanged."""
    ok_i = bool(oof["null_point"]["a_zero_model"]["ok"] and oof["null_point"]["b_i1_zero"]["ok"])
    cv = oof["cv_rule"]
    l1 = None if test_report is None else float((test_report.get("class_mass") or test_report).get("l1_after", float("nan")))
    ok_l1 = None if l1 is None else bool(l1 <= L1_MAX)
    bva, pl = oof["b_vs_a"], oof["placebo"]
    ok_b_ii = bool(bva["A"] > 0 and int(pl.get("n") or 0) >= MIN_PLACEBOS and bva["A"] > float(pl["A_mean"]))
    sp = None if splits is None else splits["summary"]["b_minus_a"]
    ok_b_iii = None if sp is None else bool(len(splits["splits"]) >= MIN_SPLITS and sp["A_mean"] > 0)
    variant = "b" if (ok_b_ii and ok_b_iii) else "a"
    overridden = bool(override is not None and not cv["ok"])
    ok_a = bool(ok_i and (cv["ok"] or overridden) and (ok_l1 is not False))
    verdict = ("deploy " + variant) if ok_a and ok_l1 else ("no deployment" if not ok_a else f"deploy {variant} (OOF part; test-side L1 open)")
    if overridden and ok_a:
        verdict += f" — (ii) missed, overridden by {override['decision']}"
    out = {"rule": RULE_NAME, "i_oof": ok_i,
           "i_file": "build script tools/builds/safety_final.sh (weight 0 -> file without second stage; chain of the strictly inductive line)",
           "ii_cv": cv, "ii_l1": {"l1_after_test": l1, "max": L1_MAX, "ok": ok_l1},
           "iii_b": {"half_F_plus_M": bva["A"], "placebo_mean": pl["A_mean"], "placebo_n": pl.get("n"), "ok_vs_placebo": ok_b_ii,
                     "splits_mean": None if sp is None else sp["A_mean"], "splits_se": None if sp is None else sp["A_se"],
                     "splits_n": None if splits is None else len(splits["splits"]), "ok_splits": ok_b_iii, "ok": bool(ok_b_ii and ok_b_iii)},
           "variant": variant, "verdict": verdict}
    if override is not None:  # only with an override: additional fields (without them the return value stays byte-identical)
        out["ii_cv"] = {**cv, "override": override["decision"], "ok_effective": bool(cv["ok"] or overridden)}
        out["override_ii"] = {**override, "applied": overridden}
    return out


def load_results(ctx: dict | None, need_splits: bool = True) -> tuple[dict, dict | None]:
    C = c120()
    if not OOF_JSON.is_file():
        abort(f"{C.rel(OOF_JSON)} missing — run python -I eda/c123_safety_set.py oof first", 3)
    oof = json.loads(OOF_JSON.read_text())
    sp = json.loads(SPLITS_JSON.read_text()) if SPLITS_JSON.is_file() else None
    if need_splits and sp is None:
        abort(f"{C.rel(SPLITS_JSON)} missing — run python -I eda/c123_safety_set.py splits first", 3)
    if ctx is not None:
        inputs, sha = C.inputs_sha256(ctx), C.sha256_file(C.STACK)
        for nm, o in ((OOF_JSON, oof), (SPLITS_JSON, sp)):
            if o is not None and (o.get("inputs") != inputs or o.get("stack_sha256") != sha):
                abort(f"{C.rel(nm)} was computed with other inputs — recompute it")
    return oof, sp


def cmd_rule(a) -> int:
    oof, sp = load_results(None)
    rep = json.loads(Path(a.test_report).read_text()) if a.test_report else None
    ovr = override_record(a.override_ii) if a.override_ii else None
    r = rule_result(oof, sp, rep, ovr)
    OUT.mkdir(parents=True, exist_ok=True)
    rule_json = RULE_JSON.with_name(f"c123_rule_{ovr['decision']}.json") if ovr else RULE_JSON
    rule_json.write_text(c120().to_json({**r, "test_report": a.test_report, "override_ii_flag": a.override_ii}))
    cv, l1, b = r["ii_cv"], r["ii_l1"], r["iii_b"]
    cv_txt = "yes" if cv["ok"] else ("NO — (ii) missed — overridden by " + ovr["decision"] if (ovr and r["override_ii"]["applied"]) else "NO")
    print(f"safety-set rule: (i) OOF null points {'bit-identical' if r['i_oof'] else 'VIOLATED'} | (ii) CV a (M, gate + guard) {cv['a_M_gated']:.6f} ≥ strictly "
          f"inductive {cv['inductive_F']:.6f} − {cv['margin']} = {cv['threshold']:.6f} -> {cv_txt}; test class masses L1 "
          + ("open (no test report)" if l1["ok"] is None else f"{l1['l1_after_test']:.4f} ≤ {L1_MAX} -> {'yes' if l1['ok'] else 'NO'}")
          + f" | (iii) b − a ½(F+M) {b['half_F_plus_M']:+.5f} vs. placebo mean {b['placebo_mean']:+.5f} -> {'yes' if b['ok_vs_placebo'] else 'no'}; over "
          f"{b['splits_n']} meta-splits {b['splits_mean']:+.5f} ± {b['splits_se']:.5f} -> {'yes' if b['ok_splits'] else 'no'} => variant {r['variant']}; "
          f"{r['verdict']}")
    return 0 if r["verdict"].startswith("deploy") else 5


# ------------------------------------------------------------------------------------------------ deployment file

def build_stage2(ctx: dict, oof: dict, rule: dict, variant: str) -> dict:
    C = c120()
    fs = VARIANT_SET[variant]
    y, Kc, folds = ctx["y"], S2.K_CAND, ctx["folds"]
    std = standard_forms(ctx)
    n = ctx["n"]
    X, cand = None, np.zeros((n, Kc), np.int64)
    names = None
    for k in range(5):
        E = np.flatnonzero(folds == k)
        Xk, ck, names = block(ctx, E, std["set"][E], std["pre"][E], variant)
        if X is None:
            X = np.zeros((n * Kc, Xk.shape[1]))
        X[(E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)] = Xk
        cand[E] = ck
    got = C.sha256_array(X)
    o = oof["sets"][variant]
    if got != o["set"].get("eval_features_sha256"):
        abort(f"null point (e) violated: features of the deployment form ({got[:16]}…) != evaluated rows of the OOF reproduction "
              f"({str(o['set'].get('eval_features_sha256'))[:16]}…) — rerun c123 oof")
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
    s036 = ART / "stage2_s036_safe.json"
    return {"kind": "second stage of the safety line with margin alignment as ONE set over the test rows: conditional logit over the first K "
                    "candidates (src/stage2.py, feature set without notes) + NONE guard (eda/c123_safety_set.py)",
            "stack": C.rel(C.STACK), "stack_sha256": C.sha256_file(C.STACK), "channels": list(ctx["stack"]["channels"]),
            "arith_beta": float(ctx["stack"]["arith_beta"]), "align_mode": "set", "align": ALIGN, "align_min": ALIGN_MIN,
            "K": Kc, "lambda": S2.LAMBDA, "null": False, "feature_set": fs, "protect_none": True,
            "draws_channel": C.M3, "draw_sources": ctx["draw_specs"], "draw_rank_source": meta["rank"]["source"], "test_draws": list(C.TEST_DRAWS),
            "model": {"features": names, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
                      "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
                      "oof_changed_top_share": float(o["set"]["changed_top_share"]), "oof_changed_top3_share": float(o["set"]["changed_top3_share"]),
                      "oof_vs_base": {f: o[f]["vs_base"]["delta"] for f in FORMS},
                      **({"oof_vs_inductive": {f: o[f]["vs_inductive"]["delta"] for f in FORMS}} if variant == "a"
                         else {"oof_vs_a": {f: o[f]["vs_a"]["delta"] for f in FORMS}}),
                      "features_sha256": got, "insample": ins},
            "novel_task_share_top5": float(o["set"]["novel_task_share_top5"]),
            "guard_rule": {"changed_top_factor": list(CHANGED_TOP_FACTOR), "class_mass_l1_max": L1_MAX,
                           "text": "(c) share of test rows with a new top code (after the NONE guard) between ½ and 2 times the OOF share, otherwise "
                                   "abort without a file; (b) share \"novel for the task\" among the first five reported only; no note guard (no note "
                                   "features); rule (ii): class masses of the test rows after the alignment at most L1 0.035 from the target (report "
                                   "class_mass; build script)"},
            "inputs_sha256": oof["inputs"],
            "oof_reproduction": {"file": C.rel(OOF_JSON), "units_gated": {f: o[f]["units_gated"] for f in FORMS}, "inductive_units": INDUCTIVE_UNITS},
            "rule_oof": rule,
            "based_on": {"stage2_s036_safe.json": C.sha256_file(s036) if s036.is_file() else None, "build": "tools/builds/safety_final.sh"},
            "order": "blend -> margin alignment (one set over the test rows) -> second stage over the first K -> NONE guard -> REL-01"}


def cmd_fit(a) -> int:
    C = c120()
    target = Path(a.out) if a.out else (PROBE_STAGE2 if a.probe else STAGE2_FILE)
    if target.resolve() in {(ART / f).resolve() for f in ("stage2_s032.json", "stage2_s034.json", "stage2_s035.json", "stage2_s036_safe.json",
                                                         "stage2_primary_i1.json")}:
        abort(f"fit never writes an existing deployment file — separate output (default {C.rel(STAGE2_FILE)})", 2)
    if a.probe and not target.resolve().is_relative_to(OUT.resolve()):
        abort(f"--probe only writes under {C.rel(OUT)} (a probe is never a deployment file)", 2)
    if a.probe and a.override_ii:
        abort("--probe and --override-ii are mutually exclusive (an override only applies to the deployment)", 2)
    ovr = override_record(a.override_ii) if a.override_ii else None
    ctx = load_context()
    oof, sp = load_results(ctx)
    rule = rule_result(oof, sp, None, ovr)
    if not rule["verdict"].startswith("deploy") and not a.probe:
        abort(f"safety-set rule not met ({rule['verdict']}: (i) {rule['i_oof']}, (ii) CV {rule['ii_cv']['ok']}) — no deployment, no file", 5)
    variant = a.variant or ("a" if a.probe else rule["variant"])
    if ovr is not None and variant != rule["variant"]:
        abort(f"with an override only the variant of the rule ({rule['variant']}), not {variant}", 2)
    out = build_stage2(ctx, oof, rule, variant)
    out["variant"] = variant
    if ovr is not None:  # recorded explicitly in the deployment file
        out["override_ii"] = ovr["decision"]
        print(f"  (ii) missed — overridden by {ovr['decision']} (decision file, status {ovr['status']}): {ovr['scope']}")
    if a.probe:  # PROBE: only for judging the test side (aggregates), never submitted; the result of the rule is in the file
        out = {**out, "probe": True, "probe_note": "PROBE: not eligible for deployment under the pre-registered rule as long as rule_oof.verdict "
                                                     "does not start with \"deploy\"; only for judging the test side (aggregates), never to be submitted"}
    text = C.to_json(out)
    m = out["model"]
    print(f"  [safety-set {variant}: {VARIANT_SET[variant]}] {len(m['features'])} features | gate (guarded) inner {m['gate']['inner']:.5f} vs. base "
          f"{m['gate']['inner_base']:.5f} -> {'ABSTAIN (weight 0)' if m['abstain'] else 'apply'} | in-sample {m['insample']['map_base']:.5f} → "
          f"{m['insample']['map_reranked_protected']:.5f} (new top code {m['insample']['changed_top_share']:.4f}, guarded {m['insample']['rows_protected']}; "
          f"OOF {m['oof_changed_top_share']:.4f})")
    if a.verify:
        if not target.is_file():
            abort(f"{C.rel(target)} missing — build it with --save first", 3)
        if target.read_text() != text:
            abort(f"{C.rel(target)} is not the recomputation")
        print(f"   VERIFIED: {C.rel(target)} = recomputation (SHA-256 {C.sha256_file(target)[:16]}…)")
    if a.save:
        if target.is_file() and target.read_text() != text:
            abort(f"{C.rel(target)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"   saved: {C.rel(target)} (SHA-256 {C.sha256_file(target)})" + ("  [PROBE — not eligible for deployment]" if a.probe else ""))
    return 0


def null_stage2(source: Path = STAGE2_FILE) -> dict:
    """Null model (weight 0, standardisation 0/1) for the deployment file: the features are computed, the score is 0 everywhere ->
    base order, the guard changes nothing (copy null point). Guard (c) is not enforced."""
    ref = json.loads(Path(source).read_text())
    nf = len(ref["model"]["features"])
    model = {"features": ref["model"]["features"], "w": [0.0] * nf, "mu": [0.0] * nf, "sd": [1.0] * nf, "abstain": True, "oof_changed_top_share": 0.0}
    keep = ("kind", "stack", "stack_sha256", "channels", "arith_beta", "align_mode", "align", "align_min", "K", "lambda", "feature_set", "protect_none",
            "draws_channel", "draw_sources", "draw_rank_source", "test_draws", "novel_task_share_top5", "guard_rule")
    return {**{k: ref[k] for k in keep}, "null": True, "model": model, "null_from": c120().rel(Path(source))}


def cmd_null(a) -> int:
    src = Path(a.source)
    if not src.is_file():
        abort(f"{c120().rel(src)} missing — run c123 fit --save first", 3)
    Path(a.out).write_text(c120().to_json(null_stage2(src)))
    print(f"null model for {c120().rel(src)} -> {a.out}")
    return 0


# ------------------------------------------------------------------------------------------------ class masses of the test side (aggregates)

def cmd_test_mass(a) -> int:
    """Rule (ii), second part — aggregates only: test blend AFTER the alignment (before the second stage and REL-01) through the unchanged
    path tools/make_submission.py (own outputs in a temporary folder, the test matrix deleted afterwards); L1 of the mean class mass
    against the target (training shares of the classes with at least --align-min rows). --align-mode inductive (strictly inductive line;
    control value 0.087 from an independent check) or set (safety line with set alignment). The test-side sources follow "--" (without
    alignment, REL-01 and second-stage switches)."""
    import subprocess
    import tempfile

    import pandas as pd

    src = list(a.src)
    if src[:1] == ["--"]:
        src = src[1:]
    bad = [x for x in src if x in ("--align", "--align-inductive", "--align-min", "--blank-rows", "--stage2", "--stage2-report", "--out", "--suffix")]
    if bad or not src:
        abort(f"test-mass: give the sources without {sorted(set(bad))} (this command sets the alignment itself)", 2)
    flag = ["--align-inductive", str(ALIGN)] if a.align_mode == "inductive" else ["--align", str(ALIGN)]
    suffix = f"_safemass_{a.align_mode}"
    mat = ART / f"test_{a.name}{suffix}.npy"
    if mat.exists():
        abort(f"{c120().rel(mat)} already exists — not overwritten", 4)
    with tempfile.TemporaryDirectory(prefix="c123_mass_") as tmp:
        cmd = [sys.executable, "-I", str(COMP / "tools" / "make_submission.py"), "--name", a.name, *src, *flag, "--align-min", str(ALIGN_MIN),
               "--out", str(Path(tmp) / "sub.csv"), "--suffix", suffix]
        r = subprocess.run(cmd, cwd=COMP, capture_output=True, text=True)
        try:
            if r.returncode != 0 or not mat.is_file():
                abort(f"make_submission code {r.returncode}: " + "; ".join(x for x in r.stderr.splitlines()[-3:]), 4)
            Q = np.load(mat)
        finally:
            if mat.exists():
                mat.unlink()
    from vlm import common as VC
    import labels

    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    K = len(classes)
    if Q.shape[1] != K:
        abort(f"test matrix with {Q.shape[1]} classes instead of {K}", 4)
    counts = np.bincount(y, minlength=K)
    target = np.where(counts >= ALIGN_MIN, counts, 0) / len(y)
    used = target > 0
    m = Q.mean(0)
    gen = [classes.index(c) for c in S2.GENERAL_CODES]
    res = {"kind": "rule (ii): class masses of the test side after the alignment (aggregates only)", "align_mode": a.align_mode,
           "name": a.name, "rows": int(Q.shape[0]), "classes_with_target": int(used.sum()), "l1_after": float(np.abs(m - target)[used].sum()),
           "none_after": float(m[0]), "none_target": float(target[0]), "general_after": float(m[gen].sum()), "general_target": float(target[gen].sum()),
           "mass_without_target": float(m[~used].sum()), "max": L1_MAX, "ok": bool(np.abs(m - target)[used].sum() <= L1_MAX)}
    out = OUT / f"c123_test_mass_{a.align_mode}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(c120().to_json(res))
    print(f"class masses of the test side after the alignment ({a.align_mode}, {res['rows']} rows, {res['classes_with_target']} classes with a target): "
          f"L1 {res['l1_after']:.4f} (limit {L1_MAX}); NONE {res['none_after']:.4f} (target {res['none_target']:.4f}), general codes "
          f"{res['general_after']:.4f} (target {res['general_target']:.4f}); written: {c120().rel(out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof", help="nested reproduction a/b, null points, placebos -> artifacts/eda/private/safety_set/")
    o.add_argument("--placebos", type=int, default=len(PLACEBO_SEEDS))
    s = sub.add_parser("splits", help="new meta-splits -> artifacts/eda/private/safety_set/")
    s.add_argument("--n", type=int, default=20)
    sub.add_parser("splits-inductive", help="additional: strictly inductive line (F inductive) on the same meta-splits, paired with a (M)")
    r = sub.add_parser("rule", help="pre-registered safety-set rule from oof + splits (+ test report)")
    r.add_argument("--test-report", help="report of make_submission --stage2-report (field class_mass)")
    r.add_argument("--override-ii", choices=OVERRIDE_DECISIONS, help="override the CV condition of (ii) — only with the pinned decision file decisions/<id>.json")
    f = sub.add_parser("fit", help="build (--save) or recompute (--verify) the deployment file artifacts/stage2_safety_final.json")
    f.add_argument("--out")
    f.add_argument("--probe", action="store_true", help="probe regardless of the result of the rule, only under artifacts/eda/private/safety_set/ (never deployed)")
    f.add_argument("--variant", choices=("a", "b"), help="variant (default: by the rule; probe: a)")
    f.add_argument("--override-ii", choices=OVERRIDE_DECISIONS, help="override the CV condition of (ii) — only with the pinned decision file decisions/<id>.json")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    nl = sub.add_parser("null", help="write the null model (weight 0) for the deployment file")
    nl.add_argument("--out", required=True)
    nl.add_argument("--source", default=str(STAGE2_FILE))
    tm = sub.add_parser("test-mass", help="aggregates only: class masses of the test side after the alignment (sources after --)")
    tm.add_argument("--name", required=True)
    tm.add_argument("--align-mode", choices=("inductive", "set"), required=True)
    tm.add_argument("src", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "oof" and not 0 <= a.placebos <= len(PLACEBO_SEEDS):
        abort(f"--placebos {a.placebos}", 2)
    return {"oof": cmd_oof, "splits": cmd_splits, "rule": cmd_rule, "fit": cmd_fit, "null": cmd_null, "test-mass": cmd_test_mass,
            "splits-inductive": cmd_splits_inductive}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
