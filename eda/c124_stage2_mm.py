"""SECOND STAGE WITH THE MINOR × MECHANICAL-MATCH COLUMN (primary line, final candidate): the second stage with interactions I1
(artifacts/stage2_primary_i1.json) with the column minor_x_mech27 (feature set main_guard_i1_mm, src/stage2.py; number comparison
src/answer_match.py without eval). eda/c118_stage2.py and eda/c121_stage2_i1.py stay unchanged and are only used as modules. Build
script: tools/builds/primary_final.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c124_stage2_mm.py parser                       # mech27 training/test (aggregates)
    python -I eda/c124_stage2_mm.py oof                          # nested: I1 form bit-identical, MM, copy null point, 15 placebos
    python -I eda/c124_stage2_mm.py splits [--n 20]              # new meta-splits
    python -I eda/c124_stage2_mm.py rule                         # pre-registered expected-value rule
    python -I eda/c124_stage2_mm.py fit (--save | --verify) [--probe]   # deployment file artifacts/stage2_primary_final.json (probe: private only)
    python -I eda/c124_stage2_mm.py null --source <file> --out <file>   # null model (weight 0) with the field mech27

FEATURE (fixed before the evaluation): minor_x_mech27[row, candidate] = 1[default_severity of the candidate = minor] · mech27[row];
mech27 = the final answer read by the pr27 channel (field student_final_answer of the last readable JSON answer) matches the answer key
mechanically. Training: artifacts/runs/pr27_rp/pr27_rp/{proto_a,proto_b}/proto_raw.jsonl (table train); test:
artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl (table test, frozen). At the end of main_guard_i1, standardised like all
features; otherwise the I1 form unchanged (K 5, λ 0.1, β choice, folds, knowledge from other folds, NONE guard also in the gate).
Motivation: by the annotation protocol the root is always a major finding (one that changes the solution path or the result); the column
lets the reranker weigh codes of default severity minor in rows whose final answer is right (the weight is fitted, not fixed).

OOF (`oof`; as c121 oof): per form F, oL, M: I1 form recomputed (c121.run_i1) = stage2_i1 row values bit-identical; MM (forced and with
gate); copy null point (column 0 -> I1 form bit-identical: row values, abstention, orders, candidates, guard marks); Δ against the I1 form
with σ over questions; placebos (row values of mech27 permuted within each block, default_rng(seed + rows of the block).permutation,
seeds 31000 + 97·s). -> artifacts/eda/private/stage2_mm/.
SPLITS (`splits`): StratifiedKFold(5, shuffle, 1000 + s) by root_code, channel OOFs fixed; per split base = column 0 (= I1 form), real,
placebo (seed 77000 + s); forms F and M.
EXPECTED-VALUE RULE (fixed before the evaluation): (i) copy null point (OOF here; file part in the build script
tools/builds/primary_final.sh); (ii) ½(F+M) against the I1 form > 0 and above the placebo mean (≥ 5); (iii) mean over ≥ 10 new
meta-splits > 0. Otherwise no deployment (`fit` exit 5; `--probe` then only writes under artifacts/eda/private/stage2_mm/probe/).
DEPLOYMENT (`fit`): first recomputes artifacts/stage2_primary_i1.json character for character (c121 fit --verify in this process), then
replaces ONLY models.main by the main_guard_i1_mm model (features of the deployment form = evaluated rows of `oof` bit-identical, the first
63 = I1 form) and appends mm_build, mech27 (test answers with SHA-256, training sources, minor codes) and derived_from ->
artifacts/stage2_primary_final.json. Test side: tools/make_submission.py --stage2 artifacts/stage2_primary_final.json (directly; the set
main_guard_i1_mm is admitted for the main model).

Training rows only for statistics; of the test side only shares. Exit 3 = input missing; 4 = checksum/recomputation/null point violated;
5 = pre-registered rule not met.
"""
from __future__ import annotations

import argparse
import hashlib
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

import answer_match as AM  # noqa: E402
import stage2 as S2  # noqa: E402

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "stage2_mm"
PARSER_JSON = OUT / "c124_parser.json"
MM_OOF_JSON = OUT / "c124_mm_oof.json"
MM_OOF_ROWS = OUT / "c124_mm_oof_rows.npz"
MM_SPLITS_JSON = OUT / "c124_mm_splits.json"
STAGE2_MM = ART / "stage2_primary_final.json"
PROBE_STAGE2 = OUT / "probe" / "stage2_primary_final_probe.json"
STACK_DEFAULT = "s032_stage2"
TRAIN_ANSWERS = ("artifacts/runs/pr27_rp/pr27_rp/proto_a/proto_raw.jsonl", "artifacts/runs/pr27_rp/pr27_rp/proto_b/proto_raw.jsonl")
TEST_ANSWERS = "artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl"
FORMS = ("fold", "pre", "set")
FORM_SHORT = {"fold": "F", "pre": "oL", "set": "M"}
PLACEBO_SEEDS = tuple(31000 + 97 * s for s in range(15))  # 15 placebos, fixed seeds
SPLIT_SEED0, SPLIT_PLACEBO_SEED0 = 1000, 77000
MIN_PLACEBOS, MIN_SPLITS = 5, 10

_MODS: dict = {}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def c121():
    """eda/c121_stage2_i1.py as a module (c118 underneath; both unchanged)."""
    if "c121" not in _MODS:
        _MODS["c121"] = _load("c124_c121", COMP / "eda" / "c121_stage2_i1.py")
    return _MODS["c121"]


def c118():
    return c121().c118()


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ row feature mech27

def taxonomy_minor(root: Path, classes: list[str]) -> np.ndarray:
    import pandas as pd

    tax = pd.read_csv(root / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)
    return AM.minor_mask(classes, dict(zip(tax["id"], tax["default_severity"])))


def train_mech(ctx: dict) -> tuple[np.ndarray, list]:
    """mech27 of the 1,500 training rows from the frozen pr27 answers (table train checked)."""
    for f in TRAIN_ANSWERS:
        p = COMP / f
        if not p.is_file():
            raise c118().Missing(f"{f} missing")
        meta = p.parent / "proto_meta.json"
        if meta.is_file() and json.loads(meta.read_text()).get("table") != "train":
            abort(f"{meta} does not name the table train")
    answers = AM.last_answers([COMP / f for f in TRAIN_ANSWERS], ctx["n"])
    return AM.mech_rows(answers, ctx["train"]["correct_answer"].tolist()), answers


def answers_sha256() -> dict:
    return {f: sha256_file(COMP / f) for f in (*TRAIN_ANSWERS, TEST_ANSWERS)}


def load_context(stack: str) -> dict:
    """Context of the primary line (c118) with mech27 and minor attached (attach_mech on ctx['static'])."""
    C = c118()
    try:
        ctx = C.context(stack)
        mech, answers = train_mech(ctx)
    except C.Missing as e:
        abort(str(e), 3)
    minor = taxonomy_minor(ctx["root"], ctx["classes"])
    S2.attach_mech(ctx["static"], mech, minor)
    ctx.update(mech27=mech, minor=minor, answers27=answers)
    return ctx


def inputs(ctx: dict) -> dict:
    C = c118()
    out = C.inputs_sha256(ctx)
    out["pr27_answers"] = answers_sha256()
    return out


# ------------------------------------------------------------------------------------------------ parser

def cmd_parser(a) -> int:
    """mech27 on training and test (shares only), minor codes."""
    from vlm import common as VC
    import labels

    root = VC.find_data_root()
    tr, te = VC.load_table(root, "train"), VC.load_table(root, "test")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    A_tr = AM.last_answers([COMP / f for f in TRAIN_ANSWERS], len(tr))
    A_te = AM.last_answers([COMP / TEST_ANSWERS], len(te))
    m_tr = AM.mech_rows(A_tr, tr["correct_answer"].tolist())
    m_te = AM.mech_rows(A_te, te["correct_answer"].tolist())
    minor = taxonomy_minor(root, classes)
    res = {"kind": "mech27 (final answer read by the pr27 channel = answer key), shares only", "answers": answers_sha256(),
           "train": {"rows": len(tr), "share_with_answer": float(np.mean([o is not None for o in A_tr])), "share_mech": float(m_tr.mean()),
                     "rows_mech": int(m_tr.sum())},
           "test": {"rows": len(te), "share_with_answer": float(np.mean([o is not None for o in A_te])), "share_mech": float(m_te.mean())},
           "minor_codes": [c for c, v in zip(classes, minor) if v],
           "train_mech_sha256": hashlib.sha256(m_tr.astype("<f8").tobytes()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True)
    PARSER_JSON.write_text(c118().to_json(res))
    print(f"mech27: training {res['train']['share_mech']:.4f} ({res['train']['rows_mech']} rows; answer {res['train']['share_with_answer']:.4f}), test "
          f"{res['test']['share_mech']:.4f} (answer {res['test']['share_with_answer']:.4f}); minor codes {len(res['minor_codes'])}; written "
          f"{c118().rel(PARSER_JSON)}")
    return 0


# ------------------------------------------------------------------------------------------------ nested reproduction

def row_perm(R: int, seed: int) -> np.ndarray:
    """Placebo: reorder the rows of a block, default_rng(seed + R).permutation(R)."""
    return np.random.default_rng(seed + R).permutation(R)


def run_mm(ctx: dict, std: dict, nested: dict, form: str, mode: str = "real", seed: int | None = None) -> dict:
    """As c121.run_i1 (same steps, NONE guard also in the gate) with the feature set main_guard_i1_mm. mode: real; zero (column 0 ->
    copy null point); placebo (mech27 of the rows permuted within each block, candidate mask unchanged; seed)."""
    if mode not in ("real", "zero", "placebo") or (mode == "placebo") != (seed is not None):
        raise ValueError(f"mode {mode!r} / seed {seed!r}")
    C = c118()
    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND
    mech, minor = ctx["mech27"], ctx["minor"]

    def prep(X: np.ndarray, names: list[str], rows: np.ndarray, cand: np.ndarray) -> np.ndarray:
        if mode == "real":
            return X
        X = X.copy()
        j = names.index("minor_x_mech27")
        if mode == "zero":
            X[:, j] = 0.0
        else:
            src = rows[row_perm(len(rows), seed)]
            X[:, j] = np.repeat(mech[src], Kc) * minor[cand.reshape(-1)]
        return X

    forced, abst, base = np.zeros(n), np.zeros(n), np.zeros(n)
    changed, changed_forced, changed3 = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    top_all, cand_all, protected = np.zeros((n, Kc), np.int64), np.zeros((n, Kc), np.int64), np.zeros(n, bool)
    choice, X_eval, names = {}, {}, None
    for k in range(5):
        forms_T = nested[k]
        E = np.flatnonzero(folds == k)
        XT, candT, rowsT = [], [], []
        for jf in [j for j in range(5) if j != k]:
            Rj = np.flatnonzero(folds == jf)
            X, cand, names_j = C.block(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], S2.MM_SET, exclude=(k,))
            XT.append(prep(X, names_j, Rj, cand))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = C.block(ctx, E, std[form][E], std["pre"][E], S2.MM_SET)
        XE = prep(XE, names, E, candE)
        X_eval[k] = XE
        labT = [S2.labels_of(c, y[r]) for c, r in zip(candT, rowsT)]
        g = S2.gate([(XT[b], labT[b], candT[b], y[rowsT[b]]) for b in range(len(XT))], protect=True)
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


def same_rows(a: dict, b: dict, keys=("forced", "abstain", "top", "cand", "protected")) -> bool:
    return bool(all(np.array_equal(a[k], b[k]) for k in keys))


def half_fm(d: dict) -> float:
    return 0.5 * (float(d["fold"]) + float(d["set"]))


def cmd_oof(a) -> int:
    t0 = time.time()
    C, M121 = c118(), c121()
    ctx = load_context(a.stack)
    ref_file = M121.I1_OOF_ROWS
    if not ref_file.is_file() or not M121.I1_OOF_JSON.is_file():
        abort(f"{C.rel(ref_file)} missing — run python -I eda/c121_stage2_i1.py oof first (build script of the I1 form)", 3)
    i1 = json.loads(M121.I1_OOF_JSON.read_text())
    if i1.get("inputs") != C.inputs_sha256(ctx) or i1.get("stack_sha256") != C.sha256_file(ctx["stack_file"]):
        abort(f"{C.rel(M121.I1_OOF_JSON)} was computed with other inputs — rerun c121 oof")
    ref = np.load(ref_file)
    y, seen = ctx["y"], ctx["seen"]
    draws_q = C.C107.question_draws(ctx["q_code"])
    std, betas, nested = M121.prepare(ctx)
    rec_rows = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS}
    seeds = PLACEBO_SEEDS[: a.placebos]
    res = {"kind": "I1 form (main_guard_i1 + NONE guard) with the column minor_x_mech27 (main_guard_i1_mm), nested as in c121 oof",
           "stack": C.rel(ctx["stack_file"]), "stack_sha256": C.sha256_file(ctx["stack_file"]), "beta_outer": betas, "K": S2.K_CAND,
           "lambda": S2.LAMBDA, "feature_set": S2.MM_SET, "protect_none": True, "train_share_mech": float(ctx["mech27"].mean()),
           "minor_codes": [c for c, v in zip(ctx["classes"], ctx["minor"]) if v], "reference_rows": C.rel(ref_file),
           "sets": {"i1": {}, "mm": {}}, "placebo": {"seeds": list(seeds), "per_seed": []}, "inputs": inputs(ctx)}
    rows_out, null_ok, pl = {}, {}, {s: {} for s in seeds}
    for form in FORMS:
        s = M121.run_i1(ctx, std, nested, form, "real")  # I1 form recomputed
        if not same_rows(s, {k: ref[f"i1_{form}_{k}"] for k in ("forced", "abstain", "top", "cand", "protected")}):
            abort(f"I1 form [{form}] does not hit the row values of {C.rel(ref_file)} bit for bit")
        res["sets"]["i1"][form] = {"map_forced": float(s["forced"].mean()), "rows_equal_reference": True,
                                   "changed_top_share_forced": float(s["changed_top_forced"].mean())}
        r = run_mm(ctx, std, nested, form, "real")
        if not np.allclose(r["base"], rec_rows[form]):
            abort(f"base of the candidate order [MM|{form}] != recipe")
        z = run_mm(ctx, std, nested, form, "zero")
        null_ok[form] = same_rows(z, s)
        d = r["forced"] - s["forced"]
        ent = {"map_forced": float(r["forced"].mean()), "map_abstain": float(r["abstain"].mean()),
               "abstain_folds": int(sum(c["abstain"] for c in r["choice"].values())), "gate": {str(k): v for k, v in r["choice"].items()},
               "n_features": len(r["features"]), "changed_top_share": float(r["changed_top"].mean()),
               "changed_top_share_forced": float(r["changed_top_forced"].mean()), "changed_top3_share_forced": float(r["changed_top3_forced"].mean()),
               "vs_i1": C.boot(d, ctx, draws_q), "vs_i1_with_abstention": C.boot(r["abstain"] - s["abstain"], ctx, draws_q),
               "vs_recipe_beta": C.boot(r["forced"] - rec_rows[form], ctx, draws_q), "rows_protected": int(r["protected"].sum()),
               "rows_other_top1_vs_i1": int((r["top"][:, 0] != s["top"][:, 0]).sum()),
               "rows_other_top3_vs_i1": int((r["top"][:, :3] != s["top"][:, :3]).any(1).sum()), "null_zero_equals_i1": null_ok[form]}
        if form == "set":
            Xe = C.eval_rows_in_table_order(ctx, r["X_eval"])
            ent["eval_features_sha256"] = C.sha256_array(Xe)
            ent["features"] = r["features"]
        res["sets"]["mm"][form] = ent
        for key in ("forced", "abstain", "top", "cand", "protected"):
            rows_out[f"mm_{form}_{key}"] = r[key]
        for sd in seeds:
            p = run_mm(ctx, std, nested, form, "placebo", sd)
            pl[sd][form] = float((p["forced"] - s["forced"]).mean())
        v = ent["vs_i1"]
        print(f"  [MM|{form:4s}] {ent['n_features']} features | MAP {ent['map_forced']:.6f} (I1 form {res['sets']['i1'][form]['map_forced']:.6f}) | "
              f"vs. I1 form {v['delta']:+.5f} ± {v['se_q']:.5f} {v['fold_units']} | null point {'bit-identical' if null_ok[form] else 'VIOLATED'} | "
              f"abstention {ent['abstain_folds']}/5 | new top code {ent['changed_top_share_forced']:.4f} | other top code/first three than the I1 form "
              f"{ent['rows_other_top1_vs_i1']}/{ent['rows_other_top3_vs_i1']} | placebo mean {np.mean([pl[q][form] for q in seeds]):+.5f}",
              flush=True)
    res["null_point_i"] = {"ok": bool(all(null_ok.values())), "forms": null_ok}
    summ = {FORM_SHORT[f]: res["sets"]["mm"][f]["vs_i1"]["delta"] for f in FORMS}
    summ["A"] = half_fm({f: res["sets"]["mm"][f]["vs_i1"]["delta"] for f in FORMS})
    summ["units"] = {FORM_SHORT[f]: res["sets"]["mm"][f]["vs_i1"]["units"] for f in FORMS}
    res["summary"] = summ
    per = [{"seed": sd, "F": pl[sd]["fold"], "oL": pl[sd]["pre"], "M": pl[sd]["set"], "A": half_fm(pl[sd])} for sd in seeds]
    pa = np.array([p["A"] for p in per]) if per else np.zeros(0)
    res["placebo"].update({"per_seed": per, "n": len(per), "A_mean": float(pa.mean()) if len(pa) else None,
                           "A_sd": float(pa.std(ddof=1)) if len(pa) > 1 else None, "A_max": float(pa.max()) if len(pa) else None,
                           "share_ge_real": float((pa >= summ["A"]).mean()) if len(pa) else None})
    OUT.mkdir(parents=True, exist_ok=True)
    MM_OOF_JSON.write_text(C.to_json(res))
    np.savez_compressed(MM_OOF_ROWS, **rows_out)
    plr = res["placebo"]
    print(f"MM vs. I1 form: F {summ['F']:+.5f} | oL {summ['oL']:+.5f} | M {summ['M']:+.5f} | ½(F+M) {summ['A']:+.5f}; placebos (n {plr['n']}) mean "
          f"{plr['A_mean']:+.5f} sd {plr['A_sd']:.5f} max {plr['A_max']:+.5f}, share ≥ real {plr['share_ge_real']:.3f}; copy null point "
          f"{'PASSED' if res['null_point_i']['ok'] else 'VIOLATED'}; written: {C.rel(MM_OOF_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0 if res["null_point_i"]["ok"] else 4


# ------------------------------------------------------------------------------------------------ new meta-splits

def cmd_splits(a) -> int:
    t0 = time.time()
    if a.n < MIN_SPLITS:
        abort(f"--n {a.n}: the pre-registered rule requires at least {MIN_SPLITS} meta-splits", 2)
    C, M121 = c118(), c121()
    ctx0 = load_context(a.stack)
    y = ctx0["y"]
    res = {"kind": "expected-value rule (iii): minor_x_mech27 over new meta-splits (channel OOFs fixed, meta-folds new)",
           "stack": C.rel(ctx0["stack_file"]), "stack_sha256": C.sha256_file(ctx0["stack_file"]), "splits": [], "inputs": inputs(ctx0)}
    for s in range(a.n):
        ctx = dict(ctx0)
        ctx["folds"] = M121.split_folds(y, SPLIT_SEED0 + s)
        std, betas, nested = M121.prepare(ctx)
        rec = {"seed": SPLIT_SEED0 + s, "placebo_seed": SPLIT_PLACEBO_SEED0 + s, "beta_outer": betas, "base_map": {}, "real": {}, "placebo": {}}
        for form in ("fold", "set"):
            b = run_mm(ctx, std, nested, form, "zero")
            if s == 0:  # copy null point also on the new meta-split: column 0 = I1 form (c121.run_i1)
                if not same_rows(b, M121.run_i1(ctx, std, nested, form, "real")):
                    abort(f"meta-split {SPLIT_SEED0}: column 0 is not the I1 form [{form}]")
            r = run_mm(ctx, std, nested, form, "real")
            p = run_mm(ctx, std, nested, form, "placebo", SPLIT_PLACEBO_SEED0 + s)
            rec["base_map"][form] = float(b["forced"].mean())
            rec["real"][form] = float((r["forced"] - b["forced"]).mean())
            rec["placebo"][form] = float((p["forced"] - b["forced"]).mean())
        res["splits"].append(rec)
        print(f"  split {s:2d} (seed {rec['seed']}): I1 form F {rec['base_map']['fold']:.4f} M {rec['base_map']['set']:.4f} | real F "
              f"{rec['real']['fold']:+.4f} M {rec['real']['set']:+.4f} | placebo F {rec['placebo']['fold']:+.4f} M {rec['placebo']['set']:+.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    summ = {}
    for key in ("real", "placebo"):
        F = np.array([r[key]["fold"] for r in res["splits"]])
        M = np.array([r[key]["set"] for r in res["splits"]])
        A = 0.5 * (F + M)
        summ[key] = {"F_mean": float(F.mean()), "M_mean": float(M.mean()), "A_mean": float(A.mean()), "A_se": float(A.std(ddof=1) / np.sqrt(len(A))),
                     "share_A_pos": float((A > 0).mean()), "share_F_pos": float((F > 0).mean()), "share_M_pos": float((M > 0).mean())}
    Ad = np.array([half_fm(r["real"]) - half_fm(r["placebo"]) for r in res["splits"]])
    summ["real_minus_placebo"] = {"A_mean": float(Ad.mean()), "A_se": float(Ad.std(ddof=1) / np.sqrt(len(Ad))), "share_A_pos": float((Ad > 0).mean())}
    res["summary"] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    MM_SPLITS_JSON.write_text(C.to_json(res))
    sr, sp = summ["real"], summ["placebo"]
    print(f"MM over {a.n} meta-splits: F {sr['F_mean']:+.5f} ({sr['share_F_pos']:.0%} > 0) | M {sr['M_mean']:+.5f} ({sr['share_M_pos']:.0%}) | ½(F+M) "
          f"{sr['A_mean']:+.5f} ± {sr['A_se']:.5f} ({sr['share_A_pos']:.0%}); placebo ½(F+M) {sp['A_mean']:+.5f}; real − placebo "
          f"{summ['real_minus_placebo']['A_mean']:+.5f} ± {summ['real_minus_placebo']['A_se']:.5f}; written: {C.rel(MM_SPLITS_JSON)} "
          f"({time.time() - t0:.1f} s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ expected-value rule

def rule_result(oof: dict, splits: dict) -> dict:
    A, pl = float(oof["summary"]["A"]), oof["placebo"]
    ok_ii = bool(A > 0 and int(pl.get("n") or 0) >= MIN_PLACEBOS and A > float(pl["A_mean"]))
    sr = splits["summary"]["real"]
    ok_iii = bool(len(splits["splits"]) >= MIN_SPLITS and float(sr["A_mean"]) > 0)
    ok_i = bool(oof["null_point_i"]["ok"])
    return {"rule": "expected-value rule", "i_oof": {"ok": ok_i, "forms": oof["null_point_i"]["forms"]},
            "i_file": "build script tools/builds/primary_final.sh: column weight 0 -> I1-form file and test matrix byte-identical (otherwise abort)",
            "ii": {"ok": ok_ii, "half_F_plus_M": A, "F": float(oof["summary"]["F"]), "M": float(oof["summary"]["M"]), "oL": float(oof["summary"]["oL"]),
                   "placebo_n": int(pl.get("n") or 0), "placebo_mean": float(pl["A_mean"]), "placebo_max": float(pl["A_max"]),
                   "placebo_share_ge_real": float(pl["share_ge_real"])},
            "iii": {"ok": ok_iii, "n_splits": len(splits["splits"]), "half_F_plus_M_mean": float(sr["A_mean"]), "se": float(sr["A_se"]),
                    "share_pos": float(sr["share_A_pos"]), "F_mean": float(sr["F_mean"]), "M_mean": float(sr["M_mean"])},
            "verdict_oof": "deploy" if (ok_i and ok_ii and ok_iii) else "no deployment"}


def load_results(ctx: dict | None) -> tuple[dict, dict]:
    C = c118()
    for f in (MM_OOF_JSON, MM_SPLITS_JSON):
        if not f.is_file():
            abort(f"{C.rel(f)} missing — run python -I eda/c124_stage2_mm.py oof and splits first", 3)
    oof, sp = json.loads(MM_OOF_JSON.read_text()), json.loads(MM_SPLITS_JSON.read_text())
    if ctx is not None:
        sha, inp = C.sha256_file(ctx["stack_file"]), inputs(ctx)
        for nm, o in ((MM_OOF_JSON, oof), (MM_SPLITS_JSON, sp)):
            if o.get("stack_sha256") != sha or o.get("inputs") != inp:
                abort(f"{C.rel(nm)} was computed with a different blend file or different inputs — recompute it")
    return oof, sp


def cmd_rule(a) -> int:
    oof, sp = load_results(None)
    r = rule_result(oof, sp)
    ii, iii = r["ii"], r["iii"]
    print(f"expected-value rule: (i) OOF null point {'bit-identical' if r['i_oof']['ok'] else 'VIOLATED'} | (ii) ½(F+M) {ii['half_F_plus_M']:+.5f} > 0 and > "
          f"placebo mean {ii['placebo_mean']:+.5f} (n {ii['placebo_n']}) -> {'yes' if ii['ok'] else 'NO'} | (iii) mean over {iii['n_splits']} "
          f"meta-splits {iii['half_F_plus_M_mean']:+.5f} ± {iii['se']:.5f} -> {'yes' if iii['ok'] else 'NO'} => {r['verdict_oof']} (file null point in the build script)")
    return 0 if r["verdict_oof"] == "deploy" else 5


# ------------------------------------------------------------------------------------------------ deployment file

def cmd_fit(a) -> int:
    C, M121 = c118(), c121()
    target = Path(a.out) if a.out else (PROBE_STAGE2 if a.probe else STAGE2_MM)
    blocked = {(ART / f).resolve() for f in ("stage2_s032.json", "stage2_s034.json", "stage2_s035.json", "stage2_s036_safe.json", "stage2_primary_i1.json")}
    if target.resolve() in blocked:
        abort(f"fit never writes an existing deployment file — separate output (default {C.rel(STAGE2_MM)})", 2)
    if a.probe and not target.resolve().is_relative_to(OUT.resolve()):
        abort(f"--probe only writes under {C.rel(OUT)} (a probe is never a deployment file)", 2)
    print("  recomputation of stage2_primary_i1.json (c121 fit --verify):", flush=True)
    rc = M121.cmd_fit(argparse.Namespace(stack=a.stack, out=None, verify=True, save=False))
    if rc != 0:
        abort(f"c121 fit --verify code {rc}")
    i1 = json.loads(M121.STAGE2_I1.read_text())
    ctx = load_context(a.stack)
    oof, sp = load_results(ctx)
    rule = rule_result(oof, sp)
    if rule["verdict_oof"] != "deploy" and not a.probe:
        abort(f"expected-value rule not met ({rule['verdict_oof']}) — no deployment, no file", 5)
    y, Kc = ctx["y"], S2.K_CAND
    forms = C.deploy_forms(ctx)
    X, cand, names = C.deploy_features(ctx, forms, S2.MM_SET)
    got = C.sha256_array(X)
    want = (oof["sets"]["mm"].get("set") or {}).get("eval_features_sha256")
    if got != want:
        abort(f"null point (e) violated: features of the deployment form ({got[:16]}…) != evaluated rows of c124 oof ({str(want)[:16]}…)")
    mm = i1["models"]["main"]
    nm = len(mm["features"])
    if mm.get("feature_set") != S2.I1_SET or names[:nm] != mm["features"] or names[nm:] != list(S2.MM_FEATS) \
            or C.sha256_array(X[:, :nm]) != mm["features_sha256"]:
        abort("the first features of main_guard_i1_mm are not the features of main_guard_i1 from stage2_primary_i1.json (bit-identical)")
    lab = S2.labels_of(cand, y)
    g = S2.gate(C.fold_blocks(ctx, X, cand, lab), protect=True)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top0 = S2.reorder(cand, S2.scores(model, X, Kc))
    top = S2.protect_none(top0, cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    o = oof["sets"]["mm"]
    gm = {"features": names, "feature_set": S2.MM_SET, "protect_none": True, "w": [float(v) for v in model["w"]],
          "mu": [float(v) for v in model["mu"]], "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
          "oof_changed_top_share": float(o["set"]["changed_top_share_forced"]),
          "oof_changed_top_share_with_abstention": float(o["set"]["changed_top_share"]),
          "oof_changed_top3_share": float(o["set"]["changed_top3_share_forced"]),
          "oof_vs_i1": {f: o[f]["vs_i1"]["delta"] for f in FORMS}, "oof_vs_recipe_beta": {f: o[f]["vs_recipe_beta"]["delta"] for f in FORMS},
          "features_sha256": got, "insample": ins}
    new = {}
    for k, v in i1.items():
        if k != "derived_from":
            new[k] = {"main": gm, "fallback": v["fallback"]} if k == "models" else v
    new["mm_build"] = {"rule": "expected-value rule", "feature_set": S2.MM_SET, "protect_none": True, "feature": list(S2.MM_FEATS),
                       "definition": "1[default_severity of the candidate = minor] · 1[final answer read by the pr27 channel matches the answer key "
                                     "mechanically] (src/answer_match.py without eval), at the end of main_guard_i1, standardised like all features",
                       "oof": C.rel(MM_OOF_JSON), "splits": C.rel(MM_SPLITS_JSON), "rule_oof": rule,
                       "fallback": "models.fallback unchanged from stage2_primary_i1.json (fallback = the fallback-form file)"}
    new["mech27"] = {"test": {"file": TEST_ANSWERS, "sha256": sha256_file(COMP / TEST_ANSWERS)},
                     "train": [{"file": f, "sha256": sha256_file(COMP / f)} for f in TRAIN_ANSWERS],
                     "minor_codes": [c for c, v in zip(ctx["classes"], ctx["minor"]) if v], "train_share_mech": float(ctx["mech27"].mean()),
                     "reader": "src/answer_match.py: last_answers (last readable JSON answer per row), mech_rows (numeric value rel. 1e-6 or text)"}
    new["derived_from"] = {"file": C.rel(M121.STAGE2_I1), "sha256": sha256_file(M121.STAGE2_I1),
                           "differs_only_in": ["models.main", "mm_build", "mech27", "derived_from"], "build": "tools/builds/primary_final.sh",
                           "previous": i1.get("derived_from")}
    if a.probe:
        new["probe"] = True
        new["probe_note"] = ("PROBE: not eligible for deployment under the pre-registered rule as long as mm_build.rule_oof.verdict_oof is not \"deploy\"; "
                             "only for judging the test side (aggregates), never to be submitted")
    text = C.to_json(new)
    print(f"  [{S2.MM_SET}] {len(names)} features (main_guard_i1 {nm} bit-identical + minor_x_mech27) | inner gate {g['inner']:.5f} vs. base "
          f"{g['inner_base']:.5f} -> {'ABSTAIN (weight 0)' if g['abstain'] else 'apply'} | in-sample {ins['map_base']:.5f} → {ins['map_reranked']:.5f} "
          f"(new top code {ins['changed_top_share']:.4f}, guarded {ins['rows_protected']}; OOF {gm['oof_changed_top_share']:.4f}) | weight minor_x_mech27 "
          f"{gm['w'][-1]:+.4f} (mean {gm['mu'][-1]:.4f}, sd {gm['sd'][-1]:.4f})")
    print(f"  expected-value rule (OOF): {rule['verdict_oof']} | fallback, note guard (widened rule), β, K, λ, reference = stage2_primary_i1.json (SHA-256 "
          f"{new['derived_from']['sha256'][:16]}…)")
    if a.verify:
        if not target.is_file():
            abort(f"{C.rel(target)} missing — build it with --save first", 3)
        if target.read_text() != text:
            abort(f"{C.rel(target)} is not the recomputation")
        print(f"   VERIFIED: {C.rel(target)} = recomputation (SHA-256 {sha256_file(target)[:16]}…)")
    if a.save:
        if target.is_file() and target.read_text() != text:
            abort(f"{C.rel(target)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"   saved: {C.rel(target)} (SHA-256 {sha256_file(target)})" + ("  [PROBE — not eligible for deployment]" if a.probe else ""))
    return 0


def cmd_null(a) -> int:
    """Null model (weight 0) for the deployment file as c118 null, with the field mech27 (the test side computes the 64 features)."""
    C = c118()
    src = Path(a.source)
    if not src.is_file():
        abort(f"{C.rel(src)} missing", 3)
    ref = json.loads(src.read_text())
    try:
        out = C.null_stage2(a.stack, src)
    except C.Missing as e:
        abort(str(e), 3)
    out["mech27"] = ref["mech27"]
    Path(a.out).write_text(C.to_json(out))
    print(f"null model for {C.rel(src)} -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("parser", help="mech27 training/test (shares only)")
    o = sub.add_parser("oof", help="nested reproduction I1 form + MM, copy null point, placebos -> artifacts/eda/private/stage2_mm/")
    o.add_argument("--stack", default=STACK_DEFAULT)
    o.add_argument("--placebos", type=int, default=len(PLACEBO_SEEDS))
    s = sub.add_parser("splits", help="new meta-splits -> artifacts/eda/private/stage2_mm/")
    s.add_argument("--stack", default=STACK_DEFAULT)
    s.add_argument("--n", type=int, default=20)
    sub.add_parser("rule", help="pre-registered expected-value rule from oof + splits")
    f = sub.add_parser("fit", help="build (--save) or recompute (--verify) the deployment file artifacts/stage2_primary_final.json")
    f.add_argument("--stack", default=STACK_DEFAULT)
    f.add_argument("--out")
    f.add_argument("--probe", action="store_true", help="probe regardless of the result of the rule, only under artifacts/eda/private/stage2_mm/ (never deployed)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    nl = sub.add_parser("null", help="write the null model (weight 0) for the deployment file (with the field mech27)")
    nl.add_argument("--stack", default=STACK_DEFAULT)
    nl.add_argument("--source", default=str(STAGE2_MM))
    nl.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "oof" and not 0 <= a.placebos <= len(PLACEBO_SEEDS):
        abort(f"--placebos {a.placebos}", 2)
    return {"parser": cmd_parser, "oof": cmd_oof, "splits": cmd_splits, "rule": cmd_rule, "fit": cmd_fit, "null": cmd_null}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
