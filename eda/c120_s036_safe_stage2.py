"""SECOND STAGE of the INDUCTIVE line — the strictly inductive recipe + a conditional logit over the first 5 candidates AFTER the
inductive alignment (src/stage2.py, feature set "fallback" without notes) + NONE guard. Build script:
tools/builds/s036_safe_stage2.sh (file sub_s036_safe_stage2.csv).

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c120_s036_safe_stage2.py oof                      # nested reproduction, per-fold form -> private JSON
    python -I eda/c120_s036_safe_stage2.py fit (--save | --verify)  # deployment file artifacts/stage2_s036_safe.json
    python -I eda/c120_s036_safe_stage2.py null --out <file>        # null model (weight 0) for the copy null point

RECIPE (fixed before the evaluation; it comes from the reranker family of the model sweep, applied to the inductive line):
inductive line = v1r, zs27t, think18sb, pr27xm, m3x3_full; floors 0.01/0.05/0.05; β 0.4; inductive alignment 0.75 for classes
with at least 3 rows (class weights from the blend of the fit rows, logs with the mask first as in make_submission.oof_mixture).
Second stage: K 5, λ 0.1, feature set "fallback" (40 features; no notes, hence no note guard), knowledge of the training rows ONLY
from the other folds, form "per fold" (the deployment form of the inductive line). NONE guard: if the base order has NONE at the
top and the second stage wants another top code, the base order of the row stays. Abstention gate as stage2.gate, but on the
GUARDED order (as applied). Order on the test side: blend -> inductive alignment -> second stage -> NONE guard -> REL-01.

OOF (`oof`): outer fold k = evaluated rows. Training rows T (folds ≠ k): base forms from the weights of the folds except {j, k},
inductive class weights from the blend of the folds except {j, k}; knowledge of the rows from fold j: folds except {j, k}.
Evaluated rows: standard cross-fit (c107.cross_fit, last step inductive; checked bit for bit), knowledge from the folds ≠ k. Gate
over the four inner folds. Outputs: anchor of the inductive line 7,366/9,000, row values forced / with the guard / with the gate,
copy null point (weight 0 -> the inductive line bit for bit), share of new top codes (reference of guard (c)), checksum of the
features of the evaluated rows (null point (e) in `fit`). Target value (fixed before the build): deployment form +0.00489
(7,410/9,000).

DEPLOYMENT (`fit`): training rows = all 1,500 in the per-fold form (standard cross-fit), knowledge of a row of one fold only from the
other folds; abstention gate over the five folds on the guarded order; weights, standardisation, feature list, K, λ, gate, OOF
shares, M3 test draws and SHA-256 of the inputs -> artifacts/stage2_s036_safe.json (without timestamps; --verify requires
character identity). Null point (e): features of the deployment form == features of the evaluated rows of the OOF reproduction
(checksum), otherwise abort. The test side is computed by tools/make_submission.py --stage2 (field align_mode "inductive" ->
apply_stage2_safe, requires --align-inductive).

Training rows only; only statistics are printed. Exit 3 = input missing; 4 = checksum/recomputation/null point violated.
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

import stage2 as S2  # noqa: E402
from vlm import common as C  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C110 = _load("c120_c110", COMP / "eda" / "c110_q397_deploy.py")  # line_specs (recipe of the inductive line)
C107 = C110.C107  # context, load_base, logs_of, target_of, apply_last, cross_fit, set_align, rows_map, units, cluster_boot
MS, A2, C21 = C107.MS, C107.A2, C107.C21
A9 = _load("c120_a9", COMP / "eda" / "a9_m3_draws_oof.py")

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "stage2_inductive"
OOF_JSON = OUT / "c120_oof.json"
STAGE2_FILE = ART / "stage2_s036_safe.json"
M3 = "m3x3_full"
STACK = ART / "stack_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar.json"
STACK_SHA = "7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c"  # blend file of the inductive line (both rotation lists)
ANCHOR_UNITS = 7366  # inductive-line recipe per fold (field cv of the blend file 0.818444)
TARGET_UNITS = 7410  # deployment form (gate + guard, per fold): +0.00489 against the base recipe, fixed before the build
FEATURE_SET = "fallback"
ALIGN_INDUCTIVE, ALIGN_MIN = 0.75, 3
TEST_DRAWS = ("runs/m3_fullte_rp/m3_test_rank/test", "runs/m3d2te_rp/m3d2_test_rank/test", "runs/m3d3te_rp/m3d3_test_rank/test")
CHANGED_TOP_FACTOR = (0.5, 2.0)
NONE_IDX = 0


class Missing(RuntimeError):
    """Input missing (exit 3)."""


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=C107.json_default) + "\n"


# ------------------------------------------------------------------------------------------------ context

def context() -> dict:
    """Training context: labels, folds, channels of the inductive-line recipe (OOF before the floor, checksums), three M3 draws, blend file."""
    root = C.find_data_root()
    ctx = C107.context(root)
    train = C.load_table(root, "train")
    n, K, seen = ctx["n"], ctx["K"], ctx["seen"]
    if ctx["classes"][NONE_IDX] != "NONE":
        abort("NONE is not class 0 — the NONE guard requires that")
    mats, sha16 = C107.load_base(n, K)
    f = ART / f"oof_{M3}.npy"
    mf = ART / f"meta_{M3}.json"
    if not f.is_file() or not mf.is_file() or not STACK.is_file():
        raise Missing(f"{rel(f)}, {rel(mf)} or {rel(STACK)} missing")
    P = np.load(f)
    meta = json.loads(mf.read_text())
    rk = meta.get("rank") or {}
    if P.shape != (n, K) or not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-6) or rk.get("rows_by_draws") != {"3": n}:
        abort(f"{f.name}: not three draws on all {n} rows or not normalised")
    mats[M3], sha16[M3] = P, sha256_file(f)[:16]
    specs = [d["source"] for d in rk["draws"]]
    if len(specs) != len(S2.DRAWS):
        abort(f"{mf.name}: {len(specs)} draws instead of {len(S2.DRAWS)}")
    draws = []
    for s in specs:
        Pd, have, _ = A9.load_draw(s, n, seen)
        if not have.all():
            abort(f"{s}: draw not on all rows")
        draws.append(Pd)
    Mean, _ = A9.mean_of_draws([(Pd, np.ones(n, bool)) for Pd in draws], n, K)
    if not np.array_equal(Mean, mats[M3]):
        abort(f"mean of the three draws != {f.name}")
    if sha256_file(STACK) != STACK_SHA:
        abort(f"{STACK.name} is not the blend file of the inductive line ({STACK_SHA[:16]}…)")
    stack = json.loads(STACK.read_text())
    spec = C110.line_specs("safe", M3)[0]
    floors = {k: float(v) for k, v in spec["floors"].items() if v}
    if stack.get("channels") != list(spec["names"]) or {k: float(v) for k, v in (stack.get("floor") or {}).items()} != floors \
            or float(stack.get("arith_beta") or 0.0) != float(spec["beta"]) or spec["last"] != "ind" or spec.get("think") or stack.get("scaled"):
        abort(f"{STACK.name} does not match the recipe of the inductive line {spec['names']}")
    grade = train["grade"].to_numpy().astype(float)
    ctx.update(root=root, train=train, mats=mats, sha16=sha16, draw_specs=specs, draws=draws, spec=spec, stack=stack, grade=grade,
               key=S2.task_keys(train["question_id"].to_numpy(), train["topic_name"].to_numpy()),
               topic_s=np.asarray(train["topic_name"].to_numpy()).astype(str),
               L=C107.logs_of(mats, spec["names"], spec["floors"], seen, False), Lm=C107.logs_of(mats, spec["names"], spec["floors"], seen, True))
    ctx["static"] = S2.Static(spec["names"], mats, draws, grade, seen, ctx["classes"], None)
    return ctx


def inputs_sha256(ctx: dict) -> dict:
    root = ctx["root"]
    out = {f"oof_{nm}.npy": sha256_file(ART / f"oof_{nm}.npy") for nm in ctx["spec"]["names"]}
    out[f"meta_{M3}.json"] = sha256_file(ART / f"meta_{M3}.json")
    out["stack"] = {"file": rel(STACK), "sha256": sha256_file(STACK)}
    out["folds.npy"] = sha256_file(ART / "folds.npy")
    out["train.csv"] = sha256_file(root / "train.csv")
    out["taxonomy_v1.csv"] = sha256_file(root / "competition_data" / "taxonomy_v1.csv")
    return out


# ------------------------------------------------------------------------------------------------ recipe (inductive)

def fit_apply(ctx: dict, fit_idx: np.ndarray, ev_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights on log loss from fit_idx (a2.fit_mix), blend with β; inductive class weights from the blend of the fit rows (logs with
    the mask first), target = frequencies of the fit rows for classes with at least 3 rows, applied to ev_idx (as c107.cross_fit with last step "ind")."""
    y, seen, K, spec = ctx["y"], ctx["seen"], ctx["K"], ctx["spec"]
    L, Lm = ctx["L"], ctx["Lm"]
    th = np.asarray(A2.fit_mix([lg[fit_idx] for lg in L], None, y[fit_idx]), dtype=float).copy()
    st = {"theta": [float(v) for v in th], "arith_beta": float(spec["beta"])}
    PE = MS.mix_channels(st, [lg[ev_idx] for lg in L], None, seen)
    PF = MS.mix_channels(st, [lg[fit_idx] for lg in Lm], None, seen)
    QE = C107.apply_last(PE, C21.fit_weights(PF, C107.target_of(y, fit_idx, K)), seen)
    return PE, QE


def standard_forms(ctx: dict) -> dict:
    """Standard cross-fit per fold (weights and class weights from the folds ≠ k)."""
    n, K, folds = ctx["n"], ctx["K"], ctx["folds"]
    P, Q = np.zeros((n, K)), np.zeros((n, K))
    for k in range(5):
        F, E = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        P[E], Q[E] = fit_apply(ctx, F, E)
    return {"fold": Q, "pre": P}


def nested_forms(ctx: dict, k: int) -> dict:
    """Base forms of the training rows T (folds ≠ k) without any label from fold k."""
    n, K, folds = ctx["n"], ctx["K"], ctx["folds"]
    P, Q = np.zeros((n, K)), np.zeros((n, K))
    for j in range(5):
        if j == k:
            continue
        F = np.flatnonzero((folds != j) & (folds != k))
        E = np.flatnonzero(folds == j)
        P[E], Q[E] = fit_apply(ctx, F, E)
    return {"fold": Q, "pre": P}


def block(ctx: dict, rows: np.ndarray, FM: np.ndarray, PM: np.ndarray, exclude=()) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features of the rows `rows` of ONE fold (feature set fallback); knowledge from the other folds except `exclude`."""
    kn = S2.knowledge_other_folds(ctx["key"], ctx["topic_s"], ctx["y"], ctx["folds"], rows, ctx["K"], exclude)
    return S2.pair_features(ctx["static"], rows, FM, PM, kn, FEATURE_SET)


# ------------------------------------------------------------------------------------------------ guard and gate

def protect_none(top: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """NONE guard: base with NONE at the top and a new top code ≠ NONE -> base order of the whole row."""
    top, cand = np.asarray(top), np.asarray(cand)
    m = (cand[:, 0] == NONE_IDX) & (top[:, 0] != NONE_IDX)
    out = top.copy()
    out[m] = cand[m]
    return out


def gate_protected(blocks: list[tuple]) -> dict:
    """Abstention gate as stage2.gate, evaluated on the guarded order. blocks: [(X, target per pair, candidates, y of the rows)]."""
    got, base = [], []
    for a in range(len(blocks)):
        tr = [b for b in range(len(blocks)) if b != a]
        m = S2.fit_clogit(np.concatenate([blocks[b][0] for b in tr]), np.concatenate([blocks[b][1] for b in tr]), S2.K_CAND, S2.LAMBDA)
        X, _, cand, yr = blocks[a]
        got.append(S2.map_rows(protect_none(S2.reorder(cand, S2.scores(m, X, S2.K_CAND)), cand), yr))
        base.append(S2.map_rows(cand, yr))
    inner, inner_base = float(np.concatenate(got).mean()), float(np.concatenate(base).mean())
    return {"inner": inner, "inner_base": inner_base, "abstain": bool(not (inner > inner_base + 1e-12))}


# ------------------------------------------------------------------------------------------------ nested reproduction

def run_nested(ctx: dict, std: dict) -> dict:
    """Per outer fold k: fit on T (nested base forms), gate over the four inner folds, application to k."""
    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND
    out = {key: np.zeros(n) for key in ("base", "forced", "forced_prot", "gated_prot", "null")}
    top_all, topp_all, cand_all = (np.zeros((n, Kc), np.int64) for _ in range(3))
    gates, X_eval, names = {}, {}, None
    for k in range(5):
        nested = nested_forms(ctx, k)
        E = np.flatnonzero(folds == k)
        blocks = []
        for j in [j for j in range(5) if j != k]:
            Rj = np.flatnonzero(folds == j)
            X, cand, _ = block(ctx, Rj, nested["fold"][Rj], nested["pre"][Rj], exclude=(k,))
            blocks.append((X, S2.labels_of(cand, y[Rj]), cand, y[Rj]))
        XE, candE, names = block(ctx, E, std["fold"][E], std["pre"][E])
        X_eval[k] = XE
        g = gate_protected(blocks)
        model = S2.fit_clogit(np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]), Kc, S2.LAMBDA)
        top = S2.reorder(candE, S2.scores(model, XE, Kc))
        topp = protect_none(top, candE)
        zero = S2.reorder(candE, S2.scores(S2.zero_model(XE.shape[1]), XE, Kc))
        yE = y[E]
        out["base"][E], out["forced"][E], out["forced_prot"][E] = S2.map_rows(candE, yE), S2.map_rows(top, yE), S2.map_rows(topp, yE)
        out["gated_prot"][E] = out["base"][E] if g["abstain"] else out["forced_prot"][E]
        out["null"][E] = S2.map_rows(zero, yE)
        top_all[E], topp_all[E], cand_all[E] = top, topp, candE
        gates[k] = g
    return {**out, "top": top_all, "top_prot": topp_all, "cand": cand_all, "gates": gates, "X_eval": X_eval, "features": names}


def eval_rows_in_table_order(ctx: dict, X_eval: dict) -> np.ndarray:
    Kc = S2.K_CAND
    X = np.zeros((ctx["n"] * Kc, next(iter(X_eval.values())).shape[1]))
    for k, Xk in X_eval.items():
        E = np.flatnonzero(ctx["folds"] == k)
        X[(E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)] = Xk
    return X


def boot(ctx: dict, d: np.ndarray, draws: np.ndarray, mask=None) -> dict:
    cb = C107.cluster_boot(np.asarray(d, float), ctx["q_code"], draws, mask)
    out = {"delta": cb["mean"], "se_q": cb["se_q"], "z": cb["z_q"], "n": cb["n_rows"]}
    if mask is None:
        out.update(units=int(np.rint(6 * np.sum(d))), fold_units=[int(np.rint(6 * d[ctx["folds"] == k].sum())) for k in range(5)])
    return out


def cmd_oof(a) -> int:
    t0 = time.time()
    try:
        ctx = context()
    except Missing as e:
        abort(str(e), 3)
    y, seen = ctx["y"], ctx["seen"]
    std = standard_forms(ctx)
    lib = C107.cross_fit(y, ctx["folds"], seen, ctx["mats"], ctx["spec"])
    if not (np.array_equal(lib["Q"], std["fold"]) and np.array_equal(lib["P"], std["pre"])):
        abort("standard cross-fit differs from c107.cross_fit (last step inductive)")
    base_rows = C107.rows_map(y, std["fold"], seen)
    u_base = int(C107.units(base_rows).sum())
    if u_base != ANCHOR_UNITS:
        abort(f"anchor of the inductive line per fold {u_base}/9000 instead of {ANCHOR_UNITS}")
    r = run_nested(ctx, std)
    if not np.array_equal(r["base"], base_rows):
        abort("base of the candidate order != recipe")
    null_ok = bool(np.array_equal(r["null"], base_rows))
    if not null_ok:
        abort("copy null point (weight 0) does not reproduce the inductive line bit for bit")
    u_dep = int(C107.units(r["gated_prot"]).sum())
    draws_q = C107.question_draws(ctx["q_code"])
    Xe = eval_rows_in_table_order(ctx, r["X_eval"])
    res = {"anchor_units": u_base, "target_units": TARGET_UNITS, "deploy_units": u_dep, "reproduction_ok": bool(u_dep == TARGET_UNITS),
           "null_point_exact": null_ok, "feature_set": FEATURE_SET, "features": r["features"], "n_features": len(r["features"]),
           "K": S2.K_CAND, "lambda": S2.LAMBDA, "gates": {str(k): v for k, v in r["gates"].items()},
           "abstain_folds": int(sum(g["abstain"] for g in r["gates"].values())),
           "vs_base": {v: boot(ctx, r[v] - base_rows, draws_q) for v in ("forced", "forced_prot", "gated_prot")},
           "parts": {p: boot(ctx, r["gated_prot"] - base_rows, draws_q, m) for p, m in (("part A", ctx["part_a"]), ("part B", ~ctx["part_a"]))},
           "changed_top_share_forced": float((r["top"][:, 0] != r["cand"][:, 0]).mean()),
           "changed_top_share_prot": float((r["top_prot"][:, 0] != r["cand"][:, 0]).mean()),
           "changed_top3_share_prot": float((r["top_prot"][:, :3] != r["cand"][:, :3]).any(1).mean()),
           "rows_protected": int(((r["cand"][:, 0] == NONE_IDX) & (r["top"][:, 0] != NONE_IDX)).sum()),
           "eval_features_sha256": sha256_array(Xe), "novel_task_share_top5": float(Xe[:, r["features"].index("novel_task")].mean()),
           "inputs": inputs_sha256(ctx)}
    res["seconds"] = round(time.time() - t0, 1)
    v = res["vs_base"]["gated_prot"]
    print(f"  anchor of the inductive line per fold {u_base}/9000; deployment form (gate + guard) {u_dep}/9000 (target {TARGET_UNITS}) -> "
          f"{'PASSED' if res['reproduction_ok'] else 'FAILED'}; vs. base recipe {v['delta']:+.5f} ± {v['se_q']:.5f} {v['fold_units']}; "
          f"forced without the guard {res['vs_base']['forced']['delta']:+.5f}; gate abstains {res['abstain_folds']}/5; copy null point {null_ok}; "
          f"new top code (with the guard) {res['changed_top_share_prot']:.4f}; guarded rows {res['rows_protected']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    OOF_JSON.write_text(to_json(res))
    np.savez_compressed(OUT / "c120_oof_rows.npz", base=base_rows, forced=r["forced"], forced_prot=r["forced_prot"], gated_prot=r["gated_prot"])
    print(f"written: {rel(OOF_JSON)} ({res['seconds']} s)")
    return 0 if res["reproduction_ok"] else 4


# ------------------------------------------------------------------------------------------------ deployment file

def deploy_features(ctx: dict, std: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair features of all 1,500 training rows in the per-fold deployment form, table row order."""
    n, Kc = ctx["n"], S2.K_CAND
    X, cand, names = None, np.zeros((n, Kc), np.int64), None
    for k in range(5):
        E = np.flatnonzero(ctx["folds"] == k)
        Xk, ck, names = block(ctx, E, std["fold"][E], std["pre"][E])
        if X is None:
            X = np.zeros((n * Kc, Xk.shape[1]))
        X[(E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)] = Xk
        cand[E] = ck
    return X, cand, names


def build_stage2(ctx: dict, oof: dict) -> dict:
    y, Kc, folds = ctx["y"], S2.K_CAND, ctx["folds"]
    std = standard_forms(ctx)
    cv_units = int(C107.units(C107.rows_map(y, std["fold"], ctx["seen"])).sum())
    if cv_units != int(round(float(ctx["stack"]["cv"]) * 6 * ctx["n"])):
        abort(f"cross value of the deployment form {cv_units}/9000 does not match the field cv of the blend file")
    X, cand, names = deploy_features(ctx, std)
    got = sha256_array(X)
    if got != oof.get("eval_features_sha256"):
        abort(f"null point (e) violated: features of the deployment form ({got[:16]}…) != evaluated rows of the OOF reproduction "
              f"({str(oof.get('eval_features_sha256'))[:16]}…) — rerun c120 oof")
    lab = S2.labels_of(cand, y)
    blocks = []
    for k in range(5):
        E = np.flatnonzero(folds == k)
        pos = (E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)
        blocks.append((X[pos], lab[pos], cand[E], y[E]))
    g = gate_protected(blocks)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top = protect_none(S2.reorder(cand, S2.scores(model, X, Kc)), cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked_protected": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean())}
    meta = json.loads((ART / f"meta_{M3}.json").read_text())
    return {"kind": "second stage of the inductive line: conditional logit over the first K candidates after the INDUCTIVE alignment "
                    "(src/stage2.py, feature set fallback without notes) + NONE guard (eda/c120_s036_safe_stage2.py)",
            "stack": rel(STACK), "stack_sha256": sha256_file(STACK), "channels": list(ctx["stack"]["channels"]),
            "arith_beta": float(ctx["stack"]["arith_beta"]), "align_mode": "inductive", "align_inductive": ALIGN_INDUCTIVE, "align_min": ALIGN_MIN,
            "K": Kc, "lambda": S2.LAMBDA, "null": False, "feature_set": FEATURE_SET, "protect_none": True,
            "draws_channel": M3, "draw_sources": ctx["draw_specs"], "draw_rank_source": meta["rank"]["source"], "test_draws": list(TEST_DRAWS),
            "model": {"features": names, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
                      "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
                      "oof_changed_top_share": float(oof["changed_top_share_prot"]), "oof_changed_top3_share": float(oof["changed_top3_share_prot"]),
                      "oof_vs_base": {k: v["delta"] for k, v in oof["vs_base"].items()}, "features_sha256": got, "insample": ins},
            "novel_task_share_top5": float(oof["novel_task_share_top5"]),
            "guard_rule": {"changed_top_factor": list(CHANGED_TOP_FACTOR),
                           "text": "(c) share of test rows with a new top code (after the NONE guard) between ½ and 2 times the OOF share, otherwise abort without a "
                                   "file; (b) share \"novel for the task\" among the first five reported only; no note guard (no note features)"},
            "inputs_sha256": oof["inputs"], "oof_reproduction": {"file": rel(OOF_JSON), "deploy_units": oof["deploy_units"], "target_units": TARGET_UNITS,
                                                                  "anchor_units": oof["anchor_units"]},
            "order": "blend -> inductive alignment (class weights from the training rows) -> second stage over the first K -> NONE guard -> REL-01"}


def cmd_fit(a) -> int:
    try:
        ctx = context()
    except Missing as e:
        abort(str(e), 3)
    if not OOF_JSON.is_file():
        abort(f"{rel(OOF_JSON)} missing — run python -I eda/c120_s036_safe_stage2.py oof first", 3)
    oof = json.loads(OOF_JSON.read_text())
    if oof.get("inputs") != inputs_sha256(ctx):
        abort(f"{rel(OOF_JSON)} was computed with different inputs — rerun c120 oof")
    if not oof.get("reproduction_ok") or not oof.get("null_point_exact"):
        abort("reproduction of the OOF numbers or copy null point failed — no deployment")
    out = build_stage2(ctx, oof)
    text = to_json(out)
    m = out["model"]
    print(f"  [{FEATURE_SET}] {len(m['features'])} features | gate (guarded) inner {m['gate']['inner']:.5f} vs. base {m['gate']['inner_base']:.5f} -> "
          f"{'ABSTAIN (weight 0)' if m['abstain'] else 'apply'} | in-sample {m['insample']['map_base']:.5f} → {m['insample']['map_reranked_protected']:.5f} "
          f"(new top code {m['insample']['changed_top_share']:.4f}; OOF {m['oof_changed_top_share']:.4f})")
    target = Path(a.out) if a.out else STAGE2_FILE
    if a.verify:
        if not target.is_file():
            abort(f"{rel(target)} missing — build it with --save first", 3)
        if target.read_text() != text:
            abort(f"{rel(target)} is not the recomputation")
        print(f"   VERIFIED: {rel(target)} = recomputation (SHA-256 {sha256_file(target)[:16]}…)")
    if a.save:
        if target.is_file() and target.read_text() != text:
            abort(f"{rel(target)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        target.write_text(text)
        print(f"   saved: {rel(target)} (SHA-256 {sha256_file(target)})")
    return 0


def null_stage2(source: Path = STAGE2_FILE) -> dict:
    """Null model (weight 0, standardisation 0/1) of the deployment file: the features are computed on the test side, the score is 0
    everywhere -> the stable sort keeps the base order, the guard changes nothing (copy null point). Guard (c) is not enforced."""
    ref = json.loads(Path(source).read_text())
    nf = len(ref["model"]["features"])
    model = {"features": ref["model"]["features"], "w": [0.0] * nf, "mu": [0.0] * nf, "sd": [1.0] * nf, "abstain": True, "oof_changed_top_share": 0.0}
    keep = ("kind", "stack", "stack_sha256", "channels", "arith_beta", "align_mode", "align_inductive", "align_min", "K", "lambda", "feature_set",
            "protect_none", "draws_channel", "draw_sources", "draw_rank_source", "test_draws", "novel_task_share_top5", "guard_rule")
    return {**{k: ref[k] for k in keep}, "null": True, "model": model, "null_from": rel(Path(source))}


def cmd_null(a) -> int:
    src = Path(a.source)
    if not src.is_file():
        abort(f"{rel(src)} missing — run c120 fit --save first", 3)
    Path(a.out).write_text(to_json(null_stage2(src)))
    print(f"null model of {rel(src)} -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("oof", help="nested reproduction (per-fold form; private JSON)")
    f = sub.add_parser("fit", help="build (--save) or recompute (--verify) the deployment file artifacts/stage2_s036_safe.json")
    f.add_argument("--out", help="output file (default artifacts/stage2_s036_safe.json)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    nl = sub.add_parser("null", help="write the null model (weight 0) of the deployment file")
    nl.add_argument("--out", required=True)
    nl.add_argument("--source", default=str(STAGE2_FILE))
    a = ap.parse_args(argv)
    return {"oof": cmd_oof, "fit": cmd_fit, "null": cmd_null}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
