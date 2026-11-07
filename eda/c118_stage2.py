"""SECOND STAGE of the primary line — β by log loss (blend file from eda/c110_q397_deploy.py) and a conditional logit over
the first 5 candidates after margin alignment (src/stage2.py). Build script: tools/builds/s032_stage2.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c118_stage2.py oof [--stack s032_stage2]                    # nested reproduction (three forms) -> private JSON
    python -I eda/c118_stage2.py fit --stack s032_stage2 (--save | --verify)  # deployment file artifacts/stage2_s032.json
    python -I eda/c118_stage2.py fit --stack s032_stage2 --guard-rule widened (--save | --verify)  # artifacts/stage2_s034.json
    python -I eda/c118_stage2.py null --stack <stem> --out <file> [--source <deployment file>]  # null model (weight 0) for the copy null points
    python -I eda/c118_stage2.py guard-oof [--stack s032_stage2]                   # nested main form + main_guard -> stage2_guard/
    python -I eda/c118_stage2.py guard-fit --stack s032_stage2 (--save | --verify)  # artifacts/stage2_s035.json

FORM (fixed before the evaluation): base recipe = six-channel recipe with q397m3 instead of q397, with β from the blend file
(argmin log loss before the last step on all 1,500 OOF rows, grid 0.1…0.8); reranker K = 5, λ = 0.1 FIXED, feature set "main" =
conditional logit without 397B notes and without 397B channel features (43 features), fallback "fallback" = "main" without
notes (40).

OOF (`oof`; an independent re-implementation of the model-sweep reranker with nested β): outer fold k = evaluated rows.
  * Training rows T (folds ≠ k): base forms from weights of the folds except {j, k}, β chosen there by log loss, last step per
    fold with the target from the folds except {j, k}; set form over T with the target from T. Knowledge of the rows from fold
    j: folds except {j, k}.
  * Evaluated rows (fold k): standard cross-fit (weights and β from the folds ≠ k); set over all 1,500 rows (convention
    c107.set_align). Knowledge: folds ≠ k.
  * Abstention: four inner folds of T (fit on three, evaluate on the fourth); abstain if the pooled inner MAP@3 is not above
    the base order. Placebo: targets of the training rows on random candidates (seed 1000 + k).
  * Comparison against the anchor of the earlier six-channel recipe (7,463 / 7,396 / 7,441 of 9,000), against the own base
    recipe (β 0.2) and against base recipe + β without the reranker; σ over tasks (c107.cluster_boot, 10,000 draws). Target
    values (independent cross-check): against the anchor +0.0083 / +0.0083 / +0.0078.
  * Share of rows with a new top code (form "set"; applied without abstention = reference of guard (c), plus with abstention)
    per feature set.
  * Checksum of the features of the evaluated rows in the set form (table row order) -> null point (e) in `fit`.

DEPLOYMENT (`fit`): training rows = all 1,500 in the deployment form: standard cross-fit of the recipe with the β of the blend
file, alignment as ONE set over 1,500 rows (c107.set_align); knowledge of a row of one fold ONLY from the other folds.
Abstention gate over the five folds. Weights, standardisation, feature list, β, K, λ, gate, OOF share of new top codes,
reference of the note guard (three training draws: mean number of codes mentioned, share mentioning the top code,
distribution of the mention count 0–3 for the first five, share of word-identical notes per pair of draws) and SHA-256 of the
inputs -> artifacts/stage2_s032.json (without timestamps; --verify requires character identity).
Null point (e): features of the deployment form == features of the evaluated rows of the OOF reproduction, bit-identical
(checksum), otherwise abort.

WIDENED RULE (`fit --guard-rule widened`): the same recomputation; it must be artifacts/stage2_s032.json character for character
(otherwise abort). Afterwards ONLY guard_rule changes — extended by the widened note guard (share_z_max 3, n_ref = number of
training rows, rule, rule_text; the old fields stay) — and the provenance field derived_from is added (file and SHA-256 of
stage2_s032.json). Output artifacts/stage2_s034.json (or --out); stage2_s032.json is never written with this rule.

FEATURE SET main_guard (`guard-oof`, `guard-fit`; the commands above are unchanged): main form (43 features) + identifiers of the
five general codes (48), together with the NONE guard (src/stage2.protect_none: base top NONE -> the base order stays); otherwise
nested exactly like the main form (same folds, β choice, K 5, λ 0.1; the gate evaluates WITH the guard).
  * guard-oof: main form and main_guard in three forms; the main form must reproduce the row values of stage2/c118_oof_rows.npz
    bit for bit (otherwise abort); copy null point (i): identifiers = 0 and guard off -> row values and orders of the main form
    bit-identical; Δ against the main form (σ over tasks), share of new top codes after the guard (reference of guard (c)),
    checksum of the evaluated features (form "set") -> artifacts/eda/private/stage2_guard/c118_guard_oof.json and
    c118_guard_oof_rows.npz (new files).
  * guard-fit: first recomputes the deployment file under the widened rule, which must be artifacts/stage2_s034.json character
    for character; then replaces ONLY models.main by the main_guard model (fields feature_set, protect_none; null point (e)
    against guard-oof) and appends guard_build and derived_from. Fallback model, reference and rule of the note guard, β, K, λ
    stay those of stage2_s034.json -> artifacts/stage2_s035.json.

Training rows only; only statistics are printed. Notes are only counted by machine (regex of the codes), no text is printed.
Exit 3 = input missing; 4 = checksum/recomputation/null point violated.
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C115 = _load("c118_c115", COMP / "eda" / "c115_q397m3_eval.py")  # context, spec_base, replaced, run_recipe
C110, C107, A9 = C115.C110, C115.C107, C115.A9
MS, A2, C21 = C107.MS, C107.A2, C107.C21
FRZ = _load("c118_frz", COMP / "tools" / "freeze_m3_texts.py")

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "stage2"
STACK_DEFAULT = "s032_stage2"
STAGE2_FILE = ART / "stage2_s032.json"
STAGE2_WIDENED = ART / "stage2_s034.json"  # the same deployment file under the widened note guard
WIDENED_SHARE_Z_MAX = 3.0  # ≈ Bonferroni for 13 statistics at α 0.05
# text of the widened rule, recorded in the deployment file (guard_rule.rule_text)
WIDENED_RULE_TEXT = ("Widened rule: a statistic of the note guard (a) passes if it lies in the old band OR — for shares only "
             "(share_top_mentioned per draw against the training value of the same draw, mention_hist_top5 per level against the training "
             "value of the same level, word-identical notes per pair against the pair's own training value) — the two-sided two-sample "
             "binomial test gives |z| ≤ share_z_max (pooled share, n = number of rows: training n_ref, test the number of test rows; rows "
             "also for the histogram levels). codes_per_row keeps the old band. The rule can only widen, never narrow.")
OOF_JSON = OUT / "c118_oof.json"
REPLACE, OLD = "q397m3", "q397"
M3 = "m3x3_full"
FORMS = ("fold", "pre", "set")
FORM_LABEL = {"fold": "per fold", "pre": "without the last step", "set": "as one set"}
ANCHOR_UNITS = {"fold": 7463, "pre": 7396, "set": 7441}  # earlier six-channel recipe: 0.82922 / 0.82178 / 0.82678
TARGET_VS_ANCHOR = {"fold": 0.00833, "pre": 0.00833, "set": 0.00778}  # independent cross-check (λ 0.1)
TOLERANCE = 0.0005
PROTO_ROWS = ART / "eda" / "private" / "stage2" / "crosscheck_rows.npz"  # row values of the independent cross-check (reference only)
TEST_NOTES = tuple(f"derived/sft/m3_d{d}_test_frozen_notes.jsonl" for d in (1, 2, 3))
TEST_RANKS = ("runs/m3_fullte_rp/m3_test_rank/test", "runs/m3d2te_rp/m3d2_test_rank/test", "runs/m3d3te_rp/m3d3_test_rank/test")
PLACEBO_SEED = 1000
GUARD_OUT = ART / "eda" / "private" / "stage2_guard"  # own outputs of the main_guard form; the files of the main and fallback forms stay untouched
GUARD_OOF_JSON = GUARD_OUT / "c118_guard_oof.json"
GUARD_OOF_ROWS = GUARD_OUT / "c118_guard_oof_rows.npz"
STAGE2_GUARD = ART / "stage2_s035.json"  # main form + NONE guard + general-code identifiers, widened note guard


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

def train_note_texts(spec: str, folds: np.ndarray) -> list[str]:
    """Note (field text of the generation, frozen as in tools/freeze_m3_texts.freeze) per training row of one M3 draw; shards with
    +, combined_from resolved; every shard covers exactly one whole fold (fold integrity). The same resolution as for the findings,
    but with the note field instead of the finding."""
    texts: dict[int, str] = {}
    for shard in str(spec).split("+"):
        run, job, tag = A9.parse_source(shard)
        meta = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json")
        if meta is None:
            raise Missing(f"{shard}: meta file missing")
        parts = [(run, job, meta)]
        if meta.get("combined_from"):
            parts = [(s.get("run", run), s["job"], A9.read_json(A9.run_dir(s.get("run", run)) / s["job"] / f"{tag}_meta.json"))
                     for s in meta["combined_from"]]
        for r, j, m in parts:
            src = (m or {}).get("sft", {}).get("sft_from")
            if not src or "/" not in str(src):
                abort(f"{r}/{j}: sft_from missing")
            gj, gt = str(src).split("/", 1)
            gen, rank = A9.run_dir(r) / gj / f"{gt}_notes.jsonl", A9.run_dir(r) / j / f"{tag}_proto_rank.jsonl"
            if not gen.is_file() or not rank.is_file():
                raise Missing(f"{r}/{j}: notes or ranking missing")
            recs, cnt = FRZ.freeze(gen, rank)
            if cnt["table"] != "train":
                abort(f"{r}/{j}: table {cnt['table']}")
            rows = np.array(sorted(int(x["row"]) for x in recs))
            fs = sorted(set(int(v) for v in folds[rows]))
            if len(fs) != 1 or not np.array_equal(rows, np.flatnonzero(folds == fs[0])):
                abort(f"{r}/{j}: rows do not cover exactly one whole fold (folds {fs})")
            for x in recs:
                i = int(x["row"])
                if i in texts:
                    abort(f"{spec}: duplicate row")
                texts[i] = str(x.get("text") or "")
    if sorted(texts) != list(range(len(folds))):
        abort(f"{spec}: {len(texts)} of {len(folds)} rows")
    return [texts[i] for i in range(len(folds))]


def context(stack_stem: str = STACK_DEFAULT) -> dict:
    """Training context: labels, folds, channels (OOF before the floor), three M3 draws, notes, blend file and recipe."""
    ctx, mats, sha16 = C115.context()
    f_r = ART / f"oof_{REPLACE}.npy"
    if not f_r.is_file():
        raise Missing(f"{rel(f_r)} missing")
    mats[REPLACE] = np.load(f_r)
    sha16[REPLACE] = sha256_file(f_r)[:16]
    meta = json.loads((ART / f"meta_{M3}.json").read_text())
    specs = [d["source"] for d in meta["rank"]["draws"]]
    if len(specs) != 3:
        abort(f"meta_{M3}.json: {len(specs)} draws instead of 3")
    n, K, seen = ctx["n"], ctx["K"], ctx["seen"]
    draws = []
    for s in specs:
        P, have, _ = A9.load_draw(s, n, seen)
        if not have.all():
            abort(f"{s}: draw not on all rows")
        draws.append(P)
    Mean, _ = A9.mean_of_draws([(P, np.ones(n, bool)) for P in draws], n, K)
    if not np.array_equal(Mean, mats[M3]):
        abort(f"mean of the three draws != oof_{M3}.npy")
    texts = [train_note_texts(s, ctx["folds"]) for s in specs]
    notes = S2.notes_features(texts, ctx["classes"])
    stack_file = ART / f"stack_{stack_stem}.json"
    if not stack_file.is_file():
        raise Missing(f"{rel(stack_file)} missing — run eda/c110_q397_deploy.py stack … --stem {stack_stem} --save first")
    stack = json.loads(stack_file.read_text())
    spec = C115.replaced(C115.spec_base(), OLD, REPLACE)
    floors = {k: float(v) for k, v in spec["floors"].items() if v}
    if stack.get("channels") != list(spec["names"]) or {k: float(v) for k, v in (stack.get("floor") or {}).items()} != floors:
        abort(f"{stack_file.name}: channels/floors {stack.get('channels')} / {stack.get('floor')} do not match the base recipe {spec['names']}")
    if (stack.get("scaled") or {}) != {spec["think"]: float(spec["factor"])}:
        abort(f"{stack_file.name}: reasoning factor {stack.get('scaled')}")
    beta = float(stack.get("arith_beta") or 0.0)
    train = ctx["train"]
    ctx.update(mats=mats, sha16=sha16, draw_specs=specs, draws=draws, note_texts=texts, notes=notes, stack=stack, stack_file=stack_file,
               stack_stem=stack_stem, spec=spec, beta=beta, key=S2.task_keys(ctx["qid"], ctx["topic"]),
               topic_s=np.asarray(ctx["topic"]).astype(str), grade=train["grade"].to_numpy().astype(float))
    ctx["static"] = S2.Static(spec["names"], mats, draws, ctx["grade"], seen, ctx["classes"], notes)
    return ctx


def inputs_sha256(ctx: dict) -> dict:
    """Checksums of all inputs of the training side."""
    root = ctx["root"]
    out = {f"oof_{nm}.npy": sha256_file(ART / f"oof_{nm}.npy") for nm in ctx["spec"]["names"]}
    out[f"meta_{M3}.json"] = sha256_file(ART / f"meta_{M3}.json")
    out["stack"] = {"file": rel(ctx["stack_file"]), "sha256": sha256_file(ctx["stack_file"])}
    out["folds.npy"] = sha256_file(ART / "folds.npy")
    out["train.csv"] = sha256_file(root / "train.csv")
    out["taxonomy_v1.csv"] = sha256_file(root / "competition_data" / "taxonomy_v1.csv")
    out["train_notes_texts"] = [hashlib.sha256("\x1e".join(t).encode("utf-8")).hexdigest() for t in ctx["note_texts"]]
    return out


# ------------------------------------------------------------------------------------------------ recipe (cross-fit, nested)

def fit_apply(spec: dict, L, y, seen, fit_idx, ev_idx, beta) -> tuple[np.ndarray, np.ndarray, float]:
    """Weights on log loss from fit_idx (a2.fit_mix, then the reasoning factor), β fixed or "select" (argmin log loss of the blend
    on fit_idx, grid 0.1…0.8), blend on ev_idx, margin alignment 0.75 (classes with at least 3 rows) with the target from fit_idx.
    -> (P, Q, β)."""
    names = list(spec["names"])
    ti = names.index(spec["think"]) if spec.get("think") in names else None
    th = np.asarray(A2.fit_mix([lg[fit_idx] for lg in L], None, y[fit_idx]), dtype=float).copy()
    if ti is not None:
        th[ti] *= float(spec["factor"])
    if beta == "select":
        best = None
        for b in S2.BETA_GRID:
            PF = MS.mix_channels({"theta": [float(v) for v in th], "arith_beta": b}, [lg[fit_idx] for lg in L], None, seen)
            ll = float(-np.log(np.clip(PF[np.arange(len(fit_idx)), y[fit_idx]], 1e-12, 1)).mean())
            if best is None or ll < best[0] - 1e-12:
                best = (ll, b)
        beta = best[1]
    st = {"theta": [float(v) for v in th], "arith_beta": float(beta)}
    PE = MS.mix_channels(st, [lg[ev_idx] for lg in L], None, seen)
    QE = C107.apply_last(PE, C21.fit_weights(PE, C107.target_of(y, fit_idx, len(seen))), seen)
    return PE, QE, float(beta)


def standard_forms(ctx: dict, beta) -> tuple[dict, list[float]]:
    """Standard cross-fit: per fold k weights (and β) from the folds ≠ k; set over all rows (c107.set_align)."""
    y, seen, folds, n, K = ctx["y"], ctx["seen"], ctx["folds"], ctx["n"], ctx["K"]
    spec = ctx["spec"]
    L = C107.logs_of(ctx["mats"], spec["names"], spec["floors"], seen, False)
    P, Q, betas = np.zeros((n, K)), np.zeros((n, K)), []
    for k in range(5):
        F, E = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        P[E], Q[E], b = fit_apply(spec, L, y, seen, F, E, beta)
        betas.append(b)
    return {"fold": Q, "pre": P, "set": C107.set_align(P, y, seen)}, betas


def nested_forms(ctx: dict, beta, k: int, L) -> dict:
    """Base forms of the training rows T (folds ≠ k) without any label from fold k."""
    y, seen, folds, n, K = ctx["y"], ctx["seen"], ctx["folds"], ctx["n"], ctx["K"]
    T = np.flatnonzero(folds != k)
    P, Q = np.zeros((n, K)), np.zeros((n, K))
    for j in range(5):
        if j == k:
            continue
        F = np.flatnonzero((folds != j) & (folds != k))
        E = np.flatnonzero(folds == j)
        P[E], Q[E], _ = fit_apply(ctx["spec"], L, y, seen, F, E, beta)
    S = np.zeros((n, K))
    S[T] = C107.apply_last(P[T], C21.fit_weights(P[T], C107.target_of(y, T, K)), seen)
    return {"fold": Q, "pre": P, "set": S}


def deploy_forms(ctx: dict) -> dict:
    """Deployment form of the training rows: standard cross-fit with the β of the blend file (c107.cross_fit), set over all 1,500 rows."""
    spec = {**ctx["spec"], "beta": ctx["beta"]}
    fit = C107.cross_fit(ctx["y"], ctx["folds"], ctx["seen"], ctx["mats"], spec)
    return {"fold": fit["Q"], "pre": fit["P"], "set": C107.set_align(fit["P"], ctx["y"], ctx["seen"])}


# ------------------------------------------------------------------------------------------------ features per fold block

def block(ctx: dict, rows: np.ndarray, FM: np.ndarray, PM: np.ndarray, feature_set: str, exclude=()) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features of the rows `rows` of ONE fold; knowledge from the other folds except `exclude`."""
    kn = S2.knowledge_other_folds(ctx["key"], ctx["topic_s"], ctx["y"], ctx["folds"], rows, ctx["K"], exclude)
    return S2.pair_features(ctx["static"], rows, FM, PM, kn, feature_set)


def deploy_features(ctx: dict, forms: dict, feature_set: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair features of all 1,500 training rows in the deployment form, table row order -> (X [n·K, f], candidates [n, K], names)."""
    n, Kc = ctx["n"], S2.K_CAND
    X, cand, names = None, np.zeros((n, Kc), np.int64), None
    for k in range(5):
        E = np.flatnonzero(ctx["folds"] == k)
        Xk, ck, names = block(ctx, E, forms["set"][E], forms["pre"][E], feature_set)
        if X is None:
            X = np.zeros((n * Kc, Xk.shape[1]))
        pos = (E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)
        X[pos] = Xk
        cand[E] = ck
    return X, cand, names


def fold_blocks(ctx: dict, X: np.ndarray, cand: np.ndarray, lab: np.ndarray) -> list[tuple]:
    """Five blocks (per fold, table row order) for the abstention gate."""
    Kc, out = S2.K_CAND, []
    for k in range(5):
        E = np.flatnonzero(ctx["folds"] == k)
        pos = (E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)
        out.append((X[pos], lab[pos], cand[E], ctx["y"][E]))
    return out


# ------------------------------------------------------------------------------------------------ nested reproduction

def run_nested(ctx: dict, std: dict, nested: dict, form: str, feature_set: str, placebo: bool = False, ids_zero: bool = False) -> dict:
    """Like the model-sweep reranker with exactly one setting (conditional logit, λ 0.1, K 5): per outer fold k fit on T (knowledge
    of the rows from fold j: folds except {j, k}), abstention gate over four inner folds, application to fold k. -> row values.
    main_guard applies the NONE guard after the reordering (also in the gate). ids_zero=True (main_guard only): identifiers = 0 and
    guard off — the copy null point (i), which must reproduce the main form bit for bit. For main/fallback everything is as before."""
    guard = feature_set == S2.GUARD_SET
    if ids_zero and not guard:
        raise ValueError("ids_zero only with the feature set main_guard")
    protect = guard and not ids_zero

    def prep(X: np.ndarray, names: list[str]) -> np.ndarray:
        if not ids_zero:
            return X
        X = X.copy()
        X[:, [names.index(nm) for nm in S2.GENERAL_ID_FEATS]] = 0.0
        return X

    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND
    forced, abst, base = np.zeros(n), np.zeros(n), np.zeros(n)
    changed, changed_forced, changed3 = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    top_all, cand_all, protected = np.zeros((n, Kc), np.int64), np.zeros((n, Kc), np.int64), np.zeros(n, bool)
    choice, X_eval = {}, {}
    for k in range(5):
        forms_T = nested[k]
        E = np.flatnonzero(folds == k)
        Tfolds = [j for j in range(5) if j != k]
        XT, candT, rowsT = [], [], []
        for j in Tfolds:
            Rj = np.flatnonzero(folds == j)
            X, cand, names_j = block(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], feature_set, exclude=(k,))
            XT.append(prep(X, names_j))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = block(ctx, E, std[form][E], std["pre"][E], feature_set)
        XE = prep(XE, names)
        X_eval[k] = XE
        labT = [S2.labels_of(c, y[r]) for c, r in zip(candT, rowsT)]
        if placebo:
            rng = np.random.default_rng(PLACEBO_SEED + k)
            labT = [S2.permute_labels(lb, Kc, rng) for lb in labT]
        g = S2.gate([(XT[b], labT[b], candT[b], y[rowsT[b]]) for b in range(len(Tfolds))], protect=protect)
        model = S2.fit_clogit(np.concatenate(XT), np.concatenate(labT), Kc, S2.LAMBDA)
        topE = S2.reorder(candE, S2.scores(model, XE, Kc))
        if protect:  # the NONE guard belongs to the feature set
            top0 = topE
            topE = S2.protect_none(topE, candE)
            protected[E] = (top0 != topE).any(1)
        rr, bb = S2.map_rows(topE, y[E]), S2.map_rows(candE, y[E])
        forced[E], abst[E], base[E] = rr, (bb if g["abstain"] else rr), bb
        changed_forced[E] = topE[:, 0] != candE[:, 0]
        changed3[E] = (topE[:, :3] != candE[:, :3]).any(1)
        changed[E] = changed_forced[E] & (not g["abstain"])
        top_all[E], cand_all[E] = topE, candE
        choice[k] = g
    return {"forced": forced, "abstain": abst, "base": base, "changed_top": changed, "changed_top_forced": changed_forced, "changed_top3_forced": changed3, "choice": choice,
            "features": names, "X_eval": X_eval, "top": top_all, "cand": cand_all, "protected": protected}


def eval_rows_in_table_order(ctx: dict, X_eval: dict) -> np.ndarray:
    Kc = S2.K_CAND
    X = np.zeros((ctx["n"] * Kc, next(iter(X_eval.values())).shape[1]))
    for k, Xk in X_eval.items():
        E = np.flatnonzero(ctx["folds"] == k)
        X[(E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)] = Xk
    return X


def boot(d: np.ndarray, ctx: dict, draws: np.ndarray) -> dict:
    cb = C107.cluster_boot(np.asarray(d, float), ctx["q_code"], draws)
    return {"delta": cb["mean"], "se_q": cb["se_q"], "z": cb["z_q"], "units": int(np.rint(6 * np.sum(d))),
            "fold_units": [int(np.rint(6 * d[ctx["folds"] == k].sum())) for k in range(5)],
            "folds_pos": int(sum(np.rint(6 * d[ctx["folds"] == k].sum()) > 0 for k in range(5)))}


def cmd_oof(a) -> int:
    t0 = time.time()
    try:
        ctx = context(a.stack)
    except Missing as e:
        abort(str(e), 3)
    y, seen = ctx["y"], ctx["seen"]
    draws_q = C107.question_draws(ctx["q_code"])
    # anchor of the earlier six-channel recipe and the own base recipe (β 0.2), three forms each
    base_b = C115.run_recipe(ctx, ctx["mats"], C115.spec_base())
    anchor = {"fold": base_b["rows"], "pre": base_b["rows_pre"], "set": base_b["rows_set"]}
    for f in FORMS:
        u = int(C107.units(anchor[f]).sum())
        if u != ANCHOR_UNITS[f]:
            abort(f"anchor [{f}] {u}/9000 instead of {ANCHOR_UNITS[f]}")
    own = C115.run_recipe(ctx, ctx["mats"], ctx["spec"])  # base recipe with β 0.2
    own_rows = {"fold": own["rows"], "pre": own["rows_pre"], "set": own["rows_set"]}
    std, betas = standard_forms(ctx, "select")
    if not np.allclose(betas, ctx["beta"]) or len(set(betas)) != 1:
        print(f"NOTE: β nested per outer fold {betas}, blend file {ctx['beta']:g} — deployment features may differ from the cross-fit",
              flush=True)
    rec_rows = {f: C107.rows_map(y, std[f], seen) for f in FORMS}
    L = C107.logs_of(ctx["mats"], ctx["spec"]["names"], ctx["spec"]["floors"], seen, False)
    nested = {k: nested_forms(ctx, "select", k, L) for k in range(5)}
    res = {"stack": rel(ctx["stack_file"]), "stack_sha256": sha256_file(ctx["stack_file"]), "beta_outer": betas, "beta_stack": ctx["beta"],
           "K": S2.K_CAND, "lambda": S2.LAMBDA, "anchor_map": {f: float(anchor[f].mean()) for f in FORMS},
           "own_recipe_map": {f: float(own_rows[f].mean()) for f in FORMS}, "recipe_beta_map": {f: float(rec_rows[f].mean()) for f in FORMS},
           "beta_only": {"vs_anchor": {f: boot(rec_rows[f] - anchor[f], ctx, draws_q) for f in FORMS},
                         "vs_own_recipe": {f: boot(rec_rows[f] - own_rows[f], ctx, draws_q) for f in FORMS}},
           "sets": {}, "inputs": inputs_sha256(ctx)}
    rows_out = {f"anchor_{f}": anchor[f] for f in FORMS} | {f"own_{f}": own_rows[f] for f in FORMS} | {f"recipe_beta_{f}": rec_rows[f] for f in FORMS}
    proto = np.load(PROTO_ROWS) if PROTO_ROWS.is_file() else None
    runs = [(fs, False) for fs in a.feature_sets.split(",")] + ([("main", True)] if a.placebo else [])
    for fs, placebo in runs:
        key = fs + ("+placebo" if placebo else "")
        res["sets"][key] = {}
        for form in FORMS:
            r = run_nested(ctx, std, nested, form, fs, placebo)
            if not np.allclose(r["base"], rec_rows[form]):
                abort(f"base of the candidate order [{key}|{form}] != recipe")
            ent = {"map_forced": float(r["forced"].mean()), "map_abstain": float(r["abstain"].mean()),
                   "abstain_folds": int(sum(c["abstain"] for c in r["choice"].values())),
                   "gate": {str(k): v for k, v in r["choice"].items()}, "n_features": len(r["features"]),
                   "changed_top_share": float(r["changed_top"].mean()), "changed_top_share_forced": float(r["changed_top_forced"].mean()),
                   "changed_top3_share_forced": float(r["changed_top3_forced"].mean())}
            for mode in ("forced", "abstain"):
                ent[mode] = {"vs_anchor": boot(r[mode] - anchor[form], ctx, draws_q), "vs_own_recipe": boot(r[mode] - own_rows[form], ctx, draws_q),
                             "vs_recipe_beta": boot(r[mode] - rec_rows[form], ctx, draws_q)}
            if form == "set" and not placebo:
                Xe = eval_rows_in_table_order(ctx, r["X_eval"])
                ent["eval_features_sha256"] = sha256_array(Xe)
                ent["features"] = r["features"]
            if proto is not None and fs == "main" and not placebo and f"noq397m3ch_{form}" in proto.files:
                pr = proto[f"noq397m3ch_{form}"]
                ent["prototype_rows_equal"] = bool(np.array_equal(pr, r["forced"]))
                ent["prototype_rows_max_abs"] = float(np.abs(pr - r["forced"]).max())
            res["sets"][key][form] = ent
            rows_out[f"{key}_{form}_forced"] = r["forced"]
            rows_out[f"{key}_{form}_abstain"] = r["abstain"]
            va, vo, vb = ent["forced"]["vs_anchor"], ent["forced"]["vs_own_recipe"], ent["forced"]["vs_recipe_beta"]
            print(f"  [{key:13s}|{form:4s}] {ent['n_features']} features | vs. anchor {va['delta']:+.5f} ± {va['se_q']:.5f} {va['fold_units']} | "
                  f"vs. base recipe {vo['delta']:+.5f} ± {vo['se_q']:.5f} | vs. base recipe+β {vb['delta']:+.5f} ± {vb['se_q']:.5f} | abstention "
                  f"{ent['abstain_folds']}/5 → vs. anchor {ent['abstain']['vs_anchor']['delta']:+.5f} | new top code {ent['changed_top_share']:.4f}"
                  + (f" | prototype bit-identical {ent['prototype_rows_equal']}" if "prototype_rows_equal" in ent else ""), flush=True)
    main = res["sets"].get("main")
    if main:
        dev = {f: abs(main[f]["forced"]["vs_anchor"]["delta"] - TARGET_VS_ANCHOR[f]) for f in FORMS}
        res["reproduction"] = {"target_vs_anchor": TARGET_VS_ANCHOR, "abs_deviation": dev, "tolerance": TOLERANCE,
                               "ok": bool(all(v <= TOLERANCE for v in dev.values()))}
        print(f"REPRODUCTION against the independent cross-check (target {TARGET_VS_ANCHOR}): deviation {', '.join(f'{f} {v:.5f}' for f, v in dev.items())} -> "
              f"{'PASSED' if res['reproduction']['ok'] else 'FAILED'} (tolerance {TOLERANCE})", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    OOF_JSON.write_text(to_json(res))
    np.savez_compressed(OUT / "c118_oof_rows.npz", **rows_out)
    print(f"written: {rel(OOF_JSON)} ({res['seconds']} s)")
    if main and not res["reproduction"]["ok"]:
        return 4
    return 0


# ------------------------------------------------------------------------------------------------ deployment file

def guard_reference(ctx: dict, cand: np.ndarray) -> dict:
    """Reference of the note guard from the three training draws (deployment form; aggregates only)."""
    st = S2.note_stats(ctx["notes"]["men_per_draw"], ctx["notes"]["m3_men"], cand, cand[:, 0])
    t = ctx["note_texts"]
    st["identical"] = {"d1-d2": S2.identical_share(t[0], t[1]), "d1-d3": S2.identical_share(t[0], t[2]), "d2-d3": S2.identical_share(t[1], t[2])}
    return st


def build_stage2(ctx: dict, oof: dict) -> tuple[dict, dict]:
    """Deployment file (without timestamps) and log."""
    y, Kc = ctx["y"], S2.K_CAND
    forms = deploy_forms(ctx)
    cv_units = int(C107.units(C107.rows_map(y, forms["fold"], ctx["seen"])).sum())
    want_cv = int(round(float(ctx["stack"]["cv"]) * 6 * ctx["n"]))
    if cv_units != want_cv:
        abort(f"cross value of the deployment form {cv_units}/9000 does not match the field cv of the blend file ({want_cv})")
    models, rep = {}, {"sets": {}}
    for fs in S2.FEATURE_SETS:
        X, cand, names = deploy_features(ctx, forms, fs)
        lab = S2.labels_of(cand, y)
        want = (oof.get("sets", {}).get(fs, {}).get("set") or {}).get("eval_features_sha256")
        got = sha256_array(X)
        if want != got:
            abort(f"null point (e) violated: features of the deployment form [{fs}] ({got[:16]}…) != evaluated rows of the OOF reproduction "
                  f"({str(want)[:16]}…) — rerun c118 oof with the same blend file")
        g = S2.gate(fold_blocks(ctx, X, cand, lab))
        model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
        # OOF share of new top codes for guard (c): the applied form (without abstention), since the deployment gate applies; if it
        # abstains, the model has weight 0 and the guard deliberately raises the alarm (to be checked manually)
        top = S2.reorder(cand, S2.scores(model, X, Kc))
        ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked": float(S2.map_rows(top, y).mean()),
               "changed_top_share": float((top[:, 0] != cand[:, 0]).mean())}
        o = oof["sets"][fs]["set"]
        models[fs] = {"features": names, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
                      "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
                      "oof_changed_top_share": float(o["changed_top_share_forced"]),
                      "oof_changed_top_share_with_abstention": float(o["changed_top_share"]),
                      "oof_changed_top3_share": float(o["changed_top3_share_forced"]),
                      "oof_vs_anchor": {f: oof["sets"][fs][f]["forced"]["vs_anchor"]["delta"] for f in FORMS},
                      "oof_vs_own_recipe": {f: oof["sets"][fs][f]["forced"]["vs_own_recipe"]["delta"] for f in FORMS},
                      "features_sha256": got, "insample": ins}
        if fs == "main":
            rep["guard_reference"] = guard_reference(ctx, cand)
            rep["novel_task_share_top5"] = float(X[:, names.index("novel_task")].mean())
        rep["sets"][fs] = {"gate": g, "insample": ins, "n_features": len(names)}
    st = ctx["stack"]
    meta = json.loads((ART / f"meta_{M3}.json").read_text())
    out = {"kind": "second stage: conditional logit over the first K candidates after the margin alignment (src/stage2.py)",
           "stack": rel(ctx["stack_file"]), "stack_sha256": sha256_file(ctx["stack_file"]), "channels": list(st["channels"]),
           "arith_beta": ctx["beta"], "align": 0.75, "align_min": 3, "K": Kc, "lambda": S2.LAMBDA, "null": False,
           "draws_channel": M3, "draw_sources": ctx["draw_specs"], "draw_rank_source": meta["rank"]["source"],
           "test_notes": [{"file": f, "rank": r} for f, r in zip(TEST_NOTES, TEST_RANKS)],
           "models": models, "guard_reference": rep["guard_reference"], "novel_task_share_top5": rep["novel_task_share_top5"],
           "guard_rule": {"notes_rel_tolerance": 0.10, "changed_top_factor": [0.5, 2.0],
                          "text": "(a) note statistics of the test draws within the range of the three training draws ± 10 % relative, otherwise "
                                  "feature set fallback; (c) share of test rows with a new top code between ½ and 2 times the OOF share of the chosen "
                                  "set, otherwise abort"},
           "inputs_sha256": oof["inputs"], "oof_reproduction": {"file": rel(OOF_JSON), "reproduction": oof.get("reproduction"),
                                                                 "beta_outer": oof.get("beta_outer")},
           "order": "blend -> margin alignment (one set over the test rows) -> reranker over the first K -> REL-01 for blank photos "
                    "(note guard and NONE band are abort guards before it)"}
    return out, rep


def with_widened_rule(out: dict, n_ref: int) -> dict:
    """Deployment file under the widened note guard. The recomputation `out` (band rule) must be artifacts/stage2_s032.json
    character for character; afterwards ONLY guard_rule changes (the old fields stay, the widened-rule fields are added), and the
    provenance field derived_from is appended. Weights, standardisation, feature lists, β, K, λ, gate, OOF shares, reference and
    checksums stay."""
    if not STAGE2_FILE.is_file():
        raise Missing(f"{rel(STAGE2_FILE)} missing — run c118 fit --stack … --save first (band rule)")
    if STAGE2_FILE.read_text() != to_json(out):
        abort(f"the recomputation (band rule) is not {rel(STAGE2_FILE)} character for character — the widened rule only applies to the unchanged deployment file")
    new = {}
    for k, v in out.items():
        new[k] = ({**v, "rule": "widened", "share_z_max": WIDENED_SHARE_Z_MAX, "n_ref": int(n_ref), "rule_text": WIDENED_RULE_TEXT}
                  if k == "guard_rule" else v)
    new["derived_from"] = {"file": rel(STAGE2_FILE), "sha256": sha256_file(STAGE2_FILE), "differs_only_in": ["guard_rule", "derived_from"],
                           "build": "tools/builds/s034_stage2_main.sh"}
    return new


def cmd_fit(a) -> int:
    target = Path(a.out) if a.out else (STAGE2_WIDENED if a.guard_rule == "widened" else STAGE2_FILE)
    if a.guard_rule == "widened" and target.resolve() == STAGE2_FILE.resolve():
        abort(f"--guard-rule widened never writes {rel(STAGE2_FILE)} — separate output (default {rel(STAGE2_WIDENED)})", 2)
    try:
        ctx = context(a.stack)
    except Missing as e:
        abort(str(e), 3)
    if not OOF_JSON.is_file():
        abort(f"{rel(OOF_JSON)} missing — run python -I eda/c118_stage2.py oof --stack {a.stack} first", 3)
    oof = json.loads(OOF_JSON.read_text())
    if oof.get("stack_sha256") != sha256_file(ctx["stack_file"]):
        abort(f"{rel(OOF_JSON)} was computed with a different blend file — rerun c118 oof")
    if oof.get("inputs") != inputs_sha256(ctx):
        abort(f"{rel(OOF_JSON)} was computed with different inputs — rerun c118 oof")
    if not (oof.get("reproduction") or {}).get("ok"):
        abort("reproduction of the OOF numbers failed — no deployment")
    out, rep = build_stage2(ctx, oof)
    if a.guard_rule == "widened":
        try:
            out = with_widened_rule(out, ctx["n"])
        except Missing as e:
            abort(str(e), 3)
        print(f"  rule of the note guard: {out['guard_rule']['rule']} (|z| ≤ {out['guard_rule']['share_z_max']:g}, n_ref {out['guard_rule']['n_ref']}); "
              f"all other fields = {rel(STAGE2_FILE)} (SHA-256 {out['derived_from']['sha256'][:16]}…)")
    text = to_json(out)
    for fs in S2.FEATURE_SETS:
        m, r = out["models"][fs], rep["sets"][fs]
        print(f"  [{fs:8s}] {r['n_features']} features | inner gate {r['gate']['inner']:.5f} vs. base {r['gate']['inner_base']:.5f} -> "
              f"{'ABSTAIN (weight 0)' if m['abstain'] else 'apply'} | in-sample {r['insample']['map_base']:.5f} → {r['insample']['map_reranked']:.5f} "
              f"(new top code {r['insample']['changed_top_share']:.4f}; OOF {m['oof_changed_top_share']:.4f})")
    g = out["guard_reference"]
    print("  reference of the note guard (training): " + "; ".join(f"{d} codes/row {v['codes_per_row']:.3f}, top mentioned {v['share_top_mentioned']:.3f}"
                                                           for d, v in g["per_draw"].items())
          + f"; mention count 0–3 for the first 5 {[round(v, 4) for v in g['mention_hist_top5']]}; word-identical {g['identical']}")
    print(f"  share \"novel for the task\" among the first 5 (OOF, knowledge from the other folds): {out['novel_task_share_top5']:.4f}")
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


def null_stage2(stack_stem: str, source: Path = STAGE2_FILE) -> dict:
    """Null model (weight 0, standardisation 0/1) for a blend file: features are computed on the test side, the score is 0
    everywhere -> the stable sort keeps the base order (copy null point). Guards are computed but not enforced."""
    stack_file = ART / f"stack_{stack_stem}.json"
    if not stack_file.is_file():
        raise Missing(f"{rel(stack_file)} missing")
    st = json.loads(stack_file.read_text())
    ref = json.loads(Path(source).read_text())
    models = {fs: {"features": ref["models"][fs]["features"], "w": [0.0] * len(ref["models"][fs]["features"]),
                   "mu": [0.0] * len(ref["models"][fs]["features"]), "sd": [1.0] * len(ref["models"][fs]["features"]),
                   "abstain": True, "oof_changed_top_share": 0.0} for fs in S2.FEATURE_SETS}
    for fs in S2.FEATURE_SETS:  # a model with its own feature set (main_guard) keeps it; without these fields bit-identical as before
        for key in ("feature_set", "protect_none"):
            if key in ref["models"][fs]:
                models[fs][key] = ref["models"][fs][key]
    return {**{k: ref[k] for k in ("kind", "align", "align_min", "K", "lambda", "draws_channel", "draw_sources", "draw_rank_source", "test_notes",
                                    "guard_reference", "novel_task_share_top5", "guard_rule")},
            "stack": rel(stack_file), "stack_sha256": sha256_file(stack_file), "channels": list(st["channels"]),
            "arith_beta": float(st.get("arith_beta") or 0.0), "null": True, "models": models, "null_from": rel(Path(source))}


def cmd_null(a) -> int:
    try:
        out = null_stage2(a.stack, Path(a.source))
    except Missing as e:
        abort(str(e), 3)
    Path(a.out).write_text(to_json(out))
    print(f"null model for stack_{a.stack}.json -> {a.out}")
    return 0


# ------------------------------------------------------------------------------------------------ feature set main_guard

def cmd_guard_oof(a) -> int:
    """Nested reproduction of the main form + main_guard (with the NONE guard) in three forms and copy null point (i); new files in stage2_guard/."""
    t0 = time.time()
    try:
        ctx = context(a.stack)
    except Missing as e:
        abort(str(e), 3)
    ref_file = OUT / "c118_oof_rows.npz"
    if not ref_file.is_file() or not OOF_JSON.is_file():
        abort(f"{rel(ref_file)} or {rel(OOF_JSON)} missing — run python -I eda/c118_stage2.py oof --stack {a.stack} first", 3)
    oof = json.loads(OOF_JSON.read_text())
    if oof.get("stack_sha256") != sha256_file(ctx["stack_file"]) or oof.get("inputs") != inputs_sha256(ctx):
        abort(f"{rel(OOF_JSON)} was computed with a different blend file or different inputs — rerun c118 oof")
    ref = np.load(ref_file)
    y, seen, G = ctx["y"], ctx["seen"], S2.GUARD_SET
    draws_q = C107.question_draws(ctx["q_code"])
    std, betas = standard_forms(ctx, "select")
    L = C107.logs_of(ctx["mats"], ctx["spec"]["names"], ctx["spec"]["floors"], seen, False)
    nested = {k: nested_forms(ctx, "select", k, L) for k in range(5)}
    rec_rows = {f: C107.rows_map(y, std[f], seen) for f in FORMS}
    res = {"kind": "main form + NONE guard + general-code identifiers (main_guard), nested as in c118 oof",
           "stack": rel(ctx["stack_file"]), "stack_sha256": sha256_file(ctx["stack_file"]), "beta_outer": betas, "K": S2.K_CAND,
           "lambda": S2.LAMBDA, "guard_set": G, "protect_none": True, "id_features": list(S2.GENERAL_ID_FEATS), "sets": {},
           "inputs": inputs_sha256(ctx)}
    rows_out, main_r, null_ok = {}, {}, {}
    for fs in ("main", G):
        res["sets"][fs] = {}
        for form in FORMS:
            r = run_nested(ctx, std, nested, form, fs)
            if not np.allclose(r["base"], rec_rows[form]):
                abort(f"base of the candidate order [{fs}|{form}] != recipe")
            if fs == "main":
                if not (np.array_equal(r["forced"], ref[f"main_{form}_forced"]) and np.array_equal(r["abstain"], ref[f"main_{form}_abstain"])):
                    abort(f"main form [{form}] does not reproduce the row values of {rel(ref_file)} bit for bit")
                main_r[form] = r
            ent = {"map_forced": float(r["forced"].mean()), "map_abstain": float(r["abstain"].mean()),
                   "abstain_folds": int(sum(c["abstain"] for c in r["choice"].values())), "gate": {str(k): v for k, v in r["choice"].items()},
                   "n_features": len(r["features"]), "changed_top_share": float(r["changed_top"].mean()),
                   "changed_top_share_forced": float(r["changed_top_forced"].mean()),
                   "changed_top3_share_forced": float(r["changed_top3_forced"].mean()), "vs_recipe_beta": boot(r["forced"] - rec_rows[form], ctx, draws_q)}
            if fs == G:
                m = main_r[form]
                ent["vs_main"] = boot(r["forced"] - m["forced"], ctx, draws_q)
                ent["vs_main_with_abstention"] = boot(r["abstain"] - m["abstain"], ctx, draws_q)
                ent["rows_protected"] = int(r["protected"].sum())
                ent["rows_other_top1_vs_main"] = int((r["top"][:, 0] != m["top"][:, 0]).sum())
                z = run_nested(ctx, std, nested, form, G, ids_zero=True)  # copy null point (i): identifiers 0, guard off -> main form
                null_ok[form] = bool(np.array_equal(z["forced"], m["forced"]) and np.array_equal(z["abstain"], m["abstain"])
                                     and np.array_equal(z["top"], m["top"]) and np.array_equal(z["cand"], m["cand"]))
                ent["null_ids_zero_equals_main"] = null_ok[form]
            if form == "set":
                ent["eval_features_sha256"] = sha256_array(eval_rows_in_table_order(ctx, r["X_eval"]))
                ent["features"] = r["features"]
            res["sets"][fs][form] = ent
            for key in ("forced", "abstain", "top", "cand", "protected"):
                rows_out[f"{fs}_{form}_{key}"] = r[key]
            vm = ent.get("vs_main")
            print(f"  [{fs:10s}|{form:4s}] {ent['n_features']} features | MAP {ent['map_forced']:.6f} | vs. base recipe+β {ent['vs_recipe_beta']['delta']:+.5f}"
                  + (f" | vs. main form {vm['delta']:+.5f} ± {vm['se_q']:.5f} {vm['fold_units']} | guarded {ent['rows_protected']} | "
                     f"null point (i) {'bit-identical' if ent['null_ids_zero_equals_main'] else 'VIOLATED'}" if vm else "")
                  + f" | abstention {ent['abstain_folds']}/5 | new top code {ent['changed_top_share_forced']:.4f}", flush=True)
    res["null_point_i"] = {"ok": bool(all(null_ok.values())), "forms": null_ok}
    res["delta_vs_main"] = {f: res["sets"][G][f]["vs_main"]["delta"] for f in FORMS}
    res["seconds"] = round(time.time() - t0, 1)
    GUARD_OUT.mkdir(parents=True, exist_ok=True)
    GUARD_OOF_JSON.write_text(to_json(res))
    np.savez_compressed(GUARD_OOF_ROWS, **rows_out)
    print(f"copy null point (i): {'PASSED' if res['null_point_i']['ok'] else 'VIOLATED'}; written: {rel(GUARD_OOF_JSON)} ({res['seconds']} s)")
    return 0 if res["null_point_i"]["ok"] else 4


def cmd_guard_fit(a) -> int:
    """Deployment file = stage2_s034.json (recomputed character for character) with the main_guard model in place of models.main."""
    target = Path(a.out) if a.out else STAGE2_GUARD
    if target.resolve() in (STAGE2_FILE.resolve(), STAGE2_WIDENED.resolve()):
        abort(f"guard-fit never writes {rel(STAGE2_FILE)} or {rel(STAGE2_WIDENED)} — separate output (default {rel(STAGE2_GUARD)})", 2)
    try:
        ctx = context(a.stack)
    except Missing as e:
        abort(str(e), 3)
    for f in (OOF_JSON, GUARD_OOF_JSON):
        if not f.is_file():
            abort(f"{rel(f)} missing — run c118 oof and c118 guard-oof --stack {a.stack} first", 3)
    oof, goof = json.loads(OOF_JSON.read_text()), json.loads(GUARD_OOF_JSON.read_text())
    sha_stack, inputs = sha256_file(ctx["stack_file"]), inputs_sha256(ctx)
    for nm, o in ((OOF_JSON, oof), (GUARD_OOF_JSON, goof)):
        if o.get("stack_sha256") != sha_stack or o.get("inputs") != inputs:
            abort(f"{rel(nm)} was computed with a different blend file or different inputs — recompute it")
    if not (oof.get("reproduction") or {}).get("ok"):
        abort("reproduction of the OOF numbers failed — no deployment")
    if not (goof.get("null_point_i") or {}).get("ok"):
        abort("copy null point (i) of guard-oof failed — no deployment")
    out, _ = build_stage2(ctx, oof)
    try:
        out = with_widened_rule(out, ctx["n"])
    except Missing as e:
        abort(str(e), 3)
    if not STAGE2_WIDENED.is_file() or STAGE2_WIDENED.read_text() != to_json(out):
        abort(f"the recomputation under the widened rule is not {rel(STAGE2_WIDENED)} character for character — main_guard only applies to the unchanged deployment file")
    y, Kc, G = ctx["y"], S2.K_CAND, S2.GUARD_SET
    forms = deploy_forms(ctx)
    X, cand, names = deploy_features(ctx, forms, G)
    got = sha256_array(X)
    want = (goof["sets"][G].get("set") or {}).get("eval_features_sha256")
    if got != want:
        abort(f"null point (e) violated: features of the deployment form [{G}] ({got[:16]}…) != evaluated rows of guard-oof ({str(want)[:16]}…)")
    mm = out["models"]["main"]
    nm = len(mm["features"])
    if names[:nm] != mm["features"] or names[nm:] != list(S2.GENERAL_ID_FEATS) or sha256_array(X[:, :nm]) != mm["features_sha256"]:
        abort("the first features of main_guard are not the features of the main form from stage2_s034.json (bit-identical)")
    lab = S2.labels_of(cand, y)
    g = S2.gate(fold_blocks(ctx, X, cand, lab), protect=True)
    model = S2.zero_model(len(names)) if g["abstain"] else S2.fit_clogit(X, lab, Kc, S2.LAMBDA)
    top0 = S2.reorder(cand, S2.scores(model, X, Kc))
    top = S2.protect_none(top0, cand)
    ins = {"map_base": float(S2.map_rows(cand, y).mean()), "map_reranked": float(S2.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    o = goof["sets"][G]
    gm = {"features": names, "feature_set": G, "protect_none": True, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
          "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
          "oof_changed_top_share": float(o["set"]["changed_top_share_forced"]),
          "oof_changed_top_share_with_abstention": float(o["set"]["changed_top_share"]),
          "oof_changed_top3_share": float(o["set"]["changed_top3_share_forced"]),
          "oof_vs_main": {f: o[f]["vs_main"]["delta"] for f in FORMS}, "oof_vs_recipe_beta": {f: o[f]["vs_recipe_beta"]["delta"] for f in FORMS},
          "features_sha256": got, "insample": ins}
    new = {}
    for k, v in out.items():
        if k != "derived_from":
            new[k] = {"main": gm, "fallback": v["fallback"]} if k == "models" else v
    new["guard_build"] = {"feature_set": G, "protect_none": True, "id_features": list(S2.GENERAL_ID_FEATS),
                          "oof": rel(GUARD_OOF_JSON), "null_point_i": goof["null_point_i"],
                          "fallback": "models.fallback unchanged from stage2_s034.json (if the note guard fails, the file equals the fallback-form file)"}
    new["derived_from"] = {"file": rel(STAGE2_WIDENED), "sha256": sha256_file(STAGE2_WIDENED), "differs_only_in": ["models.main", "guard_build", "derived_from"],
                           "build": "tools/builds/s035_stage2_guard.sh", "previous": out["derived_from"]}
    text = to_json(new)
    print(f"  [{G}] {len(names)} features (main form {nm} bit-identical + {len(S2.GENERAL_ID_FEATS)} identifiers) | inner gate {g['inner']:.5f} vs. base "
          f"{g['inner_base']:.5f} -> {'ABSTAIN (weight 0)' if g['abstain'] else 'apply'} | in-sample {ins['map_base']:.5f} → {ins['map_reranked']:.5f} "
          f"(new top code {ins['changed_top_share']:.4f}, guarded {ins['rows_protected']}; OOF {gm['oof_changed_top_share']:.4f}) | identifier weights "
          + ", ".join(f"{f} {w:+.3f}" for f, w in zip(names[nm:], gm["w"][nm:])))
    print(f"  fallback, note guard (widened rule), β, K, λ, reference = {rel(STAGE2_WIDENED)} (SHA-256 {new['derived_from']['sha256'][:16]}…)")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof", help="nested reproduction in three forms (private JSON)")
    o.add_argument("--stack", default=STACK_DEFAULT)
    o.add_argument("--feature-sets", default="main,fallback")
    o.add_argument("--no-placebo", dest="placebo", action="store_false", help="without the label placebo")
    f = sub.add_parser("fit", help="build (--save) or recompute (--verify) the deployment file artifacts/stage2_s032.json")
    f.add_argument("--stack", default=STACK_DEFAULT)
    f.add_argument("--guard-rule", choices=("band", "widened"), default="band",
                   help="rule of the note guard: band (default) or widened (only guard_rule and derived_from change)")
    f.add_argument("--out", help="output file (default: artifacts/stage2_s032.json for band, artifacts/stage2_s034.json for widened)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    nl = sub.add_parser("null", help="write a null model (weight 0) for a blend file")
    nl.add_argument("--stack", required=True)
    nl.add_argument("--out", required=True)
    nl.add_argument("--source", default=str(STAGE2_FILE))
    go = sub.add_parser("guard-oof", help="nested reproduction of the main form + main_guard (NONE guard) -> artifacts/eda/private/stage2_guard/")
    go.add_argument("--stack", default=STACK_DEFAULT)
    gf = sub.add_parser("guard-fit", help="build (--save) or recompute (--verify) the deployment file artifacts/stage2_s035.json")
    gf.add_argument("--stack", default=STACK_DEFAULT)
    gf.add_argument("--out", help="output file (default: artifacts/stage2_s035.json; never stage2_s032.json or stage2_s034.json)")
    gg = gf.add_mutually_exclusive_group(required=True)
    gg.add_argument("--save", action="store_true")
    gg.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    return {"oof": cmd_oof, "fit": cmd_fit, "null": cmd_null, "guard-oof": cmd_guard_oof, "guard-fit": cmd_guard_fit}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
