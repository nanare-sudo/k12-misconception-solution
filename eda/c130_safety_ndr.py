"""SAFETY LINE WITH REGEX-EXACT NOTE COLUMNS (safety_ndr): the safety line with note features (safety_nd, feature set fallback_i1_nd) with
the feature set "fallback_i1_ndr" in the second stage: fallback_i1_nd (58, unchanged) + the six regex-exact note columns R (src/note_regex.py;
M3 notes d1–d3, notes of the first pr27 draw). eda/c129_safety_nd.py and eda/c123_safety_set.py stay unchanged; this script only uses them
as modules. Build script: tools/builds/safety_ndr.sh.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c130_safety_ndr.py export-notes (--save | --verify)  # pr27 notes (training n1, n2; test n1) = re-export from the runs
    python -I eda/c130_safety_ndr.py oof [--placebos 15]                # nested: ND+R against ND (safety_nd), copy null point (R 0 -> ND), placebos
    python -I eda/c130_safety_ndr.py swap                               # full draw swap over the four sets from d1–d4 + sensitivity pr27 n2
    python -I eda/c130_safety_ndr.py splits [--n 20]                    # new meta-splits: (ND+R) − ND (expected-value rule)
    python -I eda/c130_safety_ndr.py rule                               # safety-ndr rule (1)–(4) with the risk rule against safety_nd
    python -I eda/c130_safety_ndr.py fit (--save | --verify)            # deployment file artifacts/stage2_safety_ndr.json
    python -I eda/c130_safety_ndr.py null --out <file>                  # null model (weight 0) for the deployment file
    python -I eda/c130_safety_ndr.py b-zero --out <file>                # copy null point: weights of safety_nd (stage2_safety_nd.json) + 0 for R

FEATURES (fixed before the evaluation): R = t_m3_logcnt, t_m3_first, t_m3_lastpos (M3 notes d1–d3, frozen under
data/derived/sft/m3_d<k>_*_frozen_notes.jsonl, as in safety_nd), t_pr_men, t_pr_last, t_pr_logcnt (reasoning of the first pass of the first
pr27 draw, frozen with tools/export_proto_notes.py: training data/derived/proto_notes/train_pr27.jsonl from pr27_rp/proto_a+proto_b, test
data/derived/proto_notes_test/test_pr27.jsonl from pr27te_rp/proto_te — the same runs as the channel pr27xm). Definition and extractor:
src/note_regex.py (mentions via stage2.mention_pattern; empty note -> 0). R is row-local and label-free, hence fold-honest without nesting;
the ND columns stay as in c129 (reference of the example similarity nested).
OOF (`oof`): as c129.run_nested_nd (gate over the four inner folds with guard, NONE guard), forms F (per fold) and M (one set); reference
ND = c129.run_nested_nd (safety_nd; rows bit-identical to artifacts/eda/private/safety_nd/c129_oof_rows.npz if present); copy null point:
R columns 0 -> ND bit-identical (row values, orders, guard); ND prefix of the features of the evaluated rows = c129.block_nd bit-identical;
placebo: the six R columns from another row of the same block at the candidates of the own row (default_rng(seed + R), 15 seeds as in
c123/c129).
DRAW SWAP (`swap`): per set of three draws from d1–d4 (123, 124, 134, 234) the channel m3x3_full (mean of the three draws), draw ranks, note
mentions, example similarity AND the M3 columns of R come from THE SAME three draws (pr27 first draw fixed); d4 only on the training side
(runs/m3d4_oof_rp, notes via c118.train_note_texts). Per set the original split and 8 meta-splits (seeds 1000–1007), per split (ND+R) − ND
(form "forced", ½(F+M)) and a placebo (seed 88000 + s). Sensitivity (reported only): set 123 with the second pr27 draw
(data/derived/proto_notes/train_pr27d2.jsonl).
SAFETY-NDR RULE (1)–(4) (fixed before the evaluation; `rule`): (1) (ND+R) − ND in the draw swap (per set the mean over the 8 splits) > 0 in
the mean of the four sets and > 0 in ≥ 3 of 4 sets; (2) original ½(F+M) > 0 and above the placebo mean (15 placebos); (3) mean over 20 new
meta-splits > 0; (4) risk rule against safety_nd (eda/risk_rule.py as in c129: class R, k 6, ρ 0.5, spread 0.0015), σ over questions of the
original split, τ_d = spread of the four sets: P(θ < −0.002) ≤ 0.05 for the observation "original" (reading of the safety-nd rule, c129)
AND "mean of the four sets" (reading of the draw swap); reported only: τ_d 0, worst set, sensitivity pr27 n2. Precondition: copy null point
of the OOF reproduction. The other conditions (copy null points of the test side, regressions, guards, positive controls) are checked by
the build script tools/builds/safety_ndr.sh, the byte-identical rebuild without network separately.
DEPLOYMENT (`fit`): training rows = all 1,500 in form M, knowledge and reference of the similarity from the other folds, gate over the five
folds; features of the deployment form = evaluated rows of the OOF reproduction (null point (e)). Guards of the test side
(tools/make_submission.py, apply_stage2_safe): as in safety_nd (widened note guard, similarity guard, guard (c)) and the guard of the columns
R (safety-ndr rule (6)) against r_reference (pr27 mention rate per row, share of rows in which pr27 mentions the top code of the base form,
mean last position of the M3 mentions per draw; band ±10 % OR |z| ≤ 3) — each ONLY as an abort without a file, never as a switch.

Training rows only for statistics; texts are never printed, test notes only by machine (checksums, comparison with the re-export). Exit 3 =
input missing; 4 = checksum/recomputation/null point violated; 5 = pre-registered rule not met (no deployment).
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

import note_regex as NR  # noqa: E402
import note_sim as NS  # noqa: E402
import stage2 as S2  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C129 = _load("c130_c129", COMP / "eda" / "c129_safety_nd.py")
C123 = C129.C123
C118 = C129.C118
EPN = _load("c130_export_proto_notes", COMP / "tools" / "export_proto_notes.py")

ART = COMP / "artifacts"
OUT = ART / "eda" / "private" / "safety_ndr"
OOF_JSON = OUT / "c130_oof.json"
OOF_ROWS = OUT / "c130_oof_rows.npz"
SWAP_JSON = OUT / "c130_swap.json"
SPLITS_JSON = OUT / "c130_splits.json"
RULE_JSON = OUT / "c130_rule.json"
STAGE2_FILE = ART / "stage2_safety_ndr.json"
STAGE2_SAFETY_ND = ART / "stage2_safety_nd.json"  # deployment file of safety_nd (eda/c129_safety_nd.py)
STAGE2_SAFETY_ND_SHA = "50217a6c48f16bf35afb97fd755611eba85c43fa81187d0d00a91dbb31cddbee"
ND_ROWS = ART / "eda" / "private" / "safety_nd" / "c129_oof_rows.npz"  # rows of the OOF reproduction of safety_nd (reference ND)
FORMS = ("fold", "set")
FORM_SHORT = {"fold": "F", "set": "M"}
PR27_NOTES = {  # frozen pr27 notes (tools/export_proto_notes.py, note_chars 16,000): name -> (file relative to the data folder, table, sources)
    "train_pr27": ("derived/proto_notes/train_pr27.jsonl", "train", ("pr27_rp/proto_a/proto", "pr27_rp/proto_b/proto")),
    "test_pr27": ("derived/proto_notes_test/test_pr27.jsonl", "test", ("pr27te_rp/proto_te/proto",)),
    "train_pr27d2": ("derived/proto_notes/train_pr27d2.jsonl", "train", ("pr27s2_rp/proto_a2/proto", "pr27s3_rp/proto_b2/proto")),
}
NOTE_CHARS = 16000
D4_SPEC = "+".join(f"runs/m3d4_oof_rp/m3d4_rank_f{k}/bench" for k in range(5))  # fourth M3 draw (training rows only)
SWAP_SETS = ("123", "124", "134", "234")
SWAP_SPLITS, SWAP_PLACEBO_SEED0 = 8, 88000
PLACEBO_SEEDS = tuple(C129.PLACEBO_SEEDS)
SPLIT_SEED0, SPLIT_PLACEBO_SEED0, MIN_SPLITS = 1000, 7900, 20
MIN_PLACEBOS = 15
RISK_SEED, RISK_K, RISK_RHO, RISK_SPREAD, RISK_PHARM = 170_181, 6, 0.5, 0.0015, 0.05
NR_COLS = len(S2.R_FEATS)


def abort(msg: str, code: int = 4) -> None:
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ pr27 notes (frozen)

def export_payload(name: str) -> tuple[bytes, str, str, int]:
    """Re-export as tools/export_proto_notes.py (collect, same rows and meta file, bytes): -> (content, text of the meta file, table,
    number of rows). By machine only; texts are never printed."""
    _rel, table_want, sources = PR27_NOTES[name]
    try:
        table, rows, used, states = EPN.collect(list(sources), ART / "runs", NOTE_CHARS)
    except (OSError, KeyError, ValueError) as e:
        abort(f"{name}: source of the notes not readable ({type(e).__name__})", 3)
    if table != table_want:
        abort(f"{name}: the sources concern the table {table}, expected {table_want}")
    if sum(bool(EPN.MARKS.search(t)) for t in rows.values()):
        abort(f"{name}: notes with markers of the chat template")
    payload = "".join(json.dumps({"table": table, "row": int(i), "state": states[i], "text": rows[i]}, ensure_ascii=False) + "\n"
                      for i in sorted(rows)).encode("utf-8")
    sha = sha256_bytes(payload)
    chars = sorted(len(t) for t in rows.values())
    by_state = {s: sum(v == s for v in states.values()) for s in tuple(EPN.RECORDED_STATE.values())}
    meta = json.dumps({"name": name, "table": table, "rows": len(rows), "sources": used, "note_chars": NOTE_CHARS,
                       "chars_median": chars[len(chars) // 2] if chars else 0, "states": by_state, "sha256": sha}, indent=1)
    return payload, meta, table, len(rows)


def pr27_texts(root: Path, name: str, n: int, want_sha: str | None = None) -> tuple[list[str], str]:
    """Frozen pr27 notes: file = checksum of the meta file (and want_sha), table as in PR27_NOTES, every row exactly once, all n rows.
    Rows are split at "\\n", never with splitlines() (U+2028 and similar characters stand raw in the texts, tools/export_proto_notes.py)."""
    rel, table, sources = PR27_NOTES[name]
    f = Path(root) / rel
    mf = f.with_suffix(".meta.json")
    if not f.is_file() or not mf.is_file():
        abort(f"{rel} or {mf.name} missing — run tools/export_proto_notes.py or c130 export-notes --save", 3)
    raw = f.read_bytes()
    sha = sha256_bytes(raw)
    meta = json.loads(mf.read_text())
    if meta.get("sha256") != sha or (want_sha is not None and sha != want_sha):
        abort(f"{rel}: checksum {sha[:16]}… does not match the meta file or the deployment file")
    if [s.get("source") for s in meta.get("sources") or []] != list(sources) or meta.get("table") != table or int(meta.get("rows", -1)) != n:
        abort(f"{rel}: the meta file names other sources, another table or another number of rows")
    texts: dict[int, str] = {}
    for line in raw.decode("utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            i = int(r["row"])
            if r.get("table") != table or i in texts or not 0 <= i < n:
                abort(f"{rel}: rows not unique or not table {table}")
            texts[i] = str(r.get("text") or "")
    if len(texts) != n:
        abort(f"{rel}: {len(texts)} of {n} rows")
    return [texts[i] for i in range(n)], sha


def cmd_export_notes(a) -> int:
    root = C123.c120().C.find_data_root()
    for name, (rel, _table, sources) in PR27_NOTES.items():
        payload, meta, table, rows = export_payload(name)
        f = root / rel
        mf = f.with_suffix(".meta.json")
        sha = sha256_bytes(payload)
        if a.verify:
            if not f.is_file() or f.read_bytes() != payload or not mf.is_file() or mf.read_text() != meta:
                abort(f"{rel} is not the re-export from {', '.join(sources)}")
            print(f"  VERIFIED: {rel} = re-export from {', '.join(sources)} (SHA-256 {sha[:16]}…, table {table}, {rows} rows)")
        else:
            if f.is_file() and (f.read_bytes() != payload or (mf.is_file() and mf.read_text() != meta)):
                abort(f"{rel} exists with different content — not overwritten")
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.is_file():
                f.write_bytes(payload)
            if not mf.is_file():
                mf.write_text(meta)
            print(f"  saved/present: {rel} (SHA-256 {sha}, table {table}, {rows} rows)")
    return 0


# ------------------------------------------------------------------------------------------------ context with R

def add_r(ctx: dict, m3_stats: list[dict], pr_stats: dict) -> None:
    """R from the mentions (M3 draws of the context, pr27) and a Static version with notes and R (feature set fallback_i1_ndr)."""
    R = NR.r_features(m3_stats, pr_stats)
    spec = ctx["spec"]
    st = S2.Static(spec["names"], ctx["mats"], ctx["draws"], ctx["grade"], ctx["seen"], ctx["classes"], ctx["notes"])
    S2.attach_r(st, R)
    ctx.update(R=R, static_ndr=st, m3_stats=m3_stats, pr_stats=pr_stats)


def ndr_context(check_source: bool = True) -> dict:
    """c129 context (notes d1–d3, similarity) + pr27 notes of the first draw (frozen) + mentions + R."""
    ctx = C129.nd_context(check_source)
    n, classes = ctx["n"], ctx["classes"]
    pr, pr_sha = pr27_texts(ctx["root"], "train_pr27", n)
    m3_stats = [NR.mention_stats(t, classes) for t in ctx["note_texts"]]
    add_r(ctx, m3_stats, NR.mention_stats(pr, classes))
    ctx["pr27_train_sha256"] = pr_sha
    ctx["inputs_ndr"] = {**ctx["inputs_nd"], "pr27_train_notes": {PR27_NOTES["train_pr27"][0]: pr_sha}, "extractor": NR.VERSION}
    return ctx


def block_ndr(ctx: dict, rows: np.ndarray, FM: np.ndarray, PM: np.ndarray, exclude=(), placebo_seed: int | None = None):
    """Features "fallback_i1_ndr" of the rows `rows` of ONE fold; knowledge and reference of the similarity as in c129.block_nd (other folds
    except `exclude`). placebo_seed: the six R columns from another row of the same block (at the candidates of the own row; ND stays real)."""
    folds, y, K = ctx["folds"], ctx["y"], ctx["K"]
    fs = set(int(v) for v in folds[rows])
    if len(fs) != 1:
        abort(f"rows from several folds {sorted(fs)}")
    f = fs.pop()
    kn = S2.knowledge_other_folds(ctx["key"], ctx["topic_s"], y, folds, rows, K, exclude)
    pool = np.flatnonzero((folds != f) & ~np.isin(folds, list(exclude)))
    kn["demo_sim"] = NS.demo_sim(ctx["SIM"][np.ix_(rows, pool)], y[pool], K)
    X, cand, names = S2.pair_features(ctx["static_ndr"], rows, FM, PM, kn, S2.NDR_SET)
    if placebo_seed is not None:
        R, Kc = len(rows), S2.K_CAND
        src = np.random.default_rng(placebo_seed + R).permutation(R)
        ii, cc = np.repeat(np.arange(R), Kc), cand.reshape(-1)
        X = X.copy()
        X[:, -NR_COLS:] = ctx["R"][rows[src]][ii, cc, :]
    return X, cand, names


def run_nested_ndr(ctx: dict, std: dict, nested: dict, form: str, mode: str = "real", seed: int | None = None) -> dict:
    """As c129.run_nested_nd with the feature set fallback_i1_ndr. mode: real, zero (R columns 0 -> ND bit-identical), placebo (seed)."""
    if mode not in ("real", "zero", "placebo") or (mode == "placebo") != (seed is not None):
        raise ValueError(f"mode {mode!r} / seed {seed!r}")
    y, folds, n, Kc = ctx["y"], ctx["folds"], ctx["n"], S2.K_CAND

    def prep(X: np.ndarray) -> np.ndarray:
        if mode != "zero":
            return X
        X = X.copy()
        X[:, -NR_COLS:] = 0.0
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
            X, cand, _ = block_ndr(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], exclude=(k,), placebo_seed=seed)
            XT.append(prep(X))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = block_ndr(ctx, E, std[form][E], std["pre"][E], placebo_seed=seed)
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


def same_gates(a: dict, b: dict) -> bool:
    return bool(json.dumps({str(k): v for k, v in a["gates"].items()}, sort_keys=True) == json.dumps({str(k): v for k, v in b["gates"].items()}, sort_keys=True))


# ------------------------------------------------------------------------------------------------ nested reproduction

def cmd_oof(a) -> int:
    t0 = time.time()
    C = C123.c120()
    ctx = ndr_context()
    y, seen = ctx["y"], ctx["seen"]
    std, nested = C123.prepare(ctx)
    rec_rows = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS}
    draws_q = C.C107.question_draws(ctx["q_code"])
    seeds = PLACEBO_SEEDS[: a.placebos]
    ref_nd = np.load(ND_ROWS) if ND_ROWS.is_file() else None
    res = {"kind": "safety_ndr: recipe of safety_nd (fallback_i1_nd) with the feature set fallback_i1_ndr (+ six regex-exact note columns R) in the "
                   "second stage", "stack": C.rel(C.STACK), "stack_sha256": C.sha256_file(C.STACK), "K": S2.K_CAND, "lambda": S2.LAMBDA,
           "feature_set": S2.NDR_SET, "r_features": list(S2.R_FEATS), "extractor": NR.VERSION, "protect_none": True,
           "base_map": {f: float(rec_rows[f].mean()) for f in FORMS}, "sets": {"nd": {}, "ndr": {}},
           "placebo": {"seeds": list(seeds), "per_seed": []}, "inputs": ctx["inputs_ndr"]}
    rows_out, null_ok, prefix_ok, nd_ok = {}, {}, {}, {}
    pl = {s: {} for s in seeds}
    for form in FORMS:
        rb = C123.run_nested(ctx, std, nested, form, "b")
        rnd = C129.run_nested_nd(ctx, std, nested, form, "real")
        rr = run_nested_ndr(ctx, std, nested, form, "real")
        rz = run_nested_ndr(ctx, std, nested, form, "zero")
        null_ok[form] = bool(C123.same_rows(rz, rnd) and same_gates(rz, rnd))
        prefix_ok[form] = bool(all(np.array_equal(rr["X_eval"][k][:, :-NR_COLS], rnd["X_eval"][k]) for k in range(5)))
        if ref_nd is not None:
            nd_ok[form] = bool(all(np.array_equal(rnd[key], ref_nd[f"nd_{form}_{key}"]) for key in ("forced", "gated", "top", "cand", "protected")))
        for nm, r_ in (("nd", rnd), ("ndr", rr)):
            ent = {"map_forced": float(r_["forced"].mean()), "map_gated": float(r_["gated"].mean()),
                   "units_forced": int(C.C107.units(r_["forced"]).sum()), "units_gated": int(C.C107.units(r_["gated"]).sum()),
                   "abstain_folds": int(sum(g["abstain"] for g in r_["gates"].values())), "gates": {str(k): v for k, v in r_["gates"].items()},
                   "n_features": len(r_["features"]), "changed_top_share": float((r_["top"][:, 0] != r_["cand"][:, 0]).mean()),
                   "changed_top3_share": float((r_["top"][:, :3] != r_["cand"][:, :3]).any(1).mean()), "rows_protected": int(r_["protected"].sum()),
                   "vs_base": C123.boot(ctx, r_["gated"] - rec_rows[form], draws_q), "vs_b": C123.boot(ctx, r_["forced"] - rb["forced"], draws_q)}
            if nm == "ndr":
                ent["vs_nd"] = C123.boot(ctx, rr["forced"] - rnd["forced"], draws_q)
                ent["vs_nd_gated"] = C123.boot(ctx, rr["gated"] - rnd["gated"], draws_q)
                ent["null_r_zero_equals_nd"] = null_ok[form]
                ent["nd_prefix_equals_c129"] = prefix_ok[form]
                ent["changed_top_vs_nd"] = int((rr["top"][:, 0] != rnd["top"][:, 0]).sum())
            if form == "set":
                Xt = C123.eval_rows_in_table_order(ctx, r_["X_eval"])
                ent["eval_features_sha256"] = C.sha256_array(Xt)
                ent["features"] = r_["features"]
                ent["novel_task_share_top5"] = float(Xt[:, r_["features"].index("novel_task")].mean())
            res["sets"][nm][form] = ent
            for key in ("forced", "gated", "top", "cand", "protected"):
                rows_out[f"{nm}_{form}_{key}"] = r_[key]
        for sd in seeds:
            p = run_nested_ndr(ctx, std, nested, form, "placebo", sd)
            pl[sd][form] = float((p["forced"] - rnd["forced"]).mean())
        e = res["sets"]["ndr"][form]
        print(f"  [{FORM_SHORT[form]}] base {res['base_map'][form]:.6f} | ND {res['sets']['nd'][form]['map_forced']:.6f} | ND+R {e['map_forced']:.6f} "
              f"(ND+R − ND {e['vs_nd']['delta']:+.5f} ± {e['vs_nd']['se_q']:.5f} {e['vs_nd']['fold_units']}; gate + guard {e['vs_nd_gated']['delta']:+.5f}; "
              f"abstention {e['abstain_folds']}/5; new top code {e['changed_top_share']:.4f}; top code changed against ND {e['changed_top_vs_nd']}; "
              f"null point {'bit-identical' if null_ok[form] else 'VIOLATED'}; ND prefix {'bit-identical' if prefix_ok[form] else 'VIOLATED'}; safety_nd rows "
              f"{'—' if form not in nd_ok else ('bit-identical' if nd_ok[form] else 'VIOLATED')}; placebo mean "
              f"{np.mean([pl[s][form] for s in seeds]) if seeds else float('nan'):+.5f}) ({time.time() - t0:.0f} s)", flush=True)
    res["null_point"] = {"r_zero_equals_nd": {"ok": bool(all(null_ok.values())), "forms": null_ok},
                         "nd_prefix_equals_c129": {"ok": bool(all(prefix_ok.values())), "forms": prefix_ok},
                         "nd_equals_safety_nd_oof_rows": {"checked": ref_nd is not None, "ok": bool(all(nd_ok.values())) if nd_ok else None, "forms": nd_ok,
                                                          "file": C.rel(ND_ROWS)}}
    d = {f: res["sets"]["ndr"][f]["vs_nd"]["delta"] for f in FORMS}
    half = 0.5 * ((rows_out["ndr_fold_forced"] - rows_out["nd_fold_forced"]) + (rows_out["ndr_set_forced"] - rows_out["nd_set_forced"]))
    res["ndr_vs_nd"] = {"F": d["fold"], "M": d["set"], "A": 0.5 * (d["fold"] + d["set"]), "A_boot": C123.boot(ctx, half, draws_q)}
    db = {f: res["sets"]["ndr"][f]["vs_b"]["delta"] for f in FORMS}
    res["ndr_vs_b"] = {"F": db["fold"], "M": db["set"], "A": 0.5 * (db["fold"] + db["set"])}
    per = [{"seed": s, "F": pl[s]["fold"], "M": pl[s]["set"], "A": 0.5 * (pl[s]["fold"] + pl[s]["set"])} for s in seeds]
    pa = np.array([p["A"] for p in per]) if per else np.zeros(0)
    res["placebo"].update({"per_seed": per, "n": len(per), "A_mean": float(pa.mean()) if len(pa) else None,
                           "A_sd": float(pa.std(ddof=1)) if len(pa) > 1 else None, "A_max": float(pa.max()) if len(pa) else None,
                           "share_ge_real": float((pa >= res["ndr_vs_nd"]["A"]).mean()) if len(pa) else None})
    OUT.mkdir(parents=True, exist_ok=True)
    OOF_JSON.write_text(C.to_json(res))
    np.savez_compressed(OOF_ROWS, **rows_out, **{f"base_{f}": rec_rows[f] for f in FORMS})
    r = res["ndr_vs_nd"]
    npt = res["null_point"]
    ok = npt["r_zero_equals_nd"]["ok"] and npt["nd_prefix_equals_c129"]["ok"] and npt["nd_equals_safety_nd_oof_rows"]["ok"] is not False
    plt = (f"{res['placebo']['A_mean']:+.5f} (max {res['placebo']['A_max']:+.5f}, n {len(per)})" if per else "— (none)")
    print(f"ND+R − ND: F {r['F']:+.5f}, M {r['M']:+.5f}, ½(F+M) {r['A']:+.5f} ± {r['A_boot']['se_q']:.5f} (over questions); placebo mean {plt}; "
          f"against safety_final ½(F+M) {res['ndr_vs_b']['A']:+.5f}; copy null point (R 0 -> ND) {'bit-identical' if npt['r_zero_equals_nd']['ok'] else 'VIOLATED'}; "
          f"written: {C.rel(OOF_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0 if ok else 4


# ------------------------------------------------------------------------------------------------ full draw swap d1–d4

def swap_context(ctx0: dict, combo: str, draws_all: dict, texts_all: dict, sim_parts: dict, m3_stats_all: dict, pr_stats: dict) -> dict:
    """Context of a set of three draws: channel m3x3_full = mean of the three draws, ranks, notes, similarity (mean of the similarities per
    draw in the order of the set, as note_sim.similarity) and R from the same three draws; pr27 as given."""
    C = C123.c120()
    c = dict(ctx0)
    n, K, seen, classes, spec = ctx0["n"], ctx0["K"], ctx0["seen"], ctx0["classes"], ctx0["spec"]
    dr = [draws_all[d] for d in combo]
    mats = dict(ctx0["mats"])
    if combo != "123":
        mats[C.M3], _ = C.A9.mean_of_draws([(P, np.ones(n, bool)) for P in dr], n, K)
    c.update(mats=mats, draws=dr, L=C.C107.logs_of(mats, spec["names"], spec["floors"], seen, False),
             Lm=C.C107.logs_of(mats, spec["names"], spec["floors"], seen, True))
    c["static"] = S2.Static(spec["names"], mats, dr, ctx0["grade"], seen, classes, None)
    tx = [texts_all[d] for d in combo]
    notes = S2.notes_features(tx, classes)
    c.update(note_texts=tx, notes=notes)
    c["static_nd"] = S2.Static(spec["names"], mats, dr, ctx0["grade"], seen, classes, notes)
    S = None
    for d in combo:
        S = sim_parts[d] if S is None else S + sim_parts[d]
    c["SIM"] = NS.mask_same_task(S / float(len(combo)), ctx0["key"], ctx0["key"])
    add_r(c, [m3_stats_all[d] for d in combo], pr_stats)
    return c


def run_variant(c: dict, y: np.ndarray, label: str, t0: float) -> dict:
    """Original split + SWAP_SPLITS meta-splits: per split (ND+R) − ND and placebo − ND (forced, F and M, ½(F+M))."""
    recs = []
    for s in range(-1, SWAP_SPLITS):
        cc = c if s < 0 else dict(c, folds=C123.split_folds(y, SPLIT_SEED0 + s))
        std, nested = C123.prepare(cc)
        pseed = SWAP_PLACEBO_SEED0 + (0 if s < 0 else s + 1)
        rec = {"seed": None if s < 0 else SPLIT_SEED0 + s, "placebo_seed": pseed}
        for form in FORMS:
            rnd = C129.run_nested_nd(cc, std, nested, form, "real")
            rr = run_nested_ndr(cc, std, nested, form, "real")
            rp = run_nested_ndr(cc, std, nested, form, "placebo", pseed)
            rec[FORM_SHORT[form]] = float((rr["forced"] - rnd["forced"]).mean())
            rec[f"pl_{FORM_SHORT[form]}"] = float((rp["forced"] - rnd["forced"]).mean())
        rec["A"] = 0.5 * (rec["F"] + rec["M"])
        rec["pl_A"] = 0.5 * (rec["pl_F"] + rec["pl_M"])
        recs.append(rec)
    orig = recs[0]
    sp = recs[1:]
    out = {"original": orig, "splits": sp}
    for key in ("A", "F", "M", "pl_A"):
        v = np.array([r[key] for r in sp])
        out[f"splits_{key}"] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "share_pos": float((v > 0).mean())}
    v = np.array([r["A"] - r["pl_A"] for r in sp])
    out["splits_A_minus_pl"] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "share_pos": float((v > 0).mean())}
    print(f"  set {label}: original (ND+R) − ND {orig['A']:+.5f} (placebo {orig['pl_A']:+.5f}) | {SWAP_SPLITS} meta-splits {out['splits_A']['mean']:+.5f} ± "
          f"{out['splits_A']['se']:.5f} ({out['splits_A']['share_pos']:.0%} > 0; F {out['splits_F']['mean']:+.5f}, M {out['splits_M']['mean']:+.5f}) | "
          f"placebo {out['splits_pl_A']['mean']:+.5f} ({time.time() - t0:.0f} s)", flush=True)
    return out


def cmd_swap(a) -> int:
    t0 = time.time()
    C = C123.c120()
    ctx0 = ndr_context()
    n, seen, y, classes = ctx0["n"], ctx0["seen"], ctx0["y"], ctx0["classes"]
    P4, have4, _ = C.A9.load_draw(D4_SPEC, n, seen)
    if not have4.all():
        abort(f"{D4_SPEC}: fourth draw not on all rows", 3)
    try:
        t4 = C118.train_note_texts(D4_SPEC, ctx0["folds"])
    except C118.Missing as e:
        abort(str(e), 3)
    draws_all = {"1": ctx0["draws"][0], "2": ctx0["draws"][1], "3": ctx0["draws"][2], "4": P4}
    texts_all = {"1": ctx0["note_texts"][0], "2": ctx0["note_texts"][1], "3": ctx0["note_texts"][2], "4": t4}
    sim_parts = {d: NS.similarity([texts_all[d]], [texts_all[d]]) for d in "1234"}
    m3_stats_all = {"1": ctx0["m3_stats"][0], "2": ctx0["m3_stats"][1], "3": ctx0["m3_stats"][2], "4": NR.mention_stats(t4, classes)}
    pr2, pr2_sha = pr27_texts(ctx0["root"], "train_pr27d2", n)
    pr_stats = {"n1": ctx0["pr_stats"], "n2": NR.mention_stats(pr2, classes)}
    d4_sha = hashlib.sha256("\n".join(t4).encode("utf-8")).hexdigest()
    res = {"kind": "safety-ndr rule (1): (ND+R) − ND under the full draw swap (four sets of three draws from d1–d4; channel, ranks, notes, "
                   "similarity, M3 columns of R from the same draws, pr27 first draw fixed); sensitivity pr27 n2 (set 123)",
           "sets": {}, "sensitivity": {}, "splits_per_set": SWAP_SPLITS, "split_seeds": [SPLIT_SEED0 + s for s in range(SWAP_SPLITS)],
           "placebo_seed0": SWAP_PLACEBO_SEED0, "d4": {"spec": D4_SPEC, "notes_sha256_joined": d4_sha},
           "pr27_n2": {"file": PR27_NOTES["train_pr27d2"][0], "sha256": pr2_sha}, "inputs": ctx0["inputs_ndr"]}
    for combo, prv in [(c, "n1") for c in SWAP_SETS] + [("123", "n2")]:
        c = swap_context(ctx0, combo, draws_all, texts_all, sim_parts, m3_stats_all, pr_stats[prv])
        if (combo, prv) == ("123", "n1"):  # the set of the deployment form is the context itself
            same = (np.array_equal(c["SIM"], ctx0["SIM"]) and np.array_equal(c["R"], ctx0["R"])
                    and all(np.array_equal(a_, b_) for a_, b_ in zip(c["L"], ctx0["L"])))
            if not same:
                abort("set 123 of the draw swap != context of the deployment form (similarity, R or logs)")
        out = run_variant(c, y, f"{combo}/{prv}", t0)
        (res["sets"] if prv == "n1" else res["sensitivity"])[combo if prv == "n1" else f"{combo}_pr27n2"] = out
    if OOF_JSON.is_file():  # set 123 in the original = OOF reproduction
        o = json.loads(OOF_JSON.read_text())
        if o.get("inputs") == res["inputs"] and abs(o["ndr_vs_nd"]["A"] - res["sets"]["123"]["original"]["A"]) > 1e-15:
            abort("draw swap, set 123 in the original != c130 oof")
    v = np.array([res["sets"][c]["splits_A"]["mean"] for c in SWAP_SETS])
    res["summary"] = {"values": dict(zip(SWAP_SETS, [float(x) for x in v])), "mean4": float(v.mean()), "sd4": float(v.std(ddof=1)), "min": float(v.min()),
                      "max": float(v.max()), "n_pos": int((v > 0).sum()), "pr27n2_splits_mean": res["sensitivity"]["123_pr27n2"]["splits_A"]["mean"],
                      "pr27n2_original": res["sensitivity"]["123_pr27n2"]["original"]["A"]}
    res["seconds_note"] = "run time not stored (file byte-identical on repetition)"
    OUT.mkdir(parents=True, exist_ok=True)
    SWAP_JSON.write_text(C.to_json(res))
    s = res["summary"]
    print(f"draw swap: sets {', '.join(f'{c} {x:+.5f}' for c, x in s['values'].items())} -> mean {s['mean4']:+.5f} (spread {s['sd4']:.5f}, "
          f"{s['n_pos']}/4 > 0); sensitivity pr27 n2 (set 123): meta-splits {s['pr27n2_splits_mean']:+.5f}, original {s['pr27n2_original']:+.5f}; "
          f"written: {C.rel(SWAP_JSON)} ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ new meta-splits

def cmd_splits(a) -> int:
    t0 = time.time()
    if a.n < MIN_SPLITS:
        abort(f"--n {a.n}: the pre-registered rule requires at least {MIN_SPLITS} meta-splits", 2)
    C = C123.c120()
    ctx0 = ndr_context()
    y = ctx0["y"]
    res = {"kind": "safety-ndr rule (3): (ND+R) − ND over new meta-splits (channel OOFs and notes fixed, meta-folds new)", "stack": C.rel(C.STACK),
           "stack_sha256": C.sha256_file(C.STACK), "splits": [], "inputs": ctx0["inputs_ndr"]}
    for s in range(a.n):
        ctx = dict(ctx0)
        ctx["folds"] = C123.split_folds(y, SPLIT_SEED0 + s)
        std, nested = C123.prepare(ctx)
        rec = {"seed": SPLIT_SEED0 + s, "placebo_seed": SPLIT_PLACEBO_SEED0 + s, "ndr_minus_nd": {}, "placebo_minus_nd": {}, "nd_map": {}}
        for form in FORMS:
            rnd = C129.run_nested_nd(ctx, std, nested, form, "real")
            rr = run_nested_ndr(ctx, std, nested, form, "real")
            rp = run_nested_ndr(ctx, std, nested, form, "placebo", SPLIT_PLACEBO_SEED0 + s)
            rec["nd_map"][form] = float(rnd["forced"].mean())
            rec["ndr_minus_nd"][form] = float((rr["forced"] - rnd["forced"]).mean())
            rec["placebo_minus_nd"][form] = float((rp["forced"] - rnd["forced"]).mean())
        res["splits"].append(rec)
        print(f"  split {s:2d} (seed {rec['seed']}): ND+R − ND F {rec['ndr_minus_nd']['fold']:+.4f} M {rec['ndr_minus_nd']['set']:+.4f} | placebo − ND "
              f"F {rec['placebo_minus_nd']['fold']:+.4f} M {rec['placebo_minus_nd']['set']:+.4f} ({time.time() - t0:.0f} s)", flush=True)
    summ = {}
    for key in ("ndr_minus_nd", "placebo_minus_nd"):
        F = np.array([r[key]["fold"] for r in res["splits"]])
        M = np.array([r[key]["set"] for r in res["splits"]])
        A = 0.5 * (F + M)
        summ[key] = {"F_mean": float(F.mean()), "M_mean": float(M.mean()), "A_mean": float(A.mean()), "A_se": float(A.std(ddof=1) / np.sqrt(len(A))),
                     "share_A_pos": float((A > 0).mean())}
    res["summary"] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(C.to_json(res))
    sr, sp = summ["ndr_minus_nd"], summ["placebo_minus_nd"]
    print(f"ND+R − ND over {a.n} meta-splits: ½(F+M) {sr['A_mean']:+.5f} ± {sr['A_se']:.5f} ({sr['share_A_pos']:.0%} > 0; F {sr['F_mean']:+.5f}, M "
          f"{sr['M_mean']:+.5f}); placebo − ND {sp['A_mean']:+.5f}; written: {C.rel(SPLITS_JSON)} ({time.time() - t0:.1f} s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ pre-registered safety-ndr rule (1)–(4)

_RR = {}


def risk_post(o: float, s: float, tau: float) -> dict:
    """Risk rule as in c129.risk_rule (eda/risk_rule.py: NPMLE prior of class R, k 6, ρ 0.5, spread 0.0015), one observation, one τ_d; a
    fresh random generator with a fixed seed per call."""
    if not _RR:
        _RR["m"] = _load("c130_risk", COMP / "eda" / "risk_rule.py")
        _RR["w"] = _RR["m"].class_r_prior()
    RR, wR = _RR["m"], _RR["w"]
    rng = np.random.default_rng(RISK_SEED)
    tol = 0.0004
    p = RR.combined_posterior(o, s, wR, rng, k=RISK_K, rho=RISK_RHO, tau_d=tau, spread=RISK_SPREAD, n_sim=400_000, chunks=8, tol=tol)
    while (p.get("ess") or 0) < 2000 and tol < 0.0032:
        tol *= 2
        p = RR.combined_posterior(o, s, wR, rng, k=RISK_K, rho=RISK_RHO, tau_d=tau, spread=RISK_SPREAD, n_sim=400_000, chunks=8, tol=tol)
    return {"o": float(o), "s": float(s), "tau_d": float(tau), "mean": None if p["mean"] is None else float(p["mean"]),
            "p_lt_m002": None if p["p_lt_m002"] is None else float(p["p_lt_m002"]), "ess": p.get("ess"), "tol": tol}


def evaluate_rule(oof: dict, swap: dict, sp: dict, risk=risk_post) -> dict:
    """Conditions (1)–(4) of the safety-ndr rule from the result files (risk: function (o, s, τ) -> posterior; replaceable for tests)."""
    npt = oof["null_point"]
    ok_null = bool(npt["r_zero_equals_nd"]["ok"] and npt["nd_prefix_equals_c129"]["ok"] and npt["nd_equals_safety_nd_oof_rows"]["ok"] is not False)
    v = [float(swap["sets"][c]["splits_A"]["mean"]) for c in SWAP_SETS]
    mean4, sd4 = float(np.mean(v)), float(np.std(v, ddof=1))
    ok1 = bool(mean4 > 0 and sum(x > 0 for x in v) >= 3)
    r, pl = oof["ndr_vs_nd"], oof["placebo"]
    ok2 = bool(r["A"] > 0 and int(pl.get("n") or 0) >= MIN_PLACEBOS and pl.get("A_mean") is not None and r["A"] > float(pl["A_mean"]))
    s = sp["summary"]["ndr_minus_nd"]
    ok3 = bool(len(sp["splits"]) >= MIN_SPLITS and s["A_mean"] > 0)
    se = float(r["A_boot"]["se_q"])
    dec = {"original": risk(float(r["A"]), se, sd4), "mean4": risk(mean4, se, sd4)}
    rep = {"original_tau0": risk(float(r["A"]), se, 0.0), "mean4_tau0": risk(mean4, se, 0.0), "worst_set": risk(float(min(v)), se, sd4),
           "worst_set_tau0": risk(float(min(v)), se, 0.0), "pr27n2_splits": risk(float(swap["sensitivity"]["123_pr27n2"]["splits_A"]["mean"]), se, sd4),
           "pr27n2_original": risk(float(swap["sensitivity"]["123_pr27n2"]["original"]["A"]), se, sd4)}
    ok4 = bool(all(d["p_lt_m002"] is not None and d["p_lt_m002"] <= RISK_PHARM for d in dec.values()))
    ok = ok_null and ok1 and ok2 and ok3 and ok4
    return {"rule": "safety-ndr rule (1)–(4)", "null_point_oof": ok_null,
            "c1_drawswap": {"sets": dict(zip(SWAP_SETS, v)), "mean4": mean4, "sd4": sd4, "n_pos": int(sum(x > 0 for x in v)), "ok": ok1},
            "c2_original": {"A": r["A"], "F": r["F"], "M": r["M"], "se_q": se, "placebo_mean": pl.get("A_mean"), "placebo_max": pl.get("A_max"),
                            "placebo_n": pl.get("n"), "ok": ok2},
            "c3_splits": {"n": len(sp["splits"]), "A_mean": s["A_mean"], "A_se": s["A_se"], "share_pos": s["share_A_pos"],
                          "placebo_A_mean": sp["summary"]["placebo_minus_nd"]["A_mean"], "ok": ok3},
            "c4_risk_rule": {"decisive": dec, "reported": rep, "p_harm_max": RISK_PHARM, "tau_d": sd4, "ok": ok4,
                             "convention": "against safety_nd; τ_d = spread of the four sets (safety-ndr rule (4)); decisive: the observation original (reading "
                                           "of the safety-nd rule) AND the mean of the four sets (reading of the draw swap); τ_d 0, worst set and pr27 n2 "
                                           "reported only"},
            "verdict": "deploy" if ok else "no deployment"}


def cmd_rule(a) -> int:
    C = C123.c120()
    for f in (OOF_JSON, SWAP_JSON, SPLITS_JSON):
        if not f.is_file():
            abort(f"{C.rel(f)} missing — run c130 oof, swap and splits first", 3)
    oof, swap, sp = (json.loads(f.read_text()) for f in (OOF_JSON, SWAP_JSON, SPLITS_JSON))
    if not (oof.get("inputs") == swap.get("inputs") == sp.get("inputs")):
        abort("oof, swap and splits were computed with different inputs")
    out = evaluate_rule(oof, swap, sp)
    OUT.mkdir(parents=True, exist_ok=True)
    RULE_JSON.write_text(C.to_json(out))
    c1, c2, c3, c4 = out["c1_drawswap"], out["c2_original"], out["c3_splits"], out["c4_risk_rule"]
    print(f"safety-ndr rule (1)–(4): copy null point OOF {'bit-identical' if out['null_point_oof'] else 'VIOLATED'} | (1) draw swap "
          f"{', '.join(f'{k} {v:+.5f}' for k, v in c1['sets'].items())} -> mean {c1['mean4']:+.5f}, {c1['n_pos']}/4 > 0 -> {'yes' if c1['ok'] else 'NO'} | "
          f"(2) original ½(F+M) {c2['A']:+.5f} ± {c2['se_q']:.5f} against the placebo mean {c2['placebo_mean']:+.5f} -> {'yes' if c2['ok'] else 'NO'} | "
          f"(3) {c3['n']} meta-splits {c3['A_mean']:+.5f} ± {c3['A_se']:.5f} -> {'yes' if c3['ok'] else 'NO'} | (4) risk rule τ_d {c4['tau_d']:.5f}: "
          f"P(θ<−0.002) original {c4['decisive']['original']['p_lt_m002']:.4f}, mean of the sets {c4['decisive']['mean4']['p_lt_m002']:.4f} "
          f"(posterior {c4['decisive']['mean4']['mean']:+.5f}; pr27 n2 {c4['reported']['pr27n2_splits']['p_lt_m002']:.4f}) -> {'yes' if c4['ok'] else 'NO'} "
          f"=> {out['verdict']}")
    return 0 if out["verdict"] == "deploy" else 5


# ------------------------------------------------------------------------------------------------ deployment file

def build_stage2(ctx: dict, oof: dict, rule: dict) -> dict:
    C = C123.c120()
    y, Kc, folds, n, seen = ctx["y"], S2.K_CAND, ctx["folds"], ctx["n"], ctx["seen"]
    std = C123.standard_forms(ctx)
    X, cand, names = None, np.zeros((n, Kc), np.int64), None
    for k in range(5):
        E = np.flatnonzero(folds == k)
        Xk, ck, names = block_ndr(ctx, E, std["set"][E], std["pre"][E])
        if X is None:
            X = np.zeros((n * Kc, Xk.shape[1]))
        X[(E[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)] = Xk
        cand[E] = ck
    got = C.sha256_array(X)
    o = oof["sets"]["ndr"]
    if got != o["set"].get("eval_features_sha256"):
        abort(f"null point (e) violated: features of the deployment form ({got[:16]}…) != evaluated rows of the OOF reproduction — rerun c130 oof")
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
    for f_, r_ in zip(C129.TEST_NOTES, C129.TEST_DRAWS):  # the test notes must belong to the test rankings of the draws (as in c129)
        b = r_.split("/")
        tm = json.loads((ctx["root"] / f_).with_suffix(".meta.json").read_text())
        if tm.get("source_rank") != f"runs/{b[1]}/{b[1]}/{b[2]}/{b[3]}" or tm.get("table") != "test":
            abort(f"{f_}: source {tm.get('source_rank')} is not the test ranking {r_}")
    te_rel, _te_table, te_src = PR27_NOTES["test_pr27"]
    te_file = ctx["root"] / te_rel
    te_meta = json.loads(te_file.with_suffix(".meta.json").read_text())
    te_sha = C.sha256_file(te_file)
    if te_meta.get("sha256") != te_sha or te_meta.get("table") != "test" or [s.get("source") for s in te_meta.get("sources") or []] != list(te_src):
        abort(f"{te_rel}: checksum, table or source do not match the meta file")
    cand_m = S2.candidates(std["set"], seen)
    sim_ref = NS.sim_stats(C129.train_top_sim(ctx))
    guard_ref = C129.guard_reference(ctx, cand_m)
    nd = json.loads(STAGE2_SAFETY_ND.read_text())
    if C.sha256_file(STAGE2_SAFETY_ND) != STAGE2_SAFETY_ND_SHA or nd.get("guard_reference") != json.loads(json.dumps(guard_ref, default=float)) \
            or nd.get("sim_reference") != json.loads(json.dumps(sim_ref, default=float)):
        abort("references of the note and similarity guards != those of safety_nd (stage2_safety_nd.json)")
    r_ref = NR.guard_stats(ctx["m3_stats"], ctx["pr_stats"], cand_m)
    rule_nd = dict(nd["guard_rule"])
    rule_nd.update({"r_rule": "safety-ndr rule (6)", "r_rel": NR.REL, "r_z_max": NR.Z_MAX,
                    "r_stats": ["pr27_codes_per_row (mean)", "pr27_share_top_mentioned (share)", "m3_last_pos_cand per draw (mean)"],
                    "text": "safety-ndr rule (6): widened note guard, similarity guard (band ±10 % OR |z| ≤ 3), guard of the columns R (band ±10 % OR "
                            "|z| ≤ 3) and guard (c) (½ to 2 times the OOF share of new top codes) each ONLY as an abort without a file, never as a "
                            "switch; class masses L1 ≤ 0.035 (build script)"})
    return {"kind": "second stage of the safety line with margin alignment as ONE set over the test rows: conditional logit over the first K "
                    "candidates (src/stage2.py, feature set fallback_i1_ndr = fallback_i1_nd + six regex-exact note columns R, src/note_regex.py) + "
                    "NONE guard (eda/c130_safety_ndr.py)",
            "stack": C.rel(C.STACK), "stack_sha256": C.sha256_file(C.STACK), "channels": list(ctx["stack"]["channels"]),
            "arith_beta": float(ctx["stack"]["arith_beta"]), "align_mode": "set", "align": C123.ALIGN, "align_min": C123.ALIGN_MIN,
            "K": Kc, "lambda": S2.LAMBDA, "null": False, "feature_set": S2.NDR_SET, "protect_none": True, "variant": "ndr",
            "draws_channel": C.M3, "draw_sources": ctx["draw_specs"], "draw_rank_source": meta["rank"]["source"], "test_draws": list(C129.TEST_DRAWS),
            "test_notes": [{"file": f_, "rank": r_} for f_, r_ in zip(C129.TEST_NOTES, C129.TEST_DRAWS)],
            "train_notes": [{"file": f_, "sha256": s_} for f_, s_ in zip(C129.TRAIN_NOTES, ctx["train_notes_sha256"])],
            "note_sim": nd["note_sim"],
            "r_notes": {"extractor": NR.VERSION, "features": list(S2.R_FEATS),
                        "m3": "M3 notes d1–d3 as test_notes / train_notes (mean or sum over the draws)",
                        "pr27_test": {"file": te_rel, "sha256": te_sha, "source": te_src[0]},
                        "pr27_train": {"file": PR27_NOTES["train_pr27"][0], "sha256": ctx["pr27_train_sha256"], "sources": list(PR27_NOTES["train_pr27"][2])}},
            "model": {"features": names, "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]],
                      "sd": [float(v) for v in model["sd"]], "gate": g, "abstain": g["abstain"],
                      "oof_changed_top_share": float(o["set"]["changed_top_share"]), "oof_changed_top3_share": float(o["set"]["changed_top3_share"]),
                      "oof_vs_base": {f: o[f]["vs_base"]["delta"] for f in FORMS}, "oof_vs_nd": {f: o[f]["vs_nd"]["delta"] for f in FORMS},
                      "features_sha256": got, "insample": ins},
            "novel_task_share_top5": float(o["set"]["novel_task_share_top5"]),
            "guard_reference": guard_ref, "sim_reference": sim_ref, "r_reference": r_ref, "guard_rule": rule_nd,
            "inputs_sha256": oof["inputs"], "rule_oof": rule,
            "based_on": {"stage2_safety_nd.json": C.sha256_file(STAGE2_SAFETY_ND), "decision": "decisions/override_cv_condition_ii.json",
                         "build": "tools/builds/safety_ndr.sh"},
            "order": "blend -> margin alignment (one set over the test rows) -> second stage over the first K -> NONE guard -> REL-01"}


def cmd_fit(a) -> int:
    C = C123.c120()
    target = Path(a.out) if a.out else STAGE2_FILE
    if target.resolve() != STAGE2_FILE.resolve() and not target.resolve().is_relative_to(OUT.resolve()):
        abort(f"fit only writes {C.rel(STAGE2_FILE)} or under {C.rel(OUT)}", 2)
    if not RULE_JSON.is_file():
        abort(f"{C.rel(RULE_JSON)} missing — run c130 rule first", 3)
    rule = json.loads(RULE_JSON.read_text())
    if rule.get("verdict") != "deploy":
        abort(f"pre-registered safety-ndr rule not met ({rule.get('verdict')}) — no deployment, no file", 5)
    ctx = ndr_context()
    oof = json.loads(OOF_JSON.read_text())
    if oof.get("inputs") != ctx["inputs_ndr"]:
        abort(f"{C.rel(OOF_JSON)} was computed with other inputs — recompute it")
    out = build_stage2(ctx, oof, rule)
    text = C.to_json(out)
    m = out["model"]
    print(f"  [safety_ndr {S2.NDR_SET}] {len(m['features'])} features | gate (guarded) inner {m['gate']['inner']:.5f} vs. base {m['gate']['inner_base']:.5f} -> "
          f"{'ABSTAIN (weight 0)' if m['abstain'] else 'apply'} | in-sample {m['insample']['map_base']:.5f} → {m['insample']['map_reranked_protected']:.5f} "
          f"(new top code {m['insample']['changed_top_share']:.4f}; OOF {m['oof_changed_top_share']:.4f}) | weights R "
          f"{', '.join(f'{nm} {w:+.3f}' for nm, w in zip(m['features'][-NR_COLS:], m['w'][-NR_COLS:]))} | reference R: pr27 codes per row "
          f"{out['r_reference']['pr27_codes_per_row']['mean']:.4f}, top code mentioned {out['r_reference']['pr27_share_top_mentioned']['share']:.4f}")
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


NULL_KEEP = tuple(C129.NULL_KEEP) + ("r_notes", "r_reference")


def cmd_null(a) -> int:
    """Null model (weight 0) for the deployment file: features and guards are computed, score 0 -> base order (copy null point)."""
    ref = json.loads(Path(a.source).read_text())
    nf = len(ref["model"]["features"])
    model = {"features": ref["model"]["features"], "w": [0.0] * nf, "mu": [0.0] * nf, "sd": [1.0] * nf, "abstain": True, "oof_changed_top_share": 0.0}
    Path(a.out).write_text(C123.c120().to_json({**{k: ref[k] for k in NULL_KEEP}, "null": True, "model": model, "null_from": str(a.source)}))
    print(f"null model for {a.source} -> {a.out}")
    return 0


def cmd_b_zero(a) -> int:
    """Copy null point (safety-ndr rule (5)): model of safety_nd (stage2_safety_nd.json, 58 features fallback_i1_nd) + weight 0 for the six
    R columns, otherwise the deployment file of safety_ndr (notes, reference, pr27 notes, guards) -> the test side must give the file of
    safety_nd byte-identical."""
    nd = json.loads(STAGE2_SAFETY_ND.read_text())
    if C123.c120().sha256_file(STAGE2_SAFETY_ND) != STAGE2_SAFETY_ND_SHA or nd.get("feature_set") != S2.ND_SET:
        abort(f"{STAGE2_SAFETY_ND.name} is not the deployment file of safety_nd", 4)
    ref = json.loads(Path(a.source).read_text())
    m_nd, mref = nd["model"], ref["model"]
    if mref["features"][:-NR_COLS] != m_nd["features"] or tuple(mref["features"][-NR_COLS:]) != S2.R_FEATS:
        abort("features of safety_nd and safety_ndr do not match", 4)
    out = json.loads(json.dumps(ref))
    out["model"].update({"w": m_nd["w"] + [0.0] * NR_COLS, "mu": m_nd["mu"] + [0.0] * NR_COLS, "sd": m_nd["sd"] + [1.0] * NR_COLS,
                         "oof_changed_top_share": m_nd["oof_changed_top_share"], "oof_changed_top3_share": m_nd["oof_changed_top3_share"]})
    out["b_zero_from"] = STAGE2_SAFETY_ND.name
    Path(a.out).write_text(C123.c120().to_json(out))
    print(f"copy null point (weights of safety_nd + 0 for {', '.join(S2.R_FEATS)}) -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-notes", help="pr27 notes (training n1/n2, test n1) = re-export from the runs")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    o = sub.add_parser("oof", help="nested reproduction ND+R against ND, null point, placebos -> artifacts/eda/private/safety_ndr/")
    o.add_argument("--placebos", type=int, default=len(PLACEBO_SEEDS))
    sub.add_parser("swap", help="full draw swap over the four sets from d1–d4 + sensitivity pr27 n2")
    s = sub.add_parser("splits", help="new meta-splits (ND+R) − ND")
    s.add_argument("--n", type=int, default=MIN_SPLITS)
    sub.add_parser("rule", help="pre-registered safety-ndr rule (1)–(4)")
    f = sub.add_parser("fit", help="deployment file artifacts/stage2_safety_ndr.json")
    f.add_argument("--out")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--verify", action="store_true")
    nl = sub.add_parser("null", help="null model for the deployment file")
    nl.add_argument("--out", required=True)
    nl.add_argument("--source", default=str(STAGE2_FILE))
    bz = sub.add_parser("b-zero", help="copy null point: weights of safety_nd + 0 for the columns R")
    bz.add_argument("--out", required=True)
    bz.add_argument("--source", default=str(STAGE2_FILE))
    a = ap.parse_args(argv)
    if a.cmd == "oof" and not 0 <= a.placebos <= len(PLACEBO_SEEDS):
        abort(f"--placebos {a.placebos}", 2)
    return {"export-notes": cmd_export_notes, "oof": cmd_oof, "swap": cmd_swap, "splits": cmd_splits,
            "rule": cmd_rule, "fit": cmd_fit, "null": cmd_null, "b-zero": cmd_b_zero}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
