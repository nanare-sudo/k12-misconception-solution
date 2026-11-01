"""THIRD M3 draw — evaluation according to the rule fixed in advance and deployment verdict for both lines.

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c108_m3_three_draws.py --status   # state of the third OOF draw (exit 0 = complete, 3 = not)
    python -I eda/c108_m3_three_draws.py            # evaluation and verdict; exit 3 with a message while d3 is missing
    python -I eda/c108_m3_three_draws.py --dry      # CODE-PATH CHECK: d3 := d2 — NO verdict, separate output files

DRAWS (training rows; per fold the adapter that did not learn this fold):
  d1  runs/m3_5f/m3_rank/bench                                     greedy, 3,200 tokens (the same draw as oof_m3p5)
  d2  runs/m3long_comb/m3l_rank/bench + runs/cv01_m3d2_f034/m3l_rank_f{0,3,4}/bench           greedy, 5,200 tokens
  d3  runs/m3d3_oof_rp/m3d3_rank_f{0..4}/bench (--d3)              sampled: temperature 0.6, top_p 0.95, top_k 20, seed 3; 5,200
  d1 and d2 are listed in artifacts/meta_m3x2_full_cv01.json (rank.source); m3x2 = mean(d1, d2) must match
  oof_m3x2_full_cv01.npy (SHA-256 e08f2dee…) bit-identically. m3x3 = mean(d1, d2, d3) via eda/a9_m3_draws_oof.py (load_draw,
  mean_of_draws, build) — the same computation as the deployed artifact m3x3_full and as tools/make_submission.py --rank-only
  on the test rows.
DEPLOYED ARTIFACT (built by tools/builds/s029_m3x3.sh; only checked here if it already exists):
    python -I eda/a9_m3_draws_oof.py m3x3_full --draw runs/m3_5f/m3_rank/bench \
      --draw runs/m3long_comb/m3l_rank/bench+runs/cv01_m3d2_f034/m3l_rank_f0/bench+runs/cv01_m3d2_f034/m3l_rank_f3/bench+runs/cv01_m3d2_f034/m3l_rank_f4/bench \
      --draw runs/m3d3_oof_rp/m3d3_rank_f0/bench+runs/m3d3_oof_rp/m3d3_rank_f1/bench+runs/m3d3_oof_rp/m3d3_rank_f2/bench+runs/m3d3_oof_rp/m3d3_rank_f3/bench+runs/m3d3_oof_rp/m3d3_rank_f4/bench \
      --rows-out derived/m3x3_full_rows.npy --ref m3p5

STEPS (training rows only, aggregates only):
 0. STATE of the third draw: every shard present, complete and exact (a8.load_rank), generation per fold (answered,
    failures, share of complete answers). If anything is missing: abort with a message (exit 3), nothing written.
 1. TECHNICAL CHECKS (abort with exit 4, nothing written): checksums of the base OOFs; m3x2 bit-identical to the deployed
    artifact; every draw covers all 1,500 rows; ranking of d3 as for d1/d2; generation of d3 as registered, system text,
    image and compute path as for d2; FOLD INTEGRITY and TRAINING REGIME per fold job of all draws (rows from exactly one
    fold, the adapter did not learn exactly this fold, gold-only, system text as in training, local adapter = merged one);
    recipes against the blend files of the primary and inductive lines (channels, floors, factor, β); ANCHORS: primary
    0.8198889 (plus weights per fold bit-identical to stack_cv01_best_l2.json), inductive 0.8120000; if
    artifacts/oof_m3x3_full.npy already exists, it must match m3x3.
 2. SINGLE CHANNELS d1, d2, d3, m3x2, m3x3: MAP@3 ± σ (rows, tasks), per fold, groups, top correct, NONE at the top, log
    loss; paired m3x3 − m3x2, d3 − d2, d3 − d1, d2 − d1; same top per pair; generation per draw.
 3. QUALITY GUARD: d3 alone worse than d2 alone by more than 0.02 with z ≥ 2 (σ over tasks) -> d3 is not used, NO
    deployment in either line.
 4. RECIPES m3x3 against m3x2: weights cross-fitted per fold from the other four folds (a2.fit_mix on log loss, then the
    reasoning factor), blend with arithmetic share β (make_submission.mix_channels), last step:
      primary      v1r, zs27q, think18nb ×1.4 (floor 0.01), pr27xm (0.05), M3 (0.05); β 0.2; margin alignment 0.75 on the
                   300 rows of the fold, target = frequencies of the other folds for classes with at least 3 rows
                   (transductive);
      inductive    v1r, zs27t, think18sb (0.01), pr27xm (0.05), M3 (0.05); β 0.4; inductive alignment 0.75 — class weights
                   from the blend of the other folds (masking before the floor as in make_submission.oof_mixture), target
                   as above.
    Δ MAP@3 with σ over rows and tasks, z, 95 %, P(Δ < −0.002), per fold (σ over folds), groups, log loss (after and before
    the last step); for information without the last step and (primary) alignment as one set.
 5. POSTERIOR (risk rule): leave-one-out family prior of class N (new draw; the candidate is not in the prior inventory),
    no selection correction (k 1, evaluation registered in advance), target = a new draw on the test side: τ_d 0.001 (MAIN);
    sensitivity τ_d 0 and 0.002, spike-and-slab, all eras, both lines pooled. Requires internal evaluation helpers
    (eda/c95_common.py) that are not part of this repository.
 6. VERDICT (fixed in advance): quality guard violated -> both NO. PRIMARY: Δ > 0 (expected-value rule) AND single
    channel m3x3 ≥ m3x2 − 0.002 — reading: Δ = the observed difference in the cross-fit; the sign of the posterior mean is
    reported next to it and flagged explicitly if it disagrees. INDUCTIVE: risk rule P(θ < −0.002) ≤ 0.05 (MAIN posterior of
    the line; not decidable with too few accepted simulations); the posterior mean and the threshold of the class "new text
    channel" (+0.0015) are reported only.
Output: artifacts/eda/private/m3_draws_eval/c108_m3_three_draws[_dry].{json,txt} and …_rows.npz (row vectors, private). The JSON
file is read by tools/builds/s029_m3x3.sh (verdict, sources, checksum of m3x3, cross values for the field cv of the blend
file).

--dry: d3 := d2 (the same probabilities). Checks all code paths on the real training artifacts without the third draw;
expectation: quality guard Δ exactly 0, m3x3 = (d1 + 2·d2)/3, Δ in both recipes close to zero (no new information;
effective number of draws 1.8 instead of 2, hence rather slightly negative). Everything marked as a CHECK, no verdict.
LIMITS: the same 1,500 labels as all measurements of the M3 axis (no new holdout); base OOFs not nested in the meta fold; the
test side is a new draw of a differently fitted model (full fit on all rows); prior from ten earlier measurements, τ_d is an
assumption.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import metric  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A9 = load("c108_a9", COMP / "eda" / "a9_m3_draws_oof.py")
C99 = load("c108_c99", COMP / "eda" / "c99_m3_deploy.py")
MS = load("c108_ms", COMP / "tools" / "make_submission.py")
A2, C21 = C99.A2, C99.C21

OUT = COMP / "artifacts" / "eda" / "private" / "m3_draws_eval"
NAME = "c108_m3_three_draws"
REF_CHANNEL = "m3x2_full_cv01"  # channel with two draws on all 1,500 rows
REF_SHA256 = "e08f2deefe90a9b3b903668222cb51aa3a07f6dd80705e8f2c078ff3b2302ef1"
ARTIFACT = "m3x3_full"  # deployed artifact (tools/builds/s029_m3x3.sh)
D3_RUN = "m3d3_oof_rp"
D3_DEFAULT = "+".join(f"runs/{D3_RUN}/m3d3_rank_f{k}/bench" for k in range(5))
D3_GEN = {"max_tokens": 5200, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 3}  # registered generation settings, configs/vlm/m3d3_oof_rp.yaml
BASE_SHA16 = {"v1r": "e1729c1fddad7ebb", "zs27q": "7a11b94751f6d458", "zs27t": "54e045dfe32c7be1", "think18nb": "9c3403b57d07a956",
              "think18sb": "275cace8a36ca757", "pr27xm": "bf3d1b58abfc57eb"}  # as in stack_cv01_best_l2 and the inductive-line blend file (oof_sha256)
FLOOR_M3 = 0.05
M3 = "m3"  # name of the M3 channel in the re-implementation (no effect on the arithmetic)
LINES = {
    # line labels (recorded in the reports)
    "best": {"label": "primary line (earlier recipe)", "names": ["v1r", "zs27q", "think18nb", "pr27xm"],
             "floors": {"think18nb": 0.01, "pr27xm": 0.05}, "think": "think18nb", "factor": 1.4, "beta": 0.2, "last": "trans",
             "anchor": 0.8198889, "stack": "cv01_best_l2", "last_label": "margin alignment 0.75 per fold (transductive)"},
    "safe": {"label": "inductive line (earlier recipe, cross-fit on 1,500 rows)", "names": ["v1r", "zs27t", "think18sb", "pr27xm"],
             "floors": {"think18sb": 0.01, "pr27xm": 0.05}, "think": None, "factor": 1.0, "beta": 0.4, "last": "ind",
             "anchor": 0.8120000, "stack": "v1r_zs27t_think18sb_pr27xm_m3x2_ar", "last_label": "inductive alignment 0.75 per fold"},
}
ANCHOR_TOL = 5e-8  # anchors given with seven digits (0.8198889 = 1,229.8333 / 1,500; 0.8120000 = 1,218 / 1,500)
GROUPS = ("NONE", "seen", "never seen", "new question")
BOOT, SEED = 10_000, 1_390_030
HARM = -0.002
# rule fixed in advance
GUARD_DROP, GUARD_Z = 0.02, 2.0  # quality guard: d3 − d2 < −0.02 with z ≤ −2
SINGLE_TOL = 0.002  # primary: single channel m3x3 not worse than m3x2 − 0.002
P_HARM = 0.05  # inductive: P(θ < −0.002) ≤ 0.05
C_TXT = 0.0015  # threshold of the posterior mean of the class "new text channel" — report only, not part of the rule
EXPECT = {"channel": (0.789, 0.797), "best": (0.0007, 0.0, 0.002), "safe": 0.0015}  # expectation registered in advance (1/K law)
TAU_MAIN, TAUS_SENS = 0.001, (0.0, 0.002)
TOL_BESTAND, TOL_MAX, MIN_ESS_FRAC, MIN_ESS_DECIDE = 0.0004, 0.0032, 0.000625, 200  # as in eda/c95_m3_full_eval.py
PROBE_PLAUSIBLE = 0.005  # --dry: |Δ| of the recipes within this range counts as "close to two identical draws"
# Run configurations outside configs/vlm/ (fallback). The second draw of folds 0/3/4 is configured in
# configs/vlm/cv01_m3d2_f034.yaml, which is always found first.
CONFIG_FALLBACK = {"cv01_m3d2_f034": "configs/vlm/cv01_m3d2_f034.yaml"}


class Technik(RuntimeError):
    """Technical finding: abort with exit 4, no verdict, nothing written."""


# ------------------------------------------------------------------------------------------------ helpers

def npy_sha256(P: np.ndarray) -> str:
    """SHA-256 of the .npy bytes of a float64 array — the same bytes as kaggle_core.artifacts.save_oof (np.save)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(P, dtype=np.float64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def context() -> dict:
    """Training table, labels, folds, tasks and groups (eda/c99_m3_deploy.context)."""
    return C99.context()


def rows_map(y: np.ndarray, P: np.ndarray, seen: np.ndarray) -> np.ndarray:
    return metric.map3_rows_from_proba(y, P, seen)


def rows_ll(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def cluster_boot(d: np.ndarray, q_code: np.ndarray, rows: np.ndarray | None = None, B: int = BOOT, seed: int = SEED) -> dict:
    """Mean of d over rows; σ, 95 % interval and P(mean < −0.002) bootstrapped over tasks (ratio estimator)."""
    rows = np.arange(len(d)) if rows is None else np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    qc = q_code[rows]
    s = np.bincount(qc, weights=d[rows], minlength=int(q_code.max()) + 1)
    c = np.bincount(qc, minlength=int(q_code.max()) + 1).astype(float)
    keep = c > 0
    s, c = s[keep], c[keep]
    idx = np.random.default_rng(seed).integers(0, len(s), size=(B, len(s)))
    vals = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(d[rows].mean()), "se_q": float(vals.std(ddof=1)), "ci95_q": [float(x) for x in np.quantile(vals, [0.025, 0.975])],
            "p_below_harm": float((vals < HARM).mean()), "p_le_0": float((vals <= 0).mean()), "n_rows": int(len(rows)), "n_q": int(len(s))}


def zval(m: float, s: float) -> float:
    return float(m / s) if s > 0 else 0.0


def paired(ctx: dict, ra: np.ndarray, rb: np.ndarray, la: np.ndarray | None = None, lb: np.ndarray | None = None,
           B: int = BOOT) -> dict:
    """Paired difference a − b on all rows: σ over rows (metric.paired_bootstrap) and tasks, z (tasks), per fold with σ over
    folds, groups, log-loss difference (optional), rows better / worse."""
    folds, grp, q = ctx["folds"], ctx["group"], ctx["q_code"]
    d = np.asarray(ra, float) - np.asarray(rb, float)
    cb = cluster_boot(d, q, B=B)
    pb = metric.paired_bootstrap(ra, rb)
    ks = sorted(set(folds.tolist()))
    pf = [float(d[folds == k].mean()) for k in ks]
    out = {"a": float(np.mean(ra)), "b": float(np.mean(rb)), **cb, "se_rows": float(pb["sigma"]), "z_q": zval(cb["mean"], cb["se_q"]),
           "per_fold": pf, "folds_positive": int(sum(v > 0 for v in pf)), "folds_negative": int(sum(v < 0 for v in pf)),
           "se_folds": float(np.std(pf, ddof=1) / np.sqrt(len(pf))) if len(pf) > 1 else 0.0,
           "rows_better": int((d > 0).sum()), "rows_worse": int((d < 0).sum()), "groups": {}}
    for g in GROUPS:
        m = grp == g
        if m.any():
            gb = cluster_boot(d, q, m, B=max(1000, B // 4))
            out["groups"][g] = {"n": int(m.sum()), "delta": gb["mean"], "se_q": gb["se_q"], "contrib": float(d[m].sum() / len(d))}
    if la is not None and lb is not None:
        lbt = cluster_boot(np.asarray(la, float) - np.asarray(lb, float), q, B=max(1000, B // 4))
        out.update(ll_a=float(np.mean(la)), ll_b=float(np.mean(lb)), ll_delta=lbt["mean"], ll_se_q=lbt["se_q"], ll_z_q=zval(lbt["mean"], lbt["se_q"]))
    return out


def fmt_pair(p: dict) -> str:
    pf = " ".join(f"{v:+.4f}" for v in p["per_fold"])
    g = " | ".join(f"{k} ({v['n']}) {v['delta']:+.4f}±{v['se_q']:.4f}" for k, v in p["groups"].items())
    ll = (f"; log loss {p['ll_a']:.4f} against {p['ll_b']:.4f}, Δ {p['ll_delta']:+.4f} ± {p['ll_se_q']:.4f} (z {p['ll_z_q']:+.2f})"
          if "ll_delta" in p else "")
    return (f"{p['a']:.5f} − {p['b']:.5f} = {p['mean']:+.5f} ± {p['se_q']:.5f} tasks (± {p['se_rows']:.5f} rows; z {p['z_q']:+.2f}; 95 % "
            f"{p['ci95_q'][0]:+.4f} … {p['ci95_q'][1]:+.4f}; P(Δ<−0.002) {p['p_below_harm']:.3f}); per fold {pf} ({p['folds_positive']}/5 positive, "
            f"σ over folds {p['se_folds']:.4f}); better/worse {p['rows_better']}/{p['rows_worse']}{ll}\n        groups: {g}")


# ------------------------------------------------------------------------------------------------ state and technical checks

def gen_meta(run: str, rank_meta: dict) -> dict | None:
    """Aggregates of the generation behind a ranking (sft.sft_from -> <tag>_meta.json in the same run)."""
    src = str((rank_meta.get("sft") or {}).get("sft_from") or "")
    if "/" not in src:
        return None
    job, tag = src.split("/", 1)
    g = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json")
    if g is None:
        return None
    st = g.get("states") or {}
    ans = int(g.get("answered") or 0)
    return {"rows": g.get("rows"), "answered": ans, "failed": int(g.get("failed") or 0), "complete": int(st.get("complete") or 0),
            "share_complete": (int(st.get("complete") or 0) / ans) if ans else None, "out_tokens_median": g.get("out_tokens_median"),
            "share_none_root": g.get("share_none_root"), "max_tokens": g.get("max_tokens")}


def shard_status(shard: str, n: int, seen: np.ndarray) -> dict:
    """State of ONE shard: files, probe complete and exact (a8.load_rank read in partial mode), generation (aggregates only)."""
    run, job, tag = A9.parse_source(shard)
    d = A9.run_dir(run) / job
    need = (f"{tag}_meta.json", f"{tag}_idx.npy", f"{tag}_proto_rank.npy", f"{tag}_proto_rank.done.npy")
    out: dict = {"shard": shard, "missing": [f"artifacts/runs/{run}/{run}/{job}/{f}" for f in need if not (d / f).is_file()], "ok": False}
    if not A9.run_dir(run).is_dir():
        out["missing"] = [f"artifacts/runs/{run}/{run}/ (run not fetched)"]
    if out["missing"]:
        return out
    try:
        _, _, st = A9.A8.load_rank(shard, n, seen, table="train", allow_partial=True)
    except SystemExit as e:
        out["error"] = str(e)
        return out
    out.update(rows=st["rows_probe"], done=st["rows_done"], exact=st["rows_exact"], mass_median=st["mass_median"],
               gen=gen_meta(run, A9.read_json(d / f"{tag}_meta.json") or {}))
    out["ok"] = bool(st["rows_probe"] > 0 and st["rows_done"] == st["rows_probe"] == st["rows_exact"])
    return out


def draw_status(spec: str, n: int, seen: np.ndarray) -> dict:
    """State of a draw made of shards (+): all shards ok and together all training rows."""
    shards = [shard_status(s, n, seen) for s in str(spec).split("+")]
    rows = sum(int(s.get("rows") or 0) for s in shards)
    return {"spec": spec, "shards": shards, "rows": rows, "ok": bool(all(s["ok"] for s in shards) and rows == n)}


def status_lines(st: dict, n: int) -> list[str]:
    out = [f"state of the third OOF draw ({len(st['shards'])} shards, together {n} rows required; found {st['rows']}):"]
    for s in st["shards"]:
        if s["missing"]:
            out.append(f"   {s['shard']}: MISSING ({', '.join(s['missing'])})")
            continue
        if s.get("error"):
            out.append(f"   {s['shard']}: ERROR {s['error']}")
            continue
        g = s.get("gen") or {}
        gtxt = (f"generation {g.get('answered')}/{g.get('rows')} answered, failures {g.get('failed')}, complete answers "
                f"{g['share_complete']:.3f}, output tokens median {g.get('out_tokens_median')}" if g.get("share_complete") is not None
                else "generation: meta file missing")
        out.append(f"   {s['shard']}: probe {s['done']}/{s['rows']} computed, {s['exact']} exact, mass median {s['mass_median']:.3f} -> "
                   f"{'COMPLETE' if s['ok'] else 'INCOMPLETE'}; {gtxt}")
    out.append(f"-> {'COMPLETE' if st['ok'] else 'INCOMPLETE'}")
    return out


def run_config(run: str) -> tuple[dict | None, str | None]:
    """Run configuration configs/vlm/<run>.yaml, otherwise the fallback path from CONFIG_FALLBACK. -> (content, relative path)."""
    import yaml

    cands = [COMP / "configs" / "vlm" / f"{run}.yaml"] + ([COMP / CONFIG_FALLBACK[run]] if run in CONFIG_FALLBACK else [])
    for f in cands:
        if f.is_file():
            return (yaml.safe_load(f.read_text()) or {}), str(f.relative_to(COMP))
    return None, None


def fold_check(spec: str, folds: np.ndarray, adapter_sha: dict) -> list[dict]:
    """FOLD INTEGRITY and TRAINING REGIME per fold job of a draw (shards and combined_from resolved): rows from exactly one fold k;
    model -> merge job (run configuration, out_model) -> training job (adapter_from) -> sft_recipe.json: fold == k (the adapter
    did NOT learn exactly fold k), gold-only (neither teacher_derived nor pool), system text as in the generation; local
    adapter = merged one (summary.json of the merge job; adapter_sha: per-file cache).
    -> per fold job: problems (findings that invalidate the measurement) and unknown (cannot be determined)."""
    out = []
    for shard in str(spec).split("+"):
        _, _, tag = A9.parse_source(shard)
        for c in A9.constituents(shard):
            r, j = c["run"], c["job"]
            rec: dict = {"run": r, "job": j, "model": c["model"], "problems": [], "unknown": []}
            idx = np.load(A9.run_dir(r) / j / f"{tag}_idx.npy")
            fs = sorted({int(v) for v in folds[idx]})
            rec.update(rows=int(len(idx)), row_folds=fs)
            if len(fs) != 1:
                rec["problems"].append(f"rows from several folds {fs}")
            cfg, src = run_config(r)
            rec["config"] = src
            if cfg is None:
                rec["unknown"].append(f"run configuration {r}.yaml missing")
                out.append(rec)
                continue
            merged = ((cfg.get("models") or {}).get(c["model"]) or {}).get("merged")
            jobs = [jb for w in cfg.get("workers") or [] for jb in w.get("jobs") or []]
            mj = [jb for jb in jobs if jb.get("kind") == "merge" and merged and jb.get("out_model") == merged]
            if len(mj) != 1:
                rec["unknown"].append(f"no unique merge job for model {c['model']!r} in {src}")
                out.append(rec)
                continue
            src_ad = str(mj[0]["adapter_from"])
            r2, j2 = src_ad[3:].split("/", 1) if src_ad.startswith("../") else (r, src_ad)
            rec.update(merge_job=mj[0]["name"], train=f"{r2}/{j2}")
            recipe = A9.read_json(A9.run_dir(r2) / j2 / "sft_recipe.json")
            if recipe is None:
                rec["unknown"].append(f"artifacts/runs/{r2}/{r2}/{j2}/sft_recipe.json missing")
            else:
                tf = recipe.get("fold")
                rec.update(train_fold=tf, gold_only=not (recipe.get("teacher_derived") or recipe.get("pool")),
                           train_system_sha256=recipe.get("system_sha256"), rows_used=recipe.get("rows_used"))
                if tf is None or len(fs) != 1 or int(tf) != fs[0]:
                    rec["problems"].append(f"adapter {r2}/{j2} holds out fold {tf}, the rows lie in fold {fs} (leak or wrong assignment)")
                if not rec["gold_only"]:
                    rec["problems"].append(f"adapter {r2}/{j2} with teacher targets (not gold-only)")
                gsys = (c.get("generation") or {}).get("system_sha256")
                if gsys and recipe.get("system_sha256") and gsys != recipe["system_sha256"]:
                    rec["problems"].append("system text of the generation differs from the system text of the training")
            summ = A9.read_json(A9.run_dir(r) / mj[0]["name"] / "summary.json")
            want = ((summ or {}).get("merge") or {}).get("adapter_sha256")
            ad = A9.run_dir(r2) / j2 / "adapter" / "adapter_model.safetensors"
            rec["adapter_matches_merge"] = None
            if want and ad.is_file():
                if str(ad) not in adapter_sha:
                    adapter_sha[str(ad)] = C99.sha256_file(ad)
                rec["adapter_matches_merge"] = adapter_sha[str(ad)] == want
                if not rec["adapter_matches_merge"]:
                    rec["problems"].append(f"local adapter {r2}/{j2} is not the merged one (checksum of merge job {mj[0]['name']})")
            out.append(rec)
    return out


def check_line_stacks(lines: dict, art: Path) -> list[str]:
    """Recipes of the re-implementation against the deployed blend files: channels (except the M3 name), floors, factor, β."""
    bad = []
    for key, L in lines.items():
        f = art / f"stack_{L['stack']}.json"
        if not f.is_file():
            bad.append(f"{key}: {f.name} missing")
            continue
        st = json.loads(f.read_text())
        ch = list(st.get("channels") or [])
        if ch[:-1] != L["names"] or not ch or not str(ch[-1]).startswith("m3x2"):
            bad.append(f"{key}: channels {ch} instead of {L['names']} + M3")
            continue
        fl = {k: float(v) for k, v in (st.get("floor") or {}).items()}
        if fl != {**L["floors"], ch[-1]: FLOOR_M3}:
            bad.append(f"{key}: floors {fl}")
        if (st.get("scaled") or {}) != ({L["think"]: L["factor"]} if L["think"] else {}):
            bad.append(f"{key}: factor {st.get('scaled')}")
        if float(st.get("arith_beta") or 0.0) != float(L["beta"]):
            bad.append(f"{key}: arith_beta {st.get('arith_beta')} instead of {L['beta']}")
    return bad


# ------------------------------------------------------------------------------------------------ recipe

def cross_fit(mats: dict[str, np.ndarray], names: list[str], floors: dict, think: str | None, factor: float, beta: float, last: str,
              y: np.ndarray, seen: np.ndarray, folds: np.ndarray) -> dict:
    """Cross-fit per fold k: weights (a2.fit_mix, log loss) only from the other folds F, then the reasoning factor; blend with
    arithmetic share β (make_submission.mix_channels) on fold k; last step
      trans  margin alignment 0.75 of the rows of k (weights on the evaluated set, target = frequencies of F, at least 3 rows),
      ind    inductive alignment 0.75: class weights from the blend for F (masking before the floor as in oof_mixture), target as for trans,
      none   no last step.
    No label from fold k enters its own prediction. -> {Q (after), P (before the last step), theta per fold}."""
    n, K = len(y), len(seen)
    fl = [float(floors.get(nm, 0.0)) for nm in names]
    L = C99.logs_of([mats[nm] for nm in names], fl, seen, mask_first=False)
    Lm = C99.logs_of([mats[nm] for nm in names], fl, seen, mask_first=True) if last == "ind" else None
    ti = names.index(think) if think and think in names else None
    Q, P = np.zeros((n, K)), np.zeros((n, K))
    thetas: dict[int, list[float]] = {}
    for k in sorted(set(folds.tolist())):
        F, E = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        th = np.asarray(A2.fit_mix([lg[F] for lg in L], None, y[F]), dtype=float).copy()
        if ti is not None:
            th[ti] *= factor
        st = {"theta": [float(v) for v in th], "arith_beta": float(beta)}
        PE = MS.mix_channels(st, [lg[E] for lg in L], None, seen)
        P[E] = PE
        tgt = C99.target_of(y, F, K)
        if last == "trans":
            Q[E] = C99.apply_last(PE, C21.fit_weights(PE, tgt), seen)
        elif last == "ind":
            Q[E] = C99.apply_last(PE, C21.fit_weights(MS.mix_channels(st, [lg[F] for lg in Lm], None, seen), tgt), seen)
        elif last == "none":
            Q[E] = PE
        else:
            raise ValueError(f"unknown last step {last}")
        thetas[int(k)] = [float(v) for v in th]
    return {"Q": Q, "P": P, "theta": thetas}


def line_cross(ctx: dict, base: dict, m3: np.ndarray, L: dict) -> dict:
    mats = {**base, M3: m3}
    return cross_fit(mats, L["names"] + [M3], {**L["floors"], M3: FLOOR_M3}, L["think"], L["factor"], L["beta"], L["last"],
                     ctx["y"], ctx["seen"], ctx["folds"])


# ------------------------------------------------------------------------------------------------ rules

def quality_guard(r_d3: np.ndarray, r_d2: np.ndarray, q_code: np.ndarray, B: int = BOOT) -> dict:
    """Quality guard: d3 alone worse than d2 alone by more than 0.02 with z ≥ 2 (σ over tasks) -> violated."""
    cb = cluster_boot(np.asarray(r_d3, float) - np.asarray(r_d2, float), q_code, B=B)
    z = zval(cb["mean"], cb["se_q"])
    return {"delta": cb["mean"], "se_q": cb["se_q"], "z": z, "ci95_q": cb["ci95_q"],
            "violated": bool(cb["mean"] < -GUARD_DROP and z <= -GUARD_Z), "rule": f"d3 − d2 < −{GUARD_DROP:g} and z ≤ −{GUARD_Z:g}"}


def decide(guard: dict, delta_best: float, single_m3x3: float, single_m3x2: float, post_best: dict | None, post_safe: dict | None) -> dict:
    """Verdict according to the rule fixed in advance. post_*: MAIN posterior (c95_common.combined_posterior) of the line."""
    g_ok = not guard["violated"]
    single_ok = bool(single_m3x3 >= single_m3x2 - SINGLE_TOL)
    best_ok = bool(g_ok and delta_best > 0 and single_ok)
    pb_mean = None if not post_best else post_best.get("mean")
    best = {"deploy": best_ok, "delta": float(delta_best), "delta_pos": bool(delta_best > 0), "single_ok": single_ok, "guard_ok": g_ok,
            "single_m3x3": float(single_m3x3), "single_m3x2": float(single_m3x2), "posterior_mean": pb_mean,
            "posterior_sign": None if pb_mean is None else bool(pb_mean > 0)}
    best["posterior_sign_disagrees"] = bool(best["posterior_sign"] is not None and best["posterior_sign"] != best["delta_pos"])
    safe = {"deploy": False, "guard_ok": g_ok, "decidable": False, "p_lt_m002": None, "posterior_mean": None}
    if post_safe and post_safe.get("decidable"):
        safe.update(decidable=True, p_lt_m002=float(post_safe["p_lt_m002"]), posterior_mean=float(post_safe["mean"]),
                    deploy=bool(g_ok and post_safe["p_lt_m002"] <= P_HARM), mean_ge_c_txt=bool(post_safe["mean"] >= C_TXT))
    return {"guard_violated": bool(guard["violated"]), "best": best, "safe": safe}


def verdict_lines(dec: dict, dry: bool) -> list[str]:
    tag = "CHECK (d3 := d2) — COMPUTATIONAL ONLY, NO VERDICT: " if dry else ""
    b, s = dec["best"], dec["safe"]
    out = [f"VERDICT (rule fixed in advance){' — CHECK' if dry else ''}:"]
    if dec["guard_violated"]:
        out.append(f"   {tag}QUALITY GUARD VIOLATED — d3 is not used (sampling hurts): NO deployment in either line")
    pm = "–" if b["posterior_mean"] is None else f"{b['posterior_mean']:+.5f}"
    out.append(f"   {tag}PRIMARY: {'DEPLOY' if b['deploy'] else 'NO DEPLOYMENT'} — Δ {b['delta']:+.5f} ({'> 0' if b['delta_pos'] else '≤ 0'}); "
               f"single channel m3x3 {b['single_m3x3']:.4f} against m3x2 {b['single_m3x2']:.4f} − {SINGLE_TOL:g} ({'met' if b['single_ok'] else 'missed'}); "
               f"guard {'held' if b['guard_ok'] else 'violated'}")
    out.append(f"      posterior mean of the line (sign reading) {pm}"
               + (" — DIFFERS FROM THE SIGN OF THE OBSERVED Δ (clarify the reading of the rule)" if b["posterior_sign_disagrees"] else ""))
    if dec.get("null_best") is not None:
        out.append(f"      null point (copy d3 := d2, report only): Δ {dec['null_best']:+.5f} — the real d3 adds {dec['new_best']:+.5f} on top")
    if not s["decidable"]:
        out.append(f"   {tag}INDUCTIVE: NOT DECIDABLE (too few accepted simulations) — no deployment")
    else:
        out.append(f"   {tag}INDUCTIVE: {'DEPLOY' if s['deploy'] else 'NO DEPLOYMENT'} — P(θ<−0.002) {s['p_lt_m002']:.3f} "
                   f"{'≤' if s['p_lt_m002'] <= P_HARM else '>'} {P_HARM:.2f}; guard {'held' if s['guard_ok'] else 'violated'}; posterior mean "
                   f"{s['posterior_mean']:+.5f} (threshold \"new text channel\" +{C_TXT:g}, report only: {'reached' if s['mean_ge_c_txt'] else 'not reached'})")
    if dec.get("null_safe") is not None:
        out.append(f"      null point (copy d3 := d2, report only): Δ {dec['null_safe']:+.5f} — the real d3 adds {dec['new_safe']:+.5f} on top")
    return out


# ------------------------------------------------------------------------------------------------ Posterior

def prior_all_eras(C95) -> tuple[np.ndarray, dict]:
    """NPMLE over all measurements of class N (all eras) — as in eda/c95_m3_full_eval.prior_all_eras (sensitivity)."""
    xs, ss, ids = [], [], []
    for (i, _nm, c, db, sb, ds, s2, _st, _pre, _note) in C95.EB.INVENTAR:
        if c != "N":
            continue
        xo, so = (db, sb) if db is not None else (ds, s2)
        xs.append(xo)
        ss.append(so)
        ids.append(i)
    w = C95.EB.npmle(np.array(xs), np.array(ss))
    return w, {"n": len(ids), "ids": ids, **C95.EB.describe_prior(w)}


def run_post(C95, o: float, s: float, w: np.ndarray, rng: np.random.Generator, tau: float, n_sim: int, chunks: int) -> dict:
    """combined_posterior with k = 1 (no selection); the acceptance window only grows with too few accepted draws (as in c95_m3_full_eval)."""
    min_ess = max(MIN_ESS_DECIDE, MIN_ESS_FRAC * n_sim * chunks)
    tol = TOL_BESTAND
    p = C95.combined_posterior(float(o), float(s), w, rng, k=1, rho=0.0, tau_d=tau, n_sim=n_sim, chunks=chunks, tol=tol)
    while (p.get("ess") or 0) < min_ess and tol < TOL_MAX:
        tol = min(2 * tol, TOL_MAX)
        p = C95.combined_posterior(float(o), float(s), w, rng, k=1, rho=0.0, tau_d=tau, n_sim=n_sim, chunks=chunks, tol=tol)
    p.pop("settings", None)
    p.update(tol=tol, tol_widened=bool(tol != TOL_BESTAND), tau_d=tau, obs=float(o), se=float(s),
             decidable=bool(p.get("mean") is not None and (p.get("ess") or 0) >= MIN_ESS_DECIDE))
    return p


def r2_posteriors(ctx: dict, d_best: np.ndarray, d_safe: np.ndarray, se: dict, n_sim: int, chunks: int, seed: int = SEED) -> dict:
    """Risk rule per line (MAIN: leave-one-out prior class N, k 1, τ_d 0.001) with sensitivities; plus both lines pooled (GLS
    with the covariance from the same task bootstrap)."""
    C95 = load("c108_c95", COMP / "eda" / "c95_common.py")
    w_loo, p_loo = C95.family_prior((), cls="N")
    w_ss, p_ss = C95.family_prior_ss((), cls="N")
    w_all, p_all = prior_all_eras(C95)
    rng = np.random.default_rng(seed)
    obs = {"best": float(d_best.mean()), "safe": float(d_safe.mean())}
    out: dict = {"priors": {"LOO N": {k: v for k, v in p_loo.items()}, "spike-and-slab N": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p_ss.items()},
                            "all eras N": p_all}, "lines": {}}
    for key in ("best", "safe"):
        o, s = obs[key], se[key]
        grid = {f"MAIN: LOO N, k 1, τ_d {TAU_MAIN:g}": run_post(C95, o, s, w_loo, rng, TAU_MAIN, n_sim, chunks)}
        for tau in TAUS_SENS:
            grid[f"LOO N, k 1, τ_d {tau:g}"] = run_post(C95, o, s, w_loo, rng, tau, n_sim, chunks)
        grid[f"spike-and-slab N, τ_d {TAU_MAIN:g}"] = run_post(C95, o, s, w_ss, rng, TAU_MAIN, n_sim, chunks)
        grid[f"all eras N (n {p_all['n']}), τ_d {TAU_MAIN:g}"] = run_post(C95, o, s, w_all, rng, TAU_MAIN, n_sim, chunks)
        out["lines"][key] = grid
    cov = C95.joint_cluster_cov(ctx, [d_best, d_safe], seed=seed, draws=4000)
    oz, sz, wz = C95.gls_pool(np.array([obs["best"], obs["safe"]]), cov)
    out["pooled"] = {"obs": oz, "se": sz, "weights": [float(v) for v in wz], "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])) if cov[0, 0] > 0 and cov[1, 1] > 0 else None,
                     "posterior": run_post(C95, oz, sz, w_loo, rng, TAU_MAIN, n_sim, chunks)}
    out["boundaries"] = boundaries(C95, w_loo, se, rng, n_sim=max(1, n_sim // 2), chunks=max(1, chunks // 2))
    return out


def main_post(grid: dict) -> dict:
    return next(v for k, v in grid.items() if k.startswith("MAIN"))


# ------------------------------------------------------------------------------------------------ inputs

def load_base(art: Path | None = None) -> dict[str, np.ndarray]:
    """Base OOFs of both lines with checksums as deployed."""
    art = COMP / "artifacts" if art is None else art
    mats = {}
    for nm, want in BASE_SHA16.items():
        f = art / f"oof_{nm}.npy"
        got = C99.sha256_file(f)[:16] if f.is_file() else "missing"
        if got != want:
            raise Technik(f"artifacts/oof_{nm}.npy: checksum {got} instead of {want} (deployed state)")
        mats[nm] = np.load(f)
    return mats


def load_draws(ctx: dict, d3_spec: str, dry: bool, art: Path | None = None) -> dict:
    """d1, d2 from the sources in meta_m3x2_full_cv01.json, d3 from d3_spec (dry: d3 := d2) — each via a9.load_draw. Checks: m3x2
    bit-identical to the artifact, every draw on all rows; without --dry additionally a9.build over all three sources (guards
    against duplicate shards and fold jobs) bit-identical to the own mean. -> sources, matrices, statistics, m3x2, m3x3."""
    art = COMP / "artifacts" if art is None else art
    n, seen = ctx["n"], ctx["seen"]
    meta = A9.read_json(art / f"meta_{REF_CHANNEL}.json") or {}
    specs = str((meta.get("rank") or {}).get("source") or "").split(",")
    if len(specs) != 2 or not all(specs):
        raise Technik(f"meta_{REF_CHANNEL}.json: rank.source does not name exactly two draws")
    try:
        loaded = [A9.load_draw(s, n, seen) for s in specs]
        if dry:
            P2, h2, i2 = loaded[1]
            loaded.append((P2.copy(), h2.copy(), {**i2, "source": f"CHECK: d3 := d2 ({specs[1]})", "probe": True}))
        else:
            loaded.append(A9.load_draw(d3_spec, n, seen))
    except SystemExit as e:
        raise Technik(f"eda/a9_m3_draws_oof.load_draw: {e}") from None
    for j, (_, h, info) in enumerate(loaded, 1):
        if not h.all():
            raise Technik(f"draw d{j} ({info['source']}) covers only {int(h.sum())} of {n} training rows")
    K = len(seen)
    m3x2, cnt2 = A9.mean_of_draws([(P, h) for P, h, _ in loaded[:2]], n, K)
    m3x3, cnt3 = A9.mean_of_draws([(P, h) for P, h, _ in loaded], n, K)
    ref = np.load(art / f"oof_{REF_CHANNEL}.npy")
    ref_sha = C99.sha256_file(art / f"oof_{REF_CHANNEL}.npy")
    if ref_sha != REF_SHA256 or not np.array_equal(m3x2, ref):
        raise Technik(f"mean(d1, d2) does not match oof_{REF_CHANNEL}.npy bit-identically (file SHA {ref_sha[:16]}…, expected {REF_SHA256[:16]}…)")
    built = None
    if not dry:
        try:
            Pb, cntb, _, _ = A9.build(specs + [d3_spec], n, seen)
        except SystemExit as e:
            raise Technik(f"eda/a9_m3_draws_oof.build: {e}") from None
        if not np.array_equal(Pb, m3x3) or not (cntb == 3).all():
            raise Technik("a9.build over three draws differs from the own mean")
        built = True
    return {"specs": specs + [d3_spec], "P": [P for P, _, _ in loaded], "info": [i for _, _, i in loaded], "m3x2": m3x2, "m3x3": m3x3,
            "rows_three": int((cnt3 == 3).sum()), "built_a9": built}


def technical_checks(ctx: dict, D: dict, dry: bool, art: Path | None = None) -> dict:
    """Ranking and generation of d3 against d1/d2 and the registered settings, fold integrity and training regime per fold job,
    deployed artifact (if present)."""
    art = COMP / "artifacts" if art is None else art
    infos = D["info"]
    out: dict = {"problems": [], "warnings": []}
    rr = [json.dumps(i.get("rank_recipe"), sort_keys=True, default=str) for i in infos]
    if len(set(rr)) != 1:
        out["problems"].append(f"ranking of the draws set differently: {sorted(set(rr))}")
    g2, g3 = infos[1].get("generation") or {}, infos[2].get("generation") or {}
    out["gen_d3"] = g3
    if not dry:
        s3 = g3.get("settings") or {}
        miss = {k: (s3.get(k), v) for k, v in D3_GEN.items() if s3.get(k) != v}
        if miss:
            out["problems"].append(f"generation of d3 differs from the registered settings (actual, registered): {miss}")
        for k in ("system_sha256", "image", "backend"):
            if g3.get(k) != g2.get(k):
                out["problems"].append(f"generation of d3: {k} {g3.get(k)!r} instead of {g2.get(k)!r} (as for d2)")
    cache: dict = {}
    out["fold_checks"] = {}
    for j, spec in enumerate(D["specs"][: 2 if dry else 3], 1):
        recs = fold_check(spec, ctx["folds"], cache)
        out["fold_checks"][f"d{j}"] = recs
        for r in recs:
            for p in r["problems"]:
                out["problems"].append(f"d{j} {r['run']}/{r['job']}: {p}")
            for u in r["unknown"]:
                (out["problems"] if j == 3 else out["warnings"]).append(f"d{j} {r['run']}/{r['job']}: cannot be determined — {u}")
        folds_cov = sorted(f for r in recs for f in r["row_folds"])
        if folds_cov != sorted(set(ctx["folds"].tolist())):
            out["problems"].append(f"d{j}: fold jobs cover the folds {folds_cov} instead of all exactly once")
    meta_art = A9.read_json(art / f"meta_{ARTIFACT}.json")
    out["artifact"] = None
    if meta_art is not None and (art / f"oof_{ARTIFACT}.npy").is_file():
        same = bool(np.array_equal(np.load(art / f"oof_{ARTIFACT}.npy"), D["m3x3"]))
        src_ok = str((meta_art.get("rank") or {}).get("source")) == ",".join(D["specs"])
        out["artifact"] = {"bit_identical": same, "source_ok": src_ok}
        if not dry and not (same and src_ok):
            out["problems"].append(f"artifacts/oof_{ARTIFACT}.npy exists but is not the evaluated mean (bit-identical {same}, sources {src_ok})")
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def channel_stats(ctx: dict, P: np.ndarray, B: int = BOOT) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    y, seen, folds, grp, q = ctx["y"], ctx["seen"], ctx["folds"], ctx["group"], ctx["q_code"]
    r, ll = rows_map(y, P, seen), rows_ll(y, P)
    top = metric.top3_indices(P, seen)[:, 0]
    cb = cluster_boot(r, q, B=B)
    st = {"map3": float(r.mean()), "se_rows": float(metric.bootstrap_sigma(r)), "se_q": cb["se_q"],
          "per_fold": [float(r[folds == k].mean()) for k in sorted(set(folds.tolist()))],
          "groups": {g: float(r[grp == g].mean()) for g in GROUPS if (grp == g).any()},
          "top1": float((top == y).mean()), "none_top": float((top == 0).mean()), "logloss": float(ll.mean())}
    return st, r, ll, top


def null_point(ctx: dict, base: dict, D: dict, lines: dict, rows: dict, B: int = BOOT) -> dict:
    """Duplicate control: m3 = mean(d1, d2, copy of d2 or d1) — the same arithmetic (a9.mean_of_draws), the same procedure.
    copy − m3x2 is the null point of a third draw without new information, m3x3 − copy the gain of the real d3 on top."""
    y, seen, n = ctx["y"], ctx["seen"], ctx["n"]
    P1, P2, _ = D["P"]
    hs = np.ones(n, bool)
    out = {}
    for cname, parts in (("copy d3 := d2", (P1, P2, P2)), ("copy d3 := d1", (P1, P2, P1))):
        Mc, _ = A9.mean_of_draws([(P, hs) for P in parts], n, len(seen))
        rec = {"single": float(rows_map(y, Mc, seen).mean()), "lines": {}}
        for key, L in lines.items():
            rc = rows_map(y, line_cross(ctx, base, Mc, L)["Q"], seen)
            rec["lines"][key] = {"null": paired(ctx, rc, rows[f"{key}_m3x2"], B=max(1000, B // 4)),
                                 "new": paired(ctx, rows[f"{key}_m3x3"], rc, B=max(1000, B // 4)), "cross": float(rc.mean())}
        out[cname] = rec
    return out


def boundaries(C95, w: np.ndarray, se: dict, rng: np.random.Generator, n_sim: int = 200_000, chunks: int = 4, step: float = 0.00025,
               lo: float = -0.002, hi: float = 0.012) -> dict:
    """Resolution of the rules at the measured σ (MAIN posterior): smallest observed Δ on the grid from which on
    P(θ<−0.002) ≤ 0.05 (inductive) or a posterior mean > 0 (primary, sign reading) holds; None = not on the grid."""
    grid = np.round(np.arange(lo, hi + step / 2, step), 6)
    out: dict = {"step": step, "n_sim": n_sim * chunks, "safe_p_harm": None, "best_mean_pos": None}
    for o in grid:
        if out["safe_p_harm"] is None:
            p = run_post(C95, float(o), se["safe"], w, rng, TAU_MAIN, n_sim, chunks)
            if p["decidable"] and p["p_lt_m002"] <= P_HARM:
                out["safe_p_harm"] = float(o)
        if out["best_mean_pos"] is None:
            p = run_post(C95, float(o), se["best"], w, rng, TAU_MAIN, n_sim, chunks)
            if p["decidable"] and p["mean"] > 0:
                out["best_mean_pos"] = float(o)
        if out["safe_p_harm"] is not None and out["best_mean_pos"] is not None:
            break
    return out


def fmt_opt(v: float | None) -> str:
    return "outside the grid" if v is None else f"{v:+.4f}"


def evaluate(ctx: dict, base: dict, D: dict, dry: bool, lines: dict | None = None, anchors: bool = True, n_sim: int = 400_000,
             chunks: int = 8, say=print, art: Path | None = None, B: int = BOOT) -> tuple[dict, dict]:
    """Steps 2–6 (module docstring). anchors=False only for synthetic tests. -> (result, row vectors)."""
    art = COMP / "artifacts" if art is None else art
    lines = LINES if lines is None else lines
    y, seen, n = ctx["y"], ctx["seen"], ctx["n"]
    P1, P2, P3 = D["P"]
    probe = "CHECK (d3 := d2) — " if dry else ""
    res: dict = {"single": {}, "pairs_single": {}, "lines": {}}
    rows: dict[str, np.ndarray] = {}
    # ---------------------------------------------------------------- 2. single channels
    say(f"2. {probe}SINGLE CHANNELS on all {n} rows (MAP@3 ± σ rows / tasks):")
    tops = {}
    for nm, P in (("d1", P1), ("d2", P2), ("d3", P3), ("m3x2", D["m3x2"]), ("m3x3", D["m3x3"])):
        st, r, ll, top = channel_stats(ctx, P, B=B)
        res["single"][nm] = st
        rows[f"single_{nm}"], rows[f"single_ll_{nm}"], tops[nm] = r, ll, top
        say(f"   {nm:<5} {st['map3']:.4f} ± {st['se_rows']:.4f} / {st['se_q']:.4f}; per fold {' '.join(f'{v:.4f}' for v in st['per_fold'])}; "
            f"top correct {st['top1']:.3f}, NONE at the top {st['none_top']:.3f} (true {np.mean(y == 0):.3f}), log loss {st['logloss']:.4f}; "
            + " | ".join(f"{g} {v:.3f}" for g, v in st["groups"].items()))
    for a, b in (("m3x3", "m3x2"), ("d3", "d2"), ("d3", "d1"), ("d2", "d1")):
        p = paired(ctx, rows[f"single_{a}"], rows[f"single_{b}"], rows[f"single_ll_{a}"], rows[f"single_ll_{b}"], B=B)
        p["same_top"] = float((tops[a] == tops[b]).mean())
        res["pairs_single"][f"{a} − {b}"] = p
        say(f"   {a} − {b}: {fmt_pair(p)}\n        same top {p['same_top']:.3f}")
    res["same_top_all_three"] = float(((tops["d1"] == tops["d2"]) & (tops["d2"] == tops["d3"])).mean())
    lo, hi = EXPECT["channel"]
    say(f"   expectation (1/K law): channel {lo:.3f} -> ≈ {hi:.3f}; measured m3x2 {res['single']['m3x2']['map3']:.4f} -> m3x3 "
        f"{res['single']['m3x3']['map3']:.4f}; same top in all three draws {res['same_top_all_three']:.3f}")
    # ---------------------------------------------------------------- 3. quality guard
    guard = quality_guard(rows["single_d3"], rows["single_d2"], ctx["q_code"], B=B)
    res["guard"] = guard
    say(f"3. {probe}QUALITY GUARD ({guard['rule']}): d3 − d2 = {guard['delta']:+.4f} ± {guard['se_q']:.4f} (z {guard['z']:+.2f}) -> "
        f"{'VIOLATED — d3 is not used' if guard['violated'] else 'held'}")
    # ---------------------------------------------------------------- 4. recipes
    say(f"4. {probe}RECIPES m3x3 against m3x2 (weights cross-fitted per fold, floors, β, last step as deployed):")
    deltas = {}
    for key, L in lines.items():
        c2 = line_cross(ctx, base, D["m3x2"], L)
        c3 = line_cross(ctx, base, D["m3x3"], L)
        r2, r3 = rows_map(y, c2["Q"], seen), rows_map(y, c3["Q"], seen)
        a2v = float(r2.mean())
        rec: dict = {"label": L["label"], "last": L["last_label"], "beta": L["beta"], "factor": L["factor"], "cross_m3x2": a2v, "cross_m3x3": float(r3.mean()),
                     "anchor": L["anchor"], "theta_m3x2": c2["theta"], "theta_m3x3": c3["theta"]}
        if anchors:
            ok = abs(a2v - L["anchor"]) < ANCHOR_TOL
            rec["anchor_ok"] = bool(ok)
            if not ok:
                raise Technik(f"{L['label']}: anchor {L['anchor']} missed (recomputed {a2v:.9f}) — check the re-implementation")
            if key == "best":
                st = json.loads((art / f"stack_{L['stack']}.json").read_text())
                ft = (st.get("cross") or {}).get("fold_theta") or {}
                dev = max(float(np.abs(np.array(c2["theta"][k]) - np.array(ft[str(k)])).max()) for k in c2["theta"]) if ft else None
                rec["fold_theta_dev_vs_stack"] = dev
                if dev is None or dev > 1e-12:
                    raise Technik(f"{L['label']}: weights per fold differ from {L['stack']} (largest deviation {dev})")
        ll2, ll3 = rows_ll(y, c2["Q"]), rows_ll(y, c3["Q"])
        p = paired(ctx, r3, r2, ll3, ll2, B=B)
        rec["delta"] = p
        rows[f"{key}_m3x2"], rows[f"{key}_m3x3"] = r2, r3
        deltas[key] = r3 - r2
        # for information: without the last step; log loss before the last step
        q2, q3 = rows_map(y, c2["P"], seen), rows_map(y, c3["P"], seen)
        rec["info_without_last_step"] = paired(ctx, q3, q2, rows_ll(y, c3["P"]), rows_ll(y, c2["P"]), B=max(1000, B // 4))
        if L["last"] == "trans":  # alignment as ONE set (target all rows; closer to the 3,498 test rows, slightly optimistic)
            tgt = C99.target_of(y, np.arange(n), len(seen))
            s2 = rows_map(y, C99.apply_last(c2["P"], C21.fit_weights(c2["P"], tgt), seen), seen)
            s3 = rows_map(y, C99.apply_last(c3["P"], C21.fit_weights(c3["P"], tgt), seen), seen)
            rec["info_one_set"] = paired(ctx, s3, s2, B=max(1000, B // 4))
        res["lines"][key] = rec
        anc = (f"anchor {L['anchor']:.7f} EXACT" + (", weights per fold bit-identical " + L["stack"] if "fold_theta_dev_vs_stack" in rec else "")
               if rec.get("anchor_ok") else "anchor not checked")
        say(f"   {L['label']} — {L['last_label']}, β {L['beta']:g}{', ' + L['think'] + ' ×' + format(L['factor'], 'g') if L['think'] else ''}: "
            f"m3x2 {a2v:.7f} ({anc}) -> m3x3 {rec['cross_m3x3']:.7f}")
        say(f"     m3x3 − m3x2: {fmt_pair(p)}")
        w = rec["info_without_last_step"]
        say(f"     (information) without last step: Δ {w['mean']:+.5f} ± {w['se_q']:.5f}; log loss before Δ {w['ll_delta']:+.4f} ± {w['ll_se_q']:.4f}"
            + (f"; alignment as one set: Δ {rec['info_one_set']['mean']:+.5f} ± {rec['info_one_set']['se_q']:.5f}" if "info_one_set" in rec else ""))
    if {"best", "safe"} - set(res["lines"]):
        raise Technik("the evaluation requires both lines (best, safe)")
    # ---------------------------------------------------------------- 4b. null point (duplicate control; report only)
    res["null_point"] = null_point(ctx, base, D, lines, rows, B=B)
    say(f"4b. {probe}NULL POINT (report only, not part of the rule): a third \"draw\" WITHOUT new information (copy of d2 or d1) "
        "with the same procedure — what the re-weighting alone brings, and what the real d3 adds on top:")
    for cname, rec in res["null_point"].items():
        say(f"   {cname}: single channel {rec['single']:.4f}; " + "; ".join(
            f"{key} copy − m3x2 {v['null']['mean']:+.5f} ± {v['null']['se_q']:.5f} ({v['null']['folds_positive']}/5 folds positive), "
            f"m3x3 − copy {v['new']['mean']:+.5f} ± {v['new']['se_q']:.5f}" for key, v in rec["lines"].items()))
    exp_b = EXPECT["best"]
    say(f"   expectation: primary {exp_b[0]:+.4f} ({exp_b[1]:+.3f} to {exp_b[2]:+.3f}), inductive {EXPECT['safe']:+.4f}; measured "
        f"{res['lines']['best']['delta']['mean']:+.4f} / {res['lines']['safe']['delta']['mean']:+.4f}")
    # ---------------------------------------------------------------- 5. Posterior
    se = {k: res["lines"][k]["delta"]["se_q"] for k in ("best", "safe")}
    post = r2_posteriors(ctx, deltas["best"], deltas["safe"], se, n_sim, chunks)
    res["posterior"] = post
    pr = post["priors"]["LOO N"]
    say(f"5. {probe}POSTERIOR (prior leave-one-out class N, n {pr['n']}: mean {pr['mean']:+.4f}, support points "
        + ", ".join(f"{g:+.4f}:{q:.2f}" for g, q in pr["support"]) + "; k 1; target: a new draw on the test side):")
    for key in ("best", "safe"):
        for lab, p in post["lines"][key].items():
            say(f"   {key:<5} {lab:<36} beob. {p['obs']:+.5f} ± {p['se']:.5f} -> " + (
                f"mean {p['mean']:+.5f}, P(θ>0) {p['p_pos']:.3f}, P(θ<−0.002) {p['p_lt_m002']:.3f}, P(θ≥0.002) {p['p_ge_002']:.3f} (ESS {p['ess']:.0f}"
                + (f", window {p['tol']:g}" if p["tol_widened"] else "") + ")" if p.get("mean") is not None else "no simulation accepted"))
    pp = post["pooled"]
    q = pp["posterior"]
    corr = "–" if pp["corr"] is None else f"{pp['corr']:+.2f}"
    say(f"   pooled (GLS, weights B/S {pp['weights'][0]:.2f}/{pp['weights'][1]:.2f}, correlation {corr}) {pp['obs']:+.5f} ± {pp['se']:.5f} -> "
        + (f"mean {q['mean']:+.5f}, P(θ<−0.002) {q['p_lt_m002']:.3f} (sensitivity)" if q.get("mean") is not None else "no simulation accepted"))
    bd = post["boundaries"]
    say(f"   RESOLUTION OF THE RULES (report only; MAIN posterior, grid {bd['step']:g}, {bd['n_sim']} simulations per point): inductive "
        f"P(θ<−0.002) ≤ {P_HARM:g} only from an observed Δ ≈ {fmt_opt(bd['safe_p_harm'])} (at σ {se['safe']:.5f}); primary line: posterior mean > 0 "
        f"(sign reading) only from Δ ≈ {fmt_opt(bd['best_mean_pos'])} (at σ {se['best']:.5f}); expectation {EXPECT['safe']:+.4f} / {EXPECT['best'][0]:+.4f}")
    # ---------------------------------------------------------------- 6. verdict
    dec = decide(guard, res["lines"]["best"]["delta"]["mean"], res["single"]["m3x3"]["map3"], res["single"]["m3x2"]["map3"],
                 main_post(post["lines"]["best"]), main_post(post["lines"]["safe"]))
    npd = res["null_point"]["copy d3 := d2"]["lines"]
    dec.update(null_best=npd["best"]["null"]["mean"], new_best=npd["best"]["new"]["mean"], null_safe=npd["safe"]["null"]["mean"],
               new_safe=npd["safe"]["new"]["mean"])
    res["decision"] = dec
    for s in verdict_lines(dec, dry):
        say(s)
    if dry:
        b, sf = res["lines"]["best"]["delta"]["mean"], res["lines"]["safe"]["delta"]["mean"]
        ok = guard["delta"] == 0.0 and abs(b) <= PROBE_PLAUSIBLE and abs(sf) <= PROBE_PLAUSIBLE
        res["probe"] = {"guard_delta_zero": bool(guard["delta"] == 0.0), "best": b, "safe": sf, "plausible": bool(ok),
                        "expected": "Δ close to zero (two identical draws: no new information, effective number 1.8 instead of 2)"}
        say(f"CHECK: guard Δ exactly 0: {guard['delta'] == 0.0}; Δ primary {b:+.5f}, inductive {sf:+.5f} — "
            f"{'CLOSE TO TWO IDENTICAL DRAWS (|Δ| ≤ ' + format(PROBE_PLAUSIBLE, 'g') + '), code paths run' if ok else 'CONSPICUOUS — check the code paths'}")
    return res, rows


# ------------------------------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d3", default=D3_DEFAULT, help="third OOF draw: runs/<run>/<job>/<tag>[+… more shards] (default: m3d3_oof_rp, five folds)")
    ap.add_argument("--dry", action="store_true", help="CODE-PATH CHECK: d3 := d2, no verdict, output …_dry")
    ap.add_argument("--status", action="store_true", help="only report the state of the third OOF draw (exit 0 = complete, 3 = not)")
    ap.add_argument("--n-sim", type=int, default=400_000, help="simulations per block of the posterior (c95_common.combined_posterior)")
    ap.add_argument("--chunks", type=int, default=8, help="blocks of the posterior simulation")
    ap.add_argument("--out-dir", help="output folder (default artifacts/eda/private/m3_draws_eval)")
    a = ap.parse_args(argv)
    t0 = time.time()
    L: list[str] = []

    def say(s: str = "") -> None:
        L.append(s)
        print(s, flush=True)

    ctx = context()
    if a.status or not a.dry:
        st = draw_status(a.d3, ctx["n"], ctx["seen"])
        for s in status_lines(st, ctx["n"]):
            say(s)
        if a.status:
            return 0 if st["ok"] else 3
        if not st["ok"]:
            print(f"ABORT: the third OOF draw ({D3_RUN}) is missing or incomplete — fetch run {D3_RUN} first, "
                  "then restart. Nothing written. The code paths can be checked without d3: python -I eda/c108_m3_three_draws.py --dry", file=sys.stderr)
            return 3
    mode = "dry" if a.dry else "real"
    say(f"c108 — third M3 draw ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}){' — CHECK: d3 := d2, NO VERDICT' if a.dry else ''}")
    try:
        base = load_base()
        D = load_draws(ctx, a.d3, a.dry)
        tech = technical_checks(ctx, D, a.dry)
        bad_lines = check_line_stacks(LINES, COMP / "artifacts")
        if bad_lines:
            tech["problems"] += [f"recipe differs from the blend file: {b}" for b in bad_lines]
        say(f"1. TECHNICAL CHECKS: base OOFs with checksums as deployed; mean(d1, d2) bit-identical to oof_{REF_CHANNEL}.npy; {D['rows_three']} rows with all three "
            f"draws; a9.build over three draws {'bit-identical' if D['built_a9'] else '(CHECK: not computed)'}")
        for j, info in enumerate(D["info"], 1):
            g = info.get("generation") or {}
            say(f"   d{j}: {info['source']} — {info['rows']} rows; max_tokens {info.get('max_tokens')}; generation {json.dumps(g.get('settings'), ensure_ascii=False)}")
        for j, recs in tech["fold_checks"].items():
            ok = all(not r["problems"] for r in recs)
            say(f"   fold integrity/training regime {j}: " + "; ".join(f"{r['run']}/{r['job']} fold {r['row_folds']} <- {r.get('train', '?')} (holds out fold {r.get('train_fold')}, "
                                                          f"{'gold-only' if r.get('gold_only') else 'NOT gold-only' if r.get('gold_only') is False else '?'}, adapter = merged one: "
                                                          f"{r.get('adapter_matches_merge')})" for r in recs) + f" -> {'ok' if ok else 'FINDING'}")
        for w in tech["warnings"]:
            say(f"   NOTE: {w}")
        if tech["artifact"] is not None:
            say(f"   deployed artifact artifacts/oof_{ARTIFACT}.npy exists: bit-identical {tech['artifact']['bit_identical']}, sources match {tech['artifact']['source_ok']}")
        if tech["problems"]:
            raise Technik("; ".join(tech["problems"]))
        res, rows = evaluate(ctx, base, D, a.dry, n_sim=a.n_sim, chunks=a.chunks, say=say)
    except Technik as e:
        print(f"ABORT (technical, exit 4): {e} — nothing written", file=sys.stderr)
        return 4
    out_dir = Path(a.out_dir) if a.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = NAME + ("_dry" if a.dry else "")
    rec = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode, "experiment": "third M3 draw", "decision_basis": "deployment rule fixed in advance",
           "specs": {"d1": D["specs"][0], "d2": D["specs"][1], "d3": f"CHECK: d3 := d2 (instead of {D['specs'][2]})" if a.dry else D["specs"][2],
                     "m3x3_source": None if a.dry else ",".join(D["specs"])},
           "m3x2_npy_sha256": npy_sha256(D["m3x2"]), "m3x3_npy_sha256": npy_sha256(D["m3x3"]), "artifact": ARTIFACT,
           "generation": [i.get("generation") for i in D["info"]], "rank_recipe": D["info"][0].get("rank_recipe"),
           "technical": {k: v for k, v in tech.items() if k != "gen_d3"}, "rules": {"guard": [GUARD_DROP, GUARD_Z], "single_tol": SINGLE_TOL, "p_harm": P_HARM,
                                                                                 "harm": HARM, "tau_main": TAU_MAIN, "prior": "leave-one-out family prior, class N (c95_common.family_prior)"},
           "expectation": EXPECT, **res, "runtime_s": time.time() - t0}
    (out_dir / f"{stem}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float) + "\n")
    np.savez_compressed(out_dir / f"{stem}_rows.npz", **rows)
    say(f"-> {out_dir / stem}.{{json,txt}} and {stem}_rows.npz ({time.time() - t0:.0f} s){' — CHECK, no verdict' if a.dry else ''}")
    (out_dir / f"{stem}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
