"""Evaluation of the reasoning 27B trained WITH pool rows (runs m3p_f1_rp, m3p_f2_rp) against the pilot WITHOUT pool rows
(m3_pilot_rp) on the same 600 rows of folds 1+2 — channel level, by cell, paired. Training rows only, aggregates only.
This experiment was not adopted (the pool version was worse); the module is kept because eda/a9_m3_draws_oof.py
--check-c98d reuses its ranking loader.

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I eda/c98b_pool_eval.py [--folds 1 2]

 1. Combine (vlm/sft.combine_rank_runs): m3p_f<k>_rp/m3p_rank_f<k> -> runs/m3p_pool/m3p_pool/m3p_rank and the namer
    control m3p_f<k>_rp/m3p_rankx_f<k> -> …/m3p_rankx (input of eda/a8_proto_oof.py --rank).
    The pilot: m3_pilot_rp/m3_rank (python -m vlm.sft combine).
 2. Channel M3p against M3 (paired row bootstrap, metric.paired_bootstrap; all 74 codes exact) — all rows, per fold,
    NONE, seen, never seen WITH the (task, code) pair among the fold's pool targets, never seen WITHOUT the pair
    (transfer of general ability), new task. Displacement control: seen and NONE.
 3. Namer control: M3p scores the PILOT'S TEXTS (fixed notes and findings) against the pilot itself — the difference
    lies only in the naming, without the spread of newly generated texts.
 4. Proposed stopping rule: STOP if channel < M3 − 0.02, NONE < M3 − 0.03 or seen < M3 − 0.04 — reported as the field
    `harm`. The contribution in the full recipe of the primary line is NOT computed here; the command for the artifact
    is printed.
Writes artifacts/eda/private/pool_eval/c98b_pool_eval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import sft as S  # noqa: E402
from vlm.engine import probs_from_logp  # noqa: E402

RUNS = COMP / "artifacts" / "runs"
OUT = COMP / "artifacts" / "eda" / "private" / "pool_eval"


def load_rank(run: str, job: str, tag: str = "bench") -> dict[int, np.ndarray]:
    d = RUNS / run / run / job
    idx, lp, done = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy"), np.load(d / f"{tag}_proto_rank.done.npy")
    return {int(i): lp[j] for j, i in enumerate(idx) if done[j]}


def cells(rows: np.ndarray, folds: np.ndarray, tq: np.ndarray, gold: np.ndarray, pool_pairs: dict[int, set]) -> dict[str, np.ndarray]:
    """Cells per row (only from the training part and the pool targets OF THE ROW'S FOLD): NONE, seen, never seen with /
    without a pool pair, new task."""
    out = {name: np.zeros(len(rows), bool) for name in ("NONE", "seen", "never seen, pair in the pool", "never seen, pair not in the pool", "new question")}
    for j, i in enumerate(rows):
        k = int(folds[i])
        if gold[i] == "NONE":
            out["NONE"][j] = True
            continue
        other = folds != k
        if tq[i] not in set(tq[other]):
            out["new question"][j] = True
        elif (tq[i], gold[i]) in set(zip(tq[other], gold[other])):
            out["seen"][j] = True
        elif (tq[i], gold[i]) in pool_pairs.get(k, set()):
            out["never seen, pair in the pool"][j] = True
        else:
            out["never seen, pair not in the pool"][j] = True
    return out


def pool_pairs_of(fold: int, pool_q: np.ndarray, data_root: Path) -> set:
    f = data_root / "derived" / "sft" / f"m3p_pool_f{fold}.jsonl"
    pairs = set()
    for line in f.read_text().split("\n"):
        if line.strip():
            r = json.loads(line)
            if r["code"] != "NONE":
                pairs.add((str(pool_q[int(r["row"])]), str(r["code"])))
    return pairs


def compare(a: dict[int, np.ndarray], b: dict[int, np.ndarray], y: np.ndarray, absent: np.ndarray, groups: dict) -> dict:
    rows = np.array(sorted(set(a) & set(b)), dtype=np.int64)
    if len(rows) < 2:
        raise SystemExit("fewer than two common rows with an exact ranking")
    pa = probs_from_logp(np.stack([a[int(i)] for i in rows]), absent=absent)
    pb = probs_from_logp(np.stack([b[int(i)] for i in rows]), absent=absent)
    ra, rb = metric.map3_rows_from_proba(y[rows], pa), metric.map3_rows_from_proba(y[rows], pb)
    out = {"rows": int(len(rows)), "same_top": float(np.mean(pa.argmax(1) == pb.argmax(1))), "groups": {}}
    for name, mask in {"all": np.ones(len(rows), bool), **{k: v(rows) for k, v in groups.items()}}.items():
        if mask.sum() >= 2:
            pb_ = metric.paired_bootstrap(ra[mask], rb[mask])
            out["groups"][name] = {"rows": int(mask.sum()), "m3p": float(ra[mask].mean()), "m3": float(rb[mask].mean()),
                                   "delta": pb_["delta"], "sigma": pb_["sigma"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--pilot", default="m3_pilot_rp/m3_rank", help="<run>/<job> of the combined pilot")
    a = ap.parse_args()
    root = C.find_data_root()
    train, pool = C.load_table(root, "train"), C.load_table(root, "unlabelled")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    absent = ~labels.seen_mask(train["root_code"], classes)
    tq, gold = train["question_id"].astype(str).to_numpy(), train["root_code"].astype(str).to_numpy()
    info = {"m3p_rank": S.combine_rank_runs(RUNS, [f"m3p_f{k}_rp/m3p_rank_f{k}" for k in a.folds], "bench", "m3p_pool", "m3p_rank"),
            "m3p_rankx": S.combine_rank_runs(RUNS, [f"m3p_f{k}_rp/m3p_rankx_f{k}" for k in a.folds], "bench", "m3p_pool", "m3p_rankx")}
    pool_q = pool["question_id"].astype(str).to_numpy()
    pp = {k: pool_pairs_of(k, pool_q, root) for k in a.folds}
    m3p, m3px = load_rank("m3p_pool", "m3p_rank"), load_rank("m3p_pool", "m3p_rankx")
    prun, pjob = a.pilot.split("/")
    m3 = load_rank(prun, pjob)

    def group_fns():
        def by(name):
            return lambda rows: cells(rows, folds, tq, gold, pp)[name]
        g = {name: by(name) for name in ("NONE", "seen", "never seen, pair in the pool", "never seen, pair not in the pool", "new question")}
        g.update({f"Fold {k}": (lambda kk: (lambda rows: folds[rows] == kk))(k) for k in a.folds})
        return g

    res = {"combine": info, "channel": compare(m3p, m3, y, absent, group_fns()), "namer_control": compare(m3px, m3, y, absent, group_fns())}
    g = res["channel"]["groups"]
    res["harm"] = {"channel_below_m3_minus_0.02": bool(g["all"]["delta"] < -0.02),
                   "none_below_m3_minus_0.03": bool("NONE" in g and g["NONE"]["delta"] < -0.03),
                   "seen_below_m3_minus_0.04": bool("seen" in g and g["seen"]["delta"] < -0.04),
                   "note": "added contribution in the primary-line recipe evaluated separately (artifact below, recipe computation as in eda/c95_m3_eval.py)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c98b_pool_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for part in ("channel", "namer_control"):
        print(f"{part}: {res[part]['rows']} rows, same top {res[part]['same_top']:.3f}")
        for name, v in res[part]["groups"].items():
            print(f"  {name:32s} n={v['rows']:4d}  M3p {v['m3p']:.4f}  M3 {v['m3']:.4f}  Δ {v['delta']:+.4f} ± {v['sigma']:.4f}")
    print(f"harm: {res['harm']}")
    print("artifact for the recipe (partial coverage as in the pilot):\n  python -I eda/a8_proto_oof.py m3pp runs/pr27_rp/proto_a/proto+runs/pr27_rp/proto_b/proto "
          "--rank runs/m3p_pool/m3p_rank/bench --rank-partial --neutral derived/sft/m3_neutral_f12.npy")
    print(f"written: {(OUT / 'c98b_pool_eval.json').relative_to(COMP)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
