"""Mean channel pr27xm: named codes (pr27) averaged with the exact ranking behind the same notes — ONE function for the
training rows (OOF artifact) and the test rows (tools/make_submission.py --rank).

    python -I eda/c95_rank_mean.py                       # builds artifacts/oof_pr27xm.npy + meta (deployed sources)
    python -I eda/c95_rank_mean.py pr27xm_d2 --proto runs/pr27s2_rp/proto_a2/proto+runs/pr27s3_rp/proto_b2/proto \\
                                             --rank runs/n1_rp/rank_d2/bench          # the same form behind draw d2

DEFINITION (the variant "mean of both as ONE channel", old-channel floor 0.05):
  * named    = channel from the answers (vlm/proto.labels_to_probs via eda/a8_proto_oof.mean_of_draws): the named codes
               with their numbers, the rest spread evenly; rows WITHOUT a valid answer: uniform distribution over the
               classes with a training example (training 8 of 1,500 rows, test 26 of 3,498).
  * rank     = exact distribution over all codes behind the notes (probe proto_rank, eda/a8_proto_oof.load_rank).
  * MEAN     = 0.5 · (named + rank) — arithmetic, BEFORE the floor. The floor 0.05 (that of the old channel pr27, NOT
               0.01 as for pr27x) is only applied in the blend (`--floor pr27xm=0.05`; make_submission applies the floor
               of the blend file). Like pr27, the channel carries mass on codes without a training example unmasked (half
               of it); the recipe evaluation masks before the floor, the blend and test paths do not.
  * Rows WITHOUT a ranking (only conceivable on the test side: no note, failed row): there the channel pr27x of the a8
               logic applies ("the other rows keep the channel from the answers"), so the mean is 0.5·(named + named) =
               named, i.e. exactly the channel pr27. On the training rows this does not occur (a ranking is required on
               all 1,500 rows, as with a8 --rank).
  * Several draws of the named codes (`,` in --proto) are averaged first, as for pr27; one ranking applies to the
               averaged channel — the deployed form uses exactly ONE draw.

Test side: tools/make_submission.py calls `channel(..., table="test")` of this file (the same function, the same row
rules); floor, factor and weights come from the blend file. Technical limits there: ranking on at least 98 % of the test
rows, median candidate mass ≥ 0.9.

Written via kaggle_core.artifacts.save_oof: artifacts/oof_<name>.npy, artifacts/meta_<name>.json (field rank_mean with
definition, sources and floor; make_submission then requires --rank and the floor 0.05 in the blend).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402

_spec = importlib.util.spec_from_file_location("c95_a8", COMP / "eda" / "a8_proto_oof.py")
A8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A8)

FLOOR = 0.05  # floor of the mean channel in the blend (the floor of the old channel pr27)
DEF_NAME = "pr27xm"
DEF_PROTO = "runs/pr27_rp/proto_a/proto+runs/pr27_rp/proto_b/proto"
DEF_RANK = "runs/px27_rp/rank_tr/bench"
# definition of the channel, recorded in meta_<name>.json (field rank_mean)
DEFINITION = ("0.5 · (named codes + exact ranking behind the notes); rows without a ranking: named codes; "
              "before the floor; floor in the blend 0.05 (variant \"mean\" of the channel-swap evaluation)")


def rank_mean(named: np.ndarray, rank: np.ndarray, has_rank: np.ndarray) -> np.ndarray:
    """The mean channel: 0.5 · (named + rank), on rows without a ranking 0.5 · (named + named) = named.

    Same floating-point computation as (old + new) / 2 and 0.5 · (old + new); both are bit-identical."""
    named = np.asarray(named, dtype=np.float64)
    rank = np.asarray(rank, dtype=np.float64)
    has_rank = np.asarray(has_rank, dtype=bool)
    if named.shape != rank.shape or named.ndim != 2 or has_rank.shape != (len(named),):
        raise ValueError(f"shapes do not match: named {named.shape}, ranking {rank.shape}, mask {has_rank.shape}")
    alt = np.where(has_rank[:, None], rank, named)  # channel pr27x of the a8 logic
    return 0.5 * (named + alt)


def exact_rank_rows(src: str, n: int, seen_cls: np.ndarray) -> np.ndarray:
    """Rows (table positions) that the probe computed completely AND exactly (all classes with a training example finite)
    — the same files and the same rule as a8.load_rank."""
    _, run, job, tag = src.split("/")
    d = A8.COMP / "artifacts" / "runs" / run / run / job
    idx, lp = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy")
    done_path = d / f"{tag}_proto_rank.done.npy"
    done = np.load(done_path) if done_path.is_file() else np.ones(len(idx), bool)
    full = done & np.isfinite(lp).any(axis=1) & np.isfinite(lp[:, seen_cls]).all(axis=1)
    out = np.zeros(n, bool)
    out[idx[full]] = True
    return out


def channel(proto_sources: list[str], rank_source: str, n: int, classes: list[str], seen_cls: np.ndarray, table: str,
            neutral=None, rank_partial: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Mean channel for one table. -> (P [n, K], number of valid answers per row, ranking present [n], statistics).

    table="train": ranking required on ALL rows, complete and exact (as with a8 --rank).
    table="test":  every row of the probe complete and exact; rows outside the probe keep the named codes.
    rank_partial:  deliberately accept an incomplete run — only rows with an exact ranking count, the others keep the named
                   codes (never for a decision on the training rows)."""
    missing = rank_files_missing(rank_source)
    if missing:
        raise SystemExit(f"ranking {rank_source}: missing {', '.join(missing)} — run not finished or not fetched")
    named, cnt = A8.mean_of_draws(list(proto_sources), n, classes, seen_cls, table=table, neutral=neutral)
    R, hr, stats = A8.load_rank(rank_source, n, seen_cls, table=table, allow_partial=rank_partial,
                                require_full_table=(table == "train" and not rank_partial))
    if rank_partial:  # in a partial state a8 also counts truncated rows (remaining mass); here only the exact form counts
        hr &= exact_rank_rows(rank_source, n, seen_cls)
    if neutral is not None:
        hr[np.asarray(neutral, dtype=int)] = False
    P = rank_mean(named, R, hr)
    due = cnt >= 0
    valid = cnt > 0
    stats = {**stats, "rows_table": int(n), "rows_due": int(due.sum()), "rows_rank": int((hr & due).sum()),
             "rows_named_only": int((~hr & (cnt > 0)).sum()), "rows_uniform_only": int((~hr & (cnt == 0)).sum()),
             "rows_without_answer": int((cnt == 0).sum()), "rows_rank_without_answer": int((hr & (cnt == 0)).sum()),
             "share_rank": float((hr & due).sum() / max(1, int(due.sum()))), "partial": bool(rank_partial),
             # aggregates (shares over rows): malfunction band and comparison with the training rows
             "none_top_rank": top_none_share(R, hr), "none_top_named": top_none_share(named, valid),
             "none_top_mean": top_none_share(P, due), "same_top_named_rank": float(np.mean(named[hr & valid].argmax(1) == R[hr & valid].argmax(1)))
             if (hr & valid).any() else float("nan")}
    return P, cnt, hr, stats


def recipe_differences(src: str, ref_src: str) -> dict:
    """Settings of two ranking runs (meta files of the probe): differences except the notes (`proto_from`) and the fold.
    Empty = the same computation (model, number format, image, beams, system text) on other rows."""
    def meta_of(s: str) -> dict:
        _, run, job, tag = s.split("/")
        return json.loads((A8.COMP / "artifacts" / "runs" / run / run / job / f"{tag}_meta.json").read_text())
    a, b = meta_of(src), meta_of(ref_src)
    ra = {k: v for k, v in (a.get("recipe") or {}).items() if k not in ("proto_from", "fold")}
    rb = {k: v for k, v in (b.get("recipe") or {}).items() if k not in ("proto_from", "fold")}
    diff = {k: (ra.get(k), rb.get(k)) for k in sorted(set(ra) | set(rb)) if ra.get(k) != rb.get(k)}
    for k in ("probes", "dtype", "quant", "adapter"):
        if a.get(k) != b.get(k):
            diff[k] = (a.get(k), b.get(k))
    return diff


def rank_files_missing(src: str) -> list[str]:
    """Files that a ranking `runs/<run>/<job>/<tag>` needs and that are (still) missing."""
    bits = src.split("/")
    if len(bits) != 4 or bits[0] != "runs":
        raise SystemExit(f"{src}: expected runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    d = A8.COMP / "artifacts" / "runs" / run / run / job
    return [str(Path("artifacts") / "runs" / run / run / job / f) for f in (f"{tag}_meta.json", f"{tag}_idx.npy", f"{tag}_proto_rank.npy",
                                                                           f"{tag}_proto_rank.done.npy") if not (d / f).is_file()]


def top_none_share(P: np.ndarray, rows: np.ndarray) -> float:
    return float(np.mean(P[rows].argmax(1) == 0)) if rows.any() else float("nan")


def build_oof(name: str, proto: str, rank: str) -> tuple[np.ndarray, dict]:
    """Training side: build the channel, check it, write it via kaggle_core.artifacts.save_oof."""
    from kaggle_core import artifacts as KA
    from kaggle_core.folds import folds_hash
    from vlm import common as C

    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    seen_cls = labels.seen_mask(train["root_code"], classes)
    n = len(y)
    P, cnt, hr, st = channel(proto.split(","), rank, n, classes, seen_cls, "train")
    if not hr.all():
        raise SystemExit(f"ranking only on {int(hr.sum())} of {n} training rows — not a channel for a decision")
    if not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-9):
        raise SystemExit("mean channel: non-finite values or rows that do not sum to 1")
    named, _ = A8.mean_of_draws(proto.split(","), n, classes, seen_cls)
    R, _, _ = A8.load_rank(rank, n, seen_cls, require_full_table=True)
    rows = metric.map3_rows_from_proba(y, P)
    cv, sigma = float(rows.mean()), float(metric.bootstrap_sigma(rows))
    r_named = float(metric.map3_rows_from_proba(y, named).mean())
    r_rank = float(metric.map3_rows_from_proba(y, R).mean())
    meta = {"folds_hash": folds_hash(folds), "model_type": "vlm-protocol", "sources": proto.split(","), "cv_score": cv, "cv_sigma": sigma,
            "feature_set": "image+task, without task knowledge", "rows_without_answer": int((cnt == 0).sum()), "neutral_rows": None,
            "rows_neutral": 0,
            "rank_mean": {"definition": DEFINITION, "proto_sources": proto.split(","), "rank_source": rank, "floor": FLOOR,
                          "origin": "variant \"mean\" of the channel-swap evaluation (eda/c58_swap_channel.py); deployed form of the mean channel",
                          "function": "eda/c95_rank_mean.py: channel() / rank_mean()",
                          "cv_named": r_named, "cv_rank": r_rank, "none_true": float(np.mean(y == 0)), **st}}
    KA.save_oof(name, P, np.zeros((0, len(classes))), meta, COMP / "artifacts")
    print(f"{name}: mean channel of {proto} and {rank}")
    print(f"  ranking: {st['rows_done']} of {st['rows_probe']} probe rows computed, {st['rows_exact']} exact; in the channel {st['rows_rank']} of {n} rows; "
          f"rows without a valid answer {st['rows_without_answer']} (of which with a ranking {st['rows_rank_without_answer']})")
    print(f"  candidate mass behind \"Root:\": median {st['mass_median']:.3f}, 5 % quantile {st['mass_p05']:.3f}, below 0.5 in {st['mass_below_half']:.3f} of the rows")
    print(f"  channel alone: named {r_named:.4f}, ranking {r_rank:.4f}, MEAN {cv:.4f} ± {sigma:.4f}; per fold {[round(float(rows[folds == k].mean()), 4) for k in range(5)]}")
    print(f"  NONE at the top: named {meta['rank_mean']['none_top_named']:.3f}, ranking {meta['rank_mean']['none_top_rank']:.3f}, "
          f"mean {meta['rank_mean']['none_top_mean']:.3f} (true {meta['rank_mean']['none_true']:.3f}); same top named/ranking "
          f"{meta['rank_mean']['same_top_named_rank']:.3f}")
    print(f"written: artifacts/oof_{name}.npy (SHA-256 {hashlib.sha256((COMP / 'artifacts' / f'oof_{name}.npy').read_bytes()).hexdigest()[:16]}…), artifacts/meta_{name}.json")
    return P, meta


def status(src: str, table: str, max_missing: float = 0.02) -> int:
    """State of a ranking run (aggregates only): 0 = complete (every row of the probe computed and exact, coverage of the
    table ≥ 1 − max_missing), 3 = missing or incomplete."""
    from vlm import common as C

    root = C.find_data_root()
    n = len(C.load_table(root, table))
    train = C.load_table(root, "train")
    seen_cls = labels.seen_mask(train["root_code"], labels.load_classes(root / "competition_data" / "taxonomy_v1.csv"))
    missing = rank_files_missing(src)
    if missing:
        print(f"{src}: MISSING ({', '.join(missing)})")
        return 3
    R, _, st = A8.load_rank(src, n, seen_cls, table=table, allow_partial=True)
    exact = exact_rank_rows(src, n, seen_cls)
    share = float(exact.sum()) / n
    complete = st["rows_done"] == st["rows_probe"] and st["rows_exact"] == st["rows_done"] and share >= 1.0 - max_missing
    print(f"{src} (table {table}, {n} rows): probe {st['rows_probe']} rows, computed {st['rows_done']}, of which exact {st['rows_exact']}; "
          f"exact ranking on {int(exact.sum())} rows ({share:.4f}); mass median {st['mass_median']:.3f}, 5 % quantile {st['mass_p05']:.3f}; "
          f"NONE at the top {top_none_share(R, exact):.3f} -> {'COMPLETE' if complete else 'INCOMPLETE'}")
    return 0 if complete else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", default=DEF_NAME)
    ap.add_argument("--status", metavar="RANK", help="only report the state of a ranking run (runs/<run>/<job>/<tag>; aggregates; exit 0 = complete, 3 = not)")
    ap.add_argument("--table", default="test", choices=["train", "test"], help="table for --status")
    ap.add_argument("--proto", default=DEF_PROTO, help="named codes: runs/<run>/<job>/<tag>[+…] (draws separated by commas)")
    ap.add_argument("--rank", default=DEF_RANK, help="ranking: runs/<run>/<job>/<tag> (probe proto_rank, training rows)")
    a = ap.parse_args()
    if a.status:
        return status(a.status, a.table)
    build_oof(a.name, a.proto, a.rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
