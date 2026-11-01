"""Save the task-agnostic channel (answers in the teacher protocol, job proto_gen) as an OOF artifact.

    python -I eda/a8_proto_oof.py <name> runs/<run>/<job>/<tag>[+runs/<run>/<job>/<tag> ...] [--mean <more draws> ...]

The channel does not depend on any fold (no labelled row in the prompt, untrained model): the answers for the training
rows are honest for every split and go unchanged into artifacts/oof_<name>.npy — the same conversion
(vlm/proto.labels_to_probs) as later on the test rows (tools/make_submission.py --proto). Rows without a valid answer get
the uniform distribution over the classes that can be a root; in the log-linear mean this has no effect. If more than
--max-missing of the rows are missing, the script aborts (an incomplete run is not a channel).

--mean <source>: another DRAW of the same channel (e.g. a second run with a different seed); per row the mean is taken
over the draws that have a valid answer (arithmetic, as for the reasoning channel).

--debias <strength>: INDUCTIVE DEBIASING — class weights that bring the channel's mean mass on the training rows with a
valid answer to their class frequency (classes with at least --debias-min rows), applied with this strength to the rows
with a valid answer. The weights are stored in the meta file (`class_weights`); tools/make_submission.py applies them
unchanged from there to the channel on the test rows. The artifact carries the weights from ALL training rows (for the
blend weights of the submission); an honest per-fold estimate requires refitting the weights inside each fold.

--rank runs/<run>/<job>/<tag>: EXACT RANKING behind the notes (probe proto_rank of a bench job) — on the rows it computed,
the distribution over all codes replaces the named codes with their stated numbers; the other rows keep the channel from
the answers. One draw, without --mean and without --debias.

Written via kaggle_core.artifacts.save_oof: artifacts/oof_<name>.npy, artifacts/meta_<name>.json.
Training rows only (the meta file of every source is checked), only aggregates in the output.
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
from kaggle_core import artifacts as A  # noqa: E402
from kaggle_core.folds import folds_hash  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import proto as PR  # noqa: E402


def load_draw(src: str, n: int, classes: list[str], seen_cls: np.ndarray, table: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """One draw from one or more answer files (joined with +; later ones complement earlier ones)."""
    P, h = np.zeros((n, len(classes))), np.zeros(n, bool)
    for part in src.split("+"):
        bits = part.split("/")
        if len(bits) != 4 or bits[0] != "runs":
            raise SystemExit(f"{part}: expected runs/<run>/<job>/<tag>")
        _, run, job, tag = bits
        d = COMP / "artifacts" / "runs" / run / run / job
        meta_file = d / f"{tag}_meta.json"
        if not meta_file.is_file():
            raise SystemExit(f"{part}: {tag}_meta.json missing — job not finished or not completely fetched")
        meta = json.loads(meta_file.read_text())
        if meta.get("table") != table:
            raise SystemExit(f"{part}: no output for rows of table {table} (table={meta.get('table')!r})")
        Pp, hp = PR.labels_to_probs(d / f"{tag}_labels.jsonl", n, classes, seen_cls)
        if "valid" in meta and int(meta["valid"]) != int(hp.sum()):  # the answer file and the statistics of the run must match
            raise SystemExit(f"{part}: {int(hp.sum())} rows with a valid answer, the meta file says {meta['valid']} — answer file incomplete?")
        if (h & hp).any():
            raise SystemExit(f"{part}: {int((h & hp).sum())} rows covered twice")
        P[hp], h[hp] = Pp[hp], True
    return P, h


def mean_of_draws(sources: list[str], n: int, classes: list[str], seen_cls: np.ndarray, table: str = "train", neutral=None) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic mean over the draws with a valid answer; rows without any answer: uniform distribution.
    neutral: table positions on which the channel counts as NOT computed (gate: e.g. rows that the fast channel confidently
    considers flawless) — uniform there, even if an answer exists; this way the channel is built identically for the training
    and the test rows.
    -> (P [n, K], number of draws per row; −1 on neutral rows)"""
    S, cnt = np.zeros((n, len(classes))), np.zeros(n, int)
    for src in sources:
        P, h = load_draw(src, n, classes, seen_cls, table)
        if neutral is not None:
            h[np.asarray(neutral, dtype=int)] = False
        S[h] += P[h]
        cnt[h] += 1
    if neutral is not None:
        cnt[np.asarray(neutral, dtype=int)] = -1
    out = np.tile(seen_cls / seen_cls.sum(), (n, 1)).astype(float)
    have = cnt > 0
    out[have] = S[have] / cnt[have, None]
    return out, cnt


def load_rank(src: str, n: int, seen_cls: np.ndarray, table: str = "train", allow_partial: bool = False,
              require_full_table: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ranking of the probe proto_rank: -> (P [n, K], computed [n], statistics). Source `runs/<run>/<job>/<tag>`.

    A COMPLETE, EXACT run is required: every row of the probe computed and a finite value for every class that can be a
    root. A partial state is not a random sample (the probe computes the longest prompts first), and after halving the
    prefixes on an out-of-memory error codes go missing silently — both abort.
    allow_partial: deliberate partial state; then codes outside the search share the remaining mass of the row
    (vlm.engine.probs_from_logp), and rows that were not computed count as absent.
    Statistics: rows of the probe, computed, exact (all classes finite) and the MASS of the candidates per row
    (Σ exp(log P) before normalisation) — if it is often far below 1, the forced answer start is unnatural for these notes."""
    from vlm.engine import probs_from_logp

    bits = src.split("/")
    if len(bits) != 4 or bits[0] != "runs":
        raise SystemExit(f"{src}: expected runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    d = COMP / "artifacts" / "runs" / run / run / job
    meta = json.loads((d / f"{tag}_meta.json").read_text())
    if meta.get("table") != table:
        raise SystemExit(f"{src}: no output for rows of table {table} (table={meta.get('table')!r})")
    idx, lp = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy")
    done_path = d / f"{tag}_proto_rank.done.npy"
    if require_full_table and not done_path.is_file():
        raise SystemExit(f"{src}: completeness mask missing — not a run that can support a decision")
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(idx) == 0:
        raise SystemExit(f"{src}: invalid row list")
    if lp.shape != (len(idx), len(seen_cls)) or len(set(idx.tolist())) != len(idx) or idx.min() < 0 or idx.max() >= n:
        raise SystemExit(f"{src}: row list and ranking do not match")
    if require_full_table and (len(idx) != n or not np.array_equal(np.sort(idx), np.arange(n))):
        raise SystemExit(f"{src}: ranking does not cover exactly all {n} training rows")
    done = np.load(done_path) if done_path.is_file() else np.ones(len(idx), bool)
    if done.shape != idx.shape:
        raise SystemExit(f"{src}: completeness mask does not match the row list")
    done = done & np.isfinite(lp).any(axis=1)  # a row without a finite candidate is not computed
    full = done & np.isfinite(lp[:, seen_cls]).all(axis=1)
    mass = np.where(np.isfinite(lp), np.exp(np.where(np.isfinite(lp), lp, 0.0)), 0.0).sum(1)[done]
    stats = {"rows_probe": int(len(idx)), "rows_expected": int(n), "rows_done": int(done.sum()), "rows_exact": int(full.sum()),
             "completion_manifest": bool(done_path.is_file()),
             "idx_sha256": hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest(),
             "mass_median": float(np.median(mass)) if done.any() else 0.0, "mass_p05": float(np.percentile(mass, 5)) if done.any() else 0.0,
             "mass_below_half": float(np.mean(mass < 0.5)) if done.any() else 0.0}
    if not allow_partial and (stats["rows_done"] < len(idx) or stats["rows_exact"] < stats["rows_done"]):
        raise SystemExit(f"{src}: {stats['rows_done']} of {len(idx)} rows computed, {stats['rows_exact']} of them with all codes — not a complete, exact run "
                         f"(to evaluate a partial state deliberately: --rank-partial)")
    P, h = np.zeros((n, len(seen_cls))), np.zeros(n, bool)
    P[idx[done]], h[idx[done]] = probs_from_logp(lp[done], absent=~seen_cls), True
    return P, h, stats


def debias_weights(P: np.ndarray, valid: np.ndarray, y: np.ndarray, rows: np.ndarray, K: int, min_count: int) -> np.ndarray:
    """Class weights from the rows `rows` (only those with a valid answer): mean mass of the channel -> class frequency of
    these rows (eda/c21_margin_align.fit_weights); classes with fewer than min_count rows there keep weight 1.
    Inductive: only training labels and the channel's answers for training rows."""
    spec = importlib.util.spec_from_file_location("c21", COMP / "eda" / "c21_margin_align.py")
    c21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c21)
    use = np.asarray(rows)[valid[rows]]
    c = np.bincount(y[use], minlength=K)
    return c21.fit_weights(P[use], np.where(c >= min_count, c, 0) / len(use))


def apply_debias(P: np.ndarray, valid: np.ndarray, w: np.ndarray, strength: float) -> np.ndarray:
    """Class weights with strength `strength` on the rows with a valid answer; the others stay unchanged (uniform).
    The same step for the training and the test rows of the channel — BEFORE the floor of the blend."""
    w = np.asarray(w, dtype=float)
    if w.shape != (P.shape[1],) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("class weights: one finite positive value per class expected")
    out = P.copy()
    Q = P[valid] * w[None, :] ** strength
    out[valid] = Q / Q.sum(1, keepdims=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("source")
    ap.add_argument("--mean", action="append", default=[])
    ap.add_argument("--max-missing", type=float, default=0.03)
    ap.add_argument("--neutral", help="row list (.npy, relative to data/): the channel counts as not computed there (gate) — as on the test rows")
    ap.add_argument("--debias", type=float, default=0.0, help="strength of the inductive debiasing of the class shares (0 = off; used value: 1.0)")
    ap.add_argument("--debias-min", type=int, default=3, help="debiasing only for classes with at least this many training rows")
    ap.add_argument("--rank", help="runs/<run>/<job>/<tag>: exact ranking behind the notes (probe proto_rank) replaces the named codes on its rows")
    ap.add_argument("--rank-partial", action="store_true", help="deliberately accept an incomplete or truncated ranking run (never for a decision)")
    # argparse interpolates percent signs in help strings using %-formatting.
    ap.add_argument("--rank-min-cover", type=float, default=1.0, help="minimum share of training rows with an exact ranking; standard for a decision: 100%%")
    a = ap.parse_args()
    if a.rank and (a.mean or a.debias):
        raise SystemExit("--rank applies to one draw without --mean and without --debias (the variants are measured separately)")
    if a.rank and not a.rank_partial and (a.neutral or not np.isclose(a.rank_min_cover, 1.0)):
        raise SystemExit("a decision with --rank requires all training rows: no --neutral and --rank-min-cover 1.0")
    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    seen_cls = labels.seen_mask(train["root_code"], classes)
    n = len(y)
    sources = [a.source] + a.mean
    neutral = np.load(root / a.neutral) if a.neutral else None
    P, cnt = mean_of_draws(sources, n, classes, seen_cls, neutral=neutral)
    miss = cnt == 0
    print(f"{a.name}: {len(sources)} draw(s); rows by number of valid draws {dict(sorted((int(k), int((cnt == k).sum())) for k in set(cnt.tolist())))}"
          + (f" (−1 = {int((cnt < 0).sum())} rows of the gate {a.neutral}, uniform there)" if a.neutral else ""))
    if miss.sum() > a.max_missing * max(1, int((cnt >= 0).sum())):
        raise SystemExit(f"{int(miss.sum())} of {int((cnt >= 0).sum())} rows to compute without a valid answer (above {a.max_missing:.0%}) — not a channel")
    extra = {}
    if a.rank:
        named = float(metric.map3_rows_from_proba(y, P).mean())
        R, hr, rs = load_rank(a.rank, n, seen_cls, allow_partial=a.rank_partial,
                              require_full_table=not a.rank_partial)
        if neutral is not None:
            hr[np.asarray(neutral, dtype=int)] = False
        cover = float(hr.sum()) / max(1, int((cnt >= 0).sum()))
        if cover < a.rank_min_cover and not a.rank_partial:  # fallback rows would sit below the floor of the exact channel in the named form
            raise SystemExit(f"ranking only for {int(hr.sum())} of {int((cnt >= 0).sum())} rows ({cover:.1%}, required {a.rank_min_cover:.0%}) — mixed channel of two forms, no evaluation")
        if not a.rank_partial and not hr.all():
            raise SystemExit("a decision with --rank requires an exact ranking for every training row")
        P = np.where(hr[:, None], R, P)
        miss = miss & ~hr
        extra = {"rank": {"source": a.rank, "rows": int(hr.sum()), "rows_named_only": int((~hr & (cnt > 0)).sum()), "cv_score_named": named, **rs,
                          "partial": bool(a.rank_partial),
                          "note": "on the computed rows the exact distribution over all codes, otherwise the named codes (labels_to_probs)"}}
        print(f"  ranking behind the notes: {rs['rows_done']} of {rs['rows_probe']} probe rows computed, {rs['rows_exact']} with all codes; in the channel {int(hr.sum())} rows exact, "
              f"{int((~hr & (cnt > 0)).sum())} with named codes only; channel from the named codes alone {named:.4f}")
        print(f"  candidate mass behind \"Root:\" per row: median {rs['mass_median']:.3f}, 5 % quantile {rs['mass_p05']:.3f}, below 0.5 in {rs['mass_below_half']:.3f} of the rows "
              f"(reasoning channel: median 0.98) — far below that, the forced answer start would be unnatural for these notes")
    if a.debias:
        valid = cnt > 0
        raw = float(metric.map3_rows_from_proba(y, P).mean())
        w = debias_weights(P, valid, y, np.arange(n), len(classes), a.debias_min)
        P = apply_debias(P, valid, w, a.debias)
        extra = {"class_weights": [float(v) for v in w], "debias": {"strength": a.debias, "min_rows": a.debias_min, "classes": int((w != 1).sum()), "cv_score_raw": raw,
                                                                     "note": "weights from all training rows with a valid answer; honest value per fold: eda/c50_channel_debias.py"}}
        print(f"  debiasing (inductive, strength {a.debias:g}, classes with at least {a.debias_min} rows): {int((w != 1).sum())} classes adjusted, weights {w[w != 1].min():.3f} to {w.max():.2f}; "
              f"channel alone raw {raw:.4f}")
    rows = metric.map3_rows_from_proba(y, P)
    cv, sigma = float(rows.mean()), float(metric.bootstrap_sigma(rows))
    # text values (model_type, feature_set, notes) are recorded in meta_<name>.json
    A.save_oof(a.name, P, np.zeros((0, len(classes))), {"folds_hash": folds_hash(folds), "model_type": "vlm-protocol", "sources": sources,
                                                         "cv_score": cv, "cv_sigma": sigma, "feature_set": "image+task, without task knowledge",
                                                         "rows_without_answer": int(miss.sum()), "neutral_rows": a.neutral,
                                                         "rows_neutral": int((cnt < 0).sum()), **extra}, COMP / "artifacts")
    print(f"  MAP@3 {cv:.4f} ± {sigma:.4f} (rows without an answer count with the uniform distribution); per fold {[round(float(rows[folds == k].mean()), 4) for k in range(5)]}")
    print(f"  NONE at the top {np.mean(P.argmax(1) == 0):.3f} (true {np.mean(y == 0):.3f}); distinct codes at the top {len(set(P.argmax(1).tolist()))}")
    print(f"written: artifacts/oof_{a.name}.npy, artifacts/meta_{a.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
