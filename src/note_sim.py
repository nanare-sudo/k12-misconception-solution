"""Example similarity of the M3 notes for the second stage of the safety line (feature set fallback_i1_nd).

Pure computational building blocks without file access. Per draw d1–d3 one TF-IDF vectoriser (word 1- and 2-grams, sublinear, min_df 2, at
most 200,000 features), fitted ONLY on the notes of the training rows of that draw (last 4,000 characters); cosine similarity of the query rows
to the reference rows, averaged over the three draws. From it, per (row, class): demo_sim = mean of the three largest similarities to reference
rows with gold = class; reference rows of the same question never count (pure analogy from other questions), the row itself never; 0 without a
reference row of the class. Guard: mean top similarity (per row the largest similarity to a reference row of another question) of the test rows
against that of the training rows (reference: all other training rows of other questions) — passes within the band ±10 % relative OR at
|z| ≤ 3 (two-sample z of the means; like the widened note guard it only widens).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

TAIL_CHARS = 4000
TOP_M = 3
SIM_REL = 0.10
SIM_Z_MAX = 3.0


def _vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)


def _tail(texts: Sequence[str]) -> list[str]:
    return [(t or "")[-TAIL_CHARS:] for t in texts]


def similarity(query_per_draw: Sequence[Sequence[str]], ref_per_draw: Sequence[Sequence[str]]) -> np.ndarray:
    """Mean cosine similarity over the draws: [len(query), len(ref)]. The vectoriser of each draw only knows the reference texts
    (training notes); if query and reference are the same list, this is the similarity of the training rows among each other."""
    if len(query_per_draw) != len(ref_per_draw) or not len(ref_per_draw):
        raise ValueError("the same number of draws is required for query and reference")
    S = None
    for q, r in zip(query_per_draw, ref_per_draw):
        v = _vectorizer()
        Xr = v.fit_transform(_tail(r))
        Xq = Xr if q is r else v.transform(_tail(q))
        part = (Xq @ Xr.T).toarray()
        S = part if S is None else S + part
    return S / float(len(ref_per_draw))


def mask_same_task(S: np.ndarray, query_keys: np.ndarray, ref_keys: np.ndarray) -> np.ndarray:
    """Similarity to reference rows of the same question set to −∞ (copy). With identical query and reference sets the row itself
    drops out as well (same key)."""
    S = np.array(S, dtype=np.float64, copy=True)
    S[np.asarray(query_keys)[:, None] == np.asarray(ref_keys)[None, :]] = -np.inf
    return S


def demo_sim(S_rows: np.ndarray, pool_y: np.ndarray, K: int, top_m: int = TOP_M) -> np.ndarray:
    """Per (query row, class): mean of the top_m largest finite similarities to reference rows with gold = class (S_rows: similarity of
    the query rows to EXACTLY the reference rows pool, −∞ = excluded); 0 if there is no such reference row. -> [R, K]."""
    S_rows = np.asarray(S_rows, dtype=np.float64)
    pool_y = np.asarray(pool_y, dtype=np.int64)
    if S_rows.shape[1] != len(pool_y):
        raise ValueError(f"similarity {S_rows.shape} does not match {len(pool_y)} reference rows")
    out = np.zeros((S_rows.shape[0], K))
    for c in np.unique(pool_y):
        cols = np.flatnonzero(pool_y == c)
        m = min(top_m, cols.size)
        top = -np.sort(-S_rows[:, cols], axis=1)[:, :m]
        fin = np.isfinite(top)
        out[:, c] = np.where(fin, top, 0.0).sum(1) / np.maximum(fin.sum(1), 1)
    return out


def top_sim(S_rows: np.ndarray) -> np.ndarray:
    """Per query row the largest finite similarity to a reference row (excluded ones left out); 0 if there is none. -> [R]."""
    S_rows = np.asarray(S_rows, dtype=np.float64)
    m = np.where(np.isfinite(S_rows), S_rows, -np.inf).max(1)
    return np.where(np.isfinite(m), m, 0.0)


def sim_stats(ts: np.ndarray) -> dict:
    """Statistic of the similarity guard (aggregate only): mean, standard deviation, number of rows."""
    ts = np.asarray(ts, dtype=np.float64)
    return {"mean": float(ts.mean()), "sd": float(ts.std(ddof=1)), "n": int(len(ts))}


def sim_guard(ref: dict, test: dict, rel: float = SIM_REL, z_max: float = SIM_Z_MAX) -> dict:
    """Similarity guard: passes within the band ref.mean·(1 ± rel) OR at |z| ≤ z_max, z = (m_test − m_ref)/sqrt(s_ref²/n_ref +
    s_test²/n_test). -> {ok, band, z, ...}."""
    lo, hi = ref["mean"] * (1.0 - rel), ref["mean"] * (1.0 + rel)
    se = math.sqrt(ref["sd"] ** 2 / ref["n"] + test["sd"] ** 2 / test["n"])
    d = test["mean"] - ref["mean"]
    z = (0.0 if d == 0.0 else math.copysign(math.inf, d)) if se == 0.0 else d / se
    ok_band = bool(lo <= test["mean"] <= hi)
    ok_z = bool(abs(z) <= z_max)
    return {"ok": bool(ok_band or ok_z), "ok_band": ok_band, "ok_z": ok_z, "band": [lo, hi], "z": z, "z_max": z_max, "rel": rel,
            "test": test, "reference": ref, "passed_by": "band" if ok_band else ("z" if ok_z else None)}
