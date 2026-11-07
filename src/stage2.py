"""Second stage of the primary line: a conditional logit over the first K candidates AFTER margin alignment.

Pure computational building blocks (numpy/scipy) without file access and without knowledge of runs: features per (row,
candidate), fitting, application, abstention gate, placebo, counting code mentions in notes, and the statistics of the test
guard. The command line with context, nested OOF reproduction and deployment file is eda/c118_stage2.py; the test side is
computed by tools/make_submission.py --stage2.

Form (fixed before the evaluation; it comes from the reranker family of the model sweep and its cross-check):
  * Candidates: the first K = 5 classes of the base form (only classes with a training example, stable sort as in
    metric.top3_indices). Base form = blend after margin alignment over the whole set; PM = the blend before that step.
  * Model: softmax over the K candidates of a row, one shared weight per feature (no class identity), features
    standardised (mean/std over all pairs of the fit), L2 with λ = 0.1 FIXED, L-BFGS-B from zero. Rows without the gold label
    among the candidates only count in the standardisation.
  * Feature set "main" (43): base form (log p, rank, gap, p), blend before the last step (log p, rank), log p and rank of
    every blend channel EXCEPT the 397B channels (q397, q397m3), number of channels with the code at the top / among the first
    three (over ALL channels of the blend, as measured), the rank in each M3 draw d1–d3, number of draws with the code at the
    top, spread of log p over the draws, task/topic knowledge from reference rows, flags "generic"/"NONE", mentions in the
    M3 notes d1–d3 (count 0–3, count of last mentions, length), row features (NONE mass, entropy, top probability,
    disagreement of the draws and of the channels, grade). No 397B note, no "q397 valid".
  * Feature set "fallback" (40): "main" without the three note features (used when the guard on the test notes fails).
  * Feature set "main_guard" (48; named before its evaluation): "main" in unchanged order + identifiers of the five general
    codes (id_<code> = 1 if the candidate is that code), appended at the end. It belongs to the NONE guard (protect_none): if the
    base form has NONE at the top, the base order of the row stays. Not in FEATURE_SETS — the deployment files stage2_s032/s034 and
    their build scripts iterate over FEATURE_SETS and stay bit-identical; a model of a deployment file names main_guard
    explicitly (field feature_set).
  * Feature set "main_guard_i1" (63; named before its evaluation): "main_guard" (48, unchanged, same order, bit-identical) + 15
    interactions I1 at the end: for each row feature r in I1_ROW_FEATS (row_ent, row_max, row_d_ntop, row_ch_ntop, task_rows —
    constant within a row and therefore without effect on their own in the conditional logit) and each candidate feature c in
    I1_CAND_FEATS (f_logp, is_general, is_none) the product of the raw values X[r]·X[c] (r outer, c inner; standardised like all
    features). Belongs to the NONE guard like main_guard. Not in FEATURE_SETS either. Knowledge features as in main_guard: test rows
    from all 1,500 training rows, not rescaled (task_rows takes effect here for the first time).
  * Feature set "fallback_i1" (55; variant b of the safety line with set alignment): "fallback" (40, unchanged, bit-identical) + the
    same 15 interactions I1 at the end; reads no notes. Not in FEATURE_SETS.
  * Feature set "main_guard_i1_mm" (64): "main_guard_i1" (63, bit-identical) + minor_x_mech27 at the end = 1[the candidate has the
    default severity "minor" in the taxonomy] · mech27[row] (the final answer read by the pr27 channel matches the answer key
    mechanically, src/answer_match.py). The caller attaches the row feature and the class mask to Static with attach_mech. Not in
    FEATURE_SETS.

KNOWLEDGE FEATURES: for training rows ONLY from reference rows of other folds — knowledge_other_folds; the row itself and
its fold never count. For test rows from all training rows.

NOTE-STATISTICS GUARD (a): guard_notes. Without further parameters it applies the band rule (range of the three training
draws ± 10 % relative). With share_z_max, n_ref, n_test it applies the widened rule (`--guard-rule widened`): shares also pass
via a two-sided two-sample binomial test (share_z); the rule can only widen, never narrow.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

GENERAL_CODES = ("ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00")  # the five capped generic codes
K_CAND = 5
LAMBDA = 0.1
RANK_CAP = 30
BETA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
GAINS = np.array([1.0, 0.5, 1.0 / 3.0])
EXCLUDED_CHANNEL_FEATURES = ("q397", "q397m3")  # no log p/rank features of the 397B channels (their test side comes from a different compute path)
DRAWS = ("d1", "d2", "d3")
KNOW_FEATS = ("task_count", "task_rows", "task_share", "novel_task", "topic_count", "topic_rows", "topic_share", "novel_topic", "glob")
NOTE_FEATS = ("m3_men", "m3_last", "m3_len")
FEATURE_SETS = ("main", "fallback")
GUARD_SET = "main_guard"  # main + identifiers of the general codes; used together with the NONE guard (protect_none)
GENERAL_ID_FEATS = tuple(f"id_{c}" for c in GENERAL_CODES)
NONE_IDX = 0  # class order NONE + taxonomy (labels.load_classes): NONE is class 0
I1_SET = "main_guard_i1"  # main_guard + 15 interactions row × candidate; used together with the NONE guard
I1_ROW_FEATS = ("row_ent", "row_max", "row_d_ntop", "row_ch_ntop", "task_rows")  # constant within a row
I1_CAND_FEATS = ("f_logp", "is_general", "is_none")
I1_FEATS = tuple(f"{r}_x_{c}" for r in I1_ROW_FEATS for c in I1_CAND_FEATS)  # order: row feature outer, candidate feature inner
FB_I1_SET = "fallback_i1"  # variant b of the safety line with set alignment: fallback + interactions I1, without notes
MM_SET = "main_guard_i1_mm"  # main_guard_i1 + minor_x_mech27 (minor code × final answer matches the key mechanically)
MM_FEATS = ("minor_x_mech27",)


# ------------------------------------------------------------------------------------------------ building blocks

def ranks_of(P: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Rank of every class per row (0 = top); classes without a training example last; stable sort as in metric.top3_indices."""
    A = np.where(seen[None, :], P, -np.inf)
    order = np.argsort(-A, axis=1, kind="stable")
    r = np.empty_like(order)
    r[np.arange(len(P))[:, None], order] = np.arange(P.shape[1])[None, :]
    return r


def candidates(FM: np.ndarray, seen: np.ndarray, Kc: int = K_CAND) -> np.ndarray:
    """The first Kc classes of the base form per row (only classes with a training example, stable sort)."""
    A = np.where(seen[None, :], FM, -np.inf)
    return np.argsort(-A, axis=1, kind="stable")[:, :Kc]


def feature_names(chan_names: Sequence[str], feature_set: str = "main") -> list[str]:
    """Feature names in fixed order (= column order of pair_features)."""
    if feature_set == I1_SET:  # main_guard unchanged + interactions I1 at the end
        return feature_names(chan_names, GUARD_SET) + list(I1_FEATS)
    if feature_set == FB_I1_SET:  # fallback unchanged + interactions I1 at the end
        return feature_names(chan_names, "fallback") + list(I1_FEATS)
    if feature_set == MM_SET:  # main_guard_i1 unchanged + minor_x_mech27 at the end
        return feature_names(chan_names, I1_SET) + list(MM_FEATS)
    if feature_set not in FEATURE_SETS and feature_set != GUARD_SET:
        raise ValueError(f"unknown feature set {feature_set!r}")
    out = ["f_logp", "f_rank", "f_gap", "f_p", "p_logp", "p_rank"]
    for nm in chan_names:
        if nm not in EXCLUDED_CHANNEL_FEATURES:
            out += [f"{nm}_logp", f"{nm}_rank"]
    out += ["n_ch_top1", "n_ch_top3", "d1_rank", "d2_rank", "d3_rank", "d_top1", "d_std", *KNOW_FEATS, "is_general", "is_none"]
    if feature_set in ("main", GUARD_SET):
        out += list(NOTE_FEATS)
    out += ["row_none", "row_ent", "row_max", "row_d_ntop", "row_ch_ntop", "row_grade"]
    if feature_set == GUARD_SET:
        out += list(GENERAL_ID_FEATS)
    return out


def uses_notes(feature_set: str) -> bool:
    """Whether a feature set reads the M3 notes (main, main_guard, main_guard_i1, main_guard_i1_mm) — a fallback model must never do that."""
    return feature_set in ("main", GUARD_SET, I1_SET, MM_SET)


def attach_mech(st: "Static", mech27: np.ndarray, minor: np.ndarray) -> None:
    """Row feature mech27 (per row of st, 0/1) and class mask minor (per class, 0/1) for the feature set main_guard_i1_mm."""
    mech27, minor = np.asarray(mech27, dtype=np.float64), np.asarray(minor, dtype=np.float64)
    if mech27.shape != (st.n,) or minor.shape != (st.K,):
        raise ValueError(f"mech27 {mech27.shape} / minor {minor.shape} instead of ({st.n},) / ({st.K},)")
    if not (np.isin(mech27, (0.0, 1.0)).all() and np.isin(minor, (0.0, 1.0)).all()):
        raise ValueError("mech27 and minor must be 0/1")
    st.mech27, st.minor = mech27, minor


def i1_interactions(X: np.ndarray, names: Sequence[str], perm: np.ndarray | None = None) -> np.ndarray:
    """The 15 interactions I1 from the raw values of a feature block with the columns `names` (main_guard or main_guard_i1): for each
    row feature r (I1_ROW_FEATS, outer) and candidate feature c (I1_CAND_FEATS, inner) the product X[:, r] · X[:, c] in float64.
    perm (placebo only): pair indices by which the row values r are permuted BEFORE the product (the candidate values stay). -> [len(X), 15]."""
    X = np.asarray(X, dtype=np.float64)
    names = list(names)
    cols = []
    for r in I1_ROW_FEATS:
        rv = X[:, names.index(r)]
        if perm is not None:
            rv = rv[np.asarray(perm)]
        for c in I1_CAND_FEATS:
            cols.append(rv * X[:, names.index(c)])
    return np.stack(cols, 1)


# ------------------------------------------------------------------------------------------------ notes (counts only)

def mention_pattern(classes: Sequence[str]) -> re.Pattern:
    """Regex of the codes (without NONE), longest first, with word boundaries; case-sensitive (codes only from the class list)."""
    codes = sorted({str(c) for c in classes if c != "NONE"}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(c) for c in codes) + r")(?![A-Za-z0-9])")


def mention_features(texts: Sequence[str], classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: which codes the text mentions (0/1), which one is mentioned last (0/1), length in characters. Counts only.
    -> (men int8 [n, K], last int8 [n, K], len float [n])."""
    pat = mention_pattern(classes)
    idx = {c: k for k, c in enumerate(classes)}
    n, K = len(texts), len(classes)
    men = np.zeros((n, K), np.int8)
    last = np.zeros((n, K), np.int8)
    ln = np.zeros(n, np.float64)
    for i, t in enumerate(texts):
        t = t or ""
        ln[i] = len(t)
        found = pat.findall(t)
        for c in set(found):
            men[i, idx[c]] = 1
        if found:
            last[i, idx[found[-1]]] = 1
    return men, last, ln


def notes_features(per_draw_texts: Sequence[Sequence[str]], classes: Sequence[str]) -> dict:
    """Note features from the three M3 draws: sum of mentions (0–3), sum of last mentions, mean length; plus the mention
    matrix per draw (for the guard). -> {m3_men, m3_last, m3_len, men_per_draw}."""
    if len(per_draw_texts) != len(DRAWS):
        raise ValueError(f"{len(per_draw_texts)} draws instead of {len(DRAWS)}")
    n = len(per_draw_texts[0])
    M = np.zeros((n, len(classes)), np.int8)
    L = np.zeros((n, len(classes)), np.int8)
    lens, per = [], []
    for texts in per_draw_texts:
        if len(texts) != n:
            raise ValueError("draws with different row counts")
        m, la, ln = mention_features(texts, classes)
        M += m
        L += la
        lens.append(ln)
        per.append(m)
    return {"m3_men": M, "m3_last": L, "m3_len": np.mean(lens, axis=0), "men_per_draw": per}


# ------------------------------------------------------------------------------------------------ fixed row/candidate features

class Static:
    """Features that depend neither on the base form nor on reference rows: blend channels, M3 draws, notes, grade.
    Channels are the probabilities BEFORE the floor (OOF artifacts or test channels before the floor)."""

    def __init__(self, chan_names: Sequence[str], chans: dict, draws: Sequence[np.ndarray], grade: np.ndarray, seen: np.ndarray,
                 classes: Sequence[str], notes: dict | None = None):
        self.seen = np.asarray(seen, bool)
        self.K = len(self.seen)
        self.chan_names = list(chan_names)
        self.n = len(grade)
        self.ch_logp, self.ch_rank = {}, {}
        ranks = []
        for nm in self.chan_names:
            P = np.asarray(chans[nm], dtype=np.float64)
            if P.shape != (self.n, self.K):
                raise ValueError(f"channel {nm}: shape {P.shape}, expected {(self.n, self.K)}")
            r = np.minimum(ranks_of(P, self.seen), RANK_CAP)
            ranks.append(r)
            if nm not in EXCLUDED_CHANNEL_FEATURES:
                self.ch_logp[nm] = np.log(np.clip(P, 1e-9, 1.0))
                self.ch_rank[nm] = r
        self.n_top1 = sum((r == 0).astype(float) for r in ranks)
        self.n_top3 = sum((r < 3).astype(float) for r in ranks)
        tops = np.stack([r.argmin(1) for r in ranks], 1)
        self.ch_ntop = np.array([len(set(t)) for t in tops], float)
        if len(draws) != len(DRAWS):
            raise ValueError(f"{len(draws)} M3 draws instead of {len(DRAWS)}")
        dr = [np.minimum(ranks_of(np.asarray(d, np.float64), self.seen), RANK_CAP) for d in draws]
        self.d_rank = dr
        self.d_top1 = sum((r == 0).astype(float) for r in dr)
        dl = np.stack([np.log(np.clip(np.asarray(d, np.float64), 1e-6, 1.0)) for d in draws], 0)
        self.d_std = dl.std(0)
        dt = np.stack([r.argmin(1) for r in dr], 1)
        self.d_ntop = np.array([len(set(t)) for t in dt], float)
        self.grade = np.asarray(grade, dtype=float)
        self.general_idx = [list(classes).index(c) for c in GENERAL_CODES]
        self.is_general = np.zeros(self.K)
        self.is_general[self.general_idx] = 1.0
        self.notes = None
        if notes is not None:
            self.notes = {"m3_men": np.asarray(notes["m3_men"]).astype(float), "m3_last": np.asarray(notes["m3_last"]).astype(float),
                          "m3_len": np.log1p(np.asarray(notes["m3_len"], dtype=np.float64))}


# ------------------------------------------------------------------------------------------------ knowledge from reference rows

def task_keys(qid: Sequence, topic: Sequence) -> np.ndarray:
    """Task key question_id|topic_name (as in the reranker family)."""
    return np.char.add(np.char.add(np.asarray(qid).astype(str), "|"), np.asarray(topic).astype(str))


def knowledge(ref_key: np.ndarray, ref_topic: np.ndarray, ref_y: np.ndarray, key: np.ndarray, topic: np.ndarray, K: int) -> dict:
    """Label counts of the reference rows per task and per topic, for the rows (key, topic). Only labels of reference rows."""
    task, top = {}, {}
    for a, b, c in zip(ref_key, ref_topic, ref_y):
        task.setdefault(a, np.zeros(K))[int(c)] += 1
        top.setdefault(b, np.zeros(K))[int(c)] += 1
    z0 = np.zeros(K)
    tc = np.stack([task.get(a, z0) for a in key]) if len(key) else np.zeros((0, K))
    pc = np.stack([top.get(b, z0) for b in topic]) if len(key) else np.zeros((0, K))
    glob = np.bincount(np.asarray(ref_y, dtype=np.int64), minlength=K) / len(ref_y)
    return {"task_count": tc, "task_rows": tc.sum(1), "topic_count": pc, "topic_rows": pc.sum(1), "glob": glob}


def knowledge_other_folds(key: np.ndarray, topic: np.ndarray, y: np.ndarray, folds: np.ndarray, rows: np.ndarray, K: int,
                          exclude: Sequence[int] = ()) -> dict:
    """Knowledge for the training rows `rows` of ONE fold: reference = rows of all other folds except `exclude` (the row itself
    and its fold never count; `exclude` = folds additionally left out by the nested reproduction)."""
    fs = set(int(v) for v in np.asarray(folds)[rows])
    if len(fs) != 1:
        raise ValueError(f"rows from several folds {sorted(fs)}")
    f = fs.pop()
    folds = np.asarray(folds)
    ref = np.flatnonzero((folds != f) & ~np.isin(folds, list(exclude)))
    return knowledge(key[ref], topic[ref], y[ref], key[rows], topic[rows], K)


# ------------------------------------------------------------------------------------------------ pair features

def pair_features(st: Static, rows: np.ndarray, FM: np.ndarray, PM: np.ndarray, kn: dict, feature_set: str = "main",
                  Kc: int = K_CAND) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features per (row, candidate) -> X [len(rows)·Kc, f], candidates [len(rows), Kc], names.
    rows: positions of the rows in `st` (table); FM/PM: base form and blend before the last step of EXACTLY these rows;
    kn: knowledge(...) of EXACTLY these rows."""
    if feature_set == MM_SET:  # main_guard_i1 bit-identical + minor_x_mech27 at the end (row feature and mask via attach_mech)
        mech27, minor = getattr(st, "mech27", None), getattr(st, "minor", None)
        if mech27 is None or minor is None:
            raise ValueError(f"feature set {MM_SET} requires attach_mech (mech27 per row, minor per class)")
        X, cand, names = pair_features(st, rows, FM, PM, kn, I1_SET, Kc)
        rows = np.asarray(rows)
        gi = rows[np.repeat(np.arange(len(rows)), Kc)]
        col = mech27[gi] * minor[cand.reshape(-1)]
        out_names = names + list(MM_FEATS)
        want = feature_names(st.chan_names, MM_SET)
        if out_names != want:
            raise AssertionError(f"feature order {out_names} != {want}")
        return np.concatenate([X, col[:, None]], 1), cand, out_names
    if feature_set in (I1_SET, FB_I1_SET):  # base set bit-identical + interactions I1 at the end (the other sets unchanged)
        X, cand, names = pair_features(st, rows, FM, PM, kn, GUARD_SET if feature_set == I1_SET else "fallback", Kc)
        out_names = names + list(I1_FEATS)
        want = feature_names(st.chan_names, feature_set)
        if out_names != want:
            raise AssertionError(f"feature order {out_names} != {want}")
        return np.concatenate([X, i1_interactions(X, names)], 1), cand, out_names
    if feature_set == "main" and st.notes is None:
        raise ValueError("feature set main requires notes")
    if feature_set == GUARD_SET and st.notes is None:
        raise ValueError(f"feature set {GUARD_SET} requires notes")
    rows = np.asarray(rows)
    R = len(rows)
    cand = candidates(FM, st.seen, Kc)
    ii = np.repeat(np.arange(R), Kc)
    cc = cand.reshape(-1)
    gi = rows[ii]
    f_logp_all = np.log(np.clip(FM, 1e-12, 1.0))
    f_logp = f_logp_all[ii, cc]
    top_logp = f_logp_all[np.arange(R), cand[:, 0]][ii]
    cols, names = [], []

    def add(nm, v):
        names.append(nm)
        cols.append(np.asarray(v, float))

    add("f_logp", f_logp)
    add("f_rank", np.tile(np.arange(Kc), R))
    add("f_gap", top_logp - f_logp)
    add("f_p", FM[ii, cc])
    p_logp_all = np.log(np.clip(PM, 1e-12, 1.0))
    add("p_logp", p_logp_all[ii, cc])
    add("p_rank", np.minimum(ranks_of(PM, st.seen), RANK_CAP)[ii, cc])
    for nm in st.chan_names:
        if nm in st.ch_logp:
            add(f"{nm}_logp", st.ch_logp[nm][gi, cc])
            add(f"{nm}_rank", st.ch_rank[nm][gi, cc])
    add("n_ch_top1", st.n_top1[gi, cc])
    add("n_ch_top3", st.n_top3[gi, cc])
    for d, rk in zip(DRAWS, st.d_rank):
        add(f"{d}_rank", rk[gi, cc])
    add("d_top1", st.d_top1[gi, cc])
    add("d_std", st.d_std[gi, cc])
    tc, pc = kn["task_count"], kn["topic_count"]
    add("task_count", tc[ii, cc])
    add("task_rows", kn["task_rows"][ii])
    add("task_share", tc[ii, cc] / np.maximum(kn["task_rows"][ii], 1))
    add("novel_task", ((tc[ii, cc] == 0) & (cc != 0)).astype(float))
    add("topic_count", pc[ii, cc])
    add("topic_rows", kn["topic_rows"][ii])
    add("topic_share", pc[ii, cc] / np.maximum(kn["topic_rows"][ii], 1))
    add("novel_topic", ((pc[ii, cc] == 0) & (cc != 0)).astype(float))
    add("glob", kn["glob"][cc])
    add("is_general", st.is_general[cc])
    add("is_none", (cc == 0).astype(float))
    if feature_set in ("main", GUARD_SET):
        add("m3_men", st.notes["m3_men"][gi, cc])
        add("m3_last", st.notes["m3_last"][gi, cc])
        add("m3_len", st.notes["m3_len"][gi])
    ent = -(FM * np.log(np.clip(FM, 1e-12, 1.0))).sum(1)
    add("row_none", FM[:, 0][ii])
    add("row_ent", ent[ii])
    add("row_max", FM.max(1)[ii])
    add("row_d_ntop", st.d_ntop[gi])
    add("row_ch_ntop", st.ch_ntop[gi])
    add("row_grade", st.grade[gi])
    if feature_set == GUARD_SET:  # identifiers of the general codes (no row information), at the end
        for nm, g in zip(GENERAL_ID_FEATS, st.general_idx):
            add(nm, (cc == g).astype(float))
    want = feature_names(st.chan_names, feature_set)
    if names != want:
        raise AssertionError(f"feature order {names} != {want}")
    return np.stack(cols, 1), cand, names


# ------------------------------------------------------------------------------------------------ model

def labels_of(cand: np.ndarray, y_rows: np.ndarray) -> np.ndarray:
    """Target per pair: 1 for the gold candidate (flat, length R·Kc)."""
    return (cand == np.asarray(y_rows)[:, None]).astype(int).reshape(-1)


def fit_clogit(X: np.ndarray, lab: np.ndarray, Kc: int = K_CAND, lam: float = LAMBDA) -> dict:
    """Conditional logit (softmax over the Kc candidates of a row), standardised features, L2 λ, L-BFGS-B from zero.
    -> {w, mu, sd} (lists of float are JSON-safe; repr is lossless)."""
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    R = len(X) // Kc
    Z3, L3 = Z.reshape(R, Kc, -1), np.asarray(lab).reshape(R, Kc).astype(float)
    keep = L3.sum(1) > 0
    Z3, L3 = Z3[keep], L3[keep]

    def f(w):
        s = Z3 @ w
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        ll = (s * L3).sum(1) - lse
        p = np.exp(s - lse[:, None])
        g = -np.einsum("rk,rkf->f", L3 - p, Z3) / len(Z3) + lam * w
        return -ll.mean() + 0.5 * lam * float(w @ w), g

    w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return {"w": w, "mu": mu, "sd": sd}


def zero_model(n_features: int) -> dict:
    """Weight 0 (copy null point): score 0 everywhere, the stable sort keeps the base order."""
    return {"w": np.zeros(n_features), "mu": np.zeros(n_features), "sd": np.ones(n_features)}


def scores(model: dict, X: np.ndarray, Kc: int = K_CAND) -> np.ndarray:
    """Score per pair -> [R, Kc]."""
    w, mu, sd = (np.asarray(model[k], dtype=np.float64) for k in ("w", "mu", "sd"))
    if X.shape[1] != len(w):
        raise ValueError(f"{X.shape[1]} features, the model has {len(w)}")
    return (((np.asarray(X, np.float64) - mu) / sd) @ w).reshape(-1, Kc)


def reorder(cand: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Candidates by descending score (stable: ties keep the base order) -> [R, Kc] class indices."""
    order = np.argsort(-score, axis=1, kind="stable")
    return np.take_along_axis(cand, order, 1)


def protect_none(top: np.ndarray, cand: np.ndarray, none_idx: int = NONE_IDX) -> np.ndarray:
    """NONE guard: if the base form has NONE at the top (cand[:, 0]) and the new order would change the top, the base order of the
    whole row stays. Otherwise unchanged. -> new order [R, Kc]."""
    top, cand = np.asarray(top), np.asarray(cand)
    if top.shape != cand.shape:
        raise ValueError(f"shape {top.shape} != {cand.shape}")
    m = (cand[:, 0] == none_idx) & (top[:, 0] != none_idx)
    out = top.copy()
    out[m] = cand[m]
    return out


def map_rows(top: np.ndarray, y_rows: np.ndarray) -> np.ndarray:
    """MAP@3 per row from ordered candidates (the first three count) — the same gains as in metric."""
    return ((np.asarray(top)[:, :3] == np.asarray(y_rows)[:, None]) * GAINS[None, :]).sum(1)


def permute_labels(lab: np.ndarray, Kc: int, rng: np.random.Generator) -> np.ndarray:
    """Label placebo: in every row with the gold label among the candidates the target is put on a random candidate."""
    L = np.asarray(lab).reshape(-1, Kc).copy()
    has = L.sum(1) > 0
    pos = rng.integers(0, Kc, size=len(L))
    L[has] = 0
    L[np.flatnonzero(has), pos[has]] = 1
    return L.reshape(-1)


def gate(blocks: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Kc: int = K_CAND, lam: float = LAMBDA,
         protect: bool = False) -> dict:
    """Abstention gate: for every block a, fit on the other blocks and evaluate on a (true labels); pooled MAP@3 against the
    base order of the same rows. blocks: [(X, target per pair, candidates, y of the rows)]. Abstain if not better.
    protect=True (main_guard only): the new order is evaluated AFTER the NONE guard (as applied); the default is unchanged."""
    got, base = [], []
    for a in range(len(blocks)):
        tr = [b for b in range(len(blocks)) if b != a]
        m = fit_clogit(np.concatenate([blocks[b][0] for b in tr]), np.concatenate([blocks[b][1] for b in tr]), Kc, lam)
        X, _, cand, yr = blocks[a]
        top = reorder(cand, scores(m, X, Kc))
        got.append(map_rows(protect_none(top, cand) if protect else top, yr))
        base.append(map_rows(cand, yr))
    inner, inner_base = float(np.concatenate(got).mean()), float(np.concatenate(base).mean())
    return {"inner": inner, "inner_base": inner_base, "abstain": bool(not (inner > inner_base + 1e-12))}


# ------------------------------------------------------------------------------------------------ test side

def apply_to_probs(P: np.ndarray, cand: np.ndarray, new_order: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, int]:
    """Transfer the new candidate order into the probability matrix: the values of the first Kc candidates (descending) are
    assigned in the new order — row sums and the other classes stay unchanged, metric.top3_indices returns the new order. Ties
    that would hide the new order are made strictly decreasing with np.nextafter (counted). With an unchanged order the result
    is bit-identical to P. -> (new P, number of rows with a tie correction)."""
    P = np.asarray(P, dtype=np.float64)
    R, Kc = cand.shape
    vals = np.take_along_axis(P, cand, 1)  # descending (base order)
    changed = (new_order != cand).any(1)
    out = P.copy()
    fixed = 0
    for i in np.flatnonzero(changed):
        v = vals[i].copy()
        if (np.diff(v) >= 0).any():  # tie: make strictly decreasing, otherwise the class index would decide
            for j in range(1, Kc):
                if v[j] >= v[j - 1]:
                    v[j] = np.nextafter(v[j - 1], -np.inf)
            fixed += 1
        out[i, new_order[i]] = v
    if changed.any():
        out[changed] = out[changed] / out[changed].sum(1, keepdims=True)
        A = np.where(seen[None, :], out[changed], -np.inf)
        got = np.argsort(-A, axis=1, kind="stable")[:, :Kc]
        if not np.array_equal(got, new_order[changed]):
            raise AssertionError("new order cannot be transferred into the probabilities")
    return out, fixed


# ------------------------------------------------------------------------------------------------ guard (aggregates only)

def note_stats(men_per_draw: Sequence[np.ndarray], m3_men: np.ndarray, cand: np.ndarray, top1: np.ndarray) -> dict:
    """Note statistics (aggregates only): per draw the mean number of codes mentioned per row and the share of rows whose note
    mentions the top code of the blend; distribution of the mention count 0–3 over d1–d3 for the first Kc candidates."""
    n = len(top1)
    per = {}
    for d, m in zip(DRAWS, men_per_draw):
        per[d] = {"codes_per_row": float(np.asarray(m).sum(1).mean()), "share_top_mentioned": float((np.asarray(m)[np.arange(n), top1] > 0).mean())}
    vals = np.take_along_axis(np.asarray(m3_men), cand, 1).reshape(-1).astype(int)
    hist = np.bincount(np.clip(vals, 0, 3), minlength=4) / len(vals)
    return {"per_draw": per, "mention_hist_top5": [float(v) for v in hist]}


def identical_share(texts_a: Sequence[str], texts_b: Sequence[str]) -> float:
    """Share of rows with a word-identical note in two draws (aggregate only)."""
    if len(texts_a) != len(texts_b):
        raise ValueError("different row counts")
    return float(np.mean([(a or "") == (b or "") for a, b in zip(texts_a, texts_b)]))


def band(train_values: Sequence[float], rel: float = 0.10) -> tuple[float, float]:
    """Range of the training values ± rel relative: [min·(1 − rel), max·(1 + rel)] (for non-negative statistics)."""
    lo, hi = float(min(train_values)), float(max(train_values))
    return lo * (1.0 - rel), hi * (1.0 + rel)


def share_z(p_ref: float, n_ref: int, p_test: float, n_test: int) -> float:
    """Two-sided two-sample binomial test for a share (widened rule): pooled share p = (x_ref + x_test)/(n_ref + n_test)
    with x = share·n, SE = sqrt(p(1 − p)(1/n_ref + 1/n_test)), z = (share_test − share_ref)/SE. If SE = 0, then z = 0 for a zero
    difference, otherwise ±infinity (alarm). If the pooled share lies outside [0, 1] (invalid input), SE = 0 and z = ±infinity;
    NaN yields NaN — both raise the alarm."""
    p_ref, p_test, n_ref, n_test = float(p_ref), float(p_test), float(n_ref), float(n_test)
    p = (p_ref * n_ref + p_test * n_test) / (n_ref + n_test)
    se = math.sqrt(max(p * (1.0 - p), 0.0) * (1.0 / n_ref + 1.0 / n_test))
    d = p_test - p_ref
    if se == 0.0:
        return 0.0 if d == 0.0 else math.copysign(math.inf, d)
    return d / se


def guard_notes(ref: dict, test: dict, rel: float = 0.10, share_z_max: float | None = None, n_ref: int | None = None,
                n_test: int | None = None) -> dict:
    """Guard (a) of the test notes against the training draws. ref/test: {per_draw: {d: {codes_per_row, share_top_mentioned}},
    mention_hist_top5: [4], identical: {d1-d2, d1-d3, d2-d3}}. Per statistic: range of the three training draws ± rel
    (distribution: the single training value ± rel). -> {checks: [...], ok}.

    Widened rule (only with share_z_max, n_ref = training rows, n_test = test rows): a statistic passes if it lies in the old band
    OR — for shares only — |share_z| ≤ share_z_max. Reference value per share: share_top_mentioned of a draw against the training
    value of THE SAME draw, mention_hist_top5 of a level against the training value of THE SAME level, word-identical notes of a
    pair against the pair's OWN training value; n = number of rows (also for the histogram levels). codes_per_row only uses the
    old band. Every check then additionally carries ok_band (old verdict), rule, ref, z, ok_z and passed_by ("band", "z" or None).
    Without share_z_max the return value is exactly that of the band rule."""
    widened = share_z_max is not None
    if widened:
        if n_ref is None or n_test is None or not (int(n_ref) > 0 and int(n_test) > 0 and float(share_z_max) > 0):
            raise ValueError("the widened rule requires share_z_max > 0, n_ref > 0 and n_test > 0")
        share_z_max, n_ref, n_test = float(share_z_max), int(n_ref), int(n_test)
    elif n_ref is not None or n_test is not None:
        raise ValueError("n_ref/n_test only together with share_z_max (widened rule)")
    checks = []

    def add(stat: str, v: float, lo: float, hi: float, ref_value: float | None) -> None:
        ok = bool(lo <= v <= hi)
        c = {"stat": stat, "test": v, "lo": lo, "hi": hi, "ok": ok}
        if widened:  # widen only: ok = old band OR binomial test (shares only)
            z = None if ref_value is None else share_z(ref_value, n_ref, v, n_test)
            ok_z = None if z is None else bool(abs(z) <= share_z_max)
            c["ok"] = bool(ok or ok_z is True)
            c.update({"ok_band": ok, "rule": "band" if z is None else "band|z", "ref": ref_value, "z": z, "ok_z": ok_z,
                      "passed_by": "band" if ok else ("z" if c["ok"] else None)})
        checks.append(c)

    for key in ("codes_per_row", "share_top_mentioned"):
        lo, hi = band([ref["per_draw"][d][key] for d in DRAWS], rel)
        for d in DRAWS:
            add(f"{key}[{d}]", test["per_draw"][d][key], lo, hi, ref["per_draw"][d][key] if key == "share_top_mentioned" else None)
    for b, (r, v) in enumerate(zip(ref["mention_hist_top5"], test["mention_hist_top5"])):
        lo, hi = band([r], rel)
        add(f"mention_hist_top5[{b}]", v, lo, hi, r)
    lo, hi = band(list(ref["identical"].values()), rel)
    for pair, v in test["identical"].items():
        if widened and pair not in ref["identical"]:
            raise ValueError(f"word-identical notes: pair {pair} missing from the reference")
        add(f"identical[{pair}]", v, lo, hi, ref["identical"][pair] if widened else None)
    out = {"checks": checks, "ok": bool(all(c["ok"] for c in checks)), "rel": rel}
    if widened:
        out.update({"rule": "widened", "share_z_max": share_z_max, "n_ref": n_ref, "n_test": n_test})
    return out
