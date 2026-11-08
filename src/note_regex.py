"""Regex-exact note columns R for the second stage of the safety line (feature set fallback_i1_ndr) — versioned extractor.

Pure functions without file access: mentions of the codes in a text via stage2.mention_pattern (longest codes first, word boundaries,
case-sensitive, NONE never). Per (row, class) and source (mention_stats):
  cnt       number of mentions (finditer),
  is_first  1 if the code is the FIRST code mentioned in the text,
  is_last   1 if the code is the LAST code mentioned in the text,
  last_rel  relative character position of the last mention = start of the last match / text length (not mentioned: 0).
The definitions are fixed with the version string VERSION; tests/test_safety_ndr.py checks them on small texts.

R (six columns per (row, class), order stage2.R_FEATS):
  t_m3_logcnt   mean over the M3 draws of log(1 + cnt)
  t_m3_first    number of M3 draws in which the code is mentioned first (0–3)
  t_m3_lastpos  mean over the M3 draws of last_rel (0 in draws without a mention)
  t_pr_men      1[pr27 mentions the code] (notes of the first pr27 draw: reasoning of the first pass, tools/export_proto_notes.py)
  t_pr_last     1[the code is the last one mentioned in the pr27 note]
  t_pr_logcnt   log(1 + cnt) in the pr27 note
Empty or missing note: all columns 0 (no mention). No labels, row-local, deterministic.

GUARD (aggregates only, abort only): three statistics, test against the reference of the training rows at the candidates of the base form
(guard_stats): pr27 mention rate per row (mean number of mentioned codes; mean), share of rows in which pr27 mentions the top of the base form
(share), per M3 draw the mean last position of the mentioned candidates (mean over the pairs). A statistic passes within the band
reference·(1 ± 0.10) OR at |z| ≤ 3 (guard): means with z = (m_test − m_ref)/sqrt(s_ref²/n_ref + s_test²/n_test) as note_sim.sim_guard,
shares with the two-sample binomial test stage2.share_z as the widened note guard — the rule can only widen.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

import stage2 as S2

VERSION = "note_regex/1"
R_FEATS = S2.R_FEATS
M3_DRAWS = S2.DRAWS
REL = 0.10
Z_MAX = 3.0
STAT_KEYS = ("cnt", "is_first", "is_last", "last_rel")


def mention_stats(texts: Sequence[str], classes: Sequence[str]) -> dict:
    """Per (row, class): cnt, is_first, is_last, last_rel (see the module docstring) in float64. -> {key: [n, K]}."""
    pat = S2.mention_pattern(classes)
    idx = {c: k for k, c in enumerate(classes)}
    n, K = len(texts), len(classes)
    out = {k: np.zeros((n, K), np.float64) for k in STAT_KEYS}
    cnt, first, last, last_rel = (out[k] for k in STAT_KEYS)
    for i, t in enumerate(texts):
        t = t or ""
        L = len(t)
        k0 = k1 = None
        for m in pat.finditer(t):
            k = idx[m.group(1)]
            cnt[i, k] += 1.0
            last_rel[i, k] = m.start() / L
            if k0 is None:
                k0 = k
            k1 = k
        if k0 is not None:
            first[i, k0] = 1.0
            last[i, k1] = 1.0
    return out


def r_features(m3_stats: Sequence[dict], pr_stats: dict) -> np.ndarray:
    """The six columns R from the mentions of the M3 draws (order of the draws irrelevant; averaged or summed) and of the pr27 note.
    -> [n, K, 6] float64 in the order stage2.R_FEATS."""
    if len(m3_stats) != len(M3_DRAWS):
        raise ValueError(f"{len(m3_stats)} M3 draws instead of {len(M3_DRAWS)}")
    shapes = {np.shape(s["cnt"]) for s in [*m3_stats, pr_stats]}
    if len(shapes) != 1:
        raise ValueError(f"mentions with different shapes {sorted(shapes)}")
    cnt = np.stack([np.asarray(s["cnt"], np.float64) for s in m3_stats], 0)
    first = np.stack([np.asarray(s["is_first"], np.float64) for s in m3_stats], 0)
    lastrel = np.where(cnt > 0, np.stack([np.asarray(s["last_rel"], np.float64) for s in m3_stats], 0), 0.0)
    pc = np.asarray(pr_stats["cnt"], np.float64)
    cols = {"t_m3_logcnt": np.log1p(cnt).mean(0), "t_m3_first": first.sum(0), "t_m3_lastpos": lastrel.mean(0),
            "t_pr_men": (pc > 0).astype(np.float64), "t_pr_last": np.asarray(pr_stats["is_last"], np.float64), "t_pr_logcnt": np.log1p(pc)}
    R = np.stack([cols[k] for k in R_FEATS], 2)
    if not np.isfinite(R).all():
        raise ValueError("R contains non-finite values")
    return R


# ------------------------------------------------------------------------------------------------ guard (aggregates only)

def _mean_sd_n(v: np.ndarray) -> dict:
    v = np.asarray(v, np.float64).reshape(-1)
    return {"mean": float(v.mean()) if v.size else 0.0, "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def guard_stats(m3_stats: Sequence[dict], pr_stats: dict, cand: np.ndarray) -> dict:
    """Statistics of the guard of the columns R at the candidates of the base form (cand [n, Kc], top = column 0); aggregates only."""
    cand = np.asarray(cand)
    n = len(cand)
    rr = np.arange(n)
    men = np.asarray(pr_stats["cnt"]) > 0
    if men.shape[0] != n or len(m3_stats) != len(M3_DRAWS):
        raise ValueError("mentions and candidates do not match")
    per = {}
    for d, st in zip(M3_DRAWS, m3_stats):
        c_men = np.take_along_axis(np.asarray(st["cnt"]), cand, 1) > 0
        c_last = np.take_along_axis(np.asarray(st["last_rel"], np.float64), cand, 1)
        per[d] = _mean_sd_n(c_last[c_men])
    return {"pr27_codes_per_row": _mean_sd_n(men.sum(1).astype(np.float64)),
            "pr27_share_top_mentioned": {"share": float(men[rr, cand[:, 0]].mean()), "n": int(n)},
            "m3_last_pos_cand": per}


def mean_check(stat: str, ref: dict, test: dict, rel: float = REL, z_max: float = Z_MAX) -> dict:
    """Mean: band ref·(1 ± rel) OR |z| ≤ z_max, z = (m_test − m_ref)/sqrt(s_ref²/n_ref + s_test²/n_test)."""
    lo, hi = ref["mean"] * (1.0 - rel), ref["mean"] * (1.0 + rel)
    se = math.sqrt(ref["sd"] ** 2 / max(ref["n"], 1) + test["sd"] ** 2 / max(test["n"], 1))
    d = test["mean"] - ref["mean"]
    z = (0.0 if d == 0.0 else math.copysign(math.inf, d)) if se == 0.0 else d / se
    ok_band, ok_z = bool(lo <= test["mean"] <= hi), bool(abs(z) <= z_max)
    return {"stat": stat, "kind": "mean", "test": test["mean"], "ref": ref["mean"], "lo": lo, "hi": hi, "z": z, "ok_band": ok_band, "ok_z": ok_z,
            "ok": bool(ok_band or ok_z), "passed_by": "band" if ok_band else ("z" if ok_z else None), "n_ref": ref["n"], "n_test": test["n"]}


def share_check(stat: str, ref: dict, test: dict, rel: float = REL, z_max: float = Z_MAX) -> dict:
    """Share: band ref·(1 ± rel) OR |stage2.share_z| ≤ z_max (two-sample binomial test, as the widened note guard)."""
    lo, hi = ref["share"] * (1.0 - rel), ref["share"] * (1.0 + rel)
    z = S2.share_z(ref["share"], ref["n"], test["share"], test["n"])
    ok_band, ok_z = bool(lo <= test["share"] <= hi), bool(abs(z) <= z_max)
    return {"stat": stat, "kind": "share", "test": test["share"], "ref": ref["share"], "lo": lo, "hi": hi, "z": z, "ok_band": ok_band, "ok_z": ok_z,
            "ok": bool(ok_band or ok_z), "passed_by": "band" if ok_band else ("z" if ok_z else None), "n_ref": ref["n"], "n_test": test["n"]}


def guard(ref: dict, test: dict, rel: float = REL, z_max: float = Z_MAX) -> dict:
    """Guard of the columns R: five checks (pr27 mention rate, top mentioned by pr27, last M3 position per draw). -> {ok, checks, rel, z_max}."""
    checks = [mean_check("pr27_codes_per_row", ref["pr27_codes_per_row"], test["pr27_codes_per_row"], rel, z_max),
              share_check("pr27_share_top_mentioned", ref["pr27_share_top_mentioned"], test["pr27_share_top_mentioned"], rel, z_max)]
    for d in M3_DRAWS:
        checks.append(mean_check(f"m3_last_pos_cand[{d}]", ref["m3_last_pos_cand"][d], test["m3_last_pos_cand"][d], rel, z_max))
    return {"ok": bool(all(c["ok"] for c in checks)), "checks": checks, "rel": rel, "z_max": z_max, "rule": "safety-ndr rule (6)"}
