"""Risk rule of the second stages of the safety line: posterior probability of a loss larger than 0.002.

A step is deployed in the safety line only if P(θ < −0.002) ≤ 0.05, where θ is the effect of the step on MAP@3 for a typical new draw of the
generated texts on the test side (docs/VALIDATION.md, risk rule). Used by eda/c129_safety_nd.py rule and eda/c130_safety_ndr.py rule.

* PRIOR: empirical Bayes. A nonparametric maximum-likelihood estimate (NPMLE, EM on a fixed grid) from the earlier paired measurements of
  the same class of component, CLASS_R_MEASUREMENTS: steps that convert information the channels already provide (calibration and
  alignment steps, rerankers, pooling steps and verifiers on existing texts). Each measurement is the paired difference in MAP@3 of the
  line the step was measured in (primary line if measured there, otherwise safety line) with its standard error. The list, its order and
  the grid are fixed; the prior is recomputed from them on every call.
* POSTERIOR: combined_posterior — the observed difference is treated as the maximum of k correlated variants (selection correction; error
  correlation rho, spread of the true variant effects `spread`), optionally with a between-draw variance tau_d of the generated texts; the
  target is the effect of a typical new draw. Monte-Carlo with a fixed seed of the caller (numpy.random.default_rng), accepted where the
  simulated maximum lies within ±tol of the observation.

Pure functions without file access; only numpy and scipy.
"""
from __future__ import annotations

import numpy as np

GRID = np.round(np.arange(-0.020, 0.0705, 0.0005), 6)

# (short description, observed paired difference in MAP@3, its standard error) — the 12 earlier measurements of class R, in their recorded order
CLASS_R_MEASUREMENTS = (
    ("inductive class reweighting of the protocol channel", 0.0011, 0.0016),
    ("inductive alignment of the blend (safety line)", 0.0109, 0.0031),
    ("pooling over test neighbours of the same question (mean)", 0.0013, 0.0014),
    ("pooling over neighbours with the same final answer", -0.0004, 0.0008),
    ("transition matrix at the class level", -0.0125, 0.0030),
    ("offset for codes new to the question", -0.0003, 0.0006),
    ("count ranker as replacement of the protocol channel", -0.0033, 0.0022),
    ("semantic comparison of findings", 0.0013, 0.0016),
    ("learned code mapping", -0.0023, 0.0019),
    ("small-encoder evidence verifier", -0.0019, 0.0021),
    ("small-encoder evidence verifier with context", -0.0003, 0.0022),
    ("fixed class correction of the verifier", 0.0012, 0.0016),
)


def npmle(x: np.ndarray, s: np.ndarray, iters: int = 5000) -> np.ndarray:
    from scipy import stats

    L = stats.norm.pdf(x[:, None], GRID[None, :], s[:, None])
    w = np.full(len(GRID), 1 / len(GRID))
    for _ in range(iters):
        post = L * w
        post /= post.sum(1, keepdims=True)
        w_new = post.mean(0)
        if np.abs(w_new - w).max() < 1e-10:
            w = w_new
            break
        w = w_new
    return w


def class_r_prior() -> np.ndarray:
    """Prior weights on GRID: NPMLE over CLASS_R_MEASUREMENTS."""
    x = np.array([m[1] for m in CLASS_R_MEASUREMENTS])
    s = np.array([m[2] for m in CLASS_R_MEASUREMENTS])
    return npmle(x, s)


def combined_posterior(o1: float, s1: float, w: np.ndarray, rng: np.random.Generator, k: int = 6, rho: float = 0.5,
                       o2: float | None = None, s2: float | None = None, r12: float = 0.0, tau_d: float = 0.0,
                       spread: float = 0.0015, n_sim: int = 400_000, tol: float = 0.0004, chunks: int = 8) -> dict:
    """Posterior of the effect of a TYPICAL draw (deployment on a new draw of the test side) from a first measurement chosen after the fact
    (d1: maximum of k correlated variants, error correlation rho, spread of the true variant effects `spread`) and optionally a repetition
    fixed in advance on a second draw (d2; sampling error correlated with that of d1 by r12, same rows and labels).

    Model: θ ~ prior (grid w); variant j: φ_j = θ + u_j, u_j ~ N(0, spread²) (k = 1: spread 0); draw d: φ + δ_d, δ_d ~ N(0, tau_d²)
    independent (difference between draws of the same kind); d1: o_j = φ_j + δ_1 + e_j, chosen j* = argmax, accepted |o_j* − o1| < tol;
    d2: o2 ~ N(φ_j* + δ_2 + β·e_j*, s2²(1 − r12²)), β = r12·s2/s1 (as a weight); target: φ_j* + δ_test. -> mean, P(target > 0),
    P(target < −0.002), P(target ≥ 0.002), effective sample size."""
    from scipy import stats as st

    grid = GRID
    sp = spread if k > 1 else 0.0
    phis, errs = [], []
    for _ in range(chunks):
        theta = rng.choice(grid, size=n_sim, p=w)
        phi_j = theta[:, None] + (rng.normal(0, sp, size=(n_sim, k)) if sp > 0 else np.zeros((n_sim, k)))
        delta1 = rng.normal(0, tau_d, size=(n_sim, 1)) if tau_d > 0 else 0.0
        common = rng.normal(0, 1, size=(n_sim, 1))
        indiv = rng.normal(0, 1, size=(n_sim, k))
        eps = s1 * (np.sqrt(rho) * common + np.sqrt(1 - rho) * indiv) if k > 1 else s1 * indiv
        obs = phi_j + delta1 + eps
        j = obs.argmax(1)
        rows = np.arange(n_sim)
        keep = np.abs(obs[rows, j] - o1) < tol
        phis.append(phi_j[rows, j][keep])
        errs.append(eps[rows, j][keep])
    phi, e1 = np.concatenate(phis), np.concatenate(errs)
    wts = np.ones(len(phi))
    if o2 is not None:
        r12 = float(np.clip(r12, -0.999, 0.999))  # a fully correlated repetition (dry run) would be degenerate
        beta = r12 * s2 / s1
        sd = np.sqrt(s2 ** 2 * (1 - r12 ** 2) + tau_d ** 2)
        wts = st.norm.pdf(o2, loc=phi + beta * e1, scale=sd)
    tot = wts.sum()
    if not len(phi) or tot <= 0:
        return {"mean": None, "p_pos": None, "p_lt_m002": None, "p_ge_002": None, "ess": 0.0, "accepted": int(len(phi))}
    if tau_d > 0:  # target = φ + δ_test: probabilities over the normal distribution of δ_test
        p_pos = float((wts * st.norm.sf(0.0, loc=phi, scale=tau_d)).sum() / tot)
        p_neg = float((wts * st.norm.cdf(-0.002, loc=phi, scale=tau_d)).sum() / tot)
        p_ge = float((wts * st.norm.sf(0.002, loc=phi, scale=tau_d)).sum() / tot)
    else:
        p_pos, p_neg, p_ge = (float(wts[phi > 0].sum() / tot), float(wts[phi < -0.002].sum() / tot), float(wts[phi >= 0.002].sum() / tot))
    return {"mean": float((wts * phi).sum() / tot), "p_pos": p_pos, "p_lt_m002": p_neg, "p_ge_002": p_ge,
            "ess": float(tot ** 2 / (wts ** 2).sum()), "accepted": int(len(phi)),
            "settings": {"k": k, "rho": rho, "tau_d": tau_d, "spread": sp, "tol": tol, "n_sim": n_sim * chunks,
                         "o1": o1, "s1": s1, "o2": o2, "s2": s2, "r12": r12}}
