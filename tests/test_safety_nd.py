"""Tests of the safety line with note features: feature set fallback_i1_nd (src/stage2.py), example similarity and similarity guard
(src/note_sim.py), positive controls of the guards (tools/make_submission.py). Synthetic only; none of these tests reads or writes artifacts/."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import note_sim as NS  # noqa: E402
import stage2 as S2  # noqa: E402

CH = ["a", "b"]
CLASSES = ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "ALG-01", "ALG-02"]


def _static(n=6, notes=True, seed=0):
    rng = np.random.default_rng(seed)
    K = len(CLASSES)

    def P():
        x = rng.random((n, K)) + 0.01
        return x / x.sum(1, keepdims=True)

    chans = {c: P() for c in CH}
    draws = [P() for _ in S2.DRAWS]
    nt = None
    if notes:
        texts = [[f"note {i} ALG-01 CAL-01" if i % 2 else f"note {i} WOP-01" for i in range(n)] for _ in S2.DRAWS]
        nt = S2.notes_features(texts, CLASSES)
    st = S2.Static(CH, chans, draws, np.full(n, 7.0), np.ones(K, bool), CLASSES, nt)
    FM, PM = P(), P()
    kn = S2.knowledge(np.array(["q0"] * 3), np.array(["t"] * 3), np.array([0, 1, 6]), np.array([f"q{i % 2}" for i in range(n)]),
                      np.array(["t"] * n), K)
    return st, FM, PM, kn


def test_feature_names_nd():
    names = S2.feature_names(CH, S2.ND_SET)
    assert names[:-3] == S2.feature_names(CH, S2.FB_I1_SET)
    assert tuple(names[-3:]) == S2.ND_FEATS
    assert S2.uses_notes(S2.ND_SET) and not S2.uses_notes(S2.FB_I1_SET)


def test_pair_features_nd_prefix_bit_identical_and_columns():
    st, FM, PM, kn = _static()
    n, K = FM.shape
    rows = np.arange(n)
    ds = np.random.default_rng(3).random((n, K))
    X0, c0, _ = S2.pair_features(st, rows, FM, PM, kn, S2.FB_I1_SET)
    X, c, names = S2.pair_features(st, rows, FM, PM, {**kn, "demo_sim": ds}, S2.ND_SET)
    assert np.array_equal(c, c0) and np.array_equal(X[:, :-3], X0)
    ii, cc = np.repeat(rows, S2.K_CAND), c.reshape(-1)
    assert np.array_equal(X[:, -1], ds[ii, cc])
    assert np.array_equal(X[:, -3], st.notes["m3_men"][ii, cc])
    assert np.array_equal(X[:, -2], st.notes["m3_last"][ii, cc])


def test_pair_features_nd_requires_inputs():
    st, FM, PM, kn = _static(notes=False)
    with pytest.raises(ValueError):
        S2.pair_features(st, np.arange(len(FM)), FM, PM, {**kn, "demo_sim": np.zeros(FM.shape)}, S2.ND_SET)
    st, FM, PM, kn = _static()
    with pytest.raises(ValueError):
        S2.pair_features(st, np.arange(len(FM)), FM, PM, kn, S2.ND_SET)


def test_demo_sim_top3_mean_and_exclusion():
    S = np.array([[0.9, 0.8, 0.1, 0.7, -np.inf], [-np.inf, -np.inf, 0.5, -np.inf, 0.2]])
    y = np.array([1, 1, 2, 1, 2])
    d = NS.demo_sim(S, y, 4)
    assert d[0, 1] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert d[0, 2] == pytest.approx(0.1)  # the −∞ column does not count
    assert d[1, 1] == 0.0  # no finite reference row of the class
    assert d[1, 2] == pytest.approx(0.35)
    assert d[0, 0] == 0.0 and d[0, 3] == 0.0


def test_similarity_and_mask_same_task():
    doc = ["aa bb cc dd", "aa bb cc dd", "xx yy zz ww ww xx yy zz"]
    tr = [list(doc), list(doc), list(doc)]
    S = NS.similarity(tr, tr)
    assert S.shape == (3, 3) and S[0, 1] == pytest.approx(1.0) and S[0, 2] == pytest.approx(0.0)
    M = NS.mask_same_task(S, np.array(["q1", "q1", "q2"]), np.array(["q1", "q1", "q2"]))
    assert np.isneginf(M[0, 0]) and np.isneginf(M[0, 1]) and M[0, 2] == pytest.approx(0.0)
    assert np.array_equal(NS.top_sim(M), np.array([0.0, 0.0, 0.0]))


def test_sim_guard_band_or_z():
    ref = {"mean": 0.30, "sd": 0.10, "n": 1500}
    assert NS.sim_guard(ref, {"mean": 0.32, "sd": 0.10, "n": 3498})["ok"]  # within the band ±10 %
    far = NS.sim_guard(ref, {"mean": 0.20, "sd": 0.10, "n": 3498})
    assert not far["ok"] and far["z"] < -3
    near = NS.sim_guard({"mean": 0.30, "sd": 0.50, "n": 30}, {"mean": 0.25, "sd": 0.50, "n": 30})
    assert near["ok"] and near["passed_by"] == "z"


def test_control_texts():
    spec = importlib.util.spec_from_file_location("ms_nd", COMP / "tools" / "make_submission.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    tx = [[f"d{d} row {i} " + "x" * 400 for i in range(5)] for d in range(3)]
    c = ms._control_texts(tx, "copy")
    assert all(t == tx[0] for t in c)
    p = ms._control_texts(tx, "permute")
    assert sorted(p[1]) == sorted(tx[1]) and p[1] != tx[1]
    t = ms._control_texts(tx, "truncate")
    assert all(len(x) == 300 for d in t for x in d)
    with pytest.raises(SystemExit):
        ms._control_texts(tx, "other")
