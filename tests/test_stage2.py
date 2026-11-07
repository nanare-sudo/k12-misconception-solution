"""Tests of the second stage: src/stage2.py, eda/c118_stage2.py, the switch --arith-beta of eda/c110_q397_deploy.py,
tools/make_submission.py --stage2 and tools/builds/s032_stage2.sh.

Synthetic (fast, without data): feature list, mention counting, knowledge only from the other folds (null point d), copy null
point of the order (weight 0 -> bit-identical), transfer of a new order into the probabilities, label placebo -> abstention
(null point c), guard of the notes, check of the test notes in make_submission. With real training artifacts (skipped if they
are missing): blend file with β (cross-check β 0.2 character by character), deployment file = recomputation including null
point (e), knowledge of the real training rows (null point d), label placebo on the deployment features and in the OOF
recomputation (c). With test channels (skipped if they are missing): copy null points (a) and (b) via make_submission —
checksums only, no test row is printed.

Widened rule of the note guard: without the new parameters exactly the old guard (compared with a verbatim copy of the band
rule); the recorded case d1–d2 passes via z; positive controls fire; the rule only widens; word-identical notes against the
pair's own training value; make_submission.apply_stage2 with and without the new guard_rule fields (synthetic); with real files
(skipped if they are missing): stage2_s034.json = stage2_s032.json except guard_rule/derived_from, c118 fit --guard-rule widened
--verify, the guard on the recorded test aggregates (aggregates only), checksums in the build script s034_stage2_main.sh."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMP / "src")]
import stage2 as S2  # noqa: E402

PY = [sys.executable, "-I"]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
SIX_CHANNELS = ["v1r", "zs27q", "think18nb", "pr27xm", "m3x3_full", "q397m3"]
REF_BASE_SHA = "768b7778ea049caeb23455ecad26b9bc90ebff52bee3cb97e356aac7cadf07e8"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classes_toy() -> list[str]:
    return ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "Dec-01", "PV-01", "ALG-06", "FRA-02"]


# ------------------------------------------------------------------------------------------------ synthetic

def test_feature_names_main_and_fallback():
    main, fb = S2.feature_names(SIX_CHANNELS, "main"), S2.feature_names(SIX_CHANNELS, "fallback")
    assert len(main) == 43 and len(fb) == 40
    assert [f for f in main if f not in fb] == list(S2.NOTE_FEATS)
    assert not any(f.startswith(("q397", "q_")) for f in main) and "row_q_valid" not in main
    assert main[:6] == ["f_logp", "f_rank", "f_gap", "f_p", "p_logp", "p_rank"]
    with pytest.raises(ValueError):
        S2.feature_names(SIX_CHANNELS, "everything")


def test_mention_features_word_boundaries_case_and_last():
    cls = classes_toy()
    texts = ["CAL-01 then Dec-01; XCAL-01 and CAL-011 no; DEC-01 no; NONE no", "", "PV-01 ... ANS-01 ... PV-01"]
    men, last, ln = S2.mention_features(texts, cls)
    idx = {c: i for i, c in enumerate(cls)}
    assert men[0, idx["CAL-01"]] == 1 and men[0, idx["Dec-01"]] == 1 and men[0].sum() == 2
    assert last[0, idx["Dec-01"]] == 1 and last[0].sum() == 1
    assert men[1].sum() == 0 and last[1].sum() == 0 and ln[1] == 0
    assert men[2, idx["PV-01"]] == 1 and men[2, idx["ANS-01"]] == 1 and last[2, idx["PV-01"]] == 1
    nf = S2.notes_features([texts, texts, ["", "", ""]], cls)
    assert nf["m3_men"][0, idx["CAL-01"]] == 2 and nf["m3_len"][1] == 0


def _toy_keys(n=60, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 6, n).astype(str)
    t = np.where(rng.random(n) < 0.5, "A", "B")
    y = rng.integers(0, 10, n)
    folds = np.arange(n) % 5
    return S2.task_keys(q, t), t, y, folds


def test_knowledge_other_folds_ignores_own_row_and_fold():
    """Null point (d): the knowledge of the training rows of a fold does not change when labels of the row itself or of its own
    fold are corrupted; positive control: corrupting another fold changes it."""
    key, topic, y, folds = _toy_keys()
    for k in range(5):
        rows = np.flatnonzero(folds == k)
        base = S2.knowledge_other_folds(key, topic, y, folds, rows, 10)
        y2 = y.copy()
        y2[rows] = (y2[rows] + 3) % 10
        own = S2.knowledge_other_folds(key, topic, y2, folds, rows, 10)
        for name in ("task_count", "task_rows", "topic_count", "topic_rows", "glob"):
            assert np.array_equal(base[name], own[name]), (k, name)
        y3 = y.copy()
        other = np.flatnonzero(folds == (k + 1) % 5)
        y3[other] = (y3[other] + 3) % 10
        assert not np.array_equal(base["glob"], S2.knowledge_other_folds(key, topic, y3, folds, rows, 10)["glob"])
    with pytest.raises(ValueError):
        S2.knowledge_other_folds(key, topic, y, folds, np.arange(10), 10)


def _toy_static(n=40, K=10, seed=1):
    rng = np.random.default_rng(seed)
    cls = classes_toy()
    seen = np.ones(K, bool)
    seen[9] = False

    def rnd():
        P = rng.dirichlet(np.ones(K), size=n)
        return P

    chans = {nm: rnd() for nm in SIX_CHANNELS}
    draws = [rnd(), rnd(), rnd()]
    notes = S2.notes_features([["CAL-01 ANS-01"] * n, ["WOP-01"] * n, [""] * n], cls)
    st = S2.Static(SIX_CHANNELS, chans, draws, rng.integers(4, 13, n).astype(float), seen, cls, notes)
    return st, seen, rng


def test_pair_features_shapes_and_zero_model_is_bitwise_identity():
    """Copy null point: weight 0 -> base order -> probabilities bit-identical."""
    st, seen, rng = _toy_static()
    n, K = st.n, st.K
    P = rng.dirichlet(np.ones(K), size=n)
    Q = np.where(seen[None, :], P, 0.0)
    Q = Q / Q.sum(1, keepdims=True)
    key, topic, y, folds = _toy_keys(n)
    kn = S2.knowledge(key, topic, y, key, topic, K)
    for fs in S2.FEATURE_SETS:
        X, cand, names = S2.pair_features(st, np.arange(n), Q, P, kn, fs)
        assert X.shape == (n * S2.K_CAND, len(names)) and names == S2.feature_names(SIX_CHANNELS, fs)
        assert not np.isin(cand, np.flatnonzero(~seen)).any()
        top = S2.reorder(cand, S2.scores(S2.zero_model(len(names)), X))
        assert np.array_equal(top, cand)
        Q2, fixed = S2.apply_to_probs(Q, cand, top, seen)
        assert fixed == 0 and Q2.tobytes() == Q.tobytes()


def test_apply_to_probs_transfers_new_order_and_fixes_ties():
    seen = np.ones(8, bool)
    Q = np.array([[0.4, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01], [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02]])
    cand = S2.candidates(Q, seen)
    new = cand[:, [2, 0, 1, 3, 4]]
    Q2, fixed = S2.apply_to_probs(Q, cand, new, seen)
    assert np.allclose(Q2.sum(1), 1.0)
    got = np.argsort(-Q2, axis=1, kind="stable")[:, :5]
    assert np.array_equal(got, new) and fixed == 1  # row 2 has ties
    assert np.allclose(Q2[:, 5:], Q[:, 5:])


def test_label_placebo_abstains_and_signal_applies():
    """Null point (c): with permuted targets the gate abstains; with a real signal it applies (positive control)."""
    rng = np.random.default_rng(7)
    R, Kc, f = 1500, S2.K_CAND, 4
    X = rng.normal(size=(R * Kc, f))
    rank = np.tile(np.arange(Kc), R).astype(float)
    X[:, 0] = -rank + rng.normal(scale=0.3, size=R * Kc)  # the base order carries
    gold = np.minimum(rng.geometric(0.55, size=R) - 1, Kc - 1)
    signal = np.zeros(R * Kc)
    signal[np.arange(R) * Kc + gold] = 1.0
    X[:, 1] = signal + rng.normal(scale=0.6, size=R * Kc)  # additional signal
    cand = np.tile(np.arange(Kc), (R, 1))
    y = gold.copy()
    lab = S2.labels_of(cand, y)
    folds = np.arange(R) % 5
    blocks = [(X[np.repeat(folds == k, Kc)], lab[np.repeat(folds == k, Kc)], cand[folds == k], y[folds == k]) for k in range(5)]
    assert not S2.gate(blocks)["abstain"]
    labp = S2.permute_labels(lab, Kc, np.random.default_rng(1000))
    blocks_p = [(b[0], labp[np.repeat(folds == k, Kc)], b[2], b[3]) for k, b in enumerate(blocks)]
    assert S2.gate(blocks_p)["abstain"]


def test_guard_notes_band():
    ref = {"per_draw": {d: {"codes_per_row": v, "share_top_mentioned": 0.6} for d, v in zip(S2.DRAWS, (2.0, 3.0, 3.5))},
           "mention_hist_top5": [0.5, 0.1, 0.1, 0.3], "identical": {"d1-d2": 0.04, "d1-d3": 0.0, "d2-d3": 0.0}}
    ok = json.loads(json.dumps(ref))
    assert S2.guard_notes(ref, ok)["ok"]
    bad = json.loads(json.dumps(ref))
    bad["per_draw"]["d1"]["codes_per_row"] = 1.7  # below 2.0 · 0.9
    g = S2.guard_notes(ref, bad)
    assert not g["ok"] and [c["stat"] for c in g["checks"] if not c["ok"]] == ["codes_per_row[d1]"]
    bad2 = json.loads(json.dumps(ref))
    bad2["identical"]["d1-d3"] = 1.0  # copies
    assert not S2.guard_notes(ref, bad2)["ok"]
    assert S2.band([0.0, 0.04]) == (0.0, pytest.approx(0.044))


# ------------------------------------------------------------------------------------------------ widened rule of the note guard

def _old_guard_notes(ref: dict, test: dict, rel: float = 0.10) -> dict:
    """Verbatim copy of guard (a) under the original band rule — reference for "without the new parameters exactly as before"."""
    checks = []
    for key in ("codes_per_row", "share_top_mentioned"):
        lo, hi = S2.band([ref["per_draw"][d][key] for d in S2.DRAWS], rel)
        for d in S2.DRAWS:
            v = test["per_draw"][d][key]
            checks.append({"stat": f"{key}[{d}]", "test": v, "lo": lo, "hi": hi, "ok": bool(lo <= v <= hi)})
    for b, (r, v) in enumerate(zip(ref["mention_hist_top5"], test["mention_hist_top5"])):
        lo, hi = S2.band([r], rel)
        checks.append({"stat": f"mention_hist_top5[{b}]", "test": v, "lo": lo, "hi": hi, "ok": bool(lo <= v <= hi)})
    lo, hi = S2.band(list(ref["identical"].values()), rel)
    for pair, v in test["identical"].items():
        checks.append({"stat": f"identical[{pair}]", "test": v, "lo": lo, "hi": hi, "ok": bool(lo <= v <= hi)})
    return {"checks": checks, "ok": bool(all(c["ok"] for c in checks)), "rel": rel}


# recorded aggregates (rounded): reference = three training draws, test = test draws (shares/means only)
GUARD_REF = {"per_draw": {"d1": {"codes_per_row": 2.757, "share_top_mentioned": 0.589}, "d2": {"codes_per_row": 3.584, "share_top_mentioned": 0.619},
                         "d3": {"codes_per_row": 3.562, "share_top_mentioned": 0.622}},
            "mention_hist_top5": [0.504, 0.127, 0.115, 0.254], "identical": {"d1-d2": 0.036, "d1-d3": 0.0, "d2-d3": 0.0007}}
GUARD_TEST = {"per_draw": {"d1": {"codes_per_row": 2.630, "share_top_mentioned": 0.571}, "d2": {"codes_per_row": 3.411, "share_top_mentioned": 0.617},
                          "d3": {"codes_per_row": 3.472, "share_top_mentioned": 0.618}},
             "mention_hist_top5": [0.509, 0.121, 0.115, 0.254], "identical": {"d1-d2": 0.046, "d1-d3": 0.0003, "d2-d3": 0.0003}}
WIDENED = {"share_z_max": 3.0, "n_ref": 1500, "n_test": 3498}


def _rnd_stats(rng) -> dict:
    return {"per_draw": {d: {"codes_per_row": float(rng.uniform(1, 4)), "share_top_mentioned": float(rng.uniform(0.3, 0.8))} for d in S2.DRAWS},
            "mention_hist_top5": [float(x) for x in rng.dirichlet(np.ones(4))],
            "identical": {"d1-d2": float(rng.uniform(0, 0.1)), "d1-d3": float(rng.choice([0.0, rng.uniform(0, 0.01)])),
                          "d2-d3": float(rng.uniform(0, 0.01))}}


def _check(g: dict, stat: str) -> dict:
    return next(c for c in g["checks"] if c["stat"] == stat)


def test_guard_notes_without_widened_params_is_exactly_the_old_rule():
    """Null point: without share_z_max/n_ref/n_test the same return value as the band-rule version (character for character as JSON)."""
    rng = np.random.default_rng(11)
    cases = [(GUARD_REF, GUARD_TEST)] + [(_rnd_stats(rng), _rnd_stats(rng)) for _ in range(300)]
    for ref, test in cases:
        for rel in (0.10, 0.05):
            assert json.dumps(S2.guard_notes(ref, test, rel)) == json.dumps(_old_guard_notes(ref, test, rel))
    g = S2.guard_notes(GUARD_REF, GUARD_TEST)
    assert not g["ok"] and [c["stat"] for c in g["checks"] if not c["ok"]] == ["identical[d1-d2]"]
    assert set(g) == {"checks", "ok", "rel"} and all(set(c) == {"stat", "test", "lo", "hi", "ok"} for c in g["checks"])
    with pytest.raises(ValueError):
        S2.guard_notes(GUARD_REF, GUARD_TEST, n_ref=1500, n_test=3498)  # sample sizes only together with share_z_max
    with pytest.raises(ValueError):
        S2.guard_notes(GUARD_REF, GUARD_TEST, share_z_max=3.0)  # widened rule without sample sizes


def test_share_z_two_sample_binomial():
    p_ref, n_ref, p_test, n_test = 0.036, 1500, 0.046, 3498
    p = (p_ref * n_ref + p_test * n_test) / (n_ref + n_test)
    want = (p_test - p_ref) / np.sqrt(p * (1 - p) * (1 / n_ref + 1 / n_test))
    assert S2.share_z(p_ref, n_ref, p_test, n_test) == pytest.approx(want, rel=1e-12)
    assert S2.share_z(p_ref, n_ref, p_test, n_test) == pytest.approx(1.597, abs=0.002)
    assert S2.share_z(p_test, n_test, p_ref, n_ref) == pytest.approx(-want, rel=1e-12)
    assert S2.share_z(0.0, 1500, 0.0, 3498) == 0.0 and S2.share_z(1.0, 1500, 1.0, 3498) == 0.0  # SE 0, difference 0
    assert S2.share_z(0.5, 2, -0.5, 2) == -np.inf  # SE 0 with a difference ≠ 0 (invalid input) -> infinite, fires
    assert np.isnan(S2.share_z(float("nan"), 1500, 0.1, 3498))


def test_widened_rule_recorded_case_d1_d2_passes_via_z():
    """The recorded case: word-identical d1–d2, test 0.046 against training 0.036, lies outside the old band (0–0.0396), z ≈ 1.6 -> passes."""
    g = S2.guard_notes(GUARD_REF, GUARD_TEST, **WIDENED)
    assert g["ok"] and g["rule"] == "widened" and (g["share_z_max"], g["n_ref"], g["n_test"]) == (3.0, 1500, 3498)
    c = _check(g, "identical[d1-d2]")
    assert not c["ok_band"] and c["ok_z"] and c["passed_by"] == "z" and c["ref"] == 0.036 and c["z"] == pytest.approx(1.597, abs=0.002)
    assert [x["stat"] for x in g["checks"] if x["passed_by"] == "z"] == ["identical[d1-d2]"]
    assert all(x["passed_by"] == "band" for x in g["checks"] if x["stat"] != "identical[d1-d2]")
    for x in g["checks"]:  # codes_per_row only within the old band, all other shares with z against their own training value
        if x["stat"].startswith("codes_per_row"):
            assert x["rule"] == "band" and x["z"] is None and x["ref"] is None
        else:
            assert x["rule"] == "band|z" and x["z"] is not None and abs(x["z"]) <= 3.0
    assert _check(g, "share_top_mentioned[d2]")["ref"] == 0.619 and _check(g, "mention_hist_top5[3]")["ref"] == 0.254


def _plant(test: dict, stat: str, value: float) -> dict:
    t = copy.deepcopy(test)
    key, arg = stat[:-1].split("[")
    if key in ("codes_per_row", "share_top_mentioned"):
        t["per_draw"][arg][key] = value
    elif key == "mention_hist_top5":
        t["mention_hist_top5"][int(arg)] = value
    else:
        t["identical"][arg] = value
    return t


def _value_for_z(p_ref: float, target: float, n_ref: int = 1500, n_test: int = 3498) -> float:
    lo, hi = (p_ref, 1.0) if target > 0 else (0.0, p_ref)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (S2.share_z(p_ref, n_ref, mid, n_test) - target) * (S2.share_z(p_ref, n_ref, lo, n_test) - target) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_widened_rule_positive_controls_still_alarm():
    """Positive controls: copied/swapped notes (word-identical 1.0), codes_per_row 1.7, top-code mention 0.40 and a planted |z| 3.5
    outside the old band fire; a planted |z| 2.5 outside the old band passes."""
    for stat, v in (("identical[d1-d2]", 1.0), ("identical[d1-d3]", 1.0), ("codes_per_row[d1]", 1.7), ("share_top_mentioned[d1]", 0.40)):
        g = S2.guard_notes(GUARD_REF, _plant(GUARD_TEST, stat, v), **WIDENED)
        assert not g["ok"] and [c["stat"] for c in g["checks"] if not c["ok"]] == [stat], stat
    assert _check(S2.guard_notes(GUARD_REF, _plant(GUARD_TEST, "codes_per_row[d1]", 1.7), **WIDENED), "codes_per_row[d1]")["z"] is None
    for stat, ref in (("identical[d1-d2]", 0.036), ("mention_hist_top5[1]", 0.127), ("mention_hist_top5[3]", 0.254)):
        for target, want in ((3.5, False), (2.5, True)):
            v = _value_for_z(ref, target)
            g = S2.guard_notes(GUARD_REF, _plant(GUARD_TEST, stat, v), **WIDENED)
            c = _check(g, stat)
            assert c["z"] == pytest.approx(target, abs=1e-6) and not c["ok_band"], (stat, target)
            assert c["ok"] is want and g["ok"] is want and c["passed_by"] == ("z" if want else None), (stat, target)
    # inside the old band even |z| 3.5 passes (the rule only widens): word-identical d1–d3 with training 0
    c = _check(S2.guard_notes(GUARD_REF, _plant(GUARD_TEST, "identical[d1-d3]", _value_for_z(0.0, 3.5)), **WIDENED), "identical[d1-d3]")
    assert c["ok"] and c["ok_band"] and not c["ok_z"] and c["passed_by"] == "band"


def test_widened_rule_only_widens_never_narrows():
    rng = np.random.default_rng(5)
    widened = 0
    for i in range(500):
        ref = _rnd_stats(rng)
        test = _rnd_stats(rng) if i % 2 else _plant(ref, "identical[d1-d2]", ref["identical"]["d1-d2"] * float(rng.uniform(0.5, 1.6)))
        old, new = S2.guard_notes(ref, test), S2.guard_notes(ref, test, **WIDENED)
        for a, b in zip(old["checks"], new["checks"]):
            assert (a["stat"], a["test"], a["lo"], a["hi"]) == (b["stat"], b["test"], b["lo"], b["hi"]) and b["ok_band"] == a["ok"]
            assert b["ok"] or not a["ok"]  # what passed before still passes
            widened += int(b["ok"] and not a["ok"])
        assert new["ok"] or not old["ok"]
    assert widened > 0


def test_widened_rule_identical_against_own_pair_reference():
    """Word-identical notes per pair against the pair's OWN training value: d1–d3 = 0.045 (training 0) fires, although the training
    value of d1–d2 (0.036) would let it pass; d1–d2 = 0.046 passes."""
    test = _plant(_plant(GUARD_TEST, "identical[d1-d3]", 0.045), "identical[d1-d2]", 0.046)
    g = S2.guard_notes(GUARD_REF, test, **WIDENED)
    c13, c12 = _check(g, "identical[d1-d3]"), _check(g, "identical[d1-d2]")
    assert c13["ref"] == 0.0 and not c13["ok"] and c13["z"] > 3 and S2.share_z(0.036, 1500, 0.045, 3498) < 3
    assert c12["ref"] == 0.036 and c12["ok"] and c12["passed_by"] == "z"
    assert _check(g, "identical[d2-d3]")["ref"] == 0.0007
    bad_ref = copy.deepcopy(GUARD_REF)
    del bad_ref["identical"]["d2-d3"]
    with pytest.raises(ValueError):
        S2.guard_notes(bad_ref, GUARD_TEST, **WIDENED)


def _toy_apply_stage2(tmp_path, guard_rule: dict):
    """Synthetic inputs for make_submission.apply_stage2 (weight 0): notes d1/d2 word-identical in 10 of 50 rows (0.20), reference
    0.17 -> old band 0–0.187 missed, z ≈ 0.36 at n_ref 40 / n_test 50."""
    import pandas as pd

    cls = classes_toy()
    K, n_test, n_train = len(cls), 50, 40
    rng = np.random.default_rng(3)
    seen = np.ones(K, bool)
    seen[9] = False

    def rnd():
        P = rng.dirichlet(np.ones(K), size=n_test) * seen[None, :]
        return P / P.sum(1, keepdims=True)

    raw, draws, Q, P_pre = {nm: rnd() for nm in SIX_CHANNELS}, [rnd(), rnd(), rnd()], rnd(), rnd()
    stack = {"channels": list(SIX_CHANNELS), "arith_beta": 0.6}
    stack_file = tmp_path / "stack_toy.json"
    stack_file.write_text(json.dumps(stack))
    train = pd.DataFrame({"root_code": [cls[i % 9] for i in range(n_train)], "question_id": [f"q{i % 7}" for i in range(n_train)],
                          "topic_name": ["A" if i % 2 else "B" for i in range(n_train)]})
    test = pd.DataFrame({"question_id": [f"q{i % 7}" for i in range(n_test)], "topic_name": ["A" if i % 3 else "B" for i in range(n_test)],
                         "grade": [7] * n_test})
    d1 = [f"CAL-01 then ANS-01 row {i}" for i in range(n_test)]
    d2 = [d1[i] if i < 10 else f"WOP-01 other {i}" for i in range(n_test)]
    d3 = [f"FRM-01 third {i}" for i in range(n_test)]
    texts = {"n1.jsonl": d1, "n2.jsonl": d2, "n3.jsonl": d3}
    notes = S2.notes_features([d1, d2, d3], cls)
    cand0 = S2.candidates(Q, seen)
    ref = S2.note_stats(notes["men_per_draw"], notes["m3_men"], cand0, cand0[:, 0])
    ref["identical"] = {"d1-d2": 0.17, "d1-d3": 0.0, "d2-d3": 0.0}
    models = {fs: {"features": S2.feature_names(SIX_CHANNELS, fs), "w": [0.0] * len(S2.feature_names(SIX_CHANNELS, fs)),
                   "mu": [0.0] * len(S2.feature_names(SIX_CHANNELS, fs)), "sd": [1.0] * len(S2.feature_names(SIX_CHANNELS, fs)),
                   "oof_changed_top_share": 0.0, "oof_changed_top3_share": 0.0} for fs in S2.FEATURE_SETS}
    s2 = {"stack_sha256": sha(stack_file), "channels": list(SIX_CHANNELS), "arith_beta": 0.6, "K": S2.K_CAND, "lambda": S2.LAMBDA,
          "null": False, "draws_channel": "m3x3_full", "test_notes": [{"file": f"n{d}.jsonl", "rank": f"runs/r{d}/job/test"} for d in (1, 2, 3)],
          "models": models, "guard_reference": ref, "novel_task_share_top5": 0.5, "guard_rule": guard_rule}
    s2_file = tmp_path / "stage2_toy.json"
    s2_file.write_text(json.dumps(s2))
    args = (s2_file, stack_file, stack, Q, P_pre, raw, {"m3x3_full": draws}, {"m3x3_full": ",".join(c["rank"] for c in s2["test_notes"])},
            train, test, cls, seen, tmp_path)
    return args, texts


def test_make_submission_apply_stage2_widened_rule_fields(tmp_path, monkeypatch):
    """make_submission reads share_z_max/n_ref from guard_rule: without the fields the old guard (fallback), with them the widened
    rule (main form); only one field or n_ref ≠ number of training rows -> abort. No command-line switch."""
    MS = _load("make_submission_stage2_widened", "tools/make_submission.py")
    old_rule = {"notes_rel_tolerance": 0.10, "changed_top_factor": [0.5, 2.0], "text": "band rule"}
    for rule, want_fs in ((old_rule, "fallback"), ({**old_rule, "rule": "widened", "share_z_max": 3.0, "n_ref": 40}, "main")):
        args, texts = _toy_apply_stage2(tmp_path, rule)
        monkeypatch.setattr(MS, "_test_notes_texts", lambda root, f, r, n: (texts[f], {"file": f, "sha256": "x", "rank": r, "rank_sha256": "y"}))
        rep = tmp_path / f"guards_{want_fs}.json"
        Q_new, info = MS.apply_stage2(*args, report_path=rep)
        gn = info["guard_notes"]
        assert info["feature_set"] == want_fs and json.loads(rep.read_text())["feature_set"] == want_fs
        c = _check(gn, "identical[d1-d2]")
        assert c["test"] == pytest.approx(0.2) and c["hi"] == pytest.approx(0.187)
        if want_fs == "fallback":
            assert "rule" not in gn and set(gn) == {"ok", "rel", "test", "reference", "checks"} and "z" not in c and not c["ok"]
        else:
            assert gn["rule"] == "widened" and (gn["share_z_max"], gn["n_ref"], gn["n_test"]) == (3.0, 40, 50)
            assert c["ok"] and c["passed_by"] == "z" and c["z"] == pytest.approx(0.363, abs=0.002)
        assert np.array_equal(Q_new, args[3])  # weight 0 -> order unchanged
    for rule, msg in (({**old_rule, "share_z_max": 3.0}, "only together"), ({**old_rule, "n_ref": 40}, "only together"),
                      ({**old_rule, "share_z_max": 3.0, "n_ref": 1500}, "number of training rows"),
                      ({**old_rule, "share_z_max": float("inf"), "n_ref": 40}, "differs from the widened rule"),
                      ({**old_rule, "share_z_max": 4.0, "n_ref": 40}, "differs from the widened rule"),
                      ({**old_rule, "notes_rel_tolerance": 0.5, "share_z_max": 3.0, "n_ref": 40}, "differs from the widened rule")):
        args, texts = _toy_apply_stage2(tmp_path, rule)
        monkeypatch.setattr(MS, "_test_notes_texts", lambda root, f, r, n: (texts[f], {"file": f, "sha256": "x", "rank": r, "rank_sha256": "y"}))
        with pytest.raises(SystemExit, match=msg):
            MS.apply_stage2(*args)
    opts = re.findall(r'ap\.add_argument\("(--[^"]+)"', (COMP / "tools" / "make_submission.py").read_text())
    assert "--stage2" in opts and not any(k in o for o in opts for k in ("guard", "share", "override", "widen"))  # no override


def test_make_submission_test_notes_guard(tmp_path, monkeypatch):
    MS = _load("make_submission_stage2_test", "tools/make_submission.py")
    root = tmp_path / "data"
    (root / "derived" / "sft").mkdir(parents=True)
    run_dir = tmp_path / "artifacts" / "runs" / "r1" / "r1" / "rank_job"
    run_dir.mkdir(parents=True)
    rank_file = run_dir / "test_proto_rank.jsonl"
    rank_file.write_text("{}\n")
    body = "".join(json.dumps({"table": "test", "row": i, "text": f"CAL-01 {i}", "finding": None, "state": "complete"}) + "\n" for i in range(3))
    f = root / "derived" / "sft" / "n.jsonl"
    f.write_text(body)
    meta = {"table": "test", "rows": 3, "source_rank": "runs/r1/r1/rank_job/test", "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "rank_sha256": sha(rank_file)}
    f.with_suffix(".meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(MS, "COMP", tmp_path)
    texts, src = MS._test_notes_texts(root, "derived/sft/n.jsonl", "runs/r1/rank_job/test", 3)
    assert texts == ["CAL-01 0", "CAL-01 1", "CAL-01 2"] and src["sha256"] == meta["sha256"]
    with pytest.raises(SystemExit, match="test ranking"):
        MS._test_notes_texts(root, "derived/sft/n.jsonl", "runs/r2/rank_job/test", 3)
    with pytest.raises(SystemExit, match="rows"):
        MS._test_notes_texts(root, "derived/sft/n.jsonl", "runs/r1/rank_job/test", 4)
    rank_file.write_text("{} \n")
    with pytest.raises(SystemExit, match="checksum"):
        MS._test_notes_texts(root, "derived/sft/n.jsonl", "runs/r1/rank_job/test", 3)


def test_c110_beta_switch_only_with_replace():
    D110 = _load("c110_stage2_test", "eda/c110_q397_deploy.py")
    for extra in (["--arith-beta", "0.6"], ["--arith-beta-select"], ["--stem", "x"]):
        with pytest.raises(SystemExit) as e:
            D110.main(["stack", "--line", "best", *extra, "--verify"])
        assert e.value.code == 2
    assert D110.BETA_GRID == S2.BETA_GRID


# ------------------------------------------------------------------------------------------------ real training artifacts

def _have(*rels) -> bool:
    return all((COMP / r).exists() for r in rels)


TRAIN_ART = ("artifacts/stack_s029_best_q397m3.json", "artifacts/stack_s032_stage2.json", "artifacts/stage2_s032.json", "artifacts/oof_q397m3.npy",
             "artifacts/oof_m3x3_full.npy", "artifacts/eda/private/stage2/c118_oof.json", "data/train.csv")
needs_train = pytest.mark.skipif(not _have(*TRAIN_ART), reason="training artifacts of the second stage missing")


@needs_train
def test_stack_beta_file_fields():
    s0 = json.loads((COMP / "artifacts/stack_s029_best_q397m3.json").read_text())
    s = json.loads((COMP / "artifacts/stack_s032_stage2.json").read_text())
    rule = s["arith_beta_rule"]
    assert s["arith_beta"] == rule["chosen"] and rule["selected"] and rule["grid"] == list(S2.BETA_GRID)
    lls = rule["logloss_pre_last_step"]
    assert rule["chosen"] == float(min(lls, key=lambda k: (lls[k], float(k))))
    changed = {"arith_beta", "cv", "cv_note", "cross", "insample", "note", "arith_beta_rule"}
    assert set(s) - set(s0) == {"arith_beta_rule"} and set(s0) <= set(s)
    assert {k for k in s0 if s0[k] != s[k]} <= changed
    assert rule["stack_beta_0_2_sha256"] == sha(COMP / "artifacts/stack_s029_best_q397m3.json")


@needs_train
def test_c110_beta_0_2_reproduces_s032_stack_char_for_char():
    """β switch with β 0.2 = exactly the blend file without the second stage (character for character; --verify compares the text)."""
    r = subprocess.run(PY + ["eda/c110_q397_deploy.py", "stack", "--line", "best", "--m3", "m3x3_full", "--replace-q397", "q397m3",
                             "--arith-beta", "0.2", "--verify"], cwd=COMP, env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERIFIED: artifacts/stack_s029_best_q397m3.json" in r.stdout


@needs_train
def test_stage2_file_fields():
    s2 = json.loads((COMP / "artifacts/stage2_s032.json").read_text())
    assert s2["K"] == S2.K_CAND == 5 and s2["lambda"] == S2.LAMBDA == 0.1 and s2["null"] is False
    assert s2["stack_sha256"] == sha(COMP / "artifacts/stack_s032_stage2.json")
    assert s2["channels"] == SIX_CHANNELS and s2["align"] == 0.75 and s2["align_min"] == 3
    for fs in S2.FEATURE_SETS:
        m = s2["models"][fs]
        assert m["features"] == S2.feature_names(SIX_CHANNELS, fs) and len(m["w"]) == len(m["features"])
        assert 0.0 < m["oof_changed_top_share"] < 0.2
    assert s2["oof_reproduction"]["reproduction"]["ok"]


@needs_train
def test_c118_fit_verify_includes_nullpoint_e():
    """Deployment file = recomputation; inside it null point (e): training features of the build path = features of the OOF
    recomputation, bit-identical."""
    r = subprocess.run(PY + ["eda/c118_stage2.py", "fit", "--stack", "s032_stage2", "--verify"], cwd=COMP, env=ENV, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    oof = json.loads((COMP / "artifacts/eda/private/stage2/c118_oof.json").read_text())
    s2 = json.loads((COMP / "artifacts/stage2_s032.json").read_text())
    for fs in S2.FEATURE_SETS:
        assert s2["models"][fs]["features_sha256"] == oof["sets"][fs]["set"]["eval_features_sha256"]


@needs_train
def test_oof_reproduction_and_label_placebo():
    """Reproduction (target values of the independent cross-check ±0.0005, prototype bit-identical if available) and label placebo:
    abstention 5/5, Δ exactly 0."""
    oof = json.loads((COMP / "artifacts/eda/private/stage2/c118_oof.json").read_text())
    assert oof["reproduction"]["ok"]
    for f in ("fold", "pre", "set"):
        e = oof["sets"]["main"][f]
        assert e["n_features"] == 43 and e.get("prototype_rows_equal", True)
        p = oof["sets"]["main+placebo"][f]
        assert p["abstain_folds"] == 5 and p["abstain"]["vs_recipe_beta"]["units"] == 0


@pytest.fixture(scope="module")
def ctx118():
    if not _have(*TRAIN_ART):
        pytest.skip("training artifacts missing")
    C118 = _load("c118_stage2_test", "eda/c118_stage2.py")
    return C118, C118.context("s032_stage2")


@needs_train
def test_real_knowledge_ignores_own_fold_and_deploy_placebo_abstains(ctx118):
    """Null point (d) on the real training rows; null point (c) on the deployment features (gate over five folds)."""
    C118, ctx = ctx118
    key, topic, y, folds, K = ctx["key"], ctx["topic_s"], ctx["y"], ctx["folds"], ctx["K"]
    for k in range(5):
        rows = np.flatnonzero(folds == k)
        base = S2.knowledge_other_folds(key, topic, y, folds, rows, K)
        y2 = y.copy()
        y2[rows] = 0
        y2[rows[0]] = 5
        got = S2.knowledge_other_folds(key, topic, y2, folds, rows, K)
        assert all(np.array_equal(base[nm], got[nm]) for nm in ("task_count", "topic_count", "glob"))
    forms = C118.deploy_forms(ctx)
    X, cand, _ = C118.deploy_features(ctx, forms, "main")
    lab = S2.labels_of(cand, y)
    assert not S2.gate(C118.fold_blocks(ctx, X, cand, lab))["abstain"]
    labp = S2.permute_labels(lab, S2.K_CAND, np.random.default_rng(C118.PLACEBO_SEED))
    assert S2.gate(C118.fold_blocks(ctx, X, cand, labp))["abstain"]


# ------------------------------------------------------------------------------------------------ test channels: copy null points (a), (b)

TEST_RUNS = ("artifacts/runs/q397m3te_rp", "artifacts/runs/m3_fullte_rp", "artifacts/runs/m3d2te_rp", "artifacts/runs/m3d3te_rp",
             "artifacts/runs/pr27te_rp", "artifacts/runs/n1_rp", "data/derived/sft/m3_d3_test_frozen_notes.jsonl", "data/test.csv")
needs_test = pytest.mark.skipif(not (_have(*TRAIN_ART) and _have(*TEST_RUNS)), reason="test channels missing")
SRC = ["--test", "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy", "--fill", "v1r=v1",
       "--test", "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test",
       "--test", "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test", "--fill", "think18nb=zs27q",
       "--test", "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy",
       "--proto", "pr27xm=runs/pr27te_rp/proto_te/proto", "--rank", "pr27xm=runs/n1_rp/rank_test/test",
       "--rank-only", "m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test",
       "--rank-only", "q397m3=runs/q397m3te_rp/q397m3_test_rank/test", "--rank-none-band-for", "q397m3=0.27,0.35",
       "--align", "0.75", "--align-min", "3", "--blank-rows", "derived/blank/test_rows.npy"]


def _ms(name: str, out: Path, suffix: str, *extra: str) -> None:
    r = subprocess.run(PY + ["tools/make_submission.py", "--name", name, *SRC, "--out", str(out), "--suffix", suffix, *extra], cwd=COMP, env=ENV,
                       capture_output=True, text=True, timeout=900)
    try:
        assert r.returncode == 0, r.stderr[-1500:]
    finally:
        (COMP / "artifacts" / f"test_{name}{suffix}.npy").unlink(missing_ok=True)


@needs_test
def test_nullpoint_a_zero_weight_beta_0_2_reproduces_s032_file(tmp_path):
    """Null point (a): null model + blend file without the second stage (β 0.2) -> sub_s029_best_q397m3.csv byte-identical (only the
    checksum is compared)."""
    null = tmp_path / "null_a.json"
    r = subprocess.run(PY + ["eda/c118_stage2.py", "null", "--stack", "s029_best_q397m3", "--out", str(null)], cwd=COMP, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _ms("s029_best_q397m3", tmp_path / "a.csv", "_pytest_a", "--stage2", str(null))
    assert sha(tmp_path / "a.csv") == REF_BASE_SHA


@needs_test
def test_nullpoint_b_zero_weight_new_beta_equals_file_without_stage2(tmp_path):
    """Null point (b): null model + new blend (β of the rule) -> byte-identical to the file without the second stage."""
    null = tmp_path / "null_b.json"
    r = subprocess.run(PY + ["eda/c118_stage2.py", "null", "--stack", "s032_stage2", "--out", str(null)], cwd=COMP, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _ms("s032_stage2", tmp_path / "base.csv", "_pytest_base")
    _ms("s032_stage2", tmp_path / "null.csv", "_pytest_nullb", "--stage2", str(null))
    assert sha(tmp_path / "base.csv") == sha(tmp_path / "null.csv")


def test_build_script_fixed_checksums_match_files():
    text = (COMP / "tools" / "builds" / "s032_stage2.sh").read_text()
    pairs = {"STACK_SHA": "artifacts/stack_s032_stage2.json", "STACK_BASE_SHA": "artifacts/stack_s029_best_q397m3.json"}
    for var, rel in pairs.items():
        want = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        if not (COMP / rel).is_file():
            pytest.skip(f"{rel} missing")
        assert want and want.group(1) == sha(COMP / rel), var
    s2 = COMP / "artifacts" / "stage2_s032.json"
    if s2.is_file():
        assert f'STAGE2_SHA:-{sha(s2)}' in text


# ------------------------------------------------------------------------------------------------ widened rule: real files

MAIN_FORM_FILE = COMP / "artifacts" / "stage2_s034.json"
needs_main_form = pytest.mark.skipif(not (_have(*TRAIN_ART) and MAIN_FORM_FILE.is_file()), reason="stage2_s034.json or training artifacts missing")


@needs_main_form
def test_stage2_s034_equals_s032_except_guard_rule():
    a = json.loads((COMP / "artifacts/stage2_s032.json").read_text())
    b = json.loads(MAIN_FORM_FILE.read_text())
    assert [k for k in b if k != "derived_from"] == list(a)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"guard_rule", "derived_from"}
    assert all(b["guard_rule"][k] == v for k, v in a["guard_rule"].items())
    new = {k: v for k, v in b["guard_rule"].items() if k not in a["guard_rule"]}
    assert set(new) == {"rule", "share_z_max", "n_ref", "rule_text"} and new["share_z_max"] == 3.0 and new["n_ref"] == 1500
    assert new["rule"] == "widened"
    assert b["derived_from"]["file"] == "artifacts/stage2_s032.json" and b["derived_from"]["sha256"] == sha(COMP / "artifacts/stage2_s032.json")


@needs_main_form
def test_c118_fit_widened_verify_reproduces_s034():
    """stage2_s034.json is reproducible: c118 fit --guard-rule widened recomputes everything, requires the band-rule recomputation to be
    character-identical to stage2_s032.json and then compares character for character with stage2_s034.json."""
    r = subprocess.run(PY + ["eda/c118_stage2.py", "fit", "--stack", "s032_stage2", "--guard-rule", "widened", "--verify"], cwd=COMP, env=ENV,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERIFIED: artifacts/stage2_s034.json" in r.stdout


def test_c118_fit_widened_never_writes_s032():
    before = sha(COMP / "artifacts/stage2_s032.json") if (COMP / "artifacts/stage2_s032.json").is_file() else None
    r = subprocess.run(PY + ["eda/c118_stage2.py", "fit", "--guard-rule", "widened", "--out", "artifacts/stage2_s032.json", "--verify"], cwd=COMP,
                       env=ENV, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "never writes" in r.stderr
    if before is not None:
        assert sha(COMP / "artifacts/stage2_s032.json") == before


def test_widened_rule_on_recorded_test_aggregates():
    """The guard on the recorded test aggregates (shares/means only, nothing is printed): the band rule fails only word-identical
    d1–d2, the widened rule passes, only this pair via z, all |z| ≤ 3."""
    f = COMP / "artifacts" / "eda" / "private" / "stage2" / "guards.json"
    if not f.is_file():
        pytest.skip("statistics of the second-stage guards missing")
    gn = json.loads(f.read_text())["guard_notes"]
    old = S2.guard_notes(gn["reference"], gn["test"], gn["rel"])
    assert [c["stat"] for c in old["checks"] if not c["ok"]] == ["identical[d1-d2]"]
    new = S2.guard_notes(gn["reference"], gn["test"], gn["rel"], **WIDENED)
    assert new["ok"] and [c["stat"] for c in new["checks"] if c["passed_by"] == "z"] == ["identical[d1-d2]"]
    assert max(abs(c["z"]) for c in new["checks"] if c["z"] is not None) <= 3.0


def test_build_script_s034_fixed_checksums_match_files():
    text = (COMP / "tools" / "builds" / "s034_stage2_main.sh").read_text()
    pairs = {"STACK_SHA": "artifacts/stack_s032_stage2.json", "STAGE2_32_SHA": "artifacts/stage2_s032.json",
             "FB_FILE_SHA": "artifacts/submissions/sub_s032_stage2.csv", "REF_BASE_SHA": "artifacts/submissions/sub_s029_best_q397m3.csv"}
    for var, rel in pairs.items():
        want = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        assert want, var
        if (COMP / rel).is_file():
            assert want.group(1) == sha(COMP / rel), var
    if MAIN_FORM_FILE.is_file():  # pinned, cannot be overridden through the environment
        assert f'STAGE2_SHA="{sha(MAIN_FORM_FILE)}"' in text and "STAGE2_SHA:-" not in text
    out = COMP / "artifacts" / "submissions" / "sub_s034_stage2_main.csv"
    m = re.search(r'\bOUT_SHA="([0-9a-f]{64})"', text)
    assert m and 'check "$OUT_SHA" "$OUT"' in text
    if out.is_file():
        assert m.group(1) == sha(out)
    assert "--guard-rule widened" in text and 'bash tools/builds/s032_stage2.sh best' in text
