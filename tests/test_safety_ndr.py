"""Tests of the safety line with regex-exact note columns: extractor of the note columns R (src/note_regex.py), feature set fallback_i1_ndr
(src/stage2.py), test side (tools/make_submission.py: apply_stage2_safe with fallback_i1_ndr, pr27 test notes, guard R, positive controls),
pre-registered rule (eda/c130_safety_ndr.py) and the build script tools/builds/safety_ndr.sh.

Synthetic (fast, without data): definitions of the columns, ND prefix bit-identical, guard (band OR z), controls, null model bit-identical,
abort instead of a switch, logic of the rule. With real files (skipped if they are missing): result files of the reproduction, deployment
file, pins in the build script. None of these tests writes to artifacts/."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import note_regex as NR  # noqa: E402
import note_sim as NS  # noqa: E402
import stage2 as S2  # noqa: E402

PRIV = COMP / "artifacts" / "eda" / "private" / "safety_ndr"
CH = ["a", "b"]
CLASSES = ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "ALG-01", "ALG-02"]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ extractor

def test_mention_stats_definitions():
    t = "x ALG-01 y CAL-01 z ALG-01 NONE XALG-02 ALG-02a"
    st = NR.mention_stats([t, "", "nothing mentioned"], CLASSES)
    k = {c: i for i, c in enumerate(CLASSES)}
    assert st["cnt"][0, k["ALG-01"]] == 2 and st["cnt"][0, k["CAL-01"]] == 1 and st["cnt"][0].sum() == 3  # XALG-02 / ALG-02a do not count
    assert st["cnt"][0, k["NONE"]] == 0 and st["cnt"][1:].sum() == 0
    assert st["is_first"][0, k["ALG-01"]] == 1 and st["is_first"][0].sum() == 1
    assert st["is_last"][0, k["ALG-01"]] == 1 and st["is_last"][0].sum() == 1
    assert st["last_rel"][0, k["ALG-01"]] == t.rindex("ALG-01 NONE") / len(t)  # start of the last match / text length
    assert st["last_rel"][0, k["CAL-01"]] == t.index("CAL-01") / len(t) and st["last_rel"][0, k["WOP-01"]] == 0.0
    assert all(st[s].dtype == np.float64 for s in NR.STAT_KEYS)


def test_r_features_definitions_and_order():
    rng = np.random.default_rng(1)
    n, K = 7, len(CLASSES)

    def rnd_stats():
        cnt = rng.integers(0, 3, size=(n, K)).astype(float)
        return {"cnt": cnt, "is_first": (rng.random((n, K)) < 0.2).astype(float), "is_last": (rng.random((n, K)) < 0.2).astype(float),
                "last_rel": np.where(cnt > 0, rng.random((n, K)), rng.random((n, K)))}  # a position without a mention must be ignored

    m3, pr = [rnd_stats() for _ in range(3)], rnd_stats()
    R = NR.r_features(m3, pr)
    assert R.shape == (n, K, 6) and NR.R_FEATS == S2.R_FEATS == ("t_m3_logcnt", "t_m3_first", "t_m3_lastpos", "t_pr_men", "t_pr_last", "t_pr_logcnt")
    cnt = np.stack([s["cnt"] for s in m3])
    assert np.array_equal(R[:, :, 0], np.log1p(cnt).mean(0))
    assert np.array_equal(R[:, :, 1], np.stack([s["is_first"] for s in m3]).sum(0))
    assert np.array_equal(R[:, :, 2], np.where(cnt > 0, np.stack([s["last_rel"] for s in m3]), 0.0).mean(0))
    assert np.array_equal(R[:, :, 3], (pr["cnt"] > 0).astype(float)) and np.array_equal(R[:, :, 4], pr["is_last"])
    assert np.array_equal(R[:, :, 5], np.log1p(pr["cnt"]))
    with pytest.raises(ValueError):
        NR.r_features(m3[:2], pr)
    z = NR.mention_stats(["", "", "", "", "", "", ""], CLASSES)
    assert not NR.r_features([z, z, z], z).any()  # empty notes -> 0


def test_guard_band_or_z_and_detects_row_shift():
    rng = np.random.default_rng(2)
    n, K = 400, len(CLASSES)
    cand = np.stack([rng.permutation(np.arange(1, K))[:5] for _ in range(n)])
    texts = [" ".join([CLASSES[cand[i, 0]], CLASSES[cand[i, 2]]]) if i % 2 else CLASSES[cand[i, 1]] for i in range(n)]
    m3 = [NR.mention_stats(texts, CLASSES) for _ in range(3)]
    pr = NR.mention_stats(texts, CLASSES)
    ref = NR.guard_stats(m3, pr, cand)
    assert ref["pr27_share_top_mentioned"]["share"] == pytest.approx(0.5) and set(ref["m3_last_pos_cand"]) == {"d1", "d2", "d3"}
    g = NR.guard(ref, NR.guard_stats(m3, pr, cand))
    assert g["ok"] and len(g["checks"]) == 5 and all(c["passed_by"] == "band" for c in g["checks"])
    shifted = NR.mention_stats(texts[1:] + texts[:1], CLASSES)  # row shift: same mention rate, top code rarely mentioned
    gs = NR.guard(ref, NR.guard_stats(m3, shifted, cand))
    bad = {c["stat"] for c in gs["checks"] if not c["ok"]}
    assert not gs["ok"] and bad == {"pr27_share_top_mentioned"}
    # mean: outside the band, but |z| ≤ 3 with a large spread -> passes via z
    c = NR.mean_check("x", {"mean": 1.0, "sd": 5.0, "n": 30}, {"mean": 1.3, "sd": 5.0, "n": 30})
    assert c["ok"] and c["passed_by"] == "z" and not c["ok_band"]
    c = NR.mean_check("x", {"mean": 1.0, "sd": 0.01, "n": 3000}, {"mean": 1.3, "sd": 0.01, "n": 3000})
    assert not c["ok"] and c["passed_by"] is None
    s = NR.share_check("s", {"share": 0.5, "n": 1500}, {"share": 0.54, "n": 3498})
    assert s["ok"] and s["passed_by"] == "band"
    s = NR.share_check("s", {"share": 0.5, "n": 1500}, {"share": 0.56, "n": 3498})  # outside the band and |z| 3.9 > 3
    assert not s["ok"] and s["z"] > 3
    s = NR.share_check("s", {"share": 0.5, "n": 40}, {"share": 0.6, "n": 60})  # outside the band, but |z| ≤ 3 with small numbers
    assert s["ok"] and s["passed_by"] == "z"


# ------------------------------------------------------------------------------------------------ feature set fallback_i1_ndr

def _static(n=6, seed=0, with_r=True):
    rng = np.random.default_rng(seed)
    K = len(CLASSES)

    def P():
        x = rng.random((n, K)) + 0.01
        return x / x.sum(1, keepdims=True)

    chans = {c: P() for c in CH}
    draws = [P() for _ in S2.DRAWS]
    texts = [[f"note {i} ALG-01 CAL-01" if i % 2 else f"note {i} WOP-01" for i in range(n)] for _ in S2.DRAWS]
    st = S2.Static(CH, chans, draws, np.full(n, 7.0), np.ones(K, bool), CLASSES, S2.notes_features(texts, CLASSES))
    R = rng.random((n, K, 6))
    if with_r:
        S2.attach_r(st, R)
    FM, PM = P(), P()
    kn = S2.knowledge(np.array(["q0"] * 3), np.array(["t"] * 3), np.array([0, 1, 6]), np.array([f"q{i % 2}" for i in range(n)]),
                      np.array(["t"] * n), K)
    kn["demo_sim"] = rng.random((n, K))
    return st, FM, PM, kn, R


def test_feature_names_ndr_and_uses_notes():
    names = S2.feature_names(CH, S2.NDR_SET)
    assert names[:-6] == S2.feature_names(CH, S2.ND_SET) and tuple(names[-6:]) == S2.R_FEATS
    assert S2.NDR_SET == "fallback_i1_ndr" and S2.NDR_SET not in S2.FEATURE_SETS and S2.uses_notes(S2.NDR_SET)


def test_pair_features_ndr_prefix_bit_identical_and_columns():
    st, FM, PM, kn, R = _static()
    rows = np.arange(len(FM))
    X0, c0, n0 = S2.pair_features(st, rows, FM, PM, kn, S2.ND_SET)
    X, c, names = S2.pair_features(st, rows, FM, PM, kn, S2.NDR_SET)
    assert np.array_equal(c, c0) and X[:, :-6].tobytes() == X0.tobytes() and names == n0 + list(S2.R_FEATS)
    ii, cc = np.repeat(rows, S2.K_CAND), c.reshape(-1)
    assert np.array_equal(X[:, -6:], R[ii, cc, :])
    sub = np.array([4, 1])  # subset of the rows: R of the right table rows
    Xs, cs, _ = S2.pair_features(st, sub, FM[sub], PM[sub], {**kn, **{k: (v[sub] if np.ndim(v) and len(v) == len(FM) else v) for k, v in kn.items()}},
                                 S2.NDR_SET)
    assert np.array_equal(Xs[:, -6:], R[sub[np.repeat(np.arange(2), S2.K_CAND)], cs.reshape(-1), :])


def test_pair_features_ndr_requires_attach_r_and_checks():
    st, FM, PM, kn, _ = _static(with_r=False)
    with pytest.raises(ValueError, match="attach_r"):
        S2.pair_features(st, np.arange(len(FM)), FM, PM, kn, S2.NDR_SET)
    with pytest.raises(ValueError):
        S2.attach_r(st, np.zeros((st.n, st.K, 5)))
    bad = np.zeros((st.n, st.K, 6))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        S2.attach_r(st, bad)


# ------------------------------------------------------------------------------------------------ test side (make_submission)

def test_control_pr27():
    MS = _load("make_submission_ndr_ctl", "tools/make_submission.py")
    pr = [f"p{i}" for i in range(5)]
    m3 = [[f"m{d}{i}" for i in range(5)] for d in range(3)]
    assert MS._control_pr27(pr, m3, "pr27_shift") == ["p1", "p2", "p3", "p4", "p0"]
    assert MS._control_pr27(pr, m3, "pr27_m3d1") == m3[0]
    assert MS._control_pr27(["x" * 400] * 2, m3, "truncate") == ["x" * 300] * 2
    assert MS._control_pr27(pr, m3, "copy") == pr and MS._control_pr27(pr, m3, "permute") == pr
    with pytest.raises(SystemExit):
        MS._control_pr27(pr, m3, "other")
    assert set(MS.R_CONTROLS) <= set(MS.STAGE2_CONTROLS) and MS.M3_CONTROLS == ("copy", "permute", "truncate")
    with pytest.raises(SystemExit):  # the M3 controls stay unchanged
        MS._control_texts(m3, "pr27_shift")


def test_pr27_notes_texts_reads_frozen_file(tmp_path):
    MS = _load("make_submission_ndr_read", "tools/make_submission.py")
    rows = [{"table": "test", "row": i, "state": "offen", "text": f"note {i} row ALG-01"} for i in (2, 0, 1)]
    f = tmp_path / "derived" / "pn" / "test_pr.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"table": "test", "rows": 3, "sources": [{"source": "r/j/t", "rows": 3, "without_notes": 0}], "sha256": sha(f)}
    f.with_suffix(".meta.json").write_text(json.dumps(meta))
    cfg = {"pr27_test": {"file": "derived/pn/test_pr.jsonl", "sha256": sha(f), "source": "r/j/t"}}
    tx, src = MS._pr27_notes_texts(tmp_path, cfg, 3)
    assert tx == [f"note {i} row ALG-01" for i in range(3)] and src == {"file": "derived/pn/test_pr.jsonl", "sha256": sha(f), "source": "r/j/t"}
    for over, n in (({"sha256": "0" * 64}, 3), ({"source": "x/y/z"}, 3), ({}, 4)):
        with pytest.raises(SystemExit):
            MS._pr27_notes_texts(tmp_path, {"pr27_test": {**cfg["pr27_test"], **over}}, n)
    with pytest.raises(SystemExit):
        MS._pr27_notes_texts(tmp_path, {}, 3)


SAFE_CHANNELS = ["v1r", "zs27t", "think18sb", "pr27xm", "m3x3_full"]


def _toy_ndr(tmp_path, MS, *, w_r: float = 0.0, oof_share: float = 0.0, feature_set: str = S2.NDR_SET):
    """Synthetic inputs for apply_stage2_safe with fallback_i1_ndr; note files via stand-ins (monkeypatch of the reading functions)."""
    import pandas as pd

    cls = ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "Dec-01", "PV-01", "ALG-06", "FRA-02"]
    K, n_test, n_train = len(cls), 60, 40
    rng = np.random.default_rng(11)
    seen = np.ones(K, bool)
    seen[9] = False

    def rnd(n=n_test):
        P = rng.dirichlet(np.ones(K), size=n) * seen[None, :]
        return P / P.sum(1, keepdims=True)

    raw, draws, Q, P_pre = {nm: rnd() for nm in SAFE_CHANNELS}, [rnd(), rnd(), rnd()], rnd(), rnd()
    Q[:6, 0] += 2.0
    Q = Q / Q.sum(1, keepdims=True)
    cand0 = S2.candidates(Q, seen)
    stack = {"channels": list(SAFE_CHANNELS), "arith_beta": 0.4}
    stack_file = tmp_path / "stack_toy.json"
    stack_file.write_text(json.dumps(stack))
    train = pd.DataFrame({"root_code": [cls[i % 9] for i in range(n_train)], "question_id": [f"q{i % 7}" for i in range(n_train)],
                          "topic_name": ["A" if i % 2 else "B" for i in range(n_train)]})
    test = pd.DataFrame({"question_id": [f"q{i % 7}" for i in range(n_test)], "topic_name": ["A" if i % 3 else "B" for i in range(n_test)],
                         "grade": [7] * n_test})
    pad = "." * 310  # control truncate (first 300 characters) deletes all mentions
    m3_test = [[f"{pad} draw {d} names {cls[cand0[i, 1]]} and then {cls[cand0[i, 2]]}" for i in range(n_test)] for d in range(3)]
    m3_train = [[f"draw {d} training {cls[1 + i % 8]}" for i in range(n_train)] for d in range(3)]
    pr_test = [f"{pad} pr27 checks {cls[cand0[i, 3]]} last {cls[cand0[i, 0]]}" for i in range(n_test)]  # pr27 names the base top code (except NONE)
    MS._test_notes_texts = lambda root, rel, rank, n: (m3_test[int(rel[-1])], {"file": rel, "rank": rank})
    MS._train_notes_texts = lambda root, rel, want, n: (m3_train[int(rel[-1])], {"file": rel, "sha256": want})
    MS._pr27_notes_texts = lambda root, cfg, n: (pr_test, {"file": "pr", "sha256": "0" * 64, "source": "x"})
    # references = statistics of the unchanged test side (all guards hold); the controls must fire
    notes = S2.notes_features(m3_test, cls)
    gref = S2.note_stats(notes["men_per_draw"], notes["m3_men"], cand0, cand0[:, 0])
    gref["identical"] = {"d1-d2": 0.0, "d1-d3": 0.0, "d2-d3": 0.0}
    Ssim = NS.mask_same_task(NS.similarity(m3_test, m3_train), S2.task_keys(test["question_id"], test["topic_name"]),
                             S2.task_keys(train["question_id"], train["topic_name"]))
    sref = NS.sim_stats(NS.top_sim(Ssim))
    m3_st = [NR.mention_stats(t, cls) for t in m3_test]
    rref = NR.guard_stats(m3_st, NR.mention_stats(pr_test, cls), cand0)
    names = S2.feature_names(SAFE_CHANNELS, feature_set)
    w = [0.0] * len(names)
    if feature_set == S2.NDR_SET:
        w[names.index("t_pr_last")] = w_r  # positive: the candidate pr27 names last (the base top code) stays in front
        w[names.index("t_pr_men")] = -w_r  # negative: the other candidate pr27 names (rank 4) loses
        w[names.index("f_rank")] = w_r  # positive: candidates further back move to the front
    ranks = ["runs/r1/job/test", "runs/r2/job/test", "runs/r3/job/test"]
    s2 = {"stack_sha256": sha(stack_file), "channels": list(SAFE_CHANNELS), "arith_beta": 0.4, "align_mode": "set", "align": 0.75, "align_min": 3,
          "K": S2.K_CAND, "lambda": S2.LAMBDA, "null": w_r == 0.0, "feature_set": feature_set, "protect_none": True, "draws_channel": "m3x3_full",
          "test_draws": ranks, "test_notes": [{"file": f"t{d}", "rank": r} for d, r in enumerate(ranks)],
          "train_notes": [{"file": f"r{d}", "sha256": "1" * 64} for d in range(3)],
          "model": {"features": names, "w": w, "mu": [0.0] * len(names), "sd": [1.0] * len(names), "oof_changed_top_share": oof_share,
                    "oof_changed_top3_share": oof_share},
          "novel_task_share_top5": 0.5, "guard_reference": {**gref, "identical": gref["identical"]}, "sim_reference": sref,
          "guard_rule": {"changed_top_factor": [0.5, 2.0], "rule": "widened", "share_z_max": 3.0, "notes_rel_tolerance": 0.1, "n_ref": n_train,
                         "sim_rel": 0.1, "sim_z_max": 3.0, "abort_only": True}}
    if feature_set == S2.NDR_SET:
        s2["guard_rule"].update({"r_rule": "safety-ndr rule (6)", "r_rel": 0.10, "r_z_max": 3.0})
        s2.update({"r_reference": rref, "r_notes": {"extractor": NR.VERSION, "features": list(S2.R_FEATS),
                                                    "pr27_test": {"file": "pr", "sha256": "0" * 64, "source": "x"}}})
    s2_file = tmp_path / "stage2_ndr_toy.json"
    s2_file.write_text(json.dumps(s2))
    identical = S2.identical_share  # notes of the test side with identical wording: 0 in the reference -> keep the test draws different
    assert identical(m3_test[0], m3_test[1]) == 0.0
    return (s2_file, stack_file, stack, Q, P_pre, raw, {"m3x3_full": draws}, {"m3x3_full": ",".join(ranks)}, train, test, cls, seen, tmp_path), cand0, pr_test


def test_apply_stage2_safe_ndr_null_bitwise_and_report(tmp_path):
    MS = _load("make_submission_ndr_null", "tools/make_submission.py")
    args, _cand0, _ = _toy_ndr(tmp_path, MS)
    rep = tmp_path / "guards.json"
    Q_new, info = MS.apply_stage2_safe(*args, report_path=rep)
    assert Q_new.tobytes() == args[3].tobytes() and info["null"] and info["feature_set"] == S2.NDR_SET and info["n_features"] == 64
    g = json.loads(rep.read_text())
    assert g["guard_r"]["ok"] and len(g["guard_r"]["checks"]) == 5 and g["r_notes"]["source"] == "x" and g["guard_notes"]["ok"] and g["guard_sim"]["ok"]


def test_apply_stage2_safe_ndr_controls_fire_without_file(tmp_path):
    MS = _load("make_submission_ndr_controls", "tools/make_submission.py")
    for ctl in ("pr27_shift", "pr27_m3d1", "truncate"):
        args, _, _ = _toy_ndr(tmp_path, MS, w_r=1.0, oof_share=0.5)
        rep = tmp_path / f"control_{ctl}.json"
        with pytest.raises(SystemExit) as e:
            MS.apply_stage2_safe(*args, report_path=rep, control=ctl)
        assert e.value.code == 5, ctl
        g = json.loads(rep.read_text())
        assert not g["guard_r"]["ok"] and g["control"] == ctl
        if ctl != "truncate":  # the pr27 controls leave the M3 notes unchanged: only the guard R fires
            assert g["guard_notes"]["ok"] and g["guard_sim"]["ok"]


def test_apply_stage2_safe_ndr_reorders_and_guard_aborts(tmp_path):
    MS = _load("make_submission_ndr_apply", "tools/make_submission.py")
    args, cand0, _ = _toy_ndr(tmp_path, MS, w_r=5.0, oof_share=0.5)
    Q_new, info = MS.apply_stage2_safe(*args)
    seen = args[11]
    t_new = S2.candidates(Q_new, seen)
    keep_none = cand0[:, 0] == 0
    assert np.array_equal(t_new[keep_none], cand0[keep_none])  # NONE guard
    assert info["guard_r"]["ok"] and info["feature_set"] == S2.NDR_SET and np.allclose(Q_new.sum(1), 1.0)
    assert (t_new[~keep_none, 0] != cand0[~keep_none, 0]).any()  # the columns R act on the order
    # guard R violated (reference shifted) -> abort without a return value, no switch
    s2 = json.loads(args[0].read_text())
    s2["r_reference"]["pr27_share_top_mentioned"]["share"] = 0.2
    args[0].write_text(json.dumps(s2))
    with pytest.raises(SystemExit, match="R guard failed"):
        MS.apply_stage2_safe(*args)


def test_apply_stage2_safe_ndr_rejects_wrong_rule_and_nd_controls(tmp_path):
    MS = _load("make_submission_ndr_reject", "tools/make_submission.py")
    args, _, _ = _toy_ndr(tmp_path, MS)
    s2 = json.loads(args[0].read_text())
    s2["guard_rule"]["r_z_max"] = 4.0
    args[0].write_text(json.dumps(s2))
    with pytest.raises(SystemExit, match="guard of the columns R"):
        MS.apply_stage2_safe(*args)
    args, _, _ = _toy_ndr(tmp_path, MS, feature_set=S2.ND_SET)  # deployment file of the form ND: pr27 controls rejected
    with pytest.raises(SystemExit, match="fallback_i1_ndr"):
        MS.apply_stage2_safe(*args, control="pr27_shift")


# ------------------------------------------------------------------------------------------------ pre-registered rule (c130)

def _rule_inputs(A=0.0008, pl=-0.0001, sets=(0.001, 0.0003, 0.0008, 0.0002), spl=0.0006, null=True):
    oof = {"null_point": {"r_zero_equals_nd": {"ok": null}, "nd_prefix_equals_c129": {"ok": True}, "nd_equals_safety_nd_oof_rows": {"ok": True}},
           "ndr_vs_nd": {"A": A, "F": A, "M": A, "A_boot": {"se_q": 0.001}}, "placebo": {"n": 15, "A_mean": pl, "A_max": 0.001}}
    swap = {"sets": {c: {"splits_A": {"mean": v}} for c, v in zip(("123", "124", "134", "234"), sets)},
            "sensitivity": {"123_pr27n2": {"splits_A": {"mean": 0.0005}, "original": {"A": 0.0004}}}}
    sp = {"splits": [{}] * 20, "summary": {"ndr_minus_nd": {"A_mean": spl, "A_se": 0.0002, "share_A_pos": 0.7}, "placebo_minus_nd": {"A_mean": 0.0}}}
    return oof, swap, sp


def test_c130_rule_logic():
    C130 = _load("c130_rule_t", "eda/c130_safety_ndr.py")
    calls = []

    def fake_risk(o, s, tau):
        calls.append((o, s, tau))
        return {"o": o, "s": s, "tau_d": tau, "mean": o, "p_lt_m002": 0.01 if o > -0.001 else 0.2}

    assert C130.evaluate_rule(*_rule_inputs(), risk=fake_risk)["verdict"] == "deploy"
    tau = float(np.std([0.001, 0.0003, 0.0008, 0.0002], ddof=1))
    assert (0.0008, 0.001, tau) in calls and (float(np.mean([0.001, 0.0003, 0.0008, 0.0002])), 0.001, tau) in calls
    for over in ({"sets": (0.001, -0.0003, -0.0008, 0.0002)}, {"sets": (-0.002, 0.0003, 0.0008, 0.0002)}, {"A": -0.0001}, {"pl": 0.0009},
                 {"spl": -0.00001}, {"null": False}, {"A": -0.002}):
        assert C130.evaluate_rule(*_rule_inputs(**over), risk=fake_risk)["verdict"] == "no deployment", over
    oof, swap, sp = _rule_inputs()
    oof["placebo"]["n"] = 10
    assert C130.evaluate_rule(oof, swap, sp, risk=fake_risk)["verdict"] == "no deployment"
    sp["splits"] = [{}] * 19
    oof["placebo"]["n"] = 15
    assert C130.evaluate_rule(oof, swap, sp, risk=fake_risk)["verdict"] == "no deployment"


def test_c130_constants_follow_the_rule():
    C130 = _load("c130_const_t", "eda/c130_safety_ndr.py")
    assert C130.SWAP_SETS == ("123", "124", "134", "234") and C130.MIN_SPLITS == 20 and C130.MIN_PLACEBOS == 15 and len(C130.PLACEBO_SEEDS) == 15
    assert C130.RISK_PHARM == 0.05 and (C130.RISK_K, C130.RISK_RHO, C130.RISK_SPREAD) == (6, 0.5, 0.0015) and C130.NR_COLS == 6
    assert C130.PR27_NOTES["test_pr27"][2] == ("pr27te_rp/proto_te/proto",) and C130.PR27_NOTES["train_pr27"][2] == ("pr27_rp/proto_a/proto", "pr27_rp/proto_b/proto")


# ------------------------------------------------------------------------------------------------ real files (skipped if they are missing)

@pytest.mark.skipif(not (PRIV / "c130_oof.json").is_file(), reason="c130 oof not computed yet")
def test_c130_oof_null_points_and_reference():
    o = json.loads((PRIV / "c130_oof.json").read_text())
    n = o["null_point"]
    assert n["r_zero_equals_nd"]["ok"] and n["nd_prefix_equals_c129"]["ok"] and n["nd_equals_safety_nd_oof_rows"]["ok"] in (True, None)
    assert o["feature_set"] == S2.NDR_SET and o["r_features"] == list(S2.R_FEATS) and o["placebo"]["n"] == 15
    assert o["sets"]["ndr"]["set"]["n_features"] == 64 and o["sets"]["nd"]["set"]["n_features"] == 58


@pytest.mark.skipif(not (COMP / "artifacts" / "stage2_safety_ndr.json").is_file(), reason="deployment file stage2_safety_ndr.json not built yet")
def test_stage2_safety_ndr_file_fields_and_bzero_prefix(tmp_path):
    ndr = json.loads((COMP / "artifacts" / "stage2_safety_ndr.json").read_text())
    nd = json.loads((COMP / "artifacts" / "stage2_safety_nd.json").read_text())
    assert ndr["feature_set"] == S2.NDR_SET and ndr["align_mode"] == "set" and ndr["protect_none"] is True and not ndr["null"]
    assert ndr["model"]["features"][:-6] == nd["model"]["features"] and tuple(ndr["model"]["features"][-6:]) == S2.R_FEATS
    assert ndr["guard_reference"] == nd["guard_reference"] and ndr["sim_reference"] == nd["sim_reference"]
    rn = ndr["r_notes"]
    assert rn["extractor"] == NR.VERSION and rn["pr27_test"]["source"] == "pr27te_rp/proto_te/proto"
    data = COMP / "data"
    if (data / rn["pr27_test"]["file"]).is_file():
        assert sha(data / rn["pr27_test"]["file"]) == rn["pr27_test"]["sha256"]
    gr = ndr["guard_rule"]
    assert (gr["r_rule"], gr["r_rel"], gr["r_z_max"], gr["abort_only"]) == ("safety-ndr rule (6)", 0.1, 3.0, True)
    C130 = _load("c130_bzero_t", "eda/c130_safety_ndr.py")
    out = tmp_path / "bzero.json"
    assert C130.main(["b-zero", "--out", str(out)]) == 0
    b = json.loads(out.read_text())
    assert b["model"]["w"] == nd["model"]["w"] + [0.0] * 6 and b["model"]["sd"] == nd["model"]["sd"] + [1.0] * 6


def test_build_script_safety_ndr_pins_and_gate():
    p = COMP / "tools" / "builds" / "safety_ndr.sh"
    text = p.read_text()
    assert "bash tools/builds/safety_nd.sh" in text and "c130_safety_ndr.py rule" in text and "c130_safety_ndr.py export-notes --verify" in text
    assert "for CTL in copy permute truncate pr27_shift pr27_m3d1" in text and '[ "$rc" = 5 ]' in text
    assert '--align 0.75 --align-min 3' in text and "--align-inductive" not in text
    for var, path in (("IOOF_SHA", PRIV / "c130_oof.json"), ("ISWAP_SHA", PRIV / "c130_swap.json"), ("ISPL_SHA", PRIV / "c130_splits.json"),
                      ("STAGE2_SHA", COMP / "artifacts" / "stage2_safety_ndr.json"), ("OUT_SHA", COMP / "artifacts" / "submissions" / "sub_safety_ndr.csv")):
        m = re.search(rf'\b{var}="([0-9a-f]{{64}})?"', text)
        assert m, var
        if m.group(1) and path.is_file():
            assert m.group(1) == sha(path), var  # pinned after the first build
    for var, path in (("SAFETY_ND_SHA", COMP / "artifacts" / "submissions" / "sub_safety_nd.csv"),
                      ("SAFETY_ND_STAGE2_SHA", COMP / "artifacts" / "stage2_safety_nd.json")):
        m = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        assert m and (not path.is_file() or m.group(1) == sha(path)), var
