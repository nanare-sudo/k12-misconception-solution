"""Tests of the feature set main_guard_i1 (main_guard + 15 interactions I1) — src/stage2.py, eda/c121_stage2_i1.py (oof, splits, fit,
submit) and the build script tools/builds/primary_i1.sh.

Synthetic (fast, without data): names, dimension and order; main_guard bit-identical as a prefix; interactions = product of the raw
values; placebo permutation; I1 weights 0 ⇒ ranking of main_guard; existing sets unchanged; test side through c121 submit (path of the
original build; make_submission.py also admits the set itself). With real files (skipped if they are missing): c121 oof (copy null point
(i)), stage2_primary_i1.json = stage2_s035.json except models.main/i1_build/derived_from, fit --verify, fixed checksums in the build
script. None of these tests writes to artifacts/."""
from __future__ import annotations

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
CH = ["v1r", "zs27q", "think18nb", "pr27xm", "m3x3_full", "q397m3"]
I1_NAMES = [f"{r}_x_{c}" for r in ("row_ent", "row_max", "row_d_ntop", "row_ch_ntop", "task_rows") for c in ("f_logp", "is_general", "is_none")]


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


def _toy(n=60, seed=1, notes=True):
    rng = np.random.default_rng(seed)
    cls = classes_toy()
    K = len(cls)
    seen = np.ones(K, bool)
    seen[9] = False

    def rnd():
        return rng.dirichlet(np.ones(K), size=n)

    chans = {nm: rnd() for nm in CH}
    nf = S2.notes_features([["CAL-01 ANS-01"] * n, ["WOP-01"] * n, [""] * n], cls) if notes else None
    st = S2.Static(CH, chans, [rnd(), rnd(), rnd()], rng.integers(4, 13, n).astype(float), seen, cls, nf)
    P = rnd()
    Q = np.where(seen[None, :], P, 0.0)
    Q = Q / Q.sum(1, keepdims=True)
    q = rng.integers(0, 6, n).astype(str)
    t = np.where(rng.random(n) < 0.5, "A", "B")
    y = rng.integers(0, K - 1, n)
    key = S2.task_keys(q, t)
    kn = S2.knowledge(key, t, y, key, t, K)
    return st, Q, P, kn, cls, seen


# ------------------------------------------------------------------------------------------------ synthetic: src/stage2.py

def test_i1_constants_names_and_existing_sets_unchanged():
    assert S2.FEATURE_SETS == ("main", "fallback") and S2.GUARD_SET == "main_guard" and S2.I1_SET == "main_guard_i1"
    assert S2.I1_SET not in S2.FEATURE_SETS
    assert list(S2.I1_FEATS) == I1_NAMES and len(S2.I1_FEATS) == 15
    assert S2.I1_ROW_FEATS == ("row_ent", "row_max", "row_d_ntop", "row_ch_ntop", "task_rows")
    assert S2.I1_CAND_FEATS == ("f_logp", "is_general", "is_none")
    main, fb, guard, i1 = (S2.feature_names(CH, fs) for fs in ("main", "fallback", S2.GUARD_SET, S2.I1_SET))
    assert (len(main), len(fb), len(guard), len(i1)) == (43, 40, 48, 63)
    assert i1[:48] == guard and i1[48:] == I1_NAMES and len(set(i1)) == 63
    assert S2.uses_notes(S2.I1_SET) and S2.uses_notes("main") and S2.uses_notes(S2.GUARD_SET) and not S2.uses_notes("fallback")
    with pytest.raises(ValueError):
        S2.feature_names(CH, "main_guard_i2")


def test_i1_pair_features_guard_prefix_bitwise_and_products():
    st, Q, P, kn, cls, seen = _toy()
    rows = np.arange(st.n)
    Xg, cg, ng = S2.pair_features(st, rows, Q, P, kn, S2.GUARD_SET)
    Xi, ci, ni = S2.pair_features(st, rows, Q, P, kn, S2.I1_SET)
    assert Xi.shape == (st.n * S2.K_CAND, 63) and Xi.dtype == np.float64
    assert np.array_equal(ci, cg) and ni == ng + I1_NAMES and ni == S2.feature_names(CH, S2.I1_SET)
    assert Xi[:, :48].tobytes() == Xg.tobytes()  # main_guard unchanged, bit-identical
    # definition: for each row feature r (outer) and candidate feature c (inner) the product of the raw values
    want = []
    for r in S2.I1_ROW_FEATS:
        rv = Xg[:, ng.index(r)]
        for c in S2.I1_CAND_FEATS:
            want.append(rv * Xg[:, ng.index(c)])
    assert Xi[:, 48:].tobytes() == np.stack(want, 1).tobytes()
    # the row features are constant within a row, the interactions are not (otherwise they would have no effect in the conditional logit)
    R = st.n
    for r in S2.I1_ROW_FEATS:
        M = Xg[:, ng.index(r)].reshape(R, -1)
        assert np.array_equal(M, np.repeat(M[:, :1], M.shape[1], 1))
    W = Xi[:, 48:].reshape(R, S2.K_CAND, 15)
    assert (W != W[:, :1, :]).any((0, 1)).all()
    st0, Q0, P0, kn0, _, _ = _toy(notes=False)
    with pytest.raises(ValueError, match="requires notes"):
        S2.pair_features(st0, np.arange(st0.n), Q0, P0, kn0, S2.I1_SET)


def test_i1_interactions_perm_and_identity():
    st, Q, P, kn, cls, seen = _toy(n=30, seed=4)
    Xg, _, ng = S2.pair_features(st, np.arange(st.n), Q, P, kn, S2.GUARD_SET)
    base = S2.i1_interactions(Xg, ng)
    Kc, R = S2.K_CAND, st.n
    ident = np.arange(R * Kc)
    assert S2.i1_interactions(Xg, ng, ident).tobytes() == base.tobytes()
    rng = np.random.default_rng(9)
    perm = np.repeat(rng.permutation(R) * Kc, Kc) + np.tile(np.arange(Kc), R)
    got = S2.i1_interactions(Xg, ng, perm)
    for j, (r, c) in enumerate((r, c) for r in S2.I1_ROW_FEATS for c in S2.I1_CAND_FEATS):
        assert np.array_equal(got[:, j], Xg[perm, ng.index(r)] * Xg[:, ng.index(c)])
    # with the 63 columns (main_guard_i1) as input the same values: the names of the interactions do not collide
    Xi, _, ni = S2.pair_features(st, np.arange(st.n), Q, P, kn, S2.I1_SET)
    assert S2.i1_interactions(Xi, ni, perm).tobytes() == got.tobytes()


def test_i1_zero_weights_give_main_guard_ranking():
    st, Q, P, kn, cls, seen = _toy(n=80, seed=2)
    rows = np.arange(st.n)
    Xg, cg, ng = S2.pair_features(st, rows, Q, P, kn, S2.GUARD_SET)
    Xi, ci, ni = S2.pair_features(st, rows, Q, P, kn, S2.I1_SET)
    rng = np.random.default_rng(5)
    mg = {"w": rng.normal(size=48), "mu": Xg.mean(0), "sd": Xg.std(0) + 1e-9}
    mi = {"w": np.r_[mg["w"], np.zeros(15)], "mu": np.r_[mg["mu"], np.zeros(15)], "sd": np.r_[mg["sd"], np.ones(15)]}
    tg = S2.reorder(cg, S2.scores(mg, Xg))
    ti = S2.reorder(ci, S2.scores(mi, Xi))
    assert np.array_equal(ti, tg) and np.array_equal(S2.protect_none(ti, ci), S2.protect_none(tg, cg))
    assert not np.array_equal(tg, cg)  # the check really reorders
    z = S2.reorder(ci, S2.scores(S2.zero_model(63), Xi))
    assert np.array_equal(z, ci)


def test_i1_fit_with_zero_columns_keeps_weights_zero():
    """Copy null point (i) in small: I1 columns = 0 -> standardisation 0, the weights of the columns stay 0 (gradient λ·w from zero)."""
    st, Q, P, kn, cls, seen = _toy(n=120, seed=6)
    rows = np.arange(st.n)
    Xi, ci, ni = S2.pair_features(st, rows, Q, P, kn, S2.I1_SET)
    y = np.random.default_rng(1).integers(0, 9, st.n)
    lab = S2.labels_of(ci, y)
    X0 = Xi.copy()
    X0[:, 48:] = 0.0
    m = S2.fit_clogit(X0, lab)
    assert np.all(m["w"][48:] == 0.0) and np.all(m["mu"][48:] == 0.0)


# ------------------------------------------------------------------------------------------------ synthetic: test side (c121 submit)

def test_make_submission_admits_i1_for_main_only_and_c121_the_same():
    """make_submission.py admits main_guard_i1 itself for the main model (a fallback model with this set is rejected). The wrapper
    c121 submit (path of the original build) behaves the same."""
    MS0 = _load("make_submission_i1_plain", "tools/make_submission.py")
    cls = classes_toy()
    assert MS0._stage2_model_form({"feature_set": S2.I1_SET, "protect_none": True}, "main", cls) == (S2.I1_SET, True)
    with pytest.raises(SystemExit, match="reads the notes"):
        MS0._stage2_model_form({"feature_set": S2.I1_SET}, "fallback", cls)
    with pytest.raises(SystemExit, match="unknown"):
        MS0._stage2_model_form({"feature_set": "main_guard_i2", "protect_none": True}, "main", cls)
    C121 = _load("c121_form_test", "eda/c121_stage2_i1.py")
    MS = C121.load_make_submission("make_submission_i1_form")
    assert MS._stage2_model_form({"feature_set": S2.I1_SET, "protect_none": True}, "main", cls) == (S2.I1_SET, True)
    assert MS._stage2_model_form({"feature_set": S2.I1_SET}, "main", cls) == (S2.I1_SET, False)
    # all other models as the original function
    assert MS._stage2_model_form({"features": []}, "main", cls) == ("main", False)
    assert MS._stage2_model_form({"features": []}, "fallback", cls) == ("fallback", False)
    assert MS._stage2_model_form({"feature_set": S2.GUARD_SET, "protect_none": True}, "main", cls) == (S2.GUARD_SET, True)
    for model, fs, msg in (({"feature_set": S2.I1_SET}, "fallback", "reads the notes"), ({"feature_set": "everything"}, "main", "unknown"),
                           ({"feature_set": S2.I1_SET, "protect_none": 1}, "main", "not a boolean"), ({"feature_set": S2.GUARD_SET}, "fallback", "reads the notes")):
        with pytest.raises(SystemExit, match=msg):
            MS._stage2_model_form(model, fs, cls)
    with pytest.raises(SystemExit, match="class 0"):
        MS._stage2_model_form({"feature_set": S2.I1_SET, "protect_none": True}, "main", cls[1:] + cls[:1])


def test_c121_apply_stage2_i1_model_zero_i1_weights_equals_guard(tmp_path, monkeypatch):
    """Test side: a main_guard_i1 model with the weights of a main_guard model and I1 weights 0 gives bit-identically the same matrix as
    the main_guard model; a model with weight only on one interaction orders by the I1 features of the test side."""
    import labels

    T = _load("test_stage2_base_for_i1", "tests/test_stage2.py")
    C121 = _load("c121_apply_test", "eda/c121_stage2_i1.py")
    MS = C121.load_make_submission("make_submission_i1_apply")
    rule = {"notes_rel_tolerance": 0.10, "changed_top_factor": [0.5, 2.0], "text": "band rule", "rule": "widened", "share_z_max": 3.0, "n_ref": 40}
    args, texts = T._toy_apply_stage2(tmp_path, rule)
    monkeypatch.setattr(MS, "_test_notes_texts", lambda root, f, r, n: (texts[f], {"file": f, "sha256": "x", "rank": r, "rank_sha256": "y"}))
    s2_file, stack, Q, P_pre, raw, draw_keep, train, test, cls, seen = (args[0], args[2], args[3], args[4], args[5], args[6], args[8], args[9],
                                                                         args[10], args[11])
    cand = S2.candidates(Q, seen)
    ans = cls.index("ANS-01")
    has = (cand == ans).any(1)
    expect = np.where(has, ans, cand[:, 0])
    expect[has & (cand[:, 0] == 0)] = 0
    changed = float((expect != cand[:, 0]).mean())
    s2 = json.loads(s2_file.read_text())
    ng, ni = S2.feature_names(T.SIX_CHANNELS, S2.GUARD_SET), S2.feature_names(T.SIX_CHANNELS, S2.I1_SET)
    wg = [0.0] * len(ng)
    wg[ng.index("id_ANS-01")] = 5.0
    base_model = {"mu": [0.0] * len(ng), "sd": [1.0] * len(ng), "protect_none": True, "oof_changed_top_share": changed, "oof_changed_top3_share": 0.0}
    s2["models"]["main"] = {"features": ng, "feature_set": S2.GUARD_SET, "w": wg, **base_model}
    s2_file.write_text(json.dumps(s2))
    Qg, info_g = MS.apply_stage2(*args)
    s2["models"]["main"] = {"features": ni, "feature_set": S2.I1_SET, "w": wg + [0.0] * 15, **base_model,
                            "mu": [0.0] * len(ni), "sd": [1.0] * len(ni)}
    s2_file.write_text(json.dumps(s2))
    Qi, info_i = MS.apply_stage2(*args, report_path=tmp_path / "gi.json")
    assert Qi.tobytes() == Qg.tobytes() and np.array_equal(S2.candidates(Qi, seen)[:, 0], expect)
    assert info_i["model_feature_set"] == S2.I1_SET and info_i["n_features"] == 63 and info_i["protect_none"] is True
    assert info_i["rows_protected"] == info_g["rows_protected"] and info_i["guard_changed_top"]["ok"]
    assert json.loads((tmp_path / "gi.json").read_text())["model_feature_set"] == S2.I1_SET
    # weight only on one interaction (guard off, null-model flag: guard (c) not enforced): order = own recomputation
    wi = [0.0] * len(ni)
    wi[ni.index("row_max_x_is_general")] = -4.0
    wi[ni.index("task_rows_x_f_logp")] = 0.7
    s2["null"] = True
    s2["models"]["main"] = {"features": ni, "feature_set": S2.I1_SET, "protect_none": False, "w": wi, "mu": [0.0] * len(ni), "sd": [1.0] * len(ni),
                            "oof_changed_top_share": 0.0, "oof_changed_top3_share": 0.0}
    s2_file.write_text(json.dumps(s2))
    Qw, info_w = MS.apply_stage2(*args)
    notes = S2.notes_features([texts[c["file"]] for c in s2["test_notes"]], cls)
    st = S2.Static(stack["channels"], raw, draw_keep["m3x3_full"], test["grade"].to_numpy().astype(float), seen, cls, notes)
    kn = S2.knowledge(S2.task_keys(train["question_id"].to_numpy(), train["topic_name"].to_numpy()), np.asarray(train["topic_name"]).astype(str),
                      labels.encode(train["root_code"], cls), S2.task_keys(test["question_id"].to_numpy(), test["topic_name"].to_numpy()),
                      np.asarray(test["topic_name"]).astype(str), len(cls))
    X, c2, names = S2.pair_features(st, np.arange(len(test)), Q, P_pre, kn, S2.I1_SET)
    want = S2.reorder(c2, S2.scores({"w": wi, "mu": [0.0] * 63, "sd": [1.0] * 63}, X))
    assert names == ni and np.array_equal(S2.candidates(Qw, seen), want) and not np.array_equal(want, cand)
    assert info_w["model_feature_set"] == S2.I1_SET and info_w["rows_protected"] == 0


# ------------------------------------------------------------------------------------------------ real files

GUARD = COMP / "artifacts" / "stage2_s035.json"
I1 = COMP / "artifacts" / "stage2_primary_i1.json"
IOOF = COMP / "artifacts" / "eda" / "private" / "stage2_i1" / "c121_i1_oof.json"
ISPL = COMP / "artifacts" / "eda" / "private" / "stage2_i1" / "c121_i1_splits.json"
needs_ioof = pytest.mark.skipif(not IOOF.is_file(), reason="c121 oof not computed yet")
needs_i1 = pytest.mark.skipif(not (I1.is_file() and GUARD.is_file()), reason="stage2_primary_i1.json missing (only after the pre-registered rule is met)")


@needs_ioof
def test_c121_oof_null_point_and_reference():
    g = json.loads(IOOF.read_text())
    assert g["null_point_i"]["ok"] and all(g["null_point_i"]["forms"].values())
    assert g["feature_set"] == S2.I1_SET and g["protect_none"] is True and g["interactions"]["features"] == I1_NAMES
    for f in ("fold", "pre", "set"):
        assert g["sets"]["guard"][f]["rows_equal_reference"] and g["sets"]["i1"][f]["n_features"] == 63
        assert g["sets"]["i1"][f]["null_i1_zero_equals_guard"]
    assert g["sets"]["i1"]["set"]["features"][48:] == I1_NAMES and g["placebo"]["n"] >= 5
    goof = json.loads((COMP / "artifacts/eda/private/stage2_guard/c118_guard_oof.json").read_text())
    for f in ("fold", "pre", "set"):
        assert g["sets"]["guard"][f]["map_forced"] == goof["sets"]["main_guard"][f]["map_forced"]
    assert g["summary"]["A"] == pytest.approx(0.5 * (g["summary"]["F"] + g["summary"]["M"]))


@needs_i1
def test_stage2_primary_i1_equals_guard_form_except_main():
    a, b = json.loads(GUARD.read_text()), json.loads(I1.read_text())
    assert [k for k in b if k not in ("i1_build", "derived_from")] == [k for k in a if k != "derived_from"]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {"models", "i1_build", "derived_from"}
    assert b["models"]["fallback"] == a["models"]["fallback"] and list(b["models"]) == ["main", "fallback"]
    m, am = b["models"]["main"], a["models"]["main"]
    assert m["feature_set"] == S2.I1_SET and m["protect_none"] is True and len(m["w"]) == 63 and not m["abstain"]
    assert m["features"] == am["features"] + I1_NAMES
    assert b["derived_from"]["file"] == "artifacts/stage2_s035.json" and b["derived_from"]["sha256"] == sha(GUARD)
    assert b["derived_from"]["previous"] == a["derived_from"]
    assert b["i1_build"]["rule_oof"]["verdict_oof"] == "deploy"
    g = json.loads(IOOF.read_text())
    assert m["oof_changed_top_share"] == g["sets"]["i1"]["set"]["changed_top_share_forced"]


@needs_i1
def test_c121_fit_verify_reproduces_primary_i1():
    r = subprocess.run(PY + ["eda/c121_stage2_i1.py", "fit", "--stack", "s032_stage2", "--verify"], cwd=COMP, env=ENV, capture_output=True,
                       text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERIFIED: artifacts/stage2_primary_i1.json" in r.stdout and "VERIFIED: artifacts/stage2_s035.json" in r.stdout


def test_c121_fit_never_writes_existing_stage2_files():
    for target in ("artifacts/stage2_s032.json", "artifacts/stage2_s034.json", "artifacts/stage2_s035.json"):
        p = COMP / target
        before = sha(p) if p.is_file() else None
        r = subprocess.run(PY + ["eda/c121_stage2_i1.py", "fit", "--out", target, "--save"], cwd=COMP, env=ENV, capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 2 and "never writes" in r.stderr
        if before is not None:
            assert sha(p) == before


@needs_i1
def test_null_stage2_from_primary_i1_keeps_feature_set():
    C118 = _load("c118_i1_null", "eda/c118_stage2.py")
    nb = C118.null_stage2("s032_stage2", I1)
    assert nb["models"]["main"]["feature_set"] == S2.I1_SET and nb["models"]["main"]["protect_none"] is True
    assert all(v == 0.0 for v in nb["models"]["main"]["w"]) and len(nb["models"]["main"]["w"]) == 63
    assert "feature_set" not in nb["models"]["fallback"]


def test_build_script_primary_i1_fixed_checksums():
    p = COMP / "tools" / "builds" / "primary_i1.sh"
    text = p.read_text()
    for var, path in (("STAGE2_SHA", I1), ("OUT_SHA", COMP / "artifacts/submissions/sub_primary_i1.csv"),
                      ("TESTMAT_SHA", COMP / "artifacts/test_s032_stage2_primary_i1.npy"), ("SPLITS_SHA", ISPL), ("IOOF_SHA", IOOF)):
        m = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        assert m, var
        if Path(path).is_file():
            assert m.group(1) == sha(path), var
    assert 'check "$OUT_SHA" "$OUT"' in text and 'check "$STAGE2_SHA" "$STAGE2"' in text
    assert "s035_stage2_guard.sh" in text and "ORIENT" in text and "c121_stage2_i1.py fit" in text and "c121_stage2_i1.py submit" in text
    assert "tools/make_submission.py" in text  # file without second stage through the unchanged path
