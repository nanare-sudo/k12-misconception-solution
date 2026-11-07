"""Tests of the feature set main_guard (main form + identifiers of the five general codes) with the NONE guard — src/stage2.py,
eda/c118_stage2.py (guard-oof, guard-fit, null), tools/make_submission.py (fields feature_set/protect_none of the deployment file).

Synthetic (fast, without data): FEATURE_SETS unchanged; main_guard = main form bit-identical + identifiers (one-hot) at the end; NONE
guard; gate without the guard exactly as before; make_submission without the new fields as before, with them the feature set and the
guard of the model (a fallback model must not read the notes). With real files (skipped if they are missing): guard-oof (copy null
point (i), main form bit-identical), stage2_s035.json = stage2_s034.json except models.main/guard_build/derived_from, guard-fit
--verify, the null model keeps the feature set, fixed checksums in the build script s035. None of these tests writes to artifacts/."""
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


# ------------------------------------------------------------------------------------------------ synthetic

def test_feature_sets_unchanged_and_guard_names():
    assert S2.FEATURE_SETS == ("main", "fallback") and S2.GUARD_SET == "main_guard"
    main, guard = S2.feature_names(CH, "main"), S2.feature_names(CH, S2.GUARD_SET)
    assert len(main) == 43 and len(guard) == 48
    assert guard[:43] == main and guard[43:] == ["id_ANS-01", "id_CAL-01", "id_WOP-01", "id_FRM-01", "id_OTH-00"]
    assert list(S2.GENERAL_ID_FEATS) == [f"id_{c}" for c in S2.GENERAL_CODES]
    assert S2.uses_notes("main") and S2.uses_notes(S2.GUARD_SET) and not S2.uses_notes("fallback")
    assert S2.NONE_IDX == 0


def test_guard_pair_features_main_prefix_bitwise_ids_onehot_and_zero_model():
    st, Q, P, kn, cls, seen = _toy()
    n = st.n
    Xm, cm, nm = S2.pair_features(st, np.arange(n), Q, P, kn, "main")
    Xg, cg, ng = S2.pair_features(st, np.arange(n), Q, P, kn, S2.GUARD_SET)
    assert np.array_equal(cg, cm) and ng[:43] == nm and Xg[:, :43].tobytes() == Xm.tobytes()
    cc = cg.reshape(-1)
    for j, c in enumerate(S2.GENERAL_CODES):
        assert np.array_equal(Xg[:, 43 + j], (cc == cls.index(c)).astype(float))
    assert Xg[:, 43:].sum(1).max() <= 1.0  # at most one identifier per pair
    top = S2.reorder(cg, S2.scores(S2.zero_model(len(ng)), Xg))
    assert np.array_equal(top, cg) and np.array_equal(S2.protect_none(top, cg), cg)
    st0, Q0, P0, kn0, _, _ = _toy(notes=False)
    with pytest.raises(ValueError, match="requires notes"):
        S2.pair_features(st0, np.arange(st0.n), Q0, P0, kn0, S2.GUARD_SET)
    with pytest.raises(ValueError):
        S2.feature_names(CH, "main_guard2")


def test_protect_none_semantics():
    cand = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [5, 0, 1, 2, 3], [0, 2, 1, 3, 4]])
    top = np.array([[1, 0, 2, 3, 4], [0, 2, 1, 3, 4], [0, 5, 1, 2, 3], [0, 2, 1, 3, 4]])
    got = S2.protect_none(top, cand)
    assert np.array_equal(got[0], cand[0])  # base top NONE, new top not NONE -> base order
    assert np.array_equal(got[1], top[1])  # top stays NONE -> the new order of the other places stays
    assert np.array_equal(got[2], top[2])  # base top not NONE -> unchanged (even if NONE moves to the top)
    assert np.array_equal(got[3], top[3])
    assert np.array_equal(top[0], [1, 0, 2, 3, 4])  # input not modified
    with pytest.raises(ValueError):
        S2.protect_none(top[:, :3], cand)


def _gate_blocks(seed=7, R=600):
    rng = np.random.default_rng(seed)
    Kc = S2.K_CAND
    X = rng.normal(size=(R * Kc, 3))
    rank = np.tile(np.arange(Kc), R).astype(float)
    X[:, 0] = -rank + rng.normal(scale=0.3, size=R * Kc)
    gold = np.minimum(rng.geometric(0.55, size=R) - 1, Kc - 1)
    sig = np.zeros(R * Kc)
    sig[np.arange(R) * Kc + gold] = 1.0
    X[:, 1] = sig + rng.normal(scale=0.6, size=R * Kc)
    cand = np.tile(np.arange(Kc), (R, 1))
    cand[::2, 0], cand[::2, 1] = 1, 0  # every second row: base top not NONE
    lab = S2.labels_of(cand, cand[np.arange(R), gold])
    y = cand[np.arange(R), gold]
    folds = np.arange(R) % 4
    return [(X[np.repeat(folds == k, Kc)], lab[np.repeat(folds == k, Kc)], cand[folds == k], y[folds == k]) for k in range(4)]


def test_gate_default_unchanged_and_protect_option():
    blocks = _gate_blocks()
    a, b = S2.gate(blocks), S2.gate(blocks, protect=False)
    assert a == b
    c = S2.gate(blocks, protect=True)
    assert c["inner_base"] == a["inner_base"] and set(c) == set(a)
    # with the guard the order is evaluated after the guard: recomputation by hand
    got = []
    for k in range(4):
        tr = [j for j in range(4) if j != k]
        m = S2.fit_clogit(np.concatenate([blocks[j][0] for j in tr]), np.concatenate([blocks[j][1] for j in tr]))
        X, _, cand, yr = blocks[k]
        got.append(S2.map_rows(S2.protect_none(S2.reorder(cand, S2.scores(m, X)), cand), yr))
    assert c["inner"] == float(np.concatenate(got).mean())


def test_make_submission_model_form_fields():
    MS = _load("make_submission_guard_form", "tools/make_submission.py")
    cls = classes_toy()
    assert MS._stage2_model_form({"features": []}, "main", cls) == ("main", False)
    assert MS._stage2_model_form({"features": []}, "fallback", cls) == ("fallback", False)
    assert MS._stage2_model_form({"feature_set": S2.GUARD_SET, "protect_none": True}, "main", cls) == (S2.GUARD_SET, True)
    for model, fs, msg in (({"feature_set": "main"}, "fallback", "notes"), ({"feature_set": S2.GUARD_SET}, "fallback", "notes"),
                           ({"feature_set": "everything"}, "main", "unknown"), ({"protect_none": 1}, "main", "boolean")):
        with pytest.raises(SystemExit, match=msg):
            MS._stage2_model_form(model, fs, cls)
    with pytest.raises(SystemExit, match="class 0"):
        MS._stage2_model_form({"protect_none": True}, "main", cls[1:] + cls[:1])


def test_make_submission_apply_stage2_guard_model(tmp_path, monkeypatch):
    """A main_guard model with weight only on id_ANS-01: ANS-01 moves to the top among the first five — except in rows with NONE at the
    base top (guard). Guard (c) counts AFTER the guard; without the fields the report stays without the new keys."""
    T = _load("test_stage2_base_for_guard", "tests/test_stage2.py")
    MS = _load("make_submission_guard_apply", "tools/make_submission.py")
    rule = {"notes_rel_tolerance": 0.10, "changed_top_factor": [0.5, 2.0], "text": "band rule", "rule": "widened", "share_z_max": 3.0, "n_ref": 40}
    args, texts = T._toy_apply_stage2(tmp_path, rule)
    monkeypatch.setattr(MS, "_test_notes_texts", lambda root, f, r, n: (texts[f], {"file": f, "sha256": "x", "rank": r, "rank_sha256": "y"}))
    Q, seen, cls = args[3], args[11], args[10]
    cand = S2.candidates(Q, seen)
    ans = cls.index("ANS-01")
    has = (cand == ans).any(1)
    expect_top = np.where(has, ans, cand[:, 0])
    blocked = has & (cand[:, 0] == 0) & (cand[:, 0] != ans)
    expect_top[blocked] = 0
    changed = float((expect_top != cand[:, 0]).mean())
    assert blocked.sum() > 0 and changed > 0  # the probe checks both
    s2_file = args[0]
    s2 = json.loads(s2_file.read_text())
    names = S2.feature_names(CH, S2.GUARD_SET)
    w = [0.0] * len(names)
    w[names.index("id_ANS-01")] = 5.0
    s2["models"]["main"] = {"features": names, "feature_set": S2.GUARD_SET, "protect_none": True, "w": w, "mu": [0.0] * len(names),
                            "sd": [1.0] * len(names), "oof_changed_top_share": changed, "oof_changed_top3_share": 0.0}
    s2_file.write_text(json.dumps(s2))
    Q_new, info = MS.apply_stage2(*args, report_path=tmp_path / "g.json")
    top_new = S2.candidates(Q_new, seen)[:, 0]
    assert np.array_equal(top_new, expect_top)
    assert info["feature_set"] == "main" and info["model_feature_set"] == S2.GUARD_SET and info["protect_none"] is True
    assert info["rows_protected"] == int(blocked.sum()) and info["n_features"] == 48
    assert info["guard_changed_top"]["test"] == pytest.approx(changed) and info["guard_changed_top"]["ok"]
    rep = json.loads((tmp_path / "g.json").read_text())
    assert rep["rows_protected"] == int(blocked.sum())
    # guard off: ANS-01 at the top there too; guard (c) with the same OOF share would fail (share above 2×?) -> only check the order
    s2["models"]["main"]["protect_none"] = False
    s2["models"]["main"]["oof_changed_top_share"] = float((np.where(has, ans, cand[:, 0]) != cand[:, 0]).mean())
    s2_file.write_text(json.dumps(s2))
    Q_off, info_off = MS.apply_stage2(*args)
    assert np.array_equal(S2.candidates(Q_off, seen)[:, 0], np.where(has, ans, cand[:, 0])) and info_off["rows_protected"] == 0
    # without the new fields: old path, no new keys in the report
    args2, texts2 = T._toy_apply_stage2(tmp_path, rule)
    _, info_old = MS.apply_stage2(*args2)
    assert not {"model_feature_set", "protect_none", "rows_protected"} & set(info_old)


# ------------------------------------------------------------------------------------------------ real files

MAIN_FORM_FILE = COMP / "artifacts" / "stage2_s034.json"
GUARD_FORM_FILE = COMP / "artifacts" / "stage2_s035.json"
GOOF = COMP / "artifacts" / "eda" / "private" / "stage2_guard" / "c118_guard_oof.json"
needs_goof = pytest.mark.skipif(not GOOF.is_file(), reason="c118 guard-oof not computed yet")
needs_guard_form = pytest.mark.skipif(not (GUARD_FORM_FILE.is_file() and MAIN_FORM_FILE.is_file()), reason="stage2_s035.json missing (built by tools/builds/s035_stage2_guard.sh)")


@needs_goof
def test_guard_oof_null_point_and_main_reproduction():
    g = json.loads(GOOF.read_text())
    assert g["null_point_i"]["ok"] and all(g["null_point_i"]["forms"].values())
    o150 = json.loads((COMP / "artifacts/eda/private/stage2/c118_oof.json").read_text())
    for f in ("fold", "pre", "set"):
        assert g["sets"]["main"][f]["map_forced"] == o150["sets"]["main"][f]["map_forced"]
        assert g["sets"]["main_guard"][f]["n_features"] == 48
    assert g["sets"]["main"]["set"]["eval_features_sha256"] == o150["sets"]["main"]["set"]["eval_features_sha256"]
    assert g["guard_set"] == S2.GUARD_SET and g["protect_none"] is True


@needs_guard_form
def test_stage2_s035_equals_s034_except_main():
    a, b = json.loads(MAIN_FORM_FILE.read_text()), json.loads(GUARD_FORM_FILE.read_text())
    assert [k for k in b if k not in ("guard_build", "derived_from")] == [k for k in a if k != "derived_from"]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {"models", "guard_build", "derived_from"}
    assert b["models"]["fallback"] == a["models"]["fallback"] and list(b["models"]) == ["main", "fallback"]
    m = b["models"]["main"]
    assert m["feature_set"] == S2.GUARD_SET and m["protect_none"] is True and len(m["w"]) == 48
    assert m["features"] == a["models"]["main"]["features"] + list(S2.GENERAL_ID_FEATS)
    assert b["derived_from"]["file"] == "artifacts/stage2_s034.json" and b["derived_from"]["sha256"] == sha(MAIN_FORM_FILE)
    assert b["derived_from"]["previous"] == a["derived_from"]
    g = json.loads(GOOF.read_text())
    assert m["oof_changed_top_share"] == g["sets"]["main_guard"]["set"]["changed_top_share_forced"]


@needs_guard_form
def test_c118_guard_fit_verify_reproduces_s035():
    r = subprocess.run(PY + ["eda/c118_stage2.py", "guard-fit", "--stack", "s032_stage2", "--verify"], cwd=COMP, env=ENV, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERIFIED: artifacts/stage2_s035.json" in r.stdout


def test_c118_guard_fit_never_writes_s032_or_s034():
    for target in ("artifacts/stage2_s032.json", "artifacts/stage2_s034.json"):
        p = COMP / target
        before = sha(p) if p.is_file() else None
        r = subprocess.run(PY + ["eda/c118_stage2.py", "guard-fit", "--out", target, "--verify"], cwd=COMP, env=ENV, capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 2 and "never writes" in r.stderr
        if before is not None:
            assert sha(p) == before


@needs_guard_form
def test_null_stage2_keeps_feature_set_only_when_present():
    C118 = _load("c118_guard_null", "eda/c118_stage2.py")
    nb = C118.null_stage2("s032_stage2", GUARD_FORM_FILE)
    assert nb["models"]["main"]["feature_set"] == S2.GUARD_SET and nb["models"]["main"]["protect_none"] is True
    assert all(v == 0.0 for v in nb["models"]["main"]["w"]) and len(nb["models"]["main"]["w"]) == 48
    assert "feature_set" not in nb["models"]["fallback"]
    na = C118.null_stage2("s032_stage2", MAIN_FORM_FILE)
    assert all(not ({"feature_set", "protect_none"} & set(na["models"][fs])) for fs in S2.FEATURE_SETS)
    assert list(na["models"]["main"]) == ["features", "w", "mu", "sd", "abstain", "oof_changed_top_share"]


def test_build_script_s035_fixed_checksums():
    p = COMP / "tools" / "builds" / "s035_stage2_guard.sh"
    if not p.is_file():
        pytest.skip("build script s035 missing")
    text = p.read_text()
    m = re.search(r'\bSTAGE2_SHA="([0-9a-f]{64})"', text)
    assert m and "STAGE2_SHA:-" not in text
    if GUARD_FORM_FILE.is_file():
        assert m.group(1) == sha(GUARD_FORM_FILE)
    out = COMP / "artifacts" / "submissions" / "sub_s035_stage2_guard.csv"
    mo = re.search(r'\bOUT_SHA="([0-9a-f]{64})"', text)
    assert mo and 'check "$OUT_SHA" "$OUT"' in text
    if out.is_file():
        assert mo.group(1) == sha(out)
    assert "s034n_stage2_main.sh" in text and "ORIENT" in text and "guard-fit" in text
