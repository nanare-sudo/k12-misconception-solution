"""Tests of the feature set main_guard_i1_mm (main_guard_i1 + minor_x_mech27) — src/stage2.py, tools/make_submission.py (admission for
the main model, test answers with checksum, column on the test side), eda/c124_stage2_mm.py and the build script
tools/builds/primary_final.sh. Synthetic; with real files (skipped if they are missing): null point, stage2_primary_final.json =
stage2_primary_i1.json except models.main/mm_build/mech27/derived_from, fixed checksums. None of these tests writes to artifacts/."""
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


def _toy(n=60, seed=1):
    T = _load("test_stage2_i1_for_mm", "tests/test_stage2_i1.py")
    return T._toy(n=n, seed=seed)


# ------------------------------------------------------------------------------------------------ src/stage2.py

def test_mm_names_and_existing_sets_unchanged():
    assert S2.MM_SET == "main_guard_i1_mm" and S2.MM_FEATS == ("minor_x_mech27",) and S2.MM_SET not in S2.FEATURE_SETS
    i1, mm = S2.feature_names(CH, S2.I1_SET), S2.feature_names(CH, S2.MM_SET)
    assert len(i1) == 63 and mm == i1 + ["minor_x_mech27"] and S2.uses_notes(S2.MM_SET)
    assert [len(S2.feature_names(CH, fs)) for fs in ("main", "fallback", S2.GUARD_SET, S2.I1_SET, S2.FB_I1_SET)] == [43, 40, 48, 63, 55]


def test_mm_pair_features_prefix_bitwise_and_column():
    st, Q, P, kn, cls, seen = _toy()
    rows = np.arange(st.n)
    with pytest.raises(ValueError, match="attach_mech"):
        S2.pair_features(st, rows, Q, P, kn, S2.MM_SET)
    rng = np.random.default_rng(3)
    mech = (rng.random(st.n) < 0.4).astype(float)
    minor = np.zeros(st.K)
    minor[cls.index("CAL-01")] = 1.0
    S2.attach_mech(st, mech, minor)
    Xi, ci, ni = S2.pair_features(st, rows, Q, P, kn, S2.I1_SET)
    Xm, cm, nm = S2.pair_features(st, rows, Q, P, kn, S2.MM_SET)
    assert np.array_equal(cm, ci) and nm == ni + ["minor_x_mech27"] and Xm[:, :63].tobytes() == Xi.tobytes()
    want = np.repeat(mech, S2.K_CAND) * minor[ci.reshape(-1)]
    assert Xm[:, 63].tobytes() == want.tobytes() and want.sum() > 0
    sub = np.arange(10, 30)
    Xs, cs, _ = S2.pair_features(st, sub, Q[sub], P[sub], {k: (v[sub] if np.ndim(v) and len(v) == st.n else v) for k, v in kn.items()}, S2.MM_SET)
    assert Xs[:, 63].tobytes() == (np.repeat(mech[sub], S2.K_CAND) * minor[cs.reshape(-1)]).tobytes()  # rows of the table, not of the block


def test_attach_mech_validates():
    st, *_ = _toy(n=20)
    with pytest.raises(ValueError):
        S2.attach_mech(st, np.zeros(19), np.zeros(st.K))
    with pytest.raises(ValueError):
        S2.attach_mech(st, np.full(20, 0.5), np.zeros(st.K))
    with pytest.raises(ValueError):
        S2.attach_mech(st, np.zeros(20), np.zeros(st.K - 1))


def test_mm_zero_weight_gives_i1_ranking():
    st, Q, P, kn, cls, seen = _toy(n=80, seed=2)
    rows = np.arange(st.n)
    S2.attach_mech(st, (np.arange(st.n) % 3 == 0).astype(float), np.eye(st.K)[cls.index("CAL-01")])
    Xi, ci, _ = S2.pair_features(st, rows, Q, P, kn, S2.I1_SET)
    Xm, cm, _ = S2.pair_features(st, rows, Q, P, kn, S2.MM_SET)
    rng = np.random.default_rng(5)
    mi = {"w": rng.normal(size=63), "mu": Xi.mean(0), "sd": Xi.std(0) + 1e-9}
    mm = {"w": np.r_[mi["w"], 0.0], "mu": np.r_[mi["mu"], 0.0], "sd": np.r_[mi["sd"], 1.0]}
    assert np.array_equal(S2.reorder(cm, S2.scores(mm, Xm)), S2.reorder(ci, S2.scores(mi, Xi)))


# ------------------------------------------------------------------------------------------------ tools/make_submission.py

def test_make_submission_admits_mm_for_main_only():
    MS = _load("make_submission_mm_form", "tools/make_submission.py")
    cls = ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "Dec-01", "PV-01", "ALG-06", "FRA-02"]
    assert MS._stage2_model_form({"feature_set": S2.MM_SET, "protect_none": True}, "main", cls) == (S2.MM_SET, True)
    with pytest.raises(SystemExit, match="reads the notes"):
        MS._stage2_model_form({"feature_set": S2.MM_SET}, "fallback", cls)


def _answers_file(tmp_path, answers: list, name="proto_te", table="test") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [{"row": i, "phase": "first", "text": json.dumps({"student_final_answer": a})} for i, a in enumerate(answers) if a is not None]
    (d / "proto_raw.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (d / "proto_meta.json").write_text(json.dumps({"table": table}))
    return d / "proto_raw.jsonl"


def _taxonomy(tmp_path, cls, minor=("CAL-01",)):
    (tmp_path / "competition_data").mkdir(parents=True, exist_ok=True)
    rows = ["id,default_severity"] + [f"{c},{'minor' if c in minor else 'major'}" for c in cls if c != "NONE"]
    (tmp_path / "competition_data" / "taxonomy_v1.csv").write_text("\n".join(rows) + "\n")


def test_make_submission_test_mech27_checks(tmp_path):
    import pandas as pd

    MS = _load("make_submission_mm_mech", "tools/make_submission.py")
    cls = ["NONE", "ANS-01", "CAL-01", "WOP-01"]
    _taxonomy(tmp_path, cls)
    test = pd.DataFrame({"correct_answer": ["3", "1/2", "7", "x"]})
    f = _answers_file(tmp_path, ["3", "0,5", "8", None])
    s2 = {"mech27": {"test": {"file": str(f), "sha256": sha(f)}, "minor_codes": ["CAL-01"], "train_share_mech": 0.3}}
    mech, minor, info = MS._test_mech27(s2, test, cls, tmp_path)
    assert mech.tolist() == [1.0, 1.0, 0.0, 0.0] and minor.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert info["share_mech"] == 0.5 and info["share_with_answer"] == 0.75 and info["minor_codes"] == 1
    for bad, msg in (({"test": {"file": str(f), "sha256": "0" * 64}, "minor_codes": ["CAL-01"]}, "SHA-256"),
                     ({"test": {"file": str(f), "sha256": sha(f)}, "minor_codes": ["CAL-01", "ANS-01"]}, "minor codes"),
                     ({"test": {"file": str(tmp_path / "missing.jsonl"), "sha256": "x"}, "minor_codes": []}, "missing")):
        with pytest.raises(SystemExit, match=msg):
            MS._test_mech27({"mech27": bad}, test, cls, tmp_path)
    g = _answers_file(tmp_path, ["3"] * 4, name="proto_tr", table="train")
    with pytest.raises(SystemExit, match="table test"):
        MS._test_mech27({"mech27": {"test": {"file": str(g), "sha256": sha(g)}, "minor_codes": ["CAL-01"]}}, test, cls, tmp_path)


def test_make_submission_apply_stage2_mm_column(tmp_path, monkeypatch):
    """Test side: a main_guard_i1_mm model with weight only on minor_x_mech27 moves CAL-01 to the end of the first five in rows with a
    mechanically correct final answer, otherwise the order stays; with the weights of an I1 model and column weight 0 bit-identical to I1."""
    T = _load("test_stage2_base_for_mm", "tests/test_stage2.py")
    MS = _load("make_submission_mm_apply", "tools/make_submission.py")
    rule = {"notes_rel_tolerance": 0.10, "changed_top_factor": [0.5, 2.0], "text": "band rule", "rule": "widened", "share_z_max": 3.0, "n_ref": 40}
    args, texts = T._toy_apply_stage2(tmp_path, rule)
    monkeypatch.setattr(MS, "_test_notes_texts", lambda root, f, r, n: (texts[f], {"file": f, "sha256": "x", "rank": r, "rank_sha256": "y"}))
    args = list(args)
    test, cls, seen, Q = args[9].copy(), args[10], args[11], args[3]
    n = len(test)
    keys = [str(i % 5) for i in range(n)]
    test["correct_answer"] = keys
    args[9] = test
    mech_true = np.array([i % 2 == 0 for i in range(n)], float)
    f = _answers_file(tmp_path, [k if m else "999" for k, m in zip(keys, mech_true)])
    _taxonomy(tmp_path, cls)
    s2_file = args[0]
    s2 = json.loads(s2_file.read_text())
    s2["mech27"] = {"test": {"file": str(f), "sha256": sha(f)}, "minor_codes": ["CAL-01"], "train_share_mech": 0.5}
    ni, nm = S2.feature_names(T.SIX_CHANNELS, S2.I1_SET), S2.feature_names(T.SIX_CHANNELS, S2.MM_SET)
    w = [0.0] * len(nm)
    w[-1] = -5.0
    s2["null"] = True  # guard (c) not enforced; the order is what counts here
    s2["models"]["main"] = {"features": nm, "feature_set": S2.MM_SET, "protect_none": False, "w": w, "mu": [0.0] * len(nm), "sd": [1.0] * len(nm),
                            "oof_changed_top_share": 0.0, "oof_changed_top3_share": 0.0}
    s2_file.write_text(json.dumps(s2))
    Q_new, info = MS.apply_stage2(*args, report_path=tmp_path / "g.json")
    cand = S2.candidates(Q, seen)
    cal = cls.index("CAL-01")
    want = cand.copy()
    for i in range(n):
        if mech_true[i] and cal in cand[i]:
            want[i] = np.r_[cand[i][cand[i] != cal], cal]
    assert np.array_equal(S2.candidates(Q_new, seen), want) and not np.array_equal(want, cand)
    assert info["model_feature_set"] == S2.MM_SET and info["n_features"] == 64 and info["mech27"]["share_mech"] == 0.5
    assert json.loads((tmp_path / "g.json").read_text())["mech27"]["minor_codes"] == 1
    # column weight 0 with the weights of an I1 model -> bit-identical to the I1 model
    rng = np.random.default_rng(1)
    wi = list(rng.normal(size=len(ni)))
    s2["models"]["main"] = {"features": ni, "feature_set": S2.I1_SET, "protect_none": True, "w": wi, "mu": [0.0] * len(ni), "sd": [1.0] * len(ni),
                            "oof_changed_top_share": 0.0, "oof_changed_top3_share": 0.0}
    s2_file.write_text(json.dumps(s2))
    Qi, _ = MS.apply_stage2(*args)
    s2["models"]["main"] = {**s2["models"]["main"], "features": nm, "feature_set": S2.MM_SET, "w": wi + [0.0], "mu": [0.0] * len(nm), "sd": [1.0] * len(nm)}
    s2_file.write_text(json.dumps(s2))
    Qm, _ = MS.apply_stage2(*args)
    assert Qm.tobytes() == Qi.tobytes()


# ------------------------------------------------------------------------------------------------ real files

PRIV = COMP / "artifacts" / "eda" / "private" / "stage2_mm"
I1 = COMP / "artifacts" / "stage2_primary_i1.json"
MMF = COMP / "artifacts" / "stage2_primary_final.json"
OOF = PRIV / "c124_mm_oof.json"
needs_oof = pytest.mark.skipif(not OOF.is_file(), reason="c124 oof not computed yet")
needs_c118_oof = pytest.mark.skipif(not (COMP / "artifacts" / "eda" / "private" / "stage2" / "c118_oof.json").is_file(),
                                    reason="report c118_oof.json missing (written by the build path, not part of the frozen package)")
needs_mm = pytest.mark.skipif(not (MMF.is_file() and I1.is_file()), reason="stage2_primary_final.json missing (only after the pre-registered rule is met)")


@needs_oof
def test_c124_oof_null_point_and_reproduction():
    g = json.loads(OOF.read_text())
    assert g["null_point_i"]["ok"] and all(g["null_point_i"]["forms"].values()) and g["feature_set"] == S2.MM_SET
    assert all(g["sets"]["i1"][f]["rows_equal_reference"] for f in ("fold", "pre", "set"))
    assert g["sets"]["mm"]["set"]["features"][-1] == "minor_x_mech27" and g["placebo"]["n"] == 15 and len(g["minor_codes"]) == 9


@needs_mm
def test_stage2_primary_final_equals_primary_i1_except_main():
    a, b = json.loads(I1.read_text()), json.loads(MMF.read_text())
    extra = ("mm_build", "mech27", "derived_from")
    assert [k for k in b if k not in extra] == [k for k in a if k != "derived_from"]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {"models", *extra}
    assert b["models"]["fallback"] == a["models"]["fallback"] and list(b["models"]) == ["main", "fallback"]
    m, am = b["models"]["main"], a["models"]["main"]
    assert m["feature_set"] == S2.MM_SET and m["protect_none"] is True and m["features"] == am["features"] + ["minor_x_mech27"] and not m["abstain"]
    assert b["derived_from"]["file"] == "artifacts/stage2_primary_i1.json" and b["derived_from"]["sha256"] == sha(I1)
    t = b["mech27"]["test"]
    assert t["file"].endswith("pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl") and t["sha256"] == sha(COMP / t["file"])
    assert b["mm_build"]["rule_oof"]["verdict_oof"] == "deploy" and len(b["mech27"]["minor_codes"]) == 9


@needs_mm
@needs_c118_oof
def test_c124_fit_verify_reproduces_primary_final():
    r = subprocess.run(PY + ["eda/c124_stage2_mm.py", "fit", "--verify"], cwd=COMP, env=ENV, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERIFIED: artifacts/stage2_primary_final.json" in r.stdout and "VERIFIED: artifacts/stage2_primary_i1.json" in r.stdout


def test_c124_fit_never_writes_existing_or_probe_outside(tmp_path):
    for target in ("artifacts/stage2_s035.json", "artifacts/stage2_primary_i1.json"):
        p = COMP / target
        before = sha(p) if p.is_file() else None
        r = subprocess.run(PY + ["eda/c124_stage2_mm.py", "fit", "--out", target, "--save"], cwd=COMP, env=ENV, capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 2 and "never writes" in r.stderr and (sha(p) if p.is_file() else None) == before
    r = subprocess.run(PY + ["eda/c124_stage2_mm.py", "fit", "--probe", "--out", str(tmp_path / "x_probe.json"), "--save"], cwd=COMP, env=ENV,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "only writes under" in r.stderr


def test_build_script_primary_final_pins():
    p = COMP / "tools" / "builds" / "primary_final.sh"
    text = p.read_text()
    for var, path in (("PARSER_SHA", PRIV / "c124_parser.json"), ("IOOF_SHA", PRIV / "c124_mm_oof.json"), ("ISPL_SHA", PRIV / "c124_mm_splits.json"),
                      ("STAGE2_SHA", MMF), ("STAGE2_I1_SHA", I1), ("OUT_SHA", COMP / "artifacts" / "submissions" / "sub_primary_final.csv")):
        m = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        assert m, var
        if Path(path).is_file():
            assert m.group(1) == sha(path), var
    assert "bash tools/builds/primary_i1.sh" in text and "c124_stage2_mm.py rule" in text and "tools/make_submission.py" in text
    assert "PRIMARY_FINAL_PROBE" in text and "c121_stage2_i1.py submit" not in text
    i1 = re.search(r'\bSTAGE2_SHA="([0-9a-f]{64})"', (COMP / "tools" / "builds" / "primary_i1.sh").read_text())
    assert i1 and re.search(rf'\bSTAGE2_I1_SHA="{i1.group(1)}"', text)  # the same pin of stage2_primary_i1.json in both build scripts
