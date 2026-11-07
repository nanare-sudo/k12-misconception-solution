"""Tests of the safety line with margin alignment over the test set — tools/make_submission.py (align_mode "set" in apply_stage2_safe,
statistic class_mass, dispatch in main), eda/c123_safety_set.py (rule, probe, placebo permutation, override by the decision file) and the
build script tools/builds/safety_final.sh.

Synthetic: the new mode computes the same path as "inductive" (same order, same guard) and also reports the class masses; files with
"inductive" keep their report without new keys; wrong modes are rejected; the command line requires --align for "set" as in the file.
The safety-set rule on artificial results. With real files (skipped if they are missing): null points and numbers of the reproduction,
test masses (control against an independent check), fixed checksums in the build script. Override of the CV condition of (ii) only
through the pinned decision file decisions/override_cv_condition_ii.json (only that condition; L1 and null points stay in force; without
the switch an unchanged return value) and the deployment file of variant b. None of these tests writes to artifacts/."""
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
PRIV = COMP / "artifacts" / "eda" / "private" / "safety_set"
DECISION = "override_cv_condition_ii"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ make_submission: align_mode "set"

def _safe_args(tmp_path, **over):
    T = _load("test_stage2_safe_for_safety_set", "tests/test_stage2_safe.py")
    return T._toy_safe(tmp_path, **over)


def test_apply_stage2_safe_set_mode_same_order_and_class_mass(tmp_path):
    MS = _load("make_submission_safety_set", "tools/make_submission.py")
    ind = _safe_args(tmp_path, w_scale=10.0, oof_share=0.7)
    Q_ind, info_ind = MS.apply_stage2_safe(*ind, report_path=tmp_path / "g_ind.json")
    st = _safe_args(tmp_path, w_scale=10.0, oof_share=0.7, align_mode="set", align=0.75)
    Q_set, info_set = MS.apply_stage2_safe(*st, report_path=tmp_path / "g_set.json")
    assert Q_set.tobytes() == Q_ind.tobytes()  # same path, same order, same guard
    assert info_set["align_mode"] == "set" and info_ind["align_mode"] == "inductive"
    assert "class_mass" not in info_ind and "class_mass" in info_set
    assert set(info_set) - set(info_ind) == {"class_mass"}
    assert info_set["order"].startswith("blend -> margin alignment (one set") and "inductive" in info_ind["order"]
    # statistic: L1 of the mean test mass per class against the training shares of the classes with at least align_min rows
    Q, P_pre, train, cls = st[3], st[4], st[8], st[10]
    import labels

    y = labels.encode(train["root_code"], cls)
    counts = np.bincount(y, minlength=len(cls))
    target = np.where(counts >= 3, counts, 0) / len(y)
    used = target > 0
    cm = info_set["class_mass"]
    assert cm["l1_after"] == pytest.approx(float(np.abs(Q.mean(0) - target)[used].sum()), abs=1e-12)
    assert cm["l1_before"] == pytest.approx(float(np.abs(P_pre.mean(0) - target)[used].sum()), abs=1e-12)
    assert cm["classes"] == int(used.sum()) and cm["none_target"] == pytest.approx(float(target[0]))
    assert json.loads((tmp_path / "g_set.json").read_text())["class_mass"]["l1_after"] == pytest.approx(cm["l1_after"])


def test_apply_stage2_safe_inductive_report_keys_unchanged(tmp_path):
    """The report of an inductive file has exactly the keys of before (the strictly inductive line stays byte-identical down to the statistics file)."""
    MS = _load("make_submission_safety_keys", "tools/make_submission.py")
    _, info = MS.apply_stage2_safe(*_safe_args(tmp_path))
    assert list(info) == ["file", "sha256", "null", "align_mode", "feature_set", "fallback", "n_features", "K", "lambda", "protect_none",
                          "rows_protected", "guard_notes", "guard_novel_task_top5", "guard_changed_top", "share_top3_changed",
                          "share_top3_changed_oof", "rows_tie_fixed", "test_draws", "order"]
    assert info["align_mode"] == "inductive" and info["order"] == "blend -> inductive alignment -> second stage -> NONE guard -> REL-01"


@pytest.mark.parametrize("mode", ["trans", "Set", None])
def test_apply_stage2_safe_rejects_other_modes(tmp_path, mode):
    MS = _load("make_submission_safety_reject", "tools/make_submission.py")
    with pytest.raises(SystemExit, match="align_mode"):
        MS.apply_stage2_safe(*_safe_args(tmp_path, align_mode=mode))


def test_set_mode_wrong_feature_set_rejected_with_label(tmp_path):
    MS = _load("make_submission_safety_fs", "tools/make_submission.py")
    with pytest.raises(SystemExit, match=r"\(transductive\).*fallback"):
        MS.apply_stage2_safe(*_safe_args(tmp_path, align_mode="set", align=0.75, feature_set="main"))


def _cli(tmp_path, s2: dict, *extra) -> subprocess.CompletedProcess:
    f = tmp_path / "s2.json"
    f.write_text(json.dumps(s2))
    return subprocess.run(PY + ["tools/make_submission.py", "--name", "__missing__", "--stage2", str(f), *extra], cwd=COMP, env=ENV,
                          capture_output=True, text=True, timeout=120)


def test_cli_set_file_requires_align_like_file(tmp_path):
    s = {"align_mode": "set", "align": 0.75, "align_min": 3}
    for extra in (("--align-inductive", "0.75", "--align-min", "3"), ("--align", "0.5", "--align-min", "3"), ("--align", "0.75", "--align-min", "2")):
        r = _cli(tmp_path, s, *extra)
        assert r.returncode != 0 and "--stage2 requires --align 0.75 --align-min 3 (transductive, as in the fit)" in r.stderr
    r = _cli(tmp_path, s, "--align", "0.75", "--align-min", "3")  # check passed; fails only at the missing blend file
    assert r.returncode != 0 and "requires" not in r.stderr and "stack___missing__.json" in r.stderr


def test_main_dispatches_set_mode_to_safe_path():
    text = (COMP / "tools" / "make_submission.py").read_text()
    assert 'apply_fn = apply_stage2_safe if s2j.get("align_mode") in ("inductive", "set") else apply_stage2' in text


# ------------------------------------------------------------------------------------------------ c123: rule and building blocks

def _oof(cv_ok=True, b_A=0.0014, pl_mean=0.0005, n_pl=15, null_ok=True):
    return {"null_point": {"a_zero_model": {"ok": null_ok}, "b_i1_zero": {"ok": True}},
            "cv_rule": {"a_M_gated": 0.822 if cv_ok else 0.8212, "inductive_F": 0.823333, "margin": 0.002, "threshold": 0.821333, "ok": cv_ok},
            "b_vs_a": {"A": b_A}, "placebo": {"n": n_pl, "A_mean": pl_mean}}


def _splits(mean=0.0007, n=20):
    return {"splits": [{}] * n, "summary": {"b_minus_a": {"A_mean": mean, "A_se": 0.0001}}}


def test_c123_rule_logic():
    M = _load("c123_rule_test", "eda/c123_safety_set.py")
    r = M.rule_result(_oof(cv_ok=False), _splits(), {"l1_after": 0.028})
    assert r["verdict"] == "no deployment" and r["variant"] == "b" and r["ii_l1"]["ok"] is True
    r = M.rule_result(_oof(), _splits(), {"class_mass": {"l1_after": 0.028}})
    assert r["verdict"] == "deploy b"
    r = M.rule_result(_oof(), _splits(mean=-0.0001), {"l1_after": 0.028})
    assert r["verdict"] == "deploy a" and r["variant"] == "a"
    r = M.rule_result(_oof(b_A=0.0004, pl_mean=0.0005), _splits(), {"l1_after": 0.028})
    assert r["variant"] == "a"  # not above the placebo mean
    r = M.rule_result(_oof(n_pl=4), _splits(), {"l1_after": 0.028})
    assert r["variant"] == "a"  # too few placebos
    r = M.rule_result(_oof(), _splits(n=9), {"l1_after": 0.028})
    assert r["variant"] == "a"  # too few meta-splits
    r = M.rule_result(_oof(), _splits(), {"l1_after": 0.036})
    assert r["verdict"] == "no deployment" and r["ii_l1"]["ok"] is False
    r = M.rule_result(_oof(), _splits(), None)
    assert r["verdict"].startswith("deploy b (OOF part") and r["ii_l1"]["ok"] is None
    r = M.rule_result(_oof(null_ok=False), _splits(), {"l1_after": 0.028})
    assert r["verdict"] == "no deployment"
    keys = {"rule", "i_oof", "i_file", "ii_cv", "ii_l1", "iii_b", "variant", "verdict"}
    assert set(r) == keys and "override" not in r["ii_cv"]  # without override: return value unchanged (probe files byte-identical)


def test_c123_rule_override_only_cv_condition_of_ii():
    M = _load("c123_rule_override_test", "eda/c123_safety_set.py")
    ovr = {"decision": DECISION, "status": "active", "title": "t", "scope": "s"}
    r = M.rule_result(_oof(cv_ok=False), _splits(), {"l1_after": 0.028}, ovr)
    assert r["verdict"] == f"deploy b — (ii) missed, overridden by {DECISION}" and r["variant"] == "b"
    assert r["ii_cv"]["ok"] is False and r["ii_cv"]["ok_effective"] is True and r["ii_cv"]["override"] == DECISION
    assert r["override_ii"]["applied"] is True and r["override_ii"]["decision"] == DECISION
    r = M.rule_result(_oof(cv_ok=False), _splits(), None, ovr)  # as in fit: L1 of the test side open
    assert r["verdict"].startswith("deploy b (OOF part") and r["verdict"].endswith(f"overridden by {DECISION}")
    r = M.rule_result(_oof(cv_ok=False), _splits(), {"l1_after": 0.036}, ovr)  # test masses stay in force
    assert r["verdict"] == "no deployment" and r["ii_l1"]["ok"] is False
    r = M.rule_result(_oof(cv_ok=False, null_ok=False), _splits(), {"l1_after": 0.028}, ovr)  # null points stay in force
    assert r["verdict"] == "no deployment"
    r = M.rule_result(_oof(cv_ok=False, b_A=0.0004), _splits(), {"l1_after": 0.028}, ovr)  # (iii) stays in force: then a
    assert r["verdict"].startswith("deploy a — (ii) missed") and r["variant"] == "a"
    r = M.rule_result(_oof(), _splits(), {"l1_after": 0.028}, ovr)  # (ii) met: nothing to override
    assert r["verdict"] == "deploy b" and r["override_ii"]["applied"] is False and r["ii_cv"]["ok_effective"] is True


def _decision(root, rec):
    d = root / "decisions"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{rec['id']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def test_c123_override_record_bound_to_pinned_decision_file(tmp_path, monkeypatch):
    """The override is carried by the versioned decision file, only with the pinned checksum, active and naming the rule."""
    M = _load("c123_override_record_file_test", "eda/c123_safety_set.py")
    with pytest.raises(SystemExit) as e:
        M.override_record("other_decision")
    assert e.value.code == 2
    ok = {"id": DECISION, "status": "active", "rule": M.RULE_NAME, "title": "build despite (ii)"}
    root = tmp_path / "pkg"
    p = _decision(root, ok)
    monkeypatch.setattr(M, "COMP", root)
    monkeypatch.setattr(M, "DECISION_SHA", {DECISION: hashlib.sha256(p.read_bytes()).hexdigest()})
    rec = M.override_record(DECISION)
    assert rec == {"decision": DECISION, "status": "active", "title": ok["title"], "scope": rec["scope"]}
    assert "only the CV condition" in rec["scope"]
    # tampered file (checksum differs) -> abort 5
    p.write_text(p.read_text() + " ", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        M.override_record(DECISION)
    assert e.value.code == 5
    # file with the pinned checksum but not active, of another rule or of another id -> abort 5
    for bad in ({**ok, "status": "withdrawn"}, {**ok, "rule": "another rule"}):
        q = _decision(root, bad)
        monkeypatch.setattr(M, "DECISION_SHA", {DECISION: hashlib.sha256(q.read_bytes()).hexdigest()})
        with pytest.raises(SystemExit) as e:
            M.override_record(DECISION)
        assert e.value.code == 5
    q = root / "decisions" / f"{DECISION}.json"
    q.write_text(json.dumps({**ok, "id": "another_id"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(M, "DECISION_SHA", {DECISION: hashlib.sha256(q.read_bytes()).hexdigest()})
    with pytest.raises(SystemExit) as e:
        M.override_record(DECISION)
    assert e.value.code == 5
    # no decision file -> abort 3
    q.unlink()
    with pytest.raises(SystemExit) as e:
        M.override_record(DECISION)
    assert e.value.code == 3


def test_c123_decision_file_in_repo_matches_pin():
    """The versioned decision file in the repository hits the pinned checksum, is active and names the rule."""
    M = _load("c123_decision_repo_test", "eda/c123_safety_set.py")
    p = COMP / "decisions" / f"{DECISION}.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == M.DECISION_SHA[DECISION]
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["id"] == DECISION and d["status"] == "active" and d["rule"] == M.RULE_NAME and d["decided_on"] == "2026-11-06"
    assert "only the cross-validation part of condition (ii)" in d["scope"]
    rec = M.override_record(DECISION)
    assert rec["decision"] == DECISION and rec["status"] == "active"


def test_c123_fit_override_refused_with_probe_or_other_decision(tmp_path):
    never = PRIV / "probe" / "_test_never_written.json"
    r = subprocess.run(PY + ["eda/c123_safety_set.py", "fit", "--probe", "--override-ii", DECISION, "--out", str(never), "--save"], cwd=COMP,
                       env=ENV, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "mutually exclusive" in r.stderr and not never.exists()
    r = subprocess.run(PY + ["eda/c123_safety_set.py", "fit", "--override-ii", "other_decision", "--out", str(tmp_path / "x.json"), "--save"], cwd=COMP,
                       env=ENV, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "invalid choice" in r.stderr and not (tmp_path / "x.json").exists()


def test_c123_block_perm_matches_c121():
    M = _load("c123_perm_test", "eda/c123_safety_set.py")
    C121 = _load("c121_perm_test", "eda/c121_stage2_i1.py")
    for R, seed in ((300, 5000), (300, 934), (7, 1)):
        assert np.array_equal(M.block_perm(R, S2.K_CAND, seed), C121.block_perm(R, S2.K_CAND, seed))
    assert M.PLACEBO_SEEDS == C121.PLACEBO_SEEDS


def test_c123_fit_probe_only_private(tmp_path):
    r = subprocess.run(PY + ["eda/c123_safety_set.py", "fit", "--probe", "--out", str(tmp_path / "x.json"), "--save"], cwd=COMP, env=ENV,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "only writes under" in r.stderr
    for target in ("artifacts/stage2_s036_safe.json", "artifacts/stage2_primary_i1.json"):
        p = COMP / target
        before = sha(p) if p.is_file() else None
        r = subprocess.run(PY + ["eda/c123_safety_set.py", "fit", "--out", target, "--save"], cwd=COMP, env=ENV, capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 2 and "never writes" in r.stderr
        assert (sha(p) if p.is_file() else None) == before


# ------------------------------------------------------------------------------------------------ real files

OOF = PRIV / "c123_oof.json"
needs_oof = pytest.mark.skipif(not OOF.is_file(), reason="c123 oof not computed yet")


@needs_oof
def test_c123_oof_null_points_and_reference():
    g = json.loads(OOF.read_text())
    assert g["null_point"]["a_zero_model"]["ok"] and g["null_point"]["b_i1_zero"]["ok"]
    assert g["inductive_units"] == 7410 and g["feature_set"] == "fallback" and g["protect_none"] is True and g["align"] == 0.75
    for f in ("fold", "pre", "set"):
        assert g["sets"]["a"][f]["n_features"] == 40 and g["sets"]["b"][f]["n_features"] == 55
        assert g["sets"]["b"][f]["null_i1_zero_equals_a"]
    assert g["sets"]["b"]["set"]["features"][40:] == list(S2.I1_FEATS)
    assert g["cv_rule"]["threshold"] == pytest.approx(7410 / 9000 - 0.002)
    assert g["cv_rule"]["ok"] == (g["sets"]["a"]["set"]["map_gated"] >= g["cv_rule"]["threshold"] - 1e-12)
    assert g["class_mass"]["M_set"]["l1"] < 0.035 and g["class_mass"]["oL"]["l1"] > g["class_mass"]["M_set"]["l1"]


@pytest.mark.skipif(not (PRIV / "c123_test_mass_inductive.json").is_file(), reason="c123 test-mass not computed yet")
def test_c123_test_mass_control_matches_independent_check():
    ind = json.loads((PRIV / "c123_test_mass_inductive.json").read_text())
    st = json.loads((PRIV / "c123_test_mass_set.json").read_text())
    assert abs(ind["l1_after"] - 0.087) < 0.001 and abs(ind["none_after"] - 0.343) < 0.0006  # independent check: 0.087 / 0.343
    assert st["align_mode"] == "set" and st["rows"] == 3498 and st["classes_with_target"] == 53 and st["l1_after"] <= 0.035


@needs_oof
def test_c123_fit_refuses_without_rule(tmp_path):
    g = json.loads(OOF.read_text())
    if g["cv_rule"]["ok"]:
        pytest.skip("rule (ii) met — this test only checks the case without deployment")
    deploy = COMP / "artifacts" / "stage2_safety_final.json"
    before = sha(deploy) if deploy.is_file() else None
    for target in (tmp_path / "x.json", None):  # own output and default target: without the override never a file
        r = subprocess.run(PY + ["eda/c123_safety_set.py", "fit", "--save"] + (["--out", str(target)] if target else []), cwd=COMP, env=ENV,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 5 and "no deployment" in r.stderr
        assert target is None or not target.exists()
    assert (sha(deploy) if deploy.is_file() else None) == before


@needs_oof
def test_c123_rule_real_files_override():
    if not (PRIV / "c123_splits.json").is_file() or not (PRIV / "c123_test_mass_set.json").is_file():
        pytest.skip("c123 splits / test-mass not computed yet")
    M = _load("c123_rule_real_test", "eda/c123_safety_set.py")
    oof, sp = M.load_results(None)
    rep = json.loads((PRIV / "c123_test_mass_set.json").read_text())
    r = M.rule_result(oof, sp, rep)
    if r["ii_cv"]["ok"]:
        pytest.skip("rule (ii) met — override irrelevant")
    assert r["verdict"] == "no deployment" and r["variant"] == "b" and r["ii_l1"]["ok"] is True
    r = M.rule_result(oof, sp, rep, M.override_record(DECISION))
    assert r["verdict"] == f"deploy b — (ii) missed, overridden by {DECISION}" and r["override_ii"]["applied"] is True


def test_safety_final_deploy_file_variant_b_with_override():
    deploy = COMP / "artifacts" / "stage2_safety_final.json"
    probe_b = PRIV / "probe" / "stage2_safety_b_probe.json"
    if not deploy.is_file():
        pytest.skip("deployment file of the safety line not built yet")
    d = json.loads(deploy.read_text())
    assert d["variant"] == "b" and d["feature_set"] == "fallback_i1" and d["align_mode"] == "set" and d["override_ii"] == DECISION
    assert "probe" not in d and len(d["model"]["features"]) == 55 and d["model"]["features"][40:] == list(S2.I1_FEATS)
    ro = d["rule_oof"]
    assert ro["ii_cv"]["ok"] is False and ro["ii_cv"]["ok_effective"] is True and ro["override_ii"]["applied"] is True
    assert ro["override_ii"]["decision"] == DECISION and ro["verdict"].endswith(f"overridden by {DECISION}")
    if probe_b.is_file():  # same model as probe b
        assert json.loads(probe_b.read_text())["model"] == d["model"]
    rj = PRIV / f"c123_rule_{DECISION}.json"  # result file of the rule with override (build script step 4, with test report)
    if rj.is_file():
        r = json.loads(rj.read_text())
        assert r["verdict"] == f"deploy b — (ii) missed, overridden by {DECISION}" and r["ii_l1"]["ok"] is True and r["ii_l1"]["l1_after_test"] <= 0.035
        assert r["ii_cv"]["ok"] is False and r["override_ii"]["applied"] is True and r["override_ii_flag"] == DECISION


def test_build_script_safety_final_pins_and_gate():
    p = COMP / "tools" / "builds" / "safety_final.sh"
    text = p.read_text()
    for var, path in (("IOOF_SHA", PRIV / "c123_oof.json"), ("ISPL_SHA", PRIV / "c123_splits.json"), ("MASS_SET_SHA", PRIV / "c123_test_mass_set.json"),
                      ("MASS_IND_SHA", PRIV / "c123_test_mass_inductive.json"), ("A_STAGE2_SHA", PRIV / "probe" / "stage2_safety_a_probe.json")):
        m = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        assert m, var
        if path.is_file():
            assert m.group(1) == sha(path), var
    assert "bash tools/builds/s036_safe_stage2.sh" in text and "c123_safety_set.py rule --test-report" in text
    assert '--align 0.75 --align-min 3' in text and "--align-inductive" not in text.split("BASE_SRC=(")[1].split(")")[0]
    assert 'SAFETY_FINAL_PROBE' in text and '$PRIV/probe' in text and "never submit" in text
    assert 'OVERRIDE_CV_II' in text and '[ "$OVERRIDE_CV_II" = override_cv_condition_ii ]' in text and '"${OVR[@]}"' in text
    assert "6b. copy null point" in text and "A_PROBE_CSV_SHA" in text and "deployment file != file of probe b" in text
    for var, path in (("STAGE2_SHA", COMP / "artifacts" / "stage2_safety_final.json"), ("OUT_SHA", COMP / "artifacts" / "submissions" / "sub_safety_final.csv")):
        m = re.search(rf'\b{var}="([0-9a-f]{{64}})"', text)
        assert m, var
        if path.is_file():
            assert m.group(1) == sha(path), var  # pinned after the first deployment build


# ------------------------------------------------------------------------------------------------ feature set fallback_i1 (variant b)

def test_fallback_i1_names_prefix_products_and_no_notes():
    T = _load("test_stage2_i1_for_safety_set", "tests/test_stage2_i1.py")
    CH = ["v1r", "zs27t", "think18sb", "pr27xm", "m3x3_full"]
    assert S2.FB_I1_SET == "fallback_i1" and S2.FB_I1_SET not in S2.FEATURE_SETS and not S2.uses_notes(S2.FB_I1_SET)
    fb, fi = S2.feature_names(CH, "fallback"), S2.feature_names(CH, S2.FB_I1_SET)
    assert len(fb) == 40 and fi == fb + list(S2.I1_FEATS) and len(fi) == 55
    st, Q, P, kn, cls, seen = T._toy(n=50, seed=3, notes=False)  # without notes: fallback_i1 requires none
    rows = np.arange(st.n)
    Xf, cf, nf = S2.pair_features(st, rows, Q, P, kn, "fallback")
    Xi, ci, ni = S2.pair_features(st, rows, Q, P, kn, S2.FB_I1_SET)
    assert np.array_equal(ci, cf) and ni == nf + list(S2.I1_FEATS) and Xi[:, :40].tobytes() == Xf.tobytes()
    assert Xi[:, 40:].tobytes() == S2.i1_interactions(Xf, nf).tobytes()


def test_apply_stage2_safe_accepts_fallback_i1(tmp_path):
    MS = _load("make_submission_safety_fbi1", "tools/make_submission.py")
    args = _safe_args(tmp_path, align_mode="set", align=0.75)
    s2 = json.loads(args[0].read_text())
    names = s2["model"]["features"] + list(S2.I1_FEATS)
    s2.update({"feature_set": S2.FB_I1_SET, "model": {**s2["model"], "features": names, "w": [0.0] * len(names), "mu": [0.0] * len(names),
                                                      "sd": [1.0] * len(names)}})
    args[0].write_text(json.dumps(s2))
    Q_new, info = MS.apply_stage2_safe(*args)
    assert np.array_equal(Q_new, args[3]) and info["feature_set"] == S2.FB_I1_SET and info["n_features"] == 55
