"""Checks of the deployment path of the 397B channel (eda/c110_q397_deploy.py, tools/builds/s029_q397.sh) and of the share by
novelty of the code in tools/make_submission.py (arith_beta_novel).

Synthetic: novelty per task and topic, reference to the other folds; mix_channels without the field unchanged, with β_new = β
bit-identical to the earlier blend, with β_new ≠ β different only on new classes, rejected without a novelty matrix; REL-01 at
rank 1 as with --blank-rows; "readable = no" from the last valid record; names and recipes of the blend files. With real
artifacts (skipped if they are missing): the saved blend files stack_s029_best/safe carry the fields that make_submission reads."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_submission_c110_test", "tools/make_submission.py")
D = _load("c110_q397_deploy_test", "eda/c110_q397_deploy.py")


def test_code_novelty_per_question_and_topic():
    ref_q = np.array(["q1", "q1", "q1", "q2"])
    ref_t = np.array(["T", "T", "U", "T"])
    ref_y = np.array([1, 0, 2, 3])
    nov = MS.code_novelty(ref_q, ref_t, ref_y, np.array(["q1", "q2", "q3", "q1"]), np.array(["T", "T", "T", "U"]), 5)
    assert nov.shape == (4, 5) and not nov[:, 0].any()  # NONE is never new
    assert nov[0].tolist() == [False, False, True, True, True]  # q1/T: code 1 seen (code 2 only under topic U)
    assert nov[1].tolist() == [False, True, True, False, True]  # q2/T: code 3 seen
    assert nov[2].tolist() == [False, True, True, True, True]  # unknown task: everything except NONE is new
    assert nov[3].tolist() == [False, True, False, True, True]  # q1/U: code 2 only


def test_train_novelty_uses_other_folds_only():
    train = pd.DataFrame({"question_id": ["a", "a", "a", "b"], "topic_name": ["T"] * 4, "root_code": ["X-01", "X-01", "Y-01", "NONE"]})
    classes = ["NONE", "X-01", "Y-01"]
    folds = np.array([0, 1, 0, 1])
    nov = MS.train_novelty(train, folds, classes)
    assert nov[0].tolist() == [False, False, True]  # row 0 (fold 0): X-01 seen for task a in fold 1, Y-01 only in its own fold
    assert nov[2].tolist() == [False, False, True]  # row 2 (fold 0): its own label does not count
    assert nov[1].tolist() == [False, False, False]  # row 1 (fold 1): X-01 and Y-01 seen for a in fold 0
    assert nov[3].tolist() == [False, True, True]  # task b only in fold 1


def _chans():
    rng = np.random.default_rng(3)
    seen = np.array([True, True, True, True, False])
    out = []
    for _ in range(3):
        p = rng.dirichlet(np.ones(5), size=6)
        p[:, 4] = 0.0
        p = p / p.sum(1, keepdims=True)
        out.append(np.log(np.clip(0.95 * p + 0.05 * seen / seen.sum(), 1e-9, 1)))
    return out, seen


def test_mix_channels_without_field_is_unchanged_and_beta_new_equal_beta_is_bitwise():
    chans, seen = _chans()
    base = {"theta": [0.5, 0.3, 0.2], "arith_beta": 0.2}
    P0 = MS.mix_channels(base, chans, None, seen)
    nov = np.zeros((6, 5), dtype=bool)
    nov[:, 2] = True
    assert np.array_equal(P0, MS.mix_channels(base, chans, None, seen, novel=nov))  # without the field: novelty has no effect
    assert np.array_equal(P0, MS.mix_channels({**base, "arith_beta_novel": 0.2}, chans, None, seen, novel=nov))  # β_new = β: bit-identical
    P1 = MS.mix_channels({**base, "arith_beta_novel": 0.7}, chans, None, seen, novel=nov)
    assert np.allclose(P1.sum(1), 1.0) and not P1[:, ~seen].any() and not np.allclose(P1, P0)
    # formula: before normalisation only the new classes differ
    Pl = np.where(seen[None, :], MS.a2.apply_mix(np.array(base["theta"]), chans, None), 0.0)
    Pl = Pl / Pl.sum(1, keepdims=True)
    w = np.array(base["theta"]) / np.sum(base["theta"])
    A = sum(wi * np.where(seen[None, :], np.exp(lg), 0.0) / np.where(seen[None, :], np.exp(lg), 0.0).sum(1, keepdims=True) for wi, lg in zip(w, chans))
    A = A / A.sum(1, keepdims=True)
    B = np.where(nov, 0.7, 0.2)
    ref = (1 - B) * Pl + B * A
    assert np.allclose(P1, ref / ref.sum(1, keepdims=True), atol=1e-12)


def test_mix_channels_beta_new_requires_novelty_and_range():
    chans, seen = _chans()
    st = {"theta": [0.5, 0.3, 0.2], "arith_beta": 0.2, "arith_beta_novel": 0.6}
    with pytest.raises(SystemExit):
        MS.mix_channels(st, chans, None, seen)  # novelty missing
    with pytest.raises(SystemExit):
        MS.mix_channels(st, chans, None, seen, novel=np.zeros((5, 5), dtype=bool))  # wrong shape
    with pytest.raises(SystemExit):
        MS.mix_channels({**st, "arith_beta_novel": 1.5}, chans, None, seen, novel=np.zeros((6, 5), dtype=bool))
    assert MS.arith_beta_novel_of({"arith_beta": 0.2}) is None


def test_oof_mixture_passes_novelty(tmp_path):
    seen = np.array([True, True, True, False])
    a = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.1, 0.8, 0.0]])
    b = np.array([[0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    np.save(tmp_path / "oof_a.npy", a)
    np.save(tmp_path / "oof_b.npy", b)
    st = {"channels": ["a", "b"], "theta": [1.0, 0.5], "floor": {"b": 0.3}, "none": [], "probes": {}, "arith_beta": 0.2}
    P0 = MS.oof_mixture(st, seen, tmp_path)
    nov = np.array([[False, True, False, False], [False, False, True, False]])
    assert np.array_equal(P0, MS.oof_mixture({**st, "arith_beta_novel": 0.2}, seen, tmp_path, novel=nov))
    assert not np.allclose(P0, MS.oof_mixture({**st, "arith_beta_novel": 0.9}, seen, tmp_path, novel=nov))
    with pytest.raises(SystemExit):
        MS.oof_mixture({**st, "arith_beta_novel": 0.9}, seen, tmp_path)


def test_force_code_like_blank_rows():
    Q = np.array([[0.5, 0.3, 0.2, 0.0], [0.1, 0.6, 0.3, 0.0]])
    P = D.force_code(Q, np.array([1]), 2)
    assert np.array_equal(P[0], Q[0]) and P[1].argmax() == 2 and np.isclose(P[1].sum(), 1.0)
    assert np.argsort(-P[1])[1:3].tolist() == [1, 0]  # the remaining order stays
    assert np.array_equal(D.force_code(Q, np.array([], dtype=int), 2), Q)


def test_readable_no_last_valid_record(tmp_path):
    f = tmp_path / "labels.jsonl"
    recs = [{"row": 0, "ok": True, "root_code": "NONE", "readable": "no"},
            {"row": 0, "ok": True, "root_code": "AAA-01", "readable": "yes"},  # the last valid one counts
            {"row": 1, "ok": True, "root_code": "AAA-01", "readable": "No "},
            {"row": 2, "ok": False, "readable": "no"},  # invalid
            {"row": 3, "ok": True, "root_code": "ZZZ-99", "readable": "no"},  # code outside the class list
            {"row": 4, "ok": True, "root_code": "NONE", "readable": "partly"}]
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    m, vals = D.readable_no([f], 5, ["NONE", "AAA-01"])
    assert m.tolist() == [False, True, False, False, False] and vals["no"] == 1 and vals["yes"] == 1 and vals["partly"] == 1


def test_stems_and_line_specs():
    assert D.stem_of("best", "m3x2_full_cv01", False) == "s029_best"
    assert D.stem_of("safe", "m3x3_full", True) == "s029_safe_m3x3_bn"
    spec5, spec6 = D.line_specs("best", "m3x3_full")
    assert spec5["names"] == ["v1r", "zs27q", "think18nb", "pr27xm", "m3x3_full"]
    assert spec6["names"][-1] == "q397" and spec6["floors"]["q397"] == 0.05 and spec6["floors"]["m3x3_full"] == 0.05
    assert "m3x2_full_cv01" not in spec6["floors"] and spec6["think"] == "think18nb" and spec6["factor"] == 1.4
    _, s6 = D.line_specs("safe", "m3x2_full_cv01")
    assert s6["names"] == ["v1r", "zs27t", "think18sb", "pr27xm", "m3x2_full_cv01", "q397"] and s6["beta"] == 0.4 and s6["last"] == "ind"


@pytest.mark.parametrize("stem", ["s029_best", "s029_safe", "s029_best_bn", "s029_safe_bn"])
def test_saved_stacks_have_the_fields_make_submission_reads(stem):
    f = COMP_ROOT / "artifacts" / f"stack_{stem}.json"
    if not f.is_file():
        pytest.skip(f"{f.name} missing")
    s = json.loads(f.read_text())
    assert s["channels"][-1] == "q397" and len(s["theta"]) == 6 and s["none"] == [] and s["probes"] == {}
    assert s["floor"]["q397"] == 0.05 and set(s["rank_only"]) == {s["channels"][4]}
    assert s["rank_only"][s["channels"][4]]["floor"] == s["floor"][s["channels"][4]] == 0.05
    assert MS.stack_fit_rows(s, 1500).tolist() == list(range(1500))
    assert MS.arith_beta_of(s) == (0.2 if "best" in stem else 0.4)
    assert (MS.arith_beta_novel_of(s) is not None) == stem.endswith("_bn")
    assert ("inductive_check" in s) == ("safe" in stem)
    assert s["cv"] > s["cross"]["without_q397"]
