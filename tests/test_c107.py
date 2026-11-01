"""Checks of eda/c107_q397_full.py (Qwen3.5-397B as the sixth channel in the primary recipe).

Synthetic: channel construction bit-identical to vlm.proto.labels_to_probs (named probabilities and fixed levels), snapshot of
a growing answer file, earlier records as an unchanged prefix, coverage and abort at partial coverage, the pre-registered rule
exactly at its boundaries, the cross-fit without a leak from the evaluated fold, the bootstrap over tasks, output and JSON.
With real artifacts (skipped if they are missing): the anchors of both lines without the new channel."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c107_q397_full_test", COMP_ROOT / "eda" / "c107_q397_full.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

from vlm import proto as PR  # noqa: E402

CLASSES = ["NONE", "AAA-01", "BBB-01", "CCC-01", "NOT-02"]
SEEN = np.array([True, True, True, True, False])


def jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in records).encode()


def rec(row: int, ok: bool = True, root: str | None = "AAA-01", top3=None, probs=None, phase: str = "first", **extra) -> dict:
    r = {"row": row, "phase": phase, "ok": ok, "stop": "stop"}
    if ok:
        r.update({"root_code": root, "top3": top3 if top3 is not None else [root], "probabilities": probs})
    r.update(extra)
    return r


EDGE_RECORDS = [
    rec(0, top3=["AAA-01", "BBB-01", "NONE"], probs=[0.7, 0.2, 0.1]),  # ordinary
    rec(1, ok=False),  # first pass at the token limit …
    rec(1, root="BBB-01", top3=["BBB-01", "NONE", "AAA-01"], probs=[0.5, 0.3, 0.1], phase="finish"),  # … finish valid
    rec(2, top3=["AAA-01", "NONE", "BBB-01"], probs=[0.4, 0.3, 0.2]),
    rec(2, root="CCC-01", top3=["CCC-01", "AAA-01", "NONE"], probs=[0.6, 0.3, 0.1], phase="finish"),  # the last valid one counts
    rec(3, top3=["AAA-01", "BBB-01", "CCC-01"], probs=[True, float("nan"), 0.3]),  # bool and NaN are not numbers
    rec(4, top3=["AAA-01", "BBB-01", "CCC-01"], probs=[0.9, 0.5, 0.2]),  # sum > 1 -> normalised
    rec(5, top3=[], probs=[]),  # no named codes -> fixed level for root_code
    rec(6, root="ZZZ-99", top3=["ZZZ-99"], probs=[1.0]),  # unknown code -> invalid
    rec(7, top3=["AAA-01", "NOT-02", "NONE"], probs=[0.5, 0.3, 0.2]),  # code without a training example named
    rec(8, top3=["BBB-01", "AAA-01", "NONE"], probs=[0.0, 0.0, 0.0]),  # zeros only -> fixed levels
    rec(10, top3=["XYZ-00", "AAA-01", "NONE"], probs=[0.5, 0.3, 0.1]),  # an unknown code in top3 is skipped
    {"row": 11, "phase": "first", "root_code": "CCC-01", "top3": ["CCC-01"], "probabilities": [0.8]},  # fine without the field
]
N_EDGE = 13  # rows 9 and 12 without a record


# ------------------------------------------------------------------------------------------------ channel construction

def test_channel_bitwise_equal_to_labels_to_probs_both_variants(tmp_path):
    raw = jsonl(EDGE_RECORDS)
    complete, records, info = M.parse_snapshot(raw)
    assert complete == raw and not info["partial_tail"] and info["records"] == len(EDGE_RECORDS)
    f = tmp_path / "labels.jsonl"
    f.write_bytes(raw)
    P_ref, h_ref = PR.labels_to_probs(f, N_EDGE, CLASSES, SEEN)
    P, h = M.proto_channel(records, N_EDGE, CLASSES, SEEN, "stated")
    assert np.array_equal(h, h_ref) and np.array_equal(P, P_ref)
    # fixed levels as in c47 (fixed_steps): probabilities = None in every record, then labels_to_probs
    f2 = tmp_path / "levels.jsonl"
    f2.write_text("".join(json.dumps({**r, "probabilities": None}) + "\n" for r in records))
    L_ref, hl_ref = PR.labels_to_probs(f2, N_EDGE, CLASSES, SEEN)
    L, hl = M.proto_channel(records, N_EDGE, CLASSES, SEEN, "levels")
    assert np.array_equal(hl, hl_ref) and np.array_equal(L, L_ref)
    # expectations by hand
    assert h.tolist() == [True, True, True, True, True, True, False, True, True, False, True, True, False]
    assert P[1].argmax() == CLASSES.index("BBB-01")  # the finish replaces the invalid first pass
    assert P[2].argmax() == CLASSES.index("CCC-01")  # the last valid record counts
    assert P[7][CLASSES.index("NOT-02")] > 0  # the mass on the code without a training example stays (as on the test path)
    assert np.isclose(L[0][CLASSES.index("AAA-01")], 0.6 + 0.05 / 4)  # fixed levels + remainder spread evenly over 4 classes
    np.testing.assert_allclose(P[h].sum(1), 1.0, atol=1e-12)


def test_fill_missing_uniform_over_seen_classes():
    P, h = M.proto_channel(M.parse_snapshot(jsonl(EDGE_RECORDS))[1], N_EDGE, CLASSES, SEEN, "stated")
    F = M.fill_missing(P, h, SEEN)
    np.testing.assert_allclose(F[~h], np.tile([0.25, 0.25, 0.25, 0.25, 0.0], ((~h).sum(), 1)))
    assert np.array_equal(F[h], P[h]) and not np.shares_memory(F, P)


# ------------------------------------------------------------------------------------------------ snapshot, earlier records

def test_parse_snapshot_drops_partial_tail_and_rejects_corrupt_line():
    raw = jsonl(EDGE_RECORDS[:3]) + b'{"row": 4, "phase": "fi'
    complete, records, info = M.parse_snapshot(raw)
    assert info["partial_tail"] and len(records) == 3 and complete.endswith(b"\n")
    assert info["sha256_complete"] == hashlib.sha256(complete).hexdigest()
    with pytest.raises(ValueError, match="not JSON"):
        M.parse_snapshot(jsonl(EDGE_RECORDS[:2]) + b"{broken\n" + jsonl(EDGE_RECORDS[2:3]))


def test_validate_records_only_training_positions():
    M.validate_records([rec(0), rec(4, table="train")], 5)
    with pytest.raises(ValueError, match="table"):
        M.validate_records([rec(0, table="test")], 5)
    for bad in (-1, 5, True, "3", 1.0):
        with pytest.raises(ValueError, match="row position"):
            M.validate_records([{**rec(0), "row": bad}], 5)


def test_legacy_prefix_found_and_tamper_detected():
    old = jsonl(EDGE_RECORDS[:5])
    sha = hashlib.sha256(old).hexdigest()
    grown = old + jsonl(EDGE_RECORDS[5:])
    assert M.legacy_prefix(grown, sha) == {"bytes": len(old), "lines": 5}
    tampered = old.replace(b"0.7", b"0.8") + jsonl(EDGE_RECORDS[5:])
    assert M.legacy_prefix(tampered, sha) is None


# ------------------------------------------------------------------------------------------------ coverage and abort

def test_coverage_full_and_problems():
    full = [rec(i) for i in range(4)]
    cov = M.coverage(full, 4, CLASSES)
    assert cov["rows_valid"] == 4 and cov["rows_missing"] == 0 and cov["rows_pending_finish"] == 0
    assert M.coverage_problems(cov, False, 0.98, True) == []
    M.require_full([], dry=False)  # full coverage: no abort


def test_partial_coverage_aborts_unless_dry():
    recs = [rec(0), rec(1), rec(2, ok=False), rec(3, ok=False), rec(3, ok=False, phase="finish")]  # row 4 missing
    cov = M.coverage(recs, 5, CLASSES)
    assert (cov["rows_missing"], cov["rows_pending_finish"], cov["rows_finish_invalid"], cov["rows_valid"]) == (1, 1, 1, 2)
    probs = M.coverage_problems(cov, partial_tail=True, min_valid=0.98, legacy_ok=False)
    assert len(probs) == 5
    assert any("without any record" in p for p in probs) and any("finishing pass" in p for p in probs)
    assert any("incomplete" in p for p in probs) and any("below 98%" in p for p in probs)
    assert any("earlier records" in p for p in probs)
    with pytest.raises(SystemExit) as e:
        M.require_full(probs, dry=False)
    assert e.value.code == 3
    M.require_full(probs, dry=True)  # --dry: continues, but without a decision


def test_finished_but_invalid_row_is_not_pending():
    recs = [rec(i) for i in range(99)] + [rec(99, ok=False), rec(99, ok=False, phase="finish")]
    cov = M.coverage(recs, 100, CLASSES)
    assert cov["rows_pending_finish"] == 0 and cov["rows_finish_invalid"] == 1
    assert M.coverage_problems(cov, False, 0.98, True) == []  # 99 % valid
    assert M.coverage_problems(cov, False, 0.995, True)  # a stricter threshold is missed


# ------------------------------------------------------------------------------------------------ pre-registered rule

def _rule_case():
    """500 rows, 5 folds of 100. Δ = 18 units = 18/3000 = exactly 0.006 (nine rows with 1/3 — just off in floating point), fixed
    levels 9 units = exactly 0.003, fold 4 with Δ = 0 from gains and losses."""
    n = 500
    folds = np.repeat(np.arange(5), 100)
    base, new, fixed = np.zeros(n), np.zeros(n), np.zeros(n)
    new[[0, 1, 100, 101, 200, 201, 300, 301, 302]] = 1.0 / 3.0  # +18 units in folds 0–3
    base[[400, 401]], new[[400, 401]] = [1.0, 1.0 / 3.0], [0.5, 1.0]  # Fold 4: −3 + 4 = +1 …
    base[402], new[402] = 0.5, 1.0 / 3.0  # … −1 = 0
    fixed[:] = base
    fixed[[0, 100, 200]] = 0.5  # +9 units
    return new, base, fixed, folds


def test_rule_passes_exactly_at_thresholds():
    new, base, fixed, folds = _rule_case()
    r = M.evaluate_rule(new, base, fixed, folds, 1.5, 3.0)
    assert r["passed"] and r["verdict"] == "PASSED"
    assert r["checks"][0]["exact"] == Fraction(6, 1000) and r["checks"][4]["exact"] == Fraction(3, 1000)
    assert r["checks"][3]["per_fold_units"] == [4, 4, 4, 6, 0]
    assert r["checks"][3]["value"] == 5  # a fold with Δ = 0 counts as "≥ 0"


def test_rule_fails_just_below_each_threshold():
    new, base, fixed, folds = _rule_case()
    below = new.copy()
    below[302] = 0.0  # 16 units < 18
    assert not M.evaluate_rule(below, base, fixed, folds, 1.5, 3.0)["checks"][0]["ok"]
    assert not M.evaluate_rule(new, base, fixed, folds, np.nextafter(1.5, 0.0), 3.0)["checks"][1]["ok"]
    assert not M.evaluate_rule(new, base, fixed, folds, 1.5, np.nextafter(3.0, 0.0))["checks"][2]["ok"]
    assert not M.evaluate_rule(new, base, fixed, folds, float("nan"), 3.0)["passed"]
    fixed_below = fixed.copy()
    fixed_below[200] = 1.0 / 3.0  # 3 + 3 + 2 = 8 units < 9
    r = M.evaluate_rule(new, base, fixed_below, folds, 1.5, 3.0)
    assert not r["checks"][4]["ok"] and not r["passed"] and r["verdict"] == "FAILED"


def test_rule_fold_count_four_of_five():
    new, base, fixed, folds = _rule_case()
    one_neg = new.copy()
    one_neg[402] = 0.0  # fold 4: −3 + 4 − 3 = −2 units -> only 4/5 folds ≥ 0
    one_neg[303] = 0.5  # compensation in fold 3 (+3): sum 19 ≥ 18 units
    r = M.evaluate_rule(one_neg, base, fixed, folds, 2.0, 4.0)
    assert r["checks"][3]["per_fold_units"] == [4, 4, 4, 9, -2]
    assert r["checks"][3]["value"] == 4 and r["checks"][3]["ok"] and r["passed"]
    two_neg = one_neg.copy()
    base2 = base.copy()
    base2[0], two_neg[0] = 1.0, 0.0  # row 0: 0 − 6 = −6
    base2[1] = 1.0  # row 1: 2 − 6 = −4 -> fold 0 with −10 units, only 3/5 folds ≥ 0
    r2 = M.evaluate_rule(two_neg, base2, fixed, folds, 2.0, 4.0)
    assert r2["checks"][3]["per_fold_units"][0] == -10
    assert r2["checks"][3]["value"] == 3 and not r2["checks"][3]["ok"] and not r2["passed"]


def test_units_rejects_values_that_are_not_map3():
    assert M.units([0.0, 1 / 3, 0.5, 1.0]).tolist() == [0, 2, 3, 6]
    for bad in ([0.25], [1.5], [-1 / 3]):
        with pytest.raises(ValueError):
            M.units(bad)


def test_ll_readings_flags_reading_that_changes_criterion():
    rng = np.random.default_rng(5)
    q = np.repeat(np.arange(100), 3)
    base = rng.uniform(0.5, 1.5, 300)
    new = base - 0.2  # clear gain in the full recipe
    pre_new = base + rng.normal(0, 0.5, 300) * 0.01  # no gain without the last step
    draws = M.question_draws(q, B=2000, seed=1)
    out = M.ll_readings(base, new, base, pre_new, q, draws, main_ok=True)
    assert out["readings"]["full recipe, σ over tasks"]["z"] >= 3
    assert any("without the last step" in f for f in out["flips"]) and not any(f.startswith("full recipe, σ") for f in out["flips"])


# ------------------------------------------------------------------------------------------------ Bootstrap

def test_cluster_boot_over_questions():
    q = np.repeat(np.arange(4), 2)
    v = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    draws = M.question_draws(q, B=20000, seed=1)
    cb = M.cluster_boot(v, q, draws)
    assert cb["mean"] == 0.5 and abs(cb["se_q"] - 0.25) < 0.02  # task means 1/0/1/0: σ = √(0.25/4) = 0.25
    rows_only = M.metric.paired_bootstrap(v, np.zeros(8), n_boot=20000, seed=1)["sigma"]
    assert rows_only < cb["se_q"] - 0.04  # over rows (≈ 0.177) too small when the rows of a task are identical
    sub = M.cluster_boot(v, q, draws, mask=np.array([1, 0, 1, 0, 1, 0, 1, 0], bool))
    assert sub["mean"] == 0.5 and sub["n_rows"] == 4
    const = M.cluster_boot(np.full(8, 0.3), q, draws)
    assert const["se_q"] < 1e-12 and np.isinf(const["z_q"])
    again = M.cluster_boot(v, q, M.question_draws(q, B=20000, seed=1))
    assert again == cb  # fixed seed: reproducible


# ------------------------------------------------------------------------------------------------ recipe

def synth(n: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = len(CLASSES)
    y = rng.choice(4, size=n, p=[0.4, 0.3, 0.2, 0.1])
    folds = np.arange(n) % 5

    def chan(strength: float) -> np.ndarray:
        z = rng.normal(size=(n, K))
        z[np.arange(n), y] += strength
        P = np.exp(z)
        P[:, ~SEEN] *= 0.01
        return P / P.sum(1, keepdims=True)

    mats = {"a": chan(2.0), "b": chan(0.7), "c": chan(0.0)}
    spec = {"label": "synth", "names": ["a", "b"], "floors": {"b": 0.05}, "think": "a", "factor": 1.4, "beta": 0.2, "last": "trans"}
    return y, folds, mats, spec


@pytest.mark.parametrize("last", ["trans", "ind", "none"])
def test_cross_fit_normalized_masked_and_without_leak(last):
    y, folds, mats, spec = synth()
    fit = M.cross_fit(y, folds, SEEN, mats, spec, last=last)
    for Q in (fit["Q"], fit["P"]):
        assert M.sane(Q, SEEN)
    th = fit["theta"]
    assert th.shape == (5, 2) and (th >= 0).all() and (th[:, 0] <= 4 * 1.4 + 1e-12).all() and (th[:, 1] <= 4 + 1e-12).all()
    # no leak: labels of the evaluated fold do not change its predictions (weights and target only from the other folds)
    y2 = y.copy()
    y2[folds == 0] = (y2[folds == 0] + 1) % 4
    fit2 = M.cross_fit(y2, folds, SEEN, mats, spec, last=last)
    assert np.array_equal(fit2["Q"][folds == 0], fit["Q"][folds == 0])


def test_transductive_step_moves_fold_mass_towards_target():
    y, folds, mats, spec = synth(seed=3)
    q_none = M.cross_fit(y, folds, SEEN, mats, spec, last="none")["Q"]
    q_trans = M.cross_fit(y, folds, SEEN, mats, spec, last="trans")["Q"]
    gap_none = gap_trans = 0.0
    for k in range(5):
        t = M.target_of(y, np.flatnonzero(folds != k), len(CLASSES))
        on = t > 0
        gap_none += np.abs(q_none[folds == k].mean(0) - t)[on].sum()
        gap_trans += np.abs(q_trans[folds == k].mean(0) - t)[on].sum()
    assert gap_trans < gap_none


def test_with_channel_appends_without_mutating():
    y, folds, mats, spec = synth()
    s2 = M.with_channel(spec, "c")
    assert s2["names"] == ["a", "b", "c"] and s2["floors"] == {"b": 0.05, "c": 0.05}
    assert spec["names"] == ["a", "b"] and spec["floors"] == {"b": 0.05}
    fit = M.cross_fit(y, folds, SEEN, mats, s2)
    w = M.weights_report(fit, "c")
    assert len(w["theta_new"]) == 5 and all(0.0 <= s <= 1.0 for s in w["share_new"])


def test_check_anchor_exact_or_abort():
    rows = np.array([1.0, 0.5, 1 / 3, 0.0, 1.0, 1.0])  # 6 + 3 + 2 + 0 + 6 + 6 = 23 units
    folds = np.array([0, 0, 1, 1, 2, 2])
    out = M.check_anchor(rows, 23, "synth", per_fold=[9 / 12, 2 / 12, 12 / 12], folds=folds)
    assert out["units"] == 23 and out["per_fold_units"] == [9, 2, 12]
    with pytest.raises(SystemExit) as e:
        M.check_anchor(rows, 22, "synth")
    assert e.value.code == 4
    with pytest.raises(SystemExit):
        M.check_anchor(rows, 23, "synth", per_fold=[8 / 12, 3 / 12, 12 / 12], folds=folds)


def test_report_and_json_roundtrip(capsys):
    y, folds, mats, spec = synth(seed=7)
    n = len(y)
    q = np.arange(n) // 5
    ctx = {"folds": folds, "q_code": q, "group": np.array(M.GROUPS)[np.arange(n) % 4], "part_a": np.arange(n) < 60}
    fb, fn = M.cross_fit(y, folds, SEEN, mats, spec), M.cross_fit(y, folds, SEEN, mats, M.with_channel(spec, "c"))
    rb, rn = M.rows_map(y, fb["Q"], SEEN), M.rows_map(y, fn["Q"], SEEN)
    lb, ln = M.rows_ll(y, fb["Q"]), M.rows_ll(y, fn["Q"])
    draws = M.question_draws(q, B=500, seed=3)
    c = M.compare(rn, rb, ctx, draws, ln, lb)
    assert c["delta_exact"] == Fraction(int((M.units(rn) - M.units(rb)).sum()), 6 * n)
    assert np.isclose(c["ll_gain"], (lb - ln).mean()) and sum(g["n"] for g in c["groups"].values()) == n
    rule = M.evaluate_rule(rn, rb, rn, folds, c["z_q"], c["ll_z_q"])
    les = M.ll_readings(lb, ln, lb, ln, q, draws, rule["checks"][2]["ok"])
    w = M.weights_report(fn, "c")
    alone = {"map3_valid": 0.5, "top1_valid": 0.4, "none_top_valid": 0.3, "none_true_valid": 0.33, "groups_valid": {g: 0.5 for g in M.GROUPS}}
    # the dict keys mirror the keys that c107 uses in its report
    M.report_decision(alone, c, c, rule, les, {"one set": c, "inductive line, earlier recipe": c},
                      {"primary": w, "fixed levels": w, "inductive": w})
    out = capsys.readouterr().out
    assert "PRE-REGISTERED RULE" in out and rule["verdict"] in out and "FOR INFORMATION ONLY" in out
    back = json.loads(M.to_json({"main": c, "rule": rule, "weights": w, "flag": np.bool_(True), "k": np.int64(3)}))
    assert back["rule"]["verdict"] == rule["verdict"] and back["flag"] is True and back["k"] == 3
    assert back["main"]["delta_exact"] == f"{c['delta_exact'].numerator}/{c['delta_exact'].denominator}"


# ------------------------------------------------------------------------------------------------ real artifacts

def test_real_anchors_without_new_channel():
    try:
        root = M.C.find_data_root()
    except FileNotFoundError:
        pytest.skip("competition data missing")
    needed = [M.ART / f"oof_{nm}.npy" for nm in M.OOF_SHA16] + [M.ART / M.STACK_BEST[0], M.ART / M.STACK_SAFE[0]]
    if not all(p.is_file() for p in needed):
        pytest.skip("artifacts missing")
    ctx = M.context(root)
    mats, _ = M.load_base(ctx["n"], ctx["K"])
    anc, _, _ = M.anchors(ctx, mats, M.check_stacks())
    assert anc["best"]["units"] == 7379 and anc["best"]["per_fold_units"] == [1469, 1540, 1431, 1475, 1464]
    assert anc["safe"]["units"] == 7308
