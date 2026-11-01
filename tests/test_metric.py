"""MAP@3 against the examples of the competition's evaluation page and against the probability form."""
import numpy as np
import pytest

import metric


def test_examples_from_evaluation_page():
    # „the three submissions below all score 1.0 for a row whose correct label is ALG-06"
    for row in (["ALG-06", "CAL-01", "ANS-01"], ["ALG-06", "ALG-06", "ALG-06"], ["ALG-06", "CAL-01", "ALG-06"]):
        assert metric.map3(["ALG-06"], [row]) == 1.0


@pytest.mark.parametrize(
    "row, expected",
    [
        (["CAL-01", "ALG-06", "ANS-01"], 0.5),
        (["CAL-01", "ANS-01", "ALG-06"], 1 / 3),
        (["CAL-01", "ANS-01", "NONE"], 0.0),
        (["ALG-06"], 1.0),  # fewer than three predictions are allowed
        (["CAL-01"], 0.0),
        ([], 0.0),  # a missing row scores 0
        (["CAL-01", "CAL-01", "ALG-06"], 1 / 3),  # a duplicate of a wrong code uses up the slot
        (["CAL-01", "ANS-01", "NONE", "ALG-06"], 0.0),  # only the first three count
    ],
)
def test_row_scores(row, expected):
    assert metric.map3_rows(["ALG-06"], [row])[0] == pytest.approx(expected)


def test_mean_over_rows():
    y = ["NONE", "CAL-01", "ALG-06", "WOP-01"]
    preds = [["NONE"], ["NONE", "CAL-01"], ["NONE", "CAL-01", "ALG-06"], ["NONE", "CAL-01", "ALG-06"]]
    assert metric.map3(y, preds) == pytest.approx((1 + 0.5 + 1 / 3 + 0) / 4)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metric.map3_rows(["NONE"], [])


def test_silent_failure_modes_raise():
    # string instead of list: would be iterated character by character and silently give 0
    with pytest.raises(TypeError, match="split"):
        metric.map3_rows(["ALG-06"], ["ALG-06 CAL-01 NONE"])
    assert metric.map3_rows(["ALG-06"], ["ALG-06 CAL-01 NONE".split()])[0] == 1.0
    # NaN probabilities: would silently map to the first three classes
    P = np.array([[0.2, 0.5, 0.2, 0.1], [np.nan, 0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="NaN"):
        metric.top3_indices(P)


def test_proba_form_matches_list_form():
    rng = np.random.default_rng(0)
    classes = [f"C{i}" for i in range(12)]
    P = rng.random((500, 12))
    y_idx = rng.integers(0, 12, size=500)
    top = metric.top3_indices(P)
    preds = [[classes[j] for j in row] for row in top]
    y = [classes[i] for i in y_idx]
    assert metric.map3_from_proba(y_idx, P) == pytest.approx(metric.map3(y, preds))


def test_top3_is_sorted_and_tie_stable():
    P = np.array([[0.2, 0.5, 0.5, 0.1], [0.25, 0.25, 0.25, 0.25]])
    top = metric.top3_indices(P)
    assert top.tolist() == [[1, 2, 0], [0, 1, 2]]  # tie: smaller class index first


def test_allowed_mask_excludes_classes():
    P = np.array([[0.9, 0.05, 0.03, 0.02]])
    allowed = np.array([False, True, True, True])
    assert metric.top3_indices(P, allowed).tolist() == [[1, 2, 3]]
    assert metric.map3_from_proba(np.array([0]), P, allowed) == 0.0
    with pytest.raises(ValueError):
        metric.top3_indices(P, np.array([True, True, False, False]))


def test_bootstrap_sigma_matches_theory():
    rng = np.random.default_rng(1)
    rows = rng.choice([0.0, 1 / 3, 0.5, 1.0], size=4000, p=[0.3, 0.05, 0.1, 0.55])
    theory = rows.std() / np.sqrt(1000)
    assert metric.bootstrap_sigma(rows, n=1000, n_boot=4000) == pytest.approx(theory, rel=0.06)


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(2)
    base = rng.choice([0.0, 1.0], size=1500, p=[0.4, 0.6])
    better = base.copy()
    flip = rng.choice(np.where(base == 0)[0], size=90, replace=False)
    better[flip] = 1.0
    res = metric.paired_bootstrap(better, base)
    assert res["delta"] == pytest.approx(0.06)
    assert res["z"] > 5 and res["p_a_better"] > 0.999
    same = metric.paired_bootstrap(base, base)
    assert same["delta"] == 0.0 and same["z"] == 0.0


def test_registered_in_kaggle_core():
    from kaggle_core.metrics import get_metric

    m = get_metric("map3")
    P = np.eye(4)[[0, 1, 2, 3]]
    assert m.fn(np.array([0, 1, 2, 3]), P) == 1.0 and m.greater_is_better
