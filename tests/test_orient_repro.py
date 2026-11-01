"""Tests of the reproducible orientation network (src/orient.py, tools/orient_net.py, tools/make_orient.py).

The core tests run on synthetic maps without competition data. The comparisons against local inputs (images, statistics
cache, frozen lists, stored weights) are skipped if these are missing; they only check equality and checksums and print
no row contents.
"""

import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import orient as O

torch = pytest.importorskip("torch")

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
# Small network of the same architecture so that the tests run in seconds; the real recipe is checked on the stored network.
SMALL = O.TrainConfig(epochs=2, rounds=2, n_folds=2, batch=16, threads=1, predict_batch=24, channels=(2, 6, 8, 12, 16), hidden=16)
QUIET = lambda s: None  # noqa: E731


def _synthetic(n: int, seed: int) -> np.ndarray:
    """Maps with an unambiguous upright orientation (bar top left), plus noise; some of them are rotated."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 40, size=(n, 2, 64, 64), dtype=np.uint8)
    X[:, :, 4:12, 4:44] = 230
    X[:, 0, 20:25, 4:28] = 190
    for i in range(0, n, 5):
        X[i] = np.rot90(X[i], 1, axes=(1, 2))
    return np.ascontiguousarray(X)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def trained():
    fit = [("train", _synthetic(120, 2)), ("unlabelled", _synthetic(72, 3))]
    Xt = _synthetic(40, 4)
    return fit, Xt, O.train_nets(fit, Xt, SMALL, log=QUIET)


def test_predict_tta_is_rotation_equivariant():
    """Class r = rotated by r quarter turns counter-clockwise: one more rotation shifts the class by 1."""
    torch.manual_seed(0)
    net = O.make_net(SMALL).eval()
    X = _synthetic(12, 1)
    P, _ = O.predict_tta(net, X, batch=5)
    Pr, _ = O.predict_tta(net, np.ascontiguousarray(np.rot90(X, 1, axes=(2, 3))), batch=5)
    assert P.dtype == np.float64 and np.allclose(P.sum(1), 1.0)
    assert np.allclose(Pr, np.roll(P, 1, axis=1), atol=1e-12)


def test_training_is_bitwise_deterministic(trained):
    fit, Xt, a = trained
    b = O.train_nets(fit, Xt, SMALL, log=QUIET)
    assert O.tensor_digest(O.state_of(a["nets"])) == O.tensor_digest(O.state_of(b["nets"]))
    assert np.array_equal(a["P_fit"], b["P_fit"]) and np.array_equal(a["P_test"], b["P_test"])
    strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731 (runtime excluded)
    assert strip(a["history"]) == strip(b["history"])
    c = O.train_nets(fit, Xt, dataclasses.replace(SMALL, seed=1), log=QUIET)
    assert O.tensor_digest(O.state_of(c["nets"])) != O.tensor_digest(O.state_of(a["nets"]))


def test_training_restores_torch_state(trained):
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    with O._TorchState(1):
        assert torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 1
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == before


def test_fold_seeds_differ_and_folds_follow_first_version():
    seeds = {O.fold_seed(0, r, f) for r in range(2) for f in range(2)}
    assert len(seeds) == 4
    assert np.array_equal(O.fold_assignment(0, 2, 500), np.random.default_rng(0).integers(0, 2, 500))


def test_saved_weights_reproduce_training_predictions(trained, tmp_path):
    fit, Xt, res = trained
    w = tmp_path / "orient_cnn.pt"
    wm = O.save_weights(w, res["nets"])
    meta = {"format": O.FORMAT, "config": dataclasses.asdict(SMALL), "weights": wm}
    O.meta_path(w).write_text(json.dumps(meta))
    nets, _ = O.load_nets(w)
    n_train = len(fit[0][1])
    fold = O.fold_assignment(SMALL.seed, SMALL.n_folds, len(res["fold"]))[:n_train]
    with O._TorchState(SMALL.threads):
        P_test = O.ensemble_predict(nets, Xt, SMALL.predict_batch)
        P_train = O.oof_predict(nets, fit[0][1], fold, SMALL.predict_batch)
    assert np.array_equal(P_test, res["P_test"])
    assert np.array_equal(P_train, res["P_fit"][:n_train])
    raw = bytearray(w.read_bytes())
    raw[len(raw) // 2] ^= 1
    w.write_bytes(bytes(raw))
    with pytest.raises(SystemExit):
        O.load_nets(w)


def test_lists_follow_make_orient_format(tmp_path):
    files = np.array([f"id{i:02d}.jpeg" for i in range(6)])
    P = np.full((6, 4), 0.02)
    for i, (k, c) in enumerate([(0, 0.94), (1, 0.94), (2, 0.85), (3, 0.70), (1, 0.55), (3, 0.97)]):
        P[i, k] = c
    preds = O.pred_frame("test", files, P)
    assert list(preds["trace_id"]) == [f"id{i:02d}" for i in range(6)]
    table = pd.DataFrame({"trace_id": [f"id{i:02d}" for i in (5, 4, 3, 2, 1, 0)]})  # a different order than the files
    turns, conf = O.from_pred_frame(preds, "test", table)
    assert turns.tolist() == [1, 3, 1, 2, 3, 0]  # straighten = (4 - orientation) mod 4
    meta = O.write_lists(tmp_path, "test", table, turns, conf, 0.8, "c80", "synthetic")
    mapping = json.loads((tmp_path / "test_c80.json").read_text())
    rows = np.load(tmp_path / "test_c80_rows.npy")
    assert mapping == {"id01": 3, "id02": 2, "id05": 1}
    assert rows.dtype == np.int64 and rows.tolist() == [0, 3, 4]
    assert meta["rows"] == 3 and meta["turns_ccw"] == {"1": 1, "2": 1, "3": 1}
    assert meta["sha256"] == _sha(tmp_path / "test_c80.json")
    assert list(json.loads((tmp_path / "test_c80.json").read_text())) == sorted(mapping)


def test_compare_and_coverage_counts(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for d, c50, c80 in ((old, {"a": 1, "b": 3, "c": 2, "d": 1}, {"a": 1, "b": 3}), (new, {"a": 1, "b": 1, "e": 2}, {"a": 1, "b": 1, "e": 2})):
        d.mkdir()
        for name, m in (("c50", c50), ("c80", c80)):
            (d / f"test_{name}.json").write_text(json.dumps(m, sort_keys=True))
            np.save(d / f"test_{name}_rows.npy", np.arange(len(m), dtype=np.int64))
    r = O.compare_lists(new, old, "test", "c80")
    assert not r["sha256_json_equal"]
    assert (r["rows_new"], r["rows_old"], r["both"], r["both_same_turn"], r["only_old"], r["only_new"]) == (3, 2, 2, 1, 0, 1)
    cv = O.coverage(new, old, "test")
    assert cv == {"rows_new": 3, "covered_same_turn": 1, "in_run_other_turn": 1, "not_in_run": 1, "needs_new_rotated_rows": 2}


def test_frozen_directory_is_protected():
    with pytest.raises(SystemExit):
        O._guard_out(O.FROZEN_DIR, force=False)
    O._guard_out(O.DEFAULT_OUT, force=False)


def test_maps_compare_counts():
    a = {"files": np.array(["x.jpeg", "y.jpeg"]), "ok": np.array([True, True]),
         "ink": np.zeros((2, 64, 64), np.uint8), "wr": np.zeros((2, 64, 64), np.uint8)}
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    assert O.compare_maps(a, b)["identical"] and O.maps_sha256(a) == O.maps_sha256(b)
    b["wr"][1, 3, 3] = 7
    r = O.compare_maps(a, b)
    assert not r["identical"] and r["both_equal"] == 1 and O.maps_sha256(a) != O.maps_sha256(b)


# ----------------------------------------------------------------------------- against local inputs
needs_data = pytest.mark.skipif(not (DATA / "train.csv").is_file(), reason="competition data missing")


@needs_data
def test_pool_sample_matches_image_analysis():
    spec = importlib.util.spec_from_file_location("i5_common_t", COMP / "eda" / "i5_common.py")
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    assert O.table_files(DATA, "unlabelled") == c.unl_sample_files()
    assert O.table_files(DATA, "train") == sorted(pd.read_csv(DATA / "train.csv", dtype=str, usecols=["image_file"])["image_file"])


@needs_data
@pytest.mark.skipif(not O.STATS_CACHE.is_file(), reason="statistics cache of the image analysis missing")
def test_maps_from_images_match_stats_cache():
    files = O.table_files(DATA, "train")[:3]
    got = O.compute_maps(DATA, "train", workers=1, files=files)
    ref = O.maps_from_cache(O.STATS_CACHE, "train", files)
    assert O.compare_maps(got, ref)["identical"]


FROZEN_C80 = [O.FROZEN_DIR / f for f in ("train_c80.json", "train_c80_rows.npy", "test_c80.json", "test_c80_rows.npy")]
OLD_PRED = COMP / "artifacts" / "eda" / "private" / "i5_rot_cnn_pred.parquet"


@needs_data
@pytest.mark.skipif(not (OLD_PRED.is_file() and all(p.is_file() for p in FROZEN_C80)), reason="first version not available locally")
def test_make_orient_reproduces_frozen_lists_from_first_version(tmp_path):
    """The frozen lists are exactly the predictions of the first version; checked only through checksums."""
    # explicitly the local data: other tests may set K12_DATA to dummy folders without restoring it
    env = {**os.environ, "K12_DATA": str(DATA)}
    subprocess.run([sys.executable, "-I", str(COMP / "tools" / "make_orient.py"), "--min-conf", "0.8", "--name", "c80",
                    "--out", str(tmp_path)], check=True, capture_output=True, env=env)
    for p in FROZEN_C80:
        assert _sha(tmp_path / p.name) == _sha(p), p.name


@pytest.mark.skipif(not O.DEFAULT_WEIGHTS.is_file(), reason="stored weights missing")
def test_saved_weights_load_and_match_description():
    nets, meta = O.load_nets(O.DEFAULT_WEIGHTS)
    assert meta["format"] == O.FORMAT and len(nets) == meta["config"]["n_folds"]
    assert meta["config"]["seed"] == 0 and meta["predict"]["dtype"] == "float64"
    assert meta["fit"]["tables"]["train"]["offset"] == 0
