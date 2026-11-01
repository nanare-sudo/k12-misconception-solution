"""Gate tool tools/replay/gates.py: rules of the reasoning gates, list format, blend, write protection.

Synthetic, without competition data; plus a check on the real training artifacts (skipped if they are missing): the
lists of the strictly inductive line are produced bit-identically to the frozen ones."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = load("gates_under_test", COMP_ROOT / "tools" / "replay" / "gates.py")


def test_thresholds_are_strict():
    # at the thresholds themselves: not inside the gate (all rules use "<")
    p = np.array([0.49, 0.5, 0.94, 0.95, 0.2])
    top = np.array([0.59, 0.59, 0.1, 0.1, 0.6])
    m = G.gate_masks(p_none_t=p, top_t=top)
    assert m["wide_t"].tolist() == [True, False, False, False, True]
    assert m["band_t"].tolist() == [False, True, True, False, False]
    assert m["tight_t"].tolist() == [True, False, False, False, False]
    assert m["tb_t"].tolist() == [True, True, True, False, False]


def test_composition_and_done_rows():
    rng = np.random.default_rng(0)
    n = 400
    pq, pt = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    tq, tt = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    folds = rng.integers(0, 5, n)
    m0 = G.gate_masks(pq, pt, tq, tt)
    done = G.done_rows(m0["wide_q"], folds)
    m = G.gate_masks(pq, pt, tq, tt, done=done, v50=rng.uniform(0, 1, n) < 0.2)
    for k in ("q", "t"):
        assert not (m[f"tight_{k}"] & ~m[f"wide_{k}"]).any()          # the narrow gate lies inside the wide one
        assert not (m[f"band_{k}"] & m[f"wide_{k}"]).any()            # the band lies outside the wide gate
        assert np.array_equal(m[f"tb_{k}"], m[f"tight_{k}"] | m[f"band_{k}"])
    assert np.array_equal(m["union"], m["wide_q"] | m["wide_t"])
    assert np.array_equal(m["tight_union"], m["tight_q"] | m["tight_t"])
    assert np.array_equal(m["band_union"], m["band_q"] | m["band_t"])
    assert np.array_equal(m0["n3"], m["tight_union"] | m["band_union"])  # test side: without D
    assert np.array_equal(m["n3"], (m["tight_union"] | m["band_union"]) & ~done)  # training side: without D
    assert np.array_equal(m["tight_union_todo"], m["tight_union"] & ~done)
    assert np.array_equal(m["union_todo"], m["union"] & ~done)
    assert not (m["rot_v50"] & ~m["wide_q"]).any()
    assert np.array_equal(done, m0["wide_q"] & np.isin(folds, [1, 2]))


def test_safe_line_needs_only_t_inputs():
    p, top = np.array([0.1, 0.7, 0.99]), np.array([0.3, 0.9, 0.9])
    m = G.gate_masks(p_none_t=p, top_t=top)
    assert set(m) == {"wide_t", "band_t", "tight_t", "tb_t"}
    keys = {k for k, _, _ in G.list_names("test", "safe")}
    assert keys == {"wide_t", "tight_t", "band_t", "tb_t"}
    assert keys <= set(m)
    assert G.LINE_CHANNELS["safe"] == ("v1", "zs27t")  # no channel with a teacher block


def test_list_names_per_table():
    tr = [nm for _, nm, _ in G.list_names("train", "all")]
    te = [nm for _, nm, _ in G.list_names("test", "all")]
    assert len(tr) == 15 and len(te) == 12
    assert {"train", "train_band", "train_n3", "train_rot_v50", "train_tight_union_todo", "train_union_todo"} <= set(tr)
    assert {"test", "test_band_q", "test_n3", "test_tb_q", "test_tb_t"} <= set(te)
    assert [nm for _, nm, _ in G.list_names("test", "best")] == ["test", "test_tight_q", "test_band_q", "test_tb_q"]


def test_npy_bytes_match_np_save(tmp_path):
    m = np.array([False, True, False, True, True])
    pos = G.positions(m)
    assert pos.dtype == np.int64 and pos.tolist() == [1, 3, 4]
    f = tmp_path / "x.npy"
    np.save(f, pos)
    assert G.npy_bytes(pos) == f.read_bytes()
    assert G.sha256(G.npy_bytes(pos)) == G.sha256(f.read_bytes())


def _channels(rng, n, k):
    a = rng.dirichlet(np.ones(k), size=n)
    b = rng.dirichlet(np.ones(k) * 0.5, size=n)
    return a, b


def test_crossfit_matches_manual_and_is_deterministic():
    rng = np.random.default_rng(1)
    n, k = 300, 6
    a, b = _channels(rng, n, k)
    y = rng.integers(0, k, n)
    folds = np.arange(n) % 5
    a2 = G._a2()
    P1 = G.crossfit_mixture([a, b], y, folds, a2)
    P2 = G.crossfit_mixture([a, b], y, folds, a2)
    assert np.array_equal(P1, P2)
    la, lb = np.log(np.clip(a, 1e-9, 1)), np.log(np.clip(b, 1e-9, 1))
    for f in range(5):
        tr, va = folds != f, folds == f
        th = a2.fit_mix([la[tr], lb[tr]], None, y[tr])
        assert np.array_equal(P1[va], a2.apply_mix(th, [la[va], lb[va]], None))
    th = G.full_theta([a, b], y, a2)
    assert np.array_equal(th, a2.fit_mix([la, lb], None, y))
    M = G.apply_theta(th, [a, b], a2)
    assert M.shape == (n, k) and np.allclose(M.sum(1), 1.0)


def test_write_lists_guards(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    masks = {"wide_t": np.array([True, False]), "tight_t": np.array([True, False]), "band_t": np.array([False, True]),
             "tb_t": np.array([True, True])}
    with pytest.raises(SystemExit):
        G.write_lists(masks, "test", "safe", frozen, frozen)  # never into the frozen folder
    out = tmp_path / "out"
    written = G.write_lists(masks, "test", "safe", out, frozen)
    assert len(written) == 4 and (out / "test_tb_t.npy").read_bytes() == G.npy_bytes(np.array([0, 1]))
    assert G.write_lists(masks, "test", "safe", out, frozen) == []  # same content: nothing to do
    masks["tb_t"] = np.array([True, False])
    with pytest.raises(SystemExit):
        G.write_lists(masks, "test", "safe", out, frozen)  # different content: do not overwrite


def test_compare_reports_counts_only(tmp_path):
    np.save(tmp_path / "test_t.npy", np.array([0, 2], dtype=np.int64))
    masks = {"wide_t": np.array([True, True, False]), "tight_t": np.array([True, False, False]), "band_t": np.array([False, False, True]),
             "tb_t": np.array([True, False, True])}
    rows = {r["list"]: r for r in G.compare(masks, "test", "safe", tmp_path)}
    r = rows["test_t.npy"]
    assert r["equal"] is False and r["differing_rows"] == 2 and r["rows"] == 2 and r["frozen_rows"] == 2
    assert "only_new" not in r  # test side: only the number of differing rows
    assert rows["test_tb_t.npy"]["equal"] is None  # no frozen list
    assert all(not isinstance(v, (list, np.ndarray)) for rec in rows.values() for v in rec.values())  # no positions


@pytest.mark.skipif(not (COMP_ROOT / "data" / "train.csv").is_file() or not (COMP_ROOT / "data" / "derived" / "think_rows" / "train_tb_t.npy").is_file()
                    or not (COMP_ROOT / "artifacts" / "oof_zs27t.npy").is_file(), reason="competition data or artifacts missing")
def test_real_train_lists_safe_line_bitgleich(tmp_path):
    rc = G.main(["check", "--table", "train", "--line", "safe", "--json", str(tmp_path / "r.json")])
    assert rc == 0
