"""Mean channel pr27xm = named codes of pr27 ⊕ exact ranking (eda/c95_rank_mean.py, tools/make_submission.py --rank).

Checked on synthetic runs: the definition (0.5 · (named + ranking), before the floor; rows without a ranking = named codes),
bit-identity of the OOF path (table train) and the test path (table test, make_submission), the whole test path in
make_submission.main against a computation by hand, and the aborts (missing files, coverage, mass, NONE band, channel form
against the meta file, floor, partial state)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["NONE", "A", "B", "C"]
SEEN = np.array([True, True, True, False])
NINF = -np.inf


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def answers(n: int, rng: np.random.Generator, missing: tuple[int, ...] = ()) -> list[dict]:
    """Protocol answers: three named codes with probabilities (sum < 1, remainder uniform)."""
    recs = []
    codes = CLASSES[:3]
    for i in range(n):
        if i in missing:
            recs.append({"row": i, "ok": False})
            continue
        order = rng.permutation(3)
        p = np.sort(rng.dirichlet(np.ones(3)))[::-1] * 0.9
        recs.append({"row": i, "ok": True, "root_code": codes[order[0]], "top3": [codes[j] for j in order],
                     "probabilities": [float(x) for x in p]})
    return recs


def rank_logp(n: int, rng: np.random.Generator, none_rows: int) -> np.ndarray:
    """Exact ranking: all classes with a training example finite, mass ≈ 0.95; the first `none_rows` rows with NONE at the top,
    the others with A or B."""
    lp = np.full((n, len(CLASSES)), NINF)
    for i in range(n):
        top = 0 if i < none_rows else 1 + i % 2
        p = rng.dirichlet(np.ones(3)) * 0.3
        p[top] += 0.65
        lp[i, :3] = np.log(p)
    return lp


def write_proto(base: Path, run: str, job: str, recs: list[dict], table: str) -> None:
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "proto_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "proto_meta.json").write_text(json.dumps({"table": table, "valid": sum(1 for r in recs if r.get("ok", True))}))


def write_rank(base: Path, run: str, job: str, tag: str, idx, lp, table: str, done=None) -> None:
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}_meta.json").write_text(json.dumps({"table": table}))
    np.save(d / f"{tag}_idx.npy", np.asarray(idx, dtype=np.int64))
    np.save(d / f"{tag}_proto_rank.npy", np.asarray(lp, dtype=float))
    np.save(d / f"{tag}_proto_rank.done.npy", np.ones(len(idx), bool) if done is None else np.asarray(done, bool))


def by_hand_named(recs: list[dict], n: int) -> np.ndarray:
    """vlm/proto.labels_to_probs + uniform distribution for rows without an answer, rebuilt by hand."""
    P = np.tile(SEEN / SEEN.sum(), (n, 1)).astype(float)
    for r in recs:
        if not r.get("ok", True):
            continue
        p = np.zeros(len(CLASSES))
        for c, w in zip(r["top3"], r["probabilities"]):
            p[CLASSES.index(c)] += w
        p = p + max(0.0, 1.0 - p.sum()) * SEEN / SEEN.sum()
        P[r["row"]] = p / p.sum()
    return P


def by_hand_rank(lp: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(lp), np.exp(np.where(np.isfinite(lp), lp, 0.0)), 0.0)
    return p / p.sum(1, keepdims=True)


def test_rank_mean_definition():
    c95 = load("c95_def", COMP_ROOT / "eda" / "c95_rank_mean.py")
    rng = np.random.default_rng(0)
    named = rng.dirichlet(np.ones(4), 7)
    rank = rng.dirichlet(np.ones(4), 7)
    has = np.array([True, True, False, True, False, True, True])
    P = c95.rank_mean(named, rank, has)
    assert np.array_equal(P[has], (named[has] + rank[has]) / 2)  # (old + new) / 2
    assert np.array_equal(P[has], 0.5 * (named[has] + rank[has]))  # 0.5 · (old + new)
    assert np.array_equal(P[~has], named[~has])  # without a ranking: exactly the named codes (channel pr27)
    assert np.allclose(P.sum(1), 1.0)
    with pytest.raises(ValueError):
        c95.rank_mean(named, rank[:, :3], has)
    with pytest.raises(ValueError):
        c95.rank_mean(named, rank, has[:5])


def test_oof_and_test_path_bit_identical(tmp_path, monkeypatch):
    """Same answers and same ranking on a training and a test table: channel(train) (OOF path) and make_submission.proto_channel
    (test path) give bit-identically the same matrix = 0.5 · (named + ranking) by hand; rows without a valid answer carry
    0.5 · (uniform + ranking), like the 8 training rows of pr27."""
    c95 = load("c95_bit", COMP_ROOT / "eda" / "c95_rank_mean.py")
    ms = load("make_submission_bit", COMP_ROOT / "tools" / "make_submission.py")
    for mod in (c95.A8, ms.a8, ms.c95.A8):
        monkeypatch.setattr(mod, "COMP", tmp_path)
    n = 50
    rng = np.random.default_rng(1)
    recs = answers(n, rng, missing=(3, 17))
    lp = rank_logp(n, rng, none_rows=17)
    for table, run in (("train", "tr"), ("test", "te")):
        write_proto(tmp_path, run, "proto_job", recs, table)
        write_rank(tmp_path, run, "rank_job", "bench" if table == "train" else "test", np.arange(n), lp, table)
    P_tr, cnt_tr, h_tr, st_tr = c95.channel(["runs/tr/proto_job/proto"], "runs/tr/rank_job/bench", n, CLASSES, SEEN, "train")
    P_te, cnt_te, info = ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/rank_job/test", n, CLASSES, SEEN, none_band=(0.30, 0.38))
    assert np.array_equal(P_tr, P_te) and np.array_equal(cnt_tr, cnt_te)
    exp = 0.5 * (by_hand_named(recs, n) + by_hand_rank(lp))
    assert np.allclose(P_te, exp, atol=1e-15, rtol=0)
    assert h_tr.all() and st_tr["rows_without_answer"] == 2 and st_tr["rows_rank_without_answer"] == 2
    assert info["rows_rank"] == n and info["share_rank"] == 1.0 and np.isclose(info["none_top_rank"], 17 / 50)
    # the row order of the probe (longest prompts first) changes nothing
    order = rng.permutation(n)
    write_rank(tmp_path, "te2", "rank_job", "test", order, lp[order], "test")
    write_proto(tmp_path, "te2", "proto_job", recs, "test")
    P_te2, _, _ = ms.proto_channel("x", ["runs/te2/proto_job/proto"], "runs/te2/rank_job/test", n, CLASSES, SEEN, none_band=(0.30, 0.38))
    assert np.array_equal(P_te2, P_te)


def test_rows_without_rank_and_aborts(tmp_path, monkeypatch):
    """Test rows without a ranking keep exactly the named codes; coverage below 98 %, missing files, small mass, NONE outside
    the band and an incomplete/truncated run abort — unless a partial state is allowed deliberately. On the training rows the
    channel requires the ranking on every row."""
    c95 = load("c95_rows", COMP_ROOT / "eda" / "c95_rank_mean.py")
    ms = load("make_submission_rows", COMP_ROOT / "tools" / "make_submission.py")
    for mod in (c95.A8, ms.a8, ms.c95.A8):
        monkeypatch.setattr(mod, "COMP", tmp_path)
    n = 50
    rng = np.random.default_rng(2)
    recs = answers(n, rng, missing=(5,))
    lp = rank_logp(n, rng, none_rows=17)
    named = by_hand_named(recs, n)
    write_proto(tmp_path, "te", "proto_job", recs, "test")
    write_proto(tmp_path, "tr", "proto_job", recs, "train")
    band = (0.30, 0.38)
    # 49 of 50 rows (98 %): admissible, the missing row carries the named codes bit-identically
    keep = np.setdiff1d(np.arange(n), [40])
    write_rank(tmp_path, "te", "r49", "test", keep, lp[keep], "test")
    P, cnt, info = ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/r49/test", n, CLASSES, SEEN, none_band=band)
    assert np.array_equal(P[40], named[40]) and info["rows_named_only"] == 1 and info["rows_rank"] == 49
    assert np.allclose(P[keep], 0.5 * (named[keep] + by_hand_rank(lp[keep])), atol=1e-15, rtol=0)
    # 48 of 50 (96 %): too few
    keep2 = np.setdiff1d(np.arange(n), [40, 41])
    write_rank(tmp_path, "te", "r48", "test", keep2, lp[keep2], "test")
    with pytest.raises(SystemExit, match="at least 98"):
        ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/r48/test", n, CLASSES, SEEN, none_band=band)
    # run not fetched yet: clean abort naming the missing files
    with pytest.raises(SystemExit, match="missing"):
        ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/missing/test", n, CLASSES, SEEN, none_band=band)
    # mass of the candidates too small (forced answer start unnatural)
    write_rank(tmp_path, "te", "low", "test", np.arange(n), lp + np.log(0.5), "test")
    with pytest.raises(SystemExit, match="candidate mass"):
        ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/low/test", n, CLASSES, SEEN, none_band=band)
    # NONE at the top outside the malfunction band
    write_rank(tmp_path, "te", "none", "test", np.arange(n), rank_logp(n, np.random.default_rng(3), none_rows=30), "test")
    with pytest.raises(SystemExit, match="malfunction band"):
        ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/none/test", n, CLASSES, SEEN, none_band=band)
    # incomplete run (row 7 not finished) and truncated row (row 9 without class B): abort without a partial state,
    # with a partial state both rows keep the named codes (no remaining-mass form in the channel)
    lp_cut = lp.copy()
    lp_cut[9, 2] = NINF
    done = np.ones(n, bool)
    done[7] = False
    write_rank(tmp_path, "te", "part", "test", np.arange(n), lp_cut, "test", done=done)
    with pytest.raises(SystemExit, match="not a complete, exact run"):
        ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/part/test", n, CLASSES, SEEN, none_band=band)
    Pp, _, ip = ms.proto_channel("x", ["runs/te/proto_job/proto"], "runs/te/part/test", n, CLASSES, SEEN, none_band=band,
                                 rank_partial=True, max_missing=0.05)
    assert np.array_equal(Pp[7], named[7]) and np.array_equal(Pp[9], named[9]) and ip["rows_rank"] == 48 and ip["partial"]
    # training rows: ranking required on every row
    write_rank(tmp_path, "tr", "r49", "bench", keep, lp[keep], "train")
    with pytest.raises(SystemExit):
        c95.channel(["runs/tr/proto_job/proto"], "runs/tr/r49/bench", n, CLASSES, SEEN, "train")
    # wrong table
    with pytest.raises(SystemExit):
        c95.channel(["runs/te/proto_job/proto"], "runs/te/r49/test", n, CLASSES, SEEN, "train")


def test_make_submission_rank_end_to_end(tmp_path, monkeypatch):
    """make_submission.main with --proto x=… --rank x=…: the test matrix is the blend of channel a and the mean channel (floor
    0.05 from the blend file) — recomputed by hand. Channel form and floor are checked against meta_x.json: --rank without
    mean-channel meta, mean-channel meta without --rank, a wrong floor and --rank without --proto abort."""
    ms = load("make_submission_e2e", COMP_ROOT / "tools" / "make_submission.py")
    for mod in (ms.a8, ms.c95.A8):
        monkeypatch.setattr(mod, "COMP", tmp_path)
    n_tr, n_te = 40, 50
    rng = np.random.default_rng(4)
    art, data = tmp_path / "artifacts", tmp_path / "data"
    (art / "submissions").mkdir(parents=True)
    (data / "competition_data").mkdir(parents=True)
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    ids = [f"{i:012x}" for i in range(n_te)]
    pd.DataFrame({"trace_id": ids, "root_code": "NONE"}).to_csv(data / "competition_data" / "sample_submission.csv", index=False)
    tables = {"train": pd.DataFrame({"trace_id": [f"t{i}" for i in range(n_tr)], "root_code": [CLASSES[c] for c in y]}),
              "test": pd.DataFrame({"trace_id": ids})}
    monkeypatch.setattr(ms, "COMP", tmp_path)
    monkeypatch.setattr(ms.C, "find_data_root", lambda hint=None: data)
    monkeypatch.setattr(ms.C, "load_table", lambda root, name: tables[name].copy())
    monkeypatch.setattr(ms.labels, "load_classes", lambda p: list(CLASSES))
    stack = {"channels": ["a", "x"], "none": [], "probes": {}, "theta": [0.6, 0.5], "floor": {"x": 0.05}}
    (art / "stack_m.json").write_text(json.dumps(stack))
    (art / "meta_x.json").write_text(json.dumps({"name": "x", "rank_mean": {"floor": 0.05, "proto_sources": ["runs/tr/proto_job/proto"],
                                                                         "rank_source": "runs/tr/rank_job/bench"}}))
    # channel a (test measurement as log probabilities)
    A = rng.dirichlet(np.ones(3), n_te)
    d = art / "runs" / "ra" / "ra" / "job"
    d.mkdir(parents=True)
    (d / "test_meta.json").write_text(json.dumps({"table": "test"}))
    np.save(d / "test_idx.npy", np.arange(n_te))
    np.save(d / "test_diag.npy", np.column_stack([np.log(A), np.full(n_te, NINF)]))
    recs = answers(n_te, rng, missing=(2,))
    lp = rank_logp(n_te, rng, none_rows=17)
    write_proto(tmp_path, "te", "proto_job", recs, "test")
    write_rank(tmp_path, "te", "rank_job", "test", np.arange(n_te), lp, "test")
    write_rank(tmp_path, "tr", "rank_job", "bench", np.arange(n_tr), rank_logp(n_tr, rng, none_rows=14), "train")
    recipe = {"proto_from": "derived/proto_notes/train.jsonl", "beams": 74, "image": {"max_side": 1024}}
    for run_, job_, tag_, tab_, pf_ in (("tr", "rank_job", "bench", "train", "train.jsonl"), ("te", "rank_job", "test", "test", "test.jsonl")):
        (art / "runs" / run_ / run_ / job_ / f"{tag_}_meta.json").write_text(json.dumps(
            {"table": tab_, "probes": ["proto_rank"], "dtype": "bf16", "quant": None, "adapter": False, "recipe": {**recipe, "proto_from": pf_}}))

    def run(*extra, name="m"):
        monkeypatch.setattr(sys, "argv", ["make_submission.py", "--name", name, "--test", "a=ra/job/test", *extra])
        return ms.main()

    assert run("--proto", "x=runs/te/proto_job/proto", "--rank", "x=runs/te/rank_job/test", "--suffix", "_xm") == 0
    got = np.load(art / "test_m_xm.npy")
    X = 0.5 * (by_hand_named(recs, n_te) + by_hand_rank(lp))
    Xf = 0.95 * X + 0.05 * SEEN / SEEN.sum()
    Pa = np.column_stack([A, np.zeros(n_te)])
    z = 0.6 * np.log(np.clip(Pa, 1e-9, 1)) + 0.5 * np.log(np.clip(Xf, 1e-9, 1))
    exp = np.where(SEEN[None, :], np.exp(z - z.max(1, keepdims=True)), 0.0)
    exp = exp / exp.sum(1, keepdims=True)
    assert np.allclose(got, exp, atol=1e-12, rtol=0)
    src = json.loads((art / "submissions" / "sub_m_xm.json").read_text())["sources"]["x"]
    assert any(s.get("rank") == "runs/te/rank_job/test" and s.get("rows_rank") == n_te for s in src)
    # ranking of the test rows with settings different from those of the training rows (here: a different number of beams)
    meta_te = art / "runs" / "te" / "te" / "rank_job" / "test_meta.json"
    good = meta_te.read_text()
    bad = json.loads(good)
    bad["recipe"]["beams"] = 18
    meta_te.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="computed differently"):
        run("--proto", "x=runs/te/proto_job/proto", "--rank", "x=runs/te/rank_job/test", "--suffix", "_bad0")
    meta_te.write_text(good)
    # channel form and floor against the meta file
    with pytest.raises(SystemExit, match="--rank x=runs/… missing"):
        run("--proto", "x=runs/te/proto_job/proto", "--suffix", "_bad1")
    with pytest.raises(SystemExit, match="also be listed under --proto"):
        run("--rank", "x=runs/te/rank_job/test", "--suffix", "_bad2")
    (art / "meta_x.json").write_text(json.dumps({"name": "x"}))
    with pytest.raises(SystemExit, match="does not describe a mean channel"):
        run("--proto", "x=runs/te/proto_job/proto", "--rank", "x=runs/te/rank_job/test", "--suffix", "_bad3")
    (art / "meta_x.json").write_text(json.dumps({"name": "x", "rank_mean": {"floor": 0.05, "proto_sources": ["p"], "rank_source": "runs/tr/rank_job/bench"}}))
    (art / "stack_m2.json").write_text(json.dumps({**stack, "floor": {"x": 0.01}}))
    with pytest.raises(SystemExit, match="floor of the blend"):
        run("--proto", "x=runs/te/proto_job/proto", "--rank", "x=runs/te/rank_job/test", "--suffix", "_bad4", name="m2")
    for sfx in ("_bad0", "_bad1", "_bad2", "_bad3", "_bad4"):
        assert not (art / "submissions" / f"sub_m{sfx}.csv").exists() and not (art / "submissions" / f"sub_m2{sfx}.csv").exists()
