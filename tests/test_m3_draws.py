"""M3 as the mean of several draws: eda/a9_m3_draws_oof.py, eda/c99_m3_deploy.py (rows with all draws, --versus, training
regime over several draws), tools/make_submission.py --rank-only <channel>=<test draw 1>,<test draw 2>.

Synthetic: a9 against a computation by hand (one draw bit-identical, two draws bit-identical to 0.5·(P_1 + P_2)) including
failure cases and --gen-status; the chain a9 -> c99 (blend file) -> make_submission with TWO faked test draws against a
computation by hand, with the guards per draw (order / max_tokens, coverage, NONE band, generation, provenance of the blend,
fit rows). Only checksums are printed."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["NONE", "A", "B", "C"]
SEEN = np.array([True, True, True, False])
U = SEEN / SEEN.sum()
NINF = -np.inf
RECIPE = {"image": {"max_side": 1024, "question_max_side": 448}, "beams": 74, "taxonomy": None, "model": "m3_f1", "fold": None, "proto_from": None}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A9 = load("t_a9", COMP_ROOT / "eda" / "a9_m3_draws_oof.py")  # puts src/ on sys.path
from vlm import common as VC  # noqa: E402
from vlm.engine import probs_from_logp  # noqa: E402


def rank_logp(n: int, rng: np.random.Generator, none_rows: int, mass: float = 0.97) -> np.ndarray:
    """Exact ranking: all classes with a training example finite; the first `none_rows` rows with NONE at the top."""
    lp = np.full((n, len(CLASSES)), NINF)
    for i in range(n):
        top = 0 if i < none_rows else 1 + i % 2
        p = rng.dirichlet(np.ones(3)) * 0.3
        p[top] += 0.7
        lp[i, :3] = np.log(p / p.sum() * mass)
    return lp


def probs(lp: np.ndarray) -> np.ndarray:
    return probs_from_logp(lp, absent=~SEEN)


def write_gen(base: Path, run: str, gen_job: str, table: str, rows: int, tokens: int, complete: int | None = None, system: str = "s1") -> None:
    """Generation behind a ranking: <gen_job>/gen_generation.json (settings) and gen_meta.json (aggregates)."""
    g = base / "artifacts" / "runs" / run / run / gen_job
    g.mkdir(parents=True, exist_ok=True)
    rec = {"table": table, "system_sha256": system, "model_dir": f"/models/{run}_{gen_job}", "backend": "vllm",
           "image": {"max_side": 1024, "min_side": None, "question_max_side": 448, "rotate_all": 0}, "rows_sha256": f"{run}/{gen_job}",
           "settings": {"max_tokens": tokens, "temperature": 0.0, "seed": None, "template_kwargs": None}}
    (g / "gen_generation.json").write_text(json.dumps({"sha256": "0", "recipe": rec}))
    c = rows if complete is None else complete
    (g / "gen_meta.json").write_text(json.dumps({"table": table, "rows": rows, "answered": rows, "failed": 0, "max_tokens": tokens,
                                                 "states": {"complete": c, "open": rows - c, "empty": 0}, "system_sha256": system}))


def write_rank(base: Path, run: str, job: str, tag: str, idx, lp, table: str, done=None, meta_extra: dict | None = None,
               gen_tokens: int | None = None, gen_rows: int | None = None) -> None:
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    meta = {"table": table, "n": int(len(idx)), "probes": ["proto_rank"], "job": job, "recipe": dict(RECIPE), "dtype": "bf16", "quant": None,
            "adapter": False, "sft": {"sft_from": f"{job}_gen/gen", "finish_tokens": 320}}
    for k, v in (meta_extra or {}).items():
        if k == "recipe":
            meta["recipe"].update(v)
        else:
            meta[k] = v
    (d / f"{tag}_meta.json").write_text(json.dumps(meta))
    np.save(d / f"{tag}_idx.npy", np.asarray(idx, dtype=np.int64))
    np.save(d / f"{tag}_proto_rank.npy", np.asarray(lp, dtype=float))
    np.save(d / f"{tag}_proto_rank.done.npy", np.ones(len(idx), bool) if done is None else np.asarray(done, bool))
    if gen_tokens is not None:
        write_gen(base, run, meta["sft"]["sft_from"].split("/")[0], table, len(idx) if gen_rows is None else gen_rows, gen_tokens)


def write_combined(base: Path, run: str, job: str, tag: str, parts: list[tuple[str, str]], idx, lp) -> None:
    """Combined ranking as in vlm/sft.combine_rank_runs: data + meta file with combined_from (run, job)."""
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{tag}_idx.npy", np.asarray(idx, dtype=np.int64))
    np.save(d / f"{tag}_proto_rank.npy", np.asarray(lp, dtype=float))
    np.save(d / f"{tag}_proto_rank.done.npy", np.ones(len(idx), bool))
    (d / f"{tag}_meta.json").write_text(json.dumps({"table": "train", "n": int(len(idx)), "probes": ["proto_rank"], "job": job,
                                                    "combined_from": [{"run": r, "job": j, "rows": 0, "done": 0} for r, j in parts]}))


def oof_draws(base: Path, n: int, rng: np.random.Generator, rows2: np.ndarray, y: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Two OOF draws: the first combined from two fold jobs (run tr, 3,200 tokens) over all rows, the second from two shards of
    different runs (+, 5,200 tokens) on rows2 only. -> (sources, ranking 1, ranking 2 [n])."""
    lp1 = rank_logp(n, rng, int((y == 0).sum()))  # y carries NONE in the first rows: NONE at the top exactly there
    lp2 = rank_logp(n, rng, int((y == 0).sum()))
    half = n // 2
    for j, rr in (("rank_f1", np.arange(half)), ("rank_f2", np.arange(half, n))):
        write_rank(base, "tr", j, "bench", rr, lp1[rr], "train", meta_extra={"recipe": {"model": f"m3_{j[-2:]}"}}, gen_tokens=3200)
    write_combined(base, "tr", "rank_c", "bench", [("tr", "rank_f1"), ("tr", "rank_f2")], np.arange(n), lp1)
    s1, s2 = rows2[: len(rows2) // 2], rows2[len(rows2) // 2:]
    write_rank(base, "l2", "rk_a", "bench", s1, lp2[s1], "train", meta_extra={"recipe": {"model": "m3l_a"}}, gen_tokens=5200)
    write_rank(base, "l3", "rk_b", "bench", s2, lp2[s2], "train", meta_extra={"recipe": {"model": "m3l_b"}}, gen_tokens=5200)
    return ["runs/tr/rank_c/bench", "runs/l2/rk_a/bench+runs/l3/rk_b/bench"], lp1, lp2


# ------------------------------------------------------------------------------------------------ a9

def test_a9_mean_of_draws(tmp_path, monkeypatch):
    """a9: draw 1 (combined) + draw 2 (two shards with +): rows with one draw carry its distribution BIT-IDENTICALLY, rows with two
    carry BIT-IDENTICALLY 0.5·(new + first); main writes artifact, meta and row list, --verify recomputes bit-identically; failure
    cases abort."""
    monkeypatch.setattr(A9.A8, "COMP", tmp_path)
    rng = np.random.default_rng(5)
    n = 40
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    rows2 = np.r_[np.arange(0, 8), np.arange(20, 28)]
    specs, lp1, lp2 = oof_draws(tmp_path, n, rng, rows2, y)
    P, cnt, draws, infos = A9.build(specs, n, SEEN)
    one = np.setdiff1d(np.arange(n), rows2)
    p1, p2 = probs(lp1), probs(lp2[rows2])
    assert (cnt[rows2] == 2).all() and (cnt[one] == 1).all()
    assert np.array_equal(P[one], p1[one])  # one draw: its distribution (a8 --rank)
    assert np.array_equal(P[rows2], 0.5 * (p2 + p1[rows2]))  # mean of two draws: 0.5·(new + pilot)
    assert infos[0]["max_tokens"] == 3200 and infos[1]["max_tokens"] == 5200
    assert [c["job"] for sh in infos[1]["shards"] for c in sh["constituents"]] == ["rk_a", "rk_b"]
    assert [c["job"] for c in infos[0]["shards"][0]["constituents"]] == ["rank_f1", "rank_f2"]
    # main: artifact, meta file, row list; --verify
    root = tmp_path / "data"
    root.mkdir()
    ctx = {"root": root, "folds": np.arange(n) % 5, "classes": list(CLASSES), "y": y, "seen": SEEN, "n": n}
    monkeypatch.setattr(A9, "COMP", tmp_path)
    monkeypatch.setattr(A9, "context", lambda: ctx)
    monkeypatch.setattr(sys, "argv", ["a9", "x2", "--draw", specs[0], "--draw", specs[1]])
    assert A9.main() == 0
    art = tmp_path / "artifacts"
    assert np.array_equal(np.load(art / "oof_x2.npy"), P)
    meta = json.loads((art / "meta_x2.json").read_text())
    rk = meta["rank"]
    assert rk["source"] == ",".join(specs) and [d["source"] for d in rk["draws"]] == specs and rk["rows"] == n and rk["rows_named_only"] == 0
    assert rk["rows_by_draws"] == {"1": n - len(rows2), "2": len(rows2)} and rk["rows_all_draws_file"] == "derived/x2_rows.npy"
    assert meta["rows_neutral"] == 0 and np.array_equal(np.load(root / "derived" / "x2_rows.npy"), np.sort(rows2))
    assert rk["rows_all_draws_sha256"] == A9.rows_sha256(np.sort(rows2))
    assert A9.verify("x2") == 0
    np.save(root / "derived" / "x2_rows.npy", np.sort(rows2)[:-1])
    assert A9.verify("x2") == 4  # the row list no longer matches
    np.save(root / "derived" / "x2_rows.npy", np.sort(rows2))
    # a foreign artifact of the same name is never overwritten
    (art / "meta_m3p5.json").write_text(json.dumps({"name": "m3p5", "rank": {"source": specs[0]}}))
    monkeypatch.setattr(sys, "argv", ["a9", "m3p5", "--draw", specs[0], "--draw", specs[1]])
    with pytest.raises(SystemExit, match="choose another name"):
        A9.main()
    assert not (art / "oof_m3p5.npy").exists()
    # failure cases of the construction
    with pytest.raises(SystemExit, match="at least two"):
        A9.build(specs[:1], n, SEEN)
    with pytest.raises(SystemExit, match="comma"):
        A9.build([specs[0], specs[0] + "," + specs[1]], n, SEEN)
    with pytest.raises(SystemExit, match="more than once"):  # the same fold job in two draws
        A9.build([specs[0], "runs/tr/rank_f1/bench"], n, SEEN)
    write_rank(tmp_path, "l4", "rk_ov", "bench", np.arange(4, 12), lp2[4:12], "train", gen_tokens=5200)
    with pytest.raises(SystemExit, match="covered twice"):
        A9.build([specs[0], "runs/l2/rk_a/bench+runs/l4/rk_ov/bench"], n, SEEN)
    with pytest.raises(SystemExit, match="without any draw"):
        A9.build(["runs/l2/rk_a/bench", "runs/l3/rk_b/bench"], n, SEEN)
    write_rank(tmp_path, "l5", "rk_mix", "bench", np.arange(30, 34), lp2[30:34], "train", gen_tokens=3200)
    with pytest.raises(SystemExit, match="generated differently"):
        A9.build([specs[0], specs[1] + "+runs/l5/rk_mix/bench"], n, SEEN)
    write_rank(tmp_path, "l6", "rk_nogen", "bench", np.arange(30, 34), lp2[30:34], "train")
    with pytest.raises(SystemExit, match="cannot be determined"):
        A9.build([specs[0], specs[1] + "+runs/l6/rk_nogen/bench"], n, SEEN)
    write_rank(tmp_path, "l9", "rk_beams", "bench", np.arange(30, 34), lp2[30:34], "train", meta_extra={"recipe": {"beams": 18}}, gen_tokens=5200)
    with pytest.raises(SystemExit, match="computed differently"):
        A9.build([specs[0], specs[1] + "+runs/l9/rk_beams/bench"], n, SEEN)
    assert meta["rank"]["draws_per_row"] == [int(c) for c in cnt]
    assert meta["rank"]["draws"][1]["rank_recipe"] == meta["rank"]["draws"][0]["rank_recipe"]
    done = np.ones(4, bool)
    done[1] = False
    write_rank(tmp_path, "l7", "rk_part", "bench", np.arange(30, 34), lp2[30:34], "train", done=done, gen_tokens=5200)
    with pytest.raises(SystemExit, match="not a complete, exact run"):
        A9.build([specs[0], specs[1] + "+runs/l7/rk_part/bench"], n, SEEN)
    write_rank(tmp_path, "l8", "rk_nodone", "bench", np.arange(30, 34), lp2[30:34], "train", gen_tokens=5200)
    (tmp_path / "artifacts" / "runs" / "l8" / "l8" / "rk_nodone" / "bench_proto_rank.done.npy").unlink()
    with pytest.raises(SystemExit, match="completeness mask"):
        A9.build([specs[0], specs[1] + "+runs/l8/rk_nodone/bench"], n, SEEN)


def test_a9_gen_status(tmp_path, monkeypatch):
    """--gen-status: rows = table, answered, failures, share of complete answers against the tipping limit (aggregates only)."""
    monkeypatch.setattr(A9.A8, "COMP", tmp_path)
    n_te = 50
    monkeypatch.setattr(VC, "find_data_root", lambda hint=None: tmp_path)
    monkeypatch.setattr(VC, "load_table", lambda root, nm: pd.DataFrame({"trace_id": [f"{i:012x}" for i in range(n_te)]}))
    rng = np.random.default_rng(2)
    write_rank(tmp_path, "te", "rk", "test", np.arange(n_te), rank_logp(n_te, rng, 17), "test", gen_tokens=5200)
    write_gen(tmp_path, "te", "rk_gen", "test", n_te, 5200, complete=30)  # share complete 0.6
    assert A9.gen_status("runs/te/rk/test", "test", 0.5, 0.05) == 0
    assert A9.gen_status("runs/te/rk/test", "test", 0.7, 0.05) == 3
    assert A9.gen_status("runs/te/rk/test", "train", 0.0, 0.05) == 3  # wrong table
    write_gen(tmp_path, "te", "rk_gen", "test", n_te - 3, 5200)  # not generated over the whole table
    assert A9.gen_status("runs/te/rk/test", "test", 0.0, 0.05) == 3
    assert A9.gen_status("runs/te/missing/test", "test", 0.0, 0.05) == 3


# ------------------------------------------------------------------------------------------------ c99

def test_c99_draws_helpers(tmp_path, monkeypatch):
    """c99: fit rows of a channel from several draws are only rows with all draws (row list with checksum); a channel from one draw
    is unchanged (None); the training-regime check resolves draws (comma) and shards (+), one test draw per call; the blend file
    records the number of draws and the row list — with one draw the entry stays as before."""
    c99 = load("c99_draws", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    root = tmp_path / "data"
    (root / "derived").mkdir(parents=True)
    rows_all = np.array([3, 5, 8, 13], dtype=np.int64)
    np.save(root / "derived" / "x_rows.npy", rows_all)
    meta = {"rank": {"source": "runs/a/r/bench,runs/b/r/bench", "draws": [{"source": "runs/a/r/bench"}, {"source": "runs/b/r/bench"}],
                     "rows_all_draws_file": "derived/x_rows.npy", "rows_all_draws_sha256": c99.rows_sha256(rows_all), "rows_by_draws": {"1": 10, "2": 4}}}
    assert c99.draws_check({"rank": {"source": "runs/a/r/bench"}}, np.arange(5), root) is None
    info = c99.draws_check(meta, np.array([3, 8]), root)
    assert info["draws"] == 2 and info["rows_all_draws"] == 4 and info["fit_rows"] == 2
    with pytest.raises(SystemExit, match="do not carry all"):
        c99.draws_check(meta, np.array([3, 4]), root)
    with pytest.raises(SystemExit, match="checksum"):
        c99.draws_check({**meta, "rank": {**meta["rank"], "rows_all_draws_sha256": "0"}}, np.array([3]), root)
    with pytest.raises(SystemExit, match="do not match"):
        c99.draws_check({**meta, "rank": {**meta["rank"], "draws": meta["rank"]["draws"][:1]}}, np.array([3]), root)
    with pytest.raises(SystemExit, match="missing"):
        c99.draws_check({**meta, "rank": {**meta["rank"], "rows_all_draws_file": "derived/nowhere.npy"}}, np.array([3]), root)
    # training regime: draws and shards resolved
    monkeypatch.setattr(c99, "COMP", tmp_path)
    rng = np.random.default_rng(1)
    for run, job in (("a", "f1"), ("a", "f2"), ("b", "s1"), ("c", "s2")):
        write_rank(tmp_path, run, job, "bench", np.arange(3), rank_logp(3, rng, 1), "train")
    write_combined(tmp_path, "a", "comb", "bench", [("a", "f1"), ("a", "f2")], np.arange(6), rank_logp(6, rng, 2))
    got = [(r, j) for r, j, _ in c99.rank_constituents("runs/a/comb/bench,runs/b/s1/bench+runs/c/s2/bench")]
    assert got == [("a", "f1"), ("a", "f2"), ("b", "s1"), ("c", "s2")]
    assert [(r, j) for r, j, _ in c99.rank_constituents("runs/b/s1/bench")] == [("b", "s1")]
    (tmp_path / "artifacts" / "meta_x.json").write_text(json.dumps({"rank": {"source": "runs/a/comb/bench"}}))
    with pytest.raises(SystemExit, match="ONE test draw"):
        c99.variant_check("runs/b/s1/bench,runs/c/s2/bench", "x", art=tmp_path / "artifacts")
    # blend file: rank_only entry
    spec = {"names": ["a", "x"], "floors": {"x": 0.05}, "think": None, "factor": 1.0, "label": "L", "last_label": "l"}
    ins = {"theta": {"with": {"names": ["a", "x"], "theta": [0.5, 0.5], "theta_before_factor": [0.5, 0.5]}}, "map3": {"with": 0.5}}
    rec1 = c99.stack_record(spec, "x", ins, None, rows_all, np.arange(20) % 5, {"x": "0" * 16}, {"rank": {"source": "runs/a/r/bench"}}, None)
    assert set(rec1["rank_only"]["x"]) == {"floor", "rank_source", "oof_sha256", "rows"}
    rec2 = c99.stack_record(spec, "x", ins, None, rows_all, np.arange(20) % 5, {"x": "0" * 16}, meta, None)
    assert rec2["rank_only"]["x"]["draws"] == 2 and rec2["rank_only"]["x"]["rows_all_draws_sha256"] == c99.rows_sha256(rows_all)
    assert rec2["rank_only"]["x"]["rank_source"] == meta["rank"]["source"]


# ------------------------------------------------------------------------------------------------ a9 -> c99 -> make_submission

def setup_ms(tmp_path: Path, monkeypatch, name: str, y: np.ndarray, n_te: int):
    """make_submission with synthetic tables; outputs under tmp_path (as in tests/test_c99_m3_deploy.py)."""
    ms = load(name, COMP_ROOT / "tools" / "make_submission.py")
    for mod in (ms.a8, ms.c95.A8):
        monkeypatch.setattr(mod, "COMP", tmp_path)
    art, data = tmp_path / "artifacts", tmp_path / "data"
    (art / "submissions").mkdir(parents=True, exist_ok=True)
    (data / "competition_data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "eda").mkdir(exist_ok=True)
    shutil.copy(COMP_ROOT / "eda" / "c21_margin_align.py", tmp_path / "eda" / "c21_margin_align.py")
    ids = [f"{i:012x}" for i in range(n_te)]
    pd.DataFrame({"trace_id": ids, "root_code": "NONE"}).to_csv(data / "competition_data" / "sample_submission.csv", index=False)
    tables = {"train": pd.DataFrame({"trace_id": [f"t{i}" for i in range(len(y))], "root_code": [CLASSES[c] for c in y]}),
              "test": pd.DataFrame({"trace_id": ids})}
    monkeypatch.setattr(ms, "COMP", tmp_path)
    monkeypatch.setattr(ms.C, "find_data_root", lambda hint=None: data)
    monkeypatch.setattr(ms.C, "load_table", lambda root, nm: tables[nm].copy())
    monkeypatch.setattr(ms.labels, "load_classes", lambda p: list(CLASSES))
    return ms, art, data


def write_diag(art: Path, run: str, P: np.ndarray) -> None:
    d = art / "runs" / run / run / "job"
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_meta.json").write_text(json.dumps({"table": "test"}))
    np.save(d / "test_idx.npy", np.arange(len(P)))
    with np.errstate(divide="ignore"):
        np.save(d / "test_diag.npy", np.log(P))


def mix_by_hand(chans: list[np.ndarray], theta: list[float]) -> np.ndarray:
    z = sum(t * np.log(np.clip(P, 1e-9, 1)) for t, P in zip(theta, chans))
    e = np.where(SEEN[None, :], np.exp(z - z.max(1, keepdims=True)), 0.0)
    return e / e.sum(1, keepdims=True)


def test_draws_chain_end_to_end(tmp_path, monkeypatch):
    """a9 builds the OOF artifact from two draws, c99 the blend file ONLY on the rows with both draws, and make_submission averages
    TWO faked test draws (3,200 / 5,200 tokens) per row — recomputed by hand (one row is missing in draw 1: there draw 2 alone;
    floor 0.05). Every guard per draw aborts without writing anything."""
    c99 = load("c99_chain", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    rng = np.random.default_rng(13)
    n, n_te = 40, 50
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    folds = np.arange(n) % 5
    ms, art, data = setup_ms(tmp_path, monkeypatch, "ms_draws", y, n_te)
    monkeypatch.setattr(A9.A8, "COMP", tmp_path)
    monkeypatch.setattr(A9, "COMP", tmp_path)
    monkeypatch.setattr(A9, "context", lambda: {"root": data, "folds": folds, "classes": list(CLASSES), "y": y, "seen": SEEN, "n": n})
    rows2 = np.flatnonzero(np.isin(folds, [1, 2]))  # the second draw only on two folds
    specs, _, _ = oof_draws(tmp_path, n, rng, rows2, y)
    monkeypatch.setattr(sys, "argv", ["a9", "x", "--draw", specs[0], "--draw", specs[1]])
    assert A9.main() == 0
    meta_x = json.loads((art / "meta_x.json").read_text())
    rows_all = np.load(data / "derived" / "x_rows.npy")
    assert np.array_equal(rows_all, rows2)
    # c99: weights only on the rows with both draws -> blend file (provenance: source, checksum, rows)
    A = rng.dirichlet(np.ones(3), n)
    mats = {"a": np.column_stack([A, np.zeros(n)]), "x": np.load(art / "oof_x.npy")}
    np.save(art / "oof_a.npy", mats["a"])
    rows, _ = c99.channel_rows(mats["x"], SEEN, meta_x, rows_all, min_rows=10)
    assert c99.draws_check(meta_x, rows, data)["draws"] == 2
    with pytest.raises(SystemExit, match="do not carry all"):
        c99.draws_check(meta_x, np.arange(n), data)
    ctx = {"y": y, "seen": SEEN, "n": n, "K": len(CLASSES), "folds": folds, "q_code": np.arange(n) % 7, "group": np.where(y == 0, "NONE", "seen")}
    spec = {"names": ["a", "x"], "names_base": ["a"], "floors": {"x": 0.05}, "think": None, "factor": 1.0, "last": "trans",
            "theta_prod": np.array([1.0]), "label": "test line", "last_label": "transductive"}
    ins = c99.insample(ctx, spec, mats, rows)
    oof_sha = {"a": "0" * 16, "x": c99.sha256_file(art / "oof_x.npy")[:16]}
    stack = c99.stack_record(spec, "x", ins, None, rows, folds, oof_sha, meta_x, None)
    (art / "stack_m.json").write_text(json.dumps(stack))
    theta = stack["theta"]
    # test side: channel a and two test draws of x (ranking settings as for the OOF draws, generation 3,200 / 5,200)
    Ta = rng.dirichlet(np.ones(3), n_te)
    write_diag(art, "ra", np.column_stack([Ta, np.zeros(n_te)]))
    lp_t1, lp_t2 = rank_logp(n_te, rng, 17), rank_logp(n_te, rng, 17)
    keep1 = np.setdiff1d(np.arange(n_te), [7])  # draw 1: 49 of 50 rows (98 %)
    write_rank(tmp_path, "te1", "rk1", "test", keep1, lp_t1[keep1], "test", meta_extra={"recipe": {"model": "m3_full"}}, gen_tokens=3200, gen_rows=n_te)
    write_rank(tmp_path, "te2", "rk2", "test", np.arange(n_te), lp_t2, "test", meta_extra={"recipe": {"model": "m3d2"}}, gen_tokens=5200)
    T1, T2 = "runs/te1/rk1/test", "runs/te2/rk2/test"

    def run(rank_only, *extra, name="m"):
        monkeypatch.setattr(sys, "argv", ["make_submission.py", "--name", name, "--test", "a=ra/job/test", "--rank-only", rank_only, *extra])
        return ms.main()

    assert run(f"x={T1},{T2}", "--suffix", "_ok") == 0
    X = 0.5 * (probs(lp_t1) + probs(lp_t2))
    X[7] = probs(lp_t2)[7]  # draw 2 only
    exp = mix_by_hand([np.column_stack([Ta, np.zeros(n_te)]), 0.95 * X + 0.05 * U], theta)
    assert np.allclose(np.load(art / "test_m_ok.npy"), exp, atol=1e-12, rtol=0)
    src = json.loads((art / "submissions" / "sub_m_ok.json").read_text())["sources"]["x"]
    assert src[0]["rank_only"] == f"{T1},{T2}" and src[0]["train_rank"] == ",".join(specs) and src[0]["draws"] == 2
    assert src[0]["rows_by_draws"] == {"1": 1, "2": n_te - 1} and src[0]["rows_uniform"] == 0 and src[0]["fit_rows"] == len(rows2)
    assert [d["rows_rank"] for d in src[0]["per_draw"]] == [n_te - 1, n_te] and src[-1] == {"floor": 0.05}
    # the inductive alignment from the fit rows works with the channel from two draws
    assert run(f"x={T1},{T2}", "--align-inductive", "0.75", "--suffix", "_ia") == 0
    assert json.loads((art / "submissions" / "sub_m_ia.json").read_text())["align_inductive"]["fit_rows"]["rows"] == len(rows2)
    # deliberate partial state: row 3 of draw 2 not computed -> there draw 1 alone
    done = np.ones(n_te, bool)
    done[3] = False
    write_rank(tmp_path, "te2", "rk2p", "test", np.arange(n_te), lp_t2, "test", done=done, meta_extra={"recipe": {"model": "m3d2"}}, gen_tokens=5200)
    assert run(f"x={T1},runs/te2/rk2p/test", "--rank-partial", "--suffix", "_part") == 0
    Xp = X.copy()
    Xp[3] = probs(lp_t1)[3]
    assert np.allclose(np.load(art / "test_m_part.npy"), mix_by_hand([np.column_stack([Ta, np.zeros(n_te)]), 0.95 * Xp + 0.05 * U], theta), atol=1e-12)
    # failure cases — none of them writes a file
    bad = []

    def expect(match, rank_only, *extra, name="m"):
        sfx = f"_bad{len(bad)}"
        bad.append((name, sfx))
        with pytest.raises(SystemExit, match=match):
            run(rank_only, *extra, "--suffix", sfx, name=name)

    expect("computed or generated differently", f"x={T2},{T1}")  # swapped: max_tokens 5,200 against OOF draw 1 (3,200)
    expect("test draw\\(s\\) given", f"x={T1}")  # one test draw for an artifact from two draws
    expect("test draw\\(s\\) given", f"x={T1},{T2},{T2}")
    expect("more than once", f"x={T1},{T1}")
    expect("not a complete, exact run", f"x={T1},runs/te2/rk2p/test")
    keep2 = np.setdiff1d(np.arange(n_te), [7, 8])  # 96 %
    write_rank(tmp_path, "te2", "r48", "test", keep2, lp_t2[keep2], "test", meta_extra={"recipe": {"model": "m3d2"}}, gen_tokens=5200, gen_rows=n_te)
    expect("at least 98", f"x={T1},runs/te2/r48/test")
    write_rank(tmp_path, "te2", "none", "test", np.arange(n_te), rank_logp(n_te, rng, 30), "test", meta_extra={"recipe": {"model": "m3d2"}}, gen_tokens=5200)
    expect("malfunction band", f"x={T1},runs/te2/none/test")
    write_rank(tmp_path, "te2", "nogen", "test", np.arange(n_te), lp_t2, "test", meta_extra={"recipe": {"model": "m3d2"}})
    expect("computed or generated differently", f"x={T1},runs/te2/nogen/test")  # generation cannot be determined
    write_rank(tmp_path, "te2", "beams", "test", np.arange(n_te), lp_t2, "test", meta_extra={"recipe": {"model": "m3d2", "beams": 18}}, gen_tokens=5200)
    expect("differently from", f"x={T1},runs/te2/beams/test")
    write_gen(tmp_path, "te2", "sys_gen", "test", n_te, 5200, system="other system text")
    write_rank(tmp_path, "te2", "sys", "test", np.arange(n_te), lp_t2, "test", meta_extra={"recipe": {"model": "m3d2"}, "sft": {"sft_from": "sys_gen/gen", "finish_tokens": 320}})
    expect("gen.system_sha256", f"x={T1},runs/te2/sys/test")
    write_rank(tmp_path, "te2", "m3k", "test", np.arange(n_te), lp_t2, "test",
               meta_extra={"recipe": {"model": "m3d2"}, "sft_examples": {"max_examples": 3, "max_chars": 900, "dropout": 0.0, "seed": 0}}, gen_tokens=5200)
    expect("sft_examples", f"x={T1},runs/te2/m3k/test")  # a test draw of another model variant against the M3 OOF draw
    expect("expected", f"x={T1}+{T2},{T2}")  # shards only exist on the OOF side
    expect("missing", f"x={T1},runs/te2/missing/test")
    for key, val, match in (("oof_sha256", "f" * 16, "is not the artifact"), ("rank_source", specs[0], "fitted on a different channel")):
        st = json.loads(json.dumps(stack))
        st["rank_only"]["x"][key] = val
        (art / f"stack_m_{key}.json").write_text(json.dumps(st))
        expect(match, f"x={T1},{T2}", name=f"m_{key}")
    st = json.loads(json.dumps(stack))
    st["fit_rows"] = sorted(st["fit_rows"] + [0])
    st["fit_rows_sha256"] = A9.rows_sha256(st["fit_rows"])
    (art / "stack_m_fit.json").write_text(json.dumps(st))
    expect("do not carry all", f"x={T1},{T2}", name="m_fit")
    np.save(data / "derived" / "x_rows.npy", rows_all[:-1])
    expect("checksum", f"x={T1},{T2}")
    np.save(data / "derived" / "x_rows.npy", rows_all)
    # several test draws for a channel from ONE draw: abort; one draw works as before
    np.save(art / "oof_y.npy", mats["x"])
    (art / "meta_y.json").write_text(json.dumps({"name": "y", "rank": {"source": specs[0], "rows_named_only": 0}}))
    st_y = {**stack, "channels": ["a", "y"], "floor": {"y": 0.05}, "rank_only": {"y": {"floor": 0.05}}}
    (art / "stack_my.json").write_text(json.dumps(st_y))
    expect("does not describe a channel made of several draws", f"y={T1},{T2}", name="my")
    assert run(f"y={T1}", "--suffix", "_single", name="my") == 0
    Y = probs(lp_t1)
    Y[7] = U
    assert np.allclose(np.load(art / "test_my_single.npy"), mix_by_hand([np.column_stack([Ta, np.zeros(n_te)]), 0.95 * Y + 0.05 * U], theta), atol=1e-12)
    for name, sfx in bad:
        assert not (art / "submissions" / f"sub_{name}{sfx}.csv").exists() and not (art / f"test_{name}{sfx}.npy").exists()
