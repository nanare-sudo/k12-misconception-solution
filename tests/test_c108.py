"""Third M3 draw: eda/c108_m3_three_draws.py, eda/a9_m3_draws_oof.py with THREE draws, tools/make_submission.py --rank-only
with three test draws (guard rank_only_draws_guard: exactly one test draw per OOF draw, order checked via generation and
ranking) and the build script tools/builds/s029_m3x3.sh.

Synthetic: a9 builds the mean of three draws (shards from several runs) bit-identical to a computation by hand; the chain
a9 -> c99 (blend file) -> make_submission averages three faked test draws and aborts on a swapped order, a missing draw, a
different temperature or a different seed; the cross-fit of c108 is honest (the label of a row never enters its own
prediction); quality guard and verdict rules; the check run (d3 := d2) passes all steps and meets the expectation "two equal
draws"; without d3, c108 aborts with exit 3; fold integrity and training-regime checks detect a leak, teacher targets and
foreign adapters. With real artifacts (skipped if they are missing): both anchors exact. Only checksums are printed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
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
RECIPE = {"image": {"max_side": 1024, "question_max_side": 448}, "beams": 74, "taxonomy": None, "model": "m", "fold": None, "proto_from": None}
G1 = {"max_tokens": 3200, "temperature": 0.0, "seed": None, "template_kwargs": None}
G2 = {"max_tokens": 5200, "temperature": 0.0, "seed": None, "template_kwargs": None}
G3 = {"max_tokens": 5200, "temperature": 0.6, "seed": 3, "template_kwargs": None, "top_p": 0.95, "top_k": 20}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C108 = load("t_c108", COMP_ROOT / "eda" / "c108_m3_three_draws.py")  # puts src/ on sys.path
A9 = C108.A9
from vlm.engine import probs_from_logp  # noqa: E402


def probs(lp: np.ndarray) -> np.ndarray:
    return probs_from_logp(lp, absent=~SEEN)


def rank_logp(n: int, rng: np.random.Generator, none_rows: int, mass: float = 0.97) -> np.ndarray:
    """Exact ranking: all classes with a training example finite; the first `none_rows` rows with NONE at the top."""
    lp = np.full((n, len(CLASSES)), NINF)
    for i in range(n):
        top = 0 if i < none_rows else 1 + i % 2
        p = rng.dirichlet(np.ones(3)) * 0.3
        p[top] += 0.7
        lp[i, :3] = np.log(p / p.sum() * mass)
    return lp


def put_gen(base: Path, run: str, gen_job: str, table: str, rows: int, settings: dict, complete: int | None = None, system: str = "s1") -> None:
    g = base / "artifacts" / "runs" / run / run / gen_job
    g.mkdir(parents=True, exist_ok=True)
    rec = {"table": table, "system_sha256": system, "model_dir": f"/models/{run}_{gen_job}", "backend": "vllm",
           "image": {"max_side": 1024, "min_side": None, "question_max_side": 448, "rotate_all": 0}, "rows_sha256": f"{run}/{gen_job}",
           "settings": dict(settings)}
    (g / "gen_generation.json").write_text(json.dumps({"sha256": "0", "recipe": rec}))
    c = rows if complete is None else complete
    (g / "gen_meta.json").write_text(json.dumps({"table": table, "rows": rows, "answered": rows, "failed": 0, "max_tokens": settings["max_tokens"],
                                                 "states": {"complete": c, "open": rows - c, "empty": 0}, "out_tokens_median": 1000}))


def put_rank(base: Path, run: str, job: str, tag: str, idx, lp, table: str, model: str, settings: dict | None, gen_rows: int | None = None,
             system: str = "s1", recipe: dict | None = None) -> None:
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    meta = {"table": table, "n": int(len(idx)), "probes": ["proto_rank"], "job": job, "recipe": {**RECIPE, "model": model, **(recipe or {})},
            "dtype": "bf16", "quant": None, "adapter": False, "sft": {"sft_from": f"{job}_gen/gen", "finish_tokens": 320}}
    (d / f"{tag}_meta.json").write_text(json.dumps(meta))
    np.save(d / f"{tag}_idx.npy", np.asarray(idx, dtype=np.int64))
    np.save(d / f"{tag}_proto_rank.npy", np.asarray(lp, dtype=float))
    np.save(d / f"{tag}_proto_rank.done.npy", np.ones(len(idx), bool))
    if settings is not None:
        put_gen(base, run, f"{job}_gen", table, len(idx) if gen_rows is None else gen_rows, settings, system=system)


def put_combined(base: Path, run: str, job: str, tag: str, parts: list[tuple[str, str]], idx, lp) -> None:
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{tag}_idx.npy", np.asarray(idx, dtype=np.int64))
    np.save(d / f"{tag}_proto_rank.npy", np.asarray(lp, dtype=float))
    np.save(d / f"{tag}_proto_rank.done.npy", np.ones(len(idx), bool))
    (d / f"{tag}_meta.json").write_text(json.dumps({"table": "train", "n": int(len(idx)), "probes": ["proto_rank"], "job": job,
                                                    "combined_from": [{"run": r, "job": j} for r, j in parts]}))


def three_oof_draws(base: Path, n: int, folds: np.ndarray, rng: np.random.Generator) -> tuple[list[str], list[np.ndarray]]:
    """d1 combined from two fold jobs (3,200, greedy), d2 two shards of different runs (5,200, greedy), d3 five fold shards of one
    run (5,200, sampled). -> (sources, rankings)."""
    lps = [rank_logp(n, rng, 14) for _ in range(3)]
    half = n // 2
    for j, rr in (("rank_f1", np.arange(half)), ("rank_f2", np.arange(half, n))):
        put_rank(base, "tr", j, "bench", rr, lps[0][rr], "train", f"m3_{j[-2:]}", G1)
    put_combined(base, "tr", "rank_c", "bench", [("tr", "rank_f1"), ("tr", "rank_f2")], np.arange(n), lps[0])
    s1, s2 = np.arange(0, 24), np.arange(24, n)
    put_rank(base, "l2", "rk_a", "bench", s1, lps[1][s1], "train", "m3l_a", G2)
    put_rank(base, "l3", "rk_b", "bench", s2, lps[1][s2], "train", "m3l_b", G2)
    for k in range(5):
        rr = np.flatnonzero(folds == k)
        put_rank(base, "d3r", f"m3d3_rank_f{k}", "bench", rr, lps[2][rr], "train", f"m3d3_f{k}", G3)
    return ["runs/tr/rank_c/bench", "runs/l2/rk_a/bench+runs/l3/rk_b/bench", "+".join(f"runs/d3r/m3d3_rank_f{k}/bench" for k in range(5))], lps


def setup_ms(tmp_path: Path, monkeypatch, name: str, y: np.ndarray, n_te: int):
    """make_submission with synthetic tables; outputs under tmp_path (as in tests/test_m3_draws.py)."""
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


# ------------------------------------------------------------------------------------------------ a9 + make_submission, three draws

def test_three_draws_chain(tmp_path, monkeypatch):
    """a9 averages THREE draws (shards from several runs) bit-identically as ((P1 + P2) + P3) / 3; c99 records three draws in the
    blend file; make_submission averages three test draws as by hand and aborts on any deviation of order, number, temperature or
    seed without writing anything."""
    c99 = load("t108_c99", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    rng = np.random.default_rng(31)
    n, n_te = 40, 50
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    folds = np.arange(n) % 5
    ms, art, data = setup_ms(tmp_path, monkeypatch, "t108_ms", y, n_te)
    monkeypatch.setattr(A9.A8, "COMP", tmp_path)
    monkeypatch.setattr(A9, "COMP", tmp_path)
    monkeypatch.setattr(A9, "context", lambda: {"root": data, "folds": folds, "classes": list(CLASSES), "y": y, "seen": SEEN, "n": n})
    specs, lps = three_oof_draws(tmp_path, n, folds, rng)
    P, cnt, _, infos = A9.build(specs, n, SEEN)
    want = np.zeros((n, len(CLASSES)))
    for lp in lps:
        want += probs(lp)
    assert np.array_equal(P, want / 3) and (cnt == 3).all()
    assert [i["max_tokens"] for i in infos] == [3200, 5200, 5200] and infos[2]["generation"]["settings"] == G3
    assert [c["job"] for sh in infos[2]["shards"] for c in sh["constituents"]] == [f"m3d3_rank_f{k}" for k in range(5)]
    with pytest.raises(SystemExit, match="more than once"):  # the same draw twice (as in the check run) is not allowed by a9.build
        A9.build([specs[0], specs[1], specs[1]], n, SEEN)
    monkeypatch.setattr(sys, "argv", ["a9", "x3", "--draw", specs[0], "--draw", specs[1], "--draw", specs[2], "--rows-out", "derived/x3_rows.npy"])
    assert A9.main() == 0
    meta = json.loads((art / "meta_x3.json").read_text())
    rk = meta["rank"]
    assert rk["source"] == ",".join(specs) and rk["rows_by_draws"] == {"3": n} and len(rk["draws"]) == 3 and rk["rows_all_draws"] == n
    assert np.array_equal(np.load(art / "oof_x3.npy"), P) and A9.verify("x3") == 0
    assert C108.npy_sha256(P) == hashlib.sha256((art / "oof_x3.npy").read_bytes()).hexdigest()  # the same bytes as save_oof
    # c99: blend file on all rows (all carry three draws)
    A = rng.dirichlet(np.ones(3), n)
    mats = {"a": np.column_stack([A, np.zeros(n)]), "x3": np.load(art / "oof_x3.npy")}
    np.save(art / "oof_a.npy", mats["a"])
    rows_all = np.load(data / "derived" / "x3_rows.npy")
    rows, _ = c99.channel_rows(mats["x3"], SEEN, meta, rows_all, min_rows=10)
    assert c99.draws_check(meta, rows, data)["draws"] == 3
    ctx = {"y": y, "seen": SEEN, "n": n, "K": len(CLASSES), "folds": folds, "q_code": np.arange(n) % 7, "group": np.where(y == 0, "NONE", "seen")}
    spec = {"names": ["a", "x3"], "names_base": ["a"], "floors": {"x3": 0.05}, "think": None, "factor": 1.0, "last": "trans",
            "theta_prod": np.array([1.0]), "label": "test line", "last_label": "transductive"}
    ins = c99.insample(ctx, spec, mats, rows)
    stack = c99.stack_record(spec, "x3", ins, None, rows, folds, {"a": "0" * 16, "x3": c99.sha256_file(art / "oof_x3.npy")[:16]}, meta, None)
    assert stack["rank_only"]["x3"]["draws"] == 3
    (art / "stack_m.json").write_text(json.dumps(stack))
    # three test draws with the settings of their OOF draw
    Ta = rng.dirichlet(np.ones(3), n_te)
    write_diag(art, "ra", np.column_stack([Ta, np.zeros(n_te)]))
    lt = [rank_logp(n_te, rng, 17) for _ in range(3)]
    put_rank(tmp_path, "te1", "rk1", "test", np.arange(n_te), lt[0], "test", "m3_full", G1)
    put_rank(tmp_path, "te2", "rk2", "test", np.arange(n_te), lt[1], "test", "m3d2", G2)
    put_rank(tmp_path, "te3", "rk3", "test", np.arange(n_te), lt[2], "test", "m3d3", G3)
    put_rank(tmp_path, "te3", "greedy", "test", np.arange(n_te), lt[2], "test", "m3d3", G2)  # d3 generated greedily
    put_rank(tmp_path, "te3", "seed4", "test", np.arange(n_te), lt[2], "test", "m3d3", {**G3, "seed": 4})
    put_rank(tmp_path, "te3", "topk", "test", np.arange(n_te), lt[2], "test", "m3d3", {k: v for k, v in G3.items() if k != "top_k"})
    T1, T2, T3 = "runs/te1/rk1/test", "runs/te2/rk2/test", "runs/te3/rk3/test"

    def run(rank_only, *extra):
        monkeypatch.setattr(sys, "argv", ["make_submission.py", "--name", "m", "--test", "a=ra/job/test", "--rank-only", rank_only, *extra])
        return ms.main()

    assert run(f"x3={T1},{T2},{T3}", "--suffix", "_ok") == 0
    X = np.zeros((n_te, len(CLASSES)))
    for lp in lt:
        X += probs(lp)
    X /= 3
    exp = mix_by_hand([np.column_stack([Ta, np.zeros(n_te)]), 0.95 * X + 0.05 * U], stack["theta"])
    assert np.allclose(np.load(art / "test_m_ok.npy"), exp, atol=1e-12, rtol=0)
    src = json.loads((art / "submissions" / "sub_m_ok.json").read_text())["sources"]["x3"][0]
    assert src["draws"] == 3 and src["rows_by_draws"] == {"3": n_te} and src["train_rank"] == ",".join(specs)
    assert [d["train_rank"] for d in src["per_draw"]] == specs
    bad = []
    for match, spec_te in (("computed or generated differently", f"x3={T1},{T3},{T2}"),  # d2 and d3 swapped (temperature)
                           ("computed or generated differently", f"x3={T2},{T1},{T3}"),  # d1 and d2 swapped (max_tokens)
                           ("test draw\\(s\\) given", f"x3={T1},{T2}"),        # the third one is missing
                           ("more than once", f"x3={T1},{T2},{T2}"),
                           ("gen.settings", f"x3={T1},{T2},runs/te3/greedy/test"),
                           ("gen.settings", f"x3={T1},{T2},runs/te3/seed4/test"),
                           ("gen.settings", f"x3={T1},{T2},runs/te3/topk/test")):
        sfx = f"_bad{len(bad)}"
        bad.append(sfx)
        with pytest.raises(SystemExit, match=match):
            run(spec_te, "--suffix", sfx)
    for sfx in bad:
        assert not (art / "submissions" / f"sub_m{sfx}.csv").exists() and not (art / f"test_m{sfx}.npy").exists()


# ------------------------------------------------------------------------------------------------ c108: recipe and rules

def synth_ctx(n: int = 150, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 3, int), rng.integers(1, 3, n - n // 3)]
    rng.shuffle(y)
    folds = np.arange(n) % 5
    q = rng.integers(0, 30, n)
    group = np.where(y == 0, "NONE", np.where(rng.random(n) < 0.5, "seen", "never seen"))
    return {"y": y, "seen": SEEN, "folds": folds, "n": n, "K": len(CLASSES), "q_code": q, "group": group}


def noisy_channel(y: np.ndarray, rng: np.random.Generator, hit: float) -> np.ndarray:
    P = rng.dirichlet(np.ones(3), len(y))
    P[np.arange(len(y)), y] += hit * rng.random(len(y))
    P = np.column_stack([P, np.zeros(len(y))])
    return P / P.sum(1, keepdims=True)


SYN_LINES = {
    "best": {"label": "B", "names": ["a", "b"], "floors": {"b": 0.01}, "think": "b", "factor": 1.4, "beta": 0.2, "last": "trans",
             "anchor": None, "stack": None, "last_label": "transductive"},
    "safe": {"label": "S", "names": ["a", "c"], "floors": {"c": 0.05}, "think": None, "factor": 1.0, "beta": 0.4, "last": "ind",
             "anchor": None, "stack": None, "last_label": "inductive"},
}


def test_cross_fit_is_honest():
    """Cross-fit of c108: the label of a row never enters its own prediction (weights, target of the alignment and inductive class
    weights only from the other folds); weights per fold = a2.fit_mix on the other folds × thinking factor."""
    ctx = synth_ctx()
    rng = np.random.default_rng(3)
    y, folds = ctx["y"], ctx["folds"]
    mats = {nm: noisy_channel(y, rng, h) for nm, h in (("a", 1.0), ("b", 2.0), ("c", 0.5))}
    for last in ("trans", "ind", "none"):
        r1 = C108.cross_fit(mats, ["a", "b"], {"b": 0.01}, "b", 1.4, 0.3, last, y, SEEN, folds)
        y2 = y.copy()
        E = np.flatnonzero(folds == 2)
        y2[E] = rng.permutation(np.r_[y[E][1:], (y[E][0] + 1) % 3])  # labels of the evaluated fold changed
        r2 = C108.cross_fit(mats, ["a", "b"], {"b": 0.01}, "b", 1.4, 0.3, last, y2, SEEN, folds)
        assert np.array_equal(r1["Q"][E], r2["Q"][E]), last
        assert np.allclose(r1["Q"].sum(1), 1.0) and (r1["Q"][:, ~SEEN] == 0).all()
    logs = C108.C99.logs_of([mats["a"], mats["b"]], [0.0, 0.01], SEEN, mask_first=False)
    F = np.flatnonzero(folds != 0)
    th = np.asarray(C108.A2.fit_mix([lg[F] for lg in logs], None, y[F]), dtype=float)
    assert r1["theta"][0] == [float(th[0]), float(th[1] * 1.4)]


def test_quality_guard():
    """Guard: violated only if d3 − d2 < −0.02 AND z ≤ −2 (σ over tasks)."""
    rng = np.random.default_rng(1)
    n = 1200
    q = rng.integers(0, 300, n)
    r2 = rng.choice([0.0, 1 / 3, 0.5, 1.0], n)
    assert not C108.quality_guard(r2.copy(), r2, q)["violated"]  # equal: Δ 0
    big = r2 - np.where(rng.random(n) < 0.3, 0.2, 0.0)  # Δ ≈ −0.06, very certain
    g = C108.quality_guard(big, r2, q)
    assert g["violated"] and g["delta"] < -0.02 and g["z"] <= -2
    small = r2 - np.where(rng.random(n) < 0.05, 0.1, 0.0)  # Δ ≈ −0.005: certain, but too small
    g = C108.quality_guard(small, r2, q)
    assert not g["violated"] and g["z"] <= -2
    noisy = r2 + 0.8 * rng.permutation(np.r_[np.ones(n // 2), -np.ones(n // 2)]) - 0.025  # Δ = −0.025 exactly, but z ≈ −1
    g = C108.quality_guard(noisy, r2, q)
    assert abs(g["delta"] + 0.025) < 1e-12 and g["z"] > -2 and not g["violated"]


def test_decide_rules():
    """Verdict rules: guard -> both NO; primary line Δ > 0 and single channel ≥ m3x2 − 0.002 (posterior sign for the report only,
    disagreement flagged); inductive line P(θ<−0.002) ≤ 0.05, not decidable -> NO."""
    ok_g, bad_g = {"violated": False}, {"violated": True}
    post = lambda m, p, dec=True: {"mean": m, "p_lt_m002": p, "decidable": dec}  # noqa: E731
    d = C108.decide(ok_g, 0.001, 0.797, 0.789, post(-0.0005, 0.08), post(0.003, 0.04))
    assert d["best"]["deploy"] and d["best"]["posterior_sign_disagrees"] and d["safe"]["deploy"] and d["safe"]["mean_ge_c_txt"]
    d = C108.decide(bad_g, 0.004, 0.80, 0.789, post(0.004, 0.0), post(0.005, 0.0))
    assert d["guard_violated"] and not d["best"]["deploy"] and not d["safe"]["deploy"]
    assert not C108.decide(ok_g, 0.0, 0.797, 0.789, None, None)["best"]["deploy"]  # Δ = 0 is not > 0
    assert not C108.decide(ok_g, 0.002, 0.7865, 0.789, None, None)["best"]["deploy"]  # single channel 0.0025 worse
    assert C108.decide(ok_g, 0.002, 0.7871, 0.789, None, None)["best"]["deploy"]  # exactly at the limit − 0.0019
    assert not C108.decide(ok_g, 0.002, 0.8, 0.789, None, post(0.004, 0.051))["safe"]["deploy"]
    s = C108.decide(ok_g, 0.002, 0.8, 0.789, None, post(None, None, dec=False))["safe"]
    assert not s["deploy"] and not s["decidable"]
    text = "\n".join(C108.verdict_lines(C108.decide(ok_g, 0.001, 0.797, 0.789, post(-0.0005, 0.08), post(0.003, 0.04)), dry=True))
    assert "CHECK" in text and "NO VERDICT" in text and "DIFFERS FROM THE SIGN" in text


def test_evaluate_probe_synthetic():
    """Check run d3 := d2 on synthetic data: all steps run; guard Δ exactly 0; m3x3 = mean(d1, d2, d2); the null point
    "copy d3 := d2" is m3x3 itself (gain on top exactly 0); cross values = cross_fit; verdict only as a check."""
    if not (COMP_ROOT / "eda" / "c95_common.py").is_file():
        pytest.skip("the posterior step of the decision rule needs eda/c95_common.py, which is not part of this repository")
    ctx = synth_ctx()
    rng = np.random.default_rng(5)
    y, n = ctx["y"], ctx["n"]
    base = {nm: noisy_channel(y, rng, h) for nm, h in (("a", 1.0), ("b", 2.0), ("c", 0.5))}
    P1, P2 = noisy_channel(y, rng, 1.5), noisy_channel(y, rng, 1.5)
    h = np.ones(n, bool)
    m3x2, _ = A9.mean_of_draws([(P1, h), (P2, h)], n, len(CLASSES))
    m3x3, _ = A9.mean_of_draws([(P1, h), (P2, h), (P2.copy(), h)], n, len(CLASSES))
    D = {"P": [P1, P2, P2.copy()], "m3x2": m3x2, "m3x3": m3x3, "specs": ["d1", "d2", "d3"], "info": [{}, {}, {}]}
    lines = []
    res, rows = C108.evaluate(ctx, base, D, True, lines=SYN_LINES, anchors=False, n_sim=20_000, chunks=1, say=lines.append, B=400)
    assert res["guard"]["delta"] == 0.0 and not res["guard"]["violated"] and res["probe"]["guard_delta_zero"]
    for key, L in SYN_LINES.items():
        c3 = C108.line_cross(ctx, base, m3x3, L)
        assert res["lines"][key]["cross_m3x3"] == float(C108.rows_map(y, c3["Q"], SEEN).mean())
        assert res["null_point"]["copy d3 := d2"]["lines"][key]["new"]["mean"] == 0.0
        assert len(res["lines"][key]["delta"]["per_fold"]) == 5 and "ll_delta" in res["lines"][key]["delta"]
    assert set(res["decision"]) >= {"guard_violated", "best", "safe"} and "boundaries" in res["posterior"]
    assert res["single"]["d3"] == res["single"]["d2"] and np.array_equal(rows["single_d3"], rows["single_d2"])
    assert any("CHECK" in s for s in lines) and any(s.startswith("VERDICT") for s in lines)


def test_main_exit3_without_d3(tmp_path, monkeypatch, capsys):
    """Without the third OOF draw: --status and the real run abort with exit 3 and a clear message, nothing written."""
    ctx = synth_ctx()
    monkeypatch.setattr(C108, "context", lambda: ctx)
    monkeypatch.setattr(A9.A8, "COMP", tmp_path)
    out_dir = tmp_path / "out"
    assert C108.main(["--status", "--out-dir", str(out_dir)]) == 3
    assert C108.main(["--out-dir", str(out_dir)]) == 3
    cap = capsys.readouterr()
    assert "run not fetched" in cap.out and "ABORT" in cap.err and "--dry" in cap.err
    assert not out_dir.exists()
    # an existing but incomplete shard reports the partial state
    lp = rank_logp(30, np.random.default_rng(0), 10)
    put_rank(tmp_path, "m3d3_oof_rp", "m3d3_rank_f0", "bench", np.flatnonzero(ctx["folds"] == 0), lp, "train", "m3d3_f0", G3)
    d = tmp_path / "artifacts" / "runs" / "m3d3_oof_rp" / "m3d3_oof_rp" / "m3d3_rank_f0"
    done = np.ones(30, bool)
    done[3] = False
    np.save(d / "bench_proto_rank.done.npy", done)
    st = C108.draw_status(C108.D3_DEFAULT, ctx["n"], SEEN)
    assert not st["ok"] and st["shards"][0]["done"] == 29 and not st["shards"][0]["ok"] and st["shards"][1]["missing"]


def test_fold_check_detects_leak(tmp_path, monkeypatch):
    """Fold integrity/training regime: rows from fold k, the adapter holds out fold k, gold-only, merged adapter = local one -> no
    finding; every deviation (other fold, teacher targets, foreign adapter, rows from two folds) is a finding; a missing
    configuration is "not determinable"."""
    monkeypatch.setattr(A9.A8, "COMP", tmp_path)
    monkeypatch.setattr(C108, "COMP", tmp_path)
    folds = np.arange(50) % 5
    cfg = {"models": {"m_f0": {"merged": "m_f0_merged"}},
           "workers": [{"jobs": [{"name": "merge_f0", "kind": "merge", "adapter_from": "../tr/train_f0", "out_model": "m_f0_merged"}]}]}
    (tmp_path / "configs" / "vlm").mkdir(parents=True)
    import yaml

    (tmp_path / "configs" / "vlm" / "r.yaml").write_text(yaml.safe_dump(cfg))
    tr = tmp_path / "artifacts" / "runs" / "tr" / "tr" / "train_f0"
    (tr / "adapter").mkdir(parents=True)
    (tr / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter-f0")
    mg = tmp_path / "artifacts" / "runs" / "r" / "r" / "merge_f0"
    mg.mkdir(parents=True)
    (mg / "summary.json").write_text(json.dumps({"merge": {"adapter_sha256": hashlib.sha256(b"adapter-f0").hexdigest()}}))

    def recipe(**kw):
        (tr / "sft_recipe.json").write_text(json.dumps({"fold": 0, "teacher_derived": False, "pool": None, "system_sha256": "s1", "rows_used": 40, **kw}))

    rows0 = np.flatnonzero(folds == 0)
    put_rank(tmp_path, "r", "rank_f0", "bench", rows0, rank_logp(len(rows0), np.random.default_rng(0), 3), "train", "m_f0", G3)
    recipe()
    rec = C108.fold_check("runs/r/rank_f0/bench", folds, {})[0]
    assert not rec["problems"] and not rec["unknown"] and rec["train_fold"] == 0 and rec["gold_only"] and rec["adapter_matches_merge"]
    recipe(fold=1)
    assert any("holds out fold 1" in p for p in C108.fold_check("runs/r/rank_f0/bench", folds, {})[0]["problems"])
    recipe(teacher_derived=True)
    assert any("teacher" in p for p in C108.fold_check("runs/r/rank_f0/bench", folds, {})[0]["problems"])
    recipe(system_sha256="other")
    assert any("system text" in p for p in C108.fold_check("runs/r/rank_f0/bench", folds, {})[0]["problems"])
    recipe()
    (tr / "adapter" / "adapter_model.safetensors").write_bytes(b"other adapter")
    assert any("not the merged one" in p for p in C108.fold_check("runs/r/rank_f0/bench", folds, {})[0]["problems"])
    (tr / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter-f0")
    mixed = np.r_[rows0[:5], np.flatnonzero(folds == 1)[:5]]
    put_rank(tmp_path, "r", "rank_mix", "bench", mixed, rank_logp(10, np.random.default_rng(1), 3), "train", "m_f0", G3)
    assert any("several folds" in p for p in C108.fold_check("runs/r/rank_mix/bench", folds, {})[0]["problems"])
    put_rank(tmp_path, "no_cfg", "rank_f0", "bench", rows0, rank_logp(len(rows0), np.random.default_rng(2), 3), "train", "m_f0", G3)
    rec = C108.fold_check("runs/no_cfg/rank_f0/bench", folds, {})[0]
    assert rec["unknown"] and not rec["problems"]


# ------------------------------------------------------------------------------------------------ build script

def test_build_script_s029():
    """s029: syntax; sources of the three draws equal to c108 (and d1/d2 equal to meta_m3x2_full_cv01.json, if present); a9
    command, c108 verdict, training regime per test draw, c99 --save on the rows with all draws, three test draws in OOF order,
    last step per line; never a submission."""
    path = COMP_ROOT / "tools" / "builds" / "s029_m3x3.sh"
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    s = path.read_text()

    def var(name: str) -> str:
        line = next(ln for ln in s.splitlines() if ln.startswith(f"{name}="))
        return line.split("=", 1)[1].strip().strip('"')

    assert var("D3") == C108.D3_DEFAULT
    assert var("EVAL") == str((C108.OUT / f"{C108.NAME}.json").relative_to(COMP_ROOT))
    meta = COMP_ROOT / "artifacts" / f"meta_{C108.REF_CHANNEL}.json"
    if meta.is_file():
        assert f"{var('D1')},{var('D2')}" == json.loads(meta.read_text())["rank"]["source"]
    for need in ('a9_m3_draws_oof.py "$CH" --draw "$D1" --draw "$D2" --draw "$D3" --rows-out "$ROWS" --ref m3p5',
                 'a9_m3_draws_oof.py --verify "$CH"', '--rank-only "$CH=$RANK1,$RANK2,$RANK3"', 'c95_rank_mean.py --status "$R" --table test',
                 '--gen-status "$RANK3" --table test --min-complete "$D3_MIN_COMPLETE"', '--variant "$R" --channel m3p5 --require-gold-only',
                 'c99_m3_deploy.py --channel "$CH" --rows "$ROWS" --lines "$LINE" --save', "--align 0.75 --align-min 3", "--align-inductive 0.75 --align-min 3",
                 'ev.get("mode") != "real"', 'm3x3_npy_sha256', "stack_cv01_best_l2.json", "inductive_check", "IGNORE_VERDICT"):
        assert need in s, need
    assert "kaggle competitions submit" not in s and not any(f"submit.{ext}" in s for ext in ("py", "sh"))
    assert var("RANK3") == "runs/m3d3te_rp/m3d3_test_rank/test"


# ------------------------------------------------------------------------------------------------ real artifacts

def _real_missing() -> str | None:
    art = COMP_ROOT / "artifacts"
    need = [COMP_ROOT / "data" / "train.csv", art / f"oof_{C108.REF_CHANNEL}.npy", art / "stack_cv01_best_l2.json",
            art / "stack_v1r_zs27t_think18sb_pr27xm_m3x2_ar.json"] + [art / f"oof_{k}.npy" for k in C108.BASE_SHA16]
    miss = [str(p.relative_to(COMP_ROOT)) for p in need if not p.exists()]
    return ", ".join(miss) if miss else None


@pytest.mark.skipif(_real_missing() is not None, reason=f"inputs missing locally: {_real_missing()}")
def test_real_anchors(monkeypatch):
    """Real training artifacts: both anchors of the re-implementation exact (primary line 0.8198889 with weights per fold
    bit-identical to its blend file, inductive line 0.8120000) and the recipes match the blend files."""
    monkeypatch.setenv("K12_DATA", str(COMP_ROOT / "data"))  # other tests set K12_DATA to dummies
    c108 = load("t_c108_real", COMP_ROOT / "eda" / "c108_m3_three_draws.py")
    ctx = c108.context()
    base = c108.load_base()
    m3x2 = np.load(COMP_ROOT / "artifacts" / f"oof_{c108.REF_CHANNEL}.npy")
    assert c108.check_line_stacks(c108.LINES, COMP_ROOT / "artifacts") == []
    stack = json.loads((COMP_ROOT / "artifacts" / "stack_cv01_best_l2.json").read_text())
    for key, L in c108.LINES.items():
        res = c108.line_cross(ctx, base, m3x2, L)
        got = float(c108.rows_map(ctx["y"], res["Q"], ctx["seen"]).mean())
        assert abs(got - L["anchor"]) < c108.ANCHOR_TOL, (key, got)
        if key == "best":
            assert all(res["theta"][k] == stack["cross"]["fold_theta"][str(k)] for k in range(5))
