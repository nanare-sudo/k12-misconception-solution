"""Deployment path of the M3 channel (eda/c99_m3_deploy.py, tools/make_submission.py --rank-only).

Synthetic: the rank-only channel along the whole test path (make_submission.main) against a computation by hand, with all
guards; the blend from c99 (channel rows only) up to the test matrix with the inductive alignment from the same rows; the
training-regime check (gold-only / teacher targets). With real artifacts (skipped if they are missing): c99 reproduces the
weights of the deployed blend files."""
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


def rank_logp(n: int, rng: np.random.Generator, none_rows: int, mass: float = 0.97) -> np.ndarray:
    """Exact ranking: all classes with a training example finite; the first `none_rows` rows with NONE at the top."""
    lp = np.full((n, len(CLASSES)), NINF)
    for i in range(n):
        top = 0 if i < none_rows else 1 + i % 2
        p = rng.dirichlet(np.ones(3)) * 0.3
        p[top] += 0.7
        lp[i, :3] = np.log(p / p.sum() * mass)
    return lp


def by_hand_rank(lp: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(lp), np.exp(np.where(np.isfinite(lp), lp, 0.0)), 0.0)
    return p / p.sum(1, keepdims=True)


def write_rank(base: Path, run: str, job: str, tag: str, idx, lp, table: str, done=None, meta_extra: dict | None = None) -> None:
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


def write_combined(base: Path, run: str, job: str, tag: str, parts: list[str]) -> None:
    """Combined ranking as in vlm/sft.combine_rank (meta file only; the fold jobs carry the settings)."""
    d = base / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}_meta.json").write_text(json.dumps({"table": "train", "probes": ["proto_rank"], "job": job,
                                                    "combined_from": [{"job": p, "rows": 10, "done": 10} for p in parts]}))


def setup_ms(tmp_path: Path, monkeypatch, name: str, y: np.ndarray, n_te: int):
    """make_submission with synthetic tables; outputs under tmp_path."""
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
    return ms, art


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


# ------------------------------------------------------------------------------------------------ --rank-only

def test_rank_only_end_to_end(tmp_path, monkeypatch):
    """make_submission.main with --rank-only x=…: channel x = exact ranking, the row outside the probe uniform, floor 0.05 from
    the blend file — recomputed by hand; guards (coverage, mass, NONE band, partial state, settings per fold job, channel form,
    floor, requirement from the blend file, duplicate listing) abort without writing anything."""
    n_tr, n_te = 40, 50
    rng = np.random.default_rng(7)
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    ms, art = setup_ms(tmp_path, monkeypatch, "ms_rank_only", y, n_te)
    stack = {"channels": ["a", "x"], "none": [], "probes": {}, "theta": [0.6, 0.5], "floor": {"x": 0.05}, "rank_only": {"x": {"floor": 0.05}}}
    (art / "stack_m.json").write_text(json.dumps(stack))
    (art / "meta_x.json").write_text(json.dumps({"name": "x", "rank": {"source": "runs/tr/rank_c/bench", "rows_named_only": 0}}))
    write_combined(tmp_path, "tr", "rank_c", "bench", ["rank_f1", "rank_f2"])
    for j, f in (("rank_f1", "m3_f1"), ("rank_f2", "m3_f2")):
        write_rank(tmp_path, "tr", j, "bench", np.arange(n_tr), rank_logp(n_tr, rng, 14), "train", meta_extra={"recipe": {"model": f}})
    A = rng.dirichlet(np.ones(3), n_te)
    write_diag(art, "ra", np.column_stack([A, np.zeros(n_te)]))
    lp = rank_logp(n_te, rng, none_rows=17)
    keep = np.setdiff1d(np.arange(n_te), [7])  # 49 of 50 rows (98 %): admissible
    test_meta = {"recipe": {"model": "m3p_full"}, "sft": {"sft_from": "m3p_test_gen/gen", "finish_tokens": 320}}
    write_rank(tmp_path, "te", "rank_job", "test", keep, lp[keep], "test", meta_extra=test_meta)

    def run(*extra, name="m"):
        monkeypatch.setattr(sys, "argv", ["make_submission.py", "--name", name, "--test", "a=ra/job/test", *extra])
        return ms.main()

    assert run("--rank-only", "x=runs/te/rank_job/test", "--suffix", "_ok") == 0
    X = by_hand_rank(lp)
    X[7] = U  # outside the probe: uniform, like the neutral rows of the OOF artifact
    exp = mix_by_hand([np.column_stack([A, np.zeros(n_te)]), 0.95 * X + 0.05 * U], [0.6, 0.5])
    assert np.allclose(np.load(art / "test_m_ok.npy"), exp, atol=1e-12, rtol=0)
    src = json.loads((art / "submissions" / "sub_m_ok.json").read_text())["sources"]["x"]
    assert src[0]["rank_only"] == "runs/te/rank_job/test" and src[0]["rows_rank"] == 49 and src[0]["rows_uniform"] == 1
    assert src[0]["train_rank"] == "runs/tr/rank_c/bench" and src[-1] == {"floor": 0.05}
    # failure cases — none of them writes a file
    bad = []

    def expect(match, *extra, name="m"):
        sfx = f"_bad{len(bad)}"
        bad.append((name, sfx))
        with pytest.raises(SystemExit, match=match):
            run(*extra, "--suffix", sfx, name=name)

    keep2 = np.setdiff1d(np.arange(n_te), [7, 8])  # 96 %
    write_rank(tmp_path, "te", "r48", "test", keep2, lp[keep2], "test", meta_extra=test_meta)
    expect("at least 98", "--rank-only", "x=runs/te/r48/test")
    write_rank(tmp_path, "te", "low", "test", np.arange(n_te), lp + np.log(0.5), "test", meta_extra=test_meta)
    expect("candidate mass", "--rank-only", "x=runs/te/low/test")
    write_rank(tmp_path, "te", "none", "test", np.arange(n_te), rank_logp(n_te, rng, none_rows=30), "test", meta_extra=test_meta)
    expect("malfunction band", "--rank-only", "x=runs/te/none/test")
    done = np.ones(n_te, bool)
    done[3] = False
    write_rank(tmp_path, "te", "part", "test", np.arange(n_te), lp, "test", done=done, meta_extra=test_meta)
    expect("not a complete, exact run", "--rank-only", "x=runs/te/part/test")
    assert run("--rank-only", "x=runs/te/part/test", "--rank-partial", "--suffix", "_part") == 0  # deliberately: row 3 uniform
    Xp = by_hand_rank(lp)
    Xp[3] = U
    assert np.allclose(np.load(art / "test_m_part.npy"), mix_by_hand([np.column_stack([A, np.zeros(n_te)]), 0.95 * Xp + 0.05 * U], [0.6, 0.5]), atol=1e-12)
    write_rank(tmp_path, "te", "beams", "test", np.arange(n_te), lp, "test", meta_extra={**test_meta, "recipe": {"model": "m3p_full", "beams": 18}})
    expect("computed differently", "--rank-only", "x=runs/te/beams/test")
    write_rank(tmp_path, "te", "fin", "test", np.arange(n_te), lp, "test", meta_extra={**test_meta, "sft": {"sft_from": "g/gen", "finish_tokens": 64}})
    expect("computed differently", "--rank-only", "x=runs/te/fin/test")
    expect("missing", "--rank-only", "x=runs/te/missing/test")
    expect("--rank-only x=runs/… missing")  # the blend file requires the channel as a ranking
    expect("must not also be listed", "--rank-only", "x=runs/te/rank_job/test", "--test", "x=ra/job/test")
    good_meta = (art / "meta_x.json").read_text()
    for meta, match in (({"name": "x"}, "does not describe a ranking channel"),
                        ({"name": "x", "rank": {"source": "runs/tr/rank_c/bench"}, "rank_mean": {"floor": 0.05}}, "describes a mean channel"),
                        ({"name": "x", "rank": {"source": "runs/tr/rank_c/bench", "rows_named_only": 3}}, "mixed form"),
                        ({"name": "x", "rank": {"source": "runs/tr/rank_c/bench"}, "class_weights": [1, 1, 1, 1]}, "debiasing")):
        (art / "meta_x.json").write_text(json.dumps(meta))
        expect(match, "--rank-only", "x=runs/te/rank_job/test")
    (art / "meta_x.json").write_text(good_meta)
    (art / "stack_m0.json").write_text(json.dumps({**stack, "floor": {}}))
    expect("sets no floor", "--rank-only", "x=runs/te/rank_job/test", name="m0")
    (art / "stack_m1.json").write_text(json.dumps({**stack, "floor": {"x": 0.01}}))
    expect("the blend file records", "--rank-only", "x=runs/te/rank_job/test", name="m1")
    for name, sfx in bad:
        assert not (art / "submissions" / f"sub_{name}{sfx}.csv").exists() and not (art / f"test_{name}{sfx}.npy").exists()


# ------------------------------------------------------------------------------------------------ c99 -> make_submission

def test_c99_mixture_on_channel_rows_to_test_matrix(tmp_path, monkeypatch):
    """c99: channel rows from the non-uniform rows (against the meta file), weights ONLY from these rows (values of the other rows
    change nothing), blend file with fit_rows and inductive_check. make_submission loads it, applies the channel with
    --rank-only and the inductive alignment from the SAME rows — recomputed by hand; the checksum of the class weights matches, a
    corrupted one aborts."""
    c99 = load("c99_rt", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    rng = np.random.default_rng(11)
    counts = [24, 20, 16, 0]
    y = rng.permutation(np.repeat(np.arange(4), counts))
    n, K, n_te = len(y), len(CLASSES), 30
    folds = np.arange(n) % 5
    rows = np.flatnonzero(np.isin(folds, [1, 2]))  # channel only on two folds

    def probs(lab, sharp, bias):
        z = rng.normal(0, 1, (len(lab), K)) + sharp * np.eye(K)[lab] + np.log(bias)[None, :]
        z[:, ~SEEN] = NINF
        P = np.exp(z - z.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)

    mats = {"a": probs(y, 1.5, np.array([1, 2.5, 1, 1.0])), "b": probs(y, 1.0, np.array([0.6, 1, 1.6, 1.0]))}
    X = np.tile(U, (n, 1))
    lp_tr = rank_logp(len(rows), rng, none_rows=int((y[rows] == 0).sum()))
    X[rows] = probs(y[rows], 2.0, np.ones(K))
    mats["x"] = X
    meta_x = {"name": "x", "rows_neutral": int(n - len(rows)), "rank": {"source": "runs/tr/rank_job/bench", "rows": int(len(rows)), "rows_named_only": 0}}
    got_rows, info = c99.channel_rows(X, SEEN, meta_x, min_rows=10)
    assert np.array_equal(got_rows, rows) and info["rows_uniform"] == n - len(rows)
    with pytest.raises(SystemExit):
        c99.channel_rows(X, SEEN, {**meta_x, "rows_neutral": 3}, min_rows=10)
    with pytest.raises(SystemExit):
        c99.channel_rows(X, SEEN, meta_x, rows=np.array([0, 1]), min_rows=1)  # rows without the channel
    # weights only from the channel rows: values of the other rows change nothing
    names, floors = ["a", "b", "x"], [0.0, 0.02, 0.05]
    logs = c99.logs_of([mats[k] for k in names], floors, SEEN)
    th, _ = c99.fit_theta(logs, y, rows, None, 1.0)
    noisy = {**mats, "a": mats["a"].copy()}
    noisy["a"][~np.isin(np.arange(n), rows)] = rng.dirichlet(np.ones(K), n - len(rows))
    th2, _ = c99.fit_theta(c99.logs_of([noisy[k] for k in names], floors, SEEN), y, rows, None, 1.0)
    assert np.array_equal(th, th2)
    ref = c99.A2.fit_mix([c[rows] for c in logs], None, y[rows])
    assert np.array_equal(th, ref)
    # blend file as written by c99 --save (line with inductive alignment)
    ctx = {"y": y, "seen": SEEN, "n": n, "K": K, "folds": folds, "q_code": np.arange(n) % 7, "group": np.where(y == 0, "NONE", "seen")}
    spec = {"names": names, "names_base": ["a", "b"], "floors": {"b": 0.02, "x": 0.05}, "think": None, "factor": 1.0, "last": "ind",
            "theta_prod": np.array([0.7, 0.4]), "label": "test line", "last_label": "inductive"}
    ins = c99.insample(ctx, spec, mats, rows)
    assert np.allclose(ins["theta"]["with"]["theta"], th, atol=0, rtol=0)
    w = ins["weights"]["with"]
    ind_check = {"strength": 0.75, "min_rows": 3, "weights_sha256": c99.weights_sha16(w)}
    stack = c99.stack_record(spec, "x", ins, None, rows, folds, {"a": "0", "b": "0", "x": "0"}, meta_x, ind_check)
    assert stack["fit_rows"] == [int(i) for i in rows] and stack["rank_only"]["x"]["floor"] == 0.05 and stack["floor"] == {"b": 0.02, "x": 0.05}
    ms, art = setup_ms(tmp_path, monkeypatch, "ms_c99", y, n_te)
    (art / "stack_s.json").write_text(json.dumps(stack))
    for k in names:
        np.save(art / f"oof_{k}.npy", mats[k])
    (art / "meta_x.json").write_text(json.dumps(meta_x))
    write_rank(tmp_path, "tr", "rank_job", "bench", rows, lp_tr, "train")
    Ta = rng.dirichlet(np.ones(3), n_te)
    Tb = rng.dirichlet(np.ones(3), n_te)
    write_diag(art, "ra", np.column_stack([Ta, np.zeros(n_te)]))
    write_diag(art, "rb", np.column_stack([Tb, np.zeros(n_te)]))
    lp_te = rank_logp(n_te, rng, none_rows=10)
    write_rank(tmp_path, "te", "rank_job", "test", np.arange(n_te), lp_te, "test", meta_extra={"recipe": {"model": "full"}})

    def run(name, *extra):
        monkeypatch.setattr(sys, "argv", ["make_submission.py", "--name", name, "--test", "a=ra/job/test", "--test", "b=rb/job/test",
                                          "--rank-only", "x=runs/te/rank_job/test", "--align-inductive", "0.75", "--align-min", "3", *extra])
        return ms.main()

    assert run("s", "--suffix", "_ia") == 0
    info = json.loads((art / "submissions" / "sub_s_ia.json").read_text())["align_inductive"]
    assert info["weights_sha256"] == ind_check["weights_sha256"] and info["fit_rows"]["rows"] == len(rows)
    # by hand: class weights from the blend of the channel rows (mask, floor), target = frequencies of all training rows from 3
    fl = {"a": 0.0, "b": 0.02, "x": 0.05}

    def floored(P, eps, masked):
        Q = np.where(SEEN[None, :], P, 0.0) if masked else P
        return (1 - eps) * Q + eps * U[None, :] if eps else Q

    P_tr = mix_by_hand([floored(mats[k][rows], fl[k], True) for k in names], stack["theta"])
    cnt = np.bincount(y, minlength=K)
    w_hand = c99.C21.fit_weights(P_tr, np.where(cnt >= 3, cnt, 0) / n)
    assert np.allclose(w_hand, w, rtol=1e-9, atol=0)  # same weights as c99 (the checksum above checks bit-identity)
    Xt = by_hand_rank(lp_te)
    P_te = mix_by_hand([floored(np.column_stack([Ta, np.zeros(n_te)]), 0, False), floored(np.column_stack([Tb, np.zeros(n_te)]), 0.02, False),
                        floored(Xt, 0.05, False)], stack["theta"])
    exp = np.where(SEEN[None, :], P_te * w_hand[None, :] ** 0.75, 0.0)
    exp = exp / exp.sum(1, keepdims=True)
    assert np.allclose(np.load(art / "test_s_ia.npy"), exp, atol=1e-12, rtol=0)
    # the same weights from ALL rows (without fit_rows) would differ: the row selection has an effect
    st_all = {k: v for k, v in stack.items() if k not in ("fit_rows", "fit_rows_sha256", "inductive_check")}
    (art / "stack_s_all.json").write_text(json.dumps(st_all))
    assert run("s_all", "--suffix", "_ia") == 0
    assert json.loads((art / "submissions" / "sub_s_all_ia.json").read_text())["align_inductive"]["weights_sha256"] != ind_check["weights_sha256"]
    # corrupted checksum or row list: abort
    (art / "stack_s_bad.json").write_text(json.dumps({**stack, "inductive_check": {**ind_check, "weights_sha256": "0" * 16}}))
    with pytest.raises(SystemExit, match="inductive_check"):
        run("s_bad", "--suffix", "_ia")
    (art / "stack_s_bad2.json").write_text(json.dumps({**stack, "fit_rows": stack["fit_rows"][:-1]}))
    with pytest.raises(SystemExit, match="fit_rows"):
        run("s_bad2", "--suffix", "_ia")
    assert not (art / "submissions" / "sub_s_bad_ia.csv").exists() and not (art / "submissions" / "sub_s_bad2_ia.csv").exists()


def test_c99_cross_and_last_steps():
    """Cross-fit on the channel rows: the weights of the evaluated fold never come from its rows; the margin alignment brings the
    set (at full strength) exactly to the target; the inductive step only applies weights from other rows."""
    c99 = load("c99_cross", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    rng = np.random.default_rng(3)
    y = rng.permutation(np.repeat(np.arange(3), [30, 30, 30]))
    n, K = len(y), len(CLASSES)
    P = rng.dirichlet(np.ones(3), n)
    P = np.column_stack([P, np.zeros(n)])
    tgt = c99.target_of(y, np.arange(n), K)
    Q, w = c99.last_step("trans", P, tgt, SEEN)
    full = c99.apply_last(P, w, SEEN, strength=1.0)
    assert np.allclose(full.mean(0)[:3], tgt[:3], atol=1e-6) and np.all(Q[:, 3] == 0) and np.allclose(Q.sum(1), 1)
    Q2, w2 = c99.last_step("ind", P[:10], tgt, SEEN, P_fit=P[10:])
    assert np.array_equal(w2, c99.C21.fit_weights(P[10:], tgt))
    with pytest.raises(ValueError):
        c99.last_step("x", P, tgt, SEEN)


# ------------------------------------------------------------------------------------------------ training regime

def _variant_tree(base: Path, run: str, models: dict, jobs: list[dict]) -> None:
    import yaml

    (base / "configs" / "vlm").mkdir(parents=True, exist_ok=True)
    (base / "configs" / "vlm" / f"{run}.yaml").write_text(yaml.safe_dump({"run": run, "models": models, "workers": [{"gpu": 0, "jobs": jobs}]}))


def test_variant_check_gold_and_teacher(tmp_path, monkeypatch):
    """--variant: training targets behind the OOF channel (fold models) and the test side (final fit) via configuration, merge
    job and sft_recipe.json or the configuration of the training job; gold-only against teacher targets, the inductive line only
    gold-only, not determinable -> 6."""
    c99 = load("c99_var", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    monkeypatch.setattr(c99, "COMP", tmp_path)
    art = tmp_path / "artifacts"
    art.mkdir()
    # OOF side: two fold models, gold-only (sft_recipe.json without pool / teacher_derived, as for the pilot)
    _variant_tree(tmp_path, "pilot", {"m_f1": {"merged": "m_f1_merged"}, "m_f2": {"merged": "m_f2_merged"}},
                  [{"name": "train_f1", "kind": "train"}, {"name": "train_f2", "kind": "train"},
                   {"name": "merge_f1", "kind": "merge", "adapter_from": "train_f1", "out_model": "m_f1_merged"},
                   {"name": "merge_f2", "kind": "merge", "adapter_from": "train_f2", "out_model": "m_f2_merged"}])
    for k in (1, 2):
        d = art / "runs" / "pilot" / "pilot" / f"train_f{k}"
        d.mkdir(parents=True)
        (d / "sft_recipe.json").write_text(json.dumps({"task": "proto_sft", "fold": k, "system_sha256": "s1"}))
        write_rank(tmp_path, "pilot", f"rank_f{k}", "bench", np.arange(5), rank_logp(5, np.random.default_rng(k), 2), "train",
                   meta_extra={"recipe": {"model": f"m_f{k}"}})
    write_combined(tmp_path, "pilot", "rank", "bench", ["rank_f1", "rank_f2"])
    (art / "meta_x.json").write_text(json.dumps({"rank": {"source": "runs/pilot/rank/bench"}}))
    # test sides: final fit gold-only (recipe fetched), with teacher targets (configuration only), unknown
    _variant_tree(tmp_path, "full_gold", {}, [{"name": "train_full", "kind": "train", "fold": None}])
    d = art / "runs" / "full_gold" / "full_gold" / "train_full"
    d.mkdir(parents=True)
    (d / "sft_recipe.json").write_text(json.dumps({"task": "proto_sft", "fold": None, "system_sha256": "s1"}))
    _variant_tree(tmp_path, "full_pool", {}, [{"name": "train_full", "kind": "train", "fold": None, "sft_pool": {"targets": "p.jsonl"}}])
    for run, src in (("test_gold", "../full_gold/train_full"), ("test_pool", "../full_pool/train_full"), ("test_unk", "../nowhere/train_full")):
        _variant_tree(tmp_path, run, {"full": {"merged": "full_merged"}},
                      [{"name": "merge", "kind": "merge", "adapter_from": src, "out_model": "full_merged"}])
        write_rank(tmp_path, run, "rank", "test", np.arange(5), rank_logp(5, np.random.default_rng(9), 2), "test",
                   meta_extra={"recipe": {"model": "full"}})
    code, lines, info = c99.variant_check("runs/test_gold/rank/test", "x", art=art)
    assert code == 0 and info["test_side"]["variant"] == "gold-only" and {s["variant"] for s in info["oof_side"]} == {"gold-only"}
    assert c99.variant_check("runs/test_gold/rank/test", "x", require_gold_only=True, art=art)[0] == 0
    code, _, info = c99.variant_check("runs/test_pool/rank/test", "x", art=art)
    assert code == 5 and info["test_side"]["variant"] == "teacher" and info["test_side"]["evidence"].startswith("configs/vlm/full_pool.yaml")
    assert c99.variant_check("runs/test_pool/rank/test", "x", allow_mismatch=True, art=art)[0] == 0
    assert c99.variant_check("runs/test_pool/rank/test", "x", require_gold_only=True, allow_mismatch=True, art=art)[0] != 0  # no exception
    assert c99.variant_check("runs/test_unk/rank/test", "x", art=art)[0] == 6
    assert c99.variant_check("runs/test_unk/rank/test", "x", allow_unknown=True, art=art)[0] == 0
    assert c99.variant_check("runs/test_unk/rank/test", "x", require_gold_only=True, allow_unknown=True, art=art)[0] == 6
    # both sides with teacher targets: same training regime, still not admissible for the inductive line
    for k in (1, 2):
        (art / "runs" / "pilot" / "pilot" / f"train_f{k}" / "sft_recipe.json").write_text(
            json.dumps({"task": "proto_sft", "fold": k, "system_sha256": "s1", "teacher_derived": True, "pool": {"n": 3}}))
    assert c99.variant_check("runs/test_pool/rank/test", "x", art=art)[0] == 0
    assert c99.variant_check("runs/test_pool/rank/test", "x", require_gold_only=True, art=art)[0] == 6


# ------------------------------------------------------------------------------------------------ real artifacts

def _real_inputs_missing() -> str | None:
    art = COMP_ROOT / "artifacts"
    need = [art / "stack_v1r_zs27q_think18nb_pr27xm_w14.json", art / "stack_v1r_zs27t_think18sb_pr27xm.json", COMP_ROOT / "data" / "train.csv"]
    need += [art / f"oof_{k}.npy" for k in ("v1r", "zs27q", "zs27t", "think18nb", "think18sb", "pr27xm")]
    miss = [str(p.relative_to(COMP_ROOT)) for p in need if not p.exists()]
    return ", ".join(miss) if miss else None


@pytest.mark.skipif(_real_inputs_missing() is not None, reason="inputs missing locally")
def test_c99_fit_reproduces_submitted_weights():
    """c99 fits with the same function as the build path: without the channel on all 1,500 rows it reproduces the weights of the
    deployed base blend files of both lines bit-identically (primary line with factor 1.4)."""
    c99 = load("c99_real", COMP_ROOT / "eda" / "c99_m3_deploy.py")
    ctx = c99.context()
    art = COMP_ROOT / "artifacts"
    mats = {k: np.load(art / f"oof_{k}.npy") for k in ("v1r", "zs27q", "zs27t", "think18nb", "think18sb", "pr27xm")}
    for key in ("best", "safe"):
        spec = c99.line_setup(key, "channel", 0.05, art)
        _, logs, _, ti = c99.variant_logs(spec, mats, ctx["seen"], False)
        th, th_base = c99.fit_theta(logs, ctx["y"], np.arange(ctx["n"]), ti, spec["factor"])
        assert np.array_equal(th, spec["theta_prod"]) and np.array_equal(th_base, spec["theta_prod_base"])
