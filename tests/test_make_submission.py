"""tools/make_submission.py and eda/a8_proto_oof.py: inductive alignment, blend with arithmetic share, protocol channels."""
import json

import pytest


def test_make_submission_inductive_align_end_to_end(tmp_path, monkeypatch):
    """tools/make_submission.main with --align-inductive on synthetic data: the class weights depend ONLY on the OOF artifacts of
    the training rows (different test predictions -> same checksum of the weights); the file is the blend of the test rows times
    weight^strength (mask, normalisation); classes below the limit keep weight 1; --align is mutually exclusive."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    comp_root = Path(__file__).resolve().parents[1]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ms = load("make_submission", comp_root / "tools" / "make_submission.py")
    c21 = load("c21", comp_root / "eda" / "c21_margin_align.py")
    classes = ["NONE"] + [f"AAA-0{i}" for i in range(1, 7)]
    counts = np.array([40, 28, 19, 8, 3, 2, 0])  # AAA-04: exactly at the limit 3, AAA-05: below, AAA-06: never
    K, n_tr, n_te = len(classes), int(counts.sum()), 30
    seen = counts > 0
    rng = np.random.default_rng(0)
    y = rng.permutation(np.repeat(np.arange(K), counts))
    bias = {"a": np.array([1, 3, 1, 1, 1, 1, 1.0]), "b": np.array([0.5, 1, 1, 2, 1, 1, 1.0])}

    def probs(r, lab, b):
        z = r.normal(0, 1, (len(lab), K)) + 2.0 * np.eye(K)[lab] + np.log(b)[None, :]
        z[:, ~seen] = -np.inf
        P = np.exp(z - z.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)

    art, data = tmp_path / "artifacts", tmp_path / "data"
    (art / "submissions").mkdir(parents=True)
    (data / "competition_data").mkdir(parents=True)
    (tmp_path / "eda").mkdir()
    (tmp_path / "eda" / "c21_margin_align.py").write_text((comp_root / "eda" / "c21_margin_align.py").read_text())
    stack = {"channels": ["a", "b"], "none": [], "probes": {}, "theta": [0.6, 0.5], "floor": {"b": 0.02}}
    (art / "stack_m.json").write_text(json.dumps(stack))
    for k in "ab":
        np.save(art / f"oof_{k}.npy", probs(rng, y, bias[k]))
    ids = [f"{i:012x}" for i in range(n_te)]
    pd.DataFrame({"trace_id": ids, "root_code": "NONE"}).to_csv(data / "competition_data" / "sample_submission.csv", index=False)
    tables = {"train": pd.DataFrame({"trace_id": [f"t{i}" for i in range(n_tr)], "root_code": [classes[c] for c in y]}), "test": pd.DataFrame({"trace_id": ids})}
    monkeypatch.setattr(ms, "COMP", tmp_path)
    monkeypatch.setattr(ms.C, "find_data_root", lambda hint=None: data)
    monkeypatch.setattr(ms.C, "load_table", lambda root, name: tables[name].copy())
    monkeypatch.setattr(ms.labels, "load_classes", lambda p: list(classes))

    def run(seed, *extra):
        r = np.random.default_rng(seed)
        lab = r.integers(0, 4, n_te)
        for k in "ab":
            d = art / "runs" / f"r{k}" / f"r{k}" / "job"
            d.mkdir(parents=True, exist_ok=True)
            (d / "test_meta.json").write_text(json.dumps({"table": "test"}))
            np.save(d / "test_idx.npy", np.arange(n_te))
            with np.errstate(divide="ignore"):
                np.save(d / "test_diag.npy", np.log(probs(r, lab, bias[k])))
        monkeypatch.setattr(sys, "argv", ["make_submission.py", "--name", "m", "--test", "a=ra/job/test", "--test", "b=rb/job/test", *extra])
        assert ms.main() == 0

    run(1, "--suffix", "_raw")
    run(1, "--align-inductive", "0.75", "--align-min", "3", "--suffix", "_ia")
    P_raw, P_ia = np.load(art / "test_m_raw.npy"), np.load(art / "test_m_ia.npy")
    info = json.loads((art / "submissions" / "sub_m_ia.json").read_text())["align_inductive"]
    target = np.where(counts >= 3, counts, 0) / n_tr
    w = c21.fit_weights(ms.oof_mixture(stack, seen, art), target)  # only OOF artifacts of the training rows
    exp = np.where(seen[None, :], P_raw * w[None, :] ** 0.75, 0.0)
    exp = exp / exp.sum(1, keepdims=True)
    assert info["classes"] == 5 and w[5] == 1.0 and w[6] == 1.0 and w[4] != 1.0  # limit: from three rows
    assert np.allclose(P_ia, exp, atol=1e-12) and not np.allclose(P_ia, P_raw, atol=1e-3)
    assert np.all(P_ia[:, 6] == 0.0) and np.allclose(P_ia.sum(1), 1.0)
    run(2, "--align-inductive", "0.75", "--align-min", "3", "--suffix", "_ia2")  # different test predictions: the same weights
    assert json.loads((art / "submissions" / "sub_m_ia2.json").read_text())["align_inductive"]["weights_sha256"] == info["weights_sha256"]
    with pytest.raises(SystemExit):
        run(1, "--align-inductive", "0.75", "--align", "0.75", "--suffix", "_x")
    assert not (art / "submissions" / "sub_m_x.csv").exists()


def test_oof_mixture_for_inductive_align(tmp_path):
    """tools/make_submission.oof_mixture (inductive alignment): predictions of the saved blend for the training rows from the
    OOF artifacts — floors and weights of the blend file, never-seen classes stay zero."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    comp_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_submission", comp_root / "tools" / "make_submission.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    seen = np.array([True, True, True, False])
    a = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.1, 0.8, 0.0]])
    b = np.array([[0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    np.save(tmp_path / "oof_a.npy", a)
    np.save(tmp_path / "oof_b.npy", b)
    stack = {"channels": ["a", "b"], "theta": [1.0, 0.5], "floor": {"b": 0.3}, "none": [], "probes": {}}
    P = ms.oof_mixture(stack, seen, tmp_path)
    bf = 0.7 * b + 0.3 * seen / 3  # floor: mixture with the uniform distribution over the admissible classes
    z = np.log(np.clip(a, 1e-9, 1)) + 0.5 * np.log(np.clip(bf, 1e-9, 1))
    ref = np.where(seen[None, :], np.exp(z), 0.0)
    assert np.allclose(P, ref / ref.sum(1, keepdims=True), atol=1e-6) and np.allclose(P.sum(1), 1.0) and np.all(P[:, 3] == 0.0)
    with pytest.raises(SystemExit):
        ms.oof_mixture({**stack, "none": ["x"]}, seen, tmp_path)


def test_mix_channels_arith_beta(tmp_path):
    """tools/make_submission.mix_channels: β = 0 is the purely log-linear blend, β = 1 the weighted arithmetic mean of the
    channels with floor (w = θ/Σθ), in between the convex combination; oof_mixture takes β from the blend file; never-seen classes
    stay zero; β outside [0, 1] and matching signals next to β are rejected."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    comp_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_submission", comp_root / "tools" / "make_submission.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    seen = np.array([True, True, True, False])
    a = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.1, 0.8, 0.0]])
    b = np.array([[0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    np.save(tmp_path / "oof_a.npy", a)
    np.save(tmp_path / "oof_b.npy", b)
    base = {"channels": ["a", "b"], "theta": [1.0, 0.5], "floor": {"b": 0.3}, "none": [], "probes": {}}
    P0 = ms.oof_mixture(base, seen, tmp_path)
    assert np.allclose(P0, ms.oof_mixture({**base, "arith_beta": 0.0}, seen, tmp_path))
    bf = 0.7 * b + 0.3 * seen / 3
    A = (1.0 * a + 0.5 * bf) / 1.5  # weighted arithmetic mean, w = θ/Σθ
    P1 = ms.oof_mixture({**base, "arith_beta": 1.0}, seen, tmp_path)
    assert np.allclose(P1, A, atol=1e-6)
    P4 = ms.oof_mixture({**base, "arith_beta": 0.4}, seen, tmp_path)
    assert np.allclose(P4, 0.6 * P0 + 0.4 * A, atol=1e-6) and np.allclose(P4.sum(1), 1.0) and np.all(P4[:, 3] == 0.0)
    assert not np.allclose(P4, P0)  # positive control: β has an effect
    with pytest.raises(SystemExit):
        ms.oof_mixture({**base, "arith_beta": 1.5}, seen, tmp_path)
    chans = [np.log(np.clip(a, 1e-9, 1)), np.log(np.clip(bf, 1e-9, 1))]
    with pytest.raises(SystemExit):
        ms.mix_channels({**base, "arith_beta": 0.4}, chans, np.zeros((2, 1)), seen)


def test_proto_debias():
    """Inductive debiasing of the protocol channel (eda/a8_proto_oof.py): the weights bring the mean mass to the class frequency
    of the rows with a valid answer, rows without an answer stay unchanged, classes below the limit keep weight 1, and unusable
    weights are rejected."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    comp_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("a8", comp_root / "eda" / "a8_proto_oof.py")
    a8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a8)
    rng = np.random.default_rng(0)
    n, K = 400, 5
    y = rng.permutation(np.repeat(np.arange(K), [160, 120, 80, 38, 2]))  # class 4: two rows, below the limit
    P = rng.dirichlet(np.ones(K), n) * np.array([1.0, 3.0, 1.0, 0.5, 1.0])  # the channel names class 1 too often, class 3 too rarely
    P = P / P.sum(1, keepdims=True)
    valid = np.ones(n, bool)
    valid[:10] = False
    P[:10] = 1.0 / K
    w = a8.debias_weights(P, valid, y, np.arange(n), K, 3)
    c = np.bincount(y[valid], minlength=K)
    target = np.where(c >= 3, c, 0) / valid.sum()
    Q = a8.apply_debias(P, valid, w, 1.0)
    assert np.allclose(Q[:10], 1.0 / K) and np.allclose(Q.sum(1), 1.0)  # rows without an answer: unchanged
    assert np.allclose(Q[valid].mean(0)[:4], target[:4], atol=2e-3)  # mean mass = frequency (fixed 200 steps: accurate to 1e-3 here, to 2e-4 on the real rows)
    assert w[4] == 1.0 and w[1] < w[2] < w[0]  # below the limit: weight 1; named too often -> smaller than those named too rarely
    assert np.allclose(a8.apply_debias(P, valid, w, 0.0), P)  # strength 0: nothing happens
    half = a8.debias_weights(P, valid, y, np.arange(n // 2), K, 3)  # only the named rows enter
    c_half = np.bincount(y[: n // 2][valid[: n // 2]], minlength=K)
    assert np.allclose(a8.apply_debias(P, valid, half, 1.0)[: n // 2][valid[: n // 2]].mean(0)[:4], (np.where(c_half >= 3, c_half, 0) / valid[: n // 2].sum())[:4], atol=2e-3)
    for bad in (w[:-1], np.where(np.arange(K) == 0, np.nan, w), np.where(np.arange(K) == 0, 0.0, w)):
        with pytest.raises(ValueError):
            a8.apply_debias(P, valid, bad, 1.0)


def test_proto_channel_for_submission(tmp_path, monkeypatch):
    """a8.mean_of_draws (deployment path of the protocol channel): last valid record, table check, duplicate rows, missing rows
    uniform, cross-check against the statistics of the run."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    comp_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("a8", comp_root / "eda" / "a8_proto_oof.py")
    a8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a8)
    monkeypatch.setattr(a8, "COMP", tmp_path)
    classes, seen = ["NONE", "A", "B", "C"], np.array([True, True, True, False])

    def run(job, recs, table="test", **meta):
        d = tmp_path / "artifacts" / "runs" / "r" / "r" / job
        d.mkdir(parents=True)
        (d / "proto_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (d / "proto_meta.json").write_text(json.dumps({"table": table, **meta}))

    run("a", [{"row": 0, "ok": False}, {"row": 0, "ok": True, "root_code": "A", "top3": ["A", "B", "NONE"], "probabilities": [0.5, 0.3, 0.1]}], valid=1)
    run("b", [{"row": 0, "ok": True, "root_code": "B", "top3": ["B"], "probabilities": [1.0]}])
    run("t", [{"row": 1, "ok": True, "root_code": "A", "top3": ["A"], "probabilities": [1.0]}], table="train")
    run("u", [{"row": 2, "ok": True, "root_code": "A", "top3": ["A"], "probabilities": [1.0]}], valid=5)  # file fetched incompletely
    P, cnt = a8.mean_of_draws(["runs/r/a/proto"], 3, classes, seen, table="test")
    assert cnt.tolist() == [1, 0, 0] and np.allclose(P[1], [1 / 3, 1 / 3, 1 / 3, 0]) and np.allclose(P.sum(1), 1)
    assert np.allclose(P[0], np.array([0.1, 0.5, 0.3, 0]) + 0.1 * seen / 3)
    for bad in (["runs/r/a/proto+runs/r/b/proto"], ["runs/r/t/proto"], ["runs/r/u/proto"], ["runs/r/missing/proto"], ["r/a/proto"]):
        with pytest.raises(SystemExit):  # row covered twice · wrong table · fewer valid rows than reported · no meta file · source without runs/
            a8.mean_of_draws(bad, 3, classes, seen, table="test")
    P2, cnt2 = a8.mean_of_draws(["runs/r/a/proto", "runs/r/b/proto"], 3, classes, seen, table="test")
    assert cnt2[0] == 2 and np.isclose(P2[0, 2], (0.3 + 0.1 / 3 + 1.0) / 2)  # mean of two draws
    P3, cnt3 = a8.mean_of_draws(["runs/r/a/proto"], 3, classes, seen, table="test", neutral=np.array([0]))
    assert cnt3.tolist() == [-1, 0, 0] and np.allclose(P3[0], [1 / 3, 1 / 3, 1 / 3, 0])  # gate: the row counts as not computed


def test_proto_rank_channel_from_probe_output(tmp_path, monkeypatch):
    """a8.load_rank: ranking of the probe proto_rank as a channel — only computed rows with a finite candidate, codes outside the
    search share the remaining mass, classes that are never a root stay zero; a wrong table and a mismatching row list abort."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    comp_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("a8", comp_root / "eda" / "a8_proto_oof.py")
    a8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a8)
    monkeypatch.setattr(a8, "COMP", tmp_path)
    seen = np.array([True, True, True, False])

    def run(job, idx, lp, done=None, table="train"):
        d = tmp_path / "artifacts" / "runs" / "r" / "r" / job
        d.mkdir(parents=True)
        (d / "bench_meta.json").write_text(json.dumps({"table": table}))
        np.save(d / "bench_idx.npy", np.array(idx))
        np.save(d / "bench_proto_rank.npy", np.array(lp, dtype=float))
        if done is not None:
            np.save(d / "bench_proto_rank.done.npy", np.array(done))

    ninf = -np.inf
    run("a", [4, 1, 2], [[np.log(0.7), np.log(0.2), ninf, ninf], [np.log(0.1), np.log(0.6), np.log(0.3), ninf], [ninf] * 4], done=[True, True, True])
    with pytest.raises(SystemExit, match="not a complete, exact run"):  # default: a partial state or truncated rows abort
        a8.load_rank("runs/r/a/bench", 6, seen)
    P, h, st = a8.load_rank("runs/r/a/bench", 6, seen, allow_partial=True)
    assert h.tolist() == [False, True, False, False, True, False]  # row 2 has no finite candidate: not computed
    assert np.allclose(P[1], [0.1, 0.6, 0.3, 0.0]) and np.allclose(P[4], [0.7, 0.2, 0.1, 0.0]) and np.allclose(P[[1, 4]].sum(1), 1)
    assert (st["rows_probe"], st["rows_done"], st["rows_exact"]) == (3, 2, 1) and np.isclose(st["mass_median"], 0.95)
    run("b", [0, 1], [[0.0, ninf, ninf, ninf], [0.0, ninf, ninf, ninf]], done=[True, False])
    assert a8.load_rank("runs/r/b/bench", 6, seen, allow_partial=True)[1].tolist() == [True, False, False, False, False, False]  # interrupted run
    run("v", [3, 0], [[np.log(0.5), np.log(0.25), np.log(0.125), ninf], [np.log(0.2), np.log(0.2), np.log(0.4), ninf]], done=[True, True])
    Pv, hv, sv = a8.load_rank("runs/r/v/bench", 6, seen)  # complete and exact: every class that can be a root is finite
    assert hv.tolist() == [True, False, False, True, False, False] and sv["rows_exact"] == 2 and np.isclose(sv["mass_median"], 0.8375)
    assert np.allclose(Pv[3], np.array([0.5, 0.25, 0.125, 0.0]) / 0.875) and np.allclose(Pv[0], [0.25, 0.25, 0.5, 0.0])
    with pytest.raises(SystemExit, match="exactly all 6 training rows"):
        a8.load_rank("runs/r/v/bench", 6, seen, require_full_table=True)  # an exact subset is not a full measurement
    full_lp = [[np.log(0.5), np.log(0.3), np.log(0.2), ninf]] * 6
    run("full", list(range(6)), full_lp, done=[True] * 6)
    Pf, hf, sf = a8.load_rank("runs/r/full/bench", 6, seen, require_full_table=True)
    assert hf.all() and sf["rows_expected"] == sf["rows_probe"] == sf["rows_done"] == sf["rows_exact"] == 6
    assert sf["completion_manifest"] and len(sf["idx_sha256"]) == 64
    run("no_manifest", list(range(6)), full_lp)
    with pytest.raises(SystemExit, match="completeness mask missing"):
        a8.load_rank("runs/r/no_manifest/bench", 6, seen, require_full_table=True)
    run("t", [0], [[0.0, ninf, ninf, ninf]], table="test")
    run("d", [0, 0], [[0.0, ninf, ninf, ninf]] * 2)
    run("w", [0], [[0.0, ninf, ninf]])
    run("o", [9], [[0.0, ninf, ninf, ninf]])
    for bad in ("runs/r/t/bench", "runs/r/d/bench", "runs/r/w/bench", "runs/r/o/bench", "r/a/bench"):
        with pytest.raises(SystemExit):  # wrong table · duplicate row · wrong width · position outside · source without runs/
            a8.load_rank(bad, 6, seen, allow_partial=True)


