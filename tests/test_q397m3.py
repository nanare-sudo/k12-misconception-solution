"""Tests for "the 397B finds, M3 names": converter (tools/make_q397m3_notes.py), configurations (configs/vlm/q397m3_oof_rp.yaml,
q397m3te_rp.yaml) and evaluation (eda/c115_q397m3_eval.py).

Converter and test side only with SYNTHETIC data in tmp_path (no test row, no real test note). With real training artifacts
(skipped if they are missing): upload folder and note file, check run of the evaluation (--dry)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

COMP = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MQ = _load("make_q397m3_notes_test", "tools/make_q397m3_notes.py")

from vlm import proto as PR  # noqa: E402
from vlm import sft as S  # noqa: E402

EXPECT_ADAPTERS = {1: "../m3_pilot_rp/m3_train_f1", 2: "../m3_pilot_rp/m3_train_f2", 0: "../m3_f034_rp/m3_train_f0",
                   3: "../m3_f034_rp/m3_train_f3", 4: "../m3_f034_rp/m3_train_f4"}
TRAIN_NOTES = "derived/sft/q397m3_train_notes.jsonl"
TEST_NOTES = "derived/sft/q397m3_test_notes.jsonl"
ANSWER = '{"root_code": "FRA-01", "top3": ["FRA-01", "CAL-01", "NONE"], "probabilities": [0.6, 0.2, 0.1]}'


# ------------------------------------------------------------------------------------------------ synthetic sources

def _tinker_records():
    """Four forms of first passes as in data/derived/open_teacher/qwen35-397b/raw.jsonl (plus one finish)."""
    return [
        {"row": 0, "phase": "first", "thinking": "Read the work. FRA-01 fits.", "text": ANSWER, "stop": "stop", "in_tokens": 9, "out_tokens": 9},
        {"row": 1, "phase": "first", "thinking": "", "text": "<think>\nLong reasoning about CAL-01 and ANS-01", "stop": "length"},
        {"row": 2, "phase": "first", "thinking": "", "text": "Reasoning done, FRA-01.\n</think>\n\n" + ANSWER[:30], "stop": "length"},
        {"row": 3, "phase": "first", "thinking": "", "text": "Draft JSON: root_code FRA-01 ...", "stop": "length"},
        {"row": 1, "phase": "finish", "thinking": "", "text": ANSWER, "stop": "stop"},
    ]


def _data_root(tmp_path: Path, n_train: int = 4, n_test: int = 3, records=None, labels_bytes: bytes = b"x\n") -> Path:
    root = tmp_path / "data"
    (root / "derived" / "open_teacher" / "qwen35-397b").mkdir(parents=True)
    pd.DataFrame({"trace_id": [f"t{i}" for i in range(n_train)]}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"trace_id": [f"s{i}" for i in range(n_test)]}).to_csv(root / "test.csv", index=False)
    d = root / "derived" / "open_teacher" / "qwen35-397b"
    d.joinpath("raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (records or _tinker_records())))
    d.joinpath("labels.jsonl").write_bytes(labels_bytes)
    return root


def _run_train(tmp_path, monkeypatch, root=None, extra=()):
    root = root or _data_root(tmp_path)
    monkeypatch.setattr(MQ, "TINKER_LABELS_SHA256", hashlib.sha256((root / MQ.TINKER_DIR / "labels.jsonl").read_bytes()).hexdigest())
    rc = MQ.main(["--data-root", str(root), "--artifacts", str(tmp_path / "art"), "--tokenizer", "", "--parity", "", *extra])
    return rc, root


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


# ------------------------------------------------------------------------------------------------ converter: training

def test_train_notes_format_and_cleaning(tmp_path, monkeypatch):
    rc, root = _run_train(tmp_path, monkeypatch)
    assert rc == 0
    out = root / TRAIN_NOTES
    table, notes = S.load_notes(out)  # this is how the runner reads sft_from
    assert table == "train" and sorted(notes) == [0, 1, 2, 3]
    assert all(f is None for _, f in notes.values())  # M3 writes the finding itself
    for text, f in notes.values():
        assert S.rank_prefix(text, f).endswith(S.HEAD)  # the runner generates the finding before scoring (needs_finding)
        assert "root_code\": " not in text and "</think>" not in text and "<think>" not in text  # no answer text, no marks
    assert notes[0][0] == "Read the work. FRA-01 fits."  # finished regularly: only the thinking
    assert notes[1][0] == "Long reasoning about CAL-01 and ANS-01"  # cut off without the end of the thinking block: everything is thinking
    assert notes[2][0] == "Reasoning done, FRA-01."  # a started answer behind </think> is dropped
    recs = _rows(out)
    st = MQ.RECORDED_STATE  # recorded labels of the note states
    assert [r["q397_state"] for r in recs] == [st["complete"], st["open"], st["open"], st["draft"]]
    assert [r["state"] for r in recs] == ["open"] * 4
    assert list(recs[0]) == ["table", "row", "state", "text", "finding", "q397_state"]
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert meta["answer_text_dropped"] == 2 and meta["tinker_closed_in_text"] == 1 and meta["rows"] == 4
    up = root / "derived" / "sft_q397m3"
    assert sorted(p.name for p in up.iterdir()) == ["q397m3_train_notes.jsonl", "q397m3_train_notes.meta.json"]
    assert (up / "q397m3_train_notes.jsonl").read_bytes() == out.read_bytes()
    body = out.read_bytes()
    assert _run_train(tmp_path, monkeypatch, root)[0] == 0 and out.read_bytes() == body  # reproducible byte for byte


def test_tinker_split_and_rank_notes_are_reused():
    """The same split as open_teacher.tinker_split and the same note as vlm/proto.rank_notes (training targets of pr27)."""
    for r in _tinker_records()[:4]:
        assert MQ.thinking_of(r, "tinker") == MQ.OT.tinker_split(r)[0]
        note, cut = MQ.note_of(MQ.thinking_of(r, "tinker"))
        assert note == PR.rank_notes({"phase": "first", "thinking": MQ.OT.tinker_split(r)[0]}) and not cut
    own = {"row": 0, "phase": "first", "thinking": "<think>\n abc \n", "text": ANSWER, "end": "stop"}
    assert MQ.note_of(MQ.thinking_of(own, "own"))[0] == PR.rank_notes(own) == "abc"
    assert MQ.note_of("x" * 20, note_chars=8) == ("x" * 8, True)


def test_train_refuses_marks_duplicates_missing_and_foreign_labels(tmp_path, monkeypatch):
    bad = _tinker_records()
    bad[0] = dict(bad[0], thinking="text with <|im_end|> mark")
    rc, root = _run_train(tmp_path / "a", monkeypatch, _data_root(tmp_path / "a", records=bad))
    assert rc == 4 and not (root / TRAIN_NOTES).exists()  # nothing written
    dup = _tinker_records() + [dict(_tinker_records()[3])]
    assert _run_train(tmp_path / "b", monkeypatch, _data_root(tmp_path / "b", records=dup))[0] == 4
    rc, root = _run_train(tmp_path / "c", monkeypatch, _data_root(tmp_path / "c", n_train=5))
    assert rc == 4 and not (root / TRAIN_NOTES).exists()  # row 4 without a first pass
    rc, root = _run_train(tmp_path / "d", monkeypatch, _data_root(tmp_path / "d", n_train=5), extra=("--allow-missing",))
    assert rc == 0
    last = _rows(root / TRAIN_NOTES)[4]
    assert last["state"] == "empty" and last["text"] == "" and last["q397_state"] == MQ.RECORDED_STATE["missing"]
    out_of_table = _tinker_records() + [{"row": 7, "phase": "first", "thinking": "x", "text": "", "stop": "stop"}]
    assert _run_train(tmp_path / "e", monkeypatch, _data_root(tmp_path / "e", records=out_of_table))[0] == 4
    root = _data_root(tmp_path / "f", labels_bytes=b"other measurement\n")  # labels.jsonl is not that of the recorded measurement
    assert MQ.main(["--data-root", str(root), "--artifacts", str(tmp_path / "f" / "art"), "--tokenizer", "", "--parity", ""]) == 4
    assert not (root / TRAIN_NOTES).exists()


def test_upload_folder_refuses_foreign_files(tmp_path, monkeypatch):
    root = _data_root(tmp_path)
    (root / "derived" / "sft_q397m3").mkdir(parents=True)
    (root / "derived" / "sft_q397m3" / "m3_d1_test_frozen_notes.jsonl").write_text("{}\n")
    rc, _ = _run_train(tmp_path, monkeypatch, root)
    assert rc == 4 and not (root / TRAIN_NOTES).exists()


# ------------------------------------------------------------------------------------------------ converter: test side (synthetic)

def _own_test_run(art: Path, n: int, table: str = "test", model: str = MQ.TEST_MODEL, rows=None, meta_rows=None):
    d = art / "runs" / "q397te_rp" / "q397te_rp" / "proto_te"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in (range(n) if rows is None else rows):
        recs.append({"row": i, "phase": "first", "thinking": f"synthetic note {i} CAL-01", "text": "" if i % 2 else ANSWER,
                     "end": "length" if i % 2 else "stop", "in_tokens": 5, "out_tokens": 5})
    d.joinpath("proto_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    d.joinpath("proto_meta.json").write_text(json.dumps({"table": table, "rows": n if meta_rows is None else meta_rows, "answered": len(recs), "failed": 0}))
    d.joinpath("summary.json").write_text(json.dumps({"model": {"repo": model}}))
    return "runs/q397te_rp/proto_te/proto"


def test_test_side_conversion_synthetic(tmp_path):
    root, art = _data_root(tmp_path, n_test=3), tmp_path / "art"
    spec = _own_test_run(art, 3)
    rc = MQ.main(["--data-root", str(root), "--artifacts", str(art), "--tokenizer", "", "--test-from", spec])
    assert rc == 0
    out = root / TEST_NOTES
    table, notes = S.load_notes(out)
    assert table == "test" and sorted(notes) == [0, 1, 2] and all(f is None for _, f in notes.values())
    assert all(ANSWER not in t for t, _ in notes.values())
    assert sorted(p.name for p in (root / "derived" / "sft_q397m3te").iterdir()) == ["q397m3_test_notes.jsonl", "q397m3_test_notes.meta.json"]
    assert not (root / "derived" / "sft_q397m3").exists()  # the training upload stays without a test file
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["table"] == "test" and meta["source_kind"] == "own" and meta["model"] == MQ.TEST_MODEL
    assert "compare_m3" not in meta and "parity_own_vs_tinker" not in meta


@pytest.mark.parametrize("kw,extra", [({"table": "train"}, ()), ({"model": "Qwen/Qwen3.8-27B"}, ()), ({"meta_rows": 2}, ()),
                                      ({"rows": [0, 1]}, ())])
def test_test_side_refusals_synthetic(tmp_path, kw, extra):
    root, art = _data_root(tmp_path, n_test=3), tmp_path / "art"
    spec = _own_test_run(art, 3, **kw)
    assert MQ.main(["--data-root", str(root), "--artifacts", str(art), "--tokenizer", "", "--test-from", spec, *extra]) == 4
    assert not (root / TEST_NOTES).exists()


# ------------------------------------------------------------------------------------------------ configurations

def _jobs(name: str):
    cfg = yaml.safe_load((COMP / "configs" / "vlm" / f"{name}.yaml").read_text())
    jobs = [j for w in cfg["workers"] for j in w["jobs"]]
    return cfg, {j["name"]: j for j in jobs}, [j["name"] for j in jobs]


RANK_FREE = {"name", "model", "sft_from", "rows"}


def test_oof_config_resolves_like_the_runner_and_is_fold_safe():
    cfg, jobs, order = _jobs("q397m3_oof_rp")
    ref_cfg, ref, _ = _jobs("m3d3_oof_rp")
    assert cfg["run"] == "q397m3_oof_rp" and len(cfg["workers"]) == 1 and cfg["workers"][0]["gpu"] == 0
    assert cfg["models"]["q38_27b"] == ref_cfg["models"]["q38_27b"]
    merged_fold, seen_folds = {}, []
    for n in order:
        j = jobs[n]
        assert j["kind"] in {"merge", "bench"}, n  # no generation: the notes come frozen from the file
        spec = cfg["models"][j["model"]]
        if j["kind"] == "merge":
            assert j["model"] == "q38_27b" and "merged" not in spec
            k = int(n.rsplit("_f", 1)[1])
            assert j["adapter_from"] == EXPECT_ADAPTERS[k]
            assert j["adapter_from"] == ref[f"m3d3_merge_f{k}"]["adapter_from"]  # the same fold adapters as d1–d4
            out = [m for m, s in cfg["models"].items() if s.get("merged") == j["out_model"]]
            assert len(out) == 1
            merged_fold[out[0]] = k
            continue
        assert spec["merged"] and j["model"] in merged_fold, (n, "model used before it was merged")
        k = merged_fold[j["model"]]
        assert j["rows"]["folds"] == [k], n  # fold rule: a row of fold k only with the adapter that did not see k
        assert j["sft_from"] == TRAIN_NOTES and j["probes"] == ["proto_rank"] and j.get("split", "train") == "train"
        assert not any(j.get(x) for x in S.NO_TASK_KEYS) and "sft_examples" not in j
        if n == "q397m3_smoke":
            assert j["must_pass"] is True and j["rows"]["limit"] == 6 and j["drop_model"] is False and order.index(n) == 1
            continue
        seen_folds.append(k)
        rk = ref[f"m3d3_rank_f{k}"]
        assert {x: v for x, v in j.items() if x not in RANK_FREE} == {x: v for x, v in rk.items() if x not in RANK_FREE}, n
        assert j["rows"] == rk["rows"] and j["drop_model"] is True
    assert sorted(seen_folds) == [0, 1, 2, 3, 4]
    smoke = jobs["q397m3_smoke"]
    main = jobs["q397m3_rank_f1"]
    assert {x: v for x, v in smoke.items() if x not in ("name", "must_pass", "rows", "save_every", "drop_model")} == \
           {x: v for x, v in main.items() if x not in ("name", "must_pass", "rows", "save_every", "drop_model")}


def test_oof_adapters_hold_out_their_fold():
    """The adapters available locally (source of the uploads) hold out exactly the fold that their ranking measures."""
    found = 0
    for k, src in EXPECT_ADAPTERS.items():
        run, job = src[3:].split("/")
        f = COMP / "artifacts" / "runs" / run / run / job / "sft_recipe.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        assert rec["fold"] == k and not rec.get("teacher_derived") and not rec.get("pool")
        found += 1
    if not found:
        pytest.skip("adapters not available locally")


def test_test_config_resolves_like_the_runner():
    cfg, jobs, order = _jobs("q397m3te_rp")
    ref_cfg, ref, _ = _jobs("m3d3te_rp")
    assert cfg["run"] == "q397m3te_rp" and cfg["models"]["q38_27b"] == ref_cfg["models"]["q38_27b"]
    assert order == ["q397m3_merge_full", "q397m3_test_smoke", "q397m3_test_rank"]
    merge = jobs["q397m3_merge_full"]
    assert merge["kind"] == "merge" and merge["adapter_from"] == ref["m3d3_merge"]["adapter_from"] == "../m3_fullte_rp/m3_train_full"
    assert cfg["models"]["q397m3_full"]["merged"] == merge["out_model"]
    for n in ("q397m3_test_smoke", "q397m3_test_rank"):
        j = jobs[n]
        assert j["kind"] == "predict" and j["fold"] is None and j["model"] == "q397m3_full" and j["sft_from"] == TEST_NOTES
        assert [p["split"] for p in j["predict"]] == ["test"] and "rows" not in j
    rk, rref = jobs["q397m3_test_rank"], ref["m3d3_test_rank"]
    assert {x: v for x, v in rk.items() if x not in RANK_FREE} == {x: v for x, v in rref.items() if x not in RANK_FREE}
    smoke = jobs["q397m3_test_smoke"]
    assert smoke["must_pass"] is True and smoke["predict"][0]["rows"] == {"limit": 6} and smoke["predict"][0]["tag"] == "smoke"


def test_real_upload_folder_holds_only_training_notes():
    up = COMP / "data" / "derived" / "sft_q397m3"
    if not up.is_dir():
        pytest.skip("upload folder not available locally")
    names = sorted(p.name for p in up.iterdir())
    assert names == ["q397m3_train_notes.jsonl", "q397m3_train_notes.meta.json"]
    f = up / "q397m3_train_notes.jsonl"
    meta = json.loads(f.with_suffix(".meta.json").read_text())
    assert meta["table"] == "train" and meta["sha256"] == S.sha256_file(f) == S.sha256_file(COMP / "data" / TRAIN_NOTES)
    table, notes = S.load_notes(f)
    assert table == "train" and sorted(notes) == list(range(1500)) and all(fd is None for _, fd in notes.values())
    assert not any(MQ.MARKS.search(t) for t, _ in notes.values())


# ------------------------------------------------------------------------------------------------ evaluation c115

E = _load("c115_q397m3_eval_test", "eda/c115_q397m3_eval.py")


def _c(du: int, z: float, ll_new: float = 1.0, ll_base: float = 1.0) -> dict:
    return {"delta_exact": Fraction(du, 9000), "z_q": z, "ll_new": ll_new, "ll_base": ll_base}


def test_decision_rule_boundaries():
    d = E.decide
    assert d(_c(18, 1.0), _c(0, 0.0), 1350, 1500)["verdict"] == "CONTINUE"  # Δ exactly +0.002, z exactly 1.0, same top exactly 0.90
    assert d(_c(17, 3.0), _c(0, 0.0), 1000, 1500)["verdict"] == "STOP"  # Δ just below +0.002
    assert d(_c(18, 0.999), _c(0, 0.0), 1000, 1500)["verdict"] == "STOP"
    assert d(_c(30, 2.0, ll_new=1.0001), _c(0, 0.0), 1000, 1500)["verdict"] == "STOP"  # log loss worse
    assert d(_c(30, 2.0), _c(0, 0.0), 1351, 1500)["verdict"] == "STOP"  # takeover
    r = d(_c(0, 0.0), _c(25, 1.2, ll_new=0.9), 1200, 1500)
    assert r["verdict"] == "CONTINUE" and r["passed_forms"] == ["replacement"]  # the replacement form alone suffices
    r = d(_c(30, 2.0, ll_new=1.1), _c(25, 0.5), 1200, 1500)  # Δ of one form, z of the other: no form passes completely
    assert r["verdict"] == "STOP" and r["passed_forms"] == []


def test_early_stop_rule():
    assert E.decide_early(290, 299)["verdict"] == "TOO EARLY"
    assert E.decide_early(285, 300)["verdict"].startswith("STOP")  # exactly 0.95
    assert E.decide_early(284, 300)["verdict"] == "KEEP COMPUTING"
    assert E.decide_early(285, 299, 300)["verdict"].startswith("STOP")  # one row without a valid q397 answer in the shard
    assert E.decide_early(200, 210, 299)["verdict"] == "TOO EARLY"


def test_same_top_ties_and_masks():
    seen = np.array([True, True, True, False])
    A = np.array([[0.4, 0.4, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.1, 0.1, 0.7]])
    B = np.array([[0.5, 0.3, 0.2, 0.0], [0.7, 0.2, 0.1, 0.0], [0.6, 0.2, 0.2, 0.0]])
    assert E.top1(A, seen).tolist() == [0, 2, 0]  # tie -> smaller index; a class without a training example never at the top
    assert E.same_top(A, B, seen) == (2, 3)
    assert E.same_top(A, B, seen, np.array([False, True, True])) == (1, 2)
    assert E.same_top(A, B, seen, np.array([1])) == (0, 1)


def test_mention_pattern_whole_words_longest_first():
    pat = E.mention_pattern(["NONE", "FRA-01", "FRA-011", "CAL-01", "Dec-01"])
    texts = {0: "FRA-011 then FRA-01, not XCAL-01 or CAL-01x; NONE; Dec-01.", 1: ""}
    assert E.mentioned(texts, pat, 3) == [{"FRA-011", "FRA-01", "Dec-01"}, set(), set()]


def _shard(art: Path, k: int, rows: np.ndarray, K: int, sft_from: str = E.NOTES_REL, beams: int = 74, done=True):
    d = art / "artifacts" / "runs" / E.RUN / E.RUN / f"q397m3_rank_f{k}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(k)
    lp = np.log(rng.dirichlet(np.ones(K), size=len(rows)))
    np.save(d / "bench_idx.npy", np.asarray(rows, dtype=np.int64))
    np.save(d / "bench_proto_rank.npy", lp)
    if done:
        np.save(d / "bench_proto_rank.done.npy", np.ones(len(rows), bool))
    meta = {"table": "train", "n": int(len(rows)), "probes": ["proto_rank"], "job": f"q397m3_rank_f{k}",
            "recipe": {"beams": beams, "image": {"max_side": 1024, "question_max_side": 448}, "model": f"q397m3_f{k}", "fold": None},
            "dtype": "bf16", "quant": None, "adapter": False, "sft": {"sft_from": sft_from, "finish_tokens": 320}}
    (d / "bench_meta.json").write_text(json.dumps(meta))
    return lp


def test_assemble_shards_fold_exact(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(E.A8, "COMP", tmp_path)
    n, K = 10, 5
    folds = np.array([0, 1, 2, 3, 4] * 2)
    seen = np.ones(K, bool)
    for k in range(5):
        _shard(tmp_path, k, np.flatnonzero(folds == k), K)
    P, have, info = E.assemble(E.SHARDS, n, seen, folds)
    assert have.all() and np.allclose(P.sum(1), 1.0) and [x["fold"] for x in info] == [0, 1, 2, 3, 4]
    st = E.check_shard_settings(E.SHARDS, E.REF_SHARDS)  # without local d3 metas in the tmp folder: only equality of the shards
    assert st["sft_from"] == E.NOTES_REL and st["same_as_d3"] is False
    _shard(tmp_path, 4, np.array([4]), K)  # shard does not cover all rows of its fold
    with pytest.raises(E.Technik, match="fold"):
        E.assemble(E.SHARDS, n, seen, folds)
    _shard(tmp_path, 4, np.array([3, 4]), K)  # wrong fold (row 3 belongs to fold 3)
    with pytest.raises(E.Technik):
        E.assemble(E.SHARDS, n, seen, folds)
    _shard(tmp_path, 4, np.flatnonzero(folds == 4), K, beams=18)
    E.assemble(E.SHARDS, n, seen, folds)
    with pytest.raises(E.Technik, match="computed differently"):
        E.check_shard_settings(E.SHARDS, E.REF_SHARDS)
    _shard(tmp_path, 4, np.flatnonzero(folds == 4), K, sft_from="m3_gen_f4/gen")
    with pytest.raises(E.Technik, match="sft_from"):
        E.check_shard_settings(E.SHARDS, E.REF_SHARDS)
    import shutil

    shutil.rmtree(tmp_path / "artifacts" / "runs" / E.RUN / E.RUN / "q397m3_rank_f4")
    with pytest.raises(E.Missing):
        E.assemble(E.SHARDS, n, seen, folds)
    P2, have2, info2 = E.assemble(E.SHARDS, n, seen, folds, require_all=False)  # early stop: only complete shards
    assert len(info2) == 4 and not have2[folds == 4].any() and have2[folds != 4].all()
    _shard(tmp_path, 4, np.flatnonzero(folds == 4), K, done=False)  # partial state without a completeness mask does not count
    assert len(E.assemble(E.SHARDS, n, seen, folds, require_all=False)[2]) == 4


def test_dry_probe_on_real_artifacts():
    """With real training artifacts: q397m3 := copy of q397 -> anchor exact, replacement bit-identical to the six-channel recipe
    (Δ exactly 0), addition = null point, same top 1.0 -> the rule would give STOP."""
    need = [COMP / "artifacts" / f"oof_{x}.npy" for x in ("q397", "m3x3_full", "v1r", "zs27q", "think18nb", "pr27xm")] + [COMP / "data" / TRAIN_NOTES]
    if not all(p.is_file() for p in need):
        pytest.skip("training artifacts not available locally")
    ctx, mats, _ = E.context()
    nt = E.check_notes(ctx["root"], ctx["n"])
    rep, rows = E.evaluate(ctx, mats, mats["q397"].copy(), nt["states_by_row"], nt["texts"], dry=True)
    assert rep["anchor"]["units"] == E.ANCHOR_UNITS
    assert rep["recipes"]["replacement"]["delta_exact"] == 0 and np.array_equal(rows["replacement"], rows["base"])
    assert rep["recipes"]["addition"]["delta_exact"] == rep["recipes"]["null_addition_copy_q397"]["delta_exact"]
    assert rep["same_top"]["q397m3/q397"]["share"] == 1.0
    assert rep["decision"]["verdict"].startswith("CHECK") and "STOP" in rep["decision"]["verdict"] and rep["decision"]["consequence"] is None
