"""Tests for the channel m3px27: converter (tools/make_m3px27_notes.py), configurations (configs/vlm/m3px27_oof_rp.yaml,
m3px27te_rp.yaml) and evaluation (eda/c132_m3px27_eval.py).

Converter only with SYNTHETIC data in tmp_path (no test row, no real test note). With real training artifacts (skipped if they are
missing): upload folder and note files (checksums, counts and equality only — no content is printed), preflight of the
configurations, copy null point and positive control on the original split."""
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


MP = _load("make_m3px27_notes_test", "tools/make_m3px27_notes.py")
E = _load("c132_m3px27_eval_test", "eda/c132_m3px27_eval.py")

from vlm import proto as PR  # noqa: E402
from vlm import sft as S  # noqa: E402

ANSWER = '{"root_code": "FRA-01", "top3": ["FRA-01", "CAL-01", "NONE"], "probabilities": [0.6, 0.2, 0.1]}'
EXPECT_ADAPTERS = E.EXPECT_ADAPTERS


# ------------------------------------------------------------------------------------------------ synthetic sources

def _first(i: int, kind: int) -> dict:
    """Four forms of first passes of the own compute path (vlm/proto.generate): complete (with thinking markers), draft (names
    root_code), open, open with a special character (U+2028)."""
    if kind == 0:
        return {"row": i, "phase": "first", "thinking": f"<think>\nRead the work {i}. FRA-01 fits.\n", "text": ANSWER, "end": "stop"}
    if kind == 1:
        return {"row": i, "phase": "first", "thinking": f"Draft {i}: root_code FRA-01 ...", "text": "", "end": "length"}
    if kind == 2:
        return {"row": i, "phase": "first", "thinking": f"  Long reasoning {i} about CAL-01 and ANS-01  ", "text": "", "end": "length"}
    return {"row": i, "phase": "first", "thinking": f"Note {i} with a special character", "text": "", "end": "length"}


def _write_run(art: Path, spec: str, table: str, recs: list[dict], model: str = MP.MODEL, revision: str = MP.REVISION, meta_rows=None):
    d, tag = MP.run_dir(spec, art)
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs + [{"row": recs[0]["row"], "phase": "finish", "text": ANSWER}])
    d.joinpath(f"{tag}_raw.jsonl").write_text(body, encoding="utf-8")
    n_first = sum(r.get("phase") == "first" for r in recs)
    d.joinpath(f"{tag}_meta.json").write_text(json.dumps({"table": table, "rows": n_first if meta_rows is None else meta_rows, "failed": 0}))
    d.joinpath("summary.json").write_text(json.dumps({"model": {"repo": model, "revision": revision}}))


def _export(root: Path, table: str, recs: list[dict], mutate=None) -> str:
    """Export file as written by tools/export_proto_notes.py (the input of px27) -> SHA-256."""
    rel, _ = MP.PX27_NOTES[table]
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sorted(recs, key=lambda x: x["row"]):
        text = PR.rank_notes(r, 16000)
        rows.append({"table": table, "row": r["row"], "state": MP.note_state(r, text), "text": text})
    if mutate:
        mutate(rows)
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return hashlib.sha256(f.read_bytes()).hexdigest()


def _setup(tmp_path: Path, n_train: int = 6, n_test: int = 4):
    root, art = tmp_path / "data", tmp_path / "art"
    root.mkdir(parents=True)
    pd.DataFrame({"trace_id": [f"t{i}" for i in range(n_train)]}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"trace_id": [f"s{i}" for i in range(n_test)]}).to_csv(root / "test.csv", index=False)
    tr = [_first(i, i % 4) for i in range(n_train)]
    te = [_first(i, (i + 1) % 4) for i in range(n_test)]
    a, b = MP.SOURCES["train"]
    _write_run(art, a, "train", tr[: n_train // 2])
    _write_run(art, b, "train", tr[n_train // 2:])
    _write_run(art, MP.SOURCES["test"][0], "test", te)
    return root, art, tr, te


def _run(root, art, table, sha, extra=()):
    return MP.main(["--data-root", str(root), "--artifacts", str(art), "--tokenizer", "", "--table", table, "--px27-sha", sha, *extra])


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


# ------------------------------------------------------------------------------------------------ converter

def test_train_notes_format_equal_to_px27_and_reproducible(tmp_path):
    root, art, tr, _ = _setup(tmp_path)
    sha = _export(root, "train", tr)
    assert _run(root, art, "train", sha) == 0
    out = root / MP.OUT["train"]
    table, notes = S.load_notes(out)  # this is how the runner reads sft_from
    assert table == "train" and sorted(notes) == list(range(6)) and all(f is None for _, f in notes.values())
    for i, (text, f) in notes.items():
        assert text == PR.rank_notes(tr[i], 16000)  # the same note as for px27 (vlm/proto.rank_notes)
        assert S.rank_prefix(text, f).endswith(S.HEAD)  # the runner generates the finding before scoring (needs_finding)
        assert "<think>" not in text and "</think>" not in text and "root_code\": " not in text  # no answer text, no markers
    recs = _rows(out)
    assert list(recs[0]) == ["table", "row", "state", "text", "finding", "pr27_state"]
    st = MP.RECORDED_STATE  # recorded labels of the note states
    assert [r["pr27_state"] for r in recs] == [st["complete"], st["draft"], st["open"], st["open"], st["complete"], st["draft"]]
    assert all(r["state"] == "open" for r in recs)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest() and meta["rows"] == 6 and meta["missing"] == 0
    assert meta["px27_notes"]["identical_notes"] == 6 and meta["px27_notes"]["sha256"] == sha
    assert [s["source"] for s in meta["sources"]] == list(MP.SOURCES["train"]) and [s["rows"] for s in meta["sources"]] == [3, 3]
    assert sorted(p.name for p in out.parent.iterdir()) == ["m3px27_train_notes.jsonl", "m3px27_train_notes.meta.json"]
    assert not (root / MP.OUT["test"]).parent.exists()  # the test upload stays untouched
    body = out.read_bytes()
    assert _run(root, art, "train", sha) == 0 and out.read_bytes() == body  # reproducible byte for byte
    assert " " in out.read_text(encoding="utf-8")  # special characters stay raw (split at \n only)


def test_test_side_fields_for_the_notes_guard(tmp_path):
    root, art, _, te = _setup(tmp_path)
    sha = _export(root, "test", te)
    assert _run(root, art, "test", sha) == 0
    out = root / MP.OUT["test"]
    table, notes = S.load_notes(out)
    assert table == "test" and sorted(notes) == [0, 1, 2, 3] and all(f is None for _, f in notes.values())
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    d, tag = MP.run_dir(MP.SOURCES["test"][0], art)
    assert meta["source"] == MP.SOURCES["test"][0] and meta["model"] == MP.MODEL and meta["with_record"] == 4
    assert meta["states"] == {"open": 4}  # note guard of make_submission: every test row with a note
    assert meta["source_sha256"] == hashlib.sha256((d / f"{tag}_raw.jsonl").read_bytes()).hexdigest()
    assert meta["meta_sha256"] == hashlib.sha256((d / f"{tag}_meta.json").read_bytes()).hexdigest()
    assert "m3_targets" not in meta
    assert not (root / MP.OUT["train"]).parent.exists()  # the training upload carries no test file


def _mut_text(rows):
    rows[2]["text"] = rows[2]["text"] + " "


def _mut_state(rows):
    rows[0]["state"] = MP.RECORDED_STATE["open"]


def _mut_drop(rows):
    rows.pop()


@pytest.mark.parametrize("mutate", [_mut_text, _mut_state, _mut_drop])
def test_refuses_notes_that_differ_from_px27(tmp_path, mutate):
    root, art, tr, _ = _setup(tmp_path)
    sha = _export(root, "train", tr, mutate)
    assert _run(root, art, "train", sha) == 4
    assert not (root / MP.OUT["train"]).exists()  # nothing written


def test_refuses_foreign_export_checksum(tmp_path):
    root, art, tr, _ = _setup(tmp_path)
    _export(root, "train", tr)
    assert _run(root, art, "train", "0" * 64) == 4
    assert MP.main(["--data-root", str(root), "--artifacts", str(art), "--tokenizer", "", "--table", "train"]) == 4  # pinned checksum
    assert not (root / MP.OUT["train"]).exists()


def test_refuses_marks_missing_duplicates_table_model_and_meta(tmp_path):
    def case(name, fn):
        root, art, tr, _ = _setup(tmp_path / name)
        fn(root, art, tr)
        sha = _export(root, "train", [r for r in tr if r.get("phase") == "first"]) if (root / "train.csv").is_file() else "0" * 64
        assert _run(root, art, "train", sha) == 4, name
        assert not (root / MP.OUT["train"]).exists(), name

    def marks(root, art, tr):
        tr[1]["thinking"] = "note with <|im_end|> marker"
        _write_run(art, MP.SOURCES["train"][0], "train", tr[:3])

    def missing(root, art, tr):
        _write_run(art, MP.SOURCES["train"][1], "train", tr[3:5])

    def duplicate(root, art, tr):
        _write_run(art, MP.SOURCES["train"][1], "train", tr[2:])

    def table(root, art, tr):
        _write_run(art, MP.SOURCES["train"][1], "test", tr[3:])

    def model(root, art, tr):
        _write_run(art, MP.SOURCES["train"][1], "train", tr[3:], model="Qwen/Qwen3.5-397B-A17B-FP8")

    def revision(root, art, tr):
        _write_run(art, MP.SOURCES["train"][1], "train", tr[3:], revision="0" * 40)

    def meta_rows(root, art, tr):
        _write_run(art, MP.SOURCES["train"][1], "train", tr[3:], meta_rows=640)

    for nm, fn in (("marks", marks), ("missing", missing), ("duplicate", duplicate), ("table", table), ("model", model),
                   ("revision", revision), ("meta_rows", meta_rows)):
        case(nm, fn)


def test_upload_folder_refuses_foreign_files(tmp_path):
    root, art, tr, _ = _setup(tmp_path)
    sha = _export(root, "train", tr)
    up = (root / MP.OUT["train"]).parent
    up.mkdir(parents=True)
    (up / "m3px27_test_notes.jsonl").write_text("{}\n")
    assert _run(root, art, "train", sha) == 4 and not (root / MP.OUT["train"]).exists()


# ------------------------------------------------------------------------------------------------ real note files (counts only)

def _real(rel: str) -> Path:
    return COMP / "data" / rel


@pytest.mark.parametrize("table,n", [("train", 1500), ("test", 3498)])
def test_real_notes_files_equal_px27_inputs(table, n):
    f = _real(str(MP.OUT[table]))
    if not f.is_file():
        pytest.skip("note file not available locally")
    assert sorted(p.name for p in f.parent.iterdir()) == [f.name, f.with_suffix(".meta.json").name]
    meta = json.loads(f.with_suffix(".meta.json").read_text())
    assert meta["sha256"] == S.sha256_file(f) and meta["table"] == table and meta["rows"] == n and meta["states"] == {"open": n}
    tab, notes = S.load_notes(f)
    assert tab == table and len(notes) == n and sorted(notes) == list(range(n))
    assert sum(fd is not None for _, fd in notes.values()) == 0  # only numbers in failure messages, never note texts
    assert sum(bool(MP.MARKS.search(t)) for t, _ in notes.values()) == 0
    rel, sha = MP.PX27_NOTES[table]
    px = _real(rel)
    if px.is_file():  # character-identical with the input of px27 (comparison only, nothing printed)
        assert S.sha256_file(px) == sha
        exp = {int(r["row"]): str(r["text"]) for r in _rows(px)}
        assert len(exp) == n and sum(exp.get(i) != notes[i][0] for i in range(n)) == 0
    if table == "train":
        assert all(v.get("identical_notes") == v.get("rows") for v in meta["m3_targets"].values() if not v.get("missing"))


# ------------------------------------------------------------------------------------------------ configurations

def _jobs(name: str):
    cfg = yaml.safe_load((COMP / "configs" / "vlm" / f"{name}.yaml").read_text())
    jobs = [j for w in cfg["workers"] for j in w["jobs"]]
    return cfg, {j["name"]: j for j in jobs}, [j["name"] for j in jobs]


def test_oof_config_is_q397m3_with_other_notes():
    """The same jobs as q397m3_oof_rp (fold adapters, pre-run, settings as d3), only another note file and other names."""
    cfg, jobs, order = _jobs("m3px27_oof_rp")
    qcfg, qjobs, qorder = _jobs("q397m3_oof_rp")
    assert cfg["run"] == "m3px27_oof_rp" and [n.replace("m3px27", "q397m3") for n in order] == qorder
    assert cfg["models"]["q38_27b"] == qcfg["models"]["q38_27b"]
    for n in order:
        j, q = jobs[n], qjobs[n.replace("m3px27", "q397m3")]
        strip = lambda d: {k: (v.replace("q397m3", "m3px27") if isinstance(v, str) else v) for k, v in d.items() if k != "sft_from"}  # noqa: E731
        assert strip(j) == strip(q), n
        if j["kind"] == "bench":
            assert j["sft_from"] == E.NOTES_REL and q["sft_from"] == "derived/sft/q397m3_train_notes.jsonl"
    cc = E.config_check()
    assert cc["problems"] == [] and sorted(cc["merged_fold"].values()) == [0, 1, 2, 3, 4]
    assert cc["adapters"] == {f"m3px27_f{k}": v for k, v in EXPECT_ADAPTERS.items()}


def test_oof_config_fold_rule_like_the_runner():
    from vlm import runner as R

    cfg, jobs, order = _jobs("m3px27_oof_rp")
    folds = np.array([k % 5 for k in range(1500)])
    merged = {}
    for n in order:
        j = jobs[n]
        if j["kind"] == "merge":
            merged[[m for m, s in cfg["models"].items() if s.get("merged") == j["out_model"]][0]] = int(n.rsplit("_f", 1)[1])
            continue
        k = merged[j["model"]]
        idx = R.select_rows(j.get("rows", {}), 1500, folds)
        assert len(idx) and np.all(folds[idx] == k), n
        assert len(idx) == (6 if n == E.SMOKE else 300)


def test_test_config_is_q397m3te_with_other_notes():
    cfg, jobs, order = _jobs("m3px27te_rp")
    qcfg, qjobs, qorder = _jobs("q397m3te_rp")
    assert cfg["run"] == "m3px27te_rp" and [n.replace("m3px27", "q397m3") for n in order] == qorder
    for n in order:
        j, q = jobs[n], qjobs[n.replace("m3px27", "q397m3")]
        strip = lambda d: {k: (v.replace("q397m3", "m3px27") if isinstance(v, str) else v) for k, v in d.items() if k != "sft_from"}  # noqa: E731
        assert strip(j) == strip(q), n
        if j["kind"] == "predict":
            assert j["sft_from"] == E.TEST_NOTES_REL and j["fold"] is None and [p["split"] for p in j["predict"]] == ["test"]
    assert jobs["m3px27_merge_full"]["adapter_from"] == E.FULL_ADAPTER


def test_oof_adapters_hold_out_their_fold():
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


# ------------------------------------------------------------------------------------------------ m3px27 rule (pure)

def _f(u: int) -> Fraction:
    return Fraction(u, 18000)


ZERO_CTRL = {k: _f(0) for k in E.CONTROLS}


def _splits(vals) -> list:
    return [_f(v) for v in vals]


def test_decide_rule_boundaries():
    neg = {"copy_px27": _f(0), "half_px27_d4": _f(-17), "d4": _f(-30), "copy_m3x3_full": _f(0)}  # controls as pre-registered
    ctrl_s = {k: _splits([-5] * 20) for k in E.CONTROLS}
    ctrl_s["copy_px27"] = _splits([0] * 20)
    ok = E.decide(_f(1), neg, _splits([1] * 20), ctrl_s)
    assert ok["verdict"] == "DEPLOY" and all(ok["checks"].values())
    assert E.decide(_f(0), neg, _splits([1] * 20), ctrl_s)["verdict"] == "NO DEPLOYMENT"  # original exactly 0: not "above"
    pos = dict(neg, d4=_f(2))  # a control above 0: then it is the bar
    assert E.decide(_f(2), pos, _splits([1] * 20), ctrl_s)["verdict"] == "NO DEPLOYMENT"
    assert E.decide(_f(3), pos, _splits([1] * 20), ctrl_s)["verdict"] == "DEPLOY"
    zero_mean = _splits([1] * 10 + [-1] * 10)  # mean exactly 0
    assert E.decide(_f(5), neg, zero_mean, ctrl_s)["verdict"] == "NO DEPLOYMENT"
    hi = {k: _splits([4] * 20) for k in E.CONTROLS}  # control mean +4 > mean of the candidate +3
    r = E.decide(_f(5), neg, _splits([3] * 20), hi)
    assert r["verdict"] == "NO DEPLOYMENT" and r["checks"]["splits_mean_above_0"] and not r["checks"]["splits_mean_above_control_mean"]
    mixed = dict(ctrl_s, d4=_splits([8] * 20))  # control mean (0 − 5 + 8 − 5)/4 = −0.5; largest control mean +8
    r = E.decide(_f(5), neg, _splits([3] * 20), mixed)
    assert r["verdict"] == "DEPLOY" and r["splits"]["strict_would_pass"] is False  # reading (2): mean of the controls
    with pytest.raises(ValueError):
        E.decide(_f(1), neg, _splits([1] * 19), ctrl_s)  # only with exactly 20 splits
    with pytest.raises(ValueError):
        E.decide(_f(1), {k: v for k, v in neg.items() if k != "d4"}, _splits([1] * 20), ctrl_s)
    assert E.half_of(7, -24) == Fraction(-17, 18000)  # ½(px27 + d4) on the original split: −0.000944 (pre-registered −0.0009)


def test_early_stop_rule():
    assert E.decide_early(290, 299)["verdict"] == "TOO EARLY"
    assert E.decide_early(285, 300)["verdict"].startswith("STOP")  # exactly 0.95
    assert E.decide_early(284, 300)["verdict"] == "KEEP COMPUTING"
    assert E.decide_early(570, 600)["verdict"].startswith("STOP")
    assert E.decide_early(200, 210, 299)["verdict"] == "TOO EARLY"


def test_none_band():
    assert E.none_band_ok(450, 1500) and E.none_band_ok(570, 1500)  # 0.30 and 0.38 included
    assert not E.none_band_ok(449, 1500) and not E.none_band_ok(571, 1500) and not E.none_band_ok(0, 0)


def test_log_check_attributes_lines_to_jobs():
    # log lines of vlm/runner.py; the recorded worker logs use an earlier German wording of both lines, which is accepted as well
    good = "\n".join(f"12:00 === job {j} (bench) ===\n12:00 sft_from: notes for 1500 rows of table train, of which without a finding 1500 (…)"
                     for j in E.RANK_JOBS)
    legacy = "\n".join(f"12:00 === Auftrag {j} (bench) ===\n12:00 sft_from: Ausgaben zu 1500 Zeilen der Tabelle train, davon ohne Befund 1500 (…)"
                       for j in E.RANK_JOBS)
    merges = "\n".join(f"11:00 === job m3px27_merge_f{k} (merge) ===" for k in range(5))
    assert E.log_check(merges + "\n" + good)["problems"] == []
    assert E.log_check(legacy)["problems"] == [] and E.log_check(legacy)["per_job"] == E.log_check(good)["per_job"]
    bad = good.replace("of which without a finding 1500", "of which without a finding 1499", 1)
    assert len(E.log_check(bad)["problems"]) == 1
    missing = "\n".join(x for x in good.split("\n") if "m3px27_rank_f4" not in x)
    probs = E.log_check(missing)["problems"]
    assert any("m3px27_rank_f3" in p for p in probs)  # the line of f4 now belongs to the job f3 (twice), f4 is missing
    assert any("m3px27_rank_f4" in p for p in probs)


def test_same_top_units_and_hash():
    seen = np.array([True, True, True, False])
    A = np.array([[0.4, 0.4, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.1, 0.1, 0.7]])
    B = np.array([[0.5, 0.3, 0.2, 0.0], [0.7, 0.2, 0.1, 0.0], [0.6, 0.2, 0.2, 0.0]])
    assert E.top1(A, seen).tolist() == [0, 2, 0]  # tie -> smaller index; a class without a training example is never on top
    assert E.same_top(A, B, seen) == (2, 3) and E.same_top(A, B, seen, np.array([False, True, True])) == (1, 2)
    assert E.units_of(np.array([1 / 6, -1 / 3, 0.5, -1.0])) == 1 - 2 + 3 - 6
    with pytest.raises(E.TechnicalError):
        E.units_of(np.array([0.1]))
    r = {f: {"forced": np.zeros(3), "top": np.zeros((3, 5), np.int64), "cand": np.zeros((3, 5), np.int64)} for f in E.FORMS}
    h = E.rows_hash(r)
    r2 = {f: {k: v.copy() for k, v in d.items()} for f, d in r.items()}
    r2["set"]["top"][1, 0] = 2
    assert E.rows_hash(r2) != h and E.rows_hash(r) == h


def _shard(art: Path, k: int, rows: np.ndarray, K: int, sft_from: str = E.NOTES_REL, beams: int = 74, done=True):
    d = art / "artifacts" / "runs" / E.RUN / E.RUN / f"m3px27_rank_f{k}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(k)
    np.save(d / "bench_idx.npy", np.asarray(rows, dtype=np.int64))
    np.save(d / "bench_proto_rank.npy", np.log(rng.dirichlet(np.ones(K), size=len(rows))))
    if done:
        np.save(d / "bench_proto_rank.done.npy", np.ones(len(rows), bool))
    meta = {"table": "train", "n": int(len(rows)), "probes": ["proto_rank"], "job": f"m3px27_rank_f{k}",
            "recipe": {"beams": beams, "image": {"max_side": 1024, "question_max_side": 448}, "model": f"m3px27_f{k}", "fold": None},
            "dtype": "bf16", "quant": None, "adapter": False, "sft": {"sft_from": sft_from, "finish_tokens": 320}}
    (d / "bench_meta.json").write_text(json.dumps(meta))


def test_assemble_shards_fold_exact(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(E.c115().A8, "COMP", tmp_path)
    n, K = 10, 5
    folds = np.array([0, 1, 2, 3, 4] * 2)
    seen = np.ones(K, bool)
    for k in range(5):
        _shard(tmp_path, k, np.flatnonzero(folds == k), K)
    P, have, info = E.assemble(n, seen, folds)
    assert have.all() and np.allclose(P.sum(1), 1.0) and [x["fold"] for x in info] == [0, 1, 2, 3, 4]
    _shard(tmp_path, 4, np.array([3, 4]), K)  # row 3 belongs to fold 3
    with pytest.raises(E.TechnicalError):
        E.assemble(n, seen, folds)
    import shutil

    shutil.rmtree(tmp_path / "artifacts" / "runs" / E.RUN / E.RUN / "m3px27_rank_f4")
    with pytest.raises(E.Missing):
        E.assemble(n, seen, folds)
    _, have2, info2 = E.assemble(n, seen, folds, require_all=False)  # early stop: only complete shards
    assert len(info2) == 4 and not have2[folds == 4].any()
    _shard(tmp_path, 4, np.flatnonzero(folds == 4), K, done=False)  # without the completeness mask the shard does not count
    assert len(E.assemble(n, seen, folds, require_all=False)[2]) == 4


# ------------------------------------------------------------------------------------------------ real artifacts (CPU, about 1 min)

def _have_real() -> bool:
    need = [COMP / "artifacts" / f"oof_{x}.npy" for x in ("pr27", "pr27x", "pr27xm", "m3x3_full", "q397m3", "v1r", "zs27q", "think18nb")]
    need += [COMP / "data" / E.NOTES_REL, E.MM_ROWS, COMP / "artifacts" / "stack_s032_stage2.json"]
    return all(p.is_file() for p in need)


@pytest.mark.skipif(not _have_real(), reason="training artifacts not available locally")
def test_real_copy_null_point_and_positive_control():
    """Copy null point (m3px27 := px27 ⇒ final primary form bit-identical) and positive control (the four controls hit their
    pre-registered values) on the original split — the same computation as the evaluation (controls from the cache where valid)."""
    ctx = E.load_ctx()
    b = E.rule_block(ctx, "primary_final", ctx[E.PX27].copy(), 0, log=lambda *a: None)
    s = b["summary"]
    assert s[E.CH]["orig_bit_identical"] and s[E.CH]["orig_uF"] == 0 and s[E.CH]["orig_uM"] == 0
    assert b["positive_control"]["ok"], b["positive_control"]
    assert s["half_px27_d4"]["orig_half"] == Fraction(-17, 18000) and s["d4"]["orig_half"] == Fraction(-30, 18000)
    assert s["copy_m3x3_full"]["orig_half"] == 0 and s["copy_px27"]["orig_bit_identical"]


@pytest.mark.skipif(not (COMP / "data" / E.NOTES_REL).is_file(), reason="note file not available locally")
def test_real_runner_emulation_and_notes_check():
    cc = E.config_check()
    rep = E.runner_emulation(cc)
    assert rep["problems"] == []
    assert all(j["check_recipe"] == "passed" and j["foreign_recipe"] == "rejected" and j["rows_in_fold"] for j in rep["jobs"].values())
    assert rep["test"]["rows"] == rep["test"]["test_rows"] == rep["test"]["without_finding"] == 3498
    from vlm import common as VC

    nt = E.check_notes(VC.find_data_root(), 1500)
    assert nt["px27_notes"]["identical_notes"] == 1500 and nt["rows"] == 1500
