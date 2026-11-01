"""Tests of the deployment path of the channel q397m3: eda/c117_q397m3_deploy.py (OOF artifact, guard of the test side),
eda/c110_q397_deploy.py stack --replace-q397 (blend file), tools/make_submission.py (fold shards as a source, note guard
rank.notes, --rank-only-placeholder) and tools/builds/s029_q397m3.sh.

Synthetic (tmp_path, no test row): comparison of settings per shard, note guard (acceptance and every refusal), placeholder,
log lines, c115 verdict, cross-check of the a8 path, switches of c110. With real training artifacts (skipped if they are
missing): oof_q397m3 bit-identical to the shards and to the channel of c115, blend file = recomputation, note guard against the
real note files (checksums and meta files only), guard of the test side with the d3 test ranking as a placeholder, fixed
checksums of the build script, exit 3 without a test side, and the existing six-channel build path byte-identical."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_submission_s029q_test", "tools/make_submission.py")
D110 = _load("c110_s029q_test", "eda/c110_q397_deploy.py")
D = _load("c117_s029q_test", "eda/c117_q397m3_deploy.py")

TRAIN_REL, TEST_REL = "derived/sft/q397m3_train_notes.jsonl", "derived/sft/q397m3_test_notes.jsonl"
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
SRC_RUN = "runs/q397te_rp/proto_te/proto"
BUILD = COMP / "tools" / "builds" / "s029_q397m3.sh"
STANDIN = "runs/m3d3te_rp/m3d3_test_rank/test"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ make_submission: shard source

def _rank_meta_file(comp: Path, run: str, job: str, tag: str, table: str, sft_from: str, beams: int = 74, model: str = "m"):
    d = comp / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    meta = {"table": table, "n": 3, "probes": ["proto_rank"], "job": job,
            "recipe": {"beams": beams, "image": {"max_side": 1024, "question_max_side": 448}, "model": model, "fold": None},
            "dtype": "bf16", "quant": None, "adapter": False, "sft": {"sft_from": sft_from, "finish_tokens": 320}}
    (d / f"{tag}_meta.json").write_text(json.dumps(meta))


def test_source_differences_per_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(MS.a8, "COMP", tmp_path)
    _rank_meta_file(tmp_path, "T", "t", "test", "test", TEST_REL, model="full")
    for j, b in (("f0", 74), ("f1", 74)):
        _rank_meta_file(tmp_path, "R", j, "bench", "train", TRAIN_REL, beams=b, model=j)
    src = "runs/R/f0/bench+runs/R/f1/bench"
    assert MS.rank_only_source_differences("runs/T/t/test", src) == {}  # model and note source are free
    assert MS.rank_only_source_differences("runs/T/t/test", "runs/R/f0/bench") == MS.rank_only_recipe_differences("runs/T/t/test", "runs/R/f0/bench") == {}
    _rank_meta_file(tmp_path, "R", "f1", "bench", "train", TRAIN_REL, beams=18, model="f1")
    diff = MS.rank_only_source_differences("runs/T/t/test", src)
    assert list(diff) == ["recipe.beams [f1]"] and diff["recipe.beams [f1]"] == (74, 18)
    with pytest.raises(SystemExit):  # the earlier function knows no shard source (the new path handles it)
        MS.rank_only_recipe_differences("runs/T/t/test", src)


# ------------------------------------------------------------------------------------------------ make_submission: note guard

def _notes(root: Path, rel: str, table: str, n: int, upload_dir: str, extra: dict) -> str:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps({"table": table, "row": i, "state": "open", "text": f"note {i}", "finding": None}) + "\n" for i in range(n))
    f.write_text(body)
    meta = {"table": table, "rows": n, "with_record": n, "states": {"open": n}, "sha256": sha(body.encode()), **extra}
    f.with_suffix(".meta.json").write_text(json.dumps(meta))
    up = root / "derived" / upload_dir / f.name
    up.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(f, up)
    return meta["sha256"]


def _world(tmp_path: Path, monkeypatch, n_train: int = 4, n_test: int = 3) -> dict:
    comp, root = tmp_path / "comp", tmp_path / "data"
    art = comp / "artifacts"
    monkeypatch.setattr(MS, "COMP", comp)
    monkeypatch.setattr(MS.a8, "COMP", comp)
    src_dir = art / "runs" / "q397te_rp" / "q397te_rp" / "proto_te"
    src_dir.mkdir(parents=True)
    (src_dir / "proto_raw.jsonl").write_text("raw\n")
    (src_dir / "proto_meta.json").write_text("{}\n")
    tr_sha = _notes(root, TRAIN_REL, "train", n_train, "sft_q397m3", {"source_kind": "tinker"})
    _notes(root, TEST_REL, "test", n_test, "sft_q397m3te", {"source": SRC_RUN, "model": MODEL, "source_sha256": sha(b"raw\n"),
                                                             "meta_sha256": sha(b"{}\n")})
    for j in ("q397m3_rank_f0", "q397m3_rank_f1"):
        _rank_meta_file(comp, "R", j, "bench", "train", TRAIN_REL, model=j)
    _rank_meta_file(comp, "T", "q397m3_test_rank", "test", "test", TEST_REL, model="full")
    source = "runs/R/q397m3_rank_f0/bench+runs/R/q397m3_rank_f1/bench"
    np.save(art / "oof_q397m3.npy", np.eye(3))
    notes = {"train": {"sft_from": TRAIN_REL, "sha256": tr_sha, "upload": "derived/sft_q397m3/q397m3_train_notes.jsonl"},
             "test": {"sft_from": TEST_REL, "upload": "derived/sft_q397m3te/q397m3_test_notes.jsonl", "source_runs": [SRC_RUN], "model": MODEL}}
    (art / "meta_q397m3.json").write_text(json.dumps({"rank": {"source": source, "notes": notes}}))
    (art / "meta_plain.json").write_text(json.dumps({"rank": {"source": "runs/R/q397m3_rank_f0/bench"}}))
    stack = {"rank_only": {"q397m3": {"floor": 0.05, "rank_source": source, "oof_sha256": sha((art / "oof_q397m3.npy").read_bytes())[:16]}}}
    return {"comp": comp, "root": root, "art": art, "stack": stack, "n_train": n_train, "n_test": n_test, "spec": "runs/T/q397m3_test_rank/test"}


def _guard(w: dict, spec: str | None = None, stack=-1, placeholder: bool = False, name: str = "q397m3"):
    return MS.rank_only_notes_guard(name, spec or w["spec"], w["root"], w["n_test"], w["n_train"], stack=w["stack"] if stack == -1 else stack,
                                    placeholder=placeholder, art_dir=w["art"])


def test_notes_guard_accepts_the_intended_sources(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    info = _guard(w)
    assert info["notes_test_source"] == SRC_RUN and info["notes_test_model"] == MODEL and info["placeholder"] is False
    assert info["notes_train_sha256"] == sha((w["root"] / TRAIN_REL).read_bytes())
    assert _guard(w, stack=None)["notes_test"] == TEST_REL  # without a blend file (c117 test-guard)
    assert _guard(w, name="plain") is None  # channel without the field rank.notes: nothing checked (all other channels)
    with pytest.raises(SystemExit, match="no note guard"):
        _guard(w, name="plain", placeholder=True)


@pytest.mark.parametrize("case,match", [
    ("test_sft_from", "notes of the 397B"),
    ("test_sha", "checksum does not match"),
    ("test_upload", "upload folder"),
    ("model", "from model"),
    ("source_run", "allowed"),
    ("raw_sha", "do not match run"),
    ("states", "not available for every row"),
    ("rows", "required test"),
    ("train_sft_from", "OOF ranking"),
    ("train_sha", "not the note file of the OOF artifact"),
    ("stack_sha", "not the artifact of the blend"),
    ("stack_source", "different ranking"),
])
def test_notes_guard_refusals(tmp_path, monkeypatch, case, match):
    w = _world(tmp_path, monkeypatch)
    root, comp, art = w["root"], w["comp"], w["art"]
    tmeta = (root / TEST_REL).with_suffix(".meta.json")
    if case == "test_sft_from":
        _rank_meta_file(comp, "T", "q397m3_test_rank", "test", "test", "m3d3_test_gen/gen", model="full")
    elif case == "test_sha":
        (root / TEST_REL).write_text((root / TEST_REL).read_text() + "\n")
    elif case == "test_upload":
        (root / "derived" / "sft_q397m3te" / "q397m3_test_notes.jsonl").write_text("different\n")
    elif case in ("model", "source_run", "states", "rows"):
        m = json.loads(tmeta.read_text())
        m.update({"model": {"model": "Qwen/Qwen3.8-27B"}, "source_run": {"source": "runs/pr27te_rp/proto_te/proto"},
                  "states": {"states": {"open": w["n_test"] - 1, "empty": 1}}, "rows": {"rows": w["n_test"] + 1}}[case])
        tmeta.write_text(json.dumps(m))
    elif case == "raw_sha":
        (art / "runs" / "q397te_rp" / "q397te_rp" / "proto_te" / "proto_raw.jsonl").write_text("new\n")
    elif case == "train_sft_from":
        _rank_meta_file(comp, "R", "q397m3_rank_f1", "bench", "train", "m3d3_gen_f1/gen", model="q397m3_rank_f1")
    elif case == "train_sha":
        m = json.loads((art / "meta_q397m3.json").read_text())
        m["rank"]["notes"]["train"]["sha256"] = "0" * 64
        (art / "meta_q397m3.json").write_text(json.dumps(m))
    elif case == "stack_sha":
        w["stack"]["rank_only"]["q397m3"]["oof_sha256"] = "0" * 16
    elif case == "stack_source":
        w["stack"]["rank_only"]["q397m3"]["rank_source"] = "runs/R/q397m3_rank_f0/bench"
    with pytest.raises(SystemExit, match=match):
        _guard(w)


def test_notes_guard_placeholder_skips_only_the_test_side(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    _rank_meta_file(w["comp"], "T", "q397m3_test_rank", "test", "test", "m3d3_test_gen/gen", model="full")  # foreign test ranking
    with pytest.raises(SystemExit, match="notes of the 397B"):
        _guard(w)
    info = _guard(w, placeholder=True)
    assert info["placeholder"] is True and "notes_test" not in info and "PLACEHOLDER" in capsys.readouterr().out
    _rank_meta_file(w["comp"], "R", "q397m3_rank_f0", "bench", "train", "m3_gen_f0/gen", model="f0")  # the training side stays checked
    with pytest.raises(SystemExit, match="OOF ranking"):
        _guard(w, placeholder=True)


def test_parse_band_for():
    ro = {"q397m3": "runs/a/b/test", "m3x3_full": "runs/c/d/test,runs/e/f/test"}
    assert MS.parse_band_for([], ro) == {}  # default: --rank-none-band for all channels
    assert MS.parse_band_for(["q397m3=0.27,0.35"], ro) == {"q397m3": (0.27, 0.35)}
    for bad in (["pr27xm=0.2,0.3"], ["q397m3=0.35,0.27"], ["q397m3=0.3"], ["q397m3"], ["q397m3=a,b"], ["q397m3=0.2,0.3", "q397m3=0.2,0.4"],
                ["q397m3=-0.1,0.3"]):
        with pytest.raises(SystemExit):
            MS.parse_band_for(bad, ro)


def test_none_band_same_in_script_and_guard():
    """The NONE band of the channel, fixed in advance, is the same in the build script and in c117; it encloses the OOF value ± 0.04."""
    m = re.search(r'^NONE_BAND_Q397M3="([0-9.]+),([0-9.]+)"', BUILD.read_text(), re.M)
    assert m and (float(m.group(1)), float(m.group(2))) == D.NONE_BAND
    assert '--rank-none-band-for "$CH=$NONE_BAND_Q397M3"' in BUILD.read_text()
    oof = ART / "oof_q397m3.npy"
    if oof.is_file():
        share = float(np.mean(np.load(oof).argmax(1) == 0))
        assert abs(share - 0.310) < 0.0005 and D.NONE_BAND == (round(share - 0.04, 2), round(share + 0.04, 2))


# ------------------------------------------------------------------------------------------------ c110: switch --replace-q397

def test_replace_stem_and_spec():
    assert D110.stem_of("best", "m3x3_full", False, "q397m3") == "s029_best_q397m3"
    assert D110.stem_of("best", "m3x3_full", False) == "s029_best_m3x3" and D110.stem_of("safe", "m3x2_full_cv01", True) == "s029_safe_bn"
    _, spec_b = D110.line_specs("best", "m3x3_full")
    spec_r = D110.replaced_spec(spec_b, "q397", "q397m3")
    assert spec_r["names"] == ["v1r", "zs27q", "think18nb", "pr27xm", "m3x3_full", "q397m3"]  # position of q397
    assert spec_r["floors"] == {"think18nb": 0.01, "pr27xm": 0.05, "m3x3_full": 0.05, "q397m3": 0.05}
    assert spec_r["beta"] == 0.2 and spec_r["think"] == "think18nb" and spec_r["factor"] == 1.4 and spec_r["last"] == "trans"
    assert spec_b["names"][-1] == "q397"  # base recipe unchanged
    with pytest.raises(SystemExit):
        D110.replaced_spec(spec_b, "doesnotexist", "q397m3")
    assert D110.REPLACE_SOURCE["q397m3"] == D.SOURCE  # the same source as the artifact from c117


@pytest.mark.parametrize("argv", [
    ["stack", "--line", "safe", "--m3", "m3x3_full", "--replace-q397", "q397m3", "--verify"],
    ["stack", "--line", "best", "--m3", "m3x2_full_cv01", "--replace-q397", "q397m3", "--verify"],
    ["stack", "--line", "best", "--m3", "m3x3_full", "--replace-q397", "q397m3", "--beta-novel", "--verify"],
    ["stack", "--line", "best", "--m3", "m3x3_full", "--replace-q397", "q397x", "--verify"],
])
def test_replace_switch_refuses_unmeasured_forms(argv):
    with pytest.raises(SystemExit) as exc:
        D110.main(argv)
    assert exc.value.code == 2


# ------------------------------------------------------------------------------------------------ c117: c115 verdict, log, a8 path

def _c115_json(path: Path, notes_sha: str, **over) -> None:
    js = {"dry": False, "decision": {"verdict": "CONTINUE", "passed_forms": ["replacement"]},
          "technical": {"notes": {"sha256": notes_sha}, "shards": [{"shard": s} for s in D.SHARDS], "channel_sha256": "ab" * 32},
          "anchor": {"units": 7463, "logloss": 0.867},
          "recipes": {"replacement": {"map_new": 7494 / 9000, "ll_new": 0.8668, "delta_exact": "31/9000", "delta": 0.00344, "se_q": 0.0028, "z_q": 1.24}}}
    for k, v in over.items():
        js[k] = v
    path.write_text(json.dumps(js))


@pytest.mark.parametrize("over,code", [
    ({}, None),
    ({"dry": True}, 4),
    ({"decision": {"verdict": "STOP", "passed_forms": []}}, 4),
    ({"decision": {"verdict": "CONTINUE", "passed_forms": ["addition"]}}, 4),
    ({"anchor": {"units": 7453, "logloss": 0.87}}, 4),
])
def test_load_c115(tmp_path, over, code):
    f = tmp_path / "c115.json"
    _c115_json(f, "n" * 64, **over)
    if code is None:
        r = D.load_c115("n" * 64, f)
        assert r["replacement_units"] == 7494 and r["anchor_units"] == 7463 and r["passed_forms"] == ["replacement"]
        with pytest.raises(SystemExit):  # a different note file
            D.load_c115("m" * 64, f)
    else:
        with pytest.raises(SystemExit) as exc:
            D.load_c115("n" * 64, f)
        assert exc.value.code == code
    with pytest.raises(SystemExit) as exc:
        D.load_c115("n" * 64, tmp_path / "missing.json")
    assert exc.value.code == 3


def test_run_log_check(tmp_path, monkeypatch):
    monkeypatch.setattr(D.A9, "run_dir", lambda run: tmp_path / run)
    spec = "runs/q397m3te_rp/q397m3_test_rank/test"
    d = tmp_path / "q397m3te_rp"
    assert D.run_log_check(spec, 3498)[0]  # log missing
    d.mkdir()
    # log line of vlm/runner.py; the recorded runs used an earlier German wording of the line, which is accepted as well
    good = "12:00:00 sft_from: notes for 3498 rows of table test, of which without a finding 3498 (findings are generated before scoring)\n"
    legacy = "12:00:00 sft_from: Ausgaben zu 3498 Zeilen der Tabelle test, davon ohne Befund 3498 (werden vor der Bewertung erzeugt)\n"
    (d / "worker0.log").write_text("Start\n" + good + "probe proto_rank [predict]: 6 rows\n" + legacy)
    problems, info = D.run_log_check(spec, 3498)
    assert problems == [] and info["lines"] == 2
    (d / "worker0.log").write_text(good + good.replace("without a finding 3498", "without a finding 1266"))  # e.g. M3's own notes
    assert any("1266" in p for p in D.run_log_check(spec, 3498)[0])
    (d / "worker0.log").write_text(good.replace("table test", "table train"))
    assert D.run_log_check(spec, 3498)[0]
    (d / "worker0.log").write_text("nothing\n")
    assert any("no line" in p for p in D.run_log_check(spec, 3498)[0])


def _shard(comp: Path, k: int, rows: np.ndarray, K: int):
    d = comp / "artifacts" / "runs" / "q397m3_oof_rp" / "q397m3_oof_rp" / f"q397m3_rank_f{k}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(10 + k)
    lp = np.log(rng.dirichlet(np.ones(K), size=len(rows)))
    lp[:, -1] = -np.inf  # class without a training example
    np.save(d / "bench_idx.npy", np.asarray(rows, dtype=np.int64))
    np.save(d / "bench_proto_rank.npy", lp)
    np.save(d / "bench_proto_rank.done.npy", np.ones(len(rows), bool))
    (d / "bench_meta.json").write_text(json.dumps({"table": "train", "n": int(len(rows)), "probes": ["proto_rank"], "job": f"q397m3_rank_f{k}",
                                                   "sft": {"sft_from": TRAIN_REL, "finish_tokens": 320}}))


def test_a8_counter_equals_assemble_synthetic(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(D.A8, "COMP", tmp_path)
    n, K = 15, 6
    folds = np.array([3, 1, 4, 0, 2] * 3)
    seen = np.array([True] * (K - 1) + [False])
    for k in range(5):
        _shard(tmp_path, k, np.flatnonzero(folds == k)[::-1], K)  # rows within the shard deliberately not sorted
    P, have, _ = D.C115.assemble(D.SHARDS, n, seen, folds)
    R, Pa8 = D.a8_counter(D.SHARDS, n, seen, ["c"] * K, named_source=None)
    assert have.all() and Pa8 is None and np.array_equal(R, P) and not P[:, ~seen].any()


# ------------------------------------------------------------------------------------------------ real artifacts

def _need(*paths: Path) -> None:
    missing = [str(p.relative_to(COMP)) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing locally: {missing[:3]}")


def _pins() -> dict:
    text = BUILD.read_text()
    return {k: re.search(rf'^{k}="([0-9a-f]{{64}})"', text, re.M).group(1) for k in ("OOF_SHA", "META_SHA", "STACK_SHA", "NOTES_TEST_SHA")}


ART = COMP / "artifacts"
STACK = ART / "stack_s029_best_q397m3.json"


def test_real_oof_artifact_is_the_evaluated_channel():
    pytest.importorskip("torch")
    _need(ART / "oof_q397m3.npy", ART / "meta_q397m3.json", D.C115_JSON, *(ART / "runs" / "q397m3_oof_rp" / "q397m3_oof_rp" / f"q397m3_rank_f{k}" for k in range(5)))
    ctx = D.C110.setup()
    P, meta, _ = D.build_oof(ctx)
    assert np.array_equal(np.load(ART / "oof_q397m3.npy"), P)
    assert (ART / "meta_q397m3.json").read_text(encoding="utf-8") == D.meta_text(P, meta, ctx["K"])
    assert D.C115.npy_sha256(P) == json.loads(D.C115_JSON.read_text())["technical"]["channel_sha256"]
    rk = meta["rank"]
    assert rk["source"] == D.SOURCE and rk["rows_named_only"] == 0 and rk["notes"]["test"]["sft_from"] == TEST_REL
    assert meta["checks"] == {"c115_bit_identical": True, "a8_combined_bit_identical": True, "a8_with_pr27_named_bit_identical": True, "fold_check_clean": True}
    assert MS.rank_only_guard("q397m3", json.loads(STACK.read_text())) == D.SOURCE
    pins = _pins()
    assert sha((ART / "oof_q397m3.npy").read_bytes()) == pins["OOF_SHA"] and sha((ART / "meta_q397m3.json").read_bytes()) == pins["META_SHA"]


def test_real_stack_fields():
    _need(STACK, ART / "stack_s029_best_m3x3.json")
    s = json.loads(STACK.read_text())
    assert s["channels"] == ["v1r", "zs27q", "think18nb", "pr27xm", "m3x3_full", "q397m3"] and len(s["theta"]) == 6
    assert "proto" not in s and set(s["rank_only"]) == {"m3x3_full", "q397m3"} and s["none"] == [] and s["probes"] == {}
    assert s["floor"] == {"think18nb": 0.01, "pr27xm": 0.05, "m3x3_full": 0.05, "q397m3": 0.05}
    assert s["rank_only"]["q397m3"]["floor"] == 0.05 and s["rank_only"]["q397m3"]["rank_source"] == D.SOURCE
    b = json.loads((ART / "stack_s029_best_m3x3.json").read_text())
    assert s["rank_only"]["m3x3_full"] == b["rank_only"]["m3x3_full"] and s["scaled"] == b["scaled"] == {"think18nb": 1.4}
    assert round(s["cv"] * 9000, 6) == 7494 and s["cross"]["six_channel"] == b["cv"] and round(b["cv"] * 9000, 6) == 7463  # units of 1/6
    assert MS.arith_beta_of(s) == 0.2
    assert MS.arith_beta_novel_of(s) is None and "inductive_check" not in s
    assert MS.stack_fit_rows(s, 1500).tolist() == list(range(1500))
    assert s["replaces"]["stack_six_channel_sha256"] == sha((ART / "stack_s029_best_m3x3.json").read_bytes())
    assert sha(STACK.read_bytes()) == _pins()["STACK_SHA"]


def test_real_stack_is_the_recomputation():
    pytest.importorskip("torch")
    _need(STACK, ART / "oof_q397m3.npy", D110.REPLACE_C115)
    stack, rep = D110.build_replaced("best", "m3x3_full", "q397m3")  # contains the cross-check against stack_s029_best_m3x3.json
    assert STACK.read_text() == D110.to_json(stack)
    assert rep["cross"]["units"] == 7494 and rep["cross"]["six_channel"]["units"] == 7463
    assert str(rep["cross"]["delta_vs_six_channel"]["delta_exact"]) == "31/9000"  # Δ exactly as in c115


def test_existing_stacks_unchanged():
    pins = {"stack_s029_best.json": "29d2a79450848e0d61c7367cce431431684d9d4a646836af7317c2731b3ae20d",
            "stack_s029_safe.json": "883f2e7a37ce3b61a1b7020a81bfddfbeec4b4a7f847ab3807232bceee19d3ef",
            "stack_s029_best_bn.json": "8d2513e43b6d715503d0f9e172a805f4d34a8dcd2687e47717c2308e20280399",
            "stack_s029_safe_bn.json": "73f163b0cabfc9ef64a9956ab808b15d58e4df3566858b67ff487e8e65ca9cab",
            "stack_s029_best_m3x3.json": "9c98234642e51f16a77f6deb746b81d8f73523c706a69f1a5690a76a63ef2d97"}
    _need(*(ART / f for f in pins))
    for f, want in pins.items():
        assert sha((ART / f).read_bytes()) == want, f


def test_real_notes_guard_with_real_notes():
    """Real note files (checksums and meta files only): the d3 test ranking is refused in real mode, a test ranking behind
    derived/sft/q397m3_test_notes.jsonl is accepted (meta file of the ranking faked)."""
    from vlm import common as C

    root = C.find_data_root()
    _need(STACK, ART / "meta_q397m3.json", root / TEST_REL, root / TRAIN_REL, ART / "runs" / "q397te_rp" / "q397te_rp" / "proto_te" / "proto_raw.jsonl",
          ART / "runs" / "m3d3te_rp" / "m3d3te_rp" / "m3d3_test_rank" / "test_meta.json")
    stack = json.loads(STACK.read_text())
    with pytest.raises(SystemExit, match="m3d3_test_gen"):
        MS.rank_only_notes_guard("q397m3", STANDIN, root, 3498, 1500, stack=stack)
    orig = MS._rank_meta

    def fake(s):
        return {"table": "test", "sft": {"sft_from": TEST_REL, "finish_tokens": 320}} if s == "runs/FAKE/x/test" else orig(s)

    MS._rank_meta = fake
    try:
        info = MS.rank_only_notes_guard("q397m3", "runs/FAKE/x/test", root, 3498, 1500, stack=stack)
    finally:
        MS._rank_meta = orig
    assert info["notes_test_source"] == SRC_RUN and info["notes_test_model"] == MODEL and info["notes_test_sha256"] == _pins()["NOTES_TEST_SHA"]


def test_real_test_guard_with_standin():
    _need(ART / "meta_q397m3.json", ART / "runs" / "m3d3te_rp" / "m3d3te_rp" / "m3d3_test_rank" / "test_meta.json",
          ART / "runs" / "m3_fullte_rp" / "m3_fullte_rp" / "m3_train_full" / "sft_recipe.json")
    code, lines, info = D.test_guard(STANDIN, placeholder=True)
    assert code == 0, lines
    assert info["settings_equal"] and info["training_regime"]["test_recipe"]["fold"] is None and info["training_regime"]["variant_code"] == 0
    code, lines, info = D.test_guard(STANDIN, placeholder=False)  # real mode: d3 is not M3 behind the 397B notes
    assert code == 5 and any("m3d3_test_gen" in p for p in info["problems"]) and any("log" in p for p in info["problems"])
    code, lines, _ = D.test_guard("runs/q397m3te_rp/doesnotexist/test")
    assert code == 3


REAL_TEST = "runs/q397m3te_rp/q397m3_test_rank/test"


def _real_test_complete() -> bool:
    r = subprocess.run([sys.executable, "-I", "eda/c95_rank_mean.py", "--status", REAL_TEST, "--table", "test"], cwd=COMP, capture_output=True, text=True,
                       timeout=600)
    return r.returncode == 0


def test_real_test_guard_on_the_fetched_run():
    """Real q397m3 test side (also a partial state): settings, 397B notes, log and training regime without a finding; exit 0 only
    when complete, otherwise 3. Aggregates only."""
    if not (ART / "runs" / "q397m3te_rp" / "q397m3te_rp" / "q397m3_test_rank" / "test_meta.json").is_file():
        pytest.skip("the q397m3 test side has not been fetched yet")
    code, lines, info = D.test_guard(REAL_TEST, notes_source=SRC_RUN)
    assert info["problems"] == [], lines
    assert code == (0 if info["coverage"]["complete"] else 3)
    assert info["notes"]["notes_test_sha256"] == _pins()["NOTES_TEST_SHA"] and info["log"]["lines"] >= 1


def test_build_script_exit3_without_test_side():
    if _real_test_complete():
        pytest.skip("the q397m3 test side is complete")
    r = subprocess.run(["bash", str(BUILD), "best"], cwd=COMP, capture_output=True, text=True, timeout=600)
    assert r.returncode == 3 and "q397m3 test side" in r.stderr
    assert not (ART / "submissions" / "sub_s029_best_q397m3.csv").exists()
    r = subprocess.run(["bash", str(BUILD), "safe"], cwd=COMP, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2


def test_make_submission_placeholder_needs_out_outside_submissions(tmp_path):
    _need(STACK, ART / "meta_q397m3.json", ART / "runs" / "m3d3te_rp" / "m3d3te_rp" / "m3d3_test_rank" / "test_meta.json")
    base = [sys.executable, "-I", "tools/make_submission.py", "--name", "s029_best_q397m3",
            "--rank-only", "m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test",
            "--rank-only", f"q397m3={STANDIN}", "--rank-only-placeholder", "q397m3"]
    for extra in ([], ["--out", str(ART / "submissions" / "sub_s029_best_q397m3_PLACEHOLDER.csv")]):
        r = subprocess.run(base + extra, cwd=COMP, capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "outside artifacts/submissions" in r.stderr
    r = subprocess.run(base[:-2], cwd=COMP, capture_output=True, text=True, timeout=600)  # without the placeholder: the note guard refuses d3
    assert r.returncode != 0 and "note guard violated" in r.stderr
    assert not (ART / "submissions" / "sub_s029_best_q397m3_PLACEHOLDER.csv").exists()


BEST_COMMON = ["--test", "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy", "--fill", "v1r=v1",
               "--test", "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test",
               "--test", "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test", "--fill", "think18nb=zs27q",
               "--test", "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy",
               "--proto", "pr27xm=runs/pr27te_rp/proto_te/proto", "--rank", "pr27xm=runs/n1_rp/rank_test/test"]
M3X3 = "m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test"


@pytest.mark.parametrize("label,name,ref,args", [
    ("six-channel file with q397", "s029_best_m3x3", "704d8ae8e6eb93051aa4fbe8ec954da143c1c68ff751fc6fe0063c681c063b18",
     BEST_COMMON + ["--proto", "q397=runs/q397te_rp/proto_te/proto", "--rank-only", M3X3, "--align", "0.75", "--align-min", "3",
                    "--blank-rows", "derived/blank/test_rows.npy"]),
])
def test_existing_build_paths_bytegleich(tmp_path, label, name, ref, args):
    """The existing six-channel build path (channel from three draws, --proto q397) with the current make_submission: the same
    file as built before. Output to tmp_path; the test matrix is deleted."""
    stack = ART / f"stack_{name}.json"
    _need(stack, ART / "runs" / "m3_fullte_rp", ART / "runs" / "n1_rp", ART / "runs" / "pr27te_rp")
    out, suffix = tmp_path / f"{name}.csv", "_pytest_bytecheck"
    try:
        r = subprocess.run([sys.executable, "-I", "tools/make_submission.py", "--name", name, *args, "--suffix", suffix, "--out", str(out)],
                           cwd=COMP, capture_output=True, text=True, timeout=900)
    finally:
        (ART / f"test_{name}{suffix}.npy").unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert sha(out.read_bytes()) == ref, label
