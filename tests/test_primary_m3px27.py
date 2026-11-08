"""Tests of the deployment path primary_m3px27: eda/c133_primary_m3px27.py, the additive paths in tools/make_submission.py (field
rank_mean.rank_compare "sft", note guard rank_mean.notes) and the additive arguments of eda/c118_stage2.build_stage2.

Note guard and comparison form only SYNTHETIC in tmp_path (no test row, no real test note). With real training artifacts (skipped if
they are missing): mean channel = matrix of the expected-value rule computation, blend and deployment file against the final primary
form, positive control of the test guard with the px27 test ranking (return code and aggregates only; no content of a test row)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
ART = COMP / "artifacts"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_submission_primary_m3px27_test", "tools/make_submission.py")
M = _load("c133_primary_m3px27_test", "eda/c133_primary_m3px27.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        pytest.skip(f"artifacts missing: {missing}")


# ------------------------------------------------------------------------------------------------ make_submission: note guard (synthetic)

def _synthetic(tmp_path: Path, monkeypatch, n_train: int = 4, n_test: int = 6) -> tuple[Path, dict, dict]:
    """Data root with frozen notes (training and test file with meta file), source run of the test notes, meta files of the rankings."""
    root, comp = tmp_path / "data", tmp_path / "comp"

    def notes(rel: str, table: str, rows: int, extra: dict) -> str:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps({"table": table, "row": i, "state": "open", "text": f"note {i}"}) + "\n" for i in range(rows)))
        sha = _sha(f)
        f.with_suffix(".meta.json").write_text(json.dumps({"table": table, "rows": rows, "sha256": sha, **extra}))
        return sha

    d = comp / "artifacts" / "runs" / "pr" / "pr" / "pj"
    d.mkdir(parents=True)
    (d / "proto_raw.jsonl").write_text('{"row": 0}\n')
    (d / "proto_meta.json").write_text('{"table": "test"}')
    tr_sha = notes("derived/a/train_notes.jsonl", "train", n_train, {})
    notes("derived/b/test_notes.jsonl", "test", n_test,
          {"model": "M27", "with_record": n_test, "states": {"open": n_test}, "source": "runs/pr/pj/proto", "source_sha256": _sha(d / "proto_raw.jsonl"),
           "meta_sha256": _sha(d / "proto_meta.json"), "px27_notes": {"sha256": "PX27", "identical_notes": n_test}})
    metas = {"runs/o/f0/bench": {"table": "train", "sft": {"sft_from": "derived/a/train_notes.jsonl"}},
             "runs/o/f1/bench": {"table": "train", "sft": {"sft_from": "derived/a/train_notes.jsonl"}},
             "runs/t/tj/test": {"table": "test", "sft": {"sft_from": "derived/b/test_notes.jsonl"}}}
    monkeypatch.setattr(MS, "_rank_meta", lambda spec: metas[spec])
    monkeypatch.setattr(MS, "COMP", comp)
    rm = {"rank_source": "runs/o/f0/bench+runs/o/f1/bench", "floor": 0.05, "rank_compare": "sft",
          "notes": {"train": {"sft_from": "derived/a/train_notes.jsonl", "sha256": tr_sha, "upload": "derived/a/train_notes.jsonl"},
                    "test": {"sft_from": "derived/b/test_notes.jsonl", "upload": "derived/b/test_notes.jsonl", "source_runs": ["runs/pr/pj/proto"],
                             "model": "M27", "px27_file": "derived/px.jsonl", "px27_sha256": "PX27"}}}
    return root, rm, metas


def test_notes_guard_without_field_is_a_noop():
    """All earlier mean channels (without rank_mean.notes): nothing checked, no file read."""
    assert MS.rank_mean_notes_guard("pr27xm", "runs/x/y/z", Path("/does/not/exist"), 10, 5, {"floor": 0.05}) is None
    assert MS.rank_mean_notes_guard("pr27xm", "runs/x/y/z", Path("/does/not/exist"), 10, 5, None) is None
    with pytest.raises(SystemExit, match="incompletely"):
        MS.rank_mean_notes_guard("m3px27xm", "runs/x/y/z", Path("/does/not/exist"), 10, 5, {"notes": {"train": {"sft_from": "a"}, "test": {}}})


def test_notes_guard_synthetic_holds_and_rejects(tmp_path, monkeypatch):
    root, rm, metas = _synthetic(tmp_path, monkeypatch)
    out = MS.rank_mean_notes_guard("m3px27xm", "runs/t/tj/test", root, 6, 4, rm)
    assert out["notes_test_model"] == "M27" and out["notes_test_px27_identical"] == 6 and out["notes_test_source"] == "runs/pr/pj/proto"
    assert out["notes_train_sha256"] == rm["notes"]["train"]["sha256"]

    def rejects(mutate, match: str, rows=(6, 4)) -> None:
        r, m = copy.deepcopy(rm), copy.deepcopy(metas)
        mutate(r, m)
        monkeypatch.setattr(MS, "_rank_meta", lambda spec: m[spec])
        with pytest.raises(SystemExit, match=match):
            MS.rank_mean_notes_guard("m3px27xm", "runs/t/tj/test", root, rows[0], rows[1], r)
        monkeypatch.setattr(MS, "_rank_meta", lambda spec: metas[spec])

    rejects(lambda r, m: r["notes"]["test"].update(model="other"), "model")
    rejects(lambda r, m: m["runs/o/f1/bench"]["sft"].update(sft_from="derived/x.jsonl"), "OOF ranking")
    rejects(lambda r, m: m["runs/t/tj/test"]["sft"].update(sft_from="derived/x.jsonl"), "test ranking ranked")
    rejects(lambda r, m: m["runs/t/tj/test"].update(table="train"), "table")
    rejects(lambda r, m: r["notes"]["train"].update(sha256="0" * 64), "note file of the OOF artifact")
    rejects(lambda r, m: r["notes"]["test"].update(source_runs=["runs/q/q/q"]), "allowed")
    rejects(lambda r, m: r["notes"]["test"].update(px27_sha256="other"), "character-identical")
    rejects(lambda r, m: None, "rows", rows=(7, 4))  # test notes not for every row
    (root / "derived/b/test_notes.jsonl").write_text("changed\n")  # the file no longer matches its meta file
    with pytest.raises(SystemExit, match="checksum does not match"):
        MS.rank_mean_notes_guard("m3px27xm", "runs/t/tj/test", root, 6, 4, rm)


def test_proto_channel_compare_form(monkeypatch):
    """rank_compare "sft" compares via rank_only_source_differences (every shard), the default stays c95.recipe_differences."""
    calls = []

    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached

    monkeypatch.setattr(MS.c95, "rank_files_missing", lambda s: [])
    monkeypatch.setattr(MS, "rank_only_source_differences", lambda a, b: calls.append(("sft", b)) or {"recipe.image [f0]": (1, 2)})
    monkeypatch.setattr(MS.c95, "recipe_differences", lambda a, b: calls.append(("c95", b)) or {})
    monkeypatch.setattr(MS.c95, "channel", reached)
    seen = np.ones(2, bool)
    with pytest.raises(SystemExit, match="computed differently"):
        MS.proto_channel("k", ["d"], "runs/a/b/c", 3, ["NONE", "X"], seen, ref_rank="runs/o/f0/bench+runs/o/f1/bench", ref_compare="sft")
    with pytest.raises(Reached):
        MS.proto_channel("k", ["d"], "runs/a/b/c", 3, ["NONE", "X"], seen, ref_rank="runs/o/f0/bench")
    assert calls == [("sft", "runs/o/f0/bench+runs/o/f1/bench"), ("c95", "runs/o/f0/bench")]
    with pytest.raises(SystemExit, match="unknown comparison form"):
        MS.proto_channel("k", ["d"], "runs/a/b/c", 3, ["NONE", "X"], seen, ref_rank="runs/o/f0/bench", ref_compare="other")


def test_build_stage2_argument_is_additive():
    sig = inspect.signature(M.c118().build_stage2).parameters
    assert list(sig) == ["ctx", "oof", "forms", "cv_units_want"]
    assert sig["forms"].default is None and sig["cv_units_want"].default is None


# ------------------------------------------------------------------------------------------------ c133: comparison form of the copy null point

def test_normalized_compare_renames_and_drops_provenance():
    d = {"stack": "artifacts/stack_primary_m3px27_copy.json", "channels": ["v1r", M.CH_COPY], "inputs_sha256": {"oof_m3px27xmc.npy": "x"},
         "models": {"main": {"features": [f"{M.CH_COPY}_logp", "v1r_logp"], "w": [1.0, 2.0]}}, "guard_build": {"oof": "a", "rule": "band"}}
    r = {"stack": "artifacts/stack_s032_stage2.json", "channels": ["v1r", M.SLOT], "inputs_sha256": {"oof_pr27xm.npy": "y"},
         "models": {"main": {"features": [f"{M.SLOT}_logp", "v1r_logp"], "w": [1.0, 2.0]}}, "guard_build": {"oof": "b", "rule": "band"}}
    assert M.diff_keys(M.normalized(d, M.CH_COPY), M.normalized(r, M.SLOT)) == []
    r2 = copy.deepcopy(r)
    r2["models"]["main"]["w"][1] = 2.5
    r2["guard_build"]["rule"] = "widened"
    assert M.diff_keys(M.normalized(d, M.CH_COPY), M.normalized(r2, M.SLOT)) == [".models.main.w", ".guard_build.rule"]


def test_log_line_pattern():
    # log line of vlm/runner.py; the recorded runs used an earlier German wording of the line, which is accepted as well
    for line in ("[w0] 01:03:21 sft_from: notes for 3498 rows of table test, of which without a finding 3498 (findings are generated before scoring)",
                 "[w0] 01:03:21 sft_from: Ausgaben zu 3498 Zeilen der Tabelle test, davon ohne Befund 3498 (werden vor der Bewertung erzeugt)"):
        m = M.LOG_LINE.search(line)
        assert m and [v for v in m.groups() if v is not None] == ["3498", "test", "3498"] and any(p in line for p in M.LOG_PREFIXES)


# ------------------------------------------------------------------------------------------------ real training artifacts

def test_mean_channel_is_the_rule_matrix():
    f = {nm: ART / f"oof_{nm}.npy" for nm in ("m3px27xm", "m3px27", "pr27", "pr27x", "pr27xm")}
    _need(*f.values(), ART / "meta_m3px27xm.json", ART / "meta_m3px27.json")
    P, R, pr27, px27 = (np.load(f[nm]) for nm in ("m3px27xm", "m3px27", "pr27", "pr27x"))
    assert np.array_equal(P, 0.5 * (pr27 + R))  # matrix of the expected-value rule computation (c132: ½(oof_pr27 + m3px27))
    assert np.array_equal(np.load(f["pr27xm"]), 0.5 * (pr27 + px27))  # copy null point: the same path with px27 = final primary form
    rm = json.loads((ART / "meta_m3px27xm.json").read_text())["rank_mean"]
    assert rm["rank_compare"] == "sft" and rm["floor"] == 0.05 and rm["proto_sources"] == [M.PROTO_TRAIN]
    assert rm["rank_source"] == M.c132().SOURCE == json.loads((ART / "meta_m3px27.json").read_text())["rank"]["source"]
    assert rm["notes"]["train"]["sft_from"] == M.c132().NOTES_REL and rm["notes"]["test"]["sft_from"] == M.c132().TEST_NOTES_REL
    assert rm["notes"]["test"]["source_runs"] == [M.PROTO_TEST] and rm["notes"]["test"]["px27_sha256"] == M.mp().PX27_NOTES["test"][1]
    assert rm["rank_artifact"]["values_sha256"] == M.npy_sha256(R)


def test_stack_and_stage2_against_primary_final():
    files = [ART / f"stack_{M.STEM}.json", ART / f"stack_{M.STEM_REF}.json", M.STAGE2, ART / "stage2_primary_final.json", ART / f"meta_{M.CH}.json"]
    _need(*files)
    stack, ref, s2, pf = (json.loads(Path(p).read_text()) for p in files[:4])
    ren = lambda s: s.replace(f"{M.SLOT}_", f"{M.CH}_", 1) if s.startswith(f"{M.SLOT}_") else (M.CH if s == M.SLOT else s)  # noqa: E731
    assert stack["channels"] == [ren(c) for c in ref["channels"]] and stack["floor"] == {ren(k): v for k, v in ref["floor"].items()}
    for k in ("rank_only", "last_step", "rows", "folds", "fit_rows_sha256", "scaled", "none", "probes"):
        assert stack[k] == ref[k], k
    assert stack["arith_beta"] in M.c110().BETA_GRID and stack["replaces"]["by"] == M.CH
    assert s2["stack_sha256"] == _sha(files[0]) and s2["channels"] == stack["channels"] and s2["arith_beta"] == stack["arith_beta"]
    assert list(s2) == [*pf, "m3px27_build"]
    for fs in ("main", "fallback"):
        assert s2["models"][fs]["features"] == [ren(x) for x in pf["models"][fs]["features"]]
    m = s2["models"]["main"]
    assert m["feature_set"] == "main_guard_i1_mm" and m["protect_none"] is True and len(m["w"]) == 64 and not m["abstain"]
    b = s2["m3px27_build"]
    assert b["kind"] == "deploy" and b["rule_rows"]["bit_identical"] and all(v["ok"] for v in b["null_point_i"].values())
    assert b["deploy_forms"]["beta_per_fold"] == b["beta_outer"] and b["deploy_forms"]["beta_stack"] == stack["arith_beta"]
    assert s2["guard_rule"] == pf["guard_rule"] and s2["mech27"] == pf["mech27"]
    assert MS.rank_mean_guard(M.CH, stack, "runs/x/y/z", 1, art_dir=ART)["rank_compare"] == "sft"  # channel form matches the blend
    with pytest.raises(SystemExit, match="draw"):
        MS.rank_mean_guard(M.CH, stack, "runs/x/y/z", 2, art_dir=ART)


def test_oof_rank_shards_compare_clean_against_their_own_source():
    """The comparison form "sft" is empty for runs with the same settings (every shard against all five: only model and fold free)."""
    src = M.c132().SOURCE
    shards = src.split("+")
    _need(*(COMP / "artifacts" / "runs" / s.split("/")[1] / s.split("/")[1] / s.split("/")[2] / f"{s.split('/')[3]}_meta.json" for s in shards))
    assert all(MS.rank_only_source_differences(s, src) == {} for s in shards)


def test_test_guard_rejects_px27_test_rank():
    """Positive control of the test guard: the ranking of the untrained 27B (px27 test) is not the test side of m3px27 (exit 5)."""
    spec = "runs/n1_rp/rank_test/test"
    _need(ART / f"meta_{M.CH}.json", ART / "runs" / "n1_rp" / "n1_rp" / "rank_test" / "test_meta.json")
    code, lines, info = M.test_guard(spec)
    assert code == 5
    text = "\n".join(lines)
    assert "note guard violated" in text and "full fit" in text and info["coverage"]["complete"]
