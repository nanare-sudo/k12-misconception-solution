"""Tests of tools/make_m3_targets.py (training targets of M3).

SYNTHETIC: record format and order of target_lines, the rule of pr27_ok (stable top code over the classes that can be a root, notes
required), the mapping of the adapter recipes to the configurations. With the private files (skipped if they are missing): the
check mode rebuilds the six target files byte-identically to the files present and to targets_sha256 of the adapter recipes.
No test row is read."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src")]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("make_m3_targets_test", "tools/make_m3_targets.py")


def test_target_lines_format_and_order():
    folds = np.array([0, 1, 2, 1])
    ok = np.array([True, False, True, False])
    notes = {0: ("n0", "s1"), 1: ("", ""), 2: ("n2 \u2028 x", "s2"), 3: ("n3", "s3")}
    findings = ["f0", "none", "f2", "f3"]
    codes = ["CAL-01", "NONE", "Dec-01", "ANS-01"]
    text = T.target_lines(np.array([0, 2, 3]), folds, ok, notes, findings, codes)
    lines = text.split("\n")
    assert lines[-1] == "" and len(lines) == 4
    recs = [json.loads(x) for x in lines[:-1]]
    assert [r["row"] for r in recs] == [0, 2, 3]
    assert list(recs[0]) == ["table", "row", "fold", "notes_loss", "notes_state", "notes", "finding", "code"]
    assert recs[1] == {"table": "train", "row": 2, "fold": 2, "notes_loss": True, "notes_state": "s2", "notes": "n2 \u2028 x",
                       "finding": "f2", "code": "Dec-01"}
    assert "\u2028" in text  # raw, not escaped (ensure_ascii=False), as in the frozen files


def test_pr27_ok_rule(tmp_path, monkeypatch):
    classes = ["NONE", "NOT-02", "CAL-01", "ANS-01"]
    P = np.array([[0.1, 0.6, 0.2, 0.1],    # NOT-02 can never be a root -> top CAL-01
                  [0.5, 0.0, 0.5, 0.0],    # tie NONE/CAL-01 -> stable: NONE
                  [0.1, 0.0, 0.2, 0.7],
                  [0.1, 0.0, 0.2, 0.7]])
    art = tmp_path / "artifacts"
    art.mkdir()
    np.save(art / "oof_pr27.npy", P)
    monkeypatch.setattr(T, "COMP", tmp_path)
    y = np.array([2, 0, 3, 3])
    have = np.array([True, True, True, False])
    assert T.pr27_ok(classes, y, have).tolist() == [True, True, True, False]


def test_recipes_match_configurations():
    import yaml

    seen = {}
    for cfg in ("m3_pilot_rp", "m3_f034_rp", "m3_fullte_rp"):
        y = yaml.safe_load((COMP / "configs" / "vlm" / f"{cfg}.yaml").read_text())
        for job in (j for w in y["workers"] for j in w["jobs"]):
            t = (job.get("train") or {}).get("sft_targets") or job.get("sft_targets")
            if t and job.get("kind") == "train" and "smoke" not in job["name"]:
                seen[job["name"]] = (cfg, Path(t).name)
    for k, rel in T.RECIPES.items():
        run, _, job = rel.split("/")
        want = "m3_full_targets.jsonl" if k is None else f"m3_fold{k}_targets.jsonl"
        assert seen.get(job) == (run, want), (k, rel, seen.get(job))


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(f"private files missing: {missing}")


def test_check_mode_rebuilds_the_frozen_targets():
    sft = COMP / "data" / "derived" / "sft"
    _need(T.MANIFEST, COMP / "artifacts" / "oof_pr27.npy", COMP / "data" / "derived" / "proto_notes" / "train_pr27.jsonl",
          COMP / "data" / "train.csv", *(sft / f"m3_fold{k}_targets.jsonl" for k in range(5)), sft / "m3_full_targets.jsonl",
          *(COMP / "artifacts" / "runs" / r / "sft_recipe.json" for r in T.RECIPES.values()))
    assert hashlib.sha256(T.MANIFEST.read_bytes()).hexdigest() == T.MANIFEST_SHA
    r = subprocess.run([sys.executable, "-I", str(COMP / "tools" / "make_m3_targets.py"), "--full", "--check"], capture_output=True, text=True,
                       cwd=COMP)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.count("equal; file present: equal") == 6
