"""configs/vlm/test_side/ and configs/vlm/rescore/: test-side jobs of the reference configurations (tools/make_inference_configs.py).

Every kept job must equal the job of the same name in the reference file, so that the GPU modes compute exactly what the
reference runs computed and write to the same folders. No competition data needed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

COMP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_inference_configs_t", COMP / "tools" / "make_inference_configs.py")
MIC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MIC)

FILES = [(folder, name) for folder, files in MIC.KEEP.items() for name in files]


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _jobs(cfg: dict) -> list[dict]:
    return [j for w in cfg["workers"] for j in w["jobs"]]


@pytest.mark.parametrize("folder,name", FILES)
def test_kept_jobs_equal_the_reference(folder, name):
    cfg = _load(COMP / "configs" / "vlm" / folder / f"{name}.yaml")
    ref = _load(COMP / "configs" / "vlm" / f"{name}.yaml")
    assert {k: v for k, v in cfg.items() if k != "workers"} == {k: v for k, v in ref.items() if k != "workers"}
    ref_jobs = {j["name"]: j for j in _jobs(ref)}
    for w in cfg["workers"]:
        ref_w = next(rw for rw in ref["workers"] if any(j["name"] == w["jobs"][0]["name"] for j in rw["jobs"]))
        assert {k: v for k, v in w.items() if k != "jobs"} == {k: v for k, v in ref_w.items() if k != "jobs"}
        for j in w["jobs"]:
            assert j == ref_jobs[j["name"]], f"{folder}/{name}: job {j['name']} differs from the reference"
    assert [j["name"] for j in _jobs(cfg)] == MIC.KEEP[folder][name]
    assert not any(j["kind"] == "train" for j in _jobs(cfg))
    if folder == "rescore":
        assert not any(j["kind"] in MIC.GENERATION_KINDS for j in _jobs(cfg))


@pytest.mark.parametrize("folder,name", FILES)
def test_text_sources(folder, name):
    """rescore: every job that reads texts of this run reads them from a generation job of the reference that is NOT in the
    file (the frozen copy). test_side: the generation job is in the same file, so the run writes its own texts first."""
    cfg = _load(COMP / "configs" / "vlm" / folder / f"{name}.yaml")
    ref_kinds = {j["name"]: j["kind"] for j in _jobs(_load(COMP / "configs" / "vlm" / f"{name}.yaml"))}
    names = [j["name"] for j in _jobs(cfg)]
    for j in _jobs(cfg):
        for key in ("think_from", "sft_from"):
            src = str(j.get(key) or "")
            if not src or src.endswith(".jsonl") or src.count("/") != 1:
                continue  # no text input, or a frozen file under data/derived/
            job = src.split("/")[0]
            assert ref_kinds.get(job) in MIC.GENERATION_KINDS, f"{name}: {key} {src} is not a generation job of the reference"
            if folder == "rescore":
                assert job not in names, f"{name}: {j['name']} would read texts generated in the same run"
            else:
                assert job in names and names.index(job) < names.index(j["name"]), f"{name}: {j['name']} reads texts generated later or elsewhere"


def test_files_are_current():
    assert MIC.main(["--check"]) == 0


def test_folders_hold_only_made_files():
    for folder, files in MIC.KEEP.items():
        on_disk = sorted(p.stem for p in (COMP / "configs" / "vlm" / folder).glob("*.yaml"))
        assert on_disk == sorted(files), folder
