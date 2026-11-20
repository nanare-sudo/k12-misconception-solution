"""tools/gpu_inference.sh --dry-run and its lists; tools/gpu_inference_inputs.py.

The dry run prints every command and runs none. These tests check that every configuration it names exists, that the
frozen texts it copies are exactly the ones the rescore configurations read, that it copies back every job folder the
two final calls read, and that the two final calls are the same as in reproduce.sh. With K12_PACKAGE set to the
unpacked private package (and the competition data in data/), the dry run must also find every input.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
SCRIPT = COMP / "tools" / "gpu_inference.sh"
spec = importlib.util.spec_from_file_location("gpu_inference_inputs_t", COMP / "tools" / "gpu_inference_inputs.py")
GI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GI)


def _array(path: Path, name: str) -> list[str]:
    """Elements of the bash array `name=( … )` in a script (the text between the brackets, split like the shell)."""
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^{name}=\((.*?)\)[ \t]*(?:#[^\n]*)?$", text, flags=re.S | re.M)
    assert m, f"{name} not found in {path.name}"
    body = "\n".join(line.split(" #")[0] for line in m.group(1).splitlines())
    return shlex.split(body)


def _fake_package(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    for d in ("artifacts/runs", "weights", "data/derived"):
        (pkg / d).mkdir(parents=True)
    (pkg / "MANIFEST.sha256").write_text("")
    return pkg


def _dry_run(mode: str, pkg: Path, out: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "K12_MODELS"}
    return subprocess.run(["bash", str(SCRIPT), mode, str(pkg), "--dry-run", "--out", str(out)], capture_output=True, text=True,
                          env=env, timeout=600)


def _configs(stdout: str) -> list[str]:
    return re.findall(r"^\+ python -u -m vlm\.runner --config (\S+) --out ", stdout, flags=re.M)


@pytest.mark.parametrize("mode", ["rescore", "full"])
def test_dry_run_names_existing_configurations_and_writes_nothing(tmp_path, mode):
    out = tmp_path / "out"
    res = _dry_run(mode, _fake_package(tmp_path), out)
    assert "dry run finished" in res.stdout, res.stdout[-2000:] + res.stderr[-2000:]
    cfgs = _configs(res.stdout)
    want = _array(SCRIPT, "GPU_STEPS_RESCORE") if mode == "rescore" else _array(SCRIPT, "GPU_STEPS_FULL") + _array(SCRIPT, "GPU_STEPS_FULL_AFTER_NOTES")
    assert [Path(c).relative_to(COMP / "configs" / "vlm").with_suffix("").as_posix() for c in cfgs] == want
    assert all(Path(c).is_file() for c in cfgs)
    assert not out.exists()


def test_frozen_texts_are_the_ones_the_rescore_configurations_read(tmp_path):
    ck = GI.Checker(_fake_package(tmp_path), None, False)
    for name in _array(SCRIPT, "GPU_STEPS_RESCORE"):
        ck.config(name)
    assert ck.frozen == set(_array(SCRIPT, "FROZEN_TEXTS"))
    full = GI.Checker(_fake_package(tmp_path / "f"), None, False)
    for name in _array(SCRIPT, "GPU_STEPS_FULL") + _array(SCRIPT, "GPU_STEPS_FULL_AFTER_NOTES"):
        full.config(name)
    assert full.frozen == set()  # mode full generates every text itself


def _read_by_final_calls() -> set[str]:
    """<run>/<job> of every test-side output that the two final make_submission calls read."""
    jobs = set()
    for arg in _array(SCRIPT, "B_ARGS") + _array(SCRIPT, "A_ARGS"):
        for part in re.split(r"[=,+]", arg):
            part = part.split("@")[0]
            bits = part.split("/")
            if bits[0] == "runs" and len(bits) == 4:
                jobs.add(f"{bits[1]}/{bits[2]}")
            elif len(bits) in (3, 4) and not part.startswith(("derived/", "artifacts/")):
                jobs.add(f"{bits[0]}/{bits[1]}")
    return jobs


def test_every_output_of_the_final_calls_is_copied_back():
    read = _read_by_final_calls()
    assert {"pr27te_rp/proto_te", "v1rot_t4/rot_test", "m3hte_b_rp/m3h_test_rank", "q397m3te_rp/q397m3_test_rank"} <= read
    rescore = set(_array(SCRIPT, "PUT_RESCORE"))
    full = set(_array(SCRIPT, "PUT_FULL")) | set(_array(SCRIPT, "PUT_FULL_AFTER_NOTES"))
    assert read - rescore == {"pr27te_rp/proto_te"}  # rescore keeps the frozen protocol answers
    assert read <= full


def test_final_calls_are_the_same_as_in_reproduce_sh():
    for name in ("A_ARGS", "B_ARGS"):
        assert _array(SCRIPT, name) == _array(COMP / "reproduce.sh", name)


def test_inputs_checker_reports_missing_files(tmp_path, capsys):
    pkg = _fake_package(tmp_path)
    assert GI.main(["--package", str(pkg), "rescore/m3d2te_rp", "no_such_configuration"]) == 1
    out = capsys.readouterr().out
    assert "no_such_configuration.yaml (configuration)" in out
    assert "m3_full/adapter/adapter_config.json" in out and "m3d2_test_gen/gen_notes.jsonl" in out


PKG = os.environ.get("K12_PACKAGE")


@pytest.mark.skipif(not (PKG and Path(PKG).is_dir() and (COMP / "data" / "competition_data" / "images" / "test").is_dir()),
                    reason="K12_PACKAGE not set, or the competition data with photos is not in data/")
@pytest.mark.parametrize("mode", ["rescore", "full"])
def test_dry_run_finds_every_input_of_the_package(tmp_path, mode):
    res = _dry_run(mode, Path(PKG), tmp_path / "out")
    assert res.returncode == 0 and "inputs complete" in res.stdout, res.stdout[-3000:]
