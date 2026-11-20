#!/usr/bin/env python3
"""Write the test-side run configurations of the GPU modes `full` and `rescore`.

    python -I tools/make_inference_configs.py            # write configs/vlm/test_side/ and configs/vlm/rescore/
    python -I tools/make_inference_configs.py --check    # compare the files in the repository with a new build

The model runner (src/vlm/runner.py) runs every job of a configuration file, one after another. Some reference
configurations also train a model or score training rows. The two folders hold copies of these files that keep only
the jobs the GPU modes need:

    configs/vlm/test_side/   mode `full`: the test-side jobs, text generation included. No training job and no job on
                             training rows, except short pre-check jobs.
    configs/vlm/rescore/     mode `rescore`: the scoring and ranking jobs only. They read the frozen texts that
                             tools/gpu_inference.sh copies from the private package into the output folder. No job
                             generates text.

Each kept job is copied unchanged. The run name, the model entries and the job names stay the same, so the outputs land
in the folders that the CPU tools read. tests/test_inference_configs.py checks every kept job against the job of the
same name in the reference file.

--check writes nothing. It exits with 1 if a file differs, is missing or is not made by this tool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

COMP = Path(__file__).resolve().parent.parent
CONFIGS = COMP / "configs" / "vlm"

# Jobs kept per folder and reference configuration, in the order of the reference file.
KEEP: dict[str, dict[str, list[str]]] = {
    "test_side": {
        "v1rot_t4": ["rot_test"],
        "zs27v_te_rp": ["gen_te", "think_te"],
        "zs27v_n3_rp": ["gen_nte", "think_nte"],
        "m3_fullte_rp": ["m3_merge_full", "m3_test_gen", "m3_test_rank"],
        "n1_rp": ["rank_smoke", "rank_test"],
    },
    "rescore": {
        "zs27v_te_rp": ["think_te"],
        "zs27v_n3_rp": ["think_nte"],
        "m3_fullte_rp": ["m3_merge_full", "m3_test_rank"],
        "m3d2te_rp": ["m3d2_merge", "m3d2_test_rank"],
        "m3d3te_rp": ["m3d3_merge", "m3d3_test_rank"],
        "m3d4te_rp": ["m3d4_merge", "m3d4_test_rank"],
        "m3hte_a_rp": ["m3h_merge", "m3h_test_rank"],
        "m3hte_b_rp": ["m3h_merge", "m3h_test_rank"],
    },
}
GENERATION_KINDS = ("think_gen", "proto_gen", "sft_gen", "rationale_gen")  # jobs that write texts with the vLLM server
MODE_OF = {"test_side": "full", "rescore": "rescore"}


class NoAliasDumper(yaml.SafeDumper):
    """Write repeated values in full instead of YAML anchors, so every job reads on its own."""

    def ignore_aliases(self, data) -> bool:  # noqa: ARG002 - signature of yaml.SafeDumper
        return True


def subset(folder: str, name: str) -> dict:
    """The reference configuration configs/vlm/<name>.yaml with only the jobs of KEEP[folder][name]."""
    want = KEEP[folder][name]
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    found = [j["name"] for w in cfg["workers"] for j in w["jobs"]]
    for job in want:
        if found.count(job) != 1:
            raise SystemExit(f"configs/vlm/{name}.yaml: job {job!r} found {found.count(job)} times (expected once)")
    for w in cfg["workers"]:
        w["jobs"] = [j for j in w["jobs"] if j["name"] in want]
    cfg["workers"] = [w for w in cfg["workers"] if w["jobs"]]
    kept = [j for w in cfg["workers"] for j in w["jobs"]]
    if [j["name"] for j in kept] != [n for n in found if n in want]:
        raise SystemExit(f"configs/vlm/{name}.yaml: kept jobs out of order")
    for j in kept:
        if j["kind"] == "train":
            raise SystemExit(f"configs/vlm/{name}.yaml: job {j['name']} trains a model and cannot be kept")
        if folder == "rescore" and j["kind"] in GENERATION_KINDS:
            raise SystemExit(f"configs/vlm/{name}.yaml: job {j['name']} generates text and cannot be kept for rescore")
    return cfg


def render(folder: str, name: str) -> str:
    jobs = ", ".join(KEEP[folder][name])
    head = (f"# Made by tools/make_inference_configs.py from configs/vlm/{name}.yaml. Do not edit by hand.\n"
            f"# Mode `{MODE_OF[folder]}`: the jobs {jobs}, copied unchanged under the same run and job names, so that\n"
            f"# the outputs land where the CPU tools read them (docs/VERIFICATION.md, section 5.2).\n")
    body = yaml.dump(subset(folder, name), Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True,
                     default_flow_style=False, width=120)
    return head + body


def build() -> dict[Path, str]:
    return {CONFIGS / folder / f"{name}.yaml": render(folder, name) for folder, files in KEEP.items() for name in files}


def extra_files(files: dict[Path, str]) -> list[Path]:
    """YAML files in the two folders that this tool does not make."""
    return sorted(p for folder in KEEP for p in (CONFIGS / folder).glob("*.yaml") if p not in files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the files in the repository with a new build; write nothing")
    a = ap.parse_args(argv)
    files = build()
    if a.check:
        bad = [f"{p.relative_to(COMP)}: missing" for p in files if not p.is_file()]
        bad += [f"{p.relative_to(COMP)}: differs from a new build" for p, text in files.items()
                if p.is_file() and p.read_text(encoding="utf-8") != text]
        bad += [f"{p.relative_to(COMP)}: not made by this tool" for p in extra_files(files)]
        for line in bad:
            print(line)
        print(f"{len(files)} files checked, {len(bad)} problems")
        return 1 if bad else 0
    for p in extra_files(files):
        p.unlink()
        print(f"removed {p.relative_to(COMP)} (not made by this tool)")
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"written {p.relative_to(COMP)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
