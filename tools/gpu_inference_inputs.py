#!/usr/bin/env python3
"""Check the inputs of the GPU modes before a run: configurations, competition data, derived files, frozen texts, weights.

    python -I tools/gpu_inference_inputs.py --package <package-dir> [--models DIR] [--strict-runtime] <config> ...

tools/gpu_inference.sh calls this tool before it runs anything, and in a dry run. <config> is a file under configs/vlm/
without .yaml, for example rescore/m3d2te_rp. For every job of every configuration the tool checks:
  - derived files named in the job (row lists, rotation lists, note files) under data/derived/ of this repository, or
    under the package if data/derived/ is not set up yet;
  - teacher labels (data/derived/teacher/<plan>/labels.jsonl) and data/unlabelled.csv, which the jobs with teacher
    labels read;
  - adapters named by `adapter:` under <package-dir>/weights/ (the runner finds them there through K12_ADAPTERS);
  - the M3 adapter that the merge jobs read; tools/gpu_inference.sh copies it from <package-dir>/weights/m3_full/;
  - texts that a job reads from a generation job of another file: the frozen copy in
    <package-dir>/artifacts/runs/<run>/<run>/<job>/ (mode rescore);
  - merged models: a merge job of the same file must write them.
It also checks the competition data (CSV files and photo folders), the base models in --models (<DIR>/<name>/config.json)
and the vLLM environment of the jobs that generate text. Missing base models and a missing vLLM environment count as
missing only with --strict-runtime; a dry run lists them without counting them.

The tool prints one line per configuration and every missing file. It reads no file contents. Exit 0 if nothing is
missing, 1 otherwise, 2 for wrong arguments.
"""
from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

import yaml

COMP = Path(__file__).resolve().parent.parent
GENERATION_KINDS = ("think_gen", "proto_gen", "sft_gen", "rationale_gen")
M3_ADAPTER = "m3_fullte_rp/m3_train_full"  # where every M3 merge job reads the adapter (relative to the output folder)
M3_FILES = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "sft_recipe.json")
VLLM_PYTHON = "/workspace/vllm_env/bin/python"  # default of src/vlm/runner.ensure_vllm_env
PHOTOS = {"train": 1500, "test": 3498, "questions": 119}


def strings(x, key: str = ""):
    """(key, string) for every string value of a nested configuration."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from strings(v, str(k))
    elif isinstance(x, list):
        for v in x:
            yield from strings(v, key)
    elif isinstance(x, str):
        yield key, x


class Checker:
    def __init__(self, pkg: Path, models: Path | None, strict_runtime: bool):
        self.pkg, self.models, self.strict = pkg, models, strict_runtime
        repo_derived = COMP / "data" / "derived"
        self.derived = repo_derived if repo_derived.is_dir() else pkg / "data" / "derived"
        self.missing: list[str] = []
        self.base: dict[str, list[str]] = {}  # base model -> configurations that need it
        self.vllm: dict[str, list[str]] = {}  # vLLM environment -> configurations that need it
        self.frozen: set[str] = set()

    def need(self, path: Path, why: str) -> bool:
        if path.exists():
            return True
        if not any(m.startswith(f"{path} (") for m in self.missing):  # each missing file once
            self.missing.append(f"{path} ({why})")
        return False

    def derived_file(self, rel: str, why: str) -> None:
        self.need(self.derived / rel[len("derived/"):], why)

    def adapter(self, name: str, why: str) -> None:
        weights = self.pkg / "weights"
        hits = list(weights.glob(f"**/{name}/adapter_config.json")) + list(weights.glob(f"**/{name}/adapter/adapter_config.json"))
        if not hits:
            self.missing.append(f"{weights}/**/{name}/adapter_config.json ({why})")

    def config(self, name: str) -> tuple[int, int]:
        """Check one configuration -> (jobs, missing before + after)."""
        before = len(self.missing)
        f = COMP / "configs" / "vlm" / f"{name}.yaml"
        if not self.need(f, "configuration"):
            return 0, len(self.missing) - before
        cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
        run = cfg["run"]
        jobs = [j for w in cfg["workers"] for j in w["jobs"]]
        names = {j["name"] for j in jobs}
        merged_by_job = {j.get("out_model") for j in jobs if j["kind"] == "merge"}
        for spec_name, spec in cfg["models"].items():
            if spec.get("merged"):
                if spec["merged"] not in merged_by_job:
                    self.missing.append(f"merged model {spec['merged']} (model {spec_name} of {name}): no merge job in this file writes it")
            elif spec.get("repo"):
                self.base.setdefault(spec["repo"].split("/")[-1], []).append(name)
        for j in jobs:
            why = f"{name}, job {j['name']}"
            for key, value in strings(j):
                for part in re.split(r"[@,+\s]", value):
                    if part.startswith("derived/"):
                        self.derived_file(part, f"{why}, {key}")
            for plan in j.get("teacher_labels") or []:
                self.derived_file(f"derived/teacher/{plan}/labels.jsonl", f"{why}, teacher_labels")
                self.need(COMP / "data" / "unlabelled.csv", f"{why}, teacher_labels")
            if j.get("adapter"):
                self.adapter(str(j["adapter"]), f"{why}, adapter")
            if j["kind"] == "merge":
                src = posixpath.normpath(posixpath.join(run, str(j["adapter_from"])))
                if src == M3_ADAPTER:
                    for rel in M3_FILES:
                        self.need(self.pkg / "weights" / "m3_full" / rel, f"{why}, adapter_from (copied to <out>/{M3_ADAPTER}/)")
                else:
                    self.missing.append(f"<out>/{src} ({why}, adapter_from): not provided by tools/gpu_inference.sh")
            for key, suffix in (("think_from", "_texts.jsonl"), ("sft_from", "_notes.jsonl")):
                ref = str(j.get(key) or "")
                if not ref or ref.endswith(".jsonl"):
                    continue
                bits = ref.split("/")
                src_run, src_job, tag = (run, bits[0], bits[1]) if len(bits) == 2 else (bits[0], bits[1], bits[2])
                if src_run == run and src_job in names:
                    continue  # written by an earlier job of the same file
                self.frozen.add(f"{src_run}/{src_job}")
                self.need(self.pkg / "artifacts" / "runs" / src_run / src_run / src_job / f"{tag}{suffix}", f"{why}, {key} (frozen copy)")
            if j["kind"] in GENERATION_KINDS:
                env = str(Path((j.get("server") or {}).get("python", VLLM_PYTHON)).parents[1])
                if name not in self.vllm.setdefault(env, []):
                    self.vllm[env].append(name)
        return len(jobs), len(self.missing) - before

    def runtime(self) -> list[tuple[bool, str]]:
        """(present, text) for every base model and vLLM environment that the configurations need."""
        out = []
        for model, cfgs in self.base.items():
            k = len(dict.fromkeys(cfgs))
            uses = f"needed by {k} configuration{'' if k == 1 else 's'}"
            if self.models is None:
                out.append((False, f"base model {model}: K12_MODELS is not set ({uses})"))
            else:
                f = self.models / model / "config.json"
                out.append((f.is_file(), f"base model {f.parent} ({uses})"))
        for env, cfgs in self.vllm.items():
            ok = (Path(env) / "bin" / "python").is_file() and (Path(env) / ".k12_ready").is_file()
            out.append((ok, f"vLLM environment {env} with the marker .k12_ready (needed by {', '.join(cfgs)})"))
        return out

    def competition_data(self) -> str:
        data = COMP / "data"
        for f in ("train.csv", "test.csv", "competition_data/taxonomy_v1.csv", "competition_data/sample_submission.csv"):
            self.need(data / f, "competition data")
        counts = []
        for table, n in PHOTOS.items():
            d = data / "competition_data" / "images" / table
            have = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
            counts.append(f"{table} {have}")
            if have < n:
                self.missing.append(f"{d} ({have} of {n} photos)")
        return ", ".join(counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--package", required=True, help="the unpacked private package")
    ap.add_argument("--models", help="folder of the base models (K12_MODELS)")
    ap.add_argument("--strict-runtime", action="store_true", help="count missing base models and a missing vLLM environment")
    ap.add_argument("configs", nargs="+", help="configurations under configs/vlm/, without .yaml")
    a = ap.parse_args(argv)
    ck = Checker(Path(a.package), Path(a.models) if a.models else None, a.strict_runtime)
    print(f"inputs: derived files from {ck.derived}")
    for name in a.configs:
        jobs, miss = ck.config(name)
        print(f"  configs/vlm/{name}.yaml: {jobs} job{'' if jobs == 1 else 's'}, " + ("inputs present" if not miss else f"MISSING {miss}"))
    photos = ck.competition_data()
    print(f"  competition data: photos {photos}")
    if ck.frozen:
        print(f"  frozen texts read from the package: {', '.join(sorted(ck.frozen))}")
    for item in ck.missing:
        print(f"    MISSING {item}")
    runtime = ck.runtime()
    for ok, item in runtime:
        print(f"    {'present' if ok else 'MISSING' if ck.strict else 'not counted in a dry run, missing:'} {item}")
    n = len(ck.missing) + (sum(not ok for ok, _ in runtime) if ck.strict else 0)
    print(f"missing: {n}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
