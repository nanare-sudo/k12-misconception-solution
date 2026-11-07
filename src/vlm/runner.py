"""Runs of the vision-language models: measurement without training, fine-tuning per fold, prediction.

    python -m vlm.runner --config configs/vlm/<run>.yaml --out <folder> [--worker N]

Without --worker the call starts one subprocess with its own GPU per entry under `workers` and waits; with --worker N
it processes the jobs of this entry one after another.

Job kinds (`kind`, selection):
    bench    probes on labelled training rows, without training (baseline, resolution, model choice)
    train    LoRA fine-tuning on the folds != k, then probes on fold k (OOF) and optionally prediction
    eval     probes on fold k with a stored adapter (examples only from the other folds)
    predict  probes on test or pool rows with a stored adapter (never on training rows)
Further kinds: think_gen, proto_gen, sft_gen, rationale_gen (text generation with a local vLLM server) and merge
(merge a LoRA adapter into the base weights).

Probes (`probes`):
    fac        "final answer correct?" directly from the image (scored yes/no)
    ans        copy the student's final answer, then score "correct?" behind the own text
    diag       code ranking directly from the image (decision first)
    diag_gold  code ranking from the cleaned gold transcription, without image (training rows only)
    diag_goldimg  code ranking from image AND cleaned gold transcription (training rows only); control "reading versus
               diagnosing": the same diagnostician and rows, only the transcription is added
    read       generate a transcription (text)
    diag_read  code ranking from image and own transcription (needs `read`)
    full       read and generate findings first, then code ranking behind the own text
    cot        like full, but every finding gives the reasoning first, then the code
    think      reason in the model's thinking mode, then code ranking behind the own text (useful only without adapter)
    why        generate answer check and description of the root error, then code ranking behind the own description
    why_no     like why, but with a forced "Final answer correct: no" (ranking of the error codes without committing to NONE)
    why_fac    the answer check at the first position of the why answer format (scored yes/no)
    proto_rank code ranking behind the notes of the teacher-protocol channel (`proto_from`) or of the fine-tuned
               diagnostician (`sft_from`)

Row-level outputs are written only to files. For test rows only aggregates are logged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Header line of a checked transcription in the prompt (control "reading versus diagnosing")
GOLD_READING_HEAD = "Transcription of this working, written and checked by a marker:"

HERE = Path(__file__).resolve()
for p in (HERE.parents[1], HERE.parents[2]):  # src/ and the root with kaggle_core/
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


YES_NO_PROBES = ("fac", "ans", "why_fac", "up")  # probes with two columns (yes, no) instead of the code ranking


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------- Obtain the model


def fetch_model(spec: dict, models_dir: Path) -> str:
    """Local path of the model; downloads the pinned revision from the Hugging Face Hub if needed.

    `merged: <name>`: a model merged in this run (job merge, vlm/sft.py) under models_dir/<name>;
    it is never downloaded and must exist."""
    if spec.get("merged"):
        local = Path(models_dir) / str(spec["merged"])
        if (local / "config.json").is_file():
            return str(local)
        raise FileNotFoundError(f"merged model {spec['merged']} missing under {models_dir} (run the merge job first)")
    if spec.get("path") and Path(spec["path"]).is_dir():
        return spec["path"]
    name = spec["repo"].split("/")[-1]
    local = models_dir / name
    if (local / "config.json").is_file():
        return str(local)
    # offline run: weights are attached as the output of a private notebook (several models, several roots)
    for hit in sorted(Path("/kaggle/input").glob(f"**/models/{name}/config.json")) if Path("/kaggle/input").is_dir() else []:
        return str(hit.parent)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(f"weights for {name} not found among the attached inputs (offline run)")
    from huggingface_hub import snapshot_download

    t = time.time()
    # large models (397B: 406 GB) with more parallel connections; default unchanged (8)
    snapshot_download(spec["repo"], revision=spec.get("revision"), local_dir=str(local), max_workers=int(spec.get("download_workers", 8)))
    size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 2**30
    log(f"model {spec['repo']}@{spec.get('revision')} loaded: {size:.1f} GB in {time.time() - t:.0f} s")
    return str(local)


def find_adapter(name: str | None) -> str | None:
    """Path of a stored adapter: as given, otherwise searched by folder name under the usual roots."""
    if not name:
        return None
    if (Path(name) / "adapter_config.json").is_file():
        return name
    for root in ("/kaggle/input", "/workspace/adapters", "/workspace/out", os.environ.get("K12_ADAPTERS", "")):
        if root and Path(root).is_dir():
            hits = sorted(Path(root).glob(f"**/{name}/adapter_config.json")) + sorted(Path(root).glob(f"**/{name}/adapter/adapter_config.json"))
            if hits:
                return str(hits[0].parent)
    raise FileNotFoundError(f"adapter {name!r} not found")


def make_engine(model_spec: dict, models_dir: Path, device: str, adapter: str | None = None):
    from vlm.engine import Engine, EngineConfig

    adapter = find_adapter(adapter)
    if adapter:
        log(f"Adapter: {adapter}")

    path = fetch_model(model_spec, models_dir)
    t = time.time()
    device = model_spec.get("device", device)  # "auto" spreads a large model over all visible GPUs
    eng = Engine(EngineConfig(model_path=path, dtype=model_spec.get("dtype", "fp16"), quant=model_spec.get("quant"),
                              device=device, adapter=adapter, attn=model_spec.get("attn"),
                              max_beam_tokens=int(model_spec.get("max_beam_tokens", 36000)),
                              max_memory=model_spec.get("max_memory"), prefix_cache=bool(model_spec.get("prefix_cache", True)),
                              head_with_adapter=bool(model_spec.get("head_with_adapter", False))))
    if getattr(eng.cfg, "head_with_adapter", False) and not eng.has_adapter():
        raise ValueError("head_with_adapter: only useful with an adapter (adapter: <name>)")
    log(f"model ready in {time.time() - t:.0f} s ({model_spec.get('dtype', 'fp16')}, quant={model_spec.get('quant')}, device {device})")
    return eng


# ---------------------------------------------------------------------------- Row selection


def ensure_vllm_env(job: dict, out_dir: Path) -> None:
    """Set up a separate Python environment for the local vLLM server (once per machine) and record it in
    job["server"]["python"]: vLLM brings other package versions than the in-house compute path."""
    spec = job.setdefault("server", {})
    py = Path(spec.get("python", "/workspace/vllm_env/bin/python"))
    spec["python"] = str(py)
    ready = py.parents[1] / ".k12_ready"  # written only after a complete setup: an aborted attempt does not count
    if py.is_file() and ready.is_file():
        return
    import shutil

    t_inst = time.time()
    shutil.rmtree(py.parents[1], ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "uv"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "venv", str(py.parents[1]), "--python", sys.executable], check=True)
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", str(py), *str(spec.get("pip", "vllm")).split()], check=True)
    frozen = subprocess.run([sys.executable, "-m", "uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True).stdout
    (out_dir / "pip_freeze_vllm.txt").write_text(frozen)
    ready.write_text(frozen)
    log(f"vLLM environment set up in {time.time() - t_inst:.0f} s: " + ", ".join(l for l in frozen.splitlines() if l.split("==")[0] in ("vllm", "torch", "transformers")))


def find_file(rel, data_root=None) -> Path:
    """Find a derived file: absolute, in the data folder, in the competition folder (locally data/… or artifacts/…) or in
    an attached private Kaggle dataset (/kaggle/input/**/<rel>); on Kaggle the competition data folder is read-only,
    derived files come from a separate dataset there."""
    from vlm import common as C

    f = Path(rel)
    if f.is_absolute():
        cands = [f]
    else:
        cands = [Path(data_root or C.find_data_root()) / f, C.comp_root() / "data" / f, C.comp_root() / f]
        cands += [Path(h) for h in sorted(glob.glob(f"/kaggle/input/**/{rel}", recursive=True))]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"{rel}: not found (checked: {[str(c) for c in cands[:3]]} and /kaggle/input)")


def load_analyses(spec, data_root, split: str = "train") -> dict[str, dict[int, list[str]]]:
    """`analysis_from`: reasoning texts of another model per table and table position, as a list per row.

    spec: list of files (applies to `split`) or {table: [files]}. Every file is the output of a think_gen job
    (`<tag>_texts.jsonl`: {"row": position, "text": raw text}). Several files = several draws: the probes take the
    first one that covers a row; training draws one of the available ones per visit."""
    from vlm import think_vllm as TV

    by_split = spec if isinstance(spec, dict) else {split: list(spec)}
    out: dict[str, dict[int, list[str]]] = {}
    for sp, files in by_split.items():
        rows: dict[int, list[str]] = {}
        for f in files:
            texts = TV.load_texts(find_file(f, data_root))
            for i, tx in texts.items():
                if tx.strip():
                    rows.setdefault(int(i), []).append(tx)
            log(f"analysis_from [{sp}]: {len(texts)} notes from {f}")
        out[sp] = rows
    return out


def only_rows(idx: np.ndarray, sel: dict, data_root=None) -> np.ndarray:
    """`only: <file.npy>`: restrict the row selection to the table positions of this file (path relative to the data folder).

    For cascades: an expensive job computes only the rows that a cheap channel has not already decided
    (a gate list as written by tools/replay/gates.py). It contains positions in the WHOLE table, no contents; it acts before
    `sample`, `limit` and `shard`.
    """
    if not sel.get("only"):
        return idx
    return idx[np.isin(idx, np.load(find_file(sel["only"], data_root)))]


def select_rows(sel: dict, n: int, folds: np.ndarray | None, whole_table: bool = True) -> np.ndarray:
    idx = np.arange(n)
    if sel.get("only") and not whole_table:
        raise ValueError("rows.only refers to positions in the whole table and is not allowed for this row selection (subset)")
    if sel.get("folds") is not None:
        if folds is None:
            raise ValueError("row selection by folds requested, but there are no folds here")
        idx = idx[np.isin(folds, sel["folds"])]
    idx = only_rows(idx, sel)
    if sel.get("sample"):
        rng = np.random.default_rng(sel.get("seed", 0))
        idx = np.sort(rng.choice(idx, size=min(int(sel["sample"]), len(idx)), replace=False))
    if sel.get("limit"):
        idx = idx[: int(sel["limit"])]
    return idx


# ---------------------------------------------------------------------------- Probes

# Question knowledge that a job with `proto_from` (probe proto_rank) must not carry: the teacher-protocol channel sees
# no labelled row (vlm/proto.py); the same list as for the proto_gen job, plus the transcription.
PROTO_NO_TASK_KEYS = ("teacher_labels", "topic_codes", "question_codes", "tax_cases", "tax_cases_for", "candidates", "analysis_from",
                      "reading_from", "gold_reading")


def proto_rows(runner, idx: np.ndarray, split: str, probes: list[str]) -> np.ndarray:
    """Row selection of the probe proto_rank: only rows of the right table for which notes exist (the others keep the
    channel from the named codes in the blend). The probe stands alone in its job; any other probe would bring
    examples of the same question into the same job."""
    if "proto_rank" not in probes:
        return idx
    if list(probes) != ["proto_rank"]:
        raise ValueError("The probe proto_rank must be the only probe in its job (no question knowledge next to it)")
    if runner.proto_notes is None:
        raise ValueError("probe proto_rank requires `proto_from` (notes of the protocol channel)")
    if runner.proto_table != split:
        raise ValueError(f"proto_from: notes of table {runner.proto_table}, but the computation runs on {split}")
    keep = np.isin(idx, np.fromiter(runner.proto_notes, dtype=np.int64, count=len(runner.proto_notes)))
    if not keep.all():
        log(f"proto_from: {int((~keep).sum())} of {len(idx)} selected rows without notes, skipped")
    return idx[keep]


class ProbeRunner:
    def __init__(self, engine, builder, classes: list[str], allowed: list[str], job: dict):
        from vlm import prompts as P
        from vlm.engine import CandidateSet

        self.P = P
        self.engine, self.builder, self.job = engine, builder, job
        self.classes, self.allowed = classes, allowed
        self.codes = CandidateSet(engine.tok, [" " + c for c in allowed])
        self.yn = CandidateSet(engine.tok, [P.YES, P.NO])
        self.col = np.array([classes.index(c) for c in allowed])
        self.batch = int(job.get("batch", 2))
        self.beams = int(job.get("beams", 12))
        self.exact = int(job.get("exact_chunk", 0))  # > 0: all codes exactly, in groups of this size (instead of the best prefixes)
        self.first_top = int(job.get("first_top", 0))  # > 0: continuations only behind the most probable first tokens
        ids = lambda t: engine.tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731 - tokenisation as in the training target (per segment)
        self.why_forced = ids(P.NO) + ids("\nError:")
        self.root_ids = ids("\nRoot:")
        self.deadline = float(os.environ.get("K12_DEADLINE", 0) or 0)  # Unix time from which probes end with a partial result
        # Reasoning texts of the probe think from a file instead of own generation (job think_gen, vlm/think_vllm.py):
        # table position -> raw text; set by the job via `think_from`.
        self.think_texts: dict[int, str] | None = None
        # Notes of another model for the probe diag (`analysis_from`): {table: {position: [texts]}}; the probe takes the
        # first note of the row. `analysis_image: false` computes without the photo (text-only variant).
        self.analysis: dict[str, dict[int, list[str]]] | None = None
        # Machine transcription IN ADDITION to the photo for the tasks diag and think (`reading_from`): table position ->
        # text; training rows only (measures what an external transcription would add).
        self.readings: dict[int, str] | None = None
        self.analysis_image = bool(job.get("analysis_image", True))
        # Notes of the protocol channel for the probe proto_rank (`proto_from`): table, table position -> text, plus the
        # system text of the ranking (verdict rules and taxonomy; vlm/proto.py). Set by the job.
        self.proto_table: str | None = None
        self.proto_notes: dict[int, str] | None = None
        self.proto_system: str | None = None
        # sft mode of the probe proto_rank (`sft_from`, vlm/sft.py): finding per row (None = generate before scoring);
        # None = ordinary ranking behind the notes of the protocol channel.
        self.sft_findings: dict[int, str | None] | None = None
        # Examples variant (`sft_examples`, vlm/sft.py): example source (fold from the model's recipe) and checksum of the
        # example text per row from the generation; None = without examples (M3).
        self.sft_examples = None
        self.sft_example_hashes: dict[int, str | None] | None = None

    def _batches(self, order: np.ndarray):
        for s in range(0, len(order), self.batch):
            yield order[s : s + self.batch]

    def _build(self, df, i, split, task, fold_of, **kw):
        if self.analysis is not None and task == "diag" and "analysis" not in kw:
            texts = self.analysis.get(split, {}).get(int(i))
            if not texts:
                raise KeyError(f"analysis_from: a row of table {split} has no note (restrict the selection with rows.only)")
            kw["analysis"] = texts[0]
            kw.setdefault("with_image", self.analysis_image)
            kw.setdefault("analysis_head", str(self.job.get("analysis_head", "")))  # own header line of the note (prompts.build_sample)
        if self.readings is not None and task in ("diag", "think") and "reading" not in kw:
            if split != "train":
                raise ValueError("reading_from exists only for training rows")
            text = self.readings.get(int(i), "")
            if text.strip():  # header line: the generic one for machine transcriptions (prompts.build_sample) or `reading_head`
                kw["reading"] = text
                if self.job.get("reading_head"):
                    kw["reading_head"] = str(self.job["reading_head"])
        if self.job.get("gold_reading") and task in ("diag", "think") and "reading" not in kw:
            # checked transcription IN ADDITION to the photo (training rows only): measures what perfect reading would add
            if split != "train":
                raise ValueError("gold_reading exists only for training rows (gold transcription)")
            kw["reading"] = self.P.clean_transcription(df.iloc[i]["transcription_steps"])
            kw["reading_head"] = GOLD_READING_HEAD
        return self.builder.build(
            df.iloc[i], split=split, task=task, exclude_fold=None if fold_of is None else int(fold_of[i]),
            with_examples=bool(self.job.get("examples", True)), **kw,
        )

    def _build_proto(self, df, i, split):
        """Sample of the probe proto_rank: protocol prompt WITHOUT question knowledge (the builder's reference set stays
        untouched; only the data root and image settings come from it), followed by the row's own notes."""
        from vlm import proto as PR

        if self.proto_notes is None or self.proto_system is None:
            raise ValueError("probe proto_rank requires `proto_from` (notes of the protocol channel)")
        if self.proto_table != split:
            raise ValueError(f"proto_from: notes of table {self.proto_table}, but the computation runs on {split}")
        findings = getattr(self, "sft_findings", None)
        if findings is not None:  # sft mode: canonical text of the fine-tuned diagnostician (notes may be empty)
            from vlm import sft as S

            if int(i) not in self.proto_notes:
                raise KeyError("sft_from: a row of the batch has no output")
            row = df.iloc[i]
            student, qimg = PR.open_images(row, split, self.builder.root, self.builder.image)
            if getattr(self, "sft_examples", None) is not None:  # examples variant: the same example text as during generation
                ex = self.sft_example_text(df, i, split)
                return S.rank_sample(row, student, qimg, self.proto_system, self.proto_notes[int(i)], findings.get(int(i)), examples=ex)
            return S.rank_sample(row, student, qimg, self.proto_system, self.proto_notes[int(i)], findings.get(int(i)))
        notes = self.proto_notes.get(int(i), "")
        if not notes.strip():
            raise KeyError("proto_from: a row of the batch has no notes")
        row = df.iloc[i]
        student, qimg = PR.open_images(row, split, self.builder.root, self.builder.image)
        return PR.rank_sample(row, student, qimg, self.proto_system, notes)

    def sft_example_text(self, df, i, split: str) -> str:
        """Examples variant: example text of a row for the ranking; must equal the one used during generation (checksum
        per row from <tag>_notes.jsonl of the sft_gen job, `sft_from`)."""
        from vlm import sft as S

        text = self.sft_examples.text(df.iloc[i], split, int(i))
        want = (self.sft_example_hashes or {}).get(int(i))
        if want is None or S.sha256_text(text) != want:
            raise ValueError("sft_examples: example text of the ranking differs from the one used during generation (sft_from)")
        return text

    def check_sft_examples(self, df, idx, split: str) -> int:
        """Examples variant, before the ranking: is the example text of ALL selected rows equal to the one used during
        generation? -> rows with examples. Aborts otherwise, before any computation (only numbers in the message)."""
        bad = n_ex = 0
        for i in idx:
            try:
                n_ex += int(bool(self.sft_example_text(df, int(i), split)))
            except ValueError:
                bad += 1
        if bad:
            raise ValueError(f"sft_examples: for {bad} of {len(idx)} rows the example text differs from the generation; aborting before the ranking")
        return n_ex

    def check_sft_prompt_tokens(self, df, idx, split: str, served: dict[int, int]) -> tuple[int, int]:
        """Examples variant, prompt fidelity: input tokens counted by the vLLM server during
        generation (`served`, sft.load_prompt_tokens) against the length of the SAME prompt on the in-house path
        (chat template and processor, without answer prefix). -> (equal, checked); rows without a server count
        (generated on the in-house path) do not count."""
        same = n = 0
        for i in idx:
            want = int(served.get(int(i), 0) or 0)
            if want <= 0:
                continue
            s = self._build_proto(df, int(i), split)
            s.prefix = ""  # only the prompt up to "<think>\n", the way the server counts it
            n += 1
            same += int(len(self.engine._encode_one(s)["input_ids"]) == want)
        return same, n

    def _sft_finish(self, samples) -> list[str | None]:
        """sft mode: rows without a complete answer get their finding from the in-house path (greedy, up to "Root:");
        the answer prefix is then closed canonically with "\\nRoot:". -> generated finding per sample (otherwise None)."""
        from vlm import sft as S

        need = [k for k, s in enumerate(samples) if S.needs_finding(s)]
        out: list[str | None] = [None] * len(samples)
        if need:
            gen = self.engine.generate([samples[k] for k in need], max_new_tokens=int(self.job.get("finish_tokens", 256)), stop_text=S.ROOT, return_ids=True)
            for k, ids in zip(need, gen):
                text = self.engine.tok.decode(ids, skip_special_tokens=True)
                samples[k].prefix = S.complete_prefix(samples[k].prefix, text)
                out[k] = " ".join(text.split(S.ROOT, 1)[0].split())
        return out

    def _score_codes(self, samples, keep_hidden: list | None = None) -> np.ndarray:
        if keep_hidden is not None:
            lp, h = self.engine.score(samples, self.codes, beams=self.beams, return_hidden=True, exact_chunk=self.exact, first_top=self.first_top)
            keep_hidden.extend(h)
        else:
            lp = self.engine.score(samples, self.codes, beams=self.beams, exact_chunk=self.exact, first_top=self.first_top)
        full = np.full((len(samples), len(self.classes)), -np.inf)
        full[:, self.col] = lp
        return full

    def _estimate_tokens(self, df, i: int, split: str) -> int:
        """Rough prompt length of a row: image tokens from the file header plus text length (without decoding the image)."""
        from PIL import Image

        from vlm import common as C

        row = df.iloc[i]
        n = (len(row["question_text"]) + len(row.get("transcription_steps", "") if hasattr(row, "get") else "")) // 3
        try:
            with Image.open(C.image_path(self.builder.root, split, row["image_file"])) as im:
                w, h = im.size
            s = min(1.0, self.builder.image.max_side / max(w, h))
            if self.builder.image.min_side and max(w, h) < self.builder.image.min_side:
                s = self.builder.image.min_side / max(w, h)
            n += int(np.ceil(w * s / 32) * np.ceil(h * s / 32))
        except OSError:
            n += 1024
        if self.proto_notes is not None:  # probe proto_rank: the row's notes are part of the prompt
            n += len(self.proto_notes.get(int(i), "")) // 3
        if getattr(self, "sft_examples", None) is not None:  # examples variant: the example block is part of the prompt
            n += len(self.sft_examples.text(row, split, int(i))) // 3
        return n + (256 if row["question_image_file"] else 0)

    def run(self, df, idx: np.ndarray, split: str, probes: list[str], out_dir: Path, tag: str, fold_of=None) -> dict:
        """Run the probes on the rows `idx`; writes one .npy or .jsonl and one meta file per probe.

        Robust for long runs: intermediate states are saved regularly and resumed after a restart; if a batch
        fails, its rows run one by one and only the row that really fails stays empty; once the run's deadline is
        reached, the probe ends with what it has (`partial`).
        """
        import torch

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{tag}_idx.npy", idx)
        (out_dir / f"{tag}_meta.json").write_text(json.dumps({
            "table": split, "n": int(len(idx)), "probes": probes, "job": self.job.get("name"),
            "recipe": {k: self.job.get(k) for k in ("taxonomy", "gold_reading", "reading_from", "reading_head", "analysis_head", "think_bridge", "think_bridge_tokens", "tax_cases", "tax_cases_for", "tax_case_chars", "tax_case_topic", "instruction_suffix", "topic_codes", "topic_none", "teacher_labels", "teacher_min_conf", "question_codes", "question_soft", "teacher_examples", "think_suffix", "think_effort", "think_cut_tokens", "candidates", "candidates_k", "analysis_from", "analysis_image", "proto_from", "proto_system", "train_targets", "image", "examples", "max_examples", "beams", "exact_chunk", "first_top", "model", "adapter", "fold")},
            "dtype": self.engine.cfg.dtype, "quant": self.engine.cfg.quant, "adapter": bool(self.engine.has_adapter()),
            "head_with_adapter": bool(getattr(self.engine.cfg, "head_with_adapter", False)),
            **({"sft": {k: self.job.get(k) for k in ("sft_from", "finish_tokens")}} if self.job.get("sft_from") else {}),
            **({"sft_examples": self.job.get("sft_examples")} if self.job.get("sft_examples") is not None else {}),  # M3k
        }, indent=1, default=str))
        # longest prompts first: memory errors show up in the first minutes, not after hours
        est = np.array([self._estimate_tokens(df, int(i), split) for i in idx])
        order = idx[np.argsort(-est, kind="stable")]
        pos = {int(i): k for k, i in enumerate(idx)}
        texts: dict[str, list] = {}
        timing: dict[str, dict] = {}
        # With adapters there is nothing to check: the header deliberately comes from the base model there (as in training),
        # except with head_with_adapter: then header and full path both use the adapter and must agree.
        if self.engine.prefix_cache and (not self.engine.has_adapter() or getattr(self.engine.cfg, "head_with_adapter", False)) and any(p.startswith("diag") or p in ("full", "cot", "think", "why", "why_no", "proto_rank") for p in probes):
            self._check_prefix_cache(df, order[-int(self.job.get("prefix_cache_check_rows", 2)):], split, fold_of, proto=all(p == "proto_rank" for p in probes))
        for probe in probes:
            try:
                self._run_probe(probe, df, idx, split, out_dir, tag, fold_of, order, pos, texts, timing)
            except Exception as exc:  # noqa: BLE001 - one probe must not take the following ones down
                import traceback

                timing[probe] = {"error": f"{type(exc).__name__}: {_mask(str(exc), split)[:300]}"}
                log(f"ERROR probe {probe} [{tag}]: {timing[probe]['error']}")
                (out_dir / f"{tag}_{probe}.error.txt").write_text(_mask(traceback.format_exc(), split))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return timing

    def _check_prefix_cache(self, df, rows, split, fold_of, proto: bool = False) -> None:
        """Before the run: does the reused prompt header give the same result as the full compute path?

        The probabilities are compared (not the log-values of improbable candidates): if one deviates by more
        than `prefix_cache_tol` (default 0.03) or the top changes, the header is switched off for this run. In bf16
        (8-bit mantissa) both paths scatter more for large models than in fp16; there the job may raise the
        threshold after the quality of both paths has been compared.
        """
        from vlm.engine import probs_from_logp

        tol = float(self.job.get("prefix_cache_tol", 0.03))

        def make(i):  # proto: a job with only the probe proto_rank checks the header on its own samples (system text)
            return self._build_proto(df, i, split) if proto else self._build(df, i, split, "diag", fold_of)

        try:
            with_cache = self.engine.score([make(i) for i in rows], self.codes, beams=4)
            self.engine.prefix_cache = False
            without = self.engine.score([make(i) for i in rows], self.codes, beams=4)
            both = np.isfinite(with_cache) & np.isfinite(without)
            pa, pb = probs_from_logp(np.where(both, with_cache, -np.inf)), probs_from_logp(np.where(both, without, -np.inf))
            dev = float(np.abs(pa - pb).max()) if both.any() else float("inf")
            same_top = bool((pa.argmax(1) == pb.argmax(1)).all())
            # If all deviations are below the threshold, the top can only change for a close pair (gap below 2 x threshold).
            # `prefix_cache_same_top: false` then keeps the header on; otherwise a single close check row switches it off and
            # the job runs twice as long.
            ok = dev < tol and (same_top or not bool(self.job.get("prefix_cache_same_top", True)))
            log(f"prompt-header cache checked: max. deviation of the probability {dev:.4f} (threshold {tol}), same top {same_top} -> {'on' if ok else 'OFF'}")
            self.engine.prefix_cache = ok
        except Exception as exc:  # noqa: BLE001
            # The full compute path (without header) needs quadratic memory for long prompts on small cards (SDPA without mask
            # with GQA falls back to the math kernel in fp32); then the comparison cannot be computed here. With
            # `prefix_cache_keep_on_oom: true` the header stays on in this case, UNCHECKED on this hardware and reported as such
            # in the log (Kaggle T4). Default: switch off.
            keep = type(exc).__name__ == "OutOfMemoryError" and bool(self.job.get("prefix_cache_keep_on_oom", False))
            self.engine.prefix_cache = keep
            log(f"prompt-header cache {'on, UNCHECKED (the comparison path without the header does not fit into memory)' if keep else 'switched off'} "
                f"({type(exc).__name__}: {str(exc)[:200]})")

    def _do_batch(self, probe: str, df, b, split: str, fold_of, texts: dict, pos: dict) -> list[dict]:
        """Compute one probe for a batch; per row {"lp": array | None, "h": array | None, "text": str | None}."""
        P = self.P
        want_h = bool(self.job.get("save_hidden", False))
        n = len(b)
        lp, hs, txt = [None] * n, [None] * n, [None] * n
        if probe in ("fac", "why_fac", "up", "diag", "diag_gold", "diag_goldimg", "diag_read"):
            keep: list | None = [] if want_h else None
            if probe in ("fac", "why_fac", "up"):
                ss = [self._build(df, i, split, {"fac": "fac", "why_fac": "why", "up": "up"}[probe], fold_of) for i in b]
                if want_h:
                    arr, h = self.engine.score(ss, self.yn, beams=2, return_hidden=True)
                    keep.extend(h)
                else:
                    arr = self.engine.score(ss, self.yn, beams=2)
            elif probe == "diag":
                arr = self._score_codes([self._build(df, i, split, "diag", fold_of) for i in b], keep)
            elif probe == "diag_gold":
                ss = [self._build(df, i, split, "diag", fold_of, with_image=False,
                                  reading=P.clean_transcription(df.iloc[i]["transcription_steps"])) for i in b]
                arr = self._score_codes(ss, keep)
            elif probe == "diag_goldimg":
                if split != "train":
                    raise ValueError("diag_goldimg exists only for training rows (gold transcription)")
                ss = [self._build(df, i, split, "diag", fold_of, reading=P.clean_transcription(df.iloc[i]["transcription_steps"]),
                                  reading_head=GOLD_READING_HEAD) for i in b]
                arr = self._score_codes(ss, keep)
            else:
                ss = [self._build(df, i, split, "diag", fold_of, reading=_transcription_part(texts["read"][pos[int(i)]])) for i in b]
                arr = self._score_codes(ss, keep)
            lp = list(arr)
            if want_h:
                hs = list(keep)
        elif probe == "ans":  # copy the final answer, then score the answer check behind the own text
            ss = [self._build(df, i, split, "ans", fold_of) for i in b]
            gen = self.engine.generate(ss, max_new_tokens=int(self.job.get("max_answer_tokens", 40)))
            for k, (g, s) in enumerate(zip(gen, ss)):
                txt[k] = (g.splitlines() or [""])[0].strip()[:160]
                s.prefix = P.PREFIX["ans"] + " " + txt[k] + "\nFinal answer correct:"
            lp = list(self.engine.score(ss, self.yn, beams=2))
        elif probe in ("read", "full", "think", "cot"):
            ss = [self._build(df, i, split, probe, fold_of) for i in b]
            limit = int(self.job.get("max_think_tokens", 900)) if probe == "think" else int(self.job.get("max_new_tokens", 320))
            if probe == "think" and self.think_texts is not None:
                missing = [int(i) for i in b if int(i) not in self.think_texts]
                if missing:
                    raise KeyError(f"think_from: {len(missing)} rows of the batch have no reasoning text")
                txt = [self.think_texts[int(i)] for i in b]
            else:
                txt = self.engine.generate(ss, max_new_tokens=limit, repetition_penalty=float(self.job.get("repetition_penalty", 1.0)))
            if probe in ("full", "think", "cot"):  # second pass: code ranking behind the own text
                for s, g in zip(ss, txt):
                    s.prefix = _upto_root(g, close_think=probe == "think")
                bridge = str(self.job.get("think_bridge", "")) if probe == "think" else ""
                if bridge:
                    # Bridge (nine of ten reasoning texts end without a verdict, and naming follows a truncated text): behind the
                    # closed thinking block the model first writes out in its own words what the root error is (answer prefix
                    # `think_bridge`, at most `think_bridge_tokens` tokens, up to "Root:"); the codes are scored behind this stated verdict.
                    for s in ss:
                        s.prefix = s.prefix[: -len("Root:")] + bridge
                    said = self.engine.generate(ss, max_new_tokens=int(self.job.get("think_bridge_tokens", 64)), stop_text="Root:")
                    txt = list(txt)
                    for k, (s, g) in enumerate(zip(ss, said)):
                        line = " ".join(g.split("Root:")[0].split())[:400]
                        s.prefix = s.prefix + (" " + line if line else "") + "\nRoot:"
                        txt[k] = txt[k] + "\n[bridge]" + bridge + " " + line
                lp = list(self._score_codes(ss))
        elif probe in ("why", "why_no"):  # describe first (generated), then score all codes behind the own description
            ss = [self._build(df, i, split, "why", fold_of) for i in b]
            forced = list(self.why_forced) if probe == "why_no" else []
            for s in ss:
                if forced:
                    s.meta["extra_ids"] = list(forced)
            gen = self.engine.generate(ss, max_new_tokens=int(self.job.get("max_why_tokens", 96)), stop_text="Root:", return_ids=True)
            for k, (s, g) in enumerate(zip(ss, gen)):
                text = self.engine.tok.decode(g, skip_special_tokens=True)
                txt[k] = text.strip()
                # Scoring appends exactly the generated tokens (no re-tokenisation of the text); if the description ran into the
                # length limit, "Root:" follows in the tokenisation of the training target.
                s.meta["extra_ids"] = forced + list(g) + ([] if "Root:" in text else list(self.root_ids))
            keep = [] if want_h else None
            lp = list(self._score_codes(ss, keep))
            if want_h:
                hs = list(keep)
        elif probe == "proto_rank" and getattr(self, "sft_findings", None) is not None:  # sft mode: generate missing findings first, then score
            ss = [self._build_proto(df, i, split) for i in b]
            made = self._sft_finish(ss)
            txt = ["" if m is None else "[finding generated] " + m for m in made]
            lp = list(self._score_codes(ss))
        elif probe == "proto_rank":  # all codes exactly behind the notes of the protocol channel
            lp = list(self._score_codes([self._build_proto(df, i, split) for i in b]))
        else:
            raise ValueError(f"unknown probe {probe}")
        return [{"lp": lp[k], "h": hs[k], "text": txt[k]} for k in range(n)]

    def _run_probe(self, probe, df, idx, split, out_dir, tag, fold_of, order, pos, texts, timing) -> None:
        import torch

        from vlm.engine import free_gpu_memory

        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        n = len(idx)
        width = 0 if probe == "read" else (2 if probe in YES_NO_PROBES else len(self.classes))
        has_text = probe in ("ans", "read", "full", "think", "cot", "why", "why_no") or (probe == "proto_rank" and getattr(self, "sft_findings", None) is not None)
        f_lp, f_h, f_txt, f_done = (out_dir / f"{tag}_{probe}{s}" for s in (".npy", "_h.npy", ".jsonl", ".done.npy"))
        res = np.full((n, width), -np.inf) if width else None
        out_txt = [""] * n
        feats: dict[int, np.ndarray] = {}
        done = np.zeros(n, bool)
        if f_done.is_file() and len(np.load(f_done)) == n:  # continuation of an interrupted run
            done = np.load(f_done)
            if res is not None and f_lp.is_file():
                res = np.load(f_lp)
            if has_text and f_txt.is_file():
                for line in open(f_txt):
                    r = json.loads(line)
                    out_txt[pos[int(r["i"])]] = r["text"]
            if f_h.is_file():
                h_old = np.load(f_h)
                feats = {k: h_old[k] for k in np.where(done)[0]} if len(h_old) == n else {}
                done &= np.isin(np.arange(n), list(feats)) if feats else done
            log(f"probe {probe} [{tag}]: resuming, {int(done.sum())} of {n} rows present")

        def save() -> None:
            if res is not None:
                np.save(f_lp, res)
            if has_text:
                with open(f_txt, "w") as fh:
                    for i, g in zip(idx, out_txt):
                        fh.write(json.dumps({"i": int(i), "text": g}, ensure_ascii=False) + "\n")
            if feats:  # hidden state at the decision position, row order as idx (missing rows: 0)
                dim = len(next(iter(feats.values())))
                np.save(f_h, np.stack([feats.get(k, np.zeros(dim, np.float16)) for k in range(n)]).astype(np.float16))
            np.save(f_done, done)

        todo = np.array([i for i in order if not done[pos[int(i)]]], dtype=np.int64)
        failed, partial = 0, False
        every = int(self.job.get("save_every", 40))
        for bi, b in enumerate(self._batches(todo)):
            if self.deadline and time.time() > self.deadline:
                partial = True
                log(f"probe {probe} [{tag}]: deadline reached after {int(done.sum())} of {n} rows")
                break
            outs, batch_error = None, ""
            try:
                outs = self._do_batch(probe, df, b, split, fold_of, texts, pos)
            except Exception as exc:  # noqa: BLE001 - batch failed: rows one by one, only the culprit stays empty
                batch_error = type(exc).__name__
            if outs is None:
                # The repetition deliberately runs AFTER the except block: as long as the exception lives, its traceback keeps
                # the tensors of the failed batch on the GPU, and every single row would fail for the same shortage.
                free_gpu_memory()
                outs = []
                for i in b:
                    one, row_error = None, ""
                    try:
                        one = self._do_batch(probe, df, [i], split, fold_of, texts, pos)
                    except Exception as exc2:  # noqa: BLE001
                        row_error = f"{type(exc2).__name__}: {_mask(str(exc2), split)[:160]}"
                    if one is None:
                        failed += 1
                        outs.append(None)
                        # Test diagnostics are aggregate-only: no selected-shard position, trace identifier,
                        # or row-specific exception text in kernel logs.
                        if failed <= 5 and split != "test":
                            log(f"probe {probe} [{tag}]: row at position {pos[int(i)]} skipped ({row_error})")
                        free_gpu_memory()
                    else:
                        outs += one
                if failed > max(20, n // 10):
                    save()
                    raise RuntimeError(f"too many failed rows ({failed}); last batch error: {batch_error}")
            for i, o in zip(b, outs):
                if o is None:
                    continue
                k = pos[int(i)]
                if res is not None and o["lp"] is not None:
                    res[k] = o["lp"]
                if o["text"] is not None:
                    out_txt[k] = o["text"]
                if o["h"] is not None:
                    feats[k] = o["h"]
                done[k] = True
            if (bi + 1) % every == 0:
                save()
        save()
        if split == "test" and failed:
            log(f"probe {probe} [{tag}]: {failed} test rows skipped (aggregate)")
        if has_text:
            texts[probe] = out_txt
        dt = time.time() - t0
        timing[probe] = {"seconds": round(dt, 1), "s_per_row": round(dt / max(1, len(todo)), 3), "rows": int(done.sum()), "failed": failed}
        if partial:
            timing[probe]["partial"] = True
        if torch.cuda.is_available():
            timing[probe]["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if self.engine.oom_retries:
            timing[probe]["oom_retries"] = int(self.engine.oom_retries)
            self.engine.oom_retries = 0
        if res is not None:
            empty = int((~np.isfinite(res[done]).any(axis=1)).sum())
            if empty:  # marker for non-finite model outputs (number format!); never accept silently
                log(f"WARNING probe {probe} [{tag}]: {empty} of {int(done.sum())} computed rows without a finite candidate")
        log(f"probe {probe} [{tag}]: {n} rows, {timing[probe]}")


def _mask(text: str, split: str) -> str:
    """Remove identifiers of single rows from error messages unless training rows are concerned."""
    import re

    return text if split == "train" else re.sub(r"[0-9a-f]{12}(\.jpe?g)?", "<row>", text)


def _transcription_part(text: str) -> str:
    """The transcription part of a generated answer (without header line and without answer check)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lower().startswith(("transcription:", "final answer correct"))]
    return "\n".join(lines)


def _upto_root(text: str, close_think: bool = False) -> str:
    """Generated text up to and including "Root:", the forced prefix for scoring the codes.

    close_think: in thinking mode the answer starts inside the "<think>" block; if generation stopped there, the
    block is closed before "Root:" follows.
    """
    cut = text.rfind("Root:")
    if cut >= 0 and not (close_think and "</think>" not in text[:cut]):
        return text[:cut] + "Root:"  # verbatim up to this position: the same tokenisation as during generation
    body = text[:cut].rstrip() if cut >= 0 else text.rstrip()
    if close_think and "</think>" not in body:
        body = body + "\n</think>\n"
    return (body + "\n" if body else "") + "Root:"


# ---------------------------------------------------------------------------- Evaluation


def evaluate(df, idx: np.ndarray, out_dir: Path, tag: str, classes: list[str], probes: list[str]) -> dict:
    """Statistics on labelled rows: MAP@3 per code probe, accuracy of the answer check."""
    import labels
    import metric
    from vlm.engine import probs_from_logp

    y = labels.encode(df["root_code"].iloc[idx], classes)
    fac = (df["final_answer_correct"].iloc[idx].str.lower() == "true").to_numpy()
    none = y == 0
    res: dict = {"n": int(len(idx))}
    for probe in probes:
        f = out_dir / f"{tag}_{probe}.npy"
        if not f.is_file():
            continue
        lp = np.load(f)
        if probe in YES_NO_PROBES:
            pred = lp[:, 0] > lp[:, 1]
            margin = lp[:, 0] - lp[:, 1]
            res[probe] = {
                "acc_vs_fac": float(np.mean(pred == fac)), "acc_vs_none": float(np.mean(pred == none)),
                "share_yes": float(pred.mean()), "auc_vs_fac": _auc(margin, fac), "auc_vs_none": _auc(margin, none),
            }
        else:
            p = probs_from_logp(lp)
            rows = metric.map3_rows_from_proba(y, p)
            res[probe] = {
                "map3": float(rows.mean()), "sigma": float(metric.bootstrap_sigma(rows)),
                "map3_none": float(rows[none].mean()) if none.any() else None,
                "map3_error": float(rows[~none].mean()) if (~none).any() else None,
                "top1": float(np.mean(p.argmax(1) == y)), "share_none_top1": float(np.mean(p.argmax(1) == 0)),
            }
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(res, indent=1))
    log(f"statistics [{tag}]:", json.dumps(res))
    return res


def _auc(score: np.ndarray, target: np.ndarray) -> float | None:
    pos, neg = score[target], score[~target]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------- Jobs


def run_job(job: dict, cfg: dict, out_root: Path, device: str) -> dict:
    import labels
    from vlm import common as C
    from vlm.data import ImageConfig, SampleBuilder

    data_root = C.find_data_root(cfg.get("data_root"))
    models_dir = Path(os.environ.get("K12_MODELS") or cfg.get("models_dir") or "/kaggle/temp/models")
    if not models_dir.is_absolute():  # relative paths count from the competition root, never from the working directory
        models_dir = C.comp_root() / models_dir
    train = C.load_table(data_root, "train")
    tax = C.load_taxonomy(data_root)
    folds = C.load_folds(len(train))
    classes = labels.load_classes(data_root / "competition_data" / "taxonomy_v1.csv")
    allowed = [c for c, s in zip(classes, labels.seen_mask(train["root_code"], classes)) if s]
    img_spec = dict(job.get("image", {}))
    rot_file = img_spec.pop("rotate_from", None)
    img = ImageConfig(**img_spec)
    if rot_file:  # orientation normalisation: JSON {trace_id: quarter turns counter-clockwise}, path as for rows.only
        img.rotate = {str(k): int(v) for k, v in json.loads(find_file(rot_file, data_root).read_text()).items()}
        log(f"orientation: {len(img.rotate)} photos of the list {rot_file} are rotated before reading")
    out_dir = out_root / job["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_spec = cfg["models"][job["model"]]
    kind = job["kind"]
    summary: dict = {"job": job["name"], "kind": kind, "model": model_spec}
    t_start = time.time()
    # Examples variant (`sft_examples`, vlm/sft.py): only in the three paths of the fine-tuned diagnostician; otherwise
    # the key would silently have no effect
    if job.get("sft_examples") is not None and not (kind == "sft_gen" or job.get("sft_from") or (
            kind == "train" and "proto_sft" in tuple((job.get("train") or {}).get("tasks", ())))):
        raise ValueError("sft_examples exists only in proto_sft jobs (train with proto_sft, sft_gen, ranking with sft_from)")

    n_cases = int(job.get("tax_cases", 0))  # annotator notes per code in the taxonomy header (prompts.code_cases)
    cases_for = {str(k): int(v) for k, v in (job.get("tax_cases_for") or {}).items()}  # individual count for single codes (case book)
    if set(cases_for) - set(classes):
        raise ValueError(f"tax_cases_for: unknown codes {sorted(set(cases_for) - set(classes))}")
    any_cases = bool(n_cases or cases_for)
    # Teacher labels for pool workings: they only extend the REFERENCE SET for examples and topic block. They carry
    # fold -1 and are therefore excluded by no `exclude_fold`. They become training rows only with the key
    # `teacher_train` (selection and weight: select_teacher_rows).
    teacher = C.load_teacher_rows(data_root, job["teacher_labels"], min_conf=float(job.get("teacher_min_conf", 0.0))) if job.get("teacher_labels") else None
    if (int(job.get("question_codes", 0)) or int(job.get("teacher_examples", 0))) and teacher is None:
        raise ValueError("question_codes and teacher_examples require teacher_labels")
    if teacher is not None and kind == "train" and not job.get("teacher_train") and not int(job.get("question_codes", 0)):
        raise ValueError("teacher_labels in a training job requires teacher_train (teacher rows as training rows) "
                         "or question_codes (teacher rows only in the question block)")

    def with_teacher(reference, ref_folds):
        if teacher is None:
            return reference, ref_folds
        both = pd.concat([reference.assign(_source="gold"), teacher], ignore_index=True)
        if ref_folds is None:  # the reference set is already the right subset (eval, predict)
            return both, None
        return both, np.concatenate([np.asarray(ref_folds, dtype=np.int64), np.full(len(teacher), -1, dtype=np.int64)])

    candidates = None
    if job.get("candidates"):  # candidates from an earlier stage: JSON {trace_id: [code, …]}, path relative to the data folder
        f = Path(job["candidates"])
        candidates = json.loads((f if f.is_absolute() else Path(data_root) / f).read_text())
        bad = {c for v in candidates.values() for c in v} - set(classes)
        if bad:
            raise ValueError(f"candidates: unknown codes {sorted(bad)[:5]}")
        log(f"candidates from an earlier stage for {len(candidates)} rows from {job['candidates']} (up to {int(job.get('candidates_k', 5))} per row)")

    def builder_for(reference, ref_folds=None, case_exclude_fold=None):
        reference, ref_folds = with_teacher(reference, ref_folds)
        return SampleBuilder(data_root, tax, tax_style=job.get("taxonomy", "full"), image=img, reference=reference,
                             reference_folds=ref_folds, max_examples=int(job.get("max_examples", 4)),
                             tax_cases=n_cases, tax_cases_for=cases_for or None, tax_case_chars=int(job.get("tax_case_chars", 200)), tax_case_topic=bool(job.get("tax_case_topic", False)),
                             instruction_suffix=str(job.get("instruction_suffix", "")), case_exclude_fold=case_exclude_fold, topic_codes=int(job.get("topic_codes", 0)), topic_none=bool(job.get("topic_none", False)),
                             question_codes=int(job.get("question_codes", 0)), question_soft=bool(job.get("question_soft", False)), question_allowed=set(allowed), teacher_examples=int(job.get("teacher_examples", 0)),
                             think_suffix=str(job.get("think_suffix", "")), think_effort=str(job.get("think_effort", "")),
                             candidates=candidates, candidates_k=int(job.get("candidates_k", 5)))

    # Notes of another model as input of the probe diag and as training input (the "namer")
    analyses = load_analyses(job["analysis_from"], data_root) if job.get("analysis_from") else None

    # Machine transcriptions per training row (`reading_from: <file.jsonl>`, {"row": position, "text": transcription}; path
    # relative to the data folder); placed after the photo under the generic header line for machine transcriptions.
    readings = None
    if job.get("reading_from"):
        if job.get("gold_reading"):
            raise ValueError("reading_from and gold_reading are mutually exclusive")
        from vlm import think_vllm as _TV

        readings = _TV.load_texts(find_file(job["reading_from"], data_root))
        log(f"reading_from: {len(readings)} transcriptions from {job['reading_from']}")

    def use_readings(runner) -> None:
        runner.readings = readings

    def use_analyses(runner) -> None:
        runner.analysis = analyses

    # Notes of the protocol channel for the probe proto_rank (`proto_from: <file.jsonl>`, the frozen notes export,
    # path relative to the data folder). Like proto_gen, the probe has no question knowledge.
    proto_notes = None
    if job.get("proto_from"):
        from vlm import proto as _PR

        if any(job.get(k) for k in PROTO_NO_TASK_KEYS):
            raise ValueError("proto_from: the probe proto_rank has no question knowledge; " + ", ".join(PROTO_NO_TASK_KEYS) + " are not allowed here")
        proto_notes = _PR.load_rank_notes(find_file(job["proto_from"], data_root))
        log(f"proto_from: notes for {len(proto_notes[1])} rows of table {proto_notes[0]} from {job['proto_from']}")

    # Output of the fine-tuned diagnostician for the probe proto_rank in sft mode (M3; vlm/sft.py): `sft_from:
    # [<run>/]<job>/<tag>` (output of an sft_gen job, <tag>_notes.jsonl) or a file relative to the data folder.
    sft_notes = None
    sft_ex_spec = sft_ex_hashes = None  # examples variant: example setting of the ranking and checksums per row from the generation
    sft_served: dict[int, int] = {}  # examples variant: input tokens per row as counted by the vLLM server during generation
    if job.get("sft_from"):
        from vlm import sft as _S

        if job.get("proto_from"):
            raise ValueError("sft_from and proto_from are mutually exclusive")
        if any(job.get(k) for k in PROTO_NO_TASK_KEYS):
            raise ValueError("sft_from: the probe proto_rank has no question knowledge; " + ", ".join(PROTO_NO_TASK_KEYS) + " are not allowed here")
        ref = str(job["sft_from"])
        if ref.endswith(".jsonl"):
            sft_file = find_file(ref, data_root)
        else:
            bits = ref.split("/")
            sft_file = (out_root if len(bits) == 2 else out_root.parent / bits[0]) / bits[-2] / f"{bits[-1]}_notes.jsonl"
        sft_notes = _S.load_notes(sft_file)
        # eda/c117_q397m3_deploy.py parses this line from the worker log of the test run (run_log_check)
        log(f"sft_from: notes for {len(sft_notes[1])} rows of table {sft_notes[0]}, of which without a finding "
            f"{sum(f is None for _, f in sft_notes[1].values())} (findings are generated before scoring)")
        sft_ex_spec = _S.examples_spec(job)
        sft_ex_hashes = _S.load_example_hashes(sft_file)
        made_with = any(h is not None for h in sft_ex_hashes.values())
        if made_with != (sft_ex_spec is not None):  # generation and ranking see the same prompt, or the job ends here
            raise ValueError(f"sft_from: generation {'with' if made_with else 'without'} examples, ranking {'with' if sft_ex_spec else 'without'}; "
                             "set sft_examples identically for generation and ranking")
        if sft_ex_spec is not None:
            sft_served = _S.load_prompt_tokens(sft_file)

    def use_proto_notes(runner) -> None:
        if sft_notes is not None:
            from vlm import sft as _S

            if runner.engine.has_adapter() and not getattr(runner.engine.cfg, "head_with_adapter", False):  # otherwise the header would come from the base model
                raise ValueError("sft_from: only with a merged model (models: {merged: …}) or with an adapter and "
                                 "head_with_adapter: true in the model (header WITH adapter, as in training)")
            runner.proto_table = sft_notes[0]
            runner.proto_notes = {i: n for i, (n, _) in sft_notes[1].items()}
            runner.sft_findings = {i: f for i, (_, f) in sft_notes[1].items()}
            runner.proto_system = _S.system_text(tax, str(job.get("taxonomy", "full")))
        if proto_notes is not None:
            from vlm import proto as _PR

            runner.proto_table, runner.proto_notes = proto_notes
            # `proto_system: protocol`: control that scores under the system text under which the notes were CREATED
            # (instruction with JSON form) instead of the ranking's. Default: rules of the ranking.
            which = str(job.get("proto_system", "rank"))
            if which not in ("rank", "protocol"):
                raise ValueError(f"proto_system: unknown value {which!r} (rank | protocol)")
            runner.proto_system = _PR.system_text(tax) if which == "protocol" else _PR.rank_system_text(tax)

    def use_sft_examples(runner, recipe) -> None:
        """Examples variant: example source of the ranking in sft mode; fold from the model's recipe (as in training and
        generation), checksums per row from the generation."""
        if sft_ex_spec is None:
            return
        from vlm import sft as _S

        runner.sft_examples = _S.ExampleSource(train, folds, sft_ex_spec, (recipe or {}).get("fold"))
        runner.sft_example_hashes = sft_ex_hashes

    def use_think_texts(runner) -> None:
        """Probe think with finished reasoning texts: `think_from: [<run>/]<job>/<tag>` points to the output of a think_gen
        job, of the same run or, with a preceding run name, of an earlier run on the same machine.
        `think_cut_tokens` truncates the texts afterwards (measurement of the token limit)."""
        if job.get("think_from"):
            from vlm import think_vllm as TV

            parts = str(job["think_from"]).split("/")
            src_job, tag = parts[-2], parts[-1]
            src_root = out_root if len(parts) == 2 else out_root.parent / parts[0]
            runner.think_texts = TV.load_texts(src_root / src_job / f"{tag}_texts.jsonl", cut_tokens=job.get("think_cut_tokens"))
            for mark in job.get("think_strip") or []:  # remove the markers of a model with a different architecture from the text
                runner.think_texts = {i: t.replace(str(mark), "") for i, t in runner.think_texts.items()}
            # Translate the markers of a model with a different architecture into the own form, as a list [[old, new], …] in this
            # order, e.g. the end of its thinking block into "</think>", so that _upto_root finds the answer behind it.
            for old, new in job.get("think_replace") or []:
                runner.think_texts = {i: t.replace(str(old), str(new)) for i, t in runner.think_texts.items()}
            log(f"think_from: {len(runner.think_texts)} reasoning texts from {src_job}/{tag}" + (f", truncated to about {job['think_cut_tokens']} tokens" if job.get("think_cut_tokens") else ""))

    if kind == "think_gen":
        # Generate reasoning texts for the probe think with a local vLLM server (vlm/think_vllm.py); scoring the codes behind
        # them is done by a later bench or predict job with `think_from`. Prompts as in those jobs.
        from vlm import think_vllm as TV

        ensure_vllm_env(job, out_dir)
        model_dir = Path(fetch_model(model_spec, models_dir))
        split = job.get("split", "train")
        with_ex = bool(job.get("examples", True))
        if split == "train":
            idx = select_rows(job.get("rows", {}), len(train), folds)
            gen_case_fold = None
            if any_cases:  # as in bench: cases in the header never from the fold that is generated on
                sel_folds = (job.get("rows") or {}).get("folds") or []
                if len(sel_folds) != 1:
                    raise ValueError("tax_cases / tax_cases_for in a think_gen job on training rows requires rows.folds with exactly one fold")
                gen_case_fold = int(sel_folds[0])
            builder = builder_for(train, folds, gen_case_fold)

            def make(i: int):  # examples never from the row's fold (as in bench)
                kw = {}
                if job.get("gold_reading"):  # as in ProbeRunner._build: transcription in addition to the photo
                    from vlm import prompts as _P

                    kw = {"reading": _P.clean_transcription(train.iloc[i]["transcription_steps"]), "reading_head": GOLD_READING_HEAD}
                elif readings is not None and readings.get(int(i), "").strip():  # machine transcription (`reading_from`)
                    kw = {"reading": readings[int(i)]}
                    if job.get("reading_head"):
                        kw["reading_head"] = str(job["reading_head"])
                return builder.build(train.iloc[i], split="train", task="think", exclude_fold=int(folds[i]), with_examples=with_ex, **kw)
        else:
            if split not in ("test", "unlabelled"):
                raise ValueError(f"think_gen: unknown table {split}")
            if readings is not None:
                raise ValueError("reading_from exists only for training rows")
            df = C.load_table(data_root, split)
            k = job.get("fold")
            builder = builder_for(train if k is None else train.iloc[np.where(folds != k)[0]].reset_index(drop=True))
            sel = job.get("rows", {})
            idx = select_by_question(df, sel, data_root, C) if sel else np.arange(len(df))

            def make(i: int):
                return builder.build(df.iloc[i], split=split, task="think", with_examples=with_ex)
        summary["think_gen"] = TV.generate(job, model_dir, make, None, idx, split, out_dir, job.get("tag", "think"), log=log)
        log(f"think_gen [{job.get('tag', 'think')}]: {summary['think_gen']}")

    elif kind == "proto_gen":
        # Diagnosis WITHOUT question knowledge in the teacher protocol (vlm/proto.py): instruction, taxonomy, question, key,
        # photo; no labelled row in the prompt, therefore no reference set and no fold.
        from vlm import proto as PR

        if any(job.get(k) for k in ("teacher_labels", "topic_codes", "question_codes", "tax_cases", "tax_cases_for", "candidates", "analysis_from", "reading_from")):
            raise ValueError("proto_gen has no question knowledge: teacher_labels, topic_codes, question_codes, tax_cases, candidates, analysis_from and reading_from are not allowed here")
        split = job.get("split", "train")
        sel = job.get("rows", {})
        if split == "train":
            df, idx = train, select_rows(sel, len(train), folds)
        elif split in ("test", "unlabelled"):
            df = C.load_table(data_root, split)
            idx = select_by_question(df, sel, data_root, C) if sel else np.arange(len(df))
        else:
            raise ValueError(f"proto_gen: unknown table {split}")
        # `first: <file.npy>`: these table positions first (early check against an existing measurement of the same rows)
        first_rows = np.load(find_file(job["first"], data_root)) if job.get("first") else None
        # environment and weights only AFTER the row selection: a missing row list shows up immediately, not after the download
        ensure_vllm_env(job, out_dir)
        model_dir = Path(fetch_model(model_spec, models_dir))
        summary["proto_gen"] = PR.generate(job, model_dir, df, idx, split, data_root, tax, classes, img, out_dir, job.get("tag", "proto"), log=log,
                                           first_rows=first_rows)
        log(f"proto_gen [{job.get('tag', 'proto')}]: {summary['proto_gen']}")

    elif kind == "merge":
        # Merge the adapter of a training job (`adapter_from: <job>` of this run or an adapter folder) into the weights of
        # the base model -> models_dir/<out_model> (vlm/sft.merge_lora; loads no model, needs no GPU).
        from vlm import sft as S

        base = Path(fetch_model(model_spec, models_dir))
        src = str(job["adapter_from"])
        train_dir = out_root / src
        adapter = train_dir / "adapter" if (train_dir / "adapter" / "adapter_config.json").is_file() else Path(find_adapter(src))
        recipe_file = train_dir / "sft_recipe.json"
        recipe = json.loads(recipe_file.read_text()) if recipe_file.is_file() else None
        if recipe is not None:
            recipe = {**recipe, "adapter_from": src}
        summary["merge"] = S.merge_lora(base, adapter, Path(models_dir) / str(job["out_model"]), log=log, recipe=recipe)

    elif kind == "sft_gen":
        # Notes, finding and root line of the fine-tuned diagnostician (vlm/sft.py; model: merged, `merged`)
        from vlm import sft as S

        S.refuse_task_knowledge(job)
        ex_spec = S.examples_spec(job)  # examples variant: examples of the same question in the prompt (None = M3)
        split = job.get("split", "train")
        sel = job.get("rows", {})
        if split == "train":
            df, idx = train, select_rows(sel, len(train), folds)
        elif split in ("test", "unlabelled"):
            df = C.load_table(data_root, split)
            idx = select_by_question(df, sel, data_root, C) if sel else np.arange(len(df))
        else:
            raise ValueError(f"sft_gen: unknown table {split}")
        model_dir = Path(fetch_model(model_spec, models_dir))
        # before environment and server: system text and example setting as in training; on training rows only the held-out fold
        recipe = S.read_recipe(model_dir)
        S.check_recipe(recipe, S.system_text(tax, str(job.get("taxonomy", "full"))), split, idx, folds if split == "train" else None, examples=ex_spec)
        ex_src = S.ExampleSource(train, folds, ex_spec, recipe.get("fold")) if ex_spec is not None else None
        engine = None
        if str(job.get("backend", "vllm")) == "vllm":
            ensure_vllm_env(job, out_dir)
        else:
            engine = make_engine(model_spec, models_dir, device)
        summary["sft_gen"] = S.generate(job, model_dir, df, idx, split, data_root, tax, classes, img, out_dir, job.get("tag", "gen"), log=log,
                                        engine=engine, folds=folds if split == "train" else None, examples=ex_src)
        log(f"sft_gen [{job.get('tag', 'gen')}]: {summary['sft_gen']}")

    elif kind == "rationale_gen":
        # Rationalisation variant (vlm/sft.py, section "rationalisation"): the own 27B WITHOUT training writes notes in the
        # teacher protocol with the gold hint of the SAME training row (hint file from a separate target builder). Training
        # rows only, base model only, no further question information; the output is label-conditioned and only a training target.
        from vlm import sft as S

        if str(job.get("split", "train")) != "train":
            raise ValueError("rationale_gen: training rows only (the hint is their label)")
        bad = [k for k in S.NO_TASK_KEYS + ("sft_from", "proto_from", "think_from", "sft_examples", "examples", "rows") if job.get(k)]
        if bad:
            raise ValueError("rationale_gen: rows come only from the hint file, no further question information; not allowed: " + ", ".join(bad))
        tails = S.load_rationale_hints(find_file(job["hints"], data_root), len(train))
        idx = np.array(sorted(tails), dtype=np.int64)
        if job.get("limit"):  # pre-run: fixed random selection of the hint rows
            idx = np.sort(idx[np.random.default_rng(int(job.get("limit_seed", 0))).permutation(len(idx))[: int(job["limit"])]])
        model_dir = Path(fetch_model(model_spec, models_dir))
        if S.read_recipe(model_dir) is not None:
            raise ValueError("rationale_gen: only with the base model without training (the model carries a proto_sft recipe)")
        engine = None
        if str(job.get("backend", "vllm")) == "vllm":
            ensure_vllm_env(job, out_dir)
        else:
            engine = make_engine(model_spec, models_dir, device)
        summary["rationale_gen"] = S.generate_rationales(job, model_dir, train, idx, data_root, tax, img, out_dir, job.get("tag", "rat"), tails,
                                                         log=log, engine=engine)
        log(f"rationale_gen [{job.get('tag', 'rat')}]: {summary['rationale_gen']}")

    elif kind == "train" and "proto_sft" in tuple((job.get("train") or {}).get("tasks", ())):
        # M3 (vlm/sft.py): fine-tuning in the teacher protocol; training targets only from rows of the other folds
        # (frozen target files), no reference set, no examples, no probes in the job (measurement: merge ->
        # sft_gen -> proto_rank with sft_from). Examples variant: with `sft_examples`, examples of the same question from gold rows (fold rule).
        from vlm import proto as _PR
        from vlm import sft as S
        from vlm.train import TrainConfig, train as fit

        S.refuse_task_knowledge(job)
        if job.get("probes") or job.get("predict"):
            raise ValueError("proto_sft: no probes in the training job; measure via merge, sft_gen and proto_rank")
        tspec = dict(job.get("train", {}))
        if tuple(tspec.get("tasks", ())) != ("proto_sft",):
            raise ValueError("proto_sft must be the only entry in train.tasks")
        # examples variant: examples of the same question in the prompt (None = M3, bit-identical); gold targets only (pool rows have no fold)
        ex_spec = S.examples_spec(job)
        if ex_spec is not None and job.get("sft_pool"):
            raise ValueError("sft_examples together with sft_pool is not supported (pool rows: no fold, teacher targets)")
        k = job.get("fold")
        # pool variant: with pool rows the gold targets are called `sft_gold_targets`; a runner without this extension aborts at
        # `sft_targets` instead of silently learning only gold
        gold_key = "sft_gold_targets" if job.get("sft_pool") else "sft_targets"
        targets, tinfo = S.load_targets(find_file(job[gold_key], data_root), k, folds, len(train))
        if job.get("train_rows"):
            targets = [targets[j] for j in select_rows(job["train_rows"], len(targets), None, whole_table=False)]
        engine = make_engine(model_spec, models_dir, device)
        tcfg = TrainConfig(**{**tspec, "tasks": ("proto_sft",)})
        if engine.prefix_cache and tcfg.frozen_prefix and not tcfg.prefix_with_adapter:
            raise ValueError("proto_sft: a header from the base model does not match the merged model in vLLM; set prefix_with_adapter: true "
                             "(or prefix_cache: false in the model)")
        system = S.system_text(tax, str(job.get("taxonomy", "full")))
        cap = int(job.get("sft_notes_max_tokens", 2500))
        notes, ntok, ncut = {}, [], 0
        for rec in targets:  # truncate the notes once in the model's tokenisation (deterministic)
            text, n_tok, cut = S.cut_notes(rec["notes"], engine.tok, cap)
            notes[int(rec["row"])] = text
            ntok.append(n_tok)
            ncut += int(cut)
        summary["sft"] = {**tinfo, "fold": k, "rows_used": len(targets), "notes_loss_used": int(sum(bool(r.get("notes_loss")) for r in targets)),
                          "share_none_used": float(np.mean([r["code"] == "NONE" for r in targets])), "notes_max_tokens": cap, "notes_cut": ncut,
                          "notes_tokens": {p: int(np.percentile(ntok, p)) for p in (5, 50, 90, 100)}, "system_sha256": S.sha256_text(system),
                          "taxonomy": str(job.get("taxonomy", "full")), "prefix_with_adapter": tcfg.prefix_with_adapter,
                          "prefix_cache": bool(engine.prefix_cache)}
        log(f"proto_sft: {len(targets)} training targets (fold {k} held out), notes with loss {summary['sft']['notes_loss_used']}, "
            f"truncated {ncut} (limit {cap} tokens), note tokens {summary['sft']['notes_tokens']}")
        ex_src = None
        if ex_spec is not None:  # examples variant: examples from folds not in {k, fold of the row} (vlm/sft.py, fold rule)
            ex_src = S.ExampleSource(train, folds, ex_spec, k)
            blocks = [ex_src.text(train.iloc[int(r["row"])], "train", int(r["row"])) for r in targets]
            btok = [len(engine.tok(S.examples_block(t), add_special_tokens=False)["input_ids"]) if t else 0 for t in blocks]
            summary["sft"]["examples"] = {**ex_spec, "model_fold": k, "rows_with_examples": int(sum(bool(t) for t in blocks)),
                                          "block_tokens_mean": float(np.mean(btok)), "block_tokens": {p: int(np.percentile(btok, p)) for p in (5, 50, 90, 100)}}
            log(f"sft_examples: {summary['sft']['examples']['rows_with_examples']} of {len(targets)} training targets with examples of the same question "
                f"(at most {ex_spec['max_examples']}, transcription up to {ex_spec['max_chars']} characters), mean block tokens "
                f"{summary['sft']['examples']['block_tokens_mean']:.0f} {summary['sft']['examples']['block_tokens']}, dropout {ex_spec['dropout']}")

        def make_sample(i: int, task: str):
            rec = targets[i]
            row = train.iloc[int(rec["row"])]
            student, qimg = _PR.open_images(row, "train", data_root, img)
            if ex_src is not None:  # examples variant: example block per visit (dropout fixed per row and visit)
                return S.train_sample(row, student, qimg, system, rec, notes[int(rec["row"])], examples=ex_src.train_text(row, int(rec["row"])))
            return S.train_sample(row, student, qimg, system, rec, notes[int(rec["row"])])

        n_items, mix = len(targets), None
        if job.get("sft_pool"):  # pool variant: pool rows with teacher targets; plan, weights and guards in vlm/sft.PoolMix
            mix = S.PoolMix.from_job(job, targets, notes, train, data_root, img, system, engine.tok, tcfg,
                                     find_file(job["sft_pool"]["targets"], data_root), k, log=log)
            make_sample, n_items = mix.make_sample, mix.n_items
            summary["sft_pool"] = mix.info

        budget = job.get("train_budget_s")
        deadline = float(os.environ.get("K12_DEADLINE", 0) or 0)
        if deadline:
            left = deadline - time.time() - float(job.get("probe_reserve_s", 1800))
            budget = max(60.0, min(budget, left) if budget else left)
        summary["train"] = fit(engine, make_sample, n_items, tcfg, out_dir, budget_s=budget)
        tlog = summary["train"].pop("log", None) or []
        if mix is not None:
            summary["sft_pool"]["served"] = dict(mix.served)
        if ex_src is not None:
            summary["sft"]["examples"]["visits"] = dict(ex_src.stats)
        (out_dir / "sft_recipe.json").write_text(json.dumps({
            "task": "proto_sft", "fold": k, "system_sha256": summary["sft"]["system_sha256"], "taxonomy": summary["sft"]["taxonomy"],
            "notes_max_tokens": cap, "targets": tinfo["file"], "targets_sha256": tinfo["sha256"], "rows_used": len(targets),
            "image": {k_: getattr(img, k_) for k_ in ("max_side", "min_side", "question_max_side")}, "model": model_spec,
            "steps_done": summary["train"].get("steps_done"), "steps_total": summary["train"].get("steps_total"),
            "pool": summary.get("sft_pool"), "teacher_derived": mix is not None,
            **({"examples": ex_spec} if ex_spec is not None else {})}, indent=1, default=str))  # examples variant: setting for check_recipe
        if job.get("must_pass"):
            _train_gate(job, summary, tlog, out_dir)

    elif kind == "bench":
        # `adapter:` only together with head_with_adapter in the model (M3 adapter unmerged on a 4-bit base model); older
        # bench jobs set no adapter and stay unchanged.
        if job.get("adapter") and not model_spec.get("head_with_adapter"):
            raise ValueError("bench with adapter: only with head_with_adapter: true in the model (otherwise use eval/predict)")
        engine = make_engine(model_spec, models_dir, device, adapter=job.get("adapter"))
        idx = select_rows(job.get("rows", {}), len(train), folds)
        case_fold = None
        if any_cases:  # the header with the cases is the same for all rows: measurement on exactly one fold, cases from the other folds
            sel_folds = (job.get("rows") or {}).get("folds") or []
            if len(sel_folds) != 1:
                raise ValueError("tax_cases / tax_cases_for in a bench job requires rows.folds with exactly one fold")
            case_fold = int(sel_folds[0])
        # examples never come from the row's fold (as in cross-validation)
        runner = ProbeRunner(engine, builder_for(train, folds, case_fold), classes, allowed, job)
        use_think_texts(runner)
        use_analyses(runner)
        use_readings(runner)
        use_proto_notes(runner)
        idx = proto_rows(runner, idx, "train", job["probes"])
        if sft_notes is not None:  # system text as in training, only rows of the fold the model has not learned
            from vlm import sft as _S

            _recipe = _S.recipe_for_engine(runner.engine, fetch_model(model_spec, models_dir))
            _S.check_recipe(_recipe, runner.proto_system, "train", idx, folds, examples=sft_ex_spec)
            use_sft_examples(runner, _recipe)
            if runner.sft_examples is not None:  # examples variant: before computing, the same example text as during generation (all rows)
                log(f"sft_examples: example text equal to the generation in all {len(idx)} rows, of which with examples "
                    f"{runner.check_sft_examples(train, idx, 'train')}")
                # fidelity: did the vLLM server see the same prompt as this compute path? (number of input tokens per row)
                same, n_chk = runner.check_sft_prompt_tokens(train, idx, "train", sft_served)
                summary["sft_prompt_tokens"] = {"same": same, "checked": n_chk}
                log(f"fidelity (sft_examples): input tokens of the vLLM server equal to the in-house path in {same} of {n_chk} rows"
                    + ("" if same == n_chk else "; WARNING: generation and ranking did not see the same prompt"))
                if same < n_chk and job.get("must_pass"):
                    raise ValueError(f"pre-run: input tokens of the vLLM server differ from the in-house path in {n_chk - same} of {n_chk} rows")
        summary["timing"] = runner.run(train, idx, "train", job["probes"], out_dir, "bench", fold_of=folds)
        summary["metrics"] = evaluate(train, idx, out_dir, "bench", classes, job["probes"])
        for pred in job.get("predict", []):  # the same probe also on test or pool without loading the model again (aggregates only)
            if pred["split"] == "train":
                raise ValueError("predict is for test and pool")
            summary[f"timing_{pred['split']}"] = _predict(runner, C, data_root, pred, out_dir)

    elif kind == "train":
        from vlm.train import TrainConfig, train as fit

        if any_cases:  # a training row would find its own note as a case of its code in the header (leak into the target)
            raise ValueError("tax_cases / tax_cases_for is not yet enabled for training (cases would have to be separated per fold)")
        k = job.get("fold")  # None = train on all rows
        tr_idx = np.arange(len(train)) if k is None else np.where(folds != k)[0]
        sel = job.get("train_rows")
        if sel:
            tr_idx = tr_idx[select_rows(sel, len(tr_idx), None, whole_table=False)]
        ref = train.iloc[tr_idx].reset_index(drop=True)
        # Examples for a training row never come from its own CV fold: it sees the other training rows the way a validation
        # or test row sees the training part (as a foreign sample). Leaving out only the row itself would be leave-one-out
        # target leakage.
        ref_folds = folds[tr_idx]
        builder = builder_for(ref, ref_folds)
        engine = make_engine(model_spec, models_dir, device)
        tspec = dict(job.get("train", {}))
        if tspec.get("init_adapter"):  # fine-tuning on top of an existing adapter (e.g. after pool pre-training)
            found = find_adapter(str(tspec["init_adapter"]))
            if not found:
                raise FileNotFoundError(f"init_adapter {tspec['init_adapter']!r}: no adapter found")
            tspec["init_adapter"] = found
            log(f"init_adapter: {found}")
        tcfg = TrainConfig(**{**tspec, "tasks": tuple(tspec.get("tasks", ("diag", "full")))})
        rng = np.random.default_rng(tcfg.seed)
        drop = float(job.get("example_dropout", 0.0))
        # Teacher rows of the pool as additional training rows: image from images/unlabelled, targets from the frozen
        # teacher outputs, own weight. Examples and topic block come from the reference set as for every row; for a pool
        # row these are the labelled rows of the training part (no fold exclusion needed: it belongs to no fold) and the
        # other teacher rows of its question.
        t_rows, t_weight = (None, None)
        if teacher is not None and job.get("teacher_train"):
            t_rows, t_weight = select_teacher_rows(teacher, job["teacher_train"], labels.seen_mask(ref["root_code"], classes), classes)
            pool_only = bool(job["teacher_train"].get("only", False))
            print(f"train: {0 if pool_only else len(ref)} labelled rows + {len(t_rows)} teacher rows (NONE {float((t_rows['root_code'] == 'NONE').mean()):.3f}, "
                  f"mean weight {float(t_weight.mean()):.3f}, {t_rows['question_id'].nunique()} questions)", flush=True)
            summary["teacher_train"] = {"rows": int(len(t_rows)), "share_none": float((t_rows["root_code"] == "NONE").mean()),
                                        "mean_weight": float(t_weight.mean()), "questions": int(t_rows["question_id"].nunique()), "only": pool_only}
        # `teacher_train.only`: learn ONLY teacher rows of the pool (pure distillation). The labelled rows remain the
        # reference set of the prompt; such a model depends on no fold only if the prompt shows no labelled rows either
        # (examples: false, no topic block); the job checks this itself.
        n_gold = 0 if (t_rows is not None and job["teacher_train"].get("only")) else len(ref)
        if n_gold == 0 and (bool(job.get("examples", True)) or int(job.get("topic_codes", 0)) or int(job.get("question_codes", 0))):
            raise ValueError("teacher_train.only requires examples: false and neither topic nor question block (otherwise the model would learn from prompts with labels of all folds)")
        items = None  # with `analysis_from`: only the rows of the training part that have a note
        if analyses is not None:
            if t_rows is not None:
                raise ValueError("analysis_from and teacher_train are mutually exclusive (pool rows have no notes)")
            have = analyses.get("train", {})
            items = np.array([j for j, i in enumerate(tr_idx) if int(i) in have], dtype=np.int64)
            n_texts = [len(have[int(tr_idx[j])]) for j in items]
            print(f"train: {len(items)} of {len(tr_idx)} rows of the training part have a note ({float(np.mean(n_texts)):.2f} per row on average); "
                  f"image {'on' if bool(job.get('analysis_image', True)) else 'off'}, targets {job.get('train_targets', 'all')}", flush=True)
            summary["analysis_train"] = {"rows": int(len(items)), "texts_per_row": float(np.mean(n_texts))}
        soft = bool((job.get("teacher_train") or {}).get("soft", False))
        if soft and tuple(tcfg.tasks) != ("diag",):  # only there the code comes first and the rest can be cut off cleanly
            raise ValueError("teacher_train.soft exists only for the task diag")
        known_codes = {c for c, ok in zip(classes, labels.seen_mask(ref["root_code"], classes)) if ok}

        def make_sample(i: int, task: str):
            with_ex = bool(job.get("examples", True)) and rng.random() >= drop
            if items is not None:  # namer: a note (one of the available draws) as input
                j = int(items[i])
                texts = have[int(tr_idx[j])]
                smp = builder.build(ref.iloc[j], split="train", task=task, with_target=True, exclude_fold=int(ref_folds[j]), with_examples=with_ex,
                                    with_image=bool(job.get("analysis_image", True)), analysis=texts[int(rng.integers(len(texts)))])
                if job.get("train_targets") == "code":  # without an image, transcription and findings are no sensible target
                    smp.target = [seg for seg in smp.target if seg[1] == "code"]
                return smp
            if i < n_gold:
                return builder.build(ref.iloc[i], split="train", task=task, with_target=True, exclude_fold=int(ref_folds[i]), with_examples=with_ex)
            t_row = t_rows.iloc[i - n_gold]
            alt = soft_code(t_row, rng, known_codes) if soft else None
            if alt is not None:  # a code other than the teacher's top was drawn: learn only the code and the answer check
                t_row = t_row.copy()
                t_row["root_code"] = alt
            smp = builder.build(t_row, split="unlabelled", task=task, with_target=True, exclude_fold=None, with_examples=with_ex)
            if alt is not None:
                # transcription and findings describe the teacher's top, not the drawn code; leave them out
                last_fac = max(k for k, (_, tag) in enumerate(smp.target) if tag == "fac")
                first_code = min(k for k, (_, tag) in enumerate(smp.target) if tag == "code")
                smp.target = [seg for k, seg in enumerate(smp.target) if k <= max(last_fac, first_code) and seg[1] != "why"]
            smp.meta["weight"] = float(t_weight[i - n_gold])
            return smp

        budget = job.get("train_budget_s")
        deadline = float(os.environ.get("K12_DEADLINE", 0) or 0)
        if deadline:  # deadline of the whole run: stop in time so that the probes can still run afterwards
            left = deadline - time.time() - float(job.get("probe_reserve_s", 1800))
            budget = max(60.0, min(budget, left) if budget else left)
        runner = ProbeRunner(engine, builder, classes, allowed, job)
        use_analyses(runner)
        use_readings(runner)
        va = None
        if k is not None and job.get("probes"):
            va = np.where(folds == k)[0]
            if analyses is not None:
                va = va[[int(i) in analyses.get("train", {}) for i in va]]
            sel = job.get("val_rows")
            if sel:
                va = va[select_rows(sel, len(va), None, whole_table=False)]

        def after_epoch(ep: int) -> None:  # learning curve on the validation fold
            if va is not None and job.get("eval_each_epoch"):
                runner.run(train, va, "train", job.get("epoch_probes", ["diag"]), out_dir, f"oof{k}_e{ep}")
                summary[f"metrics_val_e{ep}"] = evaluate(train, va, out_dir, f"oof{k}_e{ep}", classes, job.get("epoch_probes", ["diag"]))

        n_items = len(items) if items is not None else n_gold + (0 if t_rows is None else len(t_rows))
        summary["train"] = fit(engine, make_sample, n_items, tcfg, out_dir, budget_s=budget, on_epoch_end=after_epoch)
        tlog = summary["train"].pop("log", None) or []
        if job.get("must_pass"):  # pre-run of a training: too many failed or too slow steps end the worker
            done = max(1, int(summary["train"].get("steps_done", 0)))
            bad = (int(summary["train"].get("nonfinite", 0)) + int(summary["train"].get("skipped_oom", 0))) / done
            s_step = float(summary["train"].get("seconds", 0.0)) / done
            if len(tlog) >= 2 and int(tlog[-1]["step"]) > int(tlog[0]["step"]):
                # Do not count the warm-up: the first log window contains the machine warm-up (109 s for the first 8 steps on a
                # fresh machine, then 0.57 s per step; the mean over everything would have aborted a viable training).
                s_all = s_step
                s_step = (float(tlog[-1]["elapsed_s"]) - float(tlog[0]["elapsed_s"])) / (int(tlog[-1]["step"]) - int(tlog[0]["step"]))
                log(f"pre-run {job['name']}: {s_all:.2f} s per step overall, {s_step:.2f} s after the first log window (decisive)")
            log(f"pre-run {job['name']}: {done} steps, {s_step:.2f} s per step, share of failed steps {bad:.2f}")
            if bad > float(job.get("max_bad_frac", 0.25)) or (job.get("max_s_per_step") and s_step > float(job["max_s_per_step"])):
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
                raise SystemExit(f"pre-run {job['name']}: training not viable on this machine ({s_step:.2f} s per step, {bad:.2f} failed); worker stops")
        if va is not None:
            summary["timing_val"] = runner.run(train, va, "train", job["probes"], out_dir, f"oof{k}")
            summary["metrics_val"] = evaluate(train, va, out_dir, f"oof{k}", classes, job["probes"])
            # The same rows of the validation fold with the notes of OTHER draws as input (`val_analyses:
            # {<short name>: [files]}`): measures how much naming depends on the single draw and provides the building
            # blocks for averaging over draws.
            for suffix, files in (job.get("val_analyses") or {}).items():
                alt = load_analyses(files, data_root)["train"]
                runner.analysis = {**(analyses or {}), "train": alt}
                va2 = np.array([i for i in np.where(folds == k)[0] if int(i) in alt], dtype=np.int64)
                if job.get("val_rows"):
                    va2 = va2[select_rows(job["val_rows"], len(va2), None, whole_table=False)]
                summary[f"timing_val_{suffix}"] = runner.run(train, va2, "train", job["probes"], out_dir, f"oof{k}_{suffix}")
                summary[f"metrics_val_{suffix}"] = evaluate(train, va2, out_dir, f"oof{k}_{suffix}", classes, job["probes"])
            use_analyses(runner)
            use_readings(runner)
        use_readings(runner)
        for pred in job.get("predict", []):
            summary[f"timing_{pred['split']}"] = _predict(runner, C, data_root, pred, out_dir)

    elif kind == "eval":  # evaluate a stored adapter (or the base model) on a validation fold
        k = int(job["fold"])
        ref = train.iloc[np.where(folds != k)[0]].reset_index(drop=True)
        engine = make_engine(model_spec, models_dir, device, adapter=job.get("adapter"))
        runner = ProbeRunner(engine, builder_for(ref), classes, allowed, job)
        use_analyses(runner)
        use_readings(runner)
        va = np.where(folds == k)[0]
        if analyses is not None:
            va = va[[int(i) in analyses.get("train", {}) for i in va]]
        if job.get("val_rows"):
            vr = dict(job["val_rows"])
            if vr.get("only"):  # positions in the whole table (e.g. a rotation list): restrict to them first, then apply the remaining selection
                va = va[np.isin(va, np.load(find_file(vr.pop("only"), data_root)))]
            va = va[select_rows(vr, len(va), None, whole_table=False)]
        tag = job.get("tag", f"oof{k}")
        summary["timing_val"] = runner.run(train, va, "train", job["probes"], out_dir, tag)
        summary["metrics_val"] = evaluate(train, va, out_dir, tag, classes, job["probes"])

    elif kind == "predict":
        engine = make_engine(model_spec, models_dir, device, adapter=job.get("adapter"))
        k = job.get("fold")
        ref = train if k is None else train.iloc[np.where(folds != k)[0]].reset_index(drop=True)
        runner = ProbeRunner(engine, builder_for(ref), classes, allowed, job)
        use_think_texts(runner)
        use_analyses(runner)
        use_readings(runner)
        use_proto_notes(runner)
        if sft_notes is not None:  # system text as in training (the row selection checks proto_rows per table)
            from vlm import sft as _S

            _recipe = _S.recipe_for_engine(runner.engine, fetch_model(model_spec, models_dir))
            _S.check_recipe(_recipe, runner.proto_system, sft_notes[0], [], None, examples=sft_ex_spec)
            use_sft_examples(runner, _recipe)  # examples variant: check of the example text per table in _predict
        for pred in job.get("predict", []):
            if pred["split"] == "train":  # training rows would see themselves as example neighbours (leave-one-out)
                raise ValueError("predict is for test and pool; training rows are measured per fold by the job `eval`")
            summary[f"timing_{pred['split']}"] = _predict(runner, C, data_root, pred, out_dir)
    else:
        raise ValueError(f"unknown job kind {kind}")

    summary["seconds"] = round(time.time() - t_start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    failed = failed_probes(summary, strict=bool(job.get("must_pass")))
    if failed and job.get("must_pass"):  # pre-run: if it fails, the long jobs after it are not worth running
        raise SystemExit(f"pre-run {job['name']} with errors in {failed}; worker stops")
    if job.get("drop_model"):  # make room for the next model (also a merged one, `merged`)
        import shutil

        shutil.rmtree(models_dir / (model_spec.get("merged") or model_spec["repo"].split("/")[-1]), ignore_errors=True)
    return summary


def _train_gate(job: dict, summary: dict, tlog: list, out_dir: Path) -> None:
    """Pre-run of a proto_sft training (must_pass): too many failed steps, too slow steps (without the first log window,
    which contains the warm-up) or too much memory end the worker, as in the training job."""
    done = max(1, int(summary["train"].get("steps_done", 0)))
    bad = (int(summary["train"].get("nonfinite", 0)) + int(summary["train"].get("skipped_oom", 0))) / done
    s_step = float(summary["train"].get("seconds", 0.0)) / done
    if len(tlog) >= 2 and int(tlog[-1]["step"]) > int(tlog[0]["step"]):
        s_step = (float(tlog[-1]["elapsed_s"]) - float(tlog[0]["elapsed_s"])) / (int(tlog[-1]["step"]) - int(tlog[0]["step"]))
    gpu = max((float(r.get("gpu_gb", 0.0)) for r in tlog), default=0.0)
    summary["gate"] = {"s_per_step": round(s_step, 3), "bad_frac": round(bad, 3), "gpu_gb_max": gpu}
    log(f"pre-run {job['name']}: {done} steps, {s_step:.2f} s per step (after the first window), failed {bad:.2f}, memory up to {gpu:.1f} GB")
    if (bad > float(job.get("max_bad_frac", 0.25)) or (job.get("max_s_per_step") and s_step > float(job["max_s_per_step"]))
            or (job.get("max_gpu_gb") and gpu > float(job["max_gpu_gb"]))):
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        raise SystemExit(f"pre-run {job['name']}: training not viable on this machine ({s_step:.2f} s per step, {bad:.2f} failed, "
                         f"{gpu:.1f} GB); worker stops")


def failed_probes(summary: dict, strict: bool) -> list[str]:
    """Probes of a job that count as failed: always those with an error message. `strict` (pre-run with must_pass):
    additionally every probe with single failed rows or a partial result; a probe itself only aborts after more
    than max(20, n/10) failed rows, so a pre-run with 8 rows would never get there."""
    out = []
    for key in summary:
        if not key.startswith("timing") or not isinstance(summary[key], dict):
            continue
        for probe, v in summary[key].items():
            if isinstance(v, dict) and ("error" in v or (strict and (v.get("failed", 0) or v.get("partial")))):
                out.append(probe)
    return out


def soft_code(row, rng, known: set) -> str | None:
    """Soft training target by sampling: choose a code from the teacher's ranking according to its probabilities.

    Over the epochs this gives on average the gradient of the soft target sum_c q(c) * (-log P(c)), without several
    passes per sample. Returns None if the top was drawn (the sample then stays unchanged) or the data is missing.
    Only codes known to the training part.
    """
    try:
        top, probs = json.loads(row["_top3"]), json.loads(row["_probs"])
    except (KeyError, TypeError, ValueError):
        return None
    pairs = [(c, max(0.0, float(p))) for c, p in zip(top, probs) if c in known]
    if len(pairs) < 2 or pairs[0][0] != row["root_code"]:
        return None
    w = np.array([p for _, p in pairs], dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    pick = pairs[int(rng.choice(len(pairs), p=w / w.sum()))][0]
    return None if pick == row["root_code"] else pick


def select_teacher_rows(teacher, spec: dict, seen: np.ndarray, classes: list[str]):
    """Select teacher rows of the pool for training -> (rows, weights).

    spec: {min_conf: 0.6, none_share: 0.35, per_question_code: 0, weight: 0.7, conf_weight: true, seed: 0, limit: 0, soft: false}
      soft               draw the target per visit from the teacher's ranking (soft_code; acts in the training job)
      min_conf           only rows on which the teacher was at least this confident (rows without a value are dropped)
      none_share         share of NONE rows among the selected teacher rows; the pool consists of about three quarters
                         correct workings, train and test of one third; surplus rows are dropped at random
      per_question_code  at most this many rows per question and code (0 = all), most confident first
      per_question_none  separate cap per question for NONE rows (default: the same as per_question_code); a small cap
                         per error code covers many (question, code) pairs with few rows, but would leave too few
                         NONE rows for the desired share
      weight             loss weight of a teacher row relative to a labelled row (1.0)
      conf_weight        additionally multiply the weight by the teacher's confidence
    Only codes known to the training part (`seen`): the ranking cannot score others.
    """
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    known = {c for c, ok in zip(classes, seen) if ok}
    t = teacher[teacher["root_code"].isin(known)]
    conf = t["_conf"].to_numpy(dtype=float)
    min_conf = float(spec.get("min_conf", 0.0))
    if min_conf > 0:
        t, conf = t[conf >= min_conf], conf[conf >= min_conf]
    t = t.assign(_c=np.nan_to_num(conf, nan=0.5)).sort_values(["question_id", "root_code", "_c", "trace_id"], ascending=[True, True, False, True])
    cap = int(spec.get("per_question_code", 0))
    cap_none = int(spec.get("per_question_none", cap))
    if cap > 0 or cap_none > 0:
        big = np.iinfo(np.int64).max
        lim = np.where((t["root_code"] == "NONE").to_numpy(), cap_none if cap_none > 0 else big, cap if cap > 0 else big)
        t = t[t.groupby(["question_id", "root_code"]).cumcount().to_numpy() < lim]
    is_none = (t["root_code"] == "NONE").to_numpy()
    n_err = int((~is_none).sum())
    share = float(spec.get("none_share", 0.35))
    n_none = min(int(is_none.sum()), int(round(n_err * share / max(1e-9, 1.0 - share))))
    keep = np.where(~is_none)[0].tolist() + sorted(rng.choice(np.where(is_none)[0], size=n_none, replace=False).tolist())
    t = t.iloc[sorted(keep)]
    if int(spec.get("limit", 0)):
        t = t.iloc[np.sort(rng.choice(len(t), size=min(len(t), int(spec["limit"])), replace=False))]
    w = np.full(len(t), float(spec.get("weight", 0.7)))
    if spec.get("conf_weight", True):
        w = w * t["_c"].to_numpy(dtype=float)
    return t.drop(columns=["_c"]).reset_index(drop=True), w


def select_by_question(df, sel: dict, data_root, C) -> np.ndarray:
    """Row selection by question: only questions that occur in another table; at most k rows per question.

    sel: {only: file.npy, questions_from: test|train, questions_fold: k, per_question: k, seed: s, limit: n, shard: [i, n]}, all optional.
    `questions_fold` (only with questions_from: train) restricts to questions that occur in the training rows of
    this CV fold (for measurements that use one fold as a stand-in for the test set). The choice depends only on
    question identifiers (keys), never on the contents of single test rows.
    """
    idx = only_rows(np.arange(len(df)), sel, data_root)
    if sel.get("questions_from"):
        src = C.load_table(data_root, sel["questions_from"])
        if sel.get("questions_fold") is not None:
            if sel["questions_from"] != "train":
                raise ValueError("questions_fold exists only for questions_from: train")
            src = src[C.load_folds(len(src)) == int(sel["questions_fold"])]
        other = set(src["question_id"])
        idx = idx[df["question_id"].isin(other).to_numpy()]
    if sel.get("per_question"):
        rng = np.random.default_rng(int(sel.get("seed", 0)))
        order = rng.permutation(idx)
        q = df["question_id"].to_numpy()[order]
        rank = np.zeros(len(order), dtype=np.int64)
        seen: dict[str, int] = {}
        for k, name in enumerate(q):
            rank[k] = seen.get(name, 0)
            seen[name] = rank[k] + 1
        idx = np.sort(order[rank < int(sel["per_question"])])
    if sel.get("limit"):
        idx = idx[: int(sel["limit"])]
    if sel.get("shard"):
        i, n = sel["shard"]
        idx = idx[int(i) :: int(n)]
    return idx


def _predict(runner: ProbeRunner, C, data_root, pred: dict, out_dir: Path) -> dict:
    """Probes on test or pool rows; logs aggregates only."""
    split = pred["split"]
    df = C.load_table(data_root, split)
    sel = pred.get("rows", {})
    if any(k in sel for k in ("questions_from", "per_question", "shard", "only")):
        idx = select_by_question(df, sel, data_root, C)
    else:
        idx = select_rows(sel, len(df), None)
    idx = proto_rows(runner, idx, split, pred["probes"])
    if getattr(runner, "sft_examples", None) is not None:  # examples variant: before computing, the same example text as during generation (numbers only)
        log(f"sft_examples: example text equal to the generation in all {len(idx)} rows ({split}), of which with examples "
            f"{runner.check_sft_examples(df, idx, split)}")
    log(f"{split}: {len(idx)} of {len(df)} rows selected")
    timing = runner.run(df, idx, split, pred["probes"], out_dir, pred.get("tag", split))
    for probe in pred["probes"]:
        f = out_dir / f"{pred.get('tag', split)}_{probe}.npy"
        if f.is_file() and probe not in YES_NO_PROBES:
            top = np.load(f).argmax(1)
            log(f"{split}/{probe}: {len(idx)} rows, share of NONE at rank 1 {np.mean(top == 0):.3f}, distinct codes at rank 1 {len(set(top))}")
    return timing


# ---------------------------------------------------------------------------- Entry point


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out_root = Path(a.out) / cfg["run"]
    out_root.mkdir(parents=True, exist_ok=True)

    if a.worker is None:  # one subprocess with its own GPU per worker; stages (`stage`) run one after another
        # Record the code state of the run: the start scripts write the commit of the code package to build.yaml; without
        # this line in the outputs it can no longer be shown later which code state a result came from.
        build = Path(__file__).resolve().parents[2] / "build.yaml"
        commit = None
        if build.is_file():
            commit = (yaml.safe_load(build.read_text()) or {}).get("commit")
        (out_root / "build.json").write_text(json.dumps({"run": cfg["run"], "commit": None if commit is None else str(commit)}))
        log(f"code state: {commit if commit is not None else 'unknown (no build.yaml)'}")
        # Deadline of the whole run: a Kaggle run over 12 h delivers nothing, so end earlier with partial results.
        hours = cfg.get("budget_hours", 11.3 if Path("/kaggle/working").is_dir() else None)
        if hours and not os.environ.get("K12_DEADLINE"):
            os.environ["K12_DEADLINE"] = str(time.time() + float(hours) * 3600)
            log(f"deadline of the run: {hours} h from now")
        codes: dict[int, int] = {}
        stages = sorted({int(s.get("stage", 0)) for s in cfg["workers"]})
        for stage in stages:
            procs = []
            for w, spec in enumerate(cfg["workers"]):
                if int(spec.get("stage", 0)) != stage:
                    continue
                env = dict(os.environ)
                if spec.get("gpu") is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
                # changing prompt lengths fragment the allocator's memory; expandable segments avoid this
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                logf = open(out_root / f"worker{w}.log", "w")
                cmd = [sys.executable, "-u", "-m", "vlm.runner", "--config", a.config, "--out", a.out, "--worker", str(w)]
                procs.append((w, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(HERE.parents[1])), logf))
            offsets = {w: 0 for w, _, _ in procs}
            while True:
                alive = False
                for w, p, _ in procs:
                    alive |= p.poll() is None
                    f = out_root / f"worker{w}.log"
                    with open(f, errors="replace") as fh:
                        fh.seek(offsets[w])
                        chunk = fh.read()
                        offsets[w] = fh.tell()
                    for line in chunk.splitlines():
                        if line.strip() and "Warning" not in line and not line.startswith("[ERROR]"):
                            print(f"[w{w}] {line}", flush=True)
                if not alive:
                    break
                time.sleep(15)
            codes.update({w: p.returncode for w, p, _ in procs})
            (out_root / "status.json").write_text(json.dumps({"workers": codes}))
            log(f"stage {stage} finished:", {w: p.returncode for w, p, _ in procs})
        return 0 if all(c == 0 for c in codes.values()) else 1

    spec = cfg["workers"][a.worker]
    device = "cuda:0" if spec.get("gpu") is not None else "cpu"
    failed = 0
    for job in spec["jobs"]:
        try:
            log(f"=== job {job['name']} ({job['kind']}) ===")
            run_job(job, cfg, out_root, device)
        except SystemExit as exc:
            log(str(exc))
            return 1
        except Exception as exc:  # noqa: BLE001 - one job must not take the following ones down
            import traceback

            failed += 1
            log(f"ERROR in {job['name']}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            (out_root / f"{job['name']}.error.txt").write_text(traceback.format_exc())
            if job.get("must_pass"):  # a failed pre-run ends the worker, even if the job itself aborted
                log(f"pre-run {job['name']} failed — worker aborts")
                return 1
        finally:
            try:
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
