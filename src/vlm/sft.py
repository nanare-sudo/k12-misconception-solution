"""M3: fine-tuned diagnostician without question knowledge, in the teacher protocol (task proto_sft).

Before M3 the 27B was fine-tuned only in the form "code first". Here the model learns in the format of the channel
pr27 (instruction with the verdict rules, taxonomy, question, diagram, key, photo; NO labelled row, no topic or question
block, no teacher labels) to think first, then write one finding line and then name the code. The ranking of all codes
is afterwards scored exactly behind the own text (probe proto_rank in sft mode; mechanics as for the reasoning channel).

Training target per training row (only rows of the OTHER folds; frozen target files, see docs/CODEMAP.md §10):
    <think>\\n NOTES \\n</think>\\n\\nFinding: EXPLANATION\\nRoot: CODE<|im_end|>
  * NOTES = the reasoning of the first pr27 pass for this row (data/derived/proto_notes/train_pr27.jsonl). With loss
    (marker notes) only where the pr27 channel ranks the gold code first (flag pr27_ok of the target builder);
    otherwise they stand in the prompt as a closed thinking block, without loss (as in proto.rank_sample). Reasoning
    texts of the reasoning channel are not used as targets: their prompts contain examples of the same question from
    other folds, which would leak labels across folds.
  * EXPLANATION = full explanation of the annotators' root finding (`misconceptions`, field text of the finding with
    root = true), "none" for NONE. Loss "first token at full weight, rest averaged" (marker why): the first token
    decides between "none" and a description (see the description loss in vlm/train.py).
  * CODE with sum loss (marker code, -log P(code)).
  The segments are tokenised TOGETHER (`meta["joint_target"]`, engine.encode_train): the ranking later reads the same
  text as one string; tokenised separately, the boundaries (".\\n" before "Root:") would give other tokens than scoring.

Compute path (configs/vlm/m3_pilot_rp.yaml):
  1. train   (job kind: train, tasks [proto_sft], `sft_targets`)  -> adapter
  2. merge   (kind: merge)    merge the adapter into the weights of the base model (`merge_lora`, without loading the
                              model; unchanged weight files are linked) -> own model folder
  3. sft_gen (kind: sft_gen)  notes + finding + root with the merged model through the local vLLM server (or, only for
                              the local functional check, with the in-house path) -> <tag>_notes.jsonl
  4. rank    (kind: bench / predict, probe proto_rank, `sft_from`)  all codes exactly behind the canonical text
                              "NOTES\\n</think>\\n\\nFinding: FINDING\\nRoot:"; if the finding is missing (length limit),
                              the in-house path generates it before scoring.
  During training the header (system text) is computed with the CURRENT adapter (TrainConfig.prefix_with_adapter,
  without gradient, again after every optimiser step), the way the merged model computes it in vLLM and when scoring.
  A header computed with the base model during training (frozen_prefix) did not match vLLM.

Examples variant (job key `sft_examples`, section below): the same diagnostician, which ADDITIONALLY sees labelled
workings of other students for the same question in the user prompt, like the untrained reasoning channel; gold rows
only, fold rule as elsewhere. Without the key every path computes bit-identically to M3.

Further experimental variants (not part of the final lines): "read first" (field `reading`), pool rows with teacher
targets (`sft_pool`) and rationalisation targets (`rationale_gen`); each is described in its section below.

Row-level outputs stay under artifacts/runs/ and data/derived/ (gitignored); for test rows only aggregates are logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import common as C
from . import prompts as P
from . import proto as PR

TASK = "proto_sft"
FINDING = "Finding:"
ROOT = "Root:"
NONE_FINDING = "none"
CLOSE = "\n</think>"  # end of the notes (part of the note sequence: same weight as every note token)
HEAD = "\n\n" + FINDING  # between thinking block and finding
RECIPE_FILE = "k12_sft_recipe.json"  # in the merged model folder: system-text checksum, fold, note limit
SFT_INSTRUCTIONS = (PR.INSTRUCTIONS.split(PR.JSON_PART, 1)[0]
                    + "\n\nReason first. Then write one line of the form `Finding: <what the student did at the root error and what "
                      "it should have been, or none>` and finish with one last line of the form `Root: <taxonomy code or NONE>`.")
# Question knowledge a proto_sft job must not carry (as proto_gen / proto_rank, plus teacher rows in training)
NO_TASK_KEYS = ("teacher_labels", "teacher_train", "topic_codes", "question_codes", "tax_cases", "tax_cases_for", "candidates",
                "analysis_from", "reading_from", "gold_reading", "teacher_examples", "proto_from")


def system_text(tax, style: str = "full") -> str:
    """System text: verdict rules of the protocol without the JSON form, answer form finding + root, taxonomy (header for all rows)."""
    return SFT_INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(tax, style)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def refuse_task_knowledge(job: dict) -> None:
    bad = [k for k in NO_TASK_KEYS if job.get(k)]
    if bad:
        raise ValueError(f"proto_sft has no question knowledge; not allowed: {', '.join(bad)}")


# ---------------------------------------------------------------------------- Examples variant: examples of the same question
#
# Examples variant (job key `sft_examples`) = M3 with labelled workings of OTHER students for the same question in the
# user prompt, the way the untrained reasoning channel sees them (prompts.build_sample, task think): content from
# prompts.example_lines (cleaned transcription, answer check, root code, annotator explanation), header line verbatim as
# there, position as there: after the key, before the photo, in the same text part (number and order of the parts stay).
# Gold rows from train.csv only: no teacher rows, no topic or question block; the channel stays strictly inductive. It is
# the ONLY question information a proto_sft job may carry (refuse_task_knowledge still applies to everything else).
#
# Fold rule (runner.builder_for, data.SampleBuilder.examples_text: "examples for a training row never come from its own
# CV fold"): a model that has NOT learned fold k sees only rows with fold != k; a training row r additionally never
# sees rows of its own fold (no leave-one-out target leakage).
#     fold-k model, training row r (fold != k):    examples from folds not in {k, fold(r)}
#     fold-k model, validation row (fold k):        folds != k
#     full fit (k = None), training row r:          folds != fold(r)
#     test and pool rows:                           folds != k; for the full fit all 1,500 training rows
# k comes from the job during training (`fold`), and during generation and ranking from the model's recipe
# (k12_sft_recipe.json), which carries the setting (`examples`); check_recipe aborts on a different setting.
# `dropout` (training only): on each visit of a training row the whole block is dropped with this probability, fixed
# per (seed, row, visit), independent of the order of the visits.

EXAMPLES_HEAD = "Diagnosed working of other students on this same question:\n"  # verbatim as in prompts.build_sample (reasoning channel)
PHOTO_LINE = "Student's handwritten working (photo):"  # end of the text part with the key in proto.user_parts
EXAMPLES_DEFAULTS = {"max_examples": 4, "max_chars": 260, "dropout": 0.0, "seed": 0}
EXAMPLES_PROMPT_KEYS = ("source", "rule", "head", "max_examples", "max_chars")  # determine the prompt (recipe check)
# Keys with which other tasks control examples; without effect in a proto_sft job and therefore forbidden (a job with
# `max_examples: 4` would otherwise look like the examples variant and silently compute without examples)
FOREIGN_EXAMPLE_KEYS = ("examples", "max_examples", "example_dropout")


def _whole(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"sft_examples.{name}: integer expected")
    return int(value)


def examples_spec(job: dict) -> dict | None:
    """`sft_examples` of a proto_sft job, checked and complete -> None (without examples: M3, bit-identical) or
    {"source": "gold", "rule": "fold", "head", "max_examples", "max_chars", "dropout", "seed"} (defaults as for the
    reasoning channel: 4 examples, transcription up to 260 characters; dropout 0)."""
    bad = [k for k in FOREIGN_EXAMPLE_KEYS if k in job]
    if bad:
        raise ValueError("proto_sft: examples only via sft_examples; not allowed: " + ", ".join(bad))
    raw = job.get("sft_examples")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("sft_examples: expected {max_examples, max_chars, dropout, seed}")
    unknown = sorted(set(raw) - set(EXAMPLES_DEFAULTS))
    if unknown:
        raise ValueError(f"sft_examples: unknown keys {unknown}")
    sp = {**EXAMPLES_DEFAULTS, **raw}
    drop = sp["dropout"]
    if isinstance(drop, bool) or not isinstance(drop, (int, float, np.floating)) or not 0.0 <= float(drop) < 1.0:
        raise ValueError("sft_examples.dropout: number in [0, 1)")
    spec = {"source": "gold", "rule": "fold", "head": EXAMPLES_HEAD, "max_examples": _whole(sp["max_examples"], "max_examples"),
            "max_chars": _whole(sp["max_chars"], "max_chars"), "dropout": float(drop), "seed": _whole(sp["seed"], "seed")}
    if spec["max_examples"] < 1 or spec["max_chars"] < 1 or spec["seed"] < 0:
        raise ValueError("sft_examples: max_examples and max_chars at least 1, seed not negative")
    return spec


def examples_prompt(spec: dict | None) -> dict | None:
    """The part of an example setting that determines the prompt (without dropout and seed, which act only in training)."""
    return None if spec is None else {k: spec.get(k) for k in EXAMPLES_PROMPT_KEYS}


def examples_block(examples: str) -> str:
    """Block in the user prompt as for the reasoning channel (prompts.build_sample): header line, examples, blank line."""
    return EXAMPLES_HEAD + examples + "\n\n"


def user_parts(row, student_image, question_image=None, examples: str = "") -> list[tuple[str, object]]:
    """User message of the protocol (proto.user_parts). With examples, their block stands after the key and before the
    photo as for the reasoning channel, inside the text part of the key; number and order of the parts stay. Without
    examples unchanged."""
    parts = PR.user_parts(row, student_image, question_image)
    if not examples:
        return parts
    key = f"Answer key: {row['correct_answer']}\n\n"
    hit = [j for j, (kind, value) in enumerate(parts) if kind == "text" and value == key + PHOTO_LINE]
    if len(hit) != 1:
        raise ValueError("sft_examples: the user message of the protocol does not have the expected form (proto.user_parts changed?)")
    parts[hit[0]] = ("text", key + examples_block(examples) + PHOTO_LINE)
    return parts


class ExampleSource:
    """Examples of the same question for the examples variant, by the fold rule above. train: train.csv (C.load_table,
    gold rows only), folds: configs/folds.parquet (bound to row positions), spec: examples_spec(...), model_fold: the
    fold the model has NOT learned (None = full fit). Selection and text as in data.SampleBuilder.examples_text
    (prompts.example_lines)."""

    def __init__(self, train, folds, spec: dict, model_fold=None):
        if spec is None:
            raise ValueError("ExampleSource without sft_examples")
        folds = np.asarray(folds, dtype=np.int64)
        if len(train) != len(folds):
            raise ValueError("ExampleSource: folds do not match the table")
        if "root_code" not in train.columns or "_source" in train.columns:
            raise ValueError("ExampleSource: only labelled gold rows from train.csv (no teacher rows)")
        self.train = train.reset_index(drop=True)
        self.folds, self.spec = folds, dict(spec)
        self.model_fold = None if model_fold is None else int(model_fold)
        self._trace = self.train["trace_id"].astype(str).to_numpy()
        self._by_q = {str(q): np.asarray(ix, dtype=np.int64) for q, ix in self.train.groupby("question_id").indices.items()}
        self._visits: dict[int, int] = {}
        self.stats = {"visits": 0, "with_block": 0, "dropped": 0, "no_examples": 0}

    def check(self, spec: dict | None, model_fold) -> None:
        """Does this source belong to the job (spec) and the model (fold from its recipe)?"""
        if examples_prompt(self.spec) != examples_prompt(spec):
            raise ValueError("sft_examples: setting of the example source does not match the job")
        if self.model_fold != (None if model_fold is None else int(model_fold)):
            raise ValueError(f"sft_examples: example source for fold {self.model_fold}, model without fold {model_fold}")

    def rows_for(self, row, split: str, pos: int | None = None) -> np.ndarray:
        """Positions in train.csv from which the row's examples may come (before the selection in example_lines)."""
        idx = self._by_q.get(str(row["question_id"]), np.zeros(0, dtype=np.int64))
        keep = self._trace[idx] != str(row["trace_id"])
        if self.model_fold is not None:
            keep &= self.folds[idx] != self.model_fold
        if split == "train":
            if pos is None or not 0 <= int(pos) < len(self._trace) or self._trace[int(pos)] != str(row["trace_id"]):
                raise ValueError("sft_examples: training row without a matching position in train.csv")
            keep &= self.folds[idx] != self.folds[int(pos)]
        elif split not in ("test", "unlabelled"):
            raise ValueError(f"sft_examples: unknown table {split}")
        return idx[keep]

    def text(self, row, split: str, pos: int | None = None) -> str:
        """Example text of the row (prompts.example_lines); empty if the question has no allowed row."""
        return P.example_lines(self.train.iloc[self.rows_for(row, split, pos)], int(self.spec["max_examples"]), int(self.spec["max_chars"]))

    def drop(self, pos: int, visit: int) -> bool:
        """Is the block dropped on this visit of the training row? Fixed per (seed, row, visit)."""
        p = float(self.spec.get("dropout", 0.0))
        return p > 0.0 and bool(np.random.default_rng([int(self.spec.get("seed", 0)), int(pos), int(visit)]).random() < p)

    def train_text(self, row, pos: int) -> str:
        """Example text for one training visit (counts the visits per row, applies dropout)."""
        visit = self._visits[int(pos)] = self._visits.get(int(pos), 0) + 1
        text = self.text(row, "train", pos)
        self.stats["visits"] += 1
        if not text:
            self.stats["no_examples"] += 1
            return ""
        if self.drop(pos, visit):
            self.stats["dropped"] += 1
            return ""
        self.stats["with_block"] += 1
        return text


def blocks_digest(texts: dict[int, str]) -> str:
    """Checksum of the example texts of several rows (position -> text), independent of the order."""
    return sha256_text(json.dumps([[int(i), sha256_text(texts[i])] for i in sorted(texts)]))


def load_example_hashes(path) -> dict[int, str | None]:
    """Checksum of the example text per row from the output of an sft_gen job (<tag>_notes.jsonl); None if the
    generation ran without examples."""
    out: dict[int, str | None] = {}
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            r = json.loads(line)
            out[int(r["row"])] = r.get("examples_sha256")
    return out


def load_prompt_tokens(notes_path) -> dict[int, int]:
    """Input tokens per row as counted by the vLLM server during generation (<tag>_raw.jsonl next to <tag>_notes.jsonl,
    field in_tokens; 0 on the in-house path). Empty if the file does not exist. For the fidelity check of the ranking."""
    p = Path(notes_path)
    raw = p.with_name(p.name[: -len("_notes.jsonl")] + "_raw.jsonl") if p.name.endswith("_notes.jsonl") else None
    out: dict[int, int] = {}
    if raw is None or not raw.is_file():
        return out
    for line in raw.read_text().split("\n"):
        if line.strip():
            r = json.loads(line)
            out[int(r["row"])] = int(r.get("in_tokens", 0) or 0)
    return out


# ---------------------------------------------------------------------------- Training targets


def finding_text(code: str, misconceptions: str) -> str:
    """Finding line of the annotators: FULL explanation of the root finding (whitespace normalised); "none" for NONE."""
    if code == "NONE":
        return NONE_FINDING
    text = P.root_reason(misconceptions, max_chars=10**9)
    if not text:
        raise ValueError("error row without an explanation of the root finding (no finding with root = true)")
    return text


def cut_notes(text: str, tok, max_tokens: int) -> tuple[str, int, bool]:
    """Notes truncated to at most `max_tokens` tokens in the model's tokenisation, cut at the END (the way a smaller
    length limit would have produced them). -> (text, tokens, truncated)."""
    text = str(text).strip()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    if n <= int(max_tokens):
        return text, n, False
    end = int(enc["offset_mapping"][int(max_tokens) - 1][1])
    return text[:end].rstrip(), int(max_tokens), True


# ---------------------------------------------------------------------------- "Read first" variant
#
# "Read first" variant (target field `reading`) = M3 that copies the working BEFORE its notes: thinking block =
# READ_HEAD + cleaned gold transcription (transcription.strip_leaks; lines "L<n> | …", descriptive bracket annotations
# stay) + blank line + notes. EVERY row learns the transcription (marker read, weight w_read, averaged over its tokens),
# including rows without notes_loss: there the notes follow as context without loss (marker ctx) instead of in the
# prompt, because the transcription precedes them. Notes, finding and code as in M3. Generation and ranking stay
# unchanged: the transcription is the beginning of the thinking block (rank_prefix reads the same string). Without the
# field `reading` every path computes bit-identically to M3.
READ_HEAD = "Transcription of the student's working:\n"
READ_END = "\n\n"


def reading_block(reading: str) -> str:
    """Transcription as the beginning of the thinking block: header line, transcription lines, blank line."""
    text = str(reading or "").strip()
    if not text or READ_END in text or TEMPLATE_MARKS.search(text):
        raise ValueError("transcription (reading) empty, with a blank line or with markers of the chat template")
    return READ_HEAD + text + READ_END


def split_reading(notes: str) -> tuple[str | None, str]:
    """Thinking block of a "read first" answer -> (transcription or None, remaining notes). If the end of the
    transcription is missing (length limit), everything after the header line is transcription."""
    text = str(notes or "").lstrip()
    if not text.startswith(READ_HEAD):
        return None, str(notes or "").strip()
    body = text[len(READ_HEAD):]
    j = body.find(READ_END)
    if j < 0:
        return body.strip(), ""
    return body[:j].strip(), body[j + len(READ_END):].strip()


def reading_target(reading: str, notes: str, notes_loss: bool) -> list[tuple[str, str]]:
    """Beginning of the training target up to and including "Finding:" ("read first"): transcription (read), then the
    notes WITH loss (notes_loss) or as context without loss (ctx). Together with the rest (finding, root, code) this
    gives the same string as rank_prefix(transcription + notes, finding) + code."""
    block, notes = reading_block(reading), str(notes).strip()
    if notes_loss:
        if not notes:
            raise ValueError("notes_loss without notes")
        return [(block, "read"), (notes, "notes"), (CLOSE, "notes"), (HEAD, "aux")]
    if notes:
        return [(block, "read"), (notes, "ctx"), (CLOSE + HEAD, "ctx")]
    return [(block.rstrip(), "read"), (CLOSE + HEAD, "ctx")]  # empty notes: as in rank_prefix (notes are truncated)


def train_sample(row, student_image, question_image, system: str, rec: dict, notes: str, examples: str = "") -> P.Sample:
    """Sample with training target. rec: {"finding": explanation or "none", "code": gold code, "notes_loss": bool}.

    notes_loss: the own note stands WITH loss in the target (marker notes, including the closing of the thinking block);
    otherwise it stands in the prompt as a closed thinking block and only finding and code are learned.
    examples: example text of the same question (examples variant, ExampleSource.train_text); empty = M3."""
    parts = user_parts(row, student_image, question_image, examples)
    meta = {"task": TASK, "think": True, "system": system, "static": system, "joint_target": True}
    finding, code = str(rec["finding"]).strip(), str(rec["code"])
    if not finding:
        raise ValueError("empty finding line")
    tail = [(" " + finding, "why"), ("\n" + ROOT, "aux"), (" " + code, "code")]
    notes = str(notes).strip()
    if rec.get("reading") is not None:  # "read first": transcription first (above); without the field unchanged as in M3
        return P.Sample(key=str(row["trace_id"]), parts=parts, prefix="", meta=meta,
                        target=reading_target(rec["reading"], notes, bool(rec.get("notes_loss"))) + tail)
    if rec.get("notes_loss"):
        if not notes:
            raise ValueError("notes_loss without notes")
        return P.Sample(key=str(row["trace_id"]), parts=parts, prefix="", meta=meta,
                        target=[(notes, "notes"), (CLOSE, "notes"), (HEAD, "aux")] + tail)
    return P.Sample(key=str(row["trace_id"]), parts=parts, prefix=notes + CLOSE + HEAD, meta=meta, target=tail)


def load_targets(path, fold, folds: np.ndarray, n_train: int) -> tuple[list[dict], dict]:
    """Load and check the training targets of one fold (frozen target files): training rows only, no row of the fold
    `fold` (fold None = full fit on all rows), checksum as in the meta file, no row twice."""
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if meta and meta.get("sha256") != sha256_file(path):
        raise ValueError(f"{path.name}: checksum does not match the meta file (training targets changed?)")
    want = None if fold is None else int(fold)
    if meta and meta.get("fold") != want:
        raise ValueError(f"{path.name}: training targets for fold {meta.get('fold')}, requested {want}")
    recs, seen = [], set()
    for line in path.read_text().split("\n"):  # not splitlines(): U+2028 and similar characters occur raw in the texts
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["row"])
        if r.get("table") != "train" or not 0 <= i < n_train:
            raise ValueError(f"{path.name}: only rows of table train allowed")
        if i in seen:
            raise ValueError(f"{path.name}: a row appears twice in the file")
        if want is not None and int(folds[i]) == want:
            raise ValueError(f"{path.name}: a row of the held-out fold {want} appears in the training targets (leak)")
        if int(r.get("fold", folds[i])) != int(folds[i]):
            raise ValueError(f"{path.name}: fold of a row does not match configs/folds.parquet")
        if not str(r.get("finding", "")).strip() or not str(r.get("code", "")):
            raise ValueError(f"{path.name}: finding or code missing")
        if "reading" in r:  # "read first": check the transcription (raises on an empty transcription, blank line or chat-template markers)
            if not isinstance(r["reading"], str):
                raise ValueError(f"{path.name}: transcription (reading) is not text")
            reading_block(r["reading"])
        seen.add(i)
        recs.append(r)
    if not recs:
        raise ValueError(f"{path.name}: no training targets")
    info = {"file": path.name, "sha256": meta.get("sha256") or sha256_file(path), "rows": len(recs),
            "notes_loss": int(sum(bool(r.get("notes_loss")) for r in recs)),
            "share_none": float(np.mean([r["code"] == "NONE" for r in recs]))}
    n_read = int(sum("reading" in r for r in recs))
    if n_read and n_read != len(recs):
        raise ValueError(f"{path.name}: transcription (reading) only for {n_read} of {len(recs)} rows; all or none")
    if n_read:  # "read first" only; earlier targets keep their statistics unchanged
        info["reading_rows"] = n_read
    return recs, info


# ---------------------------------------------------------------------------- Generation and scoring


def gen_sample(row, student_image, question_image, system: str, examples: str = "") -> P.Sample:
    """Sample for generation (in-house path): thinking mode, the answer starts inside the thinking block. examples: as in train_sample."""
    return P.Sample(key=str(row["trace_id"]), parts=user_parts(row, student_image, question_image, examples), prefix="",
                    meta={"task": TASK, "think": True, "system": system, "static": system})


def vllm_request(row, student_image, question_image, system: str, max_tokens: int, temperature: float = 0.0, seed: int | None = None,
                 template_extra: dict | None = None, examples: str = "", top_p: float | None = None, top_k: int | None = None) -> dict:
    """Request to the local vLLM server (job sft_gen): the same user message as gen_sample / train_sample / rank_sample,
    system text as its own role, thinking mode on. top_p/top_k restrict sampling (Qwen3.8 model card for thinking
    mode: 0.95 / 20); if not given, the request stays as before."""
    body = PR.request_body(PR.to_messages(system, user_parts(row, student_image, question_image, examples)), max_tokens, thinking=True,
                           temperature=temperature, seed=seed, template_extra=template_extra)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if top_k is not None:
        body["top_k"] = int(top_k)  # extra field of the vLLM server
    return body


def rank_prefix(notes: str, finding: str | None) -> str:
    """Canonical answer prefix for scoring, the same string as in the training target. Without a finding it ends after
    "Finding:" (the finding is then generated before scoring)."""
    base = str(notes).strip() + CLOSE + HEAD
    if finding is None or not str(finding).strip():
        return base
    return base + " " + str(finding).strip() + "\n" + ROOT


def rank_sample(row, student_image, question_image, system: str, notes: str, finding: str | None, examples: str = "") -> P.Sample:
    return P.Sample(key=str(row["trace_id"]), parts=user_parts(row, student_image, question_image, examples), prefix=rank_prefix(notes, finding),
                    meta={"task": "proto_rank", "think": True, "system": system, "static": system})


def needs_finding(sample: P.Sample) -> bool:
    return sample.prefix.endswith(HEAD)


def complete_prefix(prefix: str, generated: str) -> str:
    """Append a generated finding (text up to "Root:") to an answer prefix without finding; canonical, with "\\nRoot:" at the end."""
    if not prefix.endswith(HEAD):
        raise ValueError("complete_prefix: answer prefix does not end with \"Finding:\"")
    finding = " ".join(str(generated).split(ROOT, 1)[0].split())
    return prefix + (" " + finding if finding else "") + "\n" + ROOT


def parse_generation(raw: str, classes: list[str] | None = None) -> dict:
    """Raw text of an answer in thinking mode -> {"notes", "closed", "finding", "root", "state"}.

    state: complete (thinking block closed, finding and root line present) | open (length limit or form missed; the
    finding is generated again before scoring) | empty (no notes and no finding)."""
    raw = str(raw)
    closed = "</think>" in raw
    notes, answer = PR.split_reply(raw, thinking=True)
    finding = root = None
    if closed:
        j = answer.find(FINDING)
        k = answer.find(ROOT, j + len(FINDING)) if j >= 0 else -1
        if j >= 0 and k >= 0:
            text = " ".join(answer[j + len(FINDING) : k].split())
            finding = text or None
            words = answer[k + len(ROOT):].split()
            code = words[0].strip("`*.,;:") if words else ""
            root = code if (classes is None or code in classes) else None
    state = "complete" if finding is not None and root is not None else ("empty" if not notes.strip() and not closed else "open")
    return {"notes": notes.strip(), "closed": closed, "finding": finding if state == "complete" else None, "root": root, "state": state}


def load_notes(path) -> tuple[str, dict[int, tuple[str, str | None]]]:
    """Output of an sft_gen job (<tag>_notes.jsonl): -> (table, {position: (notes, finding or None)}).
    Exactly one table; no row twice."""
    tables, out = set(), {}
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        tables.add(str(r["table"]))
        i = int(r["row"])
        if i in out:
            raise ValueError(f"{Path(path).name}: a table position appears twice in the file")
        f = r.get("finding")
        out[i] = (str(r.get("text") or ""), str(f) if f is not None and str(f).strip() else None)
    if len(tables) != 1:
        raise ValueError(f"{Path(path).name}: expected exactly one table, found {sorted(tables)}")
    return tables.pop(), out


def read_recipe(model_dir) -> dict | None:
    f = Path(model_dir) / RECIPE_FILE
    return json.loads(f.read_text()) if f.is_file() else None


def read_adapter_recipe(adapter_dir) -> dict | None:
    """Recipe of an unmerged proto_sft adapter: sft_recipe.json of the training job (next to the folder adapter/)
    or k12_sft_recipe.json in the adapter folder itself."""
    a = Path(adapter_dir)
    for f in (a / RECIPE_FILE, a / "sft_recipe.json", a.parent / "sft_recipe.json", a.parent / RECIPE_FILE):
        if f.is_file():
            return json.loads(f.read_text())
    return None


def recipe_for_engine(engine, model_dir) -> dict | None:
    """Recipe for the compute path: merged model -> model folder; adapter with header WITH adapter -> adapter folder."""
    if engine.has_adapter() and getattr(engine.cfg, "head_with_adapter", False):
        return read_adapter_recipe(engine.cfg.adapter)
    return read_recipe(model_dir)


def check_recipe(recipe: dict | None, system: str, split: str, rows, folds: np.ndarray | None, examples: dict | None = None) -> None:
    """Does the job match the merged model? Same system text as in training; the same example setting (`examples` =
    examples_spec of the job; None = without examples, as M3; a model with examples in its recipe refuses that and
    vice versa); on training rows only rows of the fold the model has NOT learned (otherwise the measurement would be
    in-sample)."""
    if recipe is None:
        raise ValueError(f"proto_sft: model without {RECIPE_FILE}; only with a merged proto_sft model (job merge)")
    if recipe.get("system_sha256") != sha256_text(system):
        raise ValueError("proto_sft: system text differs from training (check the job's taxonomy setting)")
    want, have = examples_prompt(examples), examples_prompt(recipe.get("examples"))
    if want != have:
        raise ValueError(f"proto_sft: example setting of the job (sft_examples {want}) does not match the model's recipe ({have}); "
                         "run generation and ranking with the same setting as in training")
    if split == "train":
        k = recipe.get("fold")
        if k is None:
            raise ValueError("proto_sft: the model has learned all training rows; no measurement on training rows")
        rows = np.asarray(rows, dtype=np.int64)
        if folds is None or not np.all(np.asarray(folds)[rows] == int(k)):
            raise ValueError(f"proto_sft: rows outside fold {k} selected; the model has learned them (rows.folds: [{k}])")


def _finish_reason(n_out: int, limit: int) -> str:
    return "length" if n_out >= limit else "stop"


def generate(job: dict, model_dir: Path, df, rows: np.ndarray, split: str, data_root, tax, classes: list[str], image, out_dir: Path,
             tag: str, log=print, engine=None, folds: np.ndarray | None = None, examples: ExampleSource | None = None) -> dict:
    """Generate notes, finding and root line for `rows` (thinking mode, greedy); interrupted runs resume.

    backend: vllm (default; local server as for proto_gen, only 127.0.0.1) | hf (in-house path, `engine`; only for the
    local functional check). max_tokens (3200): limit for notes and answer together; rows without a complete answer get
    their finding during scoring (probe proto_rank, sft mode). guard_rows (24) / guard_min_complete (0.5): abort if too
    few of the first rows return a complete answer (form missed). max_failed (0.05).
    examples (examples variant): example source (ExampleSource with the fold from the model's recipe), exactly when the
    job carries `sft_examples`; checksum of the example text per row in <tag>_raw.jsonl and <tag>_notes.jsonl (the
    ranking checks against it), over all rows in the generation recipe.
    Writes <tag>_raw.jsonl, <tag>_notes.jsonl (input of scoring, `sft_from`), <tag>_meta.json (aggregates only)."""
    backend = str(job.get("backend", "vllm"))
    if backend not in ("vllm", "hf"):
        raise ValueError(f"sft_gen: unknown backend {backend!r}")
    system = system_text(tax, str(job.get("taxonomy", "full")))
    ex_spec = examples_spec(job)
    model_recipe = read_recipe(model_dir)
    check_recipe(model_recipe, system, split, rows, folds, examples=ex_spec)
    if (ex_spec is None) != (examples is None):
        raise ValueError("sft_gen: sft_examples of the job and the example source do not match")
    texts: dict[int, str] | None = None
    if examples is not None:
        examples.check(ex_spec, model_recipe.get("fold"))
        texts = {int(i): examples.text(df.iloc[int(i)], split, int(i)) for i in rows}

    def ex_of(i: int) -> str:
        return "" if texts is None else texts[int(i)]

    out_dir.mkdir(parents=True, exist_ok=True)
    f_raw, f_notes, f_gen = out_dir / f"{tag}_raw.jsonl", out_dir / f"{tag}_notes.jsonl", out_dir / f"{tag}_generation.json"
    max_tokens = int(job.get("max_tokens", 3200))
    temperature, seed = float(job.get("temperature", 0.0)), job.get("seed")
    top_p, top_k = job.get("top_p"), job.get("top_k")
    settings = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed, "template_kwargs": job.get("template_kwargs")}
    settings.update({k: v for k, v in (("top_p", top_p), ("top_k", top_k)) if v is not None})  # old recipes stay the same
    recipe = {"table": split, "system_sha256": sha256_text(system), "model_dir": str(model_dir), "backend": backend,
              "image": {k: getattr(image, k) for k in ("max_side", "min_side", "question_max_side", "rotate_all")},
              "rows_sha256": hashlib.sha256(np.asarray(sorted(int(i) for i in rows), dtype="<i8").tobytes()).hexdigest(),
              "settings": settings}
    if texts is not None:  # examples variant only: without examples the recipe (and thus its checksum) stays as before
        recipe["examples"] = {**examples_prompt(ex_spec), "model_fold": examples.model_fold, "blocks_sha256": blocks_digest(texts)}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if f_gen.exists() and json.loads(f_gen.read_text()).get("sha256") != fingerprint:
        raise ValueError("sft_gen: recipe changed; use a new tag")
    f_gen.write_text(json.dumps({"sha256": fingerprint, "recipe": recipe}, indent=1))
    done: set[int] = set()
    if f_raw.is_file():
        done = {int(json.loads(line)["row"]) for line in f_raw.read_text().split("\n") if line.strip()}
    todo = [int(i) for i in rows if int(i) not in done]
    if job.get("order_seed") is not None:  # never in file order
        todo = [todo[k] for k in np.random.default_rng(int(job["order_seed"])).permutation(len(todo))]
    stats: dict = {"rows": int(len(rows)), "done_before": int(len(rows) - len(todo)), "backend": backend, "max_tokens": max_tokens}

    def images(i: int):
        return PR.open_images(df.iloc[i], split, data_root, image)

    guard_rows, guard_min = int(job.get("guard_rows", 24)), float(job.get("guard_min_complete", 0.5))
    n_err = n_seen = n_complete = 0
    t0 = time.time()

    def record(i: int, raw: str, end: str, n_in: int, n_out: int) -> dict:
        think, text = PR.split_reply(raw, thinking=True)
        rec = {"row": int(i), "phase": "first", "thinking": think, "text": text, "closed": "</think>" in raw, "end": end,
               "in_tokens": int(n_in), "out_tokens": int(n_out)}
        if texts is not None:
            rec["examples_sha256"] = sha256_text(texts[int(i)])
        return rec

    def check_guard():
        if n_seen == guard_rows and n_complete < guard_min * guard_rows:
            raise RuntimeError(f"sft_gen: only {n_complete} of {n_seen} first rows with a complete answer; form missed "
                               f"(guard_min_complete {guard_min})")

    if todo and backend == "hf":
        if engine is None:
            raise ValueError("sft_gen backend hf requires the in-house path (engine)")
        with open(f_raw, "a") as fr:
            for i in todo:
                student, qimg = images(i)
                s = gen_sample(df.iloc[i], student, qimg, system, examples=ex_of(i))
                ids = engine.generate([s], max_new_tokens=max_tokens, return_ids=True)[0]
                raw = engine.tok.decode(ids, skip_special_tokens=True)
                rec = record(i, raw, _finish_reason(len(ids), max_tokens), 0, len(ids))
                fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fr.flush()
                n_seen += 1
                n_complete += int(parse_generation(raw, classes)["state"] == "complete")
                check_guard()
        stats.update(seconds=round(time.time() - t0, 1), s_per_row=round((time.time() - t0) / max(1, len(todo)), 3), failed=0)
    elif todo:
        from . import think_vllm as TV

        spec = job.get("server", {})
        port = int(spec.get("port", 8000))
        proc = TV.start_server(model_dir, spec, out_dir / f"{tag}_server.log")
        try:
            TV.wait_ready(port, proc, int(spec.get("startup_s", 1500)))
            log(f"vLLM server ready (proto_sft); {len(todo)} rows ({split}), up to {max_tokens} tokens, {job.get('concurrency', 64)} concurrent")
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            timeout = int(job.get("request_timeout_s", 1800))

            def ask(i: int) -> dict:
                err = ""
                for attempt in range(3):
                    try:  # including building the request: an image error is a row error
                        student, qimg = images(i)
                        body = vllm_request(df.iloc[i], student, qimg, system, max_tokens, temperature=temperature,
                                            seed=None if seed is None else int(seed) * 1_000_003 + i, template_extra=job.get("template_kwargs"),
                                            examples=ex_of(i), top_p=top_p, top_k=top_k)
                        out = TV._post(url, body, timeout=timeout)
                        ch = out["choices"][0]
                        usage = out.get("usage", {})
                        return record(i, TV.raw_text(ch), str(ch.get("finish_reason")), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                    except Exception as exc:  # noqa: BLE001 - the row stays open and is counted
                        err = f"{type(exc).__name__}: {exc}"[:200]
                        time.sleep(3 * (attempt + 1))
                return {"row": int(i), "error": err}

            pool = ThreadPoolExecutor(max_workers=int(job.get("concurrency", 64)))
            try:
                with open(f_raw, "a") as fr:
                    for rec in pool.map(ask, todo):
                        n_seen += 1
                        if "error" in rec:
                            n_err += 1
                        else:
                            fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fr.flush()
                            n_complete += int(parse_generation(rec["thinking"] + ("\n</think>\n" if rec["closed"] else "") + rec["text"], classes)["state"] == "complete")
                        check_guard()
                        if n_seen % 100 == 0:
                            log(f"sft_gen: {n_seen} of {len(todo)} rows, {(time.time() - t0) / n_seen:.2f} s per row, complete {n_complete}")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            stats.update(seconds=round(time.time() - t0, 1), s_per_row=round((time.time() - t0) / max(1, len(todo)), 3), failed=n_err)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                proc.kill()

    # input of scoring and statistics: aggregates only
    want = {int(i) for i in rows}
    recs = {}
    if f_raw.is_file():
        for line in f_raw.read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                if int(r["row"]) in want:
                    recs[int(r["row"])] = r
    parsed = {i: parse_generation(r["thinking"] + ("\n</think>\n" if r.get("closed") else "") + r["text"], classes) for i, r in recs.items()}
    if texts is not None and any(r.get("examples_sha256") != sha256_text(texts[i]) for i, r in recs.items()):
        raise ValueError("sft_gen: example text of a row differs from the one used during generation (data or folds changed?)")
    with open(f_notes, "w") as fh:
        for i in sorted(parsed):
            p = parsed[i]
            entry = {"table": split, "row": i, "state": p["state"], "text": p["notes"], "finding": p["finding"]}
            if texts is not None:  # examples variant: the ranking checks its example text against this checksum
                entry["examples_sha256"] = recs[i]["examples_sha256"]
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    states = {s: sum(p["state"] == s for p in parsed.values()) for s in ("complete", "open", "empty")}
    if texts is not None:  # aggregates only
        n_ex = [t.count("\n- working: ") + int(t.startswith("- working: ")) for t in texts.values()]
        stats["examples"] = {**examples_prompt(ex_spec), "model_fold": examples.model_fold, "rows_with_examples": int(sum(n > 0 for n in n_ex)),
                             "examples_per_row_mean": float(np.mean(n_ex)) if n_ex else 0.0,
                             "block_chars_median": int(np.median([len(examples_block(t)) if t else 0 for t in texts.values()])) if texts else 0,
                             "blocks_sha256": blocks_digest(texts)}
    outs = [int(r.get("out_tokens", 0)) for r in recs.values()]
    roots = [p["root"] for p in parsed.values() if p["state"] == "complete"]
    stats.update(answered=len(recs), states=states, closed=int(sum(bool(r.get("closed")) for r in recs.values())),
                 ends={e: sum(r.get("end") == e for r in recs.values()) for e in sorted({str(r.get("end")) for r in recs.values()})},
                 out_tokens_median=int(np.median(outs)) if outs else 0, out_tokens_p90=int(np.percentile(outs, 90)) if outs else 0,
                 in_tokens_median=int(np.median([int(r.get("in_tokens", 0)) for r in recs.values()])) if recs else 0,
                 share_none_root=float(np.mean([c == "NONE" for c in roots])) if roots else 0.0, distinct_roots=len(set(roots)),
                 notes_sha256=sha256_file(f_notes), system_sha256=sha256_text(system))
    (out_dir / f"{tag}_meta.json").write_text(json.dumps({"table": split, **stats}, indent=1))
    if stats.get("failed", 0) > float(job.get("max_failed", 0.05)) * max(1, len(todo)):
        raise RuntimeError(f"sft_gen: {stats['failed']} of {len(todo)} rows without an answer (request errors); job incomplete")
    return stats


# ---------------------------------------------------------------------------- Merging


def merge_lora(base_dir, adapter_dir, out_dir, log=print, link_unchanged: bool = True, recipe: dict | None = None) -> dict:
    """Merge a LoRA adapter (PEFT, standard LoRA) into the weight files of the base model WITHOUT loading the model:
    W' = W + (alpha / r) * B @ A, computed in fp32 and stored in the number format of the file. Weight names and files
    stay as in the base model (vLLM and the in-house path load the folder like the base model); files without an
    adapter weight are linked (space), all other files (config.json, tokenizer, chat template, image preprocessing)
    are copied. Aborts if an adapter weight has no partner in the base model (name matching) or the shapes do not match."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("modules_to_save") or cfg.get("fan_in_fan_out") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("merge_lora: only standard LoRA (no DoRA, no modules_to_save, no rank/alpha_pattern)")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / (r ** 0.5 if cfg.get("use_rslora") else r)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, dict] = {}
    for key, value in state.items():
        if not key.startswith("base_model.model."):
            raise ValueError(f"merge_lora: unexpected key in the adapter: {key[:80]}")
        body = key[len("base_model.model."):]
        for part, side in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
            if body.endswith(part):
                pairs.setdefault(body[: -len(part)] + ".weight", {})[side] = value
                break
        else:
            raise ValueError(f"merge_lora: unexpected key in the adapter: {key[:80]}")
    if not pairs or any(set(ab) != {"A", "B"} for ab in pairs.values()):
        raise ValueError("merge_lora: adapter without complete A/B pairs")
    index_file = base_dir / "model.safetensors.index.json"
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:
        single = base_dir / "model.safetensors"
        with safe_open(str(single), "pt") as fh:
            weight_map = {k: single.name for k in fh.keys()}
    missing = [t for t in pairs if t not in weight_map]
    if missing:
        raise ValueError(f"merge_lora: {len(missing)} adapter weights without a partner in the base model, e.g. {missing[0]}")
    by_shard: dict[str, list[str]] = {}
    for t in pairs:
        by_shard.setdefault(weight_map[t], []).append(t)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    t0, rewritten, linked, merged, delta_rel = time.time(), 0, 0, 0, []
    for shard in sorted(set(weight_map.values())):
        src, dst = base_dir / shard, out_dir / shard
        if shard not in by_shard:
            if link_unchanged:
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            linked += 1
            continue
        with safe_open(str(src), "pt") as fh:
            meta = fh.metadata() or {"format": "pt"}
        tensors = load_file(str(src))
        for t in sorted(by_shard[shard]):
            w, a, b = tensors[t], pairs[t]["A"].float(), pairs[t]["B"].float()
            if w.ndim != 2 or tuple(w.shape) != (b.shape[0], a.shape[1]) or a.shape[0] != b.shape[1]:
                raise ValueError(f"merge_lora: shapes do not match for {t}: W {tuple(w.shape)}, A {tuple(a.shape)}, B {tuple(b.shape)}")
            delta = scale * (b @ a)
            delta_rel.append(float(delta.norm() / max(float(w.float().norm()), 1e-12)))
            tensors[t] = (w.float() + delta).to(w.dtype).contiguous()
            merged += 1
        save_file(tensors, str(dst), metadata=meta)
        rewritten += 1
        del tensors
    for f in base_dir.iterdir():  # configuration, tokenizer, chat template, index … (no weights, no folders)
        if f.is_file() and not f.name.endswith(".safetensors") and not (out_dir / f.name).exists():
            shutil.copy2(f, out_dir / f.name)
    info = {"base_dir": str(base_dir.resolve()), "adapter_dir": str(adapter_dir.resolve()), "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
            "r": r, "lora_alpha": alpha, "scale": scale, "merged_tensors": merged, "shards_rewritten": rewritten, "shards_linked": linked,
            "delta_rel_median": float(np.median(delta_rel)), "delta_rel_max": float(np.max(delta_rel)), "seconds": round(time.time() - t0, 1)}
    (out_dir / "k12_merge.json").write_text(json.dumps(info, indent=1))
    if recipe is not None:
        (out_dir / RECIPE_FILE).write_text(json.dumps(recipe, indent=1))
    log(f"merge: {merged} weights merged in {rewritten} files, {linked} linked, median relative change {info['delta_rel_median']:.2e}, "
        f"{info['seconds']} s")
    return info


# ---------------------------------------------------------------------------- Combining (evaluation)


def combine_rank(runs_dir, run: str, jobs: list[str], tag: str, out_job: str) -> dict:
    """Combine the rankings of several fold jobs (probe proto_rank, one model per fold) into ONE output in the format of
    a bench job; input for eda/a8_proto_oof.py --rank runs/<run>/<out_job>/<tag> (for a subset of the folds with
    --rank-partial). Training rows only; rows must not overlap."""
    base = Path(runs_dir) / run / run
    idx_all, lp_all, done_all, sources = [], [], [], []
    for job in jobs:
        d = base / job
        meta = json.loads((d / f"{tag}_meta.json").read_text())
        if meta.get("table") != "train":
            raise ValueError(f"{job}: no output for training rows")
        idx, lp = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy")
        if not (d / f"{tag}_proto_rank.done.npy").is_file():
            raise ValueError(f"{job}: completeness mask missing (probe not finished)")
        done = np.load(d / f"{tag}_proto_rank.done.npy")
        if lp.shape[0] != len(idx) or done.shape != idx.shape:
            raise ValueError(f"{job}: row list and ranking do not match")
        idx_all.append(idx)
        lp_all.append(lp)
        done_all.append(done)
        sources.append({"job": job, "rows": int(len(idx)), "done": int(done.sum())})
    idx, lp, done = np.concatenate(idx_all), np.concatenate(lp_all), np.concatenate(done_all)
    if len(set(idx.tolist())) != len(idx):
        raise ValueError("rows overlap between the jobs")
    order = np.argsort(idx, kind="stable")
    out = base / out_job
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{tag}_idx.npy", idx[order])
    np.save(out / f"{tag}_proto_rank.npy", lp[order])
    np.save(out / f"{tag}_proto_rank.done.npy", done[order])
    info = {"table": "train", "n": int(len(idx)), "probes": ["proto_rank"], "job": out_job, "combined_from": sources}
    (out / f"{tag}_meta.json").write_text(json.dumps(info, indent=1))
    return info


def compare_channel(run: str, job: str, tag: str = "bench", ref: str = "pr27", comp: Path | None = None) -> dict:
    """Single channel of the diagnostician against an existing channel (default pr27, artifacts/oof_<ref>.npy) on the SAME
    rows: the two harm limits of the pilot at channel level (channel < ref - 0.03, NONE < ref - 0.05) and the reported
    quantities per group. Paired row bootstrap; training rows only, aggregates only. The additional contribution in the
    full recipe of both lines is NOT computed here (eda/a8_proto_oof.py + recipe evaluation)."""
    import labels
    import metric

    from .engine import probs_from_logp

    comp = Path(comp or C.comp_root())
    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    seen_cls = labels.seen_mask(train["root_code"], classes)
    d = comp / "artifacts" / "runs" / run / run / job
    if json.loads((d / f"{tag}_meta.json").read_text()).get("table") != "train":
        raise ValueError(f"{job}: no output for training rows")
    idx, lp, done = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy"), np.load(d / f"{tag}_proto_rank.done.npy")
    exact = done & np.isfinite(lp[:, seen_cls]).all(axis=1)
    if exact.sum() < 2:
        raise ValueError(f"{job}: {int(exact.sum())} rows with an exact ranking (all codes finite); beams too small or run incomplete")
    rows = idx[exact]
    P = probs_from_logp(lp[exact], absent=~seen_cls)
    R = np.load(comp / "artifacts" / f"oof_{ref}.npy")[rows]
    yy = y[rows]
    a, b = metric.map3_rows_from_proba(yy, P), metric.map3_rows_from_proba(yy, R)
    q = train["question_id"].to_numpy()
    other = folds[rows][:, None] != folds[None, :]
    same_q = q[rows][:, None] == q[None, :]
    seen_q = (same_q & other).any(1)
    seen_code = (same_q & other & (y[rows][:, None] == y[None, :])).any(1)
    err = yy != 0
    groups = {"all": np.ones(len(rows), bool), "NONE": ~err, "error": err, "seen": err & seen_code,
              "never seen": err & seen_q & ~seen_code, "new question": err & ~seen_q}
    out: dict = {"run": run, "job": job, "ref": ref, "rows_probe": int(len(idx)), "rows_exact": int(exact.sum()),
                 "folds": sorted({int(f) for f in folds[rows]}),
                 "mass_median": float(np.median(np.exp(np.where(np.isfinite(lp[exact]), lp[exact], -np.inf)).sum(1))) if exact.any() else 0.0,
                 "top1": float(np.mean(P.argmax(1) == yy)), "top1_ref": float(np.mean(R.argmax(1) == yy)),
                 "share_none_top": float(np.mean(P.argmax(1) == 0)), "share_none_top_ref": float(np.mean(R.argmax(1) == 0)), "groups": {}}
    for name, m in groups.items():
        if m.sum() >= 2:
            pb = metric.paired_bootstrap(a[m], b[m])
            out["groups"][name] = {"rows": int(m.sum()), "map3": float(a[m].mean()), "map3_ref": float(b[m].mean()),
                                   "delta": pb["delta"], "sigma": pb["sigma"]}
    g = out["groups"]
    out["harm"] = {"channel_below_ref_minus_0.03": bool(g["all"]["delta"] < -0.03),
                   "none_below_ref_minus_0.05": bool("NONE" in g and g["NONE"]["delta"] < -0.05),
                   "note": "harm limits at the channel level only; an added contribution < −0.007 in both lines is checked separately in the recipe"}
    return out


# ---------------------------------------------------------------------------- Pool rows with teacher targets (experimental)
#
# Experimental variant, not part of the final lines: the reasoning 27B (proto_sft) learns, in addition to the gold
# targets, on pool rows with frozen teacher outputs. The prompt stays free of question knowledge (no labelled row, no
# block); the knowledge about (question, code) pairs ends up in the weights. Such a model is TEACHER-DERIVED and is only
# allowed in the primary line: the job must set `sft_pool.slot: best` explicitly, and the job statistics carry
# `teacher_derived: true`.
#
# Interface of the teacher output per pool row (data/derived/teacher/<plan>/labels.jsonl, tools/teacher_api.py
# --mode reason): row (position in unlabelled.csv), reasoning (visible reasoning), finding (finding line; "none" for
# NONE), root_code, confidence, top3 (first entry = root_code), probabilities (for top3); optional readable,
# consistent, stop_reason. Checked by `teacher_record_check`.

POOL_TABLE = "unlabelled"
NEVER_ROOT = ("NOT-02", "PRO-03")  # never a root
GENERIC = ("ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00")  # catch-all codes (gold: 57 of 994 error rows each)
# markers of the chat template in a training target would become real control tokens when tokenised (as in the export of the protocol notes)
TEMPLATE_MARKS = re.compile(r"</?think>|<\|[^|<>\n]{1,40}\|>")
# Share of correct teacher top codes per confidence level (lower bound -> precision), measured on all training rows
# (teacher model Claude Haiku 5.5 at effort xhigh without notes, user message without labelled rows; 1,499 answers),
# split by whether the own 27B in the protocol (oof_pr27, top over the codes that can be a root) puts the same code
# first. The reasoning mode of the teacher is NOT calibrated by this; the tables are an approximation until a
# calibration on training rows exists.
PRECISION = {
    "error": ((0.0, 0.36), (0.5, 0.47), (0.6, 0.71), (0.75, 0.84)),
    "error_agree": ((0.0, 0.62), (0.5, 0.60), (0.6, 0.76), (0.75, 0.89)),
    "error_disagree": ((0.0, 0.23), (0.5, 0.34), (0.6, 0.56), (0.75, 0.52)),
    "none": ((0.0, 0.81), (0.75, 0.96)),
    "none_agree": ((0.0, 0.69), (0.6, 0.90), (0.75, 0.98)),
    "none_disagree": ((0.0, 0.54),),
}
# selection of the pool training targets per fold (select_pool_rows)
POOL_SELECT_DEFAULTS = {
    "none_share": 1 / 3,       # share of NONE among the pool rows (train and test: about one third)
    "cap_pair_new": 8,         # per (question, code) pair the fold's training part does NOT know; breadth before depth
    "cap_pair_seen": 4,        # per pair the training part already knows
    "cap_none_question": 4,    # NONE rows per question
    "cap_generic_share": 0.06,  # per catch-all code at most this share of the selected error rows (gold: 5.7 %)
    "cap_specific": 150,       # per specific code
    "min_conf_new": 0.0,       # new pairs: confidence alone does not decide, the estimated precision does (min_precision)
    "min_conf_seen": 0.6,
    "min_conf_none": 0.75,
    "min_precision": 0.5,      # estimated precision (PRECISION) below this: the row is dropped
    "weight": 0.7,             # weight of a pool row = weight x estimated precision (x discount); gold rows 1.0
    "disagree_old_factor": 0.5,  # discount if the earlier teacher draw (pool_a/pool_b) named a different code
    "n_pool": 0,               # upper bound of pool rows per fold (0 = all that the rules allow)
    "seed": 0,
}
POOL_JOB_KEYS = ("targets", "slot", "gold_epochs", "pool_epochs", "chunks", "start_chunk", "seed", "pair_adjacent",
                 "notes_max_tokens", "style_fold_ok")


def teacher_record_check(r: dict, classes) -> tuple[dict | None, str]:
    """Check one teacher output (interface above) -> (cleaned record or None, reason). Reasons: ok | row |
    readable | code | never_root | top3 | probabilities | confidence | finding | inconsistent | reasoning | stop | marks."""
    try:
        row = int(r["row"])
    except (KeyError, TypeError, ValueError):
        return None, "row"
    if str(r.get("readable") or "yes").strip().lower() == "no":
        return None, "readable"
    code = str(r.get("root_code") or "")
    if code not in classes:
        return None, "code"
    if code in NEVER_ROOT:
        return None, "never_root"
    top3 = [str(c) for c in (r.get("top3") or [])]
    if not top3 or top3[0] != code or len(set(top3)) != len(top3) or any(c not in classes for c in top3):
        return None, "top3"
    try:
        probs = [float(p) for p in (r.get("probabilities") or [])]
    except (TypeError, ValueError):
        return None, "probabilities"
    if len(probs) != len(top3) or not all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in probs) or sum(probs) > 1.0 + 1e-6:
        return None, "probabilities"
    conf = r.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not np.isfinite(conf) or not 0.0 <= float(conf) <= 1.0:
        return None, "confidence"
    finding = " ".join(str(r.get("finding") or "").split())
    if not finding:
        return None, "finding"
    if (finding.strip(" .").lower() == NONE_FINDING) != (code == "NONE") or r.get("consistent") is False:
        return None, "inconsistent"
    reasoning = str(r.get("reasoning") or "").strip()
    if not reasoning:
        return None, "reasoning"
    if str(r.get("stop_reason") or "end_turn") not in ("end_turn", "stop_sequence"):
        return None, "stop"
    if TEMPLATE_MARKS.search(reasoning) or TEMPLATE_MARKS.search(finding):
        return None, "marks"
    return {"row": row, "code": code, "conf": float(conf), "top3": top3, "probabilities": probs,
            "finding": NONE_FINDING if code == "NONE" else finding, "reasoning": reasoning}, "ok"


def precision_of(conf: float, kind: str, table: dict | None = None) -> float:
    """Estimated precision of a teacher top code (PRECISION): kind error | none, with _agree / _disagree when the 27B's
    own answer for the row is available. A missing confidence counts as the lowest level."""
    steps = (table or PRECISION)[kind]
    c = float(conf) if conf is not None and np.isfinite(conf) else 0.0
    out = steps[0][1]
    for lo, p in steps:
        if c >= lo:
            out = p
    return float(out)


def choose_pool_notes(mode: str, reasoning: str, own_text: str | None, own_code: str | None, code: str):
    """Notes of a pool row per mode -> (notes, notes_loss, source) or None (the row is dropped).

    teacher  the teacher's reasoning as a note WITH loss.
    own      only the 27B's own notes (proto_gen on the pool row): with loss if the own answer names the teacher's code,
             otherwise as a closed thinking block in the prompt without loss (like the gold rows without notes_loss);
             without own notes the row is dropped. No teacher text as a long target (only finding line and code).
    mixed    own notes with loss where the own answer agrees; otherwise the teacher's reasoning with loss.
    Without teacher reasoning (rows from earlier teacher labels, an option of the target builder) a row stays
    only if own notes exist (own) or the own answer agrees (mixed)."""
    own_ok = bool(own_text and str(own_text).strip())
    agree = own_ok and own_code == code
    teacher = str(reasoning or "").strip()
    if mode == "teacher":
        return (teacher, True, "teacher") if teacher else None
    if mode == "own":
        return (str(own_text).strip(), bool(agree), "own") if own_ok else None
    if mode == "mixed":
        if agree:
            return str(own_text).strip(), True, "own"
        return (teacher, True, "teacher") if teacher else None
    raise ValueError(f"unknown note mode {mode!r} (teacher | own | mixed)")


def select_pool_rows(cand, gold_pairs: set, known_codes: set, spec: dict | None = None, priority_questions: set | None = None,
                     precision_table: dict | None = None):
    """Select and weight the pool training targets of ONE fold -> (selected rows, statistics).

    cand: one row per checked teacher output with the columns row, question_id, code, conf, ok (bool) and optionally
    own_code (top code of the own 27B, "" = no answer), old_code (earlier teacher draw, "" = none) and precision_fixed
    (fixed precision of this row, NaN = from the table).
    gold_pairs / known_codes: (question, code) pairs or codes of the TRAINING PART of this fold (folds != k; full fit:
    all rows), so the choice depends on no label of the evaluated fold (in deployment: on no test label).
    priority_questions: questions with priority (test relevance; keys only, as in select_by_question).

    Rules (POOL_SELECT_DEFAULTS): status per pair new / seen / NONE; estimated precision from PRECISION (matched
    against the own answer if available); rows below min_precision are dropped; order new before seen, breadth before
    depth (first the best row of every pair), priority questions, weight; caps per pair, per catch-all code (share) and
    per specific code; NONE rows last, first from questions with selected error rows (hard negatives: the same question
    with and without error), at most cap_none_question per question, until their share reaches none_share.
    Weight = weight x precision (x disagree_old_factor if the earlier teacher draw named a different code).
    precision_table: table in the format of PRECISION (e.g. estimated per fold from the training part only); otherwise PRECISION."""
    import pandas as pd

    sp = {**POOL_SELECT_DEFAULTS, **(spec or {})}
    unknown = set(sp) - set(POOL_SELECT_DEFAULTS)
    if unknown:
        raise ValueError(f"select_pool_rows: unknown keys {sorted(unknown)}")
    if not 0.0 <= float(sp["none_share"]) < 1.0:
        raise ValueError("none_share must lie in [0, 1)")
    rng = np.random.default_rng(int(sp["seed"]))
    c = cand.reset_index(drop=True).copy()
    for col, default in (("own_code", ""), ("old_code", "")):
        c[col] = c[col].fillna("").astype(str) if col in c else default
    c["code"] = c["code"].astype(str)
    c["question_id"] = c["question_id"].astype(str)
    none = (c["code"] == "NONE").to_numpy()
    seen = np.array([(q, k) in gold_pairs for q, k in zip(c["question_id"], c["code"])], dtype=bool) & ~none
    c["status"] = np.where(none, "none", np.where(seen, "seen", "new"))
    has_own = (c["own_code"] != "").to_numpy()
    agree = has_own & (c["own_code"] == c["code"]).to_numpy()
    base = np.where(none, "none", "error")
    c["kind"] = np.where(has_own, np.char.add(base.astype(str), np.where(agree, "_agree", "_disagree")), base)
    conf = pd.to_numeric(c["conf"], errors="coerce").to_numpy(dtype=float)
    c["precision"] = [precision_of(x, k, precision_table) for x, k in zip(conf, c["kind"])]
    if "precision_fixed" in c:  # e.g. rows with a second opinion: precision from their measurement instead of from the confidence
        fixed = pd.to_numeric(c["precision_fixed"], errors="coerce").to_numpy(dtype=float)
        c["precision"] = np.where(np.isfinite(fixed), fixed, c["precision"].to_numpy(dtype=float))
    old_diff = ((c["old_code"] != "") & (c["old_code"] != c["code"])).to_numpy()
    c["weight"] = float(sp["weight"]) * c["precision"].to_numpy() * np.where(old_diff, float(sp["disagree_old_factor"]), 1.0)
    c["tie"] = rng.random(len(c))
    stats: dict = {"candidates": int(len(c)), "dropped": {}}
    keep = c["ok"].astype(bool).to_numpy()
    stats["dropped"]["check"] = int((~keep).sum())
    for name, m in (("code_unknown_in_fold", none | c["code"].isin(known_codes).to_numpy()),
                    ("never_root", ~c["code"].isin(NEVER_ROOT).to_numpy()),
                    ("confidence", np.nan_to_num(conf, nan=-1.0) >= np.where(none, sp["min_conf_none"], np.where(seen, sp["min_conf_seen"], sp["min_conf_new"]))),
                    ("precision", c["precision"].to_numpy() >= float(sp["min_precision"]))):
        stats["dropped"][name] = int((keep & ~m).sum())
        keep &= m
    c = c[keep]
    e = c[c["status"] != "none"].copy()
    e["tier"] = (e["status"] == "seen").astype(int)
    e = e.sort_values(["question_id", "code", "weight", "tie"], ascending=[True, True, False, True])
    e["rank_pair"] = e.groupby(["question_id", "code"]).cumcount()
    e = e[e["rank_pair"] < np.where(e["status"] == "new", int(sp["cap_pair_new"]), int(sp["cap_pair_seen"]))]
    e["prio_q"] = e["question_id"].isin(priority_questions or set())
    e = e.sort_values(["tier", "rank_pair", "prio_q", "weight", "tie"], ascending=[True, True, False, False, True])
    if int(sp["n_pool"]):
        e = e.head(_half_up(int(sp["n_pool"]) * (1.0 - float(sp["none_share"]))))
    # caps per code along the order: specific codes fixed, catch-all codes as a share of the selected error rows (fixed point)
    generic = e["code"].isin(GENERIC).to_numpy()
    e["rank_code"] = e.groupby("code").cumcount().to_numpy()
    spec_keep = ~generic & (e["rank_code"].to_numpy() < int(sp["cap_specific"]))
    n_spec, cap_gen = int(spec_keep.sum()), 0
    avail = e[generic].groupby("code").size()
    for _ in range(50):
        total = n_spec + int(np.minimum(avail.to_numpy(), cap_gen).sum())
        new_cap = int(np.floor(float(sp["cap_generic_share"]) * total)) if total else 0
        if new_cap == cap_gen:
            break
        cap_gen = new_cap
    e = e[spec_keep | (generic & (e["rank_code"].to_numpy() < max(cap_gen, 0)))]
    n_err = len(e)
    share = float(sp["none_share"])
    n_none = _half_up(n_err * share / (1.0 - share)) if share > 0 else 0
    nn = c[c["status"] == "none"].copy()
    nn["hard"] = nn["question_id"].isin(set(e["question_id"]))
    nn["prio_q"] = nn["question_id"].isin(priority_questions or set())
    nn = nn.sort_values(["hard", "prio_q", "weight", "tie"], ascending=[False, False, False, True])
    nn = nn[nn.groupby("question_id").cumcount().to_numpy() < int(sp["cap_none_question"])].head(n_none)
    sel = pd.concat([e.drop(columns=["tier", "rank_pair", "prio_q", "rank_code"]), nn.drop(columns=["hard", "prio_q"])], ignore_index=True)
    sel = sel.drop(columns=["tie"]).sort_values("row").reset_index(drop=True)
    err = sel["status"] != "none"
    codes = sel.loc[err, "code"]
    stats.update(rows=int(len(sel)), errors=int(err.sum()), none=int((~err).sum()), share_none=float((~err).mean()) if len(sel) else 0.0,
                 rows_new=int((sel["status"] == "new").sum()), rows_seen=int((sel["status"] == "seen").sum()),
                 pairs_new=int(sel.loc[sel["status"] == "new", ["question_id", "code"]].drop_duplicates().shape[0]),
                 pairs_seen=int(sel.loc[sel["status"] == "seen", ["question_id", "code"]].drop_duplicates().shape[0]),
                 questions=int(sel["question_id"].nunique()), codes=int(codes.nunique()),
                 generic_share=float(codes.isin(GENERIC).mean()) if len(codes) else 0.0,
                 max_rows_per_code=int(codes.value_counts().max()) if len(codes) else 0, cap_generic=int(cap_gen),
                 none_short=int(max(0, n_none - int((~err).sum()))), weight_mean=float(sel["weight"].mean()) if len(sel) else 0.0,
                 precision_mean_errors=float(sel.loc[err, "precision"].mean()) if err.any() else 0.0,
                 kinds={str(k): int(v) for k, v in sel["kind"].value_counts().items()}, spec=sp)
    return sel, stats


def _half_up(x: float) -> int:
    """Round half up, robust against floating-point residues (11 x (1/3) / (2/3) is 5.4999…, meaning 5.5)."""
    return int(np.floor(float(x) + 0.5 + 1e-9))


def pair_coverage(selected, pairs) -> float:
    """Share of the (question, code) pairs in `pairs` (e.g. the "never seen" error rows of a fold) for which at least one
    selected pool error row with the same pair exists. Aggregate only."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    have = set(zip(selected.loc[selected["status"] != "none", "question_id"].astype(str), selected.loc[selected["status"] != "none", "code"].astype(str)))
    return float(np.mean([(str(q), str(k)) in have for q, k in pairs]))


def _negative_blocks(visits: list[int], question, code, rng) -> list[list[tuple[str, int]]]:
    """Pool visits of one round in units: two rows of the SAME question with DIFFERENT codes (also NONE versus error)
    side by side, so that they usually land in the same optimiser step (hard negatives); the rest individually."""
    by_q: dict[str, list[int]] = {}
    for j in visits:
        by_q.setdefault(str(question[j]), []).append(int(j))
    units: list[list[tuple[str, int]]] = []
    for q in sorted(by_q):
        items = [by_q[q][t] for t in rng.permutation(len(by_q[q]))]
        pending: list[int] = []
        for j in items:
            mate = next((m for m in pending if code[m] != code[j]), None)
            if mate is None:
                pending.append(j)
            else:
                pending.remove(mate)
                units.append([("pool", mate), ("pool", j)])
        units += [[("pool", j)] for j in pending]
    return units


def mix_schedule(n_gold: int, pool_question, pool_code, gold_epochs: int = 2, pool_epochs: int = 2, chunks: int = 4, seed: int = 0,
                 pair_adjacent: bool = True, pool_repeat=None) -> tuple[list[tuple[str, int]], int, int]:
    """Visit plan for gold and pool training targets -> (visits [("gold" | "pool", index)], visits per segment, padded).

    Rounds: max(gold_epochs, pool_epochs). Every gold row comes exactly once in each of the first gold_epochs rounds;
    every pool row pool_epochs x repeat times, spread over different rounds (for pool_epochs < rounds: uniformly at
    random). Shuffled within a round (with pair_adjacent: hard negatives as pairs, _negative_blocks). The plan is split
    into `chunks` segments of equal length (train() saves the adapter after each); missing slots at the end are filled
    with additional gold visits (at most chunks - 1)."""
    rng = np.random.default_rng(int(seed))
    n_pool = len(pool_question)
    reps = np.ones(n_pool, dtype=int) if pool_repeat is None else np.asarray(pool_repeat, dtype=int)
    if n_gold <= 0 or int(gold_epochs) < 1 or int(pool_epochs) < 0 or int(chunks) < 1 or len(pool_code) != n_pool or len(reps) != n_pool or (reps < 1).any():
        raise ValueError("mix_schedule: invalid settings")
    rounds = max(int(gold_epochs), int(pool_epochs))
    start = np.zeros(n_pool, dtype=int)
    start[rng.permutation(n_pool)] = np.arange(n_pool) % rounds
    per_round: list[list[int]] = [[] for _ in range(rounds)]
    for j in range(n_pool):
        for v in range(int(pool_epochs) * int(reps[j])):
            per_round[(start[j] + v) % rounds].append(j)
    schedule: list[tuple[str, int]] = []
    for r in range(rounds):
        units = [[("gold", int(i))] for i in rng.permutation(n_gold)] if r < int(gold_epochs) else []
        if pair_adjacent:
            units += _negative_blocks(per_round[r], pool_question, pool_code, rng)
        else:
            units += [[("pool", int(j))] for j in per_round[r]]
        for u in rng.permutation(len(units)):
            schedule += units[u]
    chunk = -(-len(schedule) // int(chunks))
    pad = chunk * int(chunks) - len(schedule)
    schedule += [("gold", int(i)) for i in rng.choice(n_gold, size=pad, replace=pad > n_gold)]
    return schedule, chunk, pad


def load_pool_targets(path, fold, n_pool: int, style_ok: bool = False) -> tuple[list[dict], dict]:
    """Load and check the pool training targets of one fold (frozen pool target files): only rows of the table
    unlabelled, checksum as in the meta file, fold as requested, no row twice; NO label of the evaluated fold in status
    and caps (`label_folds`) and no teacher style examples from this fold (`teacher.style_folds`; style_ok only if
    the evaluation hides the style rows)."""
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise ValueError(f"{path.name}: meta file missing; pool training targets are only accepted together with their meta file")
    meta = json.loads(meta_path.read_text())
    if meta.get("table") != POOL_TABLE:
        raise ValueError(f"{path.name}: no pool training targets (table {meta.get('table')})")
    if meta.get("sha256") != sha256_file(path):
        raise ValueError(f"{path.name}: checksum does not match the meta file (training targets changed?)")
    want = None if fold is None else int(fold)
    if meta.get("fold", "missing") != want:
        raise ValueError(f"{path.name}: pool training targets for fold {meta.get('fold')}, requested {want}")
    style = (meta.get("teacher") or {}).get("style_folds")
    if style is None:
        raise ValueError(f"{path.name}: origin of the style examples unknown (teacher.style_folds missing)")
    if want is not None and want in {int(s) for s in style} and not style_ok:
        raise ValueError(f"{path.name}: the teacher's style examples come from fold {want}, which cannot be measured honestly there (style_fold_ok only "
                         "with hidden style rows)")
    if want is not None and want in {int(f) for f in meta.get("label_folds", [want])}:
        raise ValueError(f"{path.name}: status and caps use labels of the evaluated fold {want} (label_folds): leak")
    recs, seen = [], set()
    for line in path.read_text().split("\n"):  # not splitlines(): U+2028 and similar characters occur raw in the texts
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["row"])
        if r.get("table") != POOL_TABLE or not 0 <= i < int(n_pool):
            raise ValueError(f"{path.name}: only rows of table {POOL_TABLE} allowed")
        if i in seen:
            raise ValueError(f"{path.name}: a pool row appears twice in the file")
        if not str(r.get("finding", "")).strip() or not str(r.get("code", "")):
            raise ValueError(f"{path.name}: finding or code missing")
        if r.get("notes_loss") and not str(r.get("notes", "")).strip():
            raise ValueError(f"{path.name}: notes_loss without notes")
        w = float(r.get("weight", 0.0))
        if not 0.0 < w <= 2.0 or int(r.get("repeat", 1)) < 1:
            raise ValueError(f"{path.name}: weight outside (0, 2] or repeat < 1")
        seen.add(i)
        recs.append(r)
    if not recs:
        raise ValueError(f"{path.name}: no pool training targets")
    return recs, {"file": path.name, "sha256": meta["sha256"], "rows": len(recs), "fold": want,
                  "share_none": float(np.mean([r["code"] == "NONE" for r in recs])), "notes_loss": int(sum(bool(r.get("notes_loss")) for r in recs)),
                  "notes_source": {s: int(sum(r.get("notes_source") == s for r in recs)) for s in ("own", "teacher")},
                  "weight_mean": float(np.mean([float(r["weight"]) for r in recs])), "teacher": meta.get("teacher"),
                  "label_folds": meta.get("label_folds"), "selection_label_folds": meta.get("selection_label_folds")}


def combine_rank_runs(runs_dir, specs: list[str], tag: str, out_run: str, out_job: str) -> dict:
    """Like combine_rank, but the fold jobs come from DIFFERENT runs (two machines in parallel):
    specs = ["<run>/<job>", …] -> runs/<out_run>/<out_run>/<out_job>/<tag>_* in the format of a bench job
    (input of eda/a8_proto_oof.py --rank). Training rows only; rows must not overlap."""
    idx_all, lp_all, done_all, sources = [], [], [], []
    for spec in specs:
        run, job = str(spec).split("/")
        d = Path(runs_dir) / run / run / job
        meta = json.loads((d / f"{tag}_meta.json").read_text())
        if meta.get("table") != "train":
            raise ValueError(f"{spec}: no output for training rows")
        if not (d / f"{tag}_proto_rank.done.npy").is_file():
            raise ValueError(f"{spec}: completeness mask missing (probe not finished)")
        idx, lp, done = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_proto_rank.npy"), np.load(d / f"{tag}_proto_rank.done.npy")
        if lp.shape[0] != len(idx) or done.shape != idx.shape:
            raise ValueError(f"{spec}: row list and ranking do not match")
        idx_all.append(idx)
        lp_all.append(lp)
        done_all.append(done)
        sources.append({"run": run, "job": job, "rows": int(len(idx)), "done": int(done.sum())})
    idx, lp, done = np.concatenate(idx_all), np.concatenate(lp_all), np.concatenate(done_all)
    if len(set(idx.tolist())) != len(idx):
        raise ValueError("rows overlap between the jobs")
    order = np.argsort(idx, kind="stable")
    out = Path(runs_dir) / out_run / out_run / out_job
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{tag}_idx.npy", idx[order])
    np.save(out / f"{tag}_proto_rank.npy", lp[order])
    np.save(out / f"{tag}_proto_rank.done.npy", done[order])
    info = {"table": "train", "n": int(len(idx)), "probes": ["proto_rank"], "job": out_job, "combined_from": sources}
    (out / f"{tag}_meta.json").write_text(json.dumps(info, indent=1))
    return info


class PoolMix:
    """Visit plan from gold and pool training targets for vlm.train.train().

    train() calls make_sample(i, task) exactly once per step. PoolMix works through its plan (mix_schedule) in this
    order and does not need i. With n_items = visits per segment and epochs = number of (remaining) segments, train()
    covers the plan exactly and saves the adapter after every segment (protection against losing the machine;
    resume with start_chunk and train.init_adapter)."""

    def __init__(self, schedule, chunk: int, chunks: int, gold_fn, pool_fn, start_chunk: int = 0, info: dict | None = None):
        if not 0 <= int(start_chunk) < int(chunks) or len(schedule) != int(chunk) * int(chunks):
            raise ValueError("PoolMix: plan, segment length and number of segments do not match")
        self.schedule, self.chunk, self.chunks, self.start_chunk = list(schedule), int(chunk), int(chunks), int(start_chunk)
        self.gold_fn, self.pool_fn = gold_fn, pool_fn
        self.offset, self.calls = self.start_chunk * self.chunk, 0
        self.served = {"gold": 0, "pool": 0}
        self.info = info or {}

    @property
    def n_items(self) -> int:
        return self.chunk

    @property
    def epochs(self) -> int:
        return self.chunks - self.start_chunk

    def make_sample(self, i: int, task: str):
        k = self.offset + self.calls
        if k >= len(self.schedule):
            raise IndexError("PoolMix: more steps than planned visits")
        self.calls += 1
        src, j = self.schedule[k]
        self.served[src] += 1
        return self.gold_fn(int(j)) if src == "gold" else self.pool_fn(int(j))

    @classmethod
    def from_job(cls, job: dict, gold_targets: list[dict], gold_notes: dict, train_df, data_root, img, system: str, tok, tcfg, pool_file,
                 fold, log=print) -> "PoolMix":
        """From a proto_sft training job with `sft_pool` (keys POOL_JOB_KEYS). gold_targets / gold_notes: the gold targets
        of the job and their truncated notes (as in the job without pool). train.epochs must equal chunks - start_chunk
        (one segment per "epoch" of train())."""
        spec = dict(job.get("sft_pool") or {})
        unknown = set(spec) - set(POOL_JOB_KEYS)
        if unknown:
            raise ValueError(f"sft_pool: unknown keys {sorted(unknown)}")
        if spec.get("slot") != "best":
            raise ValueError("sft_pool: teacher targets in training are allowed only in the primary line; set `slot: best` explicitly")
        pool_df = C.load_table(data_root, POOL_TABLE)
        recs, tinfo = load_pool_targets(pool_file, fold, len(pool_df), style_ok=bool(spec.get("style_fold_ok", False)))
        chunks, start = int(spec.get("chunks", 4)), int(spec.get("start_chunk", 0))
        if not 0 <= start < chunks:
            raise ValueError("sft_pool: start_chunk outside the segments")
        if abs(float(tcfg.epochs) - (chunks - start)) > 1e-9:
            raise ValueError(f"sft_pool: train.epochs ({tcfg.epochs}) must equal chunks - start_chunk ({chunks - start}); one segment per epoch")
        cap = int(spec.get("notes_max_tokens", 2500))
        pnotes, ntok, ncut = [], [], 0
        for r in recs:
            text, n_tok, cut = cut_notes(r["notes"], tok, cap) if str(r.get("notes", "")).strip() else ("", 0, False)
            pnotes.append(text)
            ntok.append(n_tok)
            ncut += int(cut)
        q = pool_df["question_id"].astype(str).to_numpy()
        sched, chunk, pad = mix_schedule(len(gold_targets), [q[int(r["row"])] for r in recs], [str(r["code"]) for r in recs],
                                         int(spec.get("gold_epochs", 2)), int(spec.get("pool_epochs", 2)), chunks, int(spec.get("seed", 0)),
                                         bool(spec.get("pair_adjacent", True)), [int(r.get("repeat", 1)) for r in recs])

        def gold_fn(j: int):
            rec = gold_targets[j]
            row = train_df.iloc[int(rec["row"])]
            student, qimg = PR.open_images(row, "train", data_root, img)
            return train_sample(row, student, qimg, system, rec, gold_notes[int(rec["row"])])

        def pool_fn(j: int):
            rec = recs[j]
            row = pool_df.iloc[int(rec["row"])]
            student, qimg = PR.open_images(row, POOL_TABLE, data_root, img)
            s = train_sample(row, student, qimg, system, rec, pnotes[j])
            s.meta["weight"] = float(rec["weight"])
            s.meta["source"] = POOL_TABLE
            return s

        info = {**tinfo, "teacher_derived": True, "slot": "best", "visits": len(sched), "visits_gold": sum(s == "gold" for s, _ in sched),
                "visits_pool": sum(s == "pool" for s, _ in sched), "padded": pad, "chunk": chunk, "chunks": chunks, "start_chunk": start,
                "gold_epochs": int(spec.get("gold_epochs", 2)), "pool_epochs": int(spec.get("pool_epochs", 2)),
                "pair_adjacent": bool(spec.get("pair_adjacent", True)), "notes_max_tokens": cap, "notes_cut": ncut,
                "notes_tokens": {p: int(np.percentile(ntok, p)) for p in (5, 50, 90, 100)} if ntok else {}}
        log(f"sft_pool: {tinfo['rows']} pool training targets ({tinfo['file']}, NONE {tinfo['share_none']:.3f}, mean weight "
            f"{tinfo['weight_mean']:.3f}, notes own/teacher {tinfo['notes_source']}) + {len(gold_targets)} gold targets -> {len(sched)} visits "
            f"in {chunks} segments of {chunk} (start at segment {start}, padded {pad})")
        return cls(sched, chunk, chunks, gold_fn, pool_fn, start, info)


# ---------------------------------------------------------------------------- Rationalisation variant (STaR, experimental)
#
# Experimental variant, not part of the final lines. Most of the M3 loss lies in the training rows where neither of
# the two pr27 draws ranks the gold code first. Exactly there M3 gets no reasoning path towards gold as a training
# target (the wrong pr27 note stands in the prompt without loss). This variant fills the gap like the rationalisation
# step of STaR: the OWN 27B WITHOUT training writes its notes in the teacher protocol (system text and user message as
# for pr27), but sees after the photo the annotators' marking of THIS row (root code, root finding, other findings)
# with the instruction never to mention it. The notes are checked locally (separate target builder: mentions of the
# hint removed line by line, at most 10 % of the characters; JSON answer or most frequently named code = gold; minimum
# length; gold code not already in the first 10 % of the note; a style gate over all hint rows) and taken into the
# training targets as notes WITH loss.
# Fold honesty: the note of a row depends only on ITS label and on the untrained model; it is a training target of
# every model whose training part contains the row; no label of another fold flows in (unlike texts of trained models).
# The outputs are label-conditioned: they are called <tag>_rationales.jsonl (no loader reads this name), carry
# `label_conditioned: true` and NEVER go anywhere as a channel, as the note source of a ranking or as a test input.

RATIONALE_TABLE = "train"
RATIONALE_HEAD = "\n\nReference marking of this working (confidential: your notes must never mention it):\n"
RATIONALE_RULES = ("\n\nYour notes will be read as a model analysis by teachers who do NOT have this reference marking. Analyse the photo "
                   "exactly as you normally would: read the handwriting line by line, solve the question yourself and compare with the "
                   "answer key. Reach the reference diagnosis through your own observations: find the step where the working goes wrong "
                   "(or confirm that it does not), state what the student did and what it should have been, and explain why the "
                   "reference code fits better than the nearest alternative codes. Never mention the reference marking, that a "
                   "diagnosis was given to you, or these instructions.")
# Mentions of the hint in the notes. The pattern covers the obvious references "the reference …", "marked as/this/it",
# "the given/expected/target code/root/diagnosis/answer", "supposed to be/reach …", "need/have to arrive at/reach/justify",
# "don't/shouldn't/mustn't/can't mention", "were given the code" and echoes of the hint format ("Root error at …",
# "(not the root)"); apostrophes also typographic. Deliberately NOT in the pattern, because it is protocol language
# (proto.INSTRUCTIONS, frequent in pr27 notes without a hint): "minor slips that do not …", "annotat…", "intended
# code/root", "Root misconception:", "The instructions say". Measured: 27 of 30 synthetic references detected (not:
# "The annotation says …", "Root misconception: …", "make sure my analysis supports …"); on the 3,000 pr27 notes WITHOUT
# a hint 174 of 243,312 lines (0.07 %; limit 0.1 %) in 136 notes, so removing lines costs almost nothing there.
RATIONALE_LEAK = re.compile(
    r"reference (?:marking|diagnosis|code|root|answer|solution)|\bthe marking\b|marking says|\bconfidential\b|"
    r"\bhint\b|i(?:['’]m| am) told|we(?:['’]re| are) told|been told|was told|were told|told (?:me|us)\b|"
    r"never mention|not (?:to )?mention|without mentioning|model analysis|teachers who do not|"
    r"diagnosis (?:was|is) given|given diagnosis|provided diagnosis|given (?:root|marking)|provided (?:root|marking)|"
    r"(?:was|were|been) given (?:the|a) (?:diagnosis|marking|code|answer|root)|"
    r"\breference\b|\bmarked (?:as|this|it)\b|\b(?:given|expected|target) (?:code|root|diagnosis|answer)\b|"
    r"\bsupposed to (?:be|reach|arrive at|get to|end up)\b|\b(?:needs?|ha(?:ve|s)) to (?:arrive at|reach|justify)\b|"
    r"\b(?:don|shouldn|mustn|can)['’]t mention\b|\broot error at\b|\(not the root\)", re.I)
RATIONALE_DEFAULTS = {"max_tokens": 2500, "note_chars": 16000, "concurrency": 64, "guard_rows": 24, "guard_min_thinking": 0.5,
                      "max_failed": 0.05, "request_timeout_s": 900}


def _finding_line(f: dict, with_severity: bool = True) -> str:
    step = str(f.get("step") or "").strip() or "?"
    code = str(f.get("category") or "").strip() or "?"
    sev = str(f.get("severity") or "").strip()
    text = " ".join(str(f.get("text") or "").split())
    head = f"{step}, {sev}, {code}" if with_severity and sev and sev.lower() != "nan" else f"{step}, {code}"
    return f"{head}: {text}"


def rationale_hint(code: str, name: str, misconceptions: str) -> str:
    """Hint block of a training row (end of the user message after the photo, proto.user_parts tail): root code with
    name and the annotators' root finding, the other findings (choice of the root), for NONE the minor slips, plus the
    instruction never to mention the hint. Only data of the row itself."""
    m = json.loads(misconceptions) if str(misconceptions or "").strip() else []
    if not isinstance(m, list):
        raise ValueError("rationale_hint: misconceptions is not a list")
    roots = [f for f in m if isinstance(f, dict) and f.get("root")]
    others = [f for f in m if isinstance(f, dict) and not f.get("root")]
    if code == "NONE":
        lines = ["Root: NONE (no major error: the method and the result are correct)."]
        if others:
            lines.append("Minor slips that do not change the method or the result: " + "; ".join(_finding_line(f, False) for f in others))
    else:
        if not roots:
            raise ValueError("rationale_hint: error row without a root finding")
        lines = [f"Root misconception: {code} ({' '.join(str(name).split())}).", "Root error at " + _finding_line(roots[0], False)]
        if others:
            lines.append("Other findings (not the root): " + "; ".join(_finding_line(f) for f in others))
    return RATIONALE_HEAD + "\n".join(lines) + RATIONALE_RULES


def rationale_notes(thinking: str, note_chars: int = 16000) -> str:
    """Notes from the reasoning of a rationalisation record, treated like the pr27 notes (proto.rank_notes and the
    export of the protocol notes): thinking-block markers removed, at most note_chars characters from the end."""
    return str(thinking or "").replace("<think>", "").replace("</think>", "").strip()[-int(note_chars):]


def strip_hint_mentions(text: str, max_share: float = 0.10) -> tuple[str | None, dict]:
    """Remove lines with a mention of the hint (RATIONALE_LEAK). -> (text or None, statistics). None if the removed
    lines carry more than max_share of the characters, nothing is left afterwards or a mention remains."""
    text = str(text or "")
    lines = text.split("\n")
    hit = [bool(RATIONALE_LEAK.search(ln)) for ln in lines]
    removed = sum(len(ln) + 1 for ln, h in zip(lines, hit) if h)
    share = removed / max(1, len(text))
    out = "\n".join(ln for ln, h in zip(lines, hit) if not h).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    info = {"lines_removed": int(sum(hit)), "share_removed": float(share)}
    if share > max_share or not out or RATIONALE_LEAK.search(out):
        return None, info
    return out, info


def lead_code(notes: str, classes: list[str]) -> str | None:
    """The code of the class list named most often in the notes (on ties the last one named); None without a mention.
    Codes only as whole words, longer ones first (no partial match, no pattern matching outside the class list)."""
    codes = sorted({str(c) for c in classes}, key=len, reverse=True)
    pat = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(c) for c in codes) + r")(?![A-Za-z0-9])")
    hits = pat.findall(str(notes or ""))
    if not hits:
        return None
    count: dict[str, int] = {}
    for h in hits:
        count[h] = count.get(h, 0) + 1
    top = max(count.values())
    return next(h for h in reversed(hits) if count[h] == top)


def rationale_consistent(notes: str, answer: str, gold: str, classes: list[str]) -> str:
    """Does the note reach the marking? -> "" (yes) or the reason. With a valid JSON answer its root_code counts; without
    an answer (token limit) the code named most often in the notes must be the gold code (lead_code), as measured on
    the truncated pr27 notes: 78 % for notes leading to gold, 9 % for the others ("named anywhere" separated weaker:
    85 % versus 29 %)."""
    parsed = PR.parse_answer(answer, classes) if str(answer or "").strip() else None
    if parsed is not None:
        return "" if parsed["root_code"] == gold else "answer_differs"
    lead = lead_code(notes, classes)
    if lead is None:
        return "no_code_mentioned"
    return "" if lead == gold else "lead_code_differs"


def load_rationale_hints(path, n_train: int) -> dict[int, str]:
    """Hint file (separate target builder, --hints): {"table": "train", "row", "tail"} per row -> {position: block}.
    Checksum as in the meta file; training rows only; no row twice; every block in the form of rationale_hint."""
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if meta and meta.get("sha256") != sha256_file(path):
        raise ValueError(f"{path.name}: checksum does not match the meta file (hints changed?)")
    out: dict[int, str] = {}
    for line in path.read_text().split("\n"):  # not splitlines(): U+2028 and similar characters occur raw in the texts
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["row"])
        if r.get("table") != RATIONALE_TABLE or not 0 <= i < n_train:
            raise ValueError(f"{path.name}: hints only for training rows")
        if i in out:
            raise ValueError(f"{path.name}: a row appears twice in the file")
        tail = str(r.get("tail") or "")
        if not (tail.startswith(RATIONALE_HEAD) and tail.endswith(RATIONALE_RULES)):
            raise ValueError(f"{path.name}: hint block of a row does not have the form of rationale_hint")
        out[i] = tail
    if not out:
        raise ValueError(f"{path.name}: no hints")
    return out


def load_rationales(path) -> dict[int, dict]:
    """Output of a rationale_gen job (<tag>_rationales.jsonl) -> {position: record}. Training rows only, each once,
    every record marked as label-conditioned."""
    out: dict[int, dict] = {}
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("table") != RATIONALE_TABLE or r.get("label_conditioned") is not True:
            raise ValueError(f"{Path(path).name}: no label-conditioned record for a training row")
        i = int(r["row"])
        if i in out:
            raise ValueError(f"{Path(path).name}: a row appears twice in the file")
        out[i] = r
    return out


def generate_rationales(job: dict, model_dir: Path, df, rows: np.ndarray, data_root, tax, image, out_dir: Path, tag: str,
                        tails: dict[int, str], log=print, engine=None) -> dict:
    """Generate rationalised notes for `rows` (training rows with a hint) in the teacher protocol as for pr27 (system
    text proto.system_text, user message proto.user_parts with the hint block at the end, thinking mode, greedy, ONE
    limit for reasoning and answer as in the first pr27 pass); no finish call (only the reasoning is needed).

    backend vllm (default; local server, only 127.0.0.1) | hf (in-house path `engine`, only local functional check).
    Interrupted runs resume (the same recipe is required). Guard: of the first guard_rows rows at most max_failed
    request errors and at least guard_min_thinking with reasoning. Writes <tag>_rationales.jsonl (per row reasoning,
    answer text, end, tokens, label_conditioned), <tag>_rationales_recipe.json and <tag>_rationales_meta.json
    (aggregates only)."""
    spec_job = {**RATIONALE_DEFAULTS, **job}
    backend = str(job.get("backend", "vllm"))
    if backend not in ("vllm", "hf"):
        raise ValueError(f"rationale_gen: unknown backend {backend!r}")
    rows = np.asarray(sorted(int(i) for i in rows), dtype=np.int64)
    missing = [int(i) for i in rows if int(i) not in tails]
    if missing:
        raise ValueError(f"rationale_gen: {len(missing)} rows without a hint")
    system = PR.system_text(tax)
    max_tokens = int(spec_job["max_tokens"])
    temperature, seed = float(job.get("temperature", 0.0)), job.get("seed")
    top_p, top_k, effort = job.get("top_p"), job.get("top_k"), str(job.get("think_effort", ""))
    settings = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed, "think_effort": effort,
                "template_kwargs": job.get("template_kwargs"), "top_p": top_p, "top_k": top_k}
    recipe = {"table": RATIONALE_TABLE, "system_sha256": sha256_text(system), "model_dir": str(model_dir), "backend": backend,
              "image": {k: getattr(image, k) for k in ("max_side", "min_side", "question_max_side", "rotate_all")},
              "rows_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
              "hints_sha256": blocks_digest({int(i): tails[int(i)] for i in rows}), "settings": settings, "label_conditioned": True}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    f_out, f_rec, f_meta = (out_dir / f"{tag}_rationales{s}" for s in (".jsonl", "_recipe.json", "_meta.json"))
    if f_rec.exists() and json.loads(f_rec.read_text()).get("sha256") != fingerprint:
        raise ValueError("rationale_gen: recipe changed; use a new tag")
    f_rec.write_text(json.dumps({"sha256": fingerprint, "recipe": recipe}, indent=1))
    have = load_rationales(f_out) if f_out.is_file() else {}
    todo = [int(i) for i in rows if int(i) not in have]
    if job.get("order_seed") is not None:  # never in file order
        todo = [todo[k] for k in np.random.default_rng(int(job["order_seed"])).permutation(len(todo))]
    stats: dict = {"rows": int(len(rows)), "done_before": int(len(rows) - len(todo)), "backend": backend, "max_tokens": max_tokens}
    guard_rows, guard_think = int(spec_job["guard_rows"]), float(spec_job["guard_min_thinking"])
    max_failed = float(spec_job["max_failed"])
    n_seen = n_err = n_think = 0
    t0 = time.time()

    def record(i: int, raw: str, end: str, n_in: int, n_out: int) -> dict:
        think, text = PR.split_reply(raw, thinking=True)
        return {"table": RATIONALE_TABLE, "row": int(i), "thinking": think, "answer": text, "closed": "</think>" in raw, "end": str(end),
                "in_tokens": int(n_in), "out_tokens": int(n_out), "tail_sha256": sha256_text(tails[int(i)]), "label_conditioned": True}

    def parts_of(i: int):
        row = df.iloc[int(i)]
        student, qimg = PR.open_images(row, RATIONALE_TABLE, data_root, image)
        return PR.user_parts(row, student, qimg, tails[int(i)])

    def check_guard():
        if n_seen == guard_rows and (n_err > max_failed * guard_rows or n_think < guard_think * guard_rows):
            raise RuntimeError(f"rationale_gen: of the first {n_seen} rows {n_err} request errors and only {n_think} with reasoning; "
                               f"form missed (max_failed {max_failed}, guard_min_thinking {guard_think})")

    def keep(rec: dict, fh) -> None:
        nonlocal n_seen, n_err, n_think
        n_seen += 1
        if "error" in rec:
            n_err += 1
        else:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            have[int(rec["row"])] = rec
            n_think += int(bool(str(rec["thinking"]).strip()))
        check_guard()

    if todo and backend == "hf":
        if engine is None:
            raise ValueError("rationale_gen backend hf requires the in-house path (engine)")
        with open(f_out, "a") as fh:
            for i in todo:
                s = P.Sample(key=str(df.iloc[i]["trace_id"]), parts=parts_of(i), prefix="",
                             meta={"task": "proto", "think": True, "system": system, "static": system})
                ids = engine.generate([s], max_new_tokens=max_tokens, return_ids=True)[0]
                raw = engine.tok.decode(ids, skip_special_tokens=True)
                keep(record(i, raw, _finish_reason(len(ids), max_tokens), 0, len(ids)), fh)
        stats.update(seconds=round(time.time() - t0, 1), s_per_row=round((time.time() - t0) / max(1, len(todo)), 3), failed=0)
    elif todo:
        from . import think_vllm as TV

        spec = job.get("server", {})
        port = int(spec.get("port", 8000))
        proc = TV.start_server(model_dir, spec, out_dir / f"{tag}_server.log")
        try:
            TV.wait_ready(port, proc, int(spec.get("startup_s", 1500)))
            log(f"vLLM server ready (rationale_gen); {len(todo)} training rows with a hint, up to {max_tokens} tokens, "
                f"{spec_job['concurrency']} concurrent")
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            timeout = int(spec_job["request_timeout_s"])

            def ask(i: int) -> dict:
                err = ""
                for attempt in range(3):
                    try:  # including building the request: an image error is a row error
                        body = PR.request_body(PR.to_messages(system, parts_of(i)), max_tokens, thinking=True, effort=effort,
                                               temperature=temperature, seed=None if seed is None else int(seed) * 1_000_003 + i,
                                               template_extra=job.get("template_kwargs"))
                        if top_p is not None:
                            body["top_p"] = float(top_p)
                        if top_k is not None:
                            body["top_k"] = int(top_k)
                        out = TV._post(url, body, timeout=timeout)
                        ch = out["choices"][0]
                        usage = out.get("usage", {})
                        return record(i, TV.raw_text(ch), str(ch.get("finish_reason")), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                    except Exception as exc:  # noqa: BLE001 - the row stays open and is counted
                        err = f"{type(exc).__name__}: {exc}"[:200]
                        time.sleep(3 * (attempt + 1))
                return {"row": int(i), "error": err}

            pool = ThreadPoolExecutor(max_workers=int(spec_job["concurrency"]))
            try:
                with open(f_out, "a") as fh:
                    for rec in pool.map(ask, todo):
                        keep(rec, fh)
                        if n_seen % 100 == 0:
                            log(f"rationale_gen: {n_seen} of {len(todo)} rows, {(time.time() - t0) / n_seen:.2f} s per row")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            stats.update(seconds=round(time.time() - t0, 1), s_per_row=round((time.time() - t0) / max(1, len(todo)), 3), failed=n_err)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                proc.kill()
    got = [have[int(i)] for i in rows if int(i) in have]
    outs = [int(r.get("out_tokens", 0)) for r in got]
    stats.update(answered=len(got), closed=int(sum(bool(r.get("closed")) for r in got)),
                 ends={e: sum(str(r.get("end")) == e for r in got) for e in sorted({str(r.get("end")) for r in got})},
                 with_thinking=int(sum(bool(str(r.get("thinking", "")).strip()) for r in got)),
                 out_tokens_median=int(np.median(outs)) if outs else 0, out_tokens_p90=int(np.percentile(outs, 90)) if outs else 0,
                 in_tokens_median=int(np.median([int(r.get("in_tokens", 0)) for r in got])) if got else 0,
                 system_sha256=sha256_text(system), hints_sha256=recipe["hints_sha256"], label_conditioned=True,
                 rationales_sha256=sha256_file(f_out) if f_out.is_file() else None)
    f_meta.write_text(json.dumps({"table": RATIONALE_TABLE, **stats}, indent=1))
    if stats.get("failed", 0) > max_failed * max(1, len(todo)):
        raise RuntimeError(f"rationale_gen: {stats['failed']} of {len(todo)} rows without an answer (request errors); job incomplete")
    return stats


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="proto_sft: combine the rankings of several fold jobs and check them as a single channel (evaluation)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--run", required=True)
    c.add_argument("--jobs", nargs="+", required=True)
    c.add_argument("--tag", default="bench")
    c.add_argument("--out", required=True, help="name of the combined job under artifacts/runs/<run>/<run>/")
    k = sub.add_parser("compare")
    k.add_argument("--run", required=True)
    k.add_argument("--job", required=True)
    k.add_argument("--tag", default="bench")
    k.add_argument("--ref", default="pr27")
    a = ap.parse_args()
    if a.cmd == "combine":
        info = combine_rank(C.comp_root() / "artifacts" / "runs", a.run, a.jobs, a.tag, a.out)
    else:
        info = compare_channel(a.run, a.job, a.tag, a.ref)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
