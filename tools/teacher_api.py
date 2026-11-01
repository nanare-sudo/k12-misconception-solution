#!/usr/bin/env python3
"""Teacher via the Anthropic API: label training and pool workings (development only) — NEVER test rows.

    python tools/teacher_api.py bake   --model claude-sonnet-5-5 [--effort high] [--bare] [--no-notes] [--limit 120]
    python tools/teacher_api.py plan   --name pool_a --per-question 40 [--rank-from 0] [--seed 7] [--exclude <earlier plans> ...]
    python tools/teacher_api.py plan   --name train_all --table train      (all training rows: a measurement, not a training target)
    python tools/teacher_api.py submit --name pool_a --model claude-haiku-5-5 [--effort high] [--notes]
    python tools/teacher_api.py collect --name pool_a [--wait]
    python tools/teacher_api.py ledger

bake     measures a teacher on the 120 training workings of the teacher measurement (the packets must have been built) —
         individual calls, result in data/derived/teacher_eval/api_<model>[_bare|_pure]_0.jsonl.
         --bare: without the topic block and without examples of the same task; --no-notes: additionally without the
         annotators' notes in the taxonomy header ("pure": the teacher then sees NO labelled row, not even indirectly).
         Rows without a usable answer are stored with an empty ranking and score 0 in the evaluation.
plan     selects pool workings and writes data/derived/teacher/<name>/plan.json. Selection as in the vision-model code
         (`runner.select_by_question`): one fixed random order of all pool rows, per task the ranks
         [--rank-from, --per-question). Same seed -> the same rows as a run with `rows: {per_question, seed}`, and later
         batches (higher ranks) extend earlier ones without overlapping them.
submit   sends the workings of a plan in batch mode (half price) — always WITHOUT task-related examples and without the
         topic block, and by default also WITHOUT the annotators' notes in the header: this way the teacher sees no labelled
         row and the pool labels carry nothing about a validation fold (the notes act like hidden examples of the same
         task — measured with them 0.84, without them 0.76, the difference lies entirely in the group "code seen for the
         task"). --notes adds the notes of all training rows; such labels are NOT honest for CV measurements.
collect  fetches finished batches, checks the answers and writes data/derived/teacher/<name>/labels.jsonl with a checksum.

Reasoning mode (additive — without --mode everything stays as above):
    python tools/teacher_api.py plan   --name pool_sft_probe60 --rows-from derived/pool_sft/pool_sel_v1_probe60.npy
    python tools/teacher_api.py submit --name pool_sft_probe60 --model claude-haiku-5-5 --effort xhigh --mode reason --max-tokens 24000
    python tools/teacher_api.py collect --name pool_sft_probe60 --wait
plan --rows-from  takes the pool positions from an .npy (relative to data/ or absolute; e.g. a selection of pool rows)
         instead of the ranks per task; pool rows only, --exclude and --limit apply as usual.
submit --mode reason  same teacher, same judging rules, same user message (NO labelled row, no topic block, no notes), but
         the answer is visible text: (1) reasoning (300–700 words), (2) "Finding: …" in the annotators' style, (3) JSON
         (without transcription and list of findings). For the style of the finding line up to five anonymised explanations
         of training rows FROM FOLD 0 are put in the header (--style, default data/derived/pool_sft/style_v1.json; the tool
         checks fold and checksum). The labels therefore depend on no label of folds 1–4 — a measurement on folds 1+2 stays
         honest, one on fold 0 does not. The header (checksum in batches.json) belongs to the plan; --notes is forbidden in
         this mode.
collect  recognises the mode from the plan and additionally writes per row `reasoning`, `finding`, word counts, the codes
         mentioned in the text, `stop_reason`, `output_tokens` and the raw text `text`; answers without usable JSON go to
         unparsed.jsonl.
reparse  repeats the parsing of a reasoning plan with the current parser (without the API) and rewrites labels.jsonl.

Rules enforced by the tool:
  * It reads only train.csv and unlabelled.csv and opens only images from images/train, images/unlabelled and
    images/questions. There is no way to send a test row.
  * The key is stored as ANTHROPIC_API_KEY in .env and is never printed.
  * Every call is booked with token counts and cost in artifacts/teacher_ledger.json; nothing is sent beyond
    TEACHER_CAP_USD. Outputs live only under data/ (not versioned); script and checksums are versioned.
The pool labels used by the solution (plans pool_a and pool_b) were produced in label mode without notes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import prompts as P  # noqa: E402

API = "https://api.anthropic.com/v1"
TEACHER_CAP_USD = 150.0  # upper limit of all API spending of this tool; raised only deliberately
LEDGER = COMP / "artifacts" / "teacher_ledger.json"
OUT = COMP / "data" / "derived" / "teacher"
EVAL = COMP / "data" / "derived" / "teacher_eval"
KEY = COMP / "artifacts" / "eda" / "private" / "teacher_eval_key.json"
# USD per million tokens: input, output, cache read, cache write (1 h); according to the provider's price list
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 8.0), "claude-sonnet-5-5": (2.0, 10.0, 0.10, 4.0),
          "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.20)}
ALLOWED_IMAGE_DIRS = ("train", "unlabelled", "questions")

INSTRUCTIONS = """You are an experienced mathematics teacher and marker. You diagnose a photo of handwritten working by an Indonesian student (grades 4-12; question and student writing are in Indonesian) and name the ROOT misconception from the taxonomy below.

How to judge:
- Read the handwriting line by line. Solve the question yourself and compare with the answer key.
- Decide whether the student's FINAL ANSWER matches the answer key.
- The root misconception is the MAJOR error from which the rest of the working follows (usually the earliest major error). A major error changes the method or the result. If the working is correct, or has only minor slips that do not change method or result, the root is NONE. Across all workings roughly one third are NONE.
- Use the code that markers would use: prefer a topic-specific code when one fits; generic codes (CAL-01, ANS-01, WOP-01, FRM-01, OTH-00) only when no specific code fits.

Answer with ONE JSON object and nothing else, in exactly this shape:
{"readable": "yes, partly or no",
 "final_answer_correct": true or false,
 "student_final_answer": "<the student's final answer exactly as written; empty string if there is none>",
 "transcription": ["L1 | <first line of the student's working>", "L2 | <next line>", "..."],
 "findings": [{"step": "L2", "code": "<taxonomy code>", "severity": "major or minor", "root": true or false, "text": "<what went wrong there and what it should have been>"}],
 "root_code": "<taxonomy code or NONE>",
 "top3": ["<best code or NONE>", "<second>", "<third>"],
 "probabilities": [<probability that the first entry of top3 is the markers' root code>, <same for the second>, <same for the third>],
 "confidence": <number between 0 and 1 for the first entry of top3>}
Rules for the JSON: `findings` is an empty list when there is no error; exactly one finding has "root": true unless root_code is NONE; `top3` holds three DIFFERENT entries, the first equals root_code; `probabilities` holds three numbers that sum to at most 1; use only codes that appear in the taxonomy, spelled exactly, or NONE. Transcribe what the student wrote; put your own remarks in [square brackets]. `readable` is "no" when the photo shows no usable working for this question."""

# ---- reasoning mode: the same judging rules as INSTRUCTIONS, but visible reasoning + finding line + JSON
JSON_PART = "\n\nAnswer with ONE JSON object"
REASON_FORMAT = """Write your answer as plain text in three parts, in this order (no headings, no markdown):

1. Your reasoning, written out for another marker who will learn from it (300 to 700 words). Work step by step: say what the student wrote and computed in each line (call the lines L1, L2, ...); solve the question yourself and say whether the student's final answer matches the answer key; find the first major error from which the rest of the working follows, or explain why there is none (correct working, or only minor slips). Then name the two to four taxonomy codes that could describe this error, always weighing the topic-specific candidates against the generic codes (CAL-01, ANS-01, WOP-01, FRM-01, OTH-00), give the reason for and against each, and say which code you choose and why. Do not state the chosen code before you have weighed the candidates.

2. One line that starts with `Finding: ` and says what the student did at the root error and what it should have been, written the way markers write their notes: one or two sentences, about 25 to 70 words, with the concrete numbers from this working, without any code. When the root is NONE, write exactly `Finding: none`.

3. ONE JSON object as the very last thing in your answer, in exactly this shape:
{"readable": "yes, partly or no",
 "final_answer_correct": true or false,
 "student_final_answer": "<the student's final answer exactly as written; empty string if there is none>",
 "root_code": "<taxonomy code or NONE>",
 "top3": ["<best code or NONE>", "<second>", "<third>"],
 "probabilities": [<probability that the first entry of top3 is the markers' root code>, <same for the second>, <same for the third>],
 "confidence": <number between 0 and 1 for the first entry of top3>}
Rules for the JSON: `root_code` is the code you chose in your reasoning; `top3` holds three DIFFERENT entries, the first equals root_code; `probabilities` holds three numbers that sum to at most 1; use only codes that appear in the taxonomy, spelled exactly, or NONE. `readable` is "no" when the photo shows no usable working for this question."""
STYLE_HEAD = ("How markers word a finding — examples from OTHER questions and other students (numbers replaced by placeholders [n1], [n2], ...). "
              "They show the wording only: they are unrelated to the working you mark, must not influence which code you choose, and in "
              "your own Finding you write the actual numbers of this working:")
STYLE_DEFAULT = COMP / "data" / "derived" / "pool_sft" / "style_v1.json"
# finding line: "Finding: …", also with an enumeration in front ("2. Finding: …" — the teacher answered like this in the probe
# for 23 of 60 rows because the instruction numbers its parts), with markdown or quotation marks
ENUM = r"(?:\(?\d+[.)]\)?[ \t]*|part[ \t]*\d+[ \t]*[—:.-]?[ \t]*)"
FIND_RE = re.compile(r"^[ \t>*_#-]*" + ENUM + r"?\**[ \t]*finding[ \t]*\**[ \t]*[:：][ \t]*\**[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
LABEL_LINE = re.compile(r"^\s*\**\s*(?:(?:\(?\d+[.)]\)?|part\s*\d+)(?:\s*[—:.-]?\s*(?:reasoning|finding(?:\s+line)?|(?:the\s+)?json(?:\s+object)?))?"
                        r"|reasoning|finding(?:\s+line)?|(?:the\s+)?json(?:\s+object)?)\s*\**\s*[:.)—-]?\s*\**\s*$", re.IGNORECASE)


def api_key() -> str:
    for line in (COMP / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    raise SystemExit("ANTHROPIC_API_KEY missing in .env — add it there.")


def env_value(name: str) -> str:
    """Value of a line NAME=… from .env (empty if it is missing). Values are never printed."""
    for line in (COMP / ".env").read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def headers() -> dict:
    """Headers of every request. If the key is not bound to a workspace, the API additionally requires
    `anthropic-workspace-id`; the identifier is configured as ANTHROPIC_WORKSPACE_ID in .env. The tool does not look it up
    itself (the account's administration endpoints are outside its scope)."""
    h = {"x-api-key": api_key(), "anthropic-version": "2023-06-01", "content-type": "application/json"}
    ws = env_value("ANTHROPIC_WORKSPACE_ID")
    if ws:
        h["anthropic-workspace-id"] = ws
    return h


def http(method: str, url: str, body: dict | None = None, timeout: int = 600):
    # Allowed are the Messages endpoints and the result file of a batch on the same host — nothing else.
    if not url.startswith("https://api.anthropic.com/v1/messages"):
        raise SystemExit("This tool only talks to the Messages endpoints.")
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(), headers=headers())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise RuntimeError(f"API {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(min(60, 5 * 2**attempt))


def ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.is_file() else {"cap_usd": TEACHER_CAP_USD, "entries": []}


def spent(led: dict) -> float:
    return float(sum(e.get("usd", 0.0) for e in led["entries"]))


def cost(model: str, usage: dict, batch: bool) -> float:
    p_in, p_out, p_read, p_write = PRICES[model]
    usd = (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out
           + usage.get("cache_read_input_tokens", 0) * p_read + usage.get("cache_creation_input_tokens", 0) * p_write) / 1e6
    return usd * (0.5 if batch else 1.0)


def book(kind: str, model: str, rows: int, usage: dict, batch: bool, note: str = "") -> float:
    led = ledger()
    usd = cost(model, usage, batch)
    led["entries"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "model": model, "rows": rows,
                           "batch": batch, "usage": usage, "usd": round(usd, 4), "note": note})
    LEDGER.write_text(json.dumps(led, indent=1))
    return usd


def check_budget(estimate: float) -> None:
    led = ledger()
    if spent(led) + estimate > led["cap_usd"]:
        raise SystemExit(f"aborted: so far {spent(led):.2f} USD + estimated {estimate:.2f} USD > cap {led['cap_usd']:.2f} USD.")


def image_block(path: Path) -> dict:
    """Image as a base64 block — only from the allowed image folders, never from images/test."""
    if path.parent.name not in ALLOWED_IMAGE_DIRS and EVAL not in path.parents:
        raise SystemExit(f"image folder {path.parent.name!r} is not allowed (only {ALLOWED_IMAGE_DIRS}).")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(path.read_bytes()).decode()}}


def system_blocks(root, notes_from=None) -> list[dict]:
    """The same header for all rows (instructions, taxonomy) — cached by the provider.

    notes_from: labelled rows from which up to three annotator notes per code go into the header; None (default) means: none —
    the teacher then sees no labelled row."""
    tax = C.load_taxonomy(root)
    cases = None if notes_from is None else P.code_cases(notes_from, 3)
    text = INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(tax, "full", cases=cases)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def user_content(root, row, table: str, extra: str = "") -> list[dict]:
    if table not in ("train", "unlabelled"):
        raise SystemExit("Only training and pool rows may be sent to a teacher.")
    parts: list[dict] = [{"type": "text", "text": f"Grade: {row['grade']} | Topic: {row['topic_name']}\nQuestion:\n{row['question_text']}\n"}]
    if row["question_image_file"]:
        parts += [{"type": "text", "text": "Question diagram:"}, image_block(C.image_path(root, "questions", row["question_image_file"]))]
    parts.append({"type": "text", "text": f"Answer key: {row['correct_answer']}\n{extra}\nStudent's handwritten working (photo):"})
    parts.append(image_block(C.image_path(root, table, row["image_file"])))
    return parts


def params(model: str, effort: str, system: list[dict], content: list[dict], max_tokens: int = 12000) -> dict:
    # max_tokens caps thinking AND answer; if it is too tight, the JSON answer is cut off (end "max_tokens")
    return {"model": model, "max_tokens": max_tokens, "system": system, "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort}, "messages": [{"role": "user", "content": content}]}


def parse(message: dict, classes: list[str]) -> dict | None:
    """First JSON answer of the message, checked; None if it is not usable."""
    text = "".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text")
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        return None
    valid = set(classes)
    top = [c for c in obj.get("top3", []) if isinstance(c, str) and c in valid]
    top = list(dict.fromkeys(top))
    if not top or obj.get("root_code") not in valid:
        return None
    obj["top3"] = top[:3]
    obj["findings"] = [f for f in obj.get("findings", []) if isinstance(f, dict) and f.get("code") in valid]
    probs = obj.get("probabilities")
    obj["probabilities"] = [float(v) for v in probs[:3]] if isinstance(probs, list) and all(isinstance(v, (int, float)) for v in probs[:3]) else None
    obj["student_final_answer"] = str(obj["student_final_answer"])[:200] if obj.get("student_final_answer") is not None else None
    obj["readable"] = str(obj["readable"]).strip().lower()[:10] if obj.get("readable") is not None else None
    return obj


# ---------------------------------------------------------------------------------------------- reasoning mode


def load_style(path: Path) -> dict:
    """Style examples of the finding line — only from fold 0, the checksum must match."""
    rec = json.loads(Path(path).read_text())
    examples = [str(e) for e in rec.get("examples", [])]
    if not 0 < len(examples) <= 5:
        raise SystemExit(f"style examples: {len(examples)} instead of 1 to 5")
    if hashlib.sha256(json.dumps(examples, ensure_ascii=False).encode()).hexdigest() != rec.get("sha256"):
        raise SystemExit("style examples: checksum does not match — regenerate the file.")
    folds = C.load_folds()
    pos = [int(p) for p in rec.get("source_positions", [])]
    if len(pos) != len(examples) or any(not 0 <= p < len(folds) or folds[p] != 0 for p in pos):
        raise SystemExit("style examples: source not entirely from fold 0 — a measurement on folds 1+2 would not stay honest.")
    return {"examples": examples, "sha256": rec["sha256"], "source_positions": pos}


def reason_system_blocks(root, style_path: Path | None = STYLE_DEFAULT) -> list[dict]:
    """Header of the reasoning mode: judging rules as in INSTRUCTIONS (without its JSON form), answer form REASON_FORMAT, style
    examples from fold 0, taxonomy without notes. The same for all rows and cached by the provider."""
    tax = C.load_taxonomy(root)
    text = INSTRUCTIONS.split(JSON_PART, 1)[0] + "\n\n" + REASON_FORMAT
    if style_path is not None:
        style = load_style(style_path)
        text += "\n\n" + STYLE_HEAD + "\n" + "\n".join(f"- {e}" for e in style["examples"])
    text += "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(tax, "full", cases=None)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def _strip_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"```[a-zA-Z]*\s*$", "", s).strip()
    return re.sub(r"^\s*```[a-zA-Z]*", "", s).strip()


def _last_json(text: str) -> tuple[int, dict] | None:
    """Last JSON object with root_code in the text -> (start position, object). Brackets in the reasoning (LaTeX, sets) or in
    strings of the JSON do not interfere: every "{" is tried from the back, the first complete decoding wins."""
    dec = json.JSONDecoder()
    for m in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = dec.raw_decode(text, m)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "root_code" in obj:
            return m, obj
    return None


def _clean_lines(text: str) -> str:
    """Remove empty outline lines ("1.", "Part 2:", "Reasoning:", fence lines), keep the rest unchanged."""
    keep = [ln for ln in text.splitlines() if not LABEL_LINE.match(ln) and not re.match(r"^\s*```[a-zA-Z]*\s*$", ln)]
    return "\n".join(keep).strip()


def code_mentions(text: str, classes: list[str]) -> list[str]:
    """Taxonomy codes that appear verbatim in the text (exact spelling, not as part of a longer token)."""
    return [c for c in classes if c != "NONE" and re.search(r"(?<![A-Za-z0-9-])" + re.escape(c) + r"(?![A-Za-z0-9])", text)]


def parse_reason(message: dict, classes: list[str]) -> dict | None:
    """Parse an answer of the reasoning mode: reasoning, finding line, JSON. None if there is no usable JSON; if only the finding
    line is missing, `finding` is None (the row is kept with a flag)."""
    text = "".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text")
    hit = _last_json(text)
    if hit is None:
        return None
    start, obj = hit
    valid = set(classes)
    top = list(dict.fromkeys(c for c in obj.get("top3", []) if isinstance(c, str) and c in valid)) if isinstance(obj.get("top3"), list) else []
    if not top or obj.get("root_code") not in valid:
        return None
    probs = obj.get("probabilities")
    out = {
        "readable": str(obj["readable"]).strip().lower()[:10] if obj.get("readable") is not None else None,
        "final_answer_correct": obj.get("final_answer_correct"),
        "student_final_answer": str(obj["student_final_answer"])[:200] if obj.get("student_final_answer") is not None else None,
        "root_code": obj["root_code"], "top3": top[:3],
        "probabilities": [float(v) for v in probs[:3]] if isinstance(probs, list) and all(isinstance(v, (int, float)) for v in probs[:3]) else None,
        "confidence": obj.get("confidence") if isinstance(obj.get("confidence"), (int, float)) else None,
    }
    out["text"] = text  # the raw text is kept: `reparse` can repeat the parsing later without a new call
    return reason_fields(out, text[:start], classes)


def split_reason_text(pre: str) -> tuple[str, str | None]:
    """Text before the JSON -> (reasoning, finding line or None). Finding line = last "Finding:" line (also "2. Finding:"),
    including continuation lines up to the first empty line; reasoning = everything before it without pure outline lines."""
    pre = _strip_fence(pre)
    found = list(FIND_RE.finditer(pre))
    if not found:
        return _clean_lines(pre), None
    m = found[-1]
    rest = []
    for ln in pre[m.end():].splitlines()[1:]:  # continuation of the finding line up to the first empty line
        if not ln.strip() or LABEL_LINE.match(ln):
            break
        rest.append(ln.strip())
    finding = " ".join([m.group(1).strip()] + rest).strip().strip("*`").strip()
    return _clean_lines(pre[: m.start()]), (finding or None)


def reason_fields(out: dict, pre: str, classes: list[str]) -> dict:
    """Enter reasoning, finding line and flags from the text before the JSON into `out`."""
    reasoning, finding = split_reason_text(pre)
    out["reasoning"] = reasoning
    out["finding"] = finding
    out["reasoning_words"] = len(reasoning.split())
    out["finding_words"] = len(finding.split()) if finding else 0
    out["codes_in_reasoning"] = code_mentions(reasoning, classes)
    out["finding_none"] = bool(finding) and finding.strip().strip(".").lower() == "none"
    out["consistent"] = bool(finding) and (out["finding_none"] == (out["root_code"] == "NONE"))
    return out


def reparse_record(rec: dict, classes: list[str]) -> dict:
    """Repeat the parsing of a stored record with the current parser (no API call). With the raw text from the record; older
    records without raw text from reasoning + finding line (the JSON fields stay unchanged)."""
    if rec.get("text"):
        hit = _last_json(rec["text"])
        pre = rec["text"][: hit[0]] if hit else rec["text"]
    else:
        pre = (rec.get("reasoning") or "") + (f"\n\nFinding: {rec['finding']}" if rec.get("finding") else "")
    return reason_fields(dict(rec), pre, classes)


def bake(a) -> int:
    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    key = json.loads(KEY.read_text())
    fold, rows = key["fold"], key["rows"]
    ref = train[folds != fold]
    tax = C.load_taxonomy(root)
    bare = a.bare or a.no_notes
    system = system_blocks(root, None if a.no_notes else ref)
    from vlm.data import SampleBuilder

    builder = SampleBuilder(root, tax, reference=train, reference_folds=folds, max_examples=4, topic_codes=10, topic_none=True)
    ids = sorted(rows)[: a.limit]
    check_budget(len(ids) * 0.08)
    tag = f"api_{a.model.replace('claude-', '').replace('-', '')}" + ("" if a.effort == "high" else f"_{a.effort}") \
        + ("_pure" if a.no_notes else "_bare" if a.bare else "") + (f"_{a.suffix}" if a.suffix else "")
    dst = EVAL / f"{tag}_0.jsonl"
    done = {json.loads(l)["id"] for l in dst.read_text().splitlines() if l.strip()} if dst.is_file() else set()
    usage_sum: dict = {}

    def one(rid: str):
        """-> (identifier, checked answer or None, usage, end of the answer, error text). Never raises: a rejected call must not
        prevent the booking of the others."""
        row = train.iloc[rows[rid]["row"]]
        extra = ""
        if not bare:
            extra = ("\nRoot errors that markers found in OTHER questions on this topic (code, how often, one marker note):\n"
                     + (builder.topic_text(row, exclude_fold=fold) or "(none)")
                     + "\n\nDiagnosed working of other students on this same question:\n" + (builder.examples_text(row, exclude_fold=fold) or "(none)") + "\n")
        try:
            msg = json.loads(http("POST", f"{API}/messages", params(a.model, a.effort, system, user_content(root, row, "train", extra), a.max_tokens)))
        except Exception as e:  # noqa: BLE001 — treat network and API errors alike: the row stays open
            return rid, None, {}, None, f"{type(e).__name__}: {e}"[:200]
        return rid, parse(msg, classes), msg.get("usage", {}), msg.get("stop_reason"), None

    todo = [r for r in ids if r not in done]
    n_ok = n_bad = n_err = 0
    ends: dict = {}
    out_tok: list[int] = []
    first_error = None
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool, open(dst, "a") as fh:
            # first row alone: it writes the shared header into the cache, the others only read it
            stream = [one(todo[0])] if todo else []
            if stream and stream[0][4] is None:
                import itertools

                stream = itertools.chain(stream, pool.map(one, todo[1:]))
            for rid, obj, usage, end, error in stream:
                for k, v in usage.items():
                    if isinstance(v, int):
                        usage_sum[k] = usage_sum.get(k, 0) + v
                if error is not None:
                    n_err += 1
                    first_error = first_error or error
                    continue  # not written: the next call retries the row
                ends[end] = ends.get(end, 0) + 1
                out_tok.append(int(usage.get("output_tokens", 0)))
                rec = {"id": rid, "end": end, "output_tokens": int(usage.get("output_tokens", 0))}
                if obj is None:
                    n_bad += 1
                    rec.update(final_answer_correct="", top3=[], why="")
                else:
                    n_ok += 1
                    rec.update(final_answer_correct="yes" if obj.get("final_answer_correct") else "no", top3=obj["top3"],
                               confidence=obj.get("confidence"), probabilities=obj.get("probabilities"), readable=obj.get("readable"),
                               why=next((f.get("text", "") for f in obj["findings"] if f.get("root")), ""))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:  # book what has been used so far, also on abort
        usd = book("bake", a.model, n_ok + n_bad, usage_sum, batch=False, note=tag) if usage_sum else 0.0
    n = max(1, n_ok + n_bad)
    print(f"{n_ok + n_bad} rows to {a.model} (effort {a.effort}{', without examples' if bare else ''}{', without notes' if a.no_notes else ''}): "
          f"{usd:.2f} USD, per row {usd / n:.4f} USD | usable {n_ok}, unusable {n_bad}, call failed {n_err} | end {ends} | "
          f"output tokens median {int(np.median(out_tok)) if out_tok else 0}, maximum {max(out_tok) if out_tok else 0} | "
          f"input {usage_sum.get('input_tokens', 0)}, cache read {usage_sum.get('cache_read_input_tokens', 0)}, written {usage_sum.get('cache_creation_input_tokens', 0)}")
    if first_error:
        print(f"first error: {first_error}")
    print(f"evaluation: score data/derived/teacher_eval/{tag}_0.jsonl against the gold labels")
    return 0 if not n_err else 1


def plan(a) -> int:
    root = C.find_data_root()
    if a.table == "train":  # teacher measurement on ALL training rows: reference for distillation, translation of the conventions
        chosen = list(range(len(C.load_table(root, "train"))))[: a.limit or None]
        d = OUT / a.name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "batches.json").is_file():
            raise SystemExit(f"plan {a.name} has already been sent — choose a new name.")
        (d / "plan.json").write_text(json.dumps({"name": a.name, "table": "train", "rows": chosen}))
        print(f"plan {a.name}: {len(chosen)} training workings (all rows of the table; a measurement, not a training target)")
        return 0
    if getattr(a, "rows_from", ""):  # reasoning mode: a fixed list of pool positions instead of ranks per task
        return plan_from_rows(a, root)
    if not a.per_question:
        raise SystemExit("--per-question missing (pool plan)")
    pool = C.load_table(root, "unlabelled")
    taken: set[int] = set()
    for name in a.exclude:
        taken |= set(json.loads((OUT / name / "plan.json").read_text())["rows"])
    # the same selection as runner.select_by_question: fixed random order, rank of the row within its task
    order = np.random.default_rng(a.seed).permutation(len(pool))
    q = pool["question_id"].to_numpy()[order]
    rank = np.zeros(len(order), dtype=np.int64)
    seen: dict[str, int] = {}
    for k, name in enumerate(q):
        rank[k] = seen.get(name, 0)
        seen[name] = rank[k] + 1
    chosen = sorted(int(i) for i in order[(rank >= a.rank_from) & (rank < a.per_question)] if int(i) not in taken)
    if a.limit:
        chosen = chosen[: a.limit]
    d = OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    if (d / "batches.json").is_file():
        raise SystemExit(f"plan {a.name} has already been sent — choose a new name.")
    (d / "plan.json").write_text(json.dumps({"name": a.name, "table": "unlabelled", "per_question": a.per_question, "rank_from": a.rank_from,
                                             "seed": a.seed, "exclude": a.exclude, "rows": chosen}))
    n_q = len(set(pool["question_id"].to_numpy()[chosen]))
    print(f"plan {a.name}: {len(chosen)} pool workings from {n_q} tasks (ranks {a.rank_from} to {a.per_question - 1} per task, seed {a.seed})")
    return 0


def plan_from_rows(a, root) -> int:
    """Plan from a fixed list of pool positions (.npy). Pool rows only."""
    if a.table != "unlabelled":
        raise SystemExit("--rows-from only applies to pool rows (unlabelled).")
    src = Path(a.rows_from)
    src = src if src.is_absolute() else COMP / "data" / src
    pool = C.load_table(root, "unlabelled")
    arr = np.load(src)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer) or len(np.unique(arr)) != len(arr):
        raise SystemExit("--rows-from: expected a one-dimensional list of distinct integers.")
    if arr.size and (arr.min() < 0 or arr.max() >= len(pool)):
        raise SystemExit("--rows-from: position outside the pool.")
    taken: set[int] = set()
    for name in a.exclude:
        taken |= set(json.loads((OUT / name / "plan.json").read_text())["rows"])
    chosen = sorted(int(i) for i in arr if int(i) not in taken)
    if a.limit:
        chosen = chosen[: a.limit]
    d = OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    if (d / "batches.json").is_file():
        raise SystemExit(f"plan {a.name} has already been sent — choose a new name.")
    rel = str(src.relative_to(COMP / "data")) if (COMP / "data") in src.parents else str(src)
    (d / "plan.json").write_text(json.dumps({"name": a.name, "table": "unlabelled", "rows_from": rel,
                                             "rows_from_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "exclude": a.exclude, "rows": chosen}))
    n_q = len(set(pool["question_id"].to_numpy()[chosen])) if chosen else 0
    print(f"plan {a.name}: {len(chosen)} pool workings from {n_q} tasks (list {rel}; {len(arr) - len(chosen)} excluded or beyond --limit)")
    return 0


def submit(a) -> int:
    root = C.find_data_root()
    train = C.load_table(root, "train")
    pool = C.load_table(root, "unlabelled")
    d = OUT / a.name
    plan_ = json.loads((d / "plan.json").read_text())
    rows, table = plan_["rows"], plan_.get("table", "unlabelled")
    if table == "train" and a.notes:
        raise SystemExit("training rows never with the notes of all training rows in the header — the row would see its own label.")
    mode = getattr(a, "mode", "label")
    if mode == "reason" and a.notes:
        raise SystemExit("reasoning mode never with annotator notes in the header.")
    src = train if table == "train" else pool
    if mode == "reason":
        system = reason_system_blocks(root, Path(a.style) if a.style else None)
    else:
        system = system_blocks(root, train if a.notes else None)
    head_sha = hashlib.sha256(system[0]["text"].encode()).hexdigest()
    state = json.loads((d / "batches.json").read_text()) if (d / "batches.json").is_file() else {
        "model": a.model, "effort": a.effort, "notes": bool(a.notes), "max_tokens": a.max_tokens, "head_sha256": head_sha, "batches": []}
    if mode == "reason":
        state.setdefault("mode", "reason")
        state.setdefault("style_sha256", load_style(Path(a.style))["sha256"] if a.style else None)
    if state.get("mode", "label") != mode:
        raise SystemExit(f"plan was started in mode {state.get('mode', 'label')!r} — choose a new name.")
    if (state["model"], state["effort"], state.get("head_sha256")) != (a.model, a.effort, head_sha):
        raise SystemExit("plan was started with a different model, effort or header — choose a new name.")
    sent = {i for b in state["batches"] for i in b["rows"]}
    todo = [i for i in rows if i not in sent]
    check_budget(len(todo) * a.usd_per_row)
    for s in range(0, len(todo), a.batch_rows):  # keep batches small: one image per request (about 0.2 MB as base64), limit 256 MB
        part = todo[s : s + a.batch_rows]
        reqs = [{"custom_id": f"u{i}", "params": params(a.model, a.effort, system, user_content(root, src.iloc[i], table), a.max_tokens)} for i in part]
        out = json.loads(http("POST", f"{API}/messages/batches", {"requests": reqs}, timeout=1800))
        state["batches"].append({"id": out["id"], "rows": part, "created": out.get("created_at")})
        (d / "batches.json").write_text(json.dumps(state))
        print(f"batch {out['id']}: {len(part)} workings sent")
    return 0


def collect_reason(a, d: Path, state: dict, classes: list[str]) -> int:
    """collect for plans in reasoning mode: per row reasoning, finding line, JSON fields, end and output tokens."""
    got: dict[int, dict] = {}
    if (d / "labels.jsonl").is_file():
        got = {json.loads(ln)["row"]: json.loads(ln) for ln in (d / "labels.jsonl").read_text().splitlines() if ln.strip()}
    bad_rows: list[dict] = []
    while True:
        pending = 0
        for b in state["batches"]:
            if b.get("collected"):
                continue
            info = json.loads(http("GET", f"{API}/messages/batches/{b['id']}"))
            if info.get("processing_status") != "ended":
                pending += 1
                continue
            usage_sum: dict = {}
            kinds: dict = {}
            ends: dict = {}
            bad = 0
            for line in http("GET", info["results_url"], timeout=1800).decode().splitlines():
                r = json.loads(line)
                res = r.get("result", {})
                kinds[res.get("type")] = kinds.get(res.get("type"), 0) + 1
                if res.get("type") != "succeeded":
                    continue
                msg = res["message"]
                row = int(r["custom_id"][1:])
                end, out_tok = msg.get("stop_reason"), int(msg.get("usage", {}).get("output_tokens", 0))
                ends[end] = ends.get(end, 0) + 1
                for k, v in msg.get("usage", {}).items():
                    if isinstance(v, int):
                        usage_sum[k] = usage_sum.get(k, 0) + v
                obj = parse_reason(msg, classes)
                if obj is None:
                    bad += 1
                    text = "".join(x.get("text", "") for x in msg.get("content", []) if x.get("type") == "text")
                    bad_rows.append({"row": row, "stop_reason": end, "output_tokens": out_tok, "text": text})
                    continue
                got[row] = {"row": row, **obj, "stop_reason": end, "output_tokens": out_tok}
            usd = book("pool_reason", state["model"], len(b["rows"]), usage_sum, batch=True, note=f"{a.name}/{b['id']}")
            b["collected"] = True
            b["result_types"], b["unusable"], b["usd"] = kinds, bad, round(usd, 4)
            (d / "batches.json").write_text(json.dumps(state))
            print(f"batch {b['id']}: {len(b['rows'])} sent, result types {kinds}, end {ends}, {bad} unusable, {usd:.4f} USD "
                  f"(cache read {usage_sum.get('cache_read_input_tokens', 0)}, written {usage_sum.get('cache_creation_input_tokens', 0)}, "
                  f"input {usage_sum.get('input_tokens', 0)}, output {usage_sum.get('output_tokens', 0)})", flush=True)
        if not (a.wait and pending):
            break
        time.sleep(60)
    (d / "batches.json").write_text(json.dumps(state))
    if bad_rows:
        with open(d / "unparsed.jsonl", "a") as fh:
            for rec in bad_rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sha, with_finding = write_reason_labels(d, state, got)
    vals = list(got.values())
    none = np.mean([v["root_code"] == "NONE" for v in vals]) if vals else float("nan")
    n_sent = sum(len(b["rows"]) for b in state["batches"])
    print(f"{a.name}: {len(got)} answers with JSON from {n_sent} sent workings (with a finding line {with_finding}, share NONE {none:.3f}), "
          f"SHA-256 {sha[:16]}…, {pending} batches still in progress; total spending {spent(ledger()):.2f} USD")
    return 0


def write_reason_labels(d: Path, state: dict, got: dict) -> tuple[str, int]:
    text = "".join(json.dumps(got[k], ensure_ascii=False) + "\n" for k in sorted(got))
    (d / "labels.jsonl").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    vals = list(got.values())
    none = np.mean([v["root_code"] == "NONE" for v in vals]) if vals else float("nan")
    with_finding = sum(1 for v in vals if v.get("finding"))
    (d / "manifest.json").write_text(json.dumps({"name": d.name, "table": "unlabelled", "mode": "reason", "model": state["model"],
                                                 "effort": state["effort"], "notes": state.get("notes"), "head_sha256": state.get("head_sha256"),
                                                 "style_sha256": state.get("style_sha256"), "sent": sum(len(b["rows"]) for b in state["batches"]),
                                                 "rows": len(got), "with_finding": with_finding, "sha256": sha,
                                                 "share_none": None if not vals else round(float(none), 4)}, indent=1))
    return sha, with_finding


def reparse(a) -> int:
    """Repeat the parsing of a reasoning plan with the current parser (without the API); rewrites labels.jsonl."""
    root = C.find_data_root()
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    d = OUT / a.name
    state = json.loads((d / "batches.json").read_text())
    if state.get("mode", "label") != "reason":
        raise SystemExit("reparse only applies to plans in reasoning mode.")
    got = {json.loads(ln)["row"]: json.loads(ln) for ln in (d / "labels.jsonl").read_text().splitlines() if ln.strip()}
    before = sum(1 for v in got.values() if v.get("finding"))
    got = {k: reparse_record(v, classes) for k, v in got.items()}
    sha, after = write_reason_labels(d, state, got)
    print(f"{a.name}: {len(got)} records re-parsed; with a finding line {before} -> {after}; SHA-256 {sha[:16]}…")
    return 0


def collect(a) -> int:
    root = C.find_data_root()
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    d = OUT / a.name
    state = json.loads((d / "batches.json").read_text())
    if state.get("mode", "label") == "reason":
        return collect_reason(a, d, state, classes)
    table = json.loads((d / "plan.json").read_text()).get("table", "unlabelled")
    got: dict[int, dict] = {}
    if (d / "labels.jsonl").is_file():
        got = {json.loads(l)["row"]: json.loads(l) for l in (d / "labels.jsonl").read_text().splitlines() if l.strip()}
    while True:
        pending = 0
        for b in state["batches"]:
            if b.get("collected"):
                continue
            info = json.loads(http("GET", f"{API}/messages/batches/{b['id']}"))
            if info.get("processing_status") != "ended":
                pending += 1
                continue
            usage_sum: dict = {}
            kinds: dict = {}
            ends: dict = {}
            bad = 0
            for line in http("GET", info["results_url"], timeout=1800).decode().splitlines():
                r = json.loads(line)
                res = r.get("result", {})
                kinds[res.get("type")] = kinds.get(res.get("type"), 0) + 1
                if res.get("type") != "succeeded":  # errored / canceled / expired: not charged, the row stays without a label
                    continue
                msg = res["message"]
                ends[msg.get("stop_reason")] = ends.get(msg.get("stop_reason"), 0) + 1
                for k, v in msg.get("usage", {}).items():
                    if isinstance(v, int):
                        usage_sum[k] = usage_sum.get(k, 0) + v
                obj = parse(msg, classes)
                if obj is None:
                    bad += 1
                    continue
                row = int(r["custom_id"][1:])
                got[row] = {"row": row, **{k: obj.get(k) for k in ("readable", "final_answer_correct", "student_final_answer", "transcription", "findings",
                                                                    "root_code", "top3", "probabilities", "confidence")}}
            usd = book("train" if table == "train" else "pool", state["model"], len(b["rows"]), usage_sum, batch=True, note=f"{a.name}/{b['id']}")
            b["collected"] = True
            b["result_types"], b["unusable"], b["usd"] = kinds, bad, round(usd, 4)
            (d / "batches.json").write_text(json.dumps(state))
            print(f"batch {b['id']}: {len(b['rows'])} sent, result types {kinds}, end {ends}, {bad} unusable, {usd:.2f} USD "
                  f"(cache read {usage_sum.get('cache_read_input_tokens', 0)}, written {usage_sum.get('cache_creation_input_tokens', 0)}, "
                  f"input {usage_sum.get('input_tokens', 0)}, output {usage_sum.get('output_tokens', 0)})", flush=True)
        if not (a.wait and pending):
            break
        time.sleep(60)
    (d / "batches.json").write_text(json.dumps(state))
    text = "".join(json.dumps(got[k], ensure_ascii=False) + "\n" for k in sorted(got))
    (d / "labels.jsonl").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    none = np.mean([v["root_code"] == "NONE" for v in got.values()]) if got else float("nan")
    n_sent = sum(len(b["rows"]) for b in state["batches"])
    (d / "manifest.json").write_text(json.dumps({"name": a.name, "table": table, "model": state["model"], "effort": state["effort"], "notes": state.get("notes"),
                                                 "head_sha256": state.get("head_sha256"), "sent": n_sent, "rows": len(got), "sha256": sha,
                                                 "share_none": None if not got else round(float(none), 4)}, indent=1))
    print(f"{a.name}: {len(got)} labels from {n_sent} sent workings (share NONE {none:.3f}), SHA-256 {sha[:16]}…, {pending} batches still in progress; "
          f"total spending {spent(ledger()):.2f} USD")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bake")
    b.add_argument("--model", required=True, choices=sorted(PRICES))
    b.add_argument("--effort", default="high")
    b.add_argument("--bare", action="store_true", help="without the topic block and without examples of the same task")
    b.add_argument("--no-notes", action="store_true", help="like --bare, additionally without the annotators' notes in the header")
    b.add_argument("--limit", type=int, default=120)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--max-tokens", type=int, default=12000)
    b.add_argument("--suffix", default="", help="suffix in the file name, e.g. for a new version of the instruction")
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--table", choices=["unlabelled", "train"], default="unlabelled", help="train: all training rows as a measurement (not a training target)")
    p.add_argument("--per-question", type=int, default=0, help="per task the ranks below this number (pool plan)")
    p.add_argument("--rank-from", type=int, default=0, help="lower rank bound (for later batches)")
    p.add_argument("--seed", type=int, default=7, help="7 = the same order as the pool run pool_ans_t4")
    p.add_argument("--limit", type=int, default=0, help="only the first n rows of the plan (trial run)")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--rows-from", default="", help="pool positions from an .npy (relative to data/ or absolute) instead of ranks per task")
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--model", required=True, choices=sorted(PRICES))
    s.add_argument("--effort", default="high")
    s.add_argument("--notes", action="store_true", help="annotator notes of all training rows in the header (not CV-honest)")
    s.add_argument("--mode", choices=["label", "reason"], default="label",
                   help="reason: visible reasoning + finding line + JSON; label: the JSON with transcription and findings")
    s.add_argument("--style", default=str(STYLE_DEFAULT), help="style examples from fold 0 (only --mode reason); empty = without examples")
    s.add_argument("--max-tokens", type=int, default=12000)
    s.add_argument("--batch-rows", type=int, default=500)
    s.add_argument("--usd-per-row", type=float, default=0.03, help="estimate for the budget check before sending")
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    c.add_argument("--wait", action="store_true", help="check every minute until all batches are finished")
    sub.add_parser("ledger")
    rp = sub.add_parser("reparse", help="re-parse a reasoning plan with the current parser (without the API)")
    rp.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "ledger":
        led = ledger()
        print(f"API spending {spent(led):.2f} of {led['cap_usd']:.2f} USD in {len(led['entries'])} bookings")
        return 0
    return {"bake": bake, "plan": plan, "submit": submit, "collect": collect, "reparse": reparse}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
