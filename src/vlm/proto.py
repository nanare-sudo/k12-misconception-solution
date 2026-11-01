"""Diagnosis WITHOUT question knowledge in the teacher protocol, on the in-house compute path.

Before this channel the blend only knew channels that know the question: examples of the same question, topic and
question block, fine-tuning on rows of the same question. On rows whose error type has not yet occurred for the
question, this knowledge pulls away from the correct code. The job `proto_gen` produces a channel that sees only the
working: instruction, taxonomy, question, key, photo, and NO labelled row in the prompt. It therefore depends on no
fold and can be evaluated honestly on all training rows; a model without training and without teacher labels
(strictly inductive).

Protocol as measured through the API (tools/open_teacher.py):
  1. First pass in thinking mode with ONE limit for reasoning and answer together (`max_tokens`, measured: 2,500).
     Answer: a JSON object with, among other fields, three different codes and their probabilities.
  2. Optional: on hitting the token limit without answer text, up to two further thinking requests, each with ALL
     previous own notes. `continuation_tokens=0` keeps the original recipe. A training pilot can generate the old
     finish separately from the identical first pass (`paired_finish_control`).
  3. Finish for rows without a valid answer: a request WITHOUT thinking, answer only the JSON object (`finish_tokens`).
     With continuation all notes are kept; in the old recipe only their end stays in the prompt.
The steps run one after another per row; the server is the same local vLLM server as for the reasoning texts
(think_vllm): only 127.0.0.1, no outbound call (the solution runs offline).

Outputs: <tag>_labels.jsonl (checked answer per row and pass) and <tag>_raw.jsonl (reasoning, answer text); row-level
derivatives that stay under artifacts/runs/ (gitignored). For test rows only aggregates are logged; the files are not
inspected. `labels_to_probs` builds the blend channel.

`INSTRUCTIONS` and `FINISH_TEXT` are verbatim copies from tools/teacher_api.py and tools/open_teacher.py (the tools are
not part of the code package shipped to a GPU machine); tests/test_vlm.py keeps them identical.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import common as C
from . import prompts as P
from . import think_vllm as TV

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

FINISH_TEXT = ("\n\nYour own analysis notes so far (they were cut off for length):\n<<<\n{notes}\n>>>\n"
               "Do not continue the analysis. Decide now on the basis of these notes and the photo, and answer with the JSON object only.")

# Opt-in continuation, deliberately a new thinking request with the full previous notes,
# not a claim of token-identical assistant-prefix continuation. The frozen recipes keep
# their original two-step behavior when continuation_tokens=0.
CONTINUE_TEXT = ("\n\nYour earlier analysis was interrupted before you produced an answer.\n"
                 "These are your own provisional notes, not an additional source of evidence:\n<<<\n{notes}\n>>>\n"
                 "Continue the unfinished analysis using these notes and the original photo. "
                 "Check any unresolved interpretation or competing misconception before deciding. "
                 "Do not restart the entire analysis or repeat the notes. Complete the diagnosis, "
                 "then return the required JSON object.")


def continuation_policy(job: dict, split: str) -> dict:
    """Validate bounded, label-free continuation before any server or image access."""
    tokens = int(job.get("continuation_tokens", 0))
    rounds = int(job.get("max_continuations", 1))
    paired = bool(job.get("paired_finish_control", False))
    if tokens < 0 or not 1 <= rounds <= 2:
        raise ValueError("continuation_tokens must be >=0; max_continuations must be 1 or 2")
    if paired and (split != "train" or tokens == 0):
        raise ValueError("paired_finish_control requires train and continuation_tokens > 0")
    if tokens and (job.get("template_kwargs") or {}).get("enable_thinking") is False:
        raise ValueError("continuation requires thinking enabled")
    if any(int(job.get(k, default)) <= 0 for k, default in (("max_tokens", 2500), ("finish_tokens", 1500), ("note_chars", 16000))):
        raise ValueError("token and note budgets must be positive")
    return {"version": 1, "continuation_tokens": tokens, "max_continuations": rounds,
            "paired_finish_control": paired}


def full_notes(record: dict) -> str:
    """Keep the entire diagnostic content; do not silently take only its tail."""
    return "\n".join(str(record.get(k) or "") for k in ("thinking", "text")).strip()


# Ranking of ALL codes behind the own notes (runner, probe proto_rank): the same verdict rules, but instead of the
# JSON object a final line "Root: <code>"; at this position the in-house path scores every code exactly (as for the
# reasoning channel). The channel then no longer consists of three named codes with named numbers.
JSON_PART = "\n\nAnswer with ONE JSON object"
RANK_INSTRUCTIONS = (INSTRUCTIONS.split(JSON_PART, 1)[0]
                     + "\n\nReason first, then finish with one last line of the form `Root: <taxonomy code or NONE>`.")


def system_text(tax) -> str:
    return INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(tax, "full")


def rank_system_text(tax) -> str:
    return RANK_INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(tax, "full")


def rank_notes(rec: dict, note_chars: int = 16000) -> str:
    """Notes of a row for the ranking: the REASONING of the first pass (record from <tag>_raw.jsonl), without the answer
    text; the answer already names its codes, and scoring behind it would only copy them. Empty if the record is not
    a first pass or carries no reasoning."""
    if rec.get("phase") != "first":
        return ""
    return str(rec.get("thinking") or "").replace("<think>", "").replace("</think>", "").strip()[-int(note_chars):]


def load_rank_notes(path) -> tuple[str, dict[int, str]]:
    """Exported notes (tools/export_proto_notes.py): per line {"table": …, "row": table position, "text": …}.
    -> (table, {position: notes}). The table is given on EVERY line: positions of different tables must never mix
    (a training row would otherwise silently receive the notes of a test row)."""
    tables, notes = set(), {}
    for line in Path(path).read_text().split("\n"):  # not splitlines(): U+2028 and similar characters occur raw in the texts
        if not line.strip():
            continue
        r = json.loads(line)
        tables.add(str(r["table"]))
        if int(r["row"]) in notes:
            raise ValueError(f"{Path(path).name}: a table position appears twice in the file")
        if str(r["text"]).strip():
            notes[int(r["row"])] = str(r["text"])
    if len(tables) != 1:
        raise ValueError(f"{Path(path).name}: expected exactly one table, found {sorted(tables)}")
    return tables.pop(), notes


def open_images(row, split: str, data_root, image):
    """Photo of the row (with the rotation from the image settings) and diagram of the question, as in `generate`."""
    student = C.open_image(C.image_path(data_root, split, row["image_file"]), image.max_side, image.min_side)
    turns = (int((image.rotate or {}).get(str(row["trace_id"]), 0)) + int(image.rotate_all)) % 4
    if turns:
        student = student.rotate(90 * turns, expand=True)
    qimg = C.open_image(C.image_path(data_root, "questions", row["question_image_file"]), image.question_max_side) if row["question_image_file"] else None
    return student, qimg


def rank_sample(row, student_image, question_image, system: str, notes: str) -> P.Sample:
    """Sample for the ranking behind the notes: system text (verdict rules and taxonomy; the same for all rows, its
    computed state is reused), user message as in the protocol (question, diagram, key, photo; NO labelled row), the
    notes as a closed thinking block, forced answer prefix "Root:". In thinking mode the answer starts inside the
    thinking block (engine.chat_text), so only its content and its end are given here."""
    if not notes.strip():
        raise ValueError("rank_sample: empty notes")
    return P.Sample(key=str(row["trace_id"]), parts=user_parts(row, student_image, question_image),
                    prefix=notes.rstrip() + "\n</think>\n\nRoot:", meta={"task": "proto_rank", "think": True, "system": system, "static": system})


def user_parts(row, student_image, question_image=None, tail: str = "") -> list[tuple[str, object]]:
    """Parts of the user message in the order of the teacher protocol: question, diagram if any, key, photo."""
    parts: list[tuple[str, object]] = [("text", f"Grade: {row['grade']} | Topic: {row['topic_name']}\nQuestion:\n{row['question_text']}\n")]
    if question_image is not None:
        parts += [("text", "Question diagram:"), ("image", question_image)]
    parts.append(("text", f"Answer key: {row['correct_answer']}\n\nStudent's handwritten working (photo):"))
    parts.append(("image", student_image))
    if tail:
        parts.append(("text", tail))
    return parts


def to_messages(system: str, parts: list[tuple[str, object]]) -> list[dict]:
    """Chat messages in OpenAI format; images lossless as PNG (as in think_vllm.to_messages)."""
    content = []
    for kind, value in parts:
        if kind == "text":
            content.append({"type": "text", "text": value})
        else:
            buf = io.BytesIO()
            value.convert("RGB").save(buf, format="PNG")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
    return [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]


def _is_number(v) -> bool:
    """A number that can be written as float (no bool, no oversized integer)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        float(v)
    except OverflowError:
        return False
    return True


def parse_answer(text: str, classes: list[str]) -> dict | None:
    """First JSON object of the answer, checked against the class list; None if it is not usable."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    valid = set(classes)
    raw_top = obj.get("top3") if isinstance(obj.get("top3"), list) else []
    top, positions = [], []
    for i, code in enumerate(raw_top):
        if isinstance(code, str) and code in valid and code not in top:
            top.append(code)
            positions.append(i)
            if len(top) == 3:
                break
    root = obj.get("root_code")
    if not top or not isinstance(root, str) or root not in valid:  # a list or an object would not be hashable
        return None
    raw_probs = obj.get("probabilities")
    probs = None
    # Filter/deduplicate code-probability pairs together; keep the first occurrence.
    # Unequal lengths give no unambiguous pairing, so retain only the ranking.
    if isinstance(raw_probs, list) and len(raw_probs) == len(raw_top):
        selected = [raw_probs[i] for i in positions]
        if all(_is_number(v) and np.isfinite(float(v)) for v in selected):
            probs = [float(v) for v in selected]
    return {"root_code": root, "top3": top,
            "probabilities": probs,
            "confidence": float(obj["confidence"]) if _is_number(obj.get("confidence")) else None,
            "final_answer_correct": obj.get("final_answer_correct") if isinstance(obj.get("final_answer_correct"), bool) else None,
            "readable": str(obj.get("readable") or "").strip().lower()[:10]}


def split_reply(raw: str, thinking: bool) -> tuple[str, str]:
    """-> (reasoning, answer text) from the raw text of an answer. In thinking mode the answer starts inside the thinking
    block; if its end is missing, everything is reasoning (cut off at the token limit)."""
    if not thinking:
        return "", raw
    if "</think>" in raw:
        think, answer = raw.split("</think>", 1)
        return think.replace("<think>", "").strip(), answer
    return raw.replace("<think>", "").strip(), ""


def finish_notes(raw: str, think: str, text: str, end: str | None) -> str:
    """Notes for the finish in the form of the measurement (tools/open_teacher.py reads them from renderer.parse_response):
    if the answer was cut off at the token limit, it is the RAW text (trailing whitespace is kept, and after an
    already closed thinking block the started answer belongs to it, with the block start the measurement added).
    If the answer ended regularly and was merely unusable, it is the reasoning."""
    if end == "stop":
        return think or text
    return ("<think>\n" + raw) if "</think>" in raw and not raw.startswith("<think>") else raw


def request_body(messages: list[dict], max_tokens: int, thinking: bool, effort: str = "", temperature: float = 0.0, seed: int | None = None,
                 template_extra: dict | None = None) -> dict:
    template: dict = {"enable_thinking": bool(thinking)}
    if thinking and effort:  # reasoning effort; empty = default of the chat template (Qwen3.8: xhigh)
        template["reasoning_effort"] = effort
    template.update(template_extra or {})  # extra arguments for the chat template of a model with a different architecture (job field template_kwargs)
    body = {"model": "k12", "messages": messages, "max_tokens": int(max_tokens), "temperature": float(temperature),
            "chat_template_kwargs": template, "skip_special_tokens": False}
    if seed is not None:
        body["seed"] = int(seed)
    return body


def labels_to_probs(path, n: int, classes: list[str], seen_cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Channel from the answers: the named codes with their probabilities, the remaining mass spread evenly over the
    classes that can be a root. Per row the last valid record counts (a finish replaces the first pass).
    -> (P [n, K], present [n]); rows without a valid answer stay zero and count as not present."""
    index = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    probs, has = np.zeros((n, K)), np.zeros(n, bool)
    seen_cls = np.asarray(seen_cls, dtype=bool)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ok", True) or r.get("root_code") not in index:
            continue
        p = np.zeros(K)
        for c, w in zip(r.get("top3") or [], r.get("probabilities") or []):
            if c in index and _is_number(w) and np.isfinite(float(w)):  # NaN, infinity or true/false are not probabilities
                p[index[c]] += max(0.0, float(w))
        if p.sum() <= 0:  # no usable probabilities: fixed levels for the named codes
            for c, w in zip(r.get("top3") or [r["root_code"]], (0.6, 0.25, 0.1)):
                if c in index:
                    p[index[c]] += w
        if p.sum() > 1:
            p = p / p.sum()
        p = p + max(0.0, 1.0 - p.sum()) * seen_cls / seen_cls.sum()
        probs[int(r["row"])], has[int(r["row"])] = p / p.sum(), True
    return probs, has


def generate(job: dict, model_dir: Path, df, rows: np.ndarray, split: str, data_root, tax, classes: list[str], image, out_dir: Path,
             tag: str, log=print, first_rows=None) -> dict:
    """Generate answers in the teacher protocol for `rows` (positions in `df`); interrupted runs resume.

    first_rows: positions computed FIRST (if they are in `rows`), e.g. the rows of an earlier measurement, so that a
    run can be checked against them after a few minutes. The order changes no result.

    job: reply_replace ([[old, new], …]: translate the markers of a model with a different architecture into the own
    form, e.g. the end of its thinking block into "</think>"; applied in this order to the raw text of every answer),
    template_kwargs (extra arguments for its chat template),
    max_tokens (2500), finish_tokens (1500), note_chars (16000), think_effort (""), temperature (0), seed,
    continuation_tokens (0): optional extra thinking per round, only for length-limited responses without answer text.
    max_continuations (1, at most 2): hard limit; continued rows keep ALL prior notes, including at finalization.
    paired_finish_control (False): train-only, separate old-policy control from the SAME first response.
    concurrency (64), guard_rows (24) / guard_min_valid (0.5): abort if too few of the first rows return a valid
    answer (request errors count as well), so that an unattended run does not carry a format error through all rows.
    order_seed: shuffle the processing order (default: table order). max_failed (0.1): if more rows than this share
    stay open at the end because of request errors, the job fails (after writing the statistics)."""
    policy = continuation_policy(job, split)
    extra_tokens, max_rounds, paired_control = (policy[k] for k in ("continuation_tokens", "max_continuations", "paired_finish_control"))
    spec = job.get("server", {})
    port = int(spec.get("port", 8000))
    out_dir.mkdir(parents=True, exist_ok=True)
    f_lab, f_raw = out_dir / f"{tag}_labels.jsonl", out_dir / f"{tag}_raw.jsonl"
    f_ctl, f_ctl_raw = out_dir / f"{tag}_control_labels.jsonl", out_dir / f"{tag}_control_raw.jsonl"
    # Resuming with another policy would silently combine different experiments.
    # Legacy files stay readable; enabling continuation requires a new output tag.
    f_policy = out_dir / f"{tag}_generation.json"
    recipe = {"table": split, "policy": policy, "system": system_text(tax),
              "model_dir": str(model_dir), "image": asdict(image), "classes": list(classes),
              "rows_sha256": hashlib.sha256(np.asarray(sorted(int(i) for i in rows), dtype="<i8").tobytes()).hexdigest(),
              "continuation_prompt": CONTINUE_TEXT, "finish_prompt": FINISH_TEXT,
              "settings": {k: job.get(k, default) for k, default in
                           (("max_tokens", 2500), ("finish_tokens", 1500), ("note_chars", 16000),
                            ("think_effort", ""), ("temperature", 0.0), ("seed", None),
                            ("template_kwargs", None), ("reply_replace", None))}}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if f_policy.exists():
        if json.loads(f_policy.read_text()).get("sha256") != fingerprint:
            raise ValueError("proto_gen: generation policy changed; use a new output tag")
    elif extra_tokens:
        if any(p.exists() and p.stat().st_size for p in (f_lab, f_raw, f_ctl, f_ctl_raw)):
            raise ValueError("proto_gen: legacy output cannot be resumed with continuation; use a new tag")
        f_policy.write_text(json.dumps({"sha256": fingerprint, "recipe": recipe}, indent=1))
    done: set[int] = set()  # rows that have passed both steps (valid, or invalid after the finish)
    if f_lab.is_file():
        for line in f_lab.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok") or r.get("phase") == "finish":
                    done.add(int(r["row"]))
    if paired_control:
        controlled = {int(json.loads(line)["row"]) for line in f_ctl.read_text().splitlines() if line.strip()} if f_ctl.exists() else set()
        done.intersection_update(controlled)
    todo = [int(i) for i in rows if int(i) not in done]
    if job.get("order_seed") is not None:  # never in file order: an aborted run stays a random sample
        todo = [todo[k] for k in np.random.default_rng(int(job["order_seed"])).permutation(len(todo))]
    if first_rows is not None:
        head = {int(i) for i in first_rows}
        todo = [i for i in todo if i in head] + [i for i in todo if i not in head]
    stats: dict = {"rows": int(len(rows)), "done_before": int(len(rows) - len(todo))}
    max_tokens, finish_tokens = int(job.get("max_tokens", 2500)), int(job.get("finish_tokens", 1500))
    note_chars = int(job.get("note_chars", 16000))
    effort, temperature, seed = str(job.get("think_effort", "")), float(job.get("temperature", 0.0)), job.get("seed")
    replace = [(str(o), str(nw)) for o, nw in (job.get("reply_replace") or [])]
    template_extra = job.get("template_kwargs")
    system = system_text(tax)

    def parts_for(i: int, tail: str = "") -> list[tuple[str, object]]:
        row = df.iloc[i]
        student = C.open_image(C.image_path(data_root, split, row["image_file"]), image.max_side, image.min_side)
        turns = (int((image.rotate or {}).get(str(row["trace_id"]), 0)) + int(image.rotate_all)) % 4
        if turns:
            student = student.rotate(90 * turns, expand=True)
        qimg = C.open_image(C.image_path(data_root, "questions", row["question_image_file"]), image.question_max_side) if row["question_image_file"] else None
        return user_parts(row, student, qimg, tail)

    if todo:
        proc = TV.start_server(model_dir, spec, out_dir / f"{tag}_server.log")
        try:
            TV.wait_ready(port, proc, int(spec.get("startup_s", 1500)))
            log(f"vLLM server ready; {len(todo)} rows ({split}) in the teacher protocol, first pass up to {max_tokens} tokens, finish up to {finish_tokens}, "
                f"{job.get('concurrency', 64)} concurrent; extra thinking {max_rounds if extra_tokens else 0} x {extra_tokens}, "
                f"paired control {paired_control}")
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            timeout = int(job.get("request_timeout_s", 1800))

            def ask(i: int, finish: bool, notes: str = "", continuing: bool = False) -> dict:
                err = ""
                phase = "continue" if continuing else "finish" if finish else "first"
                tail = CONTINUE_TEXT.format(notes=notes) if continuing else FINISH_TEXT.format(notes=notes) if finish else ""
                for attempt in range(3):
                    try:  # including building the request: an image error is a row error, not a reason to end the job
                        body = request_body(to_messages(system, parts_for(i, tail)),
                                            extra_tokens if continuing else finish_tokens if finish else max_tokens, thinking=not finish, effort=effort, temperature=temperature,
                                            seed=None if seed is None else int(seed) * 1_000_003 + i, template_extra=template_extra)
                        out = TV._post(url, body, timeout=timeout)
                        ch = out["choices"][0]
                        raw = TV.raw_text(ch)
                        for old, new in replace:
                            raw = raw.replace(old, new)
                        think, text = split_reply(raw, thinking=(not finish) or "</think>" in raw)  # a thinking block despite thinking being switched off: the answer follows it
                        return {"row": i, "phase": phase, "thinking": think, "text": text, "end": ch.get("finish_reason"), "_raw": raw,
                                "in_tokens": int(out.get("usage", {}).get("prompt_tokens", 0)), "out_tokens": int(out.get("usage", {}).get("completion_tokens", 0))}
                    except Exception as exc:  # noqa: BLE001 - the row stays open and is counted at the end
                        err = f"{type(exc).__name__}: {exc}"[:200]
                        time.sleep(3 * (attempt + 1))
                return {"row": i, "phase": phase, "error": err}

            def safe_parse(text: str) -> dict | None:
                try:
                    return parse_answer(text, classes)
                except Exception:  # noqa: BLE001 - an unreadable answer is invalid, not a reason to end the job
                    return None

            def one(i: int) -> tuple[list[dict], dict | None]:
                first = ask(i, finish=False)
                if "error" in first:
                    return [first], None
                raw = first.pop("_raw")
                first["parsed"] = safe_parse(first["text"])
                if first["parsed"] is not None:
                    return [first], dict(first, phase="control", reused=True) if paired_control else None
                baseline_notes = finish_notes(raw, first["thinking"], first["text"], first["end"])[-note_chars:]
                eligible = extra_tokens and first["end"] == "length" and not first["text"].strip()
                control = None
                if eligible and paired_control:
                    control = ask(i, finish=True, notes=baseline_notes)
                    control.pop("_raw", None)
                    control["phase"] = "control"
                    if "error" in control:
                        return [first], control
                    control["parsed"] = safe_parse(control["text"])
                records, notes = [first], full_notes(first)
                if eligible:
                    for turn in range(max_rounds):
                        step = ask(i, finish=False, notes=notes, continuing=True)
                        if "error" in step:
                            return records + [step], control
                        step.pop("_raw", None)
                        step["continuation_index"] = turn + 1
                        step["notes_input_chars"] = len(notes)
                        step["parsed"] = safe_parse(step["text"])
                        records.append(step)
                        notes += "\n\n" + full_notes(step)
                        if step["parsed"] is not None:
                            return records, control
                        if step["end"] != "length" or step["text"].strip():
                            break
                # Only after the bounded extra thinking, or for output-format repair.
                second = ask(i, finish=True, notes=notes if eligible else baseline_notes)
                if "error" not in second:
                    second.pop("_raw", None)
                    second["parsed"] = safe_parse(second["text"])
                    if eligible:
                        second["notes_input_chars"] = len(notes)
                        second["reasoning_budget_exhausted"] = records[-1]["end"] == "length" and not records[-1]["text"].strip()
                if paired_control and not eligible:
                    control = dict(second, phase="control", reused=True)
                return records + [second], control

            t0 = time.time()
            n_err = n_rows = n_valid = n_seen = 0
            guard_rows, guard_min = int(job.get("guard_rows", 24)), float(job.get("guard_min_valid", 0.5))
            pool = ThreadPoolExecutor(max_workers=int(job.get("concurrency", 64)))
            try:
                from contextlib import ExitStack
                with ExitStack() as stack:
                    fl, fr = stack.enter_context(open(f_lab, "a")), stack.enter_context(open(f_raw, "a"))
                    fc = stack.enter_context(open(f_ctl, "a")) if paired_control else None
                    fcr = stack.enter_context(open(f_ctl_raw, "a")) if paired_control else None

                    def write_record(r, labels_file, raw_file):
                        parsed = r.pop("parsed")
                        raw_file.write(json.dumps(r, ensure_ascii=False) + "\n")
                        lab = {"row": r["row"], "phase": r["phase"], "ok": parsed is not None, "end": r["end"], "in_tokens": r["in_tokens"],
                               "out_tokens": r["out_tokens"], "think_chars": len(r["thinking"]), "answer_chars": len(r["text"].strip())}
                        lab.update({k: r[k] for k in ("continuation_index", "notes_input_chars", "reasoning_budget_exhausted", "reused") if k in r})
                        lab.update(parsed or {})
                        labels_file.write(json.dumps(lab, ensure_ascii=False) + "\n")
                        return parsed

                    for recs, control in pool.map(one, todo):
                        n_seen += 1
                        errors = [r for r in recs + ([control] if control else []) if "error" in r]
                        if errors:
                            n_err += 1  # write nothing: the row is repeated completely at the next start
                            if n_seen == guard_rows and n_valid < guard_min * guard_rows:
                                raise RuntimeError(f"proto_gen: only {n_valid} of {n_seen} first rows with a valid answer ({n_err} request errors, last: "
                                                   f"{errors[0]['error']}); aborting (guard_min_valid {guard_min})")
                            continue
                        if control is not None:
                            write_record(control, fc, fcr)
                            fc.flush()
                            fcr.flush()
                        for r in recs:
                            parsed = write_record(r, fl, fr)
                        fl.flush()
                        fr.flush()
                        n_rows += 1
                        n_valid += int(parsed is not None)
                        if n_seen == guard_rows and n_valid < guard_min * guard_rows:
                            raise RuntimeError(f"proto_gen: only {n_valid} of {n_seen} first rows with a valid answer; aborting (guard_min_valid {guard_min})")
                        if n_rows % 100 == 0:
                            log(f"proto_gen: {n_rows} of {len(todo)} rows, {(time.time() - t0) / n_rows:.2f} s per row, valid {n_valid}")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            stats.update(seconds=round(time.time() - t0, 1), s_per_row=round((time.time() - t0) / max(1, len(todo)), 3), failed=n_err)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                proc.kill()

    # statistics: aggregates only
    want = {int(i) for i in rows}
    first, last_ok, fin, continued = {}, {}, {}, {}
    logged_tokens = 0
    if f_lab.is_file():
        for line in f_lab.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["row"])
            if i not in want:
                continue
            if r.get("phase") == "continue":
                continued[(i, r["continuation_index"])] = r
            elif r.get("phase") == "finish":
                fin[i] = r
            elif r.get("phase") == "first":
                first[i] = r
            logged_tokens += int(r.get("out_tokens", 0))
            if r.get("ok"):
                last_ok[i] = r
    ends: dict = {}
    for r in first.values():
        ends[str(r.get("end"))] = ends.get(str(r.get("end")), 0) + 1
    out1 = [r.get("out_tokens", 0) for r in first.values()]
    out2 = [r.get("out_tokens", 0) for r in fin.values()]
    stats.update(answered=len(first), valid=len(last_ok), valid_first=int(sum(bool(r.get("ok")) for r in first.values())), finished=len(fin), ends_first=ends,
                 out_tokens_first_median=int(np.median(out1)) if out1 else 0, out_tokens_finish_median=int(np.median(out2)) if out2 else 0,
                 in_tokens_first_median=int(np.median([r.get("in_tokens", 0) for r in first.values()])) if first else 0,
                 share_none_top=float(np.mean([r["root_code"] == "NONE" for r in last_ok.values()])) if last_ok else 0.0,
                 distinct_top=len({r["root_code"] for r in last_ok.values()}),
                 sampling={k: job.get(k) for k in ("temperature", "seed", "think_effort") if job.get(k) not in (None, "")})
    if extra_tokens:
        stats.update(generation_policy=policy, generation_sha256=fingerprint,
                     continued_rows=len({i for i, _ in continued}), continuation_requests=len(continued),
                     valid_continuations=sum(bool(r.get("ok")) for r in continued.values()),
                     first_length_without_answer=sum(r.get("end") == "length" and r.get("answer_chars") == 0 for r in first.values()),
                     exhausted_reasoning_rows=sum(bool(r.get("reasoning_budget_exhausted")) for r in fin.values()),
                     logged_generated_tokens=logged_tokens)
        if paired_control:
            controls = {int(r["row"]): r for line in f_ctl.read_text().splitlines() if line.strip() for r in [json.loads(line)]} if f_ctl.exists() else {}
            controls = {i: r for i, r in controls.items() if i in want}
            control_stats = {"table": split, "rows": len(want), "answered": len(controls),
                             "valid": sum(bool(r.get("ok")) for r in controls.values()),
                             "additional_generated_tokens": sum(int(r.get("out_tokens", 0)) for r in controls.values() if not r.get("reused")),
                             "paired_first_response": True, "generation_sha256": fingerprint}
            (out_dir / f"{tag}_control_meta.json").write_text(json.dumps(control_stats, indent=1))
            stats["control"] = control_stats
    (out_dir / f"{tag}_meta.json").write_text(json.dumps({"table": split, **stats}, indent=1))
    if stats.get("failed", 0) > float(job.get("max_failed", 0.1)) * max(1, len(todo)):  # after the statistics: must_pass acts on this exception
        raise RuntimeError(f"proto_gen: {stats['failed']} of {len(todo)} rows without an answer (request errors); job incomplete")
    return stats
