#!/usr/bin/env python3
"""Measure open models via the Tinker API as diagnosticians on TRAINING ROWS (development only).

    <env>/bin/python tools/open_teacher.py check
    <env>/bin/python tools/open_teacher.py run --model inkling-small (--rows <positions.npy> | --limit n)
                                                   [--max-tokens 6000] [--effort 0.9] [--workers 8] [--tag <suffix>]
    <env>/bin/python tools/open_teacher.py finish --model inkling-small   # finish rows that stopped at the token limit
    <env>/bin/python tools/open_teacher.py continue --model qwen35-397b --rows <positions.npy> --tag <new>
                                                   [--source-tag ''] [--max-tokens 5000] [--finish-tokens 1500]
                                                   [--note-chars 40000] [--allow-partial] [--dry-run]
    <env>/bin/python tools/open_teacher.py ledger

`continue` (the continuation form of the own compute path src/vlm/proto.py): for rows whose first pass stopped at the token
limit WITHOUT an answer text, a new thinking request with all own notes (CONTINUE_TEXT from vlm.proto); if its answer is
invalid, the finishing pass without thinking with ALL notes (first pass + continuation). The source are the existing first
passes in <model>[_<source-tag>]/raw.jsonl (unchanged); output is only written into the NEW folder <model>_<tag>/. Before
the start an upper cost estimate for all rows; during the run a row only starts if booked + reserved + its upper estimate
stays below the cap.

The host allows models up to 550 billion parameters and names this fine-tuning API itself. Measured is whether a large OPEN
model diagnoses the worked solutions better than our 27B without any task knowledge. Protocol as for the teacher
(tools/teacher_api.py): the same instruction, the taxonomy, task, answer key and the photo — NO labelled row in the prompt (no
examples, no topic or task block). The answers therefore depend on no fold and can be evaluated honestly on all training rows.

Rules enforced by the tool:
* Only rows of the table `train` and only images from images/train and images/questions — test and pool rows are not
  accepted (no API ever processes test rows).
* The key `TINKER_API_KEY` comes from `.env` and is never printed.
* Cost ledger `artifacts/open_ledger.json` with a hard cap (CAP_USD). Every request is booked with its token counts; before
  every batch the cost is estimated and the run aborts if the cap would be exceeded.
* Outputs only under data/derived/open_teacher/<model>[_<suffix>]/ (not versioned): labels.jsonl (checked answer per row) and
  raw.jsonl (thinking and answer text). Only aggregates appear on the console.

The outputs are measurements, not training targets. The package `tinker` lives in a separate environment; this script runs
with that environment's Python.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src"), str(COMP / "tools")]

import labels  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import prompts as P  # noqa: E402
from vlm import proto as PR  # noqa: E402  — CONTINUE_TEXT, split_reply, full_notes of the continuation form (not copied)

CAP_USD = 66.5  # hard spending cap of the cost ledger (USD); raised only deliberately
LEDGER = COMP / "artifacts" / "open_ledger.json"
OUT = COMP / "data" / "derived" / "open_teacher"
# prices per million tokens according to the provider's price list: (input, cached input, output)
MODELS = {
    "inkling-small": {"base": "thinkingmachines/Inkling-Small", "renderer": "tml_v0", "price": (1.16, 0.116, 2.88), "vision": True},
    "qwen35-397b": {"base": "Qwen/Qwen3.5-397B-A17B", "renderer": "qwen3_5", "price": (1.65, 0.33, 7.50), "vision": True},
    "qwen38-27b": {"base": "Qwen/Qwen3.8-27B", "renderer": "qwen3_8_xhigh_reasoning", "price": (0.6975, 0.1395, 5.595), "vision": True},
}
# finishing call (`finish`): template without further thinking; for the Inkling model the thinking effort controls this
FINISH_RENDERER = {"inkling-small": "tml_v0", "qwen35-397b": "qwen3_5_disable_thinking", "qwen38-27b": "qwen3_8_disable_thinking"}
FINISH_TEXT = ("\n\nYour own analysis notes so far (they were cut off for length):\n<<<\n{notes}\n>>>\n"
               "Do not continue the analysis. Decide now on the basis of these notes and the photo, and answer with the JSON object only.")


def load_key() -> None:
    for line in (COMP / ".env").read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                os.environ["TINKER_API_KEY"] = v
                return
    raise SystemExit("TINKER_API_KEY missing in .env — add it there.")


def ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.is_file() else {"cap_usd": CAP_USD, "entries": []}


def spent(led: dict) -> float:
    return float(sum(e.get("usd", 0.0) for e in led["entries"]))


def cost(model: str, n_in: int, n_cache: int, n_out: int) -> float:
    p_in, p_cache, p_out = MODELS[model]["price"]
    return ((n_in - n_cache) * p_in + n_cache * p_cache + n_out * p_out) / 1e6


def book(model: str, rows: int, n_in: int, n_cache: int, n_out: int, note: str) -> float:
    led = ledger()
    usd = cost(model, n_in, n_cache, n_out)
    led["entries"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model, "rows": rows,
                           "in_tokens": n_in, "cache_tokens": n_cache, "out_tokens": n_out, "usd": round(usd, 4), "note": note})
    tmp = LEDGER.with_name(LEDGER.name + ".tmp")  # atomic: an abort in the middle of writing never leaves half a cost ledger behind
    tmp.write_text(json.dumps(led, indent=1))
    os.replace(tmp, LEDGER)
    return usd


def system_text(root) -> str:
    import teacher_api as T  # the same instruction as for the teacher; the module sends nothing on import

    return T.INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(C.load_taxonomy(root), "full")


def load_image(path: Path, max_side: int):
    from PIL import Image, ImageOps

    if path.parent.name not in ("train", "questions"):
        raise SystemExit(f"image folder {path.parent.name!r} is not allowed (only train and questions).")
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    return im


def messages_for(root, row, system: str, tail: str = "") -> list[dict]:
    parts: list[dict] = [{"type": "text", "text": f"Grade: {row['grade']} | Topic: {row['topic_name']}\nQuestion:\n{row['question_text']}\n"}]
    if row["question_image_file"]:
        parts += [{"type": "text", "text": "Question diagram:"},
                  {"type": "image", "image": load_image(C.image_path(root, "questions", row["question_image_file"]), 448)}]
    parts.append({"type": "text", "text": f"Answer key: {row['correct_answer']}\n\nStudent's handwritten working (photo):"})
    parts.append({"type": "image", "image": load_image(C.image_path(root, "train", row["image_file"]), 1024)})
    if tail:
        parts.append({"type": "text", "text": tail})
    return [{"role": "system", "content": system}, {"role": "user", "content": parts}]


def _is_number(v) -> bool:
    """A number that can be written as a float (no bool, no oversized integer)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        float(v)
    except OverflowError:
        return False
    return True


def parse_answer(text: str, classes: list[str]) -> dict | None:
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
    if not top or not isinstance(root, str) or root not in valid:
        return None
    raw_probs = obj.get("probabilities")
    probs = None
    # Keep this pairing policy identical to src/vlm/proto.py (offline inference).
    if isinstance(raw_probs, list) and len(raw_probs) == len(raw_top):
        selected = [raw_probs[i] for i in positions]
        if all(_is_number(v) and np.isfinite(float(v)) for v in selected):
            probs = [float(v) for v in selected]
    return {"root_code": root, "top3": top,
            "probabilities": probs,
            "confidence": float(obj["confidence"]) if _is_number(obj.get("confidence")) else None,
            "final_answer_correct": obj.get("final_answer_correct") if isinstance(obj.get("final_answer_correct"), bool) else None,
            "student_final_answer": str(obj.get("student_final_answer") or "")[:200],
            "readable": str(obj.get("readable") or "").strip().lower()[:10],
            "transcription": [str(x) for x in obj.get("transcription", [])][:60] if isinstance(obj.get("transcription"), list) else [],
            # isinstance before `in valid`: a code given as a list/object is not hashable (a TypeError would have ended the run)
            "findings": [f for f in (obj.get("findings") if isinstance(obj.get("findings"), list) else [])
                         if isinstance(f, dict) and isinstance(f.get("code"), str) and f.get("code") in valid][:12]}


def split_message(message: dict) -> tuple[str, str]:
    """-> (thinking, answer text) from the parsed message of the template."""
    content = message.get("content")
    if isinstance(content, str):
        return "", content
    think = "".join(p.get("thinking", "") for p in content if p.get("type") == "thinking")
    text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return think, text


def cmd_check(a) -> int:
    load_key()
    import tinker

    sc = tinker.ServiceClient()
    try:
        sc._check_accessible()
    except Exception as e:  # type and status only, never headers
        print(f"access: ERROR {type(e).__name__} (status {getattr(e, 'status_code', '?')})")
        return 2
    names = {getattr(m, "model_name", str(m)) for m in sc.get_server_capabilities().supported_models}
    for k, m in MODELS.items():
        print(f"  {k:<14} {m['base']:<40} {'available' if m['base'] in names else 'NOT in the model list'}")
    led = ledger()
    print(f"cost ledger: {spent(led):.2f} of {led['cap_usd']:.2f} USD")
    return 0


async def run_async(a) -> int:
    load_key()
    import tinker
    from tinker import types
    from tinker_cookbook import renderers
    from tinker_cookbook.image_processing_utils import get_image_processor
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    spec = MODELS[a.model]
    root = C.find_data_root()
    train = C.load_table(root, "train")  # training rows only
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    out = OUT / (a.model + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    finish = a.cmd == "finish"
    notes: dict[int, str] = {}
    if finish:
        # rows whose answer stopped at the token limit: finish them from the own notes (the thinking so far)
        state = {}
        for line in (out / "labels.jsonl").open():
            r = json.loads(line)
            state[int(r["row"])] = bool(r.get("ok")) or state.get(int(r["row"]), False)
        for line in (out / "raw.jsonl").open():
            r = json.loads(line)
            if not state.get(int(r["row"]), False) and r.get("phase") != "finish":
                notes[int(r["row"])] = (r.get("thinking") or r.get("text") or "")[-a.note_chars :]
        todo = sorted(notes)
    else:
        rows = np.load(C.comp_root() / "data" / a.rows) if a.rows else np.arange(len(train))
        rows = np.asarray(rows, dtype=int)
        if rows.min() < 0 or rows.max() >= len(train):
            raise SystemExit("row positions outside train.csv")
        done = set()
        if (out / "labels.jsonl").is_file():
            done = {int(json.loads(line)["row"]) for line in (out / "labels.jsonl").open() if line.strip()}
        todo = [int(i) for i in rows if int(i) not in done]
        if a.order_seed >= 0:  # never the first n of a sorted list: a partial measurement must be a random sample
            todo = [todo[k] for k in np.random.default_rng(a.order_seed).permutation(len(todo))]
    if a.limit:
        todo = todo[: a.limit]
    if not todo:
        print("nothing to do: all requested rows exist")
        return 0

    tokenizer = get_tokenizer(spec["base"])
    rname = FINISH_RENDERER[a.model] if finish else spec["renderer"]
    renderer = renderers.get_renderer(rname, tokenizer, get_image_processor(spec["base"]) if spec["vision"] else None, model_name=spec["base"])
    system = system_text(root)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=spec["base"])
    params = types.SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, stop=renderer.get_stop_sequences())
    sem = asyncio.Semaphore(a.workers)
    led = ledger()
    budget_left = led["cap_usd"] - spent(led)
    est_row = cost(a.model, a.est_in, 0, a.max_tokens)  # upper estimate per row
    if est_row * min(len(todo), a.workers) > budget_left:
        raise SystemExit(f"aborted: remaining budget {budget_left:.2f} USD is not enough for the first batch.")
    stats = {"rows": 0, "ok": 0, "in": 0, "cache": 0, "out": 0, "usd": 0.0, "stop_length": 0, "failed": 0}
    lock = asyncio.Lock()
    t0 = time.time()
    stop_all = False

    def build(i: int):
        msgs = messages_for(root, train.iloc[i], system, FINISH_TEXT.format(notes=notes[i]) if finish else "")
        if rname == "tml_v0":
            return renderer.build_generation_prompt(msgs, effort=0.01 if finish else a.effort)
        return renderer.build_generation_prompt(msgs)

    async def one(i: int) -> None:
        nonlocal stop_all
        async with sem:
            if stop_all:
                return
            try:
                prompt = await asyncio.to_thread(build, i)
                res = await sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
                seq = res.sequences[0]
                toks = list(seq.tokens)
                msg, _term = renderer.parse_response(toks)
                think, text = split_message(msg)
                n_in, n_cache, n_out = int(prompt.length), int(getattr(res, "prompt_cache_hit_tokens", 0) or 0), len(toks)
                parsed = parse_answer(text, classes)
                stop = str(getattr(seq.stop_reason, "value", seq.stop_reason))
            except Exception as e:  # a row may fail; never print headers or keys
                async with lock:
                    stats["failed"] += 1
                    if stats["failed"] <= 3:
                        print(f"  row failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
                return
            async with lock:
                usd = book(a.model, 1, n_in, n_cache, n_out, a.note)
                stats["rows"] += 1
                stats["in"] += n_in
                stats["cache"] += n_cache
                stats["out"] += n_out
                stats["usd"] += usd
                stats["stop_length"] += int("length" in stop.lower())
                with (out / "raw.jsonl").open("a") as fh:
                    fh.write(json.dumps({"row": i, "phase": "finish" if finish else "first", "thinking": think, "text": text, "stop": stop, "in_tokens": n_in, "cache_tokens": n_cache, "out_tokens": n_out}, ensure_ascii=False) + "\n")
                rec = {"row": i, "phase": "finish" if finish else "first", "ok": parsed is not None, "stop": stop, "in_tokens": n_in, "out_tokens": n_out, "think_chars": len(think)}
                if parsed is not None:
                    stats["ok"] += 1
                    rec.update(parsed)
                with (out / "labels.jsonl").open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if stats["rows"] % 25 == 0:
                    print(f"  {stats['rows']} of {len(todo)} rows, {stats['usd']:.2f} USD, {(time.time() - t0) / stats['rows']:.1f} s per row (wall clock), "
                          f"valid {stats['ok']}, at the token limit {stats['stop_length']}", flush=True)
                led_now = ledger()
                if spent(led_now) + est_row * a.workers > led_now["cap_usd"]:
                    stop_all = True

    await asyncio.gather(*(one(i) for i in todo))
    n = max(1, stats["rows"])
    print(f"{a.model}: {stats['rows']} rows ({stats['failed']} failed), valid answers {stats['ok']} ({stats['ok'] / n:.1%}), at the token limit "
          f"{stats['stop_length']}; tokens per row: input {stats['in'] / n:.0f} (of which cached {stats['cache'] / n:.0f}), output {stats['out'] / n:.0f}; "
          f"{stats['usd']:.2f} USD = {stats['usd'] / n:.4f} per row; wall clock {time.time() - t0:.0f} s"
          + (" — ABORTED at the budget cap" if stop_all else ""))
    led = ledger()
    print(f"cost ledger: {spent(led):.2f} of {led['cap_usd']:.2f} USD")
    return 0


# ------------------------------------------------------------------------------------------------ continuation
# Form as in the own compute path (src/vlm/proto.py, ask/full_notes): a NEW thinking request with the complete own notes of the
# first pass appended (CONTINUE_TEXT), greedy; if the answer is not valid JSON, the finishing pass without thinking
# (FINISH_RENDERER, FINISH_TEXT) with ALL notes — first pass + "\n\n" + continuation, without the 16,000-character truncation
# of the old finishing pass (note_chars is only a hard upper bound from the end). At most one continuation per row.
CONT_TAIL_TOKENS = 256  # upper estimate for the appended text without notes (CONTINUE_TEXT ≈ 100, FINISH_TEXT ≈ 60 tokens)
NOTE_TOKEN_FACTOR, NOTE_TOKEN_PAD = 1.05, 64  # notes re-tokenised: at most +5 % + 64 tokens compared with the generated ones (checked)
# Only for displaying an expectation (the 27B in the same protocol): mean output of the continuation at +5,000 tokens
# 4,058.5; afterwards without a valid answer (finishing pass needed) 53.2 % of the rows.
CONT_OUT_TOKENS_TYPICAL, FINISH_SHARE_TYPICAL = 4058.5, 0.532
FINISH_OUT_DEFAULT = 403  # mean output of the finishing pass for q397 if the source contains no finishing pass
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}")
RECIPE_FILE = "continue_generation.json"


def is_length(rec: dict) -> bool:
    return "length" in str(rec.get("stop", rec.get("end", ""))).lower()


def tinker_split(rec: dict) -> tuple[str, str]:
    """(thinking, answer text) of a record in thinking mode, split as in vlm.proto.split_reply.

    If the template cannot split an answer (stopped at the token limit), raw.jsonl holds the raw text under `text` and
    `thinking` is empty: without an end of the thinking block everything is thinking, otherwise it is split at its end. A
    regularly finished answer without a thinking part stays answer text."""
    think, text = str(rec.get("thinking") or ""), str(rec.get("text") or "")
    if think or ("</think>" not in text and not is_length(rec)):
        return think, text
    return PR.split_reply(text, thinking=True)


def eligible_first(rec: dict) -> bool:
    """First pass at the token limit WITHOUT an answer text — as `eligible` in vlm.proto.generate."""
    return rec.get("phase") == "first" and is_length(rec) and not tinker_split(rec)[1].strip()


def record_notes(rec: dict) -> str:
    """All own notes of a record (thinking and answer text): vlm.proto.full_notes on the split above."""
    think, text = tinker_split(rec)
    return PR.full_notes({"thinking": think, "text": text})


def cut_notes(notes: str, note_chars: int) -> tuple[str, bool]:
    """Hard upper bound from the end (not reached for the q397 notes of ≤ 11,000 + ≤ 15,000 characters)."""
    return (notes[-int(note_chars):], True) if len(notes) > int(note_chars) else (notes, False)


def finish_notes(first: dict, cont: dict, note_chars: int) -> tuple[str, bool]:
    """Notes of the finishing pass after the continuation: first pass and continuation COMPLETE (proto.py: notes +=
    "\\n\\n" + full_notes(step)); truncated only at the upper bound note_chars."""
    return cut_notes(record_notes(first) + "\n\n" + record_notes(cont), note_chars)


def read_jsonl(path: Path) -> list[dict]:
    """Records of an output file (split at \\n only: U+2028 and similar stand raw in the texts)."""
    if not path.is_file():
        return []
    out = []
    for k, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            raise SystemExit(f"{path.parent.name}/{path.name}: line {k + 1} is not JSON (interrupted write?) — check first") from None
    return out


def check_train_record(rec: dict, n: int) -> None:
    """Training rows only: position integer in [0, n); a field `table` must be 'train'."""
    row = rec.get("row")
    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < n:
        raise SystemExit(f"record with an invalid row position {row!r} — only rows of train.csv")
    if rec.get("table", "train") != "train":
        raise SystemExit(f"record of table {rec.get('table')!r} — only train is allowed")


def load_rows(rel: str, n: int) -> list[int]:
    """Row list (.npy, relative to data/) in its order: unique, integer, positions in train.csv."""
    rows = np.asarray(np.load(C.comp_root() / "data" / rel))
    if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer):
        raise SystemExit("--rows: a one-dimensional, non-empty list of integers expected")
    if rows.min() < 0 or rows.max() >= n:
        raise SystemExit("row positions outside train.csv")
    if len(set(rows.tolist())) != len(rows):
        raise SystemExit("--rows contains a position twice")
    return [int(i) for i in rows]


def rows_sha256(rows) -> str:
    return hashlib.sha256(np.asarray(sorted(int(i) for i in rows), dtype="<i8").tobytes()).hexdigest()


def continue_dirs(model: str, source_tag: str, tag: str) -> tuple[Path, Path]:
    """-> (source, new output folder). --tag is mandatory, a simple name and never equal to the source."""
    if not tag or not TAG_RE.fullmatch(tag):
        raise SystemExit("--tag is mandatory: a new folder suffix of letters, digits, _ . - (the source stays untouched)")
    if source_tag and not TAG_RE.fullmatch(source_tag):
        raise SystemExit("--source-tag: invalid name")
    src = OUT / (model + (f"_{source_tag}" if source_tag else ""))
    out = OUT / f"{model}_{tag}"
    if tag == source_tag or out.resolve() == src.resolve():
        raise SystemExit("output folder equal to the source — the continuation always writes into a NEW folder (--tag)")
    return src, out


def source_firsts(src: Path, n: int) -> tuple[dict[int, dict], int]:
    """First passes of the source per row (exactly one per row) and the mean output of their finishing passes (for display)."""
    firsts: dict[int, dict] = {}
    fin_out = []
    for r in read_jsonl(src / "raw.jsonl"):
        check_train_record(r, n)
        if r.get("phase") == "first":
            if int(r["row"]) in firsts:
                raise SystemExit(f"source {src.name}: several first passes for one row — not unique")
            firsts[int(r["row"])] = r
        elif r.get("phase") == "finish":
            fin_out.append(int(r.get("out_tokens", 0)))
    return firsts, (round(float(np.mean(fin_out))) if fin_out else FINISH_OUT_DEFAULT)


def continue_plan(firsts: dict[int, dict], rows: list[int], out_lab: list[dict], out_raw: list[dict]) -> dict:
    """What remains to be done per row, in the order of the row list: 'continue' (continuation and, if needed, finishing pass)
    or 'finish' (continuation exists but is invalid, finishing pass missing — resumption without a new continuation). Skipped:
    rows without a first pass in the source, rows whose first pass did not end at the limit without an answer text, and
    completed rows (valid continuation or finishing pass present)."""
    lab = {(int(r["row"]), r.get("phase")): r for r in out_lab}
    raw = {(int(r["row"]), r.get("phase")): r for r in out_raw}
    if set(lab) != set(raw):
        raise SystemExit("output folder inconsistent: labels.jsonl and raw.jsonl do not contain the same steps — check first")
    want = set(rows)
    if any(i not in want for i, _ in lab):
        raise SystemExit("output folder contains rows outside the row list")
    todo, counts = [], {"rows": len(rows), "missing_first": 0, "not_eligible": 0, "done": 0, "resume_finish": 0}
    cont_raw = {}
    for i in rows:
        f = firsts.get(i)
        if f is None:
            counts["missing_first"] += 1
        elif not eligible_first(f):
            counts["not_eligible"] += 1
        elif (i, "finish") in lab or bool(lab.get((i, "continue"), {}).get("ok")):
            counts["done"] += 1
        elif (i, "continue") in lab:
            todo.append((i, "finish"))
            cont_raw[i] = raw[(i, "continue")]
            counts["resume_finish"] += 1
        else:
            todo.append((i, "continue"))
    return {"todo": todo, "counts": counts, "cont_raw": cont_raw}


def upper_usd(model: str, first: dict, max_tokens: int, finish_tokens: int, cont: dict | None = None) -> tuple[float, float]:
    """Upper cost estimate of a row -> (continuation, finishing pass), without cache: input = prompt of the first pass (measured
    tokens) + notes (+5 % + 64) + appended text; output = full limit; the finishing pass is always included. With `cont`
    (resumption) the continuation is dropped and its measured output replaces the limit."""
    base_in = int(first["in_tokens"])
    n1 = math.ceil(NOTE_TOKEN_FACTOR * int(first["out_tokens"])) + NOTE_TOKEN_PAD
    n2 = math.ceil(NOTE_TOKEN_FACTOR * (int(cont["out_tokens"]) if cont else int(max_tokens))) + NOTE_TOKEN_PAD
    u_cont = 0.0 if cont else cost(model, base_in + n1 + CONT_TAIL_TOKENS, 0, int(max_tokens))
    return u_cont, cost(model, base_in + n1 + n2 + CONT_TAIL_TOKENS, 0, int(finish_tokens))


def typical_usd(model: str, first: dict, max_tokens: int, finish_out: int, cont_out: float, finish_share: float) -> float:
    """Display: cache as in the first pass, continuation with cont_out output tokens, finishing pass in finish_share of the rows
    with finish_out tokens."""
    base_in, cache, n1 = int(first["in_tokens"]), int(first.get("cache_tokens", 0) or 0), int(first["out_tokens"])
    c = cost(model, base_in + n1 + 100, cache, min(cont_out, max_tokens))
    return c + finish_share * cost(model, base_in + n1 + int(min(cont_out, max_tokens)) + 60, cache, finish_out)


def budget_prefix(uppers: list[float], left: float) -> int:
    """Number of rows at the start of the list whose upper estimates together fit into the remaining budget."""
    tot, k = 0.0, 0
    for u in uppers:
        if tot + u > left + 1e-12:
            break
        tot, k = tot + u, k + 1
    return k


def budget_check(uppers: list[float], left: float, allow_partial: bool) -> int:
    """Before the start: upper estimate for ALL rows against the remaining budget. If it is not enough, abort — except with
    allow_partial: then the start of the list that is safely affordable counts (return: number of these rows)."""
    total = float(sum(uppers))
    if total <= left + 1e-12:
        return len(uppers)
    k = budget_prefix(uppers, left)
    if not allow_partial:
        raise SystemExit(f"aborted: upper estimate {total:.2f} USD for {len(uppers)} rows exceeds the remaining budget {left:.2f} USD "
                         f"(the first {k} would be safely affordable). With --allow-partial the run computes, in the order of the list, only as many "
                         "rows as are safely affordable.")
    if k == 0:
        raise SystemExit(f"aborted: the remaining budget {left:.2f} USD is not enough for the first row by the upper estimate.")
    return k


class BudgetGate:
    """Reserves the upper estimates of running rows: a row only starts if booked + reserved + its upper estimate stays below the
    cap — the cap holds even if every running row needs its maximum. Booked requests release their reservation; this way, with
    --allow-partial, further rows of the list move up."""

    def __init__(self, cap: float):
        self.cap, self.reserved = float(cap), 0.0

    def fits(self, usd: float, spent_now: float) -> bool:
        return spent_now + self.reserved + usd <= self.cap + 1e-12

    def reserve(self, usd: float) -> None:
        self.reserved += float(usd)

    def release(self, usd: float) -> None:
        self.reserved = max(0.0, self.reserved - float(usd))


def continue_recipe(a, src: Path, firsts: dict[int, dict], rows: list[int], system: str) -> dict:
    spec = MODELS[a.model]
    src_firsts = [firsts[i] for i in sorted(rows) if i in firsts]
    return {"version": 1, "table": "train", "model": a.model, "base": spec["base"], "renderer": spec["renderer"],
            "finish_renderer": FINISH_RENDERER[a.model], "source": src.name,
            "source_first_sha256": hashlib.sha256(json.dumps(src_firsts, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "rows_sha256": rows_sha256(rows), "max_continuations": 1, "max_tokens": int(a.max_tokens),
            "finish_tokens": int(a.finish_tokens), "note_chars": int(a.note_chars), "temperature": float(a.temperature),
            "effort": float(a.effort), "continuation_prompt": PR.CONTINUE_TEXT, "finish_prompt": FINISH_TEXT,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest()}


def prepare_continue(a, n: int, system: str, dry: bool = False) -> dict:
    """Everything offline: folder, row list, source, recipe check, plan, upper estimate and budget check (abort before any call).
    n = rows of train.csv; only this table is ever read."""
    for k in ("max_tokens", "finish_tokens", "note_chars", "workers"):
        if int(getattr(a, k)) <= 0:
            raise SystemExit(f"--{k.replace('_', '-')} must be positive")
    src, out = continue_dirs(a.model, a.source_tag, a.tag)
    if not (src / "raw.jsonl").is_file():
        raise SystemExit(f"source {src.name}/raw.jsonl missing")
    rows = load_rows(a.rows, n)
    firsts, finish_out = source_firsts(src, n)
    out_lab, out_raw = read_jsonl(out / "labels.jsonl"), read_jsonl(out / "raw.jsonl")
    for r in out_lab + out_raw:
        check_train_record(r, n)
        if r.get("phase") not in ("continue", "finish"):
            raise SystemExit(f"output folder {out.name} contains a record of phase {r.get('phase')!r} — not a continuation folder")
    recipe = continue_recipe(a, src, firsts, rows, system)
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if (out / RECIPE_FILE).is_file():
        if json.loads((out / RECIPE_FILE).read_text()).get("sha256") != fingerprint:
            raise SystemExit(f"{out.name}: recipe of the continuation differs (rows, limits, texts or source) — use a new --tag")
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out.name} exists and does not come from `continue` — use another --tag")
    plan = continue_plan(firsts, rows, out_lab, out_raw)
    todo = plan["todo"][: a.limit] if a.limit else plan["todo"]
    uppers = [upper_usd(a.model, firsts[i], a.max_tokens, a.finish_tokens, plan["cont_raw"].get(i)) for i, _ in todo]
    if not LEDGER.is_file():  # otherwise ledger() would fall back to an EMPTY ledger (booked 0) — never compute without the existing bookings
        raise SystemExit(f"cost ledger {LEDGER.name} missing — no continuation without the existing bookings")
    led = ledger()
    cap = min(float(led["cap_usd"]), CAP_USD)
    left = cap - spent(led)
    totals = [u + v for u, v in uppers]
    k = budget_prefix(totals, left) if dry else budget_check(totals, left, a.allow_partial)
    return {"src": src, "out": out, "rows": rows, "firsts": firsts, "finish_out": finish_out, "recipe": recipe,
            "fingerprint": fingerprint, "plan": plan, "todo": todo, "uppers": uppers, "cap": cap, "left": left,
            "safe_rows": k, "upper_total": float(sum(totals))}


def print_estimate(a, prep: dict) -> None:
    """Cost estimate for the full list (aggregates only)."""
    todo, firsts = prep["todo"], prep["firsts"]
    full = [i for i, s in todo if s == "continue"]
    pess = sum(typical_usd(a.model, firsts[i], a.max_tokens, prep["finish_out"], a.max_tokens, 1.0) for i in full)
    exp = sum(typical_usd(a.model, firsts[i], a.max_tokens, prep["finish_out"], CONT_OUT_TOKENS_TYPICAL * a.max_tokens / 5000, FINISH_SHARE_TYPICAL) for i in full)
    rest = sum(v for (_i, s), (_u, v) in zip(todo, prep["uppers"]) if s == "finish")
    c = prep["plan"]["counts"]
    print(f"plan: {c['rows']} rows in the list — to do {len(todo)} (of which finishing pass only {c['resume_finish']}), done {c['done']}, "
          f"skipped: without a first pass {c['missing_first']}, not at the limit without an answer text {c['not_eligible']}")
    print(f"cost (USD) for the {len(todo)} rows: upper estimate {prep['upper_total']:.2f} (no cache, full limits, "
          f"always a finishing pass) | unfavourable {pess + rest:.2f} (cache as in the first pass, full continuation, always a finishing pass "
          f"with {prep['finish_out']} tokens) | expectation from the mean of all continued rows of the 27B {exp + rest:.2f} "
          "(open rows are more expensive than the average)")
    print(f"remaining budget {prep['left']:.2f} USD (cap {prep['cap']:.2f}); safely affordable by the upper estimate: the first "
          f"{prep['safe_rows']} of {len(todo)} rows")


def continue_dry(a) -> int:
    """Offline dry run: plan and cost for the full list; for 3 rows build the messages locally (and, if template and tokenizer
    are available offline, the prompts with token counts). No key, no request, no file."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = C.find_data_root()
    train = C.load_table(root, "train")  # training rows only
    system = system_text(root)
    prep = prepare_continue(a, len(train), system, dry=True)
    print_estimate(a, prep)
    probe = [i for i, s in prep["todo"] if s == "continue"][:3]
    if not probe:
        return 0
    renderer = frenderer = None
    try:  # local files only (HF_HUB_OFFLINE); without tinker_cookbook only the message structure and note lengths
        from tinker_cookbook import renderers
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        spec = MODELS[a.model]
        tok = get_tokenizer(spec["base"])
        ip = get_image_processor(spec["base"]) if spec["vision"] else None
        renderer = renderers.get_renderer(spec["renderer"], tok, ip, model_name=spec["base"])
        frenderer = renderers.get_renderer(FINISH_RENDERER[a.model], tok, ip, model_name=spec["base"])
    except Exception as e:  # noqa: BLE001
        print(f"  template/tokenizer not available offline ({type(e).__name__}) — message structure only")
    agg: dict[str, list] = {k: [] for k in ("notes_chars", "same_prefix", "first_len_match", "cont_len", "cont_est", "fin_len", "fin_est")}
    for i in probe:
        row, first = train.iloc[i], prep["firsts"][i]
        notes, _ = cut_notes(record_notes(first), a.note_chars)
        m0 = messages_for(root, row, system, "")
        m1 = messages_for(root, row, system, PR.CONTINUE_TEXT.format(notes=notes))
        p0, p1 = m0[1]["content"], m1[1]["content"]
        same = len(p1) == len(p0) + 1 and all(
            (x["type"] == y["type"] and (x.get("text") == y.get("text") if x["type"] == "text" else x["image"].size == y["image"].size))
            for x, y in zip(p0, p1)) and p1[-1] == {"type": "text", "text": PR.CONTINUE_TEXT.format(notes=notes)}
        agg["notes_chars"].append(len(notes))
        agg["same_prefix"].append(bool(same and m1[0] == m0[0]))
        u_c, u_f = upper_usd(a.model, first, a.max_tokens, a.finish_tokens)
        if renderer is not None:
            proxy = {"thinking": record_notes(first), "text": "", "stop": "length"}  # placeholder: continuation as long as the first pass
            fin_notes, _ = finish_notes(first, proxy, a.note_chars)
            m2 = messages_for(root, row, system, FINISH_TEXT.format(notes=fin_notes))
            l0 = int(renderer.build_generation_prompt(m0).length)
            l1 = int(renderer.build_generation_prompt(m1).length)
            l2 = int(frenderer.build_generation_prompt(m2).length)
            agg["first_len_match"].append(l0 == int(first["in_tokens"]))
            agg["cont_len"].append(l1)
            agg["cont_est"].append(int(first["in_tokens"]) + math.ceil(NOTE_TOKEN_FACTOR * int(first["out_tokens"])) + NOTE_TOKEN_PAD + CONT_TAIL_TOKENS)
            agg["fin_len"].append(l2)
            agg["fin_est"].append(int(first["in_tokens"]) + 2 * (math.ceil(NOTE_TOKEN_FACTOR * int(first["out_tokens"])) + NOTE_TOKEN_PAD) + CONT_TAIL_TOKENS)
    print(f"dry run on {len(probe)} rows: notes on average {np.mean(agg['notes_chars']):.0f} characters; message of the continuation = "
          f"first pass + appended CONTINUE_TEXT: {sum(agg['same_prefix'])}/{len(probe)}")
    if agg["cont_len"]:
        print(f"  prompt of the first pass built locally = measured input tokens: {sum(agg['first_len_match'])}/{len(probe)}; continuation "
              f"{np.mean(agg['cont_len']):.0f} tokens (upper estimate {np.mean(agg['cont_est']):.0f}, exceeded {sum(x > y for x, y in zip(agg['cont_len'], agg['cont_est']))}); "
              f"finishing pass with placeholder notes {np.mean(agg['fin_len']):.0f} (estimate {np.mean(agg['fin_est']):.0f}, exceeded "
              f"{sum(x > y for x, y in zip(agg['fin_len'], agg['fin_est']))})")
    return 0


async def continue_async(a, env: dict | None = None) -> int:
    """Continuation via the API. env (tests only): root, train, classes, system, renderer, finish_renderer, sampler,
    make_params — without env the real objects (key from .env, tinker).

    Only ONE continuation process at a time (lock under OUT, released by the operating system at process end): two processes
    would plan the same rows (duplicate requests) and overwrite each other's bookings in the cost ledger."""
    import fcntl

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".continue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another continuation is still running (lock data/derived/open_teacher/.continue.lock) — never two at the same time") from None
        return await _continue_locked(a, env)


async def _continue_locked(a, env: dict | None) -> int:
    if env is None:
        root = C.find_data_root()
        train = C.load_table(root, "train")  # training rows only
        classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
        system = system_text(root)
    else:
        root, train, classes, system = env["root"], env["train"], env["classes"], env["system"]
    prep = prepare_continue(a, len(train), system)  # budget check BEFORE every call
    print_estimate(a, prep)
    todo, uppers, firsts, out = prep["todo"], prep["uppers"], prep["firsts"], prep["out"]
    if not todo:
        print("nothing to do: all rows of the list are continued or skipped")
        return 0
    if env is None:
        load_key()
        import tinker
        from tinker import types
        from tinker_cookbook import renderers
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        spec = MODELS[a.model]
        tok = get_tokenizer(spec["base"])
        ip = get_image_processor(spec["base"]) if spec["vision"] else None
        renderer = renderers.get_renderer(spec["renderer"], tok, ip, model_name=spec["base"])
        frenderer = renderers.get_renderer(FINISH_RENDERER[a.model], tok, ip, model_name=spec["base"])
        sampler = tinker.ServiceClient().create_sampling_client(base_model=spec["base"])

        def make_params(max_tokens: int, rend):
            return types.SamplingParams(max_tokens=max_tokens, temperature=a.temperature, stop=rend.get_stop_sequences())
    else:
        renderer, frenderer, sampler, make_params = env["renderer"], env["finish_renderer"], env["sampler"], env["make_params"]
    out.mkdir(parents=True, exist_ok=True)
    if not (out / RECIPE_FILE).is_file():
        (out / RECIPE_FILE).write_text(json.dumps({"sha256": prep["fingerprint"], "recipe": prep["recipe"]}, indent=1, ensure_ascii=False))
    rname, fname = MODELS[a.model]["renderer"], FINISH_RENDERER[a.model]
    p_cont, p_fin = make_params(a.max_tokens, renderer), make_params(a.finish_tokens, frenderer)
    gate = BudgetGate(prep["cap"])
    cond = asyncio.Condition()
    stats = {"rows": 0, "continued": 0, "cont_ok": 0, "cont_length": 0, "finished": 0, "fin_ok": 0, "failed": 0, "budget_skipped": 0,
             "in": 0, "cache": 0, "out": 0, "usd": 0.0}
    state = {"pos": 0, "in_flight": 0, "closed": "", "stop_all": False, "outcomes": 0, "valid": 0, "unbooked_max": 0.0, "api_failed": 0}
    max_failed = int(getattr(a, "max_failed", 8))
    t0 = time.time()

    def build(i: int, tail: str, rend, name: str, effort: float):
        msgs = messages_for(root, train.iloc[i], system, tail)
        return rend.build_generation_prompt(msgs, effort=effort) if name == "tml_v0" else rend.build_generation_prompt(msgs)

    def safe_parse(text: str) -> dict | None:  # as in vlm.proto.generate: an unreadable answer is invalid, no reason for a crash
        try:
            return parse_answer(text, classes)
        except Exception:  # noqa: BLE001
            return None

    def write(raw_rec: dict, parsed: dict | None) -> None:
        lab = {k: raw_rec[k] for k in ("row", "table", "phase", "stop", "in_tokens", "out_tokens") if k in raw_rec}
        lab.update({"ok": parsed is not None, "think_chars": len(raw_rec["thinking"]), "answer_chars": len(raw_rec["text"].strip())})
        lab.update({k: raw_rec[k] for k in ("continuation_index", "notes_input_chars", "notes_cut", "reasoning_budget_exhausted", "parse_error")
                    if k in raw_rec})
        lab.update(parsed or {})
        with (out / "raw.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
        with (out / "labels.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lab, ensure_ascii=False) + "\n")

    def after_booking() -> None:  # under cond: an impending overrun (e.g. foreign bookings) stops everything
        if spent(ledger()) + gate.reserved > gate.cap + 1e-9:
            state["stop_all"] = True

    def outcome(valid: bool) -> None:  # under cond: guard against a format error on all rows
        state["outcomes"] += 1
        state["valid"] += int(valid)
        if state["outcomes"] == a.guard_rows and state["valid"] < a.guard_min_valid * a.guard_rows:
            state["stop_all"] = True
            print(f"  guard: only {state['valid']} of {a.guard_rows} first rows with a valid answer — abort", flush=True)

    async def request(i: int, tail: str, rend, name: str, effort: float, params, max_out: int, held: list) -> dict | None:
        """Build the prompt, bring the reservation to the exact upper estimate, send, book. None = not sent/failed."""
        try:  # the build, too: an image error is a row error, no reason to end the run
            prompt = await asyncio.to_thread(build, i, tail, rend, name, effort)
            exact = cost(a.model, int(prompt.length), 0, max_out)
        except Exception as e:  # noqa: BLE001
            async with cond:
                stats["failed"] += 1
                if stats["failed"] <= 3:
                    print(f"  prompt not built: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return None
        async with cond:
            if state["stop_all"] or not gate.fits(exact - held[1], spent(ledger())):
                stats["budget_skipped"] += 1
                return None
            gate.reserve(exact - held[1])  # reservation of this step: estimate -> exact prompt
            held[0], held[1] = held[0] + exact - held[1], exact
        try:
            res = await sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        except Exception as e:  # a row may fail; never print headers or keys
            async with cond:
                # Whether the provider charged the failed request is unknown: its upper estimate stays reserved for the rest of the run
                # (the cap still holds); from max_failed failed requests on nothing new starts.
                stats["failed"] += 1
                state["api_failed"] += 1
                state["unbooked_max"] += exact
                held[0] -= exact
                held[1] = 0.0
                if stats["failed"] <= 3:
                    print(f"  row failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
                if state["api_failed"] >= max_failed and not state["stop_all"]:
                    state["stop_all"] = True
                    print(f"  {state['api_failed']} failed requests — abort (nothing new; running requests are still booked)", flush=True)
            return None
        try:  # the answer exists: from here on it is ALWAYS booked, even if splitting fails
            seq = res.sequences[0]
            toks = list(seq.tokens)
            n_out, stop = len(toks), str(getattr(seq.stop_reason, "value", seq.stop_reason))
            n_cache = int(getattr(res, "prompt_cache_hit_tokens", 0) or 0)
        except Exception:  # noqa: BLE001 — token count unknown: book the upper estimate
            toks, n_out, stop, n_cache = [], int(max_out), "error", 0
        rec = {"thinking": "", "text": "", "stop": stop, "in_tokens": int(prompt.length), "cache_tokens": n_cache, "out_tokens": n_out}
        async with cond:
            stats["usd"] += book(a.model, 1, rec["in_tokens"], rec["cache_tokens"], rec["out_tokens"], a.note)
            gate.release(exact)
            held[0] -= exact
            held[1] = 0.0
            for k, v in (("in", "in_tokens"), ("cache", "cache_tokens"), ("out", "out_tokens")):
                stats[k] += rec[v]
            after_booking()
        try:
            msg, _term = rend.parse_response(toks)
            rec["thinking"], rec["text"] = split_message(msg)
        except Exception as e:  # noqa: BLE001 — booked, but not splittable: counts as an invalid answer
            rec["parse_error"] = type(e).__name__
        return rec

    async def row_job(i: int, step: str, u_c: float, u_f: float) -> None:
        held = [u_c + u_f, u_c if step == "continue" else u_f]  # [still reserved for the row, reservation of the running step]
        try:
            first = firsts[i]
            if step == "continue":
                notes, cut = cut_notes(record_notes(first), a.note_chars)
                rec = await request(i, PR.CONTINUE_TEXT.format(notes=notes), renderer, rname, a.effort, p_cont, a.max_tokens, held)
                if rec is None:
                    return
                rec["thinking"], rec["text"] = tinker_split(rec)
                parsed = safe_parse(rec["text"])
                cont = {"row": i, "table": "train", "phase": "continue", "continuation_index": 1, **rec,
                        "notes_input_chars": len(notes), "notes_cut": cut}
                async with cond:
                    write(cont, parsed)
                    stats["continued"] += 1
                    stats["cont_ok"] += int(parsed is not None)
                    stats["cont_length"] += int(is_length(rec))
                    if parsed is not None:
                        stats["rows"] += 1
                        outcome(True)
                        return
                    held[1] = u_f
            else:
                cont = prep["plan"]["cont_raw"][i]
            fin_notes, cut = finish_notes(first, cont, a.note_chars)
            rec = await request(i, FINISH_TEXT.format(notes=fin_notes), frenderer, fname, 0.01, p_fin, a.finish_tokens, held)
            if rec is None:
                return
            parsed = safe_parse(rec["text"])
            fin = {"row": i, "table": "train", "phase": "finish", **rec, "notes_input_chars": len(fin_notes), "notes_cut": cut,
                   "reasoning_budget_exhausted": is_length(cont) and not tinker_split(cont)[1].strip()}
            async with cond:
                write(fin, parsed)
                stats["finished"] += 1
                stats["fin_ok"] += int(parsed is not None)
                stats["rows"] += 1
                outcome(parsed is not None)
        except Exception as e:  # noqa: BLE001 — unexpected (e.g. a write error): a crash would drop running, already charged requests
            async with cond:  # unbooked; instead start nothing new and finish booking the running ones
                stats["failed"] += 1
                state["stop_all"] = True
                print(f"  unexpected error in a row: {type(e).__name__}: {str(e)[:160]} — abort after the running requests", flush=True)
        finally:
            async with cond:
                gate.release(held[0])
                state["in_flight"] -= 1
                cond.notify_all()
                if stats["rows"] and stats["rows"] % 25 == 0 and state.get("last_print") != stats["rows"]:
                    state["last_print"] = stats["rows"]
                    print(f"  {stats['rows']} of {len(todo)} rows, {stats['usd']:.2f} USD, {(time.time() - t0) / stats['rows']:.1f} s per row "
                          f"(wall clock); continuation valid {stats['cont_ok']}/{stats['continued']}, finishing pass valid {stats['fin_ok']}/{stats['finished']}",
                          flush=True)

    async def worker() -> None:
        while True:
            async with cond:
                while True:
                    if state["stop_all"] or state["closed"] or state["pos"] >= len(todo):
                        return
                    k = state["pos"]
                    u_c, u_f = uppers[k]
                    if gate.fits(u_c + u_f, spent(ledger())):  # in the fixed order of the list: never move a later row ahead
                        gate.reserve(u_c + u_f)
                        state["pos"] += 1
                        state["in_flight"] += 1
                        break
                    if state["in_flight"] == 0:
                        state["closed"] = f"budget: row {k + 1} of the list no longer fits into the remaining budget by the upper estimate"
                        return
                    await cond.wait()
            i, step = todo[k]
            await row_job(i, step, u_c, u_f)

    await asyncio.gather(*(worker() for _ in range(a.workers)))
    n = max(1, stats["continued"] + stats["finished"])
    print(f"{a.model} continuation → {out.name}: {stats['rows']} rows completed of {len(todo)} planned ({stats['failed']} requests failed, "
          f"{stats['budget_skipped']} not sent because of the budget); continuation valid {stats['cont_ok']}/{stats['continued']} "
          f"(at the limit {stats['cont_length']}), finishing pass valid {stats['fin_ok']}/{stats['finished']}; tokens per request: input "
          f"{stats['in'] / n:.0f} (cache {stats['cache'] / n:.0f}), output {stats['out'] / n:.0f}; {stats['usd']:.2f} USD; wall clock "
          f"{time.time() - t0:.0f} s" + (" — STOPPED (budget or guard)" if state["stop_all"] else "")
          + (f" — {state['closed']}" if state["closed"] else ""))
    led = ledger()
    print(f"cost ledger: {spent(led):.2f} of {led['cap_usd']:.2f} USD"
          + (f"; plus at most {state['unbooked_max']:.2f} USD from {state['api_failed']} failed requests that the provider may have "
             "charged (not booked — check the balance with the provider)" if state["unbooked_max"] > 0 else ""))
    return 0


def cmd_ledger(a) -> int:
    led = ledger()
    by: dict[str, list[float]] = {}
    for e in led["entries"]:
        by.setdefault(e["model"], [0, 0.0])
        by[e["model"]][0] += e["rows"]
        by[e["model"]][1] += e["usd"]
    for k, (n, usd) in by.items():
        print(f"  {k:<14} {int(n):>5} rows  {usd:6.2f} USD")
    print(f"total {spent(led):.2f} of {led['cap_usd']:.2f} USD")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=sorted(MODELS))
    r.add_argument("--rows", default="", help="positions in train.csv as .npy, relative to data/ (e.g. derived/open_teacher/rows_stage1.npy)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--order-seed", type=int, default=0, help="shuffle the processing order (fixed seed) so that --limit draws a random sample; −1 = order of the list")
    r.add_argument("--max-tokens", type=int, default=6000, help="limit for thinking AND answer together")
    r.add_argument("--effort", type=float, default=0.9, help="thinking effort of the Inkling model in [0, 1)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--est-in", type=int, default=8000, help="upper estimate of the input tokens per row for the budget check")
    r.add_argument("--tag", default="")
    r.add_argument("--note", default="")
    f = sub.add_parser("finish", help="finish the stopped rows of a run from the own notes (without further thinking)")
    f.add_argument("--model", required=True, choices=sorted(MODELS))
    f.add_argument("--tag", default="")
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--max-tokens", type=int, default=1500)
    f.add_argument("--note-chars", type=int, default=16000, help="this many characters from the end of the thinking so far are passed on as a note")
    f.add_argument("--temperature", type=float, default=0.0)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--est-in", type=int, default=12000)
    f.add_argument("--effort", type=float, default=0.01)
    f.add_argument("--note", default="finish")
    f.add_argument("--rows", default="")
    c = sub.add_parser("continue", help="continue first passes that stopped at the token limit without an answer text, new folder")
    c.add_argument("--model", required=True, choices=sorted(MODELS))
    c.add_argument("--rows", required=True, help="positions in train.csv as .npy, relative to data/; processed in this order")
    c.add_argument("--source-tag", default="", help="source <model>[_<source-tag>]/raw.jsonl (first passes); default: without a suffix")
    c.add_argument("--tag", required=True, help="new output folder <model>_<tag> (mandatory, never equal to the source)")
    c.add_argument("--max-tokens", type=int, default=5000, help="limit of the continuation (thinking and answer)")
    c.add_argument("--finish-tokens", type=int, default=1500, help="limit of the finishing pass without thinking")
    c.add_argument("--note-chars", type=int, default=40000, help="hard upper bound of the notes (characters from the end)")
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--effort", type=float, default=0.9, help="thinking effort of the Inkling model in [0, 1)")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--limit", type=int, default=0, help="only the first n open rows of the list")
    c.add_argument("--allow-partial", action="store_true",
                   help="if the remaining budget is not enough for all rows by the upper estimate: only as many rows, in the order of the list, as are safely affordable")
    c.add_argument("--dry-run", action="store_true", help="offline: plan, cost, prompts for 3 rows; nothing is written")
    c.add_argument("--guard-rows", type=int, default=24)
    c.add_argument("--guard-min-valid", type=float, default=0.5)
    c.add_argument("--max-failed", type=int, default=8, help="from this many failed requests on, nothing new starts")
    c.add_argument("--note", default="continuation")
    sub.add_parser("ledger")
    a = ap.parse_args()
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "ledger":
        return cmd_ledger(a)
    if a.cmd == "continue":
        return continue_dry(a) if a.dry_run else asyncio.run(continue_async(a))
    return asyncio.run(run_async(a))


if __name__ == "__main__":
    raise SystemExit(main())
