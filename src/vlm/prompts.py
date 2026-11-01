"""Prompts and training targets for the vision-language model runs.

A sample (`Sample`) is a chat input of text and image parts, a forced beginning of the answer (`prefix`) and, for
training, the target as a sequence of marked segments. The segments are tokenised separately and concatenated, so
the model sees the codes during training in exactly the token split in which they are scored later
(`engine.CandidateSet`).

Tasks (`task`):
    diag   decision first: "Root: <code>", then (target only) answer check, transcription, findings
    full   read first, then decide: transcription, answer check, findings, "Root: <code>"
    cot    like full, but every finding gives the reasoning first and then the code ("=> code")
    why    describe first, then name: answer check, the root error in one or two sentences (without code),
           "Root: <code>"; then (target only) transcription and findings. The decision is taken at ONE place behind
           the own description, unlike full/cot, where "Root:" only copies the code from the list of findings.
    read   read only: transcription and final-answer check
    fac    only the question "does the final answer match the key?"
    ans    copy the student's final answer, then the check (measurement only, no training target)
    think  reason in the model's thinking mode first, then "Root: <code>" (measurement only, no training target)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import transcription as tr

SYSTEM = (
    "You are an experienced mathematics teacher. You read photos of handwritten working by "
    "Indonesian students (grades 4-12; questions and student writing are in Indonesian) and "
    "diagnose the misconception behind their errors."
)

NONE_LINE = "NONE | No major error | The working is correct, or has only minor slips that do not change the method or the result."

INSTRUCTIONS = {
    "diag": (
        "Name the root misconception of this working: the major error from which the rest of the "
        "working follows (usually the earliest major error), or NONE if there is no major error. "
        "Answer with `Root: <code>`."
    ),
    "full": (
        "Work in this order and use exactly this format:\n"
        "Transcription:\n"
        "L1 | <first line of the student's working>\n"
        "L2 | <next line> ...\n"
        "Final answer correct: yes or no (does the student's final answer match the answer key?)\n"
        "Findings:\n"
        "- L<n> [<code>, major or minor]: <what went wrong>   (write `- none` if there is no error)\n"
        "Root: <code of the major error from which the rest follows, or NONE>"
    ),
    "cot": (
        "Work like a marker and use exactly this format:\n"
        "Transcription:\n"
        "L1 | <first line of the student's working>\n"
        "L2 | <next line> ...\n"
        "Final answer correct: yes or no (does the student's final answer match the answer key?)\n"
        "Findings:\n"
        "- L<n>: <what went wrong there and what it should have been> => <code> (major or minor)   (write `- none` if there is no error)\n"
        "Root: <code of the major error from which the rest follows, or NONE>"
    ),
    "why": (
        "Mark this working in three steps and use exactly this format:\n"
        "Final answer correct: yes or no (does the student's final answer match the answer key?)\n"
        "Error: <one or two sentences: what the student did at the major error from which the rest of the working "
        "follows, and what it should have been — or `none` if there is no major error>\n"
        "Root: <code of that error, or NONE>"
    ),
    "read": (
        "Transcribe the student's working line by line and then say whether the student's final "
        "answer matches the answer key. Use exactly this format:\n"
        "Transcription:\n"
        "L1 | <first line of the student's working>\n"
        "L2 | <next line> ...\n"
        "Final answer correct: yes or no"
    ),
    "fac": (
        "Does the student's final answer match the answer key? "
        "Answer with `Final answer correct: yes` or `Final answer correct: no`."
    ),
    "think": (
        "Reason before you answer: read what the student wrote line by line, solve the question yourself, and find "
        "the first step where the student's working goes wrong and why. Then name the root misconception: the major "
        "error from which the rest of the working follows, or NONE if there is no major error. "
        "Finish with one last line of the form `Root: <code>`."
    ),
    # Orientation probe: only the photo, without the question. It is evaluated for all four rotations of the same photo;
    # the rotation with the most confident "yes" counts as upright (the model reads; it does not have to name a direction).
    "up": (
        "Is this photo the right way up, so that the handwriting can be read normally, from left to right, without "
        "turning the photo? Answer with `Upright: yes` or `Upright: no`."
    ),
    "ans": (
        "Find the student's final answer in the working and write it exactly as the student wrote it "
        "(only the answer itself, on one line). Then say whether it matches the answer key. Use exactly this format:\n"
        "Student's final answer: <answer>\n"
        "Final answer correct: yes or no"
    ),
}
PREFIX = {"diag": "Root:", "full": "", "cot": "", "read": "", "think": "", "fac": "Final answer correct:", "ans": "Student's final answer:",
          "why": "Final answer correct:", "up": "Upright:"}
# Reasoning of another model (probe think) as a note in the prompt, for a component that learns on the labelled rows
# to name the code from the note (and the image) the way the annotators do (the "namer").
ANALYSIS_HEAD = ("Notes of another marker who studied THIS working (step-by-step reasoning written before a code was chosen; "
                 "it can be wrong, and the code it leans towards is often not the code our markers assign):\n")
STATIC_TASKS = ("diag", "full", "think", "cot", "why")  # tasks with a taxonomy header and examples of the same question
YES, NO = " yes", " no"


@dataclass
class Sample:
    key: str
    parts: list  # ("text", str) | ("image", PIL.Image)
    prefix: str = ""
    target: list = field(default_factory=list)  # [(text, marker)], marker: code | fac | aux | why
    meta: dict = field(default_factory=dict)


def taxonomy_text(tax: pd.DataFrame, style: str = "full", codes: list[str] | None = None, cases: dict | None = None) -> str:
    """Taxonomy as lines "code | name | description" (style=full) or "code | name" (name).

    cases: {code: [note, …]}, annotator notes on real cases of the code (from the reference set), as indented lines
    "e.g. …" below the code. The header stays the same for all rows of a run and is computed only once; the cases
    cost nothing per row.
    """
    rows = []
    if cases:
        rows.append("(Indented lines starting with `e.g.` are notes that markers wrote about real student work with the code above.)")
    for r in tax.itertuples():
        if codes is not None and r.id not in codes:
            continue
        rows.append(f"{r.id} | {r.name} | {r.description}" if style == "full" else f"{r.id} | {r.name}")
        rows += [f"    e.g. {c}" for c in (cases or {}).get(r.id, [])]
    rows.append(NONE_LINE if style == "full" else "NONE | No major error")
    return "\n".join(rows)


def code_cases(reference: pd.DataFrame, per_code: int, max_chars: int = 200, per_code_for: dict | None = None,
               with_topic: bool = False) -> dict:
    """Per root code up to `per_code` annotator notes (explanation of the root finding) from labelled rows.

    Different questions first, ordered by trace_id (deterministic). Rows without an explanation are dropped.
    per_code_for: {code: count}, an individual upper bound for single codes; codes without an entry get `per_code`
    (0 = none). Intended for a case book of the catch-all codes: what "fits no category" means is written in no
    description, only in the notes on the cases. with_topic: topic of the working in square brackets before the note.
    """
    out: dict[str, list[str]] = {}
    limits = {str(k): int(v) for k, v in (per_code_for or {}).items()}
    if reference is None or (per_code <= 0 and not limits):
        return out
    ref = reference[reference["root_code"] != "NONE"].sort_values("trace_id")
    for code, grp in ref.groupby("root_code", sort=True):
        n_max = limits.get(str(code), per_code)
        if n_max <= 0:
            continue
        first = grp.drop_duplicates("question_id")
        chosen = pd.concat([first, grp.drop(first.index)])
        notes = []
        for r in chosen.itertuples():
            why = root_reason(r.misconceptions, max_chars)
            if why and with_topic:
                why = f"[{r.topic_name}] {why}"
            if why and why not in notes:
                notes.append(why)
            if len(notes) >= n_max:
                break
        if notes:
            out[str(code)] = notes
    return out


def yes_no(flag: str) -> str:
    return YES if str(flag).strip().lower() == "true" else NO


def findings_text(misconceptions: str, max_chars: int = 320) -> str:
    """List of findings as lines "- L3 [ALG-06, major, root]: explanation"."""
    try:
        items = json.loads(misconceptions) if misconceptions else []
    except json.JSONDecodeError:
        items = []
    lines = []
    for it in items:
        tags = [str(it.get("category", "")), str(it.get("severity", ""))]
        if it.get("root"):
            tags.append("root")
        text = " ".join(str(it.get("text", "")).split())[:max_chars]
        lines.append(f"- {it.get('step', '')} [{', '.join(t for t in tags if t and t != 'nan')}]: {text}")
    return "\n".join(lines) if lines else "- none"


def findings_reasoned(misconceptions: str, max_chars: int = 320) -> str:
    """List of findings in the order "reason first, then name": "- L3: explanation => ALG-06 (major, root)".

    The explanation comes before the code, so that a model generating these lines commits to the code only after
    the reasoning (unlike `findings_text`, where the code comes first).
    """
    try:
        items = json.loads(misconceptions) if misconceptions else []
    except json.JSONDecodeError:
        items = []
    lines = []
    for it in items:
        tags = [str(it.get("severity", ""))]
        if it.get("root"):
            tags.append("root")
        tags = [x for x in tags if x and x != "nan"]
        text = " ".join(str(it.get("text", "")).split())[:max_chars]
        lines.append(f"- {it.get('step', '')}: {text} => {it.get('category', '')}" + (f" ({', '.join(tags)})" if tags else ""))
    return "\n".join(lines) if lines else "- none"


def root_reason(misconceptions: str, max_chars: int = 220) -> str:
    """Explanation of the root finding (for examples of the same question)."""
    try:
        items = json.loads(misconceptions) if misconceptions else []
    except json.JSONDecodeError:
        return ""
    for it in items:
        if it.get("root"):
            return " ".join(str(it.get("text", "")).split())[:max_chars]
    return ""


def why_text(code: str, misconceptions: str, max_chars: int = 320) -> str:
    """Description of the root error in words (annotator explanation of the root finding), without the code.

    "none" for workings without a major error; if an error row has no explanation, the text is "not described".
    """
    if code == "NONE":
        return "none"
    return root_reason(misconceptions, max_chars) or "not described"


def clean_transcription(raw: str) -> str:
    """Transcription without label leaks (annotator verdicts removed)."""
    return tr.strip_leaks(raw).strip()


def example_lines(examples: pd.DataFrame, max_examples: int = 4, max_chars: int = 260) -> str:
    """Labelled workings of other students for the same question (from the training part)."""
    if examples is None or len(examples) == 0:
        return ""
    # Variety before quantity: first one example per (code, answer check), then fill up.
    ex = examples.sort_values("trace_id")
    if "_source" in ex.columns:  # labelled training rows before teacher rows of the pool (stable sort, otherwise unchanged)
        ex = ex.iloc[np.argsort((ex["_source"] == "teacher").to_numpy(), kind="stable")]
    first = ex.drop_duplicates(["root_code", "final_answer_correct"])
    rest = ex.drop(first.index)
    chosen = pd.concat([first, rest]).head(max_examples)
    out = []
    for r in chosen.itertuples():
        work = " / ".join(t for _, t in tr.step_lines(clean_transcription(r.transcription_steps)))[:max_chars]
        fac = "correct" if str(r.final_answer_correct).lower() == "true" else "wrong"
        why = root_reason(r.misconceptions)
        out.append(f"- working: {work}\n  final answer {fac}; root: {r.root_code}" + (f" ({why})" if why else ""))
    return "\n".join(out)


def topic_lines(rows: pd.DataFrame, max_codes: int = 8, max_chars: int = 160, n_none: int | None = None) -> str:
    """Root errors of other questions of the same topic: per code its frequency and one annotator note.

    `rows` are labelled error rows (without NONE) of other questions; most frequent codes first, ties by code.
    n_none: number of workings without a major error for these questions, as the first line, so that the list of
    errors is not read as "there is always an error here" (without it the 27B lost 0.02 on NONE rows).
    """
    if rows is None or max_codes <= 0 or (len(rows) == 0 and not n_none):
        return ""
    out = []
    if n_none is not None:
        out.append(f"- NONE ({int(n_none)}x of {int(n_none) + len(rows)} marked workings): no major error")
    if len(rows) == 0:
        return "\n".join(out)
    counts = rows["root_code"].value_counts()
    order = sorted(counts.index, key=lambda c: (-int(counts[c]), str(c)))[:max_codes]
    for code in order:
        grp = rows[rows["root_code"] == code].sort_values("trace_id")
        note = next((w for w in (root_reason(m, max_chars) for m in grp["misconceptions"]) if w), "")
        out.append(f"- {code} ({int(counts[code])}x)" + (f": {note}" if note else ""))
    return "\n".join(out)


def question_lines(rows: pd.DataFrame, max_codes: int = 8, max_chars: int = 150, max_answers: int = 2, soft: bool = False,
                   allowed: set | None = None) -> str:
    """Teacher-labelled pool workings of the SAME question, condensed (the question block).

    First line: how many of the workings have no major error. Then per root code (most frequent first, ties by
    code): frequency, the most frequent final answers of students with this error and the note of the working the
    teacher was most confident about. The lines carry no information from labelled training rows (the teacher saw
    none) and therefore need no fold exclusion.

    soft: the teacher's second and third guesses count as well (`_top3`, `_probs`), taken from the workings it rates
    as erroneous (for error-free ones the same filler pair almost always appears there). The codes are then ordered by
    their summed probability, and codes the teacher never chose as root but considered more often enter the list with
    "0x". Motivation: with hard counting the correct code of a "never seen" row is among the first eight in only 62 % of
    the rows, with soft counting among the first twelve in 79 %. `allowed` restricts the soft list to codes that can be
    a root (classes with a training example).
    """
    if rows is None or len(rows) == 0 or max_codes <= 0:
        return ""
    counts = rows["root_code"].value_counts()
    n = len(rows)
    out = [f"- NONE ({int(counts.get('NONE', 0))}x of {n} workings): no major error"]
    also: dict[str, int] = {}
    if soft and "_top3" in rows.columns and "_probs" in rows.columns:
        mass: dict[str, float] = {}
        confs = rows["_conf"] if "_conf" in rows.columns else pd.Series(float("nan"), index=rows.index)
        for root, t3, pr, cf in zip(rows["root_code"], rows["_top3"], rows["_probs"], confs):
            if root == "NONE":
                continue
            codes, probs = json.loads(t3), json.loads(pr)
            pairs = dict(zip(codes, probs))
            if root not in pairs:  # root missing from its own list: count it with its confidence
                pairs[root] = float(cf) if cf == cf else 0.5
            for c, p in pairs.items():
                if c == "NONE" or (allowed is not None and c not in allowed):
                    continue
                mass[str(c)] = mass.get(str(c), 0.0) + float(p)
                if c != root:
                    also[str(c)] = also.get(str(c), 0) + 1
        order = sorted(mass, key=lambda c: (-mass[c], str(c)))[:max_codes]
    else:
        order = sorted((c for c in counts.index if c != "NONE"), key=lambda c: (-int(counts[c]), str(c)))[:max_codes]
    for code in order:
        hard = int(counts.get(code, 0))
        extra = f"; {also[code]}x as second or third guess" if also.get(code) else ""
        if hard == 0:  # only considered, never chosen as root: no final answers, no note
            out.append(f"- {code} (0x{extra})")
            continue
        grp = rows[rows["root_code"] == code]
        conf = grp["_conf"].fillna(-1.0) if "_conf" in grp.columns else pd.Series(0.0, index=grp.index)
        grp = grp.assign(_c=conf).sort_values(["_c", "trace_id"], ascending=[False, True])
        note = next((w for w in (root_reason(m, max_chars) for m in grp["misconceptions"]) if w), "")
        line = f"- {code} ({hard}x{extra})"
        if "_answer" in grp.columns:
            ans = pd.Series([" ".join(str(a).split())[:30] for a in grp["_answer"] if str(a).strip()], dtype=object)
            if len(ans):
                top = ans.value_counts()
                top = sorted(top.index, key=lambda a: (-int(top[a]), str(a)))[:max_answers]
                line += "; student answers: " + " | ".join(top)
        out.append(line + (f": {note}" if note else ""))
    return "\n".join(out)


def clean_analysis(text: str) -> str:
    """Raw reasoning text (probe think) as a note for the prompt: without the markers of the thinking block and without
    repeated blank lines; a final line "Root: X" becomes "Suggested root: X", because the answer format "Root:" is
    reserved for the model's own decision."""
    out: list[str] = []
    for ln in text.replace("<think>", "").replace("</think>", "\n").strip().splitlines():
        ln = ln.rstrip()
        if ln or (out and out[-1]):
            out.append(ln)
    while out and not out[-1]:
        out.pop()
    if out and out[-1].lstrip().startswith("Root:"):
        out[-1] = "Suggested root:" + out[-1].lstrip()[len("Root:"):]
    return "\n".join(out)


def build_sample(
    row,
    *,
    task: str,
    tax_text: str,
    student_image=None,
    question_image=None,
    examples_text: str = "",
    topic_text: str = "",
    question_block: str = "",
    teacher_examples_text: str = "",
    reading: str | None = None,
    with_target: bool = False,
    instruction_suffix: str = "",
    candidates_text: str = "",
    analysis: str = "",
    reading_head: str = "",
    analysis_head: str = "",
) -> Sample:
    """Build the sample for one row. `row` is a table row (Series or namedtuple-like).

    student_image=None gives the text-only variant (diagnosis from a transcription in `reading`).
    instruction_suffix: text appended to the instruction (only for measurements without a training target, e.g. a
    length hint for the reasoning of the think task).
    candidates_text: codes that an earlier stage (blend of other channels) proposes for THIS working, as one line
    "A, B, C"; placed after the image, before the instruction (second stage: check instead of find).
    analysis: reasoning of another model about THIS working (cleaned with `clean_analysis`); placed after the
    image, before the instruction. Without image and transcription it is the only view of the working.
    analysis_head: own header line of the note instead of ANALYSIS_HEAD (e.g. when the note is the annotators'
    description of the root error, for the measurement "naming given the finding").
    """
    g = row.get if hasattr(row, "get") else lambda k, d="": getattr(row, k, d)
    if task == "up":  # orientation probe: short prompt with only the photo, independent of question and key
        if student_image is None:
            raise ValueError("The task up needs the photo")
        return Sample(key=str(g("trace_id")), prefix=PREFIX["up"], meta={"task": "up", "static": "", "think": False},
                      parts=[("text", "Photo of a student's handwritten maths working:\n"), ("image", student_image), ("text", "\n\n" + INSTRUCTIONS["up"])])
    parts: list = []
    head = static = ""
    if task in STATIC_TASKS:
        # Same header for all rows; deliberately placed first so that its computed state can be reused.
        static = "Misconception taxonomy (code | name | description):\n" + tax_text + "\n\n"
        head += static
    head += f"Question (grade {g('grade')}, topic: {g('topic_name')}):\n{g('question_text')}\n"
    parts.append(("text", head))
    if question_image is not None:
        parts.append(("text", "Question diagram:\n"))
        parts.append(("image", question_image))
        parts.append(("text", "\n"))
    mid = f"Answer key: {g('correct_answer')}\n\n"
    if topic_text and task in STATIC_TASKS:
        mid += "Root errors that markers found in OTHER questions on this topic (code, how often, one marker note):\n" + topic_text + "\n\n"
    if question_block and task in STATIC_TASKS:
        mid += ("Machine-marked workings of other students on this SAME question (code, how often, typical student answers, "
                "one note; machine marks can be wrong):\n" + question_block + "\n\n")
    if examples_text and task in STATIC_TASKS:
        mid += "Diagnosed working of other students on this same question:\n" + examples_text + "\n\n"
    if teacher_examples_text and task in STATIC_TASKS:
        mid += ("Machine-marked working of other students on this same question (one example per error type; machine marks "
                "can be wrong):\n" + teacher_examples_text + "\n\n")
    if student_image is not None:
        mid += "Student's handwritten working:\n"
        parts.append(("text", mid))
        parts.append(("image", student_image))
        tail = "\n"
        if reading:  # reading_head: own header line, e.g. for a checked transcription (control "reading versus diagnosing")
            tail += (reading_head.strip() or "Machine transcription of this working (may contain reading errors):") + "\n" + reading.strip() + "\n"
    elif analysis and not reading:  # text-only variant from the note: no image, no transcription
        tail = ""
        parts.append(("text", mid))
    else:
        tail = "Transcription of the student's working:\n" + (reading or "").strip() + "\n"
        parts.append(("text", mid))
    if analysis and task in STATIC_TASKS:
        tail += (analysis_head.strip() + "\n" if analysis_head.strip() else ANALYSIS_HEAD) + analysis.strip() + "\n"
    if candidates_text and task in STATIC_TASKS:
        tail += ("\nRoot codes that other automatic markers proposed for THIS working, most likely first (they can be wrong — "
                 "check each against the working, and consider codes outside this list too):\n" + candidates_text.strip() + "\n")
    tail += "\n" + INSTRUCTIONS[task] + (" " + instruction_suffix.strip() if instruction_suffix.strip() else "")
    parts.append(("text", tail))

    sample = Sample(key=str(g("trace_id")), parts=parts, prefix=PREFIX[task], meta={"task": task, "static": static, "think": task == "think"})
    if with_target:
        code = str(g("root_code"))
        fac = yes_no(g("final_answer_correct"))
        trans = clean_transcription(g("transcription_steps"))
        finds = findings_text(g("misconceptions"))
        if task == "diag":
            sample.target = [
                (" " + code, "code"),
                ("\nFinal answer correct:", "aux"),
                (fac, "fac"),
                ("\nTranscription:\n" + trans + "\nFindings:\n" + finds, "aux"),
            ]
        elif task == "full":
            sample.target = [
                ("Transcription:\n" + trans + "\nFinal answer correct:", "aux"),
                (fac, "fac"),
                ("\nFindings:\n" + finds + "\nRoot:", "aux"),
                (" " + code, "code"),
            ]
        elif task == "cot":
            sample.target = [
                ("Transcription:\n" + trans + "\nFinal answer correct:", "aux"),
                (fac, "fac"),
                ("\nFindings:\n" + findings_reasoned(g("misconceptions")) + "\nRoot:", "aux"),
                (" " + code, "code"),
            ]
        elif task == "why":
            sample.target = [
                (fac, "fac"),
                ("\nError:", "aux"),
                (" " + why_text(code, g("misconceptions")), "why"),
                ("\nRoot:", "aux"),
                (" " + code, "code"),
                ("\nTranscription:\n" + trans + "\nFindings:\n" + finds, "aux"),
            ]
        elif task == "read":
            sample.target = [("Transcription:\n" + trans + "\nFinal answer correct:", "aux"), (fac, "fac")]
        elif task == "fac":
            sample.target = [(fac, "fac")]
    return sample
