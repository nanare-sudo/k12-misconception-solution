"""Build samples from table rows: load images, attach examples of the same question.

A `SampleBuilder` knows the data root, the taxonomy and a **reference set** of labelled rows from which examples of
the same question are drawn. For a validation or test row this is the whole training part. For a row that is itself
part of the reference set: **never leave-one-out**. Its examples come only from the other CV folds
(`reference_folds` and `exclude_fold`), not from "all rows except itself". Otherwise the model learns from the gap
that its own row leaves in the examples (leave-one-out target leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import common as C
from . import prompts as P


@dataclass
class ImageConfig:
    max_side: int = 1024
    min_side: int | None = None
    question_max_side: int = 448
    # Orientation normalisation: trace_id -> quarter turns counter-clockwise for the row's photo; set by the runner
    # from `image.rotate_from` (tools/make_orient.py). Question diagrams are left untouched.
    rotate: dict | None = None
    rotate_all: int = 0  # additionally rotate EVERY photo by this many quarter turns counter-clockwise (orientation probe "up")


class SampleBuilder:
    def __init__(
        self,
        data_root,
        tax: pd.DataFrame,
        *,
        tax_style: str = "full",
        image: ImageConfig | None = None,
        reference: pd.DataFrame | None = None,
        reference_folds: np.ndarray | None = None,
        max_examples: int = 4,
        tax_cases: int = 0,
        tax_cases_for: dict | None = None,
        tax_case_chars: int = 200,
        tax_case_topic: bool = False,
        case_exclude_fold: int | None = None,
        topic_codes: int = 0,
        topic_none: bool = False,
        question_codes: int = 0,
        question_soft: bool = False,
        question_allowed: set | None = None,
        teacher_examples: int = 0,
        think_suffix: str = "",
        think_effort: str = "",
        instruction_suffix: str = "",
        candidates: dict | None = None,
        candidates_k: int = 5,
    ):
        """tax_cases > 0: put this many annotator notes per code from the reference set into the taxonomy header.
        The header is the same for all rows, so the cases must not come from a fold that is evaluated
        (`case_exclude_fold`, if the reference set contains that fold).
        tax_cases_for: {code: count}, an individual count for single codes (case book for catch-all codes);
        tax_case_chars: length of one note; tax_case_topic: topic before the note.
        instruction_suffix: text appended to the instruction of EVERY task with a taxonomy header (e.g. a strictness rule)."""
        self.root = data_root
        self.instruction_suffix = instruction_suffix.strip()
        self.think_suffix = think_suffix  # appended to the instruction of the think task (e.g. a length hint for the reasoning)
        # Reasoning effort for the think task: the Qwen3.8 chat template knows reasoning_effort = xhigh (default),
        # medium and low, and puts one sentence per level into the system text. Empty = template default.
        if think_effort not in ("", "xhigh", "medium", "low"):
            raise ValueError(f"think_effort: unknown level {think_effort!r}")
        self.think_effort = think_effort
        # Candidates from an earlier stage per row: {trace_id: [code, …]} (best first); the first `candidates_k` go into
        # the prompt. For training rows they must come from out-of-fold predictions (eda/c9_candidates.py).
        self.candidates = candidates
        self.candidates_k = int(candidates_k)
        cases = None
        if (tax_cases > 0 or tax_cases_for) and reference is not None:
            src = reference
            if case_exclude_fold is not None:
                if reference_folds is None:
                    raise ValueError("case_exclude_fold requires reference_folds")
                src = reference[reference_folds != case_exclude_fold]
            if "_source" in src.columns:  # notes in the header come only from annotators, never from a teacher model
                src = src[src["_source"] != "teacher"]
            cases = P.code_cases(src, tax_cases, max_chars=int(tax_case_chars), per_code_for=tax_cases_for, with_topic=bool(tax_case_topic))
        self.tax_text = P.taxonomy_text(tax, tax_style, cases=cases)
        self.image = image or ImageConfig()
        self.max_examples = max_examples
        self.topic_codes = topic_codes  # > 0: root errors of other questions of the same topic in the prompt (this many codes)
        self.topic_none = topic_none  # plus the number of workings without a major error as the first line
        # > 0: teacher rows of the same question as a condensed block (this many codes); examples then only from labelled rows
        self.question_codes = question_codes
        self.question_soft = question_soft  # question block with the teacher's second and third guesses (prompts.question_lines)
        self.question_allowed = question_allowed  # codes that can be a root (only for the soft block)
        # > 0: additionally this many teacher-labelled workings of the same question as written-out examples (one per error code)
        self.teacher_examples = teacher_examples
        self.reference = reference
        self.reference_folds = reference_folds
        self._by_question: dict[str, np.ndarray] = {}
        self._by_topic: dict[str, np.ndarray] = {}
        self._teacher_by_question: dict[str, np.ndarray] = {}
        self._is_teacher = None
        if reference is not None:
            if "_source" in reference.columns:
                self._is_teacher = (reference["_source"] == "teacher").to_numpy()
            for q, idx in reference.groupby("question_id").indices.items():
                idx = np.asarray(idx)
                if self._is_teacher is not None:
                    self._teacher_by_question[q] = idx[self._is_teacher[idx]]
                    if question_codes > 0:  # teacher rows go into the block, not among the examples
                        idx = idx[~self._is_teacher[idx]]
                self._by_question[q] = idx
            for t, idx in reference.groupby("topic_name").indices.items():
                idx = np.asarray(idx)
                # topic block only from labelled rows (as measured): teacher rows of the pool have a different class mix
                self._by_topic[t] = idx if self._is_teacher is None else idx[~self._is_teacher[idx]]

    def question_block(self, row) -> str:
        """Teacher findings for pool workings of the same question (without the row itself if it is a pool row)."""
        if self.reference is None or self.question_codes <= 0:
            return ""
        idx = self._teacher_by_question.get(row["question_id"])
        if idx is None or len(idx) == 0:
            return ""
        ref = self.reference.iloc[idx]
        return P.question_lines(ref[(ref["trace_id"] != row["trace_id"]).to_numpy()], self.question_codes, soft=self.question_soft,
                                allowed=self.question_allowed)

    def topic_text(self, row, exclude_fold: int | None = None) -> str:
        """Root errors of OTHER questions of the same topic from the reference set (never from the excluded fold)."""
        if self.reference is None or self.topic_codes <= 0:
            return ""
        idx = self._by_topic.get(row["topic_name"])
        if idx is None:
            return ""
        ref = self.reference.iloc[idx]
        other = (ref["question_id"] != row["question_id"]).to_numpy()
        if exclude_fold is not None and self.reference_folds is not None:
            other = other & (self.reference_folds[idx] != exclude_fold)
        is_none = (ref["root_code"] == "NONE").to_numpy()
        n_none = int((other & is_none).sum()) if self.topic_none else None
        return P.topic_lines(ref[other & ~is_none], self.topic_codes, n_none=n_none)

    def teacher_examples_text(self, row) -> str:
        """Written-out teacher examples of the same question: per error code the working the teacher was most confident
        about, most frequent codes first; without NONE (correct workings need no example) and without the row itself.
        Adds what the question block lacks: what such a working looks like."""
        if self.reference is None or self.teacher_examples <= 0:
            return ""
        idx = self._teacher_by_question.get(row["question_id"])
        if idx is None or len(idx) == 0:
            return ""
        ref = self.reference.iloc[idx]
        ref = ref[((ref["trace_id"] != row["trace_id"]) & (ref["root_code"] != "NONE")).to_numpy()]
        if len(ref) == 0:
            return ""
        counts = ref["root_code"].value_counts()
        conf = ref["_conf"].fillna(-1.0) if "_conf" in ref.columns else 0.0
        ref = ref.assign(_c=conf, _n=ref["root_code"].map(counts)).sort_values(["_n", "root_code", "_c", "trace_id"], ascending=[False, True, False, True])
        chosen = ref.drop_duplicates("root_code").head(self.teacher_examples)
        return P.example_lines(chosen.drop(columns=["_c", "_n"]), self.teacher_examples)

    def examples_text(self, row, exclude_fold: int | None = None) -> str:
        """Labelled workings of the same question from the reference set (without the row itself)."""
        if self.reference is None or self.max_examples <= 0:
            return ""
        idx = self._by_question.get(row["question_id"])
        if idx is None:
            return ""
        ref = self.reference.iloc[idx]
        keep = (ref["trace_id"] != row["trace_id"]).to_numpy()
        if exclude_fold is not None and self.reference_folds is not None:
            keep = keep & (self.reference_folds[idx] != exclude_fold)  # not in place: pandas 3 returns read-only arrays
        return P.example_lines(ref[keep], self.max_examples)

    def build(
        self,
        row,
        *,
        split: str,
        task: str,
        with_target: bool = False,
        with_image: bool = True,
        with_examples: bool = True,
        exclude_fold: int | None = None,
        reading: str | None = None,
        analysis: str = "",
        reading_head: str = "",
        analysis_head: str = "",
    ) -> P.Sample:
        student = None
        if with_image:
            student = C.open_image(
                C.image_path(self.root, split, row["image_file"]), self.image.max_side, self.image.min_side
            )
            turns = (int((self.image.rotate or {}).get(str(row["trace_id"]), 0)) + int(self.image.rotate_all)) % 4
            if turns:
                student = student.rotate(90 * turns, expand=True)  # lossless, the long side is kept
        qimg = None
        if row["question_image_file"] and task != "up":
            qimg = C.open_image(C.image_path(self.root, "questions", row["question_image_file"]), self.image.question_max_side)
        ex = self.examples_text(row, exclude_fold) if (with_examples and task != "up") else ""
        sample = P.build_sample(
            row, task=task, tax_text=self.tax_text, student_image=student, question_image=qimg,
            examples_text=ex, topic_text=self.topic_text(row, exclude_fold), question_block=self.question_block(row),
            teacher_examples_text=self.teacher_examples_text(row) if with_examples else "", reading=reading, with_target=with_target,
            instruction_suffix=" ".join(t for t in (self.instruction_suffix, self.think_suffix if task == "think" else "") if t),
            candidates_text=", ".join((self.candidates.get(str(row["trace_id"])) or [])[: self.candidates_k]) if self.candidates else "",
            analysis=P.clean_analysis(analysis) if analysis else "", reading_head=reading_head, analysis_head=analysis_head,
        )
        if task == "think" and self.think_effort:
            sample.meta["effort"] = self.think_effort  # read by engine.chat_text and think_vllm.request_body
        return sample
