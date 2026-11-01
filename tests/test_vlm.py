"""Tests of the vision-language model code that run without a model and without competition data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vlm import prompts as P  # noqa: E402
from vlm.engine import CandidateSet, _TrieProcessor, probs_from_logp  # noqa: E402
from vlm.runner import _auc, _transcription_part, _upto_root, select_by_question, select_rows  # noqa: E402


class FakeTokenizer:
    """Tokenises into single characters; enough to check the prefix tree and the mapping."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


ROW = {
    "trace_id": "t1", "question_id": "q1", "grade": "7", "topic_name": "Fractions",
    "question_text": "1/2 + 1/3 = ...", "question_image_file": "", "correct_answer": "5/6",
    "root_code": "FRA-01", "final_answer_correct": "False",
    "transcription_steps": "L1 | 1/2 + 1/3\nL2 | = 2/5",
    "misconceptions": json.dumps([{"step": "L2", "span": "2/5", "category": "FRA-01", "severity": "major", "root": True, "text": "Added tops and bottoms."}]),
}
TAX = pd.DataFrame({"id": ["FRA-01", "CAL-01"], "name": ["Adding tops and bottoms", "Calculation slip"], "description": ["d1", "d2"]})


def test_candidate_set_trie_and_index():
    cs = CandidateSet(FakeTokenizer(), [" AB", " AC", " D"], terminator="\n")
    assert cs.term_id == ord("\n") and len(cs) == 3 and cs.max_len == 4
    assert cs.trie[()] == [ord(" ")]
    assert cs.trie[(ord(" "),)] == sorted([ord("A"), ord("D")])
    assert cs.trie[(ord(" "), ord("A"))] == sorted([ord("B"), ord("C")])
    assert cs.index[tuple(map(ord, " AC\n"))] == 1


def test_candidate_set_rejects_prefix_and_duplicates():
    with pytest.raises(ValueError):
        CandidateSet(FakeTokenizer(), [" A", " A"])
    with pytest.raises(ValueError):
        CandidateSet(FakeTokenizer(), [" A", " A\nB"])  # "A + terminator" would be a prefix of the second one


def test_trie_processor_masks_and_frees_terminator():
    cs = CandidateSet(FakeTokenizer(), [" AB", " D"], terminator="\n")
    proc = _TrieProcessor(cs, prompt_len=2)
    scores = torch.full((3, 128), -1.0)
    ids = torch.tensor([[1, 2, ord(" "), 0], [1, 2, ord(" "), ord("D")], [1, 2, ord(" "), ord("A")]])
    out = proc(ids[:1, :3], scores[:1])  # after " ": A or D
    assert torch.isfinite(out[0]).nonzero().flatten().tolist() == sorted([ord("A"), ord("D")])
    assert out[0, ord("A")].item() == -1.0  # the model value is kept
    out = proc(ids[1:2], scores[:1])  # " D" is complete: the terminator costs nothing
    assert out[0, cs.term_id].item() == 0.0 and torch.isfinite(out[0]).sum().item() == 1
    out = proc(ids[2:3], scores[:1])  # " A": only B
    assert torch.isfinite(out[0]).nonzero().flatten().tolist() == [ord("B")]


def test_probs_from_logp_normalises_and_floors():
    lp = np.array([[-1.0, -2.0, -np.inf], [-np.inf, -0.5, -np.inf]])
    p = probs_from_logp(lp, floor=1e-6)
    assert np.allclose(p.sum(1), 1.0) and (p > 0).all()
    assert p[0, 0] > p[0, 1] >= p[0, 2] and p[1].argmax() == 1  # missing candidates never ahead of returned ones


def test_probs_from_logp_residual_mass_and_absent_columns():
    lp = np.full((30, 6), -np.inf)
    lp[:, 0], lp[:, 1] = np.log(0.5), np.log(0.3)
    lp[:10, 2] = np.log(0.1)
    p = probs_from_logp(lp)
    assert np.allclose(p[:, 3:], 0.0)  # columns that were never a candidate stay at 0
    assert np.allclose(p[20], [0.5, 0.3, 0.2, 0, 0, 0])  # the remaining mass goes to the dropped candidates
    assert np.allclose(p[0, :3], np.array([0.5, 0.3, 0.1]) / 0.9)


def test_build_sample_diag_target_segments():
    s = P.build_sample(ROW, task="diag", tax_text=P.taxonomy_text(TAX, "full"), student_image=object(), with_target=True,
                       examples_text="- working: x\n  final answer wrong; root: CAL-01")
    kinds = [k for k, _ in s.parts]
    assert kinds == ["text", "text", "image", "text"] and s.prefix == "Root:"
    assert "FRA-01 | Adding tops and bottoms | d1" in s.parts[0][1] and "NONE |" in s.parts[0][1]
    assert "Diagnosed working of other students" in s.parts[1][1]
    assert [tag for _, tag in s.target] == ["code", "aux", "fac", "aux"]
    assert s.target[0][0] == " FRA-01" and s.target[2][0] == " no"
    assert "L2 | = 2/5" in s.target[3][0] and "- L2 [FRA-01, major, root]: Added tops and bottoms." in s.target[3][0]


def test_build_sample_full_and_text_only():
    s = P.build_sample(ROW, task="full", tax_text="T", student_image=object(), with_target=True)
    assert s.prefix == "" and [tag for _, tag in s.target] == ["aux", "fac", "aux", "code"]
    assert s.target[2][0].endswith("\nRoot:") and s.target[3][0] == " FRA-01"
    t = P.build_sample(ROW, task="diag", tax_text="T", student_image=None, reading="L1 | a")
    assert all(k == "text" for k, _ in t.parts) and "Transcription of the student's working:\nL1 | a" in t.parts[-1][1]
    f = P.build_sample(ROW, task="fac", tax_text="T", student_image=object(), with_target=True)
    assert "taxonomy" not in f.parts[0][1].lower() and f.target == [(" no", "fac")]


def test_question_image_comes_before_student_image():
    s = P.build_sample(ROW, task="diag", tax_text="T", student_image="S", question_image="Q")
    images = [v for k, v in s.parts if k == "image"]
    assert images == ["Q", "S"]


def test_findings_text_handles_empty_and_broken():
    assert P.findings_text("[]") == "- none" and P.findings_text("") == "- none" and P.findings_text("{broken") == "- none"
    assert P.yes_no("True") == " yes" and P.yes_no("False") == " no"


def test_example_lines_prefers_variety():
    df = pd.DataFrame([
        dict(ROW, trace_id="a", root_code="FRA-01"), dict(ROW, trace_id="b", root_code="FRA-01"),
        dict(ROW, trace_id="c", root_code="NONE", final_answer_correct="True", misconceptions="[]"),
    ])
    txt = P.example_lines(df, max_examples=2)
    assert txt.count("- working:") == 2 and "root: NONE" in txt and "root: FRA-01 (Added tops and bottoms.)" in txt
    assert P.example_lines(df.iloc[:0]) == ""


def test_runner_text_helpers():
    gen = "Transcription:\nL1 | a\nL2 | b\nFinal answer correct: no\nFindings:\n- L2 [CAL-01, major]: x\nRoot: CAL-01"
    assert _upto_root(gen).endswith("- L2 [CAL-01, major]: x\nRoot:") and _upto_root("") == "Root:"
    assert _upto_root("Transcription:\nL1 | a") == "Transcription:\nL1 | a\nRoot:"
    assert _transcription_part("Transcription:\nL1 | a\nL2 | b\nFinal answer correct: yes") == "L1 | a\nL2 | b"
    assert _upto_root("still thinking", close_think=True) == "still thinking\n</think>\n\nRoot:"
    assert _upto_root("x\n</think>\n\nRoot: CAL-01", close_think=True) == "x\n</think>\n\nRoot:"


def test_select_rows_and_auc():
    folds = np.array([0, 1, 2, 0, 1, 2])
    assert select_rows({"folds": [1]}, 6, folds).tolist() == [1, 4]
    a = select_rows({"sample": 3, "seed": 5}, 6, folds)
    assert len(a) == 3 and a.tolist() == select_rows({"sample": 3, "seed": 5}, 6, folds).tolist()
    assert select_rows({"limit": 2}, 6, None).tolist() == [0, 1]


def test_select_rows_only_file(tmp_path):
    """rows.only restricts to table positions from a file, before limit and shard, never for subsets."""
    np.save(tmp_path / "keep.npy", np.array([5, 1, 4]))
    folds = np.array([0, 1, 2, 0, 1, 2])
    sel = {"only": str(tmp_path / "keep.npy")}
    assert select_rows(sel, 6, folds).tolist() == [1, 4, 5]
    assert select_rows({**sel, "folds": [1]}, 6, folds).tolist() == [1, 4]
    assert select_rows({**sel, "limit": 2}, 6, folds).tolist() == [1, 4]
    with pytest.raises(ValueError):
        select_rows(sel, 6, None, whole_table=False)
    df = pd.DataFrame({"question_id": list("aabbcc")})
    assert select_by_question(df, {"only": "keep.npy"}, tmp_path, None).tolist() == [1, 4, 5]
    assert select_by_question(df, {"only": "keep.npy", "shard": [1, 2]}, tmp_path, None).tolist() == [4]
    assert _auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([True, True, False, False])) == 1.0
    assert _auc(np.array([1.0, 2.0]), np.array([True, True])) is None


def test_select_by_question_caps_and_shards():
    df = pd.DataFrame({"question_id": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})

    class FakeC:
        @staticmethod
        def load_table(root, name):
            return pd.DataFrame({"question_id": ["a", "c"]})

    idx = select_by_question(df, {"questions_from": "test", "per_question": 2, "seed": 1}, None, FakeC)
    q = df["question_id"].iloc[idx].tolist()
    assert sorted(q) == ["a", "a", "c", "c"] and list(idx) == sorted(idx)
    assert list(idx) == list(select_by_question(df, {"questions_from": "test", "per_question": 2, "seed": 1}, None, FakeC))
    a = select_by_question(df, {"per_question": 2, "shard": [0, 2]}, None, FakeC)
    b = select_by_question(df, {"per_question": 2, "shard": [1, 2]}, None, FakeC)
    assert len(a) + len(b) == 6 and not set(a) & set(b)

    class FakeFolds:  # questions of a CV fold: a lies in fold 0, c in fold 1
        @staticmethod
        def load_table(root, name):
            return pd.DataFrame({"question_id": ["a", "c"]})

        @staticmethod
        def load_folds(n):
            return np.array([0, 1])

    only_c = select_by_question(df, {"questions_from": "train", "questions_fold": 1}, None, FakeFolds)
    assert df["question_id"].iloc[only_c].tolist() == ["c", "c"]
    with pytest.raises(ValueError):
        select_by_question(df, {"questions_from": "test", "questions_fold": 1}, None, FakeFolds)


def test_cot_target_reasons_before_naming():
    s = P.build_sample(ROW, task="cot", tax_text="T", student_image=object(), with_target=True)
    assert [tag for _, tag in s.target] == ["aux", "fac", "aux", "code"] and s.prefix == ""
    findings = s.target[2][0]
    assert "- L2: Added tops and bottoms. => FRA-01 (major, root)" in findings and findings.endswith("\nRoot:")
    assert findings.index("Added tops") < findings.index("FRA-01")  # reasoning first, then the code
    assert P.findings_reasoned("[]") == "- none"


def test_why_target_describes_before_naming():
    s = P.build_sample(ROW, task="why", tax_text="T", student_image=object(), with_target=True,
                       examples_text="- working: x\n  final answer wrong; root: CAL-01")
    assert s.prefix == "Final answer correct:" and "Diagnosed working of other students" in s.parts[0][1] + s.parts[1][1]
    assert [tag for _, tag in s.target] == ["fac", "aux", "why", "aux", "code", "aux"]
    assert s.target[0][0] == " no" and s.target[1][0] == "\nError:" and s.target[2][0] == " Added tops and bottoms."
    assert s.target[3][0] == "\nRoot:" and s.target[4][0] == " FRA-01" and "FRA-01" not in s.target[2][0]
    none = P.build_sample(dict(ROW, root_code="NONE", final_answer_correct="True", misconceptions="[]"), task="why", tax_text="T",
                          student_image=object(), with_target=True)
    assert none.target[0][0] == " yes" and none.target[2][0] == " none" and none.target[4][0] == " NONE"
    assert P.why_text("CAL-01", "[]") == "not described"


def test_code_cases_in_taxonomy_head():
    df = pd.DataFrame([
        dict(ROW, trace_id="a", question_id="q1"), dict(ROW, trace_id="b", question_id="q1", misconceptions=json.dumps(
            [{"step": "L1", "category": "FRA-01", "severity": "major", "root": True, "text": "Second note."}])),
        dict(ROW, trace_id="c", question_id="q2", misconceptions=json.dumps(
            [{"step": "L1", "category": "FRA-01", "severity": "major", "root": True, "text": "Other question."}])),
        dict(ROW, trace_id="d", root_code="NONE", misconceptions="[]"),
    ])
    cases = P.code_cases(df, per_code=2)
    assert cases == {"FRA-01": ["Added tops and bottoms.", "Other question."]}  # different questions first, NONE without cases
    txt = P.taxonomy_text(TAX, "full", cases=cases)
    lines = txt.splitlines()
    assert lines[0].startswith("(Indented lines") and lines[2] == "    e.g. Added tops and bottoms." and lines[3] == "    e.g. Other question."
    assert lines[4].startswith("CAL-01 |") and P.taxonomy_text(TAX, "full").splitlines()[0].startswith("FRA-01 |")
    assert P.code_cases(df, per_code=0) == {}


def test_topic_lines_and_prompt_block():
    other = pd.DataFrame([
        dict(ROW, trace_id="a", question_id="q2", root_code="CAL-01", misconceptions=json.dumps(
            [{"step": "L1", "category": "CAL-01", "severity": "major", "root": True, "text": "7x8=54."}])),
        dict(ROW, trace_id="b", question_id="q3", root_code="CAL-01", misconceptions="[]"),
        dict(ROW, trace_id="c", question_id="q3", root_code="FRA-01"),
    ])
    txt = P.topic_lines(other, max_codes=2)
    assert txt.splitlines() == ["- CAL-01 (2x): 7x8=54.", "- FRA-01 (1x): Added tops and bottoms."]
    assert P.topic_lines(other, max_codes=1).count("\n") == 0 and P.topic_lines(other.iloc[:0]) == ""
    s = P.build_sample(ROW, task="diag", tax_text="T", student_image=object(), topic_text=txt, examples_text="- working: x")
    mid = s.parts[1][1]
    assert mid.index("OTHER questions on this topic") < mid.index("on this same question") and "- CAL-01 (2x): 7x8=54." in mid
    f = P.build_sample(ROW, task="fac", tax_text="T", student_image=object(), topic_text=txt)
    assert "OTHER questions" not in f.parts[0][1] + f.parts[1][1]


def test_topic_lines_none_header():
    other = pd.DataFrame([
        dict(ROW, trace_id="a", question_id="q2", root_code="CAL-01", misconceptions=json.dumps(
            [{"step": "L1", "category": "CAL-01", "severity": "major", "root": True, "text": "7x8=54."}])),
        dict(ROW, trace_id="b", question_id="q3", root_code="CAL-01", misconceptions="[]"),
        dict(ROW, trace_id="c", question_id="q3", root_code="FRA-01"),
    ])
    with_none = P.topic_lines(other, max_codes=1, n_none=5).splitlines()
    assert with_none == ["- NONE (5x of 8 marked workings): no major error", "- CAL-01 (2x): 7x8=54."]
    assert P.topic_lines(other.iloc[:0], n_none=2) == "- NONE (2x of 2 marked workings): no major error"
    assert P.topic_lines(other.iloc[:0]) == ""


def test_teacher_rows_extend_reference_gold_first(tmp_path):
    from vlm import common as C
    from vlm.data import SampleBuilder

    pool = pd.DataFrame([dict(ROW, trace_id=f"p{i}", question_id="q1", image_file=f"p{i}.jpeg") for i in range(3)]).drop(
        columns=["root_code", "final_answer_correct", "transcription_steps", "misconceptions"])
    d = tmp_path / "teach"
    d.mkdir()
    (d / "labels.jsonl").write_text(json.dumps({
        "row": 1, "final_answer_correct": False, "transcription": ["L1 | 1/2 + 1/3", "L2 | = 2/6"], "root_code": "CAL-01", "top3": ["CAL-01"],
        "findings": [{"step": "L2", "code": "CAL-01", "severity": "major", "root": True, "text": "Slip in the sum."}]}) + "\n")
    orig = C.load_table
    C.load_table = lambda root, name: pool
    try:
        t = C.load_teacher_rows(tmp_path, "x", teacher_dir=d)
    finally:
        C.load_table = orig
    assert list(t["trace_id"]) == ["p1"] and t["root_code"].iloc[0] == "CAL-01" and t["final_answer_correct"].iloc[0] == "False"
    assert t["transcription_steps"].iloc[0] == "L1 | 1/2 + 1/3\nL2 | = 2/6" and t["_source"].iloc[0] == "teacher"
    assert P.root_reason(t["misconceptions"].iloc[0]) == "Slip in the sum."

    gold = pd.DataFrame([dict(ROW, trace_id="zz", question_id="q1"), dict(ROW, trace_id="yy", question_id="q1", root_code="CAL-01")]).assign(_source="gold")
    both = pd.concat([gold, t], ignore_index=True)
    b = SampleBuilder(tmp_path, TAX, reference=both, reference_folds=np.array([0, 1, -1]), max_examples=4, tax_cases=2)
    ex = b.examples_text(dict(ROW, trace_id="new", question_id="q1"), exclude_fold=1)
    assert ex.count("- working:") == 2 and "root: FRA-01" in ex and "Slip in the sum." in ex  # fold 1 is missing, the teacher row stays
    assert ex.index("root: FRA-01") < ex.index("root: CAL-01")  # labelled row before the teacher row
    assert "Slip in the sum." not in b.tax_text and "Added tops and bottoms." in b.tax_text  # notes in the header only from annotators


def _teacher_frame():
    def find(text):
        return json.dumps([{"step": "L1", "category": "X", "severity": "major", "root": True, "text": text}])

    rows = [
        dict(ROW, trace_id="p1", question_id="q1", root_code="NONE", misconceptions="[]", _conf=0.9, _answer="12"),
        dict(ROW, trace_id="p2", question_id="q1", root_code="CAL-01", misconceptions=find("7x8=54."), _conf=0.5, _answer="54"),
        dict(ROW, trace_id="p3", question_id="q1", root_code="CAL-01", misconceptions=find("Carried wrongly."), _conf=0.8, _answer=" 54 "),
        dict(ROW, trace_id="p4", question_id="q1", root_code="CAL-01", misconceptions=find("Other slip."), _conf=float("nan"), _answer="45"),
        dict(ROW, trace_id="p5", question_id="q1", root_code="FRA-01", misconceptions=find("Added tops and bottoms."), _conf=0.7, _answer=""),
        dict(ROW, trace_id="p6", question_id="q2", root_code="ALG-01", misconceptions=find("Sign lost."), _conf=0.7, _answer="x=3"),
    ]
    return pd.DataFrame(rows).assign(_source="teacher")


def test_question_lines_condense_teacher_rows():
    t = _teacher_frame()
    q1 = t[t["question_id"] == "q1"]
    assert P.question_lines(q1, max_codes=8).splitlines() == [
        "- NONE (1x of 5 workings): no major error",
        "- CAL-01 (3x); student answers: 54 | 45: Carried wrongly.",  # note of the most confident working; final answers by frequency
        "- FRA-01 (1x): Added tops and bottoms.",
    ]
    assert P.question_lines(q1, max_codes=1).count("\n") == 1 and P.question_lines(q1.iloc[:0]) == "" and P.question_lines(q1, max_codes=0) == ""


def test_question_block_keeps_teacher_rows_out_of_examples_and_topic(tmp_path):
    from vlm.data import SampleBuilder

    gold = pd.DataFrame([dict(ROW, trace_id="g1", question_id="q1", root_code="FRA-01"),
                         dict(ROW, trace_id="g2", question_id="q2", root_code="ALG-01")]).assign(_source="gold")
    both = pd.concat([gold, _teacher_frame()], ignore_index=True)
    folds = np.array([0, 1] + [-1] * 6)
    row = dict(ROW, trace_id="new", question_id="q1")
    b = SampleBuilder(tmp_path, TAX, reference=both, reference_folds=folds, max_examples=4, topic_codes=5, topic_none=True, question_codes=8)
    block = b.question_block(row)
    assert block.startswith("- NONE (1x of 5 workings)") and "CAL-01 (3x)" in block and "ALG-01" not in block
    ex = b.examples_text(row, exclude_fold=3)
    assert ex.count("- working:") == 1 and "root: FRA-01" in ex  # only the labelled row; teacher rows are in the block
    assert b.topic_text(row, exclude_fold=3) == "- NONE (0x of 1 marked workings): no major error\n- ALG-01 (1x): Added tops and bottoms."  # without teacher rows
    s = P.build_sample(row, task="diag", tax_text="T", student_image=object(), topic_text="t", question_block=block, examples_text=ex)
    mid = s.parts[1][1]
    assert mid.index("OTHER questions on this topic") < mid.index("on this SAME question") < mid.index("Diagnosed working of other students")
    # without question_codes the old behaviour stays: teacher rows fill up the examples, no block
    b0 = SampleBuilder(tmp_path, TAX, reference=both, reference_folds=folds, max_examples=4)
    assert b0.question_block(row) == "" and b0.examples_text(row, exclude_fold=3).count("- working:") == 4
    # a pool row does not see itself in the block
    assert "of 4 workings" in b.question_block(dict(ROW, trace_id="p1", question_id="q1"))


def test_select_teacher_rows_balances_none_and_weights():
    from vlm.runner import select_teacher_rows

    classes = ["NONE", "CAL-01", "FRA-01", "ALG-01"]
    rows = []
    for k in range(40):  # 30 NONE, 6 CAL-01, 3 FRA-01, 1 ALG-01 (unknown to the training part)
        code = "NONE" if k < 30 else "CAL-01" if k < 36 else "FRA-01" if k < 39 else "ALG-01"
        rows.append(dict(ROW, trace_id=f"p{k:02d}", question_id=f"q{k % 2}", root_code=code, _conf=0.5 + 0.01 * k, _source="teacher", _answer=""))
    t = pd.DataFrame(rows)
    seen = np.array([True, True, True, False])
    sel, w = select_teacher_rows(t, {"min_conf": 0.6, "none_share": 0.4, "weight": 0.5, "seed": 1}, seen, classes)
    assert "ALG-01" not in set(sel["root_code"]) and (sel["_conf"] >= 0.6).all()
    n_err = int((sel["root_code"] != "NONE").sum())
    assert n_err == 9 and int((sel["root_code"] == "NONE").sum()) == 6  # 9 error rows -> 6 NONE give 40 %
    assert np.allclose(w, 0.5 * sel["_conf"].to_numpy()) and "_c" not in sel.columns
    sel2, _ = select_teacher_rows(t, {"min_conf": 0.6, "none_share": 0.4, "weight": 0.5, "seed": 1}, seen, classes)
    assert list(sel2["trace_id"]) == list(sel["trace_id"])  # same selection with the same seed
    capped, w1 = select_teacher_rows(t, {"per_question_code": 1, "none_share": 0.5, "weight": 1.0, "conf_weight": False}, seen, classes)
    assert capped.groupby(["question_id", "root_code"]).size().max() == 1 and np.allclose(w1, 1.0)
    # separate cap for NONE: error codes stay at one row per question and code, NONE may contribute more
    wide, _ = select_teacher_rows(t, {"per_question_code": 1, "per_question_none": 5, "none_share": 0.5, "weight": 1.0, "conf_weight": False}, seen, classes)
    by = wide.groupby(["question_id", "root_code"]).size()
    assert by[[i for i in by.index if i[1] != "NONE"]].max() == 1 and int((wide["root_code"] == "NONE").sum()) == int((wide["root_code"] != "NONE").sum())
    assert int((wide["root_code"] == "NONE").sum()) > int((capped["root_code"] == "NONE").sum())


def test_soft_code_samples_from_teacher_ranking():
    from vlm.runner import soft_code

    row = dict(ROW, root_code="CAL-01", _top3=json.dumps(["CAL-01", "FRA-01", "ALG-01"]), _probs=json.dumps([0.5, 0.3, 0.2]))
    known = {"NONE", "CAL-01", "FRA-01"}  # ALG-01 is unknown to the training part
    rng = np.random.default_rng(0)
    draws = [soft_code(row, rng, known) for _ in range(4000)]
    share_alt = np.mean([d == "FRA-01" for d in draws])
    assert set(draws) == {None, "FRA-01"} and abs(share_alt - 0.3 / 0.8) < 0.03  # the top stays in 5/8 of the draws
    assert soft_code(dict(row, _probs="broken"), rng, known) is None
    assert soft_code(dict(row, root_code="FRA-01"), rng, known) is None  # top of the ranking does not match the label: draw nothing
    assert soft_code(dict(row, _top3=json.dumps(["CAL-01"]), _probs=json.dumps([0.9])), rng, known) is None


def test_teacher_examples_one_per_error_code(tmp_path):
    from vlm.data import SampleBuilder

    gold = pd.DataFrame([dict(ROW, trace_id="g1", question_id="q1", root_code="FRA-01")]).assign(_source="gold")
    both = pd.concat([gold, _teacher_frame()], ignore_index=True)
    b = SampleBuilder(tmp_path, TAX, reference=both, reference_folds=np.array([0] + [-1] * 6), max_examples=4, question_codes=8, teacher_examples=2)
    row = dict(ROW, trace_id="new", question_id="q1")
    txt = b.teacher_examples_text(row)
    # two error codes for q1: CAL-01 (3 rows, the most confident is p3) and FRA-01 (p5); NONE (p1) and the other question are missing
    assert txt.count("- working:") == 2 and "root: CAL-01 (Carried wrongly.)" in txt and "root: FRA-01" in txt and "root: NONE" not in txt and "ALG-01" not in txt
    assert SampleBuilder(tmp_path, TAX, reference=both, reference_folds=np.array([0] + [-1] * 6), question_codes=8).teacher_examples_text(row) == ""
    s = P.build_sample(row, task="diag", tax_text="T", student_image=object(), examples_text="- working: x", teacher_examples_text=txt, question_block="b")
    mid = s.parts[1][1]
    assert mid.index("Diagnosed working of other students") < mid.index("Machine-marked working of other students")
    # a pool row does not see itself: instead of p3 the next most confident CAL-01 working (p2) appears
    assert "root: CAL-01 (7x8=54.)" in b.teacher_examples_text(dict(ROW, trace_id="p3", question_id="q1"))


def test_think_vllm_messages_keep_order_and_pixels(tmp_path):
    import base64
    import io

    from PIL import Image

    from vlm import think_vllm as TV

    img = Image.fromarray((np.arange(12 * 10 * 3) % 251).astype(np.uint8).reshape(12, 10, 3))
    s = P.Sample(key="k", parts=[("text", "A"), ("image", img), ("text", "B")], prefix="", meta={"task": "think", "think": True})
    msgs = TV.to_messages(s)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"][0]["text"] == P.SYSTEM
    kinds = [c["type"] for c in msgs[1]["content"]]
    assert kinds == ["text", "image_url", "text"] and msgs[1]["content"][0]["text"] == "A" and msgs[1]["content"][2]["text"] == "B"
    url = msgs[1]["content"][1]["image_url"]["url"]
    back = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert url.startswith("data:image/png;base64,") and np.array_equal(np.asarray(back), np.asarray(img))  # lossless
    body = TV.request_body(s, "k12", 700)
    assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": True} and body["max_tokens"] == 700
    with pytest.raises(ValueError):
        TV.request_body(P.Sample(key="k", parts=[("text", "A")], prefix="Root:", meta={}), "k12", 10)
    # raw text: split-off reasoning is prepended again; already closed blocks stay as they are
    assert TV.raw_text({"message": {"content": "x\n</think>\n\nRoot: CAL-01"}}) == "x\n</think>\n\nRoot: CAL-01"
    assert TV.raw_text({"message": {"content": "Root: NONE", "reasoning_content": "because"}}) == "because\n</think>\n\nRoot: NONE"
    f = tmp_path / "t_texts.jsonl"
    f.write_text(json.dumps({"row": 7, "text": "a"}) + "\n" + json.dumps({"row": 3, "text": "b"}) + "\n")
    assert TV.load_texts(f) == {7: "a", 3: "b"}


def test_think_texts_cut(tmp_path):
    """Reasoning texts can be truncated afterwards to a token count (character share = token share)."""
    from vlm import think_vllm as TV

    f = tmp_path / "think_texts.jsonl"
    f.write_text(json.dumps({"row": 3, "text": "a" * 100, "out_tokens": 50}) + "\n" + json.dumps({"row": 7, "text": "b" * 40, "out_tokens": 10}) + "\n")
    assert TV.load_texts(f) == {3: "a" * 100, 7: "b" * 40}
    cut = TV.load_texts(f, cut_tokens=25)
    assert cut[3] == "a" * 50 and cut[7] == "b" * 40


def test_think_request_body_sampling():
    """Request to the local server: greedy by default, a random draw only with temperature and seed."""
    from vlm import think_vllm as TV

    s = P.Sample(key="k", parts=[("text", "Question")])
    body = TV.request_body(s, "k12", 700)
    assert body["temperature"] == 0.0 and "seed" not in body and body["chat_template_kwargs"] == {"enable_thinking": True}
    body = TV.request_body(s, "k12", 700, temperature=0.7, seed=5)
    assert body["temperature"] == 0.7 and body["seed"] == 5 and body["max_tokens"] == 700
    assert "top_p" not in body and "top_k" not in body
    body = TV.request_body(s, "k12", 700, temperature=1.0, seed=5, top_p=0.95, top_k=20)
    assert body["temperature"] == 1.0 and body["top_p"] == 0.95 and body["top_k"] == 20


def test_build_sample_instruction_suffix_only_changes_the_last_part():
    """The instruction suffix comes at the very end; the shared header stays unchanged (cache)."""
    img = object()
    a = P.build_sample(ROW, task="think", tax_text="T", student_image=img)
    b = P.build_sample(ROW, task="think", tax_text="T", student_image=img, instruction_suffix=" Keep it short. ")
    assert a.parts[:-1] == b.parts[:-1] and a.meta["static"] == b.meta["static"]
    assert b.parts[-1][1] == a.parts[-1][1] + " Keep it short." and b.meta["think"] is True


def test_think_effort_reaches_server_request_and_only_think_tasks():
    """The reasoning effort is stored in the sample and passed to the server's chat template as reasoning_effort."""
    from vlm import think_vllm as TV

    s = P.Sample(key="k", parts=[("text", "Question")], meta={"think": True, "effort": "low"})
    assert TV.request_body(s, "k12", 500)["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
    s = P.Sample(key="k", parts=[("text", "Question")], meta={"think": True})
    assert TV.request_body(s, "k12", 500)["chat_template_kwargs"] == {"enable_thinking": True}


def test_build_sample_candidates_block_sits_between_image_and_instruction():
    """Candidates from an earlier stage come after the image and before the instruction; the shared header stays the same."""
    img = object()
    a = P.build_sample(ROW, task="think", tax_text="T", student_image=img)
    b = P.build_sample(ROW, task="think", tax_text="T", student_image=img, candidates_text="CAL-01, NONE, FRA-01")
    assert a.parts[:-1] == b.parts[:-1] and a.meta["static"] == b.meta["static"]
    tail = b.parts[-1][1]
    assert "CAL-01, NONE, FRA-01\n" in tail and tail.index("CAL-01, NONE") < tail.index(P.INSTRUCTIONS["think"])
    assert P.build_sample(ROW, task="fac", tax_text="T", student_image=img, candidates_text="CAL-01").parts[-1][1].count("CAL-01") == 0


def test_floor_probs_limits_the_veto_of_a_sure_channel():
    """Floor for an almost always confident channel: the sum stays 1, disallowed classes stay at 0, eps = 0 changes nothing."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("a2", Path(__file__).resolve().parent.parent / "eda" / "a2_probe_stack.py")
    a2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a2)
    P = np.array([[1.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]])
    allowed = np.array([True, True, True, False])
    assert a2.floor_probs(P, 0.0, allowed) is P
    Q = a2.floor_probs(P, 0.03, allowed)
    assert np.allclose(Q.sum(1), 1.0) and np.all(Q[:, 3] == 0.0)
    assert np.allclose(Q[0], [0.97 + 0.01, 0.01, 0.01, 0.0])
    assert Q[1, 1] > Q[1, 0] > Q[1, 2] > 0


def test_question_lines_soft_counts_second_and_third_guesses():
    """Soft counting: codes by summed probability over the error rows; considered codes are added with 0x."""
    t = _teacher_frame()
    q1 = t[t["question_id"] == "q1"].copy()
    q1["_top3"] = [json.dumps(v) for v in (["NONE", "NOT-01", "INT-04"], ["CAL-01", "INT-04", "NONE"], ["CAL-01", "FRA-01", "INT-04"],
                                          ["CAL-01", "INT-04", "OPS-01"], ["INT-04", "FRA-01", "CAL-01"])]  # last row: root FRA-01 not at the top
    q1["_probs"] = [json.dumps(v) for v in ([0.9, 0.05, 0.05], [0.5, 0.4, 0.1], [0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.5, 0.3, 0.2])]
    lines = P.question_lines(q1, max_codes=8, soft=True).splitlines()
    assert lines[0] == "- NONE (1x of 5 workings): no major error"
    assert lines[1].startswith("- CAL-01 (3x; 1x as second or third guess); student answers: 54 | 45: Carried wrongly.")
    assert lines[2] == "- INT-04 (0x; 4x as second or third guess)"  # mass 1.3: ahead of FRA-01 (0.4), although never chosen as root
    assert lines[3].startswith("- FRA-01 (1x; 1x as second or third guess)")
    assert lines[4] == "- OPS-01 (0x; 1x as second or third guess)" and len(lines) == 5  # NOT-01 only in the error-free row: missing
    assert P.question_lines(q1, max_codes=2, soft=True).count("\n") == 2
    assert "INT-04" not in P.question_lines(q1, max_codes=8, soft=True, allowed={"CAL-01", "FRA-01", "OPS-01"})
    assert P.question_lines(q1, max_codes=8) == P.question_lines(q1.drop(columns=["_top3", "_probs"]), max_codes=8, soft=True)  # without lists: hard counting


def _proto_root(tmp_path):
    """Small data folder with two photos and one diagram for the tests of the teacher protocol."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for split, name, size in (("train", "a.jpeg", (40, 30)), ("train", "b.jpeg", (1500, 600)), ("questions", "q.png", (60, 50))):
        d = tmp_path / "competition_data" / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)).save(d / name)
    rows = [dict(ROW, trace_id="a", image_file="a.jpeg", question_text="Question A"),
            dict(ROW, trace_id="b", image_file="b.jpeg", question_text="Question B", question_image_file="q.png")]
    return pd.DataFrame(rows)


def test_proto_texts_are_the_measured_protocol(tmp_path):
    """Instruction, finish text, user parts and pixels of the in-house path are those of the measured protocol
    (tools/teacher_api.py, tools/open_teacher.py); otherwise the in-house path would measure something else than the probe."""
    import open_teacher as OT
    import teacher_api as T
    from vlm import common as C
    from vlm import proto as PR

    assert PR.INSTRUCTIONS == T.INSTRUCTIONS and PR.FINISH_TEXT == OT.FINISH_TEXT
    system = PR.system_text(TAX)
    assert system == T.INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(TAX, "full")
    df = _proto_root(tmp_path)
    for i, tail in ((0, ""), (1, PR.FINISH_TEXT.format(notes="my notes"))):
        row = df.iloc[i]
        ref = OT.messages_for(tmp_path, row, system, tail)[1]["content"]
        qimg = C.open_image(C.image_path(tmp_path, "questions", row["question_image_file"]), 448) if row["question_image_file"] else None
        mine = PR.user_parts(row, C.open_image(C.image_path(tmp_path, "train", row["image_file"]), 1024), qimg, tail)
        assert [k for k, _ in mine] == [p["type"] for p in ref]
        for (kind, value), p in zip(mine, ref):
            if kind == "text":
                assert value == p["text"]
            else:
                assert value.size == p["image"].size and np.array_equal(np.asarray(value), np.asarray(p["image"]))
    msgs = PR.to_messages(system, mine)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"][0]["text"] == system
    assert [c["type"] for c in msgs[1]["content"]] == ["text", "text", "image_url", "text", "image_url", "text"]
    assert PR.request_body(msgs, 2500, thinking=True)["chat_template_kwargs"] == {"enable_thinking": True}
    assert PR.request_body(msgs, 1500, thinking=False, effort="low", seed=3)["chat_template_kwargs"] == {"enable_thinking": False}
    assert PR.request_body(msgs, 2500, thinking=True, effort="medium")["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_proto_parse_split_and_channel(tmp_path):
    import open_teacher as OT
    from vlm import proto as PR

    classes = ["NONE", "FRA-01", "CAL-01", "ALG-06"]
    texts = ['x {"root_code": "FRA-01", "top3": ["FRA-01", "XXX", "CAL-01", "NONE"], "probabilities": [0.6, 0.2, 0.1], "confidence": 0.6, "final_answer_correct": false, "readable": "Yes"}<|im_end|>',
             '{"root_code": "NONE", "top3": ["NONE"], "probabilities": "many", "confidence": true}',
             '{"root_code": "XXX", "top3": ["FRA-01"]}', '{"root_code": "FRA-01", "top3": []}', "no object", '{"root_code": "FRA-01", "top3": ["FRA-01"',
             '{"root_code": "FRA-01", "top3": "FRA-01"}', "[1, 2] {3} "]
    for t in texts:
        mine, ref = PR.parse_answer(t, classes), None
        try:
            ref = OT.parse_answer(t, classes)
        except Exception:  # noqa: BLE001 - the measurement tool may fail on nonsense, the in-house path may not
            pass
        if ref is not None:
            assert mine is not None and all(mine[k] == ref[k] for k in ("root_code", "top3", "probabilities", "final_answer_correct", "readable"))
    a = PR.parse_answer(texts[0], classes)
    assert a["top3"] == ["FRA-01", "CAL-01", "NONE"] and a["probabilities"] is None and a["final_answer_correct"] is False and a["readable"] == "yes"
    assert PR.parse_answer(texts[1], classes) == {"root_code": "NONE", "top3": ["NONE"], "probabilities": None, "confidence": None, "final_answer_correct": None, "readable": ""}
    assert all(PR.parse_answer(t, classes) is None for t in texts[2:])
    assert PR.split_reply("<think>\nbecause\n</think>\n\n{...}", True) == ("because", "\n\n{...}")
    assert PR.split_reply("still reasoning", True) == ("still reasoning", "") and PR.split_reply("{...}", False) == ("", "{...}")
    f = tmp_path / "proto_labels.jsonl"
    recs = [{"row": 0, "phase": "first", "ok": True, "root_code": "FRA-01", "top3": ["FRA-01", "CAL-01"], "probabilities": [0.5, 0.3]},
            {"row": 1, "phase": "first", "ok": False},
            {"row": 1, "phase": "finish", "ok": True, "root_code": "NONE", "top3": ["NONE", "CAL-01", "FRA-01"], "probabilities": None},
            {"row": 2, "phase": "first", "ok": True, "root_code": "CAL-01", "top3": ["CAL-01", "FRA-01", "NONE"], "probabilities": [0.9, 0.6, 0.5]}]
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    seen = np.array([True, True, True, False])
    probs, has = PR.labels_to_probs(f, 4, classes, seen)
    assert has.tolist() == [True, True, True, False] and np.allclose(probs[:3].sum(1), 1.0) and np.all(probs[:, 3] == 0.0)
    rest = 0.2 / 3
    assert np.allclose(probs[0], [rest, 0.5 + rest, 0.3 + rest, 0.0])  # the rest spread evenly over the classes that can be a root
    assert np.allclose(probs[1], [0.6 + 0.05 / 3, 0.1 + 0.05 / 3, 0.25 + 0.05 / 3, 0.0])  # without probabilities: fixed levels
    assert np.allclose(probs[2], [0.25, 0.3, 0.45, 0.0])  # a sum above 1 is normalised
    # non-finite or boolean "probabilities" do not count: the others apply; if all are unusable, the fixed levels apply
    f.write_text('{"row": 0, "ok": true, "root_code": "FRA-01", "top3": ["FRA-01", "CAL-01"], "probabilities": [NaN, 0.4]}\n'
                 '{"row": 1, "ok": true, "root_code": "NONE", "top3": ["NONE", "CAL-01"], "probabilities": [Infinity, true]}\n')
    probs, has = PR.labels_to_probs(f, 4, classes, seen)
    assert has.tolist() == [True, True, False, False] and np.isfinite(probs).all() and np.allclose(probs[:2].sum(1), 1.0)
    assert np.allclose(probs[0], [0.2, 0.2, 0.4 + 0.2, 0.0]) and np.allclose(probs[1], [0.6 + 0.15 / 3, 0.15 / 3, 0.25 + 0.15 / 3, 0.0])


def test_proto_generate_two_steps_resume_and_guard(tmp_path, monkeypatch):
    """First pass, finish from the own notes, resume and the guard against form errors, with a mocked server (no model)."""
    from vlm import proto as PR
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    df = _proto_root(tmp_path)
    classes = ["NONE", "FRA-01", "CAL-01"]
    calls = {"start": 0, "first": 0, "finish": 0}
    answer = json.dumps({"root_code": "FRA-01", "top3": ["FRA-01", "CAL-01", "NONE"], "probabilities": [0.7, 0.2, 0.05]})

    class Proc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_start(model_dir, spec, log_path):
        calls["start"] += 1
        return Proc()

    def fake_post(url, body, timeout):
        texts = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        calls["first" if thinking else "finish"] += 1
        if thinking and "Question A" in texts[0]:  # finished in the first pass
            assert body["max_tokens"] == 2500
            return {"choices": [{"message": {"content": "brief thought\n</think>\n\n" + answer}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        if thinking:  # cut off at the token limit
            return {"choices": [{"message": {"content": "long notes TAIL"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2500}}
        assert body["max_tokens"] == 1500 and texts[-1] == PR.FINISH_TEXT.format(notes="TAIL")  # only the end of the own notes
        return {"choices": [{"message": {"content": answer.replace("FRA-01", "XX", 1).replace("XX", "CAL-01", 1).replace('"top3": ["FRA-01", "CAL-01"', '"top3": ["CAL-01", "FRA-01"')}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30}}

    monkeypatch.setattr(TV, "start_server", fake_start)
    monkeypatch.setattr(TV, "wait_ready", lambda port, proc, timeout_s: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    job = {"note_chars": 4, "concurrency": 2}
    out = tmp_path / "out"
    stats = PR.generate(job, tmp_path, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), out, "proto", log=lambda *a: None)
    assert calls == {"start": 1, "first": 2, "finish": 1}
    assert stats["valid"] == 2 and stats["valid_first"] == 1 and stats["finished"] == 1 and stats["ends_first"] == {"stop": 1, "length": 1}
    labs = [json.loads(line) for line in (out / "proto_labels.jsonl").read_text().splitlines()]
    assert [(r["row"], r["phase"], r["ok"]) for r in labs] == [(0, "first", True), (1, "first", False), (1, "finish", True)]
    assert labs[0]["root_code"] == "FRA-01" and labs[2]["root_code"] == "CAL-01" and labs[2]["top3"][0] == "CAL-01"
    raw = [json.loads(line) for line in (out / "proto_raw.jsonl").read_text().splitlines()]
    assert raw[0]["thinking"] == "brief thought" and raw[1]["thinking"] == "long notes TAIL" and raw[1]["text"] == ""
    probs, has = PR.labels_to_probs(out / "proto_labels.jsonl", 2, classes, np.array([True, True, True]))
    assert has.all() and probs[0].argmax() == 1 and probs[1].argmax() == 2
    # resume: everything exists, no server, no request
    stats2 = PR.generate(job, tmp_path, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), out, "proto", log=lambda *a: None)
    assert calls == {"start": 1, "first": 2, "finish": 1} and stats2["valid"] == 2 and stats2["done_before"] == 2
    # `first_rows`: these rows first; the order of the output follows the processing
    monkeypatch.setattr(TV, "_post", fake_post)
    PR.generate(job, tmp_path, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), tmp_path / "out3", "proto", log=lambda *a: None,
                first_rows=np.array([1, 7]))
    assert [json.loads(line)["row"] for line in (tmp_path / "out3" / "proto_labels.jsonl").read_text().splitlines()] == [1, 1, 0]
    # guard: if the first rows return no valid answer, the job aborts
    monkeypatch.setattr(TV, "_post", lambda url, body, timeout: {"choices": [{"message": {"content": "nonsense"}, "finish_reason": "stop"}], "usage": {}})
    with pytest.raises(RuntimeError):
        PR.generate({"guard_rows": 2, "concurrency": 1}, tmp_path, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), tmp_path / "out2", "proto", log=lambda *a: None)


class _ProtoProc:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.mark.parametrize("finishes", [True, False])
def test_proto_continuation_preserves_notes_pairs_control_and_stops(tmp_path, monkeypatch, finishes):
    from vlm import proto as PR
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    df = _proto_root(tmp_path)
    classes = ["NONE", "FRA-01", "CAL-01"]
    answer = json.dumps({"root_code": "FRA-01", "top3": classes[1:] + classes[:1]})
    # In the paired control the old tail limit is retained; the treatment must keep the start too.
    notes = "DECISIVE FIRST STEP " + "working " * 2400 + "END"
    calls = []

    def fake_post(url, body, timeout):
        texts = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        if len(texts) == 3:  # original question, key, diagram caption (row B)
            phase, content, end = "first", notes, "length"
            assert body["max_tokens"] == 2500
        elif thinking:
            phase = "continue"
            assert notes in texts[-1] and body["max_tokens"] == 2500
            if calls.count("continue"):
                assert "new reasoning 1" in texts[-1]
            content = "new reasoning 1\n</think>\n" + answer if finishes else "new reasoning 1"
            end = "stop" if finishes else "length"
        elif texts[-1] == PR.FINISH_TEXT.format(notes="END"):
            phase, content, end = "control", answer, "stop"
        else:
            phase, content, end = "finish", answer, "stop"
            assert notes in texts[-1] and "new reasoning 1" in texts[-1]
        calls.append(phase)
        return {"choices": [{"message": {"content": content}, "finish_reason": end}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2500 if thinking else 100}}

    monkeypatch.setattr(TV, "start_server", lambda *a: _ProtoProc())
    monkeypatch.setattr(TV, "wait_ready", lambda *a: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    job = {"concurrency": 1, "note_chars": 3, "continuation_tokens": 2500,
           "max_continuations": 2, "paired_finish_control": True}
    out = tmp_path / "out"
    args = (job, tmp_path, df, np.array([1]), "train", tmp_path, TAX, classes, ImageConfig(), out, "proto")
    stats = PR.generate(*args, log=lambda *a: None)
    assert calls == (["first", "control", "continue"] if finishes else ["first", "control", "continue", "continue", "finish"])
    assert stats["answered"] == 1 and stats["valid_first"] == 0 and stats["valid"] == 1
    assert stats["first_length_without_answer"] == 1 and stats["continued_rows"] == 1
    assert stats["exhausted_reasoning_rows"] == int(not finishes)
    assert stats["continuation_requests"] == (1 if finishes else 2)
    assert stats["logged_generated_tokens"] == (5000 if finishes else 7600)
    assert stats["control"]["additional_generated_tokens"] == 100
    before = len(calls)
    assert PR.generate(*args, log=lambda *a: None)["done_before"] == 1 and len(calls) == before
    controls = [json.loads(s) for s in (out / "proto_control_labels.jsonl").read_text().splitlines()]
    labs = [json.loads(s) for s in (out / "proto_labels.jsonl").read_text().splitlines()]
    assert len(controls) == 1 and controls[0]["phase"] == "control"
    assert all(r["phase"] != "control" for r in labs)
    with pytest.raises(ValueError, match="policy changed"):
        PR.generate(dict(job, continuation_tokens=3000), *args[1:], log=lambda *a: None)


@pytest.mark.parametrize("first_kind", ["valid", "partial_json", "bad_stop"])
def test_proto_continuation_does_not_rethink_finished_diagnoses(tmp_path, monkeypatch, first_kind):
    from vlm import proto as PR
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    df = _proto_root(tmp_path)
    classes = ["NONE", "FRA-01", "CAL-01"]
    answer = json.dumps({"root_code": "NONE", "top3": classes})
    replies = {"valid": ("done\n</think>\n" + answer, "stop"),
               "partial_json": ('done\n</think>\n{"root_code":', "length"),
               "bad_stop": ("done\n</think>\ninvalid", "stop")}
    calls = []

    def fake_post(url, body, timeout):
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        calls.append(thinking)
        content, end = replies[first_kind] if thinking else (answer, "stop")
        return {"choices": [{"message": {"content": content}, "finish_reason": end}], "usage": {}}

    monkeypatch.setattr(TV, "start_server", lambda *a: _ProtoProc())
    monkeypatch.setattr(TV, "wait_ready", lambda *a: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    stats = PR.generate({"continuation_tokens": 2500, "paired_finish_control": True}, tmp_path, df, np.array([0]), "train",
                        tmp_path, TAX, classes, ImageConfig(), tmp_path / "out", "proto", log=lambda *a: None)
    assert calls == ([True] if first_kind == "valid" else [True, False])
    assert stats["continued_rows"] == 0 and stats["control"]["additional_generated_tokens"] == 0
    assert stats["valid"] == 1 and stats["control"]["valid"] == 1


def test_proto_continuation_preflight_guards(tmp_path):
    from vlm import proto as PR
    from vlm.data import ImageConfig

    for job, split in (({"continuation_tokens": -1}, "train"),
                       ({"continuation_tokens": 2000, "max_continuations": 3}, "train"),
                       ({"paired_finish_control": True}, "train"),
                       ({"continuation_tokens": 2000, "paired_finish_control": True}, "test"),
                       ({"continuation_tokens": 2000, "template_kwargs": {"enable_thinking": False}}, "train")):
        with pytest.raises(ValueError):
            PR.continuation_policy(job, split)
    (tmp_path / "proto_labels.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="legacy output"):
        PR.generate({"continuation_tokens": 2500}, tmp_path, None, np.array([0]), "train", tmp_path, TAX,
                    ["NONE", "FRA-01"], ImageConfig(), tmp_path, "proto")


@pytest.mark.parametrize("broken_phase", ["continue", "control"])
def test_proto_continuation_network_failure_is_not_a_finished_pair(tmp_path, monkeypatch, broken_phase):
    from vlm import proto as PR
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    df = _proto_root(tmp_path)
    answer = json.dumps({"root_code": "NONE", "top3": ["NONE", "FRA-01", "CAL-01"]})
    calls = []

    def fake_post(url, body, timeout):
        texts = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
        phase = "first" if len(texts) == 2 else "continue" if body["chat_template_kwargs"]["enable_thinking"] else "control"
        calls.append(phase)
        if phase == broken_phase:
            raise ConnectionError("synthetic failure")
        content = "unfinished" if phase == "first" else answer
        return {"choices": [{"message": {"content": content}, "finish_reason": "length" if phase == "first" else "stop"}]}

    monkeypatch.setattr(TV, "start_server", lambda *a: _ProtoProc())
    monkeypatch.setattr(TV, "wait_ready", lambda *a: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    monkeypatch.setattr(PR.time, "sleep", lambda *a: None)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="incomplete"):
        PR.generate({"continuation_tokens": 2500, "paired_finish_control": True, "max_failed": 0}, tmp_path,
                    df, np.array([0]), "train", tmp_path, TAX, ["NONE", "FRA-01", "CAL-01"],
                    ImageConfig(), out, "proto", log=lambda *a: None)
    assert calls.count(broken_phase) == 3  # bounded retries, no endless request loop
    meta = json.loads((out / "proto_meta.json").read_text())
    assert meta["failed"] == 1 and meta["answered"] == 0 and meta["control"]["answered"] == 0
    assert (out / "proto_labels.jsonl").read_text() == ""
    assert (out / "proto_control_labels.jsonl").read_text() == ""


def test_proto_parse_never_raises_and_finish_notes_follow_the_measured_form():
    """Review findings: the answer parser must not fail on any answer (a non-hashable root_code or an oversized number
    would otherwise end the job), and the notes of the finish have the form of the API measurement."""
    from vlm import proto as PR

    classes = ["NONE", "FRA-01", "CAL-01", "ALG-06"]
    assert PR.parse_answer('{"root_code": ["CAL-01"], "top3": ["CAL-01"]}', classes) is None
    assert PR.parse_answer('{"root_code": {"a": 1}, "top3": ["CAL-01"]}', classes) is None
    assert PR.parse_answer('{"root_code": "CAL-01", "top3": ["CAL-01"], "probabilities": [1' + "0" * 400 + "]}", classes)["probabilities"] is None
    assert PR.parse_answer('{"root_code": "CAL-01", "top3": ["CAL-01"], "confidence": 1' + "0" * 400 + "}", classes)["confidence"] is None
    # cut off at the token limit, thinking block open: raw text, trailing whitespace stays
    raw = "line 1 …\nso it is \n"
    assert PR.finish_notes(raw, *PR.split_reply(raw, True), "length") == raw
    # cut off AFTER a closed thinking block: raw text including the started answer, with the block start of the measurement
    raw = 'thought\n</think>\n\n{"readable": "yes", "transc'
    assert PR.finish_notes(raw, *PR.split_reply(raw, True), "length") == "<think>\n" + raw
    # ended regularly but unusable: the reasoning, whitespace removed
    raw = "\nthought \n</think>\n\nno JSON"
    assert PR.finish_notes(raw, *PR.split_reply(raw, True), "stop") == "thought"


def test_proto_generate_row_errors_order_and_failure_limits(tmp_path, monkeypatch):
    """Review findings: shuffled processing order (priority rows stay in front), request and image errors are row
    errors, the guard counts them, and too many open rows make the job fail."""
    from PIL import Image

    from vlm import proto as PR
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    d = tmp_path / "competition_data" / "images" / "train"
    d.mkdir(parents=True, exist_ok=True)
    names = [f"r{k}.jpeg" for k in range(12)]
    for nm in names:
        Image.fromarray(np.full((30, 40, 3), 200, np.uint8)).save(d / nm)
    df = pd.DataFrame([dict(ROW, trace_id=f"t{k}", image_file=nm, question_text=f"Question {k:02d}") for k, nm in enumerate(names)])
    classes = ["NONE", "FRA-01", "CAL-01"]
    good = json.dumps({"root_code": "FRA-01", "top3": ["FRA-01", "CAL-01", "NONE"], "probabilities": [0.7, 0.2, 0.05]})
    notes_seen = {}

    def reply(text, end="stop"):
        return {"choices": [{"message": {"content": text}, "finish_reason": end}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def fake_post(url, body, timeout):
        texts = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
        k = int(texts[0].split("Question ")[1][:2])
        if not body["chat_template_kwargs"]["enable_thinking"]:
            notes_seen[k] = texts[-1]
            return reply(good)
        if k == 3:  # finished answer, but root_code as a list: formerly an exception in the job
            return reply('brief\n</think>\n\n{"root_code": ["FRA-01"], "top3": ["FRA-01"]}')
        if k == 4:  # cut off in the middle of the reasoning, trailing whitespace
            return reply("long notes \n", "length")
        return reply("thought\n</think>\n\n" + good)

    monkeypatch.setattr(TV, "start_server", lambda model_dir, spec, log_path: _ProtoProc())
    monkeypatch.setattr(TV, "wait_ready", lambda port, proc, timeout_s: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    monkeypatch.setattr(PR.time, "sleep", lambda s: None)
    rows = np.arange(12)

    def run(job, out, **kw):
        return PR.generate(dict({"concurrency": 3}, **job), tmp_path, df, rows, "train", tmp_path, TAX, classes, ImageConfig(), tmp_path / out, "proto",
                           log=lambda *a: None, **kw)

    def order(out):
        return list(dict.fromkeys(json.loads(line)["row"] for line in (tmp_path / out / "proto_labels.jsonl").read_text().splitlines()))

    stats = run({"order_seed": 0}, "o1", first_rows=np.array([7, 2]))
    got = order("o1")
    assert stats["valid"] == 12 and sorted(got) == list(range(12)) and set(got[:2]) == {2, 7} and got[2:] != sorted(got[2:])  # shuffled, priority in front
    assert notes_seen[3] == PR.FINISH_TEXT.format(notes="brief") and notes_seen[4] == PR.FINISH_TEXT.format(notes="long notes \n")
    run({}, "o2")
    assert order("o2") == list(range(12))  # without order_seed: table order
    # missing image file: the row stays open, the job continues (below the limit max_failed)
    (d / "r5.jpeg").unlink()
    stats = run({"max_failed": 0.2}, "o3")
    assert stats["failed"] == 1 and stats["valid"] == 11 and 5 not in order("o3")
    # too many open rows: the job fails AFTER writing the statistics
    with pytest.raises(RuntimeError, match="incomplete"):
        run({"max_failed": 0.05}, "o4")
    assert json.loads((tmp_path / "o4" / "proto_meta.json").read_text())["failed"] == 1
    # only request errors: the guard acts after guard_rows rows (formerly the job finished "successfully")
    def broken(url, body, timeout):
        raise ConnectionError("server gone")

    monkeypatch.setattr(TV, "_post", broken)
    with pytest.raises(RuntimeError, match="request errors"):
        run({"guard_rows": 4}, "o5")


def test_proto_generate_translates_the_marks_of_another_model(tmp_path, monkeypatch):
    """Model with a different architecture (second model in the protocol): the markers of its thinking block are translated
    into the own form (reply_replace), extra arguments go to its chat template (template_kwargs); if it writes a
    thinking block in the finish despite thinking being off, the text behind it counts as the answer."""
    from vlm import proto as PR
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    df = _proto_root(tmp_path)
    classes = ["NONE", "FRA-01", "CAL-01"]
    good = json.dumps({"root_code": "CAL-01", "top3": ["CAL-01", "FRA-01", "NONE"], "probabilities": [0.6, 0.3, 0.1]})
    seen = {"kwargs": [], "notes": []}

    def fake_post(url, body, timeout):
        texts = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
        seen["kwargs"].append(body["chat_template_kwargs"])
        if body["chat_template_kwargs"]["enable_thinking"]:
            if "Question A" in texts[0]:  # finished: thinking block in a foreign form, then the answer
                return {"choices": [{"message": {"content": "<|channel>thought\nbrief thought<channel|>" + good}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 9}}
            return {"choices": [{"message": {"content": "<|channel>thought\nlong notes"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 9, "completion_tokens": 2500}}
        seen["notes"].append(texts[-1])
        return {"choices": [{"message": {"content": "<|channel>thought\none more thought<channel|>" + good}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 30}}

    monkeypatch.setattr(TV, "start_server", lambda model_dir, spec, log_path: _ProtoProc())
    monkeypatch.setattr(TV, "wait_ready", lambda port, proc, timeout_s: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    job = {"concurrency": 1, "template_kwargs": {"image_budget": 3},
           "reply_replace": [["<|channel>thought\n", ""], ["<|channel>", ""], ["<channel|>", "\n</think>\n\n"]]}
    stats = PR.generate(job, tmp_path, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), tmp_path / "out", "proto", log=lambda *a: None)
    assert stats["valid"] == 2 and stats["valid_first"] == 1 and stats["finished"] == 1
    assert all(k["image_budget"] == 3 for k in seen["kwargs"]) and [k["enable_thinking"] for k in seen["kwargs"]] == [True, True, False]
    assert seen["notes"] == [PR.FINISH_TEXT.format(notes="long notes")]  # markers of the foreign thinking block are not part of the notes
    raw = [json.loads(line) for line in (tmp_path / "out" / "proto_raw.jsonl").read_text().splitlines()]
    assert raw[0]["thinking"] == "brief thought" and raw[2]["thinking"] == "one more thought" and raw[2]["text"].endswith(good)


def test_proto_rank_prompt_and_notes(tmp_path):
    """Ranking behind the notes of the protocol channel: verdict rules without the JSON format, user message as in the
    protocol (no labelled row), the notes as a closed thinking block before "Root:". Only the reasoning of the first
    pass counts as notes; the answer text already names its codes."""
    from vlm import common as C
    from vlm import proto as PR
    from vlm.data import ImageConfig

    assert PR.JSON_PART in PR.INSTRUCTIONS and PR.RANK_INSTRUCTIONS.startswith(PR.INSTRUCTIONS.split(PR.JSON_PART)[0])
    assert "JSON" not in PR.RANK_INSTRUCTIONS and "prefer a topic-specific code" in PR.RANK_INSTRUCTIONS
    assert PR.RANK_INSTRUCTIONS.endswith("`Root: <taxonomy code or NONE>`.")
    system = PR.rank_system_text(TAX)
    assert system == PR.RANK_INSTRUCTIONS + "\n\nMisconception taxonomy (code | name | description):\n" + P.taxonomy_text(TAX, "full")

    assert PR.rank_notes({"phase": "first", "thinking": " <think>read first,\nthen compute </think> ", "text": '{"root_code": "CAL-01"}'}) == "read first,\nthen compute"
    assert PR.rank_notes({"phase": "finish", "thinking": "thought of the finish", "text": "{}"}) == ""
    assert PR.rank_notes({"phase": "first", "thinking": "", "text": "answer only"}) == ""
    assert PR.rank_notes({"phase": "first", "thinking": "abcdefgh"}, note_chars=3) == "fgh"  # the end counts, as in the finish

    df = _proto_root(tmp_path)
    row = df.iloc[1]
    student, qimg = PR.open_images(row, "train", tmp_path, ImageConfig())
    assert student.size == C.open_image(C.image_path(tmp_path, "train", "b.jpeg"), 1024).size and qimg is not None
    turned, _ = PR.open_images(row, "train", tmp_path, ImageConfig(rotate={"b": 1}))
    assert turned.size == student.size[::-1]  # quarter turn from the rotation list, as during generation
    s = PR.rank_sample(row, student, qimg, system, "line 2 is wrong because \n")
    assert [k for k, _ in s.parts] == [k for k, _ in PR.user_parts(row, student, qimg)] and s.parts[0][1] == PR.user_parts(row, student, qimg)[0][1]
    assert s.prefix == "line 2 is wrong because\n</think>\n\nRoot:"
    assert s.meta["think"] is True and s.meta["system"] == system and s.meta["static"] == system
    text = " ".join(v for k, v in s.parts if k == "text")
    assert "FRA-01" not in text and "Diagnosed working" not in text  # no label, no example of the same question
    with pytest.raises(ValueError, match="empty notes"):
        PR.rank_sample(row, student, qimg, system, " \n")

    f = tmp_path / "n.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in ({"table": "train", "row": 4, "text": "a"}, {"table": "train", "row": 7, "text": " "}, {"table": "train", "row": 9, "text": "b"})))
    assert PR.load_rank_notes(f) == ("train", {4: "a", 9: "b"})  # empty notes count as missing
    f.write_text("".join(json.dumps(r) + "\n" for r in ({"table": "train", "row": 4, "text": "a"}, {"table": "test", "row": 5, "text": "b"})))
    with pytest.raises(ValueError, match="exactly one table"):
        PR.load_rank_notes(f)
    f.write_text("".join(json.dumps(r) + "\n" for r in ({"table": "test", "row": 4, "text": "a"}, {"table": "test", "row": 4, "text": "b"})))
    with pytest.raises(ValueError, match="appears twice"):
        PR.load_rank_notes(f)


def test_proto_rank_rows_table_and_probe_stand_alone(tmp_path):
    """The probe proto_rank computes only rows of the table the notes come from, stands alone in its job and builds its
    sample without the builder's reference set (no question knowledge)."""
    from types import SimpleNamespace

    from vlm import proto as PR
    from vlm.data import ImageConfig
    from vlm.runner import PROTO_NO_TASK_KEYS, ProbeRunner, proto_rows

    r = SimpleNamespace(proto_notes={1: "a", 3: "b"}, proto_table="train")
    idx = np.array([0, 1, 2, 3])
    assert proto_rows(r, idx, "train", ["proto_rank"]).tolist() == [1, 3]
    assert proto_rows(r, idx, "test", ["diag"]) is idx  # other probes: row selection unchanged
    with pytest.raises(ValueError, match="table train"):
        proto_rows(r, idx, "test", ["proto_rank"])
    with pytest.raises(ValueError, match="only probe"):
        proto_rows(r, idx, "train", ["proto_rank", "diag"])
    with pytest.raises(ValueError, match="proto_from"):
        proto_rows(SimpleNamespace(proto_notes=None, proto_table=None), idx, "train", ["proto_rank"])
    assert {"teacher_labels", "topic_codes", "question_codes", "analysis_from", "gold_reading"} <= set(PROTO_NO_TASK_KEYS)

    df = _proto_root(tmp_path)
    system = PR.rank_system_text(TAX)
    fake = SimpleNamespace(proto_notes={1: "notes on the second row"}, proto_table="train", proto_system=system,
                           builder=SimpleNamespace(root=tmp_path, image=ImageConfig()))
    s = ProbeRunner._build_proto(fake, df, 1, "train")
    assert s.prefix == "notes on the second row\n</think>\n\nRoot:" and s.meta["system"] == system
    assert [k for k, _ in s.parts] == ["text", "text", "image", "text", "image"]
    with pytest.raises(KeyError, match="has no notes"):
        ProbeRunner._build_proto(fake, df, 0, "train")
    with pytest.raises(ValueError, match="computation runs on test"):
        ProbeRunner._build_proto(fake, df, 1, "test")


def test_engine_chat_text_takes_the_system_text_of_the_sample():
    """A sample can carry its own system text (teacher protocol: verdict rules and taxonomy); otherwise the shared system
    text of all other tasks applies."""
    from types import SimpleNamespace

    from vlm.engine import Engine

    class Proc:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking, **kw):
            self.seen = {"think": enable_thinking, "kw": kw}
            return "<s>" + messages[0]["content"][0]["text"] + "</s><u>" + "".join(c.get("text", "[img]") for c in messages[1]["content"]) + "</u><a>"

    stub = SimpleNamespace(proc=Proc())
    plain = P.Sample(key="a", parts=[("text", "Question")], prefix="Root:", meta={"think": False})
    assert Engine.chat_text(stub, plain) == "<s>" + P.SYSTEM + "</s><u>Question</u><a>Root:" and stub.proc.seen == {"think": False, "kw": {}}
    own = P.Sample(key="b", parts=[("text", "Question"), ("image", None)], prefix="notes\n</think>\n\nRoot:", meta={"think": True, "system": "RULES", "static": "RULES"})
    assert Engine.chat_text(stub, own) == "<s>RULES</s><u>Question[img]</u><a>notes\n</think>\n\nRoot:" and stub.proc.seen["think"] is True


def test_smoke_job_fails_on_single_failed_rows():
    """A pre-run (must_pass) also counts as failed if only single rows fail or the probe ends with a partial result;
    otherwise the long job would start behind a pre-run in which every row failed."""
    from vlm.runner import failed_probes

    summary = {"job": "x", "timing": {"proto_rank": {"rows": 0, "failed": 8}, "diag": {"rows": 8, "failed": 0}},
               "timing_test": {"think": {"rows": 5, "failed": 0, "partial": True}, "fac": {"error": "RuntimeError: …"}}, "seconds": 3.0}
    assert failed_probes(summary, strict=False) == ["fac"]
    assert sorted(failed_probes(summary, strict=True)) == ["fac", "proto_rank", "think"]
    assert failed_probes({"timing": {"diag": {"rows": 8, "failed": 0}}}, strict=True) == []



def test_start_server_bin_on_path_only_with_switch(tmp_path, monkeypatch):
    """bin_on_path puts the bin folder of the server's Python in front of PATH; without the switch PATH stays unchanged."""
    import os
    import subprocess
    from vlm import think_vllm as TV

    seen = []

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, env=None):
            seen.append(env)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    py = str(tmp_path / "venv" / "bin" / "python")
    TV.start_server(tmp_path, {"python": py}, tmp_path / "a.log")
    TV.start_server(tmp_path, {"python": py, "bin_on_path": True, "env": {"VLLM_ALLREDUCE_USE_FLASHINFER": "0"}}, tmp_path / "b.log")
    assert seen[0]["PATH"] == "/usr/bin:/bin"
    assert seen[1]["PATH"] == str(tmp_path / "venv" / "bin") + os.pathsep + "/usr/bin:/bin"
    assert seen[1]["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0" and "VLLM_ALLREDUCE_USE_FLASHINFER" not in seen[0]
