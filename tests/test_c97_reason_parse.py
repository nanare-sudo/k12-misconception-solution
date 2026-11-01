"""Tests for the reasoning mode of tools/teacher_api.py: split into reasoning / finding line / JSON, style examples only from
fold 0, head without notes — and the earlier label mode stays byte-identical."""
import hashlib
import json

import numpy as np
import pytest

import teacher_api as T
from vlm import common as C

CLASSES = ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "FRA-01", "ALG-02", "Dec-01"]


def _msg(text: str) -> dict:
    return {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}]}


def _json(**kw) -> str:
    base = {"readable": "yes", "final_answer_correct": False, "student_final_answer": "4/6", "root_code": "FRA-01",
            "top3": ["FRA-01", "CAL-01", "ANS-01"], "probabilities": [0.7, 0.15, 0.05], "confidence": 0.7}
    base.update(kw)
    return json.dumps(base)


REASONING = ("In L1 the student writes the sum of the two fractions. In L2 the pupil sums tops and bottoms (zz). "
             "Candidates: FRA-01 (tops and bottoms summed) versus CAL-01 (slip) or ANS-01. "
             "The error is systematic, so FRA-01 fits better than CAL-01.")


def test_parse_reason_splits_three_parts():
    text = REASONING + "\n\nFinding: The pupil turned 3/4 + 1/2 into 4/6 (tops plus tops, bottoms plus bottoms); zz expects 5/4.\n\n" + _json()
    out = T.parse_reason(_msg(text), CLASSES)
    assert out["root_code"] == "FRA-01" and out["top3"] == ["FRA-01", "CAL-01", "ANS-01"]
    assert out["reasoning"].startswith("In L1") and "Finding" not in out["reasoning"]
    assert out["finding"].startswith("The pupil turned") and out["finding"].endswith("5/4.")
    assert out["codes_in_reasoning"] == ["ANS-01", "CAL-01", "FRA-01"] and out["consistent"] is True
    assert out["finding_words"] == len(out["finding"].split()) and out["reasoning_words"] == len(out["reasoning"].split())


def test_parse_reason_survives_braces_fences_and_markdown_labels():
    text = ("1. Reasoning:\n" + REASONING + " The key gives \\frac{5}{4} and the set {1, 2}.\n\n2.\n**Finding:** The pupil summed "
            "tops and bottoms (zz).\n\n```json\n" + _json(student_final_answer="\\frac{4}{6}") + "\n```\n")
    out = T.parse_reason(_msg(text), CLASSES)
    assert out is not None and out["student_final_answer"] == "\\frac{4}{6}"
    assert out["finding"] == "The pupil summed tops and bottoms (zz)."
    assert not out["reasoning"].startswith("1.") and "\\frac{5}{4}" in out["reasoning"]


def test_parse_reason_none_and_missing_finding():
    ok = T.parse_reason(_msg("All lines are correct.\nFinding: none\n" + _json(root_code="NONE", top3=["NONE", "CAL-01", "ANS-01"])), CLASSES)
    assert ok["finding_none"] and ok["consistent"]
    miss = T.parse_reason(_msg(REASONING + "\n" + _json()), CLASSES)
    assert miss["finding"] is None and miss["consistent"] is False and miss["reasoning"].startswith("In L1")
    contra = T.parse_reason(_msg("x\nFinding: none\n" + _json()), CLASSES)
    assert contra["consistent"] is False


def test_parse_reason_rejects_bad_json_or_codes():
    assert T.parse_reason(_msg(REASONING + "\nFinding: x\n{not json}"), CLASSES) is None
    assert T.parse_reason(_msg("Finding: x\n" + _json(root_code="DEC-01", top3=["DEC-01"])), CLASSES) is None
    assert T.parse_reason(_msg("Finding: x\n" + _json(top3=["XYZ-01"])), CLASSES) is None


def test_finding_continuation_lines_are_joined():
    text = REASONING + "\nFinding: The pupil summed tops\nand bottoms one by one (zz).\n\n" + _json()
    assert T.parse_reason(_msg(text), CLASSES)["finding"] == "The pupil summed tops and bottoms one by one (zz)."


def test_code_mentions_exact_case():
    assert T.code_mentions("Dec-01 fits, DEC-01 is not a code, xCAL-01y neither, but (CAL-01) does.", CLASSES) == ["CAL-01", "Dec-01"]


def _data_root():
    try:
        return C.find_data_root()
    except FileNotFoundError:
        pytest.skip("competition data not available")


def _style_file(tmp_path, positions):
    examples = [f"The student wrote [n1] instead of [n2] ({i})." for i in range(len(positions))]
    rec = {"examples": examples, "source_positions": positions,
           "sha256": hashlib.sha256(json.dumps(examples, ensure_ascii=False).encode()).hexdigest()}
    p = tmp_path / "style.json"
    p.write_text(json.dumps(rec))
    return p


def test_style_only_from_fold_0(tmp_path):
    folds = C.load_folds()
    f0 = [int(i) for i in np.where(folds == 0)[0][:3]]
    f1 = int(np.where(folds == 1)[0][0])
    assert T.load_style(_style_file(tmp_path, f0))["source_positions"] == f0
    with pytest.raises(SystemExit):
        T.load_style(_style_file(tmp_path, f0[:2] + [f1]))
    bad = _style_file(tmp_path, f0)
    rec = json.loads(bad.read_text())
    rec["examples"][0] += " changed"
    bad.write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        T.load_style(bad)


def test_reason_head_has_rules_format_style_taxonomy_but_no_notes(tmp_path):
    root = _data_root()
    folds = C.load_folds()
    style = _style_file(tmp_path, [int(i) for i in np.where(folds == 0)[0][:2]])
    text = T.reason_system_blocks(root, style)[0]["text"]
    assert text.startswith(T.INSTRUCTIONS.split(T.JSON_PART, 1)[0])
    assert T.JSON_PART not in text and "Finding: none" in text and "Misconception taxonomy" in text
    assert "notes that markers wrote" not in text and "\n    e.g. " not in text  # no annotator notes in the head
    assert "The student wrote [n1] instead of [n2] (0)." in text


def test_label_mode_head_unchanged():
    """The earlier head (label mode) must stay byte-identical — checksum from the manifests of its plans."""
    root = _data_root()
    head = T.system_blocks(root, None)[0]["text"]
    assert hashlib.sha256(head.encode()).hexdigest() == "5f3ecb6698f71f94045c9e6a221ac128b6333786ac8760f648df690236451bb1"


def test_plan_from_rows_writes_pool_plan(tmp_path, monkeypatch):
    root = _data_root()
    monkeypatch.setattr(T, "OUT", tmp_path / "teacher")
    rows = np.array([5, 1, 9], dtype=np.int64)
    src = tmp_path / "rows.npy"
    np.save(src, rows)

    class A:
        name, table, rows_from, exclude, limit = "p", "unlabelled", str(src), [], 0

    assert T.plan_from_rows(A, root) == 0
    plan = json.loads((tmp_path / "teacher" / "p" / "plan.json").read_text())
    assert plan["rows"] == [1, 5, 9] and plan["table"] == "unlabelled" and plan["rows_from_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    A.table = "train"
    with pytest.raises(SystemExit):
        T.plan_from_rows(A, root)


def test_numbered_parts_as_answered_in_the_probe():
    """In a probe, 23 of 60 answers wrote "2. Finding: …" (the instruction numbers its parts)."""
    text = "1. " + REASONING + "\n\n2. Finding: The pupil summed tops and bottoms (zz).\n\n3.\n" + _json()
    out = T.parse_reason(_msg(text), CLASSES)
    assert out["finding"] == "The pupil summed tops and bottoms (zz)." and out["consistent"]
    none = T.parse_reason(_msg(REASONING + "\n\n2. Finding: none\n\n" + _json(root_code="NONE", top3=["NONE", "CAL-01", "ANS-01"])), CLASSES)
    assert none["finding_none"] and none["consistent"]
    label = T.parse_reason(_msg(REASONING + "\n\n2. Finding line:\nFinding: The pupil summed tops (zz).\n\n" + _json()), CLASSES)
    assert label["finding"] == "The pupil summed tops (zz)." and "Finding line" not in label["reasoning"]


def test_reparse_record_recovers_finding_from_text_or_old_fields():
    raw = REASONING + "\n\n2. Finding: The pupil summed tops (zz).\n\n" + _json()
    rec = T.parse_reason(_msg(raw), CLASSES)
    assert rec["text"] == raw
    old = {k: v for k, v in rec.items() if k != "text"}
    old.update(reasoning=REASONING + "\n\n2. Finding: The pupil summed tops (zz).", finding=None, consistent=False)
    fixed = T.reparse_record(old, CLASSES)
    assert fixed["finding"] == "The pupil summed tops (zz)." and fixed["consistent"] and fixed["root_code"] == "FRA-01"
    assert T.reparse_record(rec, CLASSES)["finding"] == rec["finding"]
