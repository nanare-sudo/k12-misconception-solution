"""Synthetic parser regressions of the protocol answers (in-house path and API measurement tool); no model, images or
stored predictions required."""
import json

import pytest

import open_teacher
from vlm import proto


CLASSES = ["NONE", "CAL-01", "Dec-01", "Dec-02"]


@pytest.mark.parametrize("top, probabilities, expected_top, expected_probabilities", [
    (["CAL-01", "Dec-01", "NONE"], [.6, .3, .1], ["CAL-01", "Dec-01", "NONE"], [.6, .3, .1]),
    (["INVALID", "Dec-01", "NONE"], [.8, .15, .05], ["Dec-01", "NONE"], [.15, .05]),
    (["CAL-01", "INVALID", "NONE"], [.6, .3, .1], ["CAL-01", "NONE"], [.6, .1]),
    (["CAL-01", "CAL-01", "NONE"], [.6, .3, .1], ["CAL-01", "NONE"], [.6, .1]),
    (["CAL-01", "Dec-01", "Dec-01", "NONE"], [.5, .3, .15, .05], ["CAL-01", "Dec-01", "NONE"], [.5, .3, .05]),
    (["CAL-01", "Dec-01", "Dec-02", "NONE"], [.5, .3, .15, .05], ["CAL-01", "Dec-01", "Dec-02"], [.5, .3, .15]),
    ([[], "Dec-01", {}], [.8, .15, .05], ["Dec-01"], [.15]),
    (["CAL-01", "INVALID", "Dec-01", "NONE"], [.6, .3, .1], ["CAL-01", "Dec-01", "NONE"], None),
    (["CAL-01"], [.6, .4], ["CAL-01"], None),
    (["CAL-01"], [], ["CAL-01"], None),
    (["CAL-01"], "0.6", ["CAL-01"], None),
    (["CAL-01"], [True], ["CAL-01"], None),
    (["CAL-01"], [float("nan")], ["CAL-01"], None),
    (["CAL-01"], [float("inf")], ["CAL-01"], None),
    (["CAL-01"], [10 ** 400], ["CAL-01"], None),
    (["CAL-01"], None, ["CAL-01"], None),
])
def test_proto_parsers_preserve_code_probability_pairs_and_parity(top, probabilities, expected_top, expected_probabilities):
    text = json.dumps({"root_code": "CAL-01", "top3": top, "probabilities": probabilities,
                       "confidence": .7, "final_answer_correct": False, "readable": "Yes"})
    offline = proto.parse_answer(text, CLASSES)
    teacher = open_teacher.parse_answer(text, CLASSES)
    assert offline is not None and teacher is not None
    assert offline["top3"] == expected_top
    assert offline["probabilities"] == expected_probabilities
    assert {key: teacher[key] for key in offline} == offline


@pytest.mark.parametrize("obj", [
    {"root_code": [], "top3": ["CAL-01"]},
    {"root_code": {}, "top3": ["CAL-01"]},
    {"root_code": "INVALID", "top3": ["CAL-01"]},
    {"root_code": "CAL-01", "top3": "CAL-01"},
    {"root_code": "CAL-01", "top3": ["INVALID"]},
])
def test_proto_parsers_reject_malformed_rankings_without_raising(obj):
    text = json.dumps(obj)
    assert proto.parse_answer(text, CLASSES) is None
    assert open_teacher.parse_answer(text, CLASSES) is None
