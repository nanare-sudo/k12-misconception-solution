"""Tests of src/answer_match.py: number comparison "final answer as read = answer key" without eval (AST evaluation), reader of the
frozen answers, minor mask. Synthetic only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMP / "src")]
import answer_match as AM  # noqa: E402


def test_module_has_no_eval():
    import ast

    tree = ast.parse((COMP / "src" / "answer_match.py").read_text())
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"eval", "exec", "compile", "__import__"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("eval", "exec") for n in ast.walk(tree))


@pytest.mark.parametrize("tok, want", [("1.234.567", "1234567"), ("1.234,5", "1234.5"), ("3,5", "3.5"), ("12.5", "12.5"), ("1.250", "1250"),
                                       ("7", "7")])
def test_num_token_indonesian(tok, want):
    assert AM.num_token(tok) == want


@pytest.mark.parametrize("s, want", [
    ("12", 12.0), ("3,5", 3.5), ("1.250", 1250.0), ("Rp 1.500", 1500.0), ("x = 5", 5.0), ("50%", 50.0), ("5 cm", 5.0), ("-4", -4.0),
    ("$\\frac{3}{4}$", 0.75), ("\\dfrac{1}{8}", 0.125), ("2\\frac{1}{2}", 2.5), ("2^3", 8.0), ("2³", 8.0),
    ("3 × 4", 12.0), ("12 ÷ 4", 3.0), ("2(3)", 6.0), ("1/4", 0.25), ("10^-2", 0.01),
    ("\\sqrt{16}", 16.0), ("√9", 9.0),  # peculiarity kept from the reference: the word sqrt is dropped, roots are not taken
])
def test_parse_value_known(s, want):
    v, _ = AM.parse_value(s)
    assert v is not None and math.isclose(v, want, rel_tol=1e-12, abs_tol=1e-12)


def test_parse_value_root_and_info():
    v, info = AM.parse_value("\\sqrt[3]{27}")
    assert AM.close(v, 3.0)
    _, i2 = AM.parse_value("12 dan 13")  # Indonesian "and"
    assert i2["multi"] == 1
    _, i3 = AM.parse_value("50%")
    assert i3["pct"] == 1
    _, i4 = AM.parse_value("5 cm")
    assert i4["unit"] == 1
    _, i5 = AM.parse_value("-4")
    assert i5["neg"] == 1


@pytest.mark.parametrize("e", ["", "1/0", "(-8)**(1/3)", "sqrt(-1)", "2**99", "9**9**9", "(10**30)**30", "1" * 200, "__import__('os')",
                               "sqrt", "2 3", "05", "1.2.3", "(1", "abs(2)", "sqrt(4, 2)", "2**(500*3)"])
def test_eval_expr_rejects(e):
    assert AM.eval_expr(e) is None  # 9**9**9, 2**(500*3): run-time guard (evaluated exponent > 1,000); otherwise as the reference


@pytest.mark.parametrize("e, want", [("1+2*3", 7.0), ("-2**2", -4.0), ("2**-1", 0.5), ("(1+2)*(3-1)", 6.0), ("sqrt(4)", 2.0), ("+5", 5.0),
                                     ("1/3*3", 1.0), ("2**3**2", 512.0), ("00", 0.0), ("0.5", 0.5), ("2**(1+30)", 2147483648.0)])
def test_eval_expr_values(e, want):
    assert AM.eval_expr(e) == want


def test_close_and_norm():
    assert AM.close(1.0, 1.0000001) and AM.close(0.0, 1e-7) and not AM.close(1.0, 1.01)
    assert not AM.close(None, 1.0) and not AM.close(1.0, None)
    assert AM.norm_str("$\\dfrac{1}{2}$ ") == "\\frac{1}{2}" and AM.norm_str(None) == "" and AM.norm_str("A B") == "ab"


@pytest.mark.parametrize("ans, key, want", [("1/2", "\\frac{1}{2}", True), ("0,5", "1/2", True), ("3 cm", "3", True), ("Rp 1.500", "1500", True),
                                            ("", "3", False), (None, "3", False), ("4", "3", False), ("x = 7", "7", True),
                                            ("odd", "Odd", True), ("2", "2 dan 3", False), (0, "0", False)])
def test_mech_match(ans, key, want):
    assert AM.mech_match(ans, key) is want


def test_last_answers_last_readable_wins_and_mech_rows(tmp_path):
    p = tmp_path / "proto_raw.jsonl"
    lines = [{"row": 0, "phase": "first", "text": 'note {"student_final_answer": "1", "readable": "yes"} end'},
             {"row": 0, "phase": "finish", "text": '{"student_final_answer": "2"}'},
             {"row": 1, "phase": "first", "text": "no JSON"},
             {"row": 2, "phase": "first", "text": '{"other": 1}'},
             {"row": 3, "phase": "first", "text": '{"student_final_answer": "1/2"}'},
             {"row": 3, "phase": "finish", "text": "{broken"}]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    A = AM.last_answers([p], 4)
    assert A[0]["student_final_answer"] == "2" and A[1] is None and A[2] is None and A[3]["student_final_answer"] == "1/2"
    m = AM.mech_rows(A, ["2", "1", "1", "0,5"])
    assert m.tolist() == [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        AM.mech_rows(A, ["1"])
    with pytest.raises(ValueError):
        AM.last_answers([p], 2)  # row 3 outside


def test_minor_mask():
    cls = ["NONE", "AAA-01", "BBB-02", "CCC-03"]
    m = AM.minor_mask(cls, {"NONE": "minor", "AAA-01": "minor", "BBB-02": "major", "CCC-03": " Minor "})
    assert m.tolist() == [0.0, 1.0, 0.0, 1.0]
