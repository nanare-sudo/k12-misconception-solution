"""Mechanical comparison "final answer as read = answer key" for the feature minor_x_mech27 of the second stage.

Number parser WITHOUT eval: the arithmetic expression is read with ast.parse and only evaluated for number literals, + − (also
unary), ·, /, ** and the call sqrt(…) with exactly one argument; every other node is rejected. The pre-checks (only digits, point,
operators, brackets, spaces and "sqrt"; length ≤ 120; at most two "**"; exponent literals |e| ≤ 30) and the arithmetic (Python
operators on the same int/float values, math.sqrt) are those of the independent reference implementation used during development
(a guarded eval; not included) — so the value or the rejection is the same as there (complex results, division by zero, overflow,
non-finite values -> None). One deliberate difference: an EVALUATED exponent above 1,000 is rejected (the reference only checks
literals; 9**9**9 would hang there). On all real answers and answer keys (training and test) there is no difference to the
reference. A peculiarity kept from the reference: the word "sqrt" is dropped together with the letters, so roots are NOT taken
(\\sqrt{16} gives 16, √9 gives 9); n-th roots \\sqrt[n]{x} are computed as x**(1/n).

Indonesian number format: the thousands point (1.234.567) is dropped, the decimal comma becomes a point. LaTeX: \\frac / \\dfrac /
\\tfrac, mixed number a\\frac{b}{c} = a + b/c, \\sqrt{…}, \\sqrt[n]{…}, ×/·/÷, the power signs ² ³ and ^. Of an expression with "="
the part after the last "=" counts; units and words are dropped (form flags in `info`). The answer keys are Indonesian, so the
words "dan" (and) and "persen" (percent) are recognised.

Comparison (mech_match): numeric value of the answer equal to the numeric value of the key (relative 1e-6, at least absolute
1e-6) OR normalised text equal (lower case, without $ and spaces, \\dfrac/\\tfrac -> \\frac). Model answers: the last readable JSON
answer per row with the field student_final_answer (the finishing pass replaces the first pass).

Pure building blocks except `last_answers` (reads frozen answer files). No row is printed.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRTN = re.compile(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_SAFE = re.compile(r"^[0-9.+\-*/() ]+$")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_LEN = 120
MAX_POW = 2
MAX_EXP = 30.0  # pre-check as in the reference: exponent literals
MAX_EXP_EVAL = 1000.0  # run-time guard against hanging (9**9**9); the reference has none, without effect on all real inputs


def num_token(tok: str) -> str:
    """Indonesian number format: thousands point dropped, decimal comma -> point."""
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", tok):
        tok = tok.replace(".", "")
    return tok.replace(",", ".")


def to_expr(s) -> tuple[str, dict]:
    """Text/LaTeX -> arithmetic expression (only numbers, + - * / ** ( ) and sqrt) and form flags."""
    info = {"frac": 0, "dec": 0, "pct": 0, "sqrt": 0, "unit": 0, "text": 0, "neg": 0, "multi": 0, "pow": 0}
    t = str(s or "")
    t = t.replace("\r", " ").replace("\n", " ; ")
    if re.search(r";|\bdan\b|\band\b", t) or len(re.findall(r"=", t)) >= 2:
        info["multi"] = 1
    t = t.replace("$", "").replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    t = t.replace("\\displaystyle", "").replace("\\text", "").replace("\\mathrm", "")
    t = t.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/").replace("\\times", "*")
    t = t.replace("\\cdot", "*").replace("\\div", "/").replace("²", "^2").replace("³", "^3").replace("√", "\\sqrtU")
    if "%" in t or "persen" in t.lower():
        info["pct"] = 1
    if re.search(r"\\[dt]?frac|\d\s*/\s*\d", t):
        info["frac"] = 1
    if re.search(r"\d,\d", t) or re.search(r"\d\.\d(?!\d{2}\b)", t):
        info["dec"] = 1
    if "sqrt" in t:
        info["sqrt"] = 1
    if "^" in t:
        info["pow"] = 1
    if "=" in t:  # only the part after the last "="
        t = t.split("=")[-1]
    letters = re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", " ", t))
    if letters:
        info["unit"] = 1
        if len(letters) >= 2 or any(len(w) > 6 for w in letters):
            info["text"] = 1
    t = re.sub(r"(\d)\s*(?=\\[dt]?frac)", r"\1+", t)  # mixed number a\frac{b}{c} -> a+b/c
    for _ in range(4):
        t2 = _FRAC.sub(r"((\1)/(\2))", t)
        t2 = _SQRTN.sub(r"((\2)**(1/(\1)))", t2)
        t2 = _SQRT.sub(r"(sqrt(\1))", t2)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\\sqrtU\s*\(([^()]*)\)", r"(sqrt(\1))", t)
    t = re.sub(r"\\sqrtU\s*([0-9.,]+)", r"(sqrt(\1))", t)
    t = t.replace("{", "(").replace("}", ")").replace("^", "**")
    t = re.sub(r"\\[A-Za-z]+", " ", t)  # remaining macros
    t = re.sub(r"[A-Za-z°²³%]+", " ", t.replace("sqrt", "SQRT"))
    t = t.replace("SQRT", "sqrt")
    t = re.sub(r"\d[\d.,]*", lambda m: num_token(m.group(0)), t)
    t = re.sub(r"(\d|\))\s*\(", r"\1*(", t)  # implicit multiplication 2(…)
    t = re.sub(r"(\d|\))\s*sqrt", r"\1*sqrt", t)
    t = re.sub(r"\s+", " ", t).strip(" ;:.,")
    if t.startswith("-"):
        info["neg"] = 1
    return t, info


def _ast_value(node: ast.AST):
    """Value of an admitted node; everything else -> ValueError."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_ast_value(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _ast_value(node.left), _ast_value(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXP_EVAL:  # run-time guard: evaluated exponents (e.g. 9**9**9) are bounded
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1
            and not node.keywords):
        return math.sqrt(_ast_value(node.args[0]))
    raise ValueError(f"expression not admitted {type(node).__name__}")


def eval_expr(e: str):
    """Value of the expression e without eval (AST, only numbers, + − · / ** and sqrt) or None; pre-checks as in the reference."""
    if not e or len(e) > MAX_LEN:
        return None
    if not _SAFE.match(e.replace("sqrt", "")):
        return None
    if e.count("**") > MAX_POW:
        return None
    for m in re.finditer(r"\*\*\s*\(?\s*(-?[\d.]+)", e):  # only small exponents (guard against 9**9**9)
        try:
            if abs(float(m.group(1))) > MAX_EXP:
                return None
        except ValueError:
            return None
    try:
        v = float(_ast_value(ast.parse(e, mode="eval").body))
    except Exception:  # syntax, node not admitted, division by zero, complex result, overflow, math.sqrt(<0)
        return None
    return v if math.isfinite(v) else None


def parse_value(s) -> tuple[float | None, dict]:
    e, info = to_expr(s)
    return eval_expr(e), info


def close(a, b, rel: float = 1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def norm_str(s) -> str:
    return str(s or "").lower().replace("$", "").replace(" ", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def mech_match(answer, key: str, key_value: float | None = None) -> bool:
    """The final answer `answer` as read matches the answer key `key` mechanically (numeric value or normalised text).
    key_value: parse_value(key)[0] computed beforehand (otherwise computed here)."""
    if answer in (None, ""):
        return False
    kv = parse_value(key)[0] if key_value is None else key_value
    return bool(close(parse_value(answer)[0], kv) or norm_str(answer) == norm_str(key))


def mech_rows(answers: Sequence, keys: Sequence[str]) -> np.ndarray:
    """Per row 1.0 if the answer (dict with student_final_answer, or None) matches the answer key mechanically, otherwise 0.0."""
    if len(answers) != len(keys):
        raise ValueError(f"{len(answers)} answers, {len(keys)} keys")
    out = np.zeros(len(keys))
    for i, (o, k) in enumerate(zip(answers, keys)):
        if o is not None:
            out[i] = float(mech_match(o.get("student_final_answer"), k))
    return out


def last_answers(paths: Iterable[Path], n: int) -> list[dict | None]:
    """Last readable JSON answer per row from proto_raw.jsonl files (in this order; the finishing pass replaces the first pass):
    the text between the first "{" and the last "}" as JSON, only objects with the field student_final_answer."""
    out: list[dict | None] = [None] * n
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("text") or ""
                i, j = t.find("{"), t.rfind("}")
                if i < 0 or j <= i:
                    continue
                try:
                    o = json.loads(t[i : j + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(o, dict) and "student_final_answer" in o:
                    row = int(r["row"])
                    if not 0 <= row < n:
                        raise ValueError(f"{path}: row {row} outside 0…{n - 1}")
                    out[row] = o
    return out


def minor_mask(classes: Sequence[str], severity_by_code: dict) -> np.ndarray:
    """1.0 for classes with the default severity "minor" in the taxonomy (default_severity), otherwise 0.0; never NONE."""
    m = np.zeros(len(classes))
    for i, c in enumerate(classes):
        if c != "NONE" and str(severity_by_code.get(c, "")).strip().lower() == "minor":
            m[i] = 1.0
    return m
