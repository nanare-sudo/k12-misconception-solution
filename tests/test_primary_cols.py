"""Tests of the intermediate build primary_cols (extra columns of the second stage): src/stage2_sources.py (column sources from frozen
protocol answers, test guard, positive controls), the additive path in tools/make_submission.py (field extra_columns, --stage2-xcontrol),
eda/c151_primary_cols.py (comparison form of the copy null point, exit codes) and the build path tools/builds/primary_cols.sh (usage,
constants).

SYNTHETIC with files under tmp_path; with real training artifacts (skipped if they are missing): deployment file against its base, matrices of
the training side from the frozen sources = checksums of the deployment file; built files only through their checksum against the pin in the
build path. No test row is read or printed."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
ART = COMP / "artifacts"
sys.path[:0] = [str(COMP / "src")]

import stage2 as S2  # noqa: E402
import stage2_sources as XS  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_submission_primary_cols_test", "tools/make_submission.py")
C151 = _load("c151_primary_cols_test", "eda/c151_primary_cols.py")
SH = COMP / "tools" / "builds" / "primary_cols.sh"
# base/variant built by the build path (deployment file; file and test matrix as submitted)
PINNED = (("m3x4", "qfp"),)
CLASSES =["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "ALG-01", "FRA-02", "DEC-03"]
K = len(CLASSES)
ST = XS.RECORDED_STATE  # recorded labels of the note states


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        pytest.skip(f"artifacts missing: {missing}")


def _answer(root: str, top: list[str], findings: list[dict] | None = None, sfa=None) -> str:
    obj = {"root_code": root, "top3": top, "probabilities": [0.6, 0.3, 0.1][: len(top)], "findings": findings or []}
    if sfa is not None:
        obj["student_final_answer"] = sfa
    return "Answer:\n" + json.dumps(obj)


def _write_jsonl(path: Path, recs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return path


# ------------------------------------------------------------------------------------------------ parsers and reading paths

def test_clean_answer_and_json_obj():
    valid = set(CLASSES)
    assert XS.json_obj("no json") is None
    a = XS.clean_answer(XS.json_obj(_answer("ALG-01", ["ALG-01", "XYZ-99", "ANS-01"], [{"code": "ALG-01", "severity": " Major ", "step": "L2"}], "12")), valid)
    assert a["root_code"] == "ALG-01" and a["top3"] == ["ALG-01", "ANS-01"]
    assert a["findings"][0]["severity"] == "major" and a["findings"][0]["code"] == "ALG-01"
    assert a["student_final_answer"] == "12"
    assert XS.clean_answer(XS.json_obj(_answer("XYZ-99", ["ALG-01"])), valid) is None  # root outside the classes
    assert XS.clean_answer({"root_code": "ALG-01", "top3": []}, valid) is None  # without valid top3


def test_tinker_parse_matches_open_teacher():
    saved = list(sys.path)
    OT = _load("open_teacher_primary_cols_test", "tools/open_teacher.py")
    sys.path[:] = saved
    texts = [_answer("ALG-01", ["ALG-01", "ANS-01"], [{"code": "ALG-01", "severity": "major"}, {"code": "BAD", "severity": "minor"}] * 8, 7),
             _answer("NONE", ["NONE"], None, None), "nothing", '{"root_code": ["x"], "top3": ["NONE"]}',
             _answer("CAL-01", ["CAL-01", "CAL-01", "NONE"], [{"code": "CAL-01"}], "x" * 300)]
    for t in texts:
        assert XS.tinker_parse(t, CLASSES) == OT.parse_answer(t, CLASSES)
    p = XS.tinker_parse(texts[0], CLASSES)
    assert len(p["findings"]) == 8 and p["student_final_answer"] == "7"  # only valid codes (at most 12), number as text
    assert len(XS.tinker_parse(texts[4], CLASSES)["student_final_answer"]) == 200


def test_rank_notes_and_state():
    rec = {"phase": "first", "thinking": "<think>  abc root_code  </think>", "end": "length"}
    assert XS.rank_notes(rec) == "abc root_code"
    assert XS.rank_notes({**rec, "phase": "finish"}) == ""
    assert XS.rank_notes({"phase": "first", "thinking": "x" * 20}, note_chars=5) == "xxxxx"
    assert XS.note_state(rec, "abc root_code") == ST["draft"]
    assert XS.note_state(rec, "abc") == ST["open"]
    assert XS.note_state({**rec, "end": "stop"}, "abc") == ST["complete"]
    assert XS.note_state({"stop": "stop"}, "") == ST["complete"]  # Tinker: field stop


def _own_run(tmp: Path, name: str, recs: list[dict], table: str = "train", rows: int = 3) -> Path:
    d = tmp / name
    raw = _write_jsonl(d / "proto_raw.jsonl", recs)
    (d / "proto_meta.json").write_text(json.dumps({"table": table, "rows": rows}))
    return raw


def test_read_own_last_valid_answer_notes_and_states(tmp_path):
    recs = [{"row": 0, "phase": "first", "thinking": "think ALG-01 then CAL-01", "text": "", "end": "length"},
            {"row": 0, "phase": "finish", "thinking": "", "text": _answer("CAL-01", ["CAL-01"]), "end": "stop"},
            {"row": 1, "phase": "first", "thinking": "", "text": _answer("NONE", ["NONE"]), "end": "stop"},
            {"row": 2, "phase": "first", "thinking": "x", "text": "broken", "end": "stop"}]
    raw = _own_run(tmp_path, "a", recs)
    rows, cnt = XS.read_own([raw], 3, CLASSES, "json", "pr27")
    assert cnt["first"] == 3 and cnt["phases"] == {"first": 3, "finish": 1}
    assert rows[0]["ans"]["root_code"] == "CAL-01" and rows[0]["first_end"] == "length" and not rows[0]["first_valid"]
    assert rows[0]["notes"] == "think ALG-01 then CAL-01" and rows[0]["state"] == ST["open"]
    assert rows[1]["state"] is None and rows[1]["first_valid"] and rows[1]["notes"] == ""  # pr27: no state without a note
    assert rows[2]["ans"] is None
    rows_q, _ = XS.read_own([raw], 3, CLASSES, "tinker", "q397")
    assert rows_q[1]["state"] == ST["complete"]  # q397: state for every first pass, also without a note
    with pytest.raises(XS.SourceError):
        XS.read_own([_own_run(tmp_path, "b", recs + [recs[0]])], 3, CLASSES, "json", "pr27")


def test_own_labels_check_detects_parser_mismatch(tmp_path):
    recs = [{"row": 0, "phase": "first", "thinking": "", "text": _answer("ALG-01", ["ALG-01"]), "end": "stop"}]
    raw = _own_run(tmp_path, "c", recs, rows=1)
    good = _write_jsonl(tmp_path / "c" / "proto_labels.jsonl", [{"row": 0, "phase": "first", "ok": True, "root_code": "ALG-01", "top3": ["ALG-01"]}])
    assert XS.own_labels_check([raw], [good], CLASSES, "json") == {"records": 1, "mismatch": 0, "same_keys": True}
    bad = _write_jsonl(tmp_path / "c" / "bad_labels.jsonl", [{"row": 0, "phase": "first", "ok": True, "root_code": "CAL-01", "top3": ["CAL-01"]}])
    assert XS.own_labels_check([raw], [bad], CLASSES, "tinker")["mismatch"] == 1


def test_read_tinker_uses_labels_and_q397_state(tmp_path):
    lab = _write_jsonl(tmp_path / "lab.jsonl", [
        {"row": 0, "phase": "first", "ok": False, "stop": "length"},
        {"row": 0, "phase": "finish", "ok": True, "stop": "stop", "root_code": "ANS-01", "top3": ["ANS-01"], "findings": [], "student_final_answer": ""},
        {"row": 1, "phase": "first", "ok": True, "stop": "stop", "root_code": "NONE", "top3": ["NONE"], "findings": [], "student_final_answer": "5"}])
    notes = _write_jsonl(tmp_path / "notes.jsonl", [{"table": "train", "row": 0, "state": "open", "text": "t0", "finding": None, "q397_state": ST["draft"]},
                                                    {"table": "train", "row": 1, "state": "open", "text": "t1", "finding": None, "q397_state": ST["complete"]}])
    rows, cnt = XS.read_tinker(lab, notes, 2, CLASSES)
    assert cnt == {"records": 3, "first": 2}
    assert rows[0]["ans"]["root_code"] == "ANS-01" and rows[0]["first_end"] == "length" and rows[0]["state"] == ST["draft"]
    assert rows[1]["first_valid"] and rows[1]["notes"] == "t1"


# ------------------------------------------------------------------------------------------------ building blocks and columns

def _rows(spec: list[tuple]) -> list[dict]:
    out = []
    for root, notes, state, end, findings, sfa in spec:
        ans = None if root is None else XS.clean_answer({"root_code": root, "top3": [root], "findings": findings, "student_final_answer": sfa}, set(CLASSES))
        out.append({"ans": ans, "first_end": end, "first_valid": ans is not None and end == "stop", "notes": notes, "state": state})
    return out


def test_parts_formed_hierarchy_last_mention_major_and_mech():
    rows = _rows([("ALG-01", "first ANS-01, finally NONE", None, "length", [{"code": "ALG-01", "severity": "major"}, {"code": "CAL-01", "severity": "minor"}], "4"),
                  ("CAL-01", 'draft "root_code": "CAL-01"', None, "length", [], None),
                  ("NONE", "Root: NONE", None, "length", [], "3"),
                  ("ANS-01", "no verdict", None, "stop", [], ""),
                  (None, "", ST["complete"], "length", [], None)])
    P = XS.parts(rows, CLASSES, ["4", "1", "3", "9", "0"])
    assert P["root"].tolist() == [6, 2, 0, 1, -1]
    assert P["last"][0].tolist() == [1] + [0] * (K - 1)  # last code mentioned: NONE
    assert P["formed"].tolist() == [False, True, True, True, True]  # open / draft / spoken / stop+valid / state complete
    assert P["f_major"][0, 6] == 1 and P["f_major"][0].sum() == 1
    assert P["mech"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0] and P["has_read"].tolist() == [True, False, True, False, False]
    assert P["valid"].tolist() == [True, True, True, True, False]


def _two_sources(n: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    P = {}
    for s in XS.SOURCES:
        root = rng.integers(-1, K, n)
        P[s] = {"root": root, "last": (rng.random((n, K)) < 0.1).astype(np.int8), "f_major": (rng.random((n, K)) < 0.1).astype(np.int8),
                "formed": rng.random(n) < 0.5, "mech": (rng.random(n) < 0.4).astype(float), "has_read": np.ones(n, bool), "valid": root >= 0}
    return P


def _inputs(n: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + 100)
    keys = np.array([f"q{i % 3}|t{i % 2}" for i in range(n)])
    chans = [rng.dirichlet(np.ones(K), n) for _ in range(3)]
    draws = [rng.dirichlet(np.ones(K), n) for _ in range(3)]
    sections = np.array(["NONE", "CATCH-ALL", "CROSS-STRAND", "CROSS-STRAND", "CATCH-ALL", "CATCH-ALL", "SECTION-A", "SECTION-B",
                         "SECTION-C"])
    return {"classes": CLASSES, "P": _two_sources(n, seed), "minor": np.array([0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]), "row_keys": keys, "ref_keys": keys,
            "ref_y": rng.integers(0, K, n), "fac": (rng.random(n) < 0.4).astype(float), "sev_minor": np.array([0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]),
            "chans": chans, "draws": draws, "seen": np.ones(K, bool), "row_topics": np.array([f"t{i % 2}" for i in range(n)]),
            "ref_topics": np.array([f"t{i % 2}" for i in range(n)]), "sections": sections}


def test_label_free_matrices_definitions():
    P = _two_sources()
    minor = np.zeros(K)
    minor[2] = 1.0
    V = XS.label_free(P, minor)
    oh = {s: XS.onehot(P[s]["root"], K) for s in XS.SOURCES}
    assert np.array_equal(V["c_vote[pr27+q397]"], oh["pr27"] + oh["q397"])
    assert np.array_equal(V["c_formed[pr27+q397]"], oh["pr27"] * P["pr27"]["formed"][:, None] + oh["q397"] * P["q397"]["formed"][:, None])
    assert np.array_equal(V["b_last[pr27+q397]"], P["pr27"]["last"] + P["q397"]["last"])
    cons = P["pr27"]["mech"] + P["q397"]["mech"]
    assert np.array_equal(V["d_minor[pr27+q397]"][:, 2], -cons) and np.signbit(V["d_minor[pr27+q397]"][:, 0]).all()  # −0.0 as in the measurement
    assert all(v.dtype == np.float64 for v in V.values())


def test_novel_column_reads_labels_only_from_ref():
    keys = np.array(["q1|t", "q1|t", "q2|t", "q2|t"])
    y = np.array([1, 0, 2, 2])
    W = np.zeros((4, K))
    W[np.arange(4), [1, 2, 2, 0]] = 1.0
    col = XS.NovelColumn(keys, W, keys, y)
    cand = np.tile(np.arange(5), (4, 1))
    out = col.cols(np.arange(4), cand, np.array([2, 3])).reshape(4, 5)  # reference only the q2 rows: q1 knows nothing
    assert out[0, 1] == 1.0 and out[1, 2] == 1.0 and out[2, 2] == 0.0 and out[3, 0] == 0.0  # NONE never new; code 2 known for q2
    y2 = y.copy()
    y2[[0, 1]] = [4, 4]  # labels outside ref change nothing
    assert np.array_equal(XS.NovelColumn(keys, W, keys, y2).cols(np.arange(4), cand, np.array([2, 3])), out.reshape(-1, 1))
    test_col = XS.NovelColumn(np.array(["q9|t"]), np.eye(K)[[1]], keys, y)  # task without a training row: new
    assert test_col.cols(np.array([0]), np.array([[1, 0, 2, 3, 4]]), np.arange(4))[0, 0] == 1.0


def test_fac_novelty_section_columns():
    inp = _inputs(6)
    cols = XS.build_columns(list(XS.SOURCE_ORDER), inp)
    assert cols.features == XS.feature_names(XS.SOURCE_ORDER) and len(cols.features) == 15
    cand = np.tile(np.array([5, 1, 2, 0, 6]), (6, 1))  # OTH-00, ANS-01, CAL-01 (minor), NONE, ALG-01
    E = cols.cols(np.arange(6), cand, np.arange(3)).reshape(6, 5, 15)
    j = cols.features.index("fac27_rule:minor")
    assert np.array_equal(E[:, 2, j], inp["fac"]) and not E[:, 0, j].any() and np.array_equal(E[:, 1, j + 1], inp["fac"])
    o = cols.features.index("oth_novelty:ent")
    assert (E[:, 1:, o:o + 6] == 0).all() and (E[:, 0, o] > 0).all()  # only the candidate OTH-00 carries values
    F5 = XS.novelty_rows(inp["chans"], inp["draws"], inp["seen"], np.isin(np.array(CLASSES), XS.GENERAL))
    assert np.array_equal(E[:, 0, o:o + 5], F5)
    cnt = {k: int((inp["ref_keys"][:3] == k).sum()) for k in set(inp["row_keys"])}
    assert np.array_equal(E[:, 0, o + 5], [cnt[k] for k in inp["row_keys"]])  # task count only over ref
    s = cols.features.index("psy_adm_sec")
    assert (E[:, 0, s] == 1).all() and (E[:, 3, s] == 1).all()  # CATCH-ALL and NONE always admissible
    unknown = XS.build_columns(["psy_adm_sec"], {**inp, "row_topics": np.full(6, "nowhere")}).cols(np.arange(6), cand, np.arange(6)).reshape(6, 5)
    assert unknown[:, 4].sum() == 0 and unknown[:, 0].all()  # unknown topic: only the always admissible sections


def test_columns_order_zero_and_build():
    inp = _inputs(4)
    cols = XS.build_columns(XS.VARIANTS["q"], inp)
    assert cols.names == list(XS.COLUMNS) and cols.features == list(XS.COLUMNS)
    cand = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 0], [1, 2, 3, 4, 5], [6, 7, 8, 0, 1]])
    E = cols.cols(np.arange(4), cand, np.arange(4))
    assert E.shape == (20, 6) and E.dtype == np.float64
    V = XS.label_free(inp["P"], inp["minor"])["c_vote[pr27+q397]"]
    assert np.array_equal(E[:, 0], V[np.repeat(np.arange(4), 5), cand.reshape(-1)])  # row outside, candidate inside
    z = XS.build_columns(XS.SOURCE_ORDER, _inputs(4)).zero()
    assert z.cols(np.arange(4), cand, np.arange(4)).shape == (20, 15) and not z.cols(np.arange(4), cand, np.arange(4)).any()
    with pytest.raises(KeyError):
        XS.feature_names(["does_not_exist"])
    assert XS.variant_of(XS.VARIANTS["qfo"]) == "qfo" and XS.variant_of(["fac27_rule"]) is None


def test_guard_cols_use_all_rows_only_for_task_count():
    inp = _inputs(6)
    cols = XS.build_columns(list(XS.SOURCE_ORDER), inp)
    cand = np.tile(np.array([5, 1, 2, 0, 6]), (6, 1))
    a = cols.cols(np.arange(3), cand[:3], np.arange(3, 6))
    b = cols.guard_cols(np.arange(3), cand[:3], np.arange(3, 6), np.arange(6))
    t = cols.features.index("oth_novelty:task_rows")
    diff = np.flatnonzero((a != b).any(0))
    assert set(diff.tolist()) <= {t}  # only the task count changes its reference


# ------------------------------------------------------------------------------------------------ test guard and positive controls

def _stats(seed: int, n: int = 400, shift: float = 0.0) -> dict:
    inp = _inputs(n, seed)
    cols = XS.build_columns(list(XS.SOURCE_ORDER), inp)
    cand = np.tile(np.arange(5), (n, 1))
    E = cols.cols(np.arange(n), cand, np.arange(n))
    if shift:
        E = E + shift * (E != 0)
    return XS.guard_stats(cols, E, cand, inp, np.arange(n), np.arange(n))


def test_guard_rule_frozen():
    assert XS.GUARD_RULE["rel"] == 0.15 and XS.GUARD_RULE["z_max"] == 3.0 and XS.GUARD_RULE["floors"] == {"share_valid": 0.97}
    assert XS.CONTROLS == ("q397_drop10", "q397_unformed", "mech_zero", "q397_rotate", "fac_flip", "chan_rotate", "topic_unknown")
    assert set(XS.CONTROL_TARGET) == set(XS.CONTROLS) == set(XS.CONTROL_NEEDS) and set(XS.CONTROL_NEEDS.values()) <= set(XS.SOURCE_ORDER)


def test_guard_check_passes_same_distribution_and_fails_shift():
    ref = _stats(1)
    g = XS.guard_check(ref, _stats(2), floors={})
    assert g["ok"], g["failed"]
    assert len(g["checks"]) == 4 * 15 + 3 * len(XS.SOURCES) + 1 + 7
    bad = XS.guard_check(ref, _stats(3, shift=1.0), floors={})
    assert not bad["ok"] and any(c["stat"] == "mean" for c in bad["failed"])


def test_guard_check_band_or_z_and_floor():
    ref = {"rows": 1500, "columns": {}, "sources": {"q397": {"share_valid": 0.997, "share_formed": 0.40, "share_mech": 0.30}}, "sources_se": {}}
    t = {"rows": 3498, "columns": {}, "sources": {"q397": {"share_valid": 0.90, "share_formed": 0.47, "share_mech": 0.10}}, "sources_se": {}}
    g = XS.guard_check(ref, t)
    by = {c["stat"]: c for c in g["checks"]}
    assert by["share_valid"]["ok_band"] and not by["share_valid"]["ok"] and by["share_valid"]["ok_floor"] is False  # floor 0.97 (addendum)
    assert not by["share_formed"]["ok_band"] and by["share_formed"]["passed_by"] is None  # +17.5 % and z > 3
    assert not by["share_mech"]["ok"] and not g["ok"]
    t["sources"]["q397"].update(share_valid=0.995, share_formed=0.41, share_mech=0.29)
    assert XS.guard_check(ref, t)["ok"]


def test_column_stats_definitions():
    cand = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    E = np.array([[2, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0]], float).reshape(-1, 1)
    st = XS.column_stats(E, cand, ["x"])["columns"]["x"]
    assert st["share_rows_nonzero"] == pytest.approx(2 / 3)
    assert st["share_rows_support_top"] == pytest.approx(2 / 3)  # row 2 constant: no support; row 3: a tie at the top counts
    assert st["mean"] == pytest.approx(5 / 15) and st["sd"] == pytest.approx(np.std(E))


def test_controls_change_their_target_and_do_not_mutate():
    inp = _inputs(200, 5)
    inp["P"]["q397"]["formed"][:] = True
    keep = {s: {k: v.copy() for k, v in inp["P"][s].items()} for s in XS.SOURCES}
    fac0, ch0, top0 = inp["fac"].copy(), inp["chans"][0].copy(), inp["row_topics"].copy()
    for c in XS.CONTROLS:
        Q = XS.apply_control(inp, c)
        assert all(np.array_equal(inp["P"][s][k], keep[s][k]) for s in XS.SOURCES for k in keep[s])
        assert np.array_equal(inp["fac"], fac0) and np.array_equal(inp["chans"][0], ch0) and np.array_equal(inp["row_topics"], top0)
        if c == "q397_drop10":
            assert (Q["P"]["q397"]["root"] == -1).sum() >= 20 and Q["P"]["q397"]["valid"].sum() <= inp["P"]["q397"]["valid"].sum()
        elif c == "q397_unformed":
            assert not Q["P"]["q397"]["formed"].any()
        elif c == "mech_zero":
            assert not Q["P"]["pr27"]["mech"].any() and not Q["P"]["q397"]["mech"].any()
        elif c == "q397_rotate":
            assert np.array_equal(Q["P"]["q397"]["root"], np.roll(inp["P"]["q397"]["root"], 1))
        elif c == "fac_flip":
            assert np.array_equal(Q["fac"], 1.0 - fac0)
        elif c == "chan_rotate":
            assert np.array_equal(Q["chans"][0], np.roll(ch0, 1, axis=0)) and Q["chans"][1] is inp["chans"][1]
        else:
            assert (Q["row_topics"] == XS.UNKNOWN_TOPIC).all()
    with pytest.raises(ValueError):
        XS.apply_control(inp, "unknown")


def test_sources_digest_detects_change(tmp_path):
    spec = {"pr27": {"kind": "own_json", "raw": ["artifacts/r/p/proto_raw.jsonl"], "labels_check": ["artifacts/r/p/proto_labels.jsonl"]},
            "q397": {"kind": "tinker", "labels": "artifacts/r/q/proto_labels.jsonl", "notes": "derived/n.jsonl"}}
    for rel in ("artifacts/r/p/proto_raw.jsonl", "artifacts/r/p/proto_labels.jsonl", "artifacts/r/p/proto_meta.json", "artifacts/r/p/summary.json",
                "artifacts/r/q/proto_labels.jsonl"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    (tmp_path / "data" / "derived").mkdir(parents=True)
    (tmp_path / "data" / "derived" / "n.jsonl").write_text("n")
    d1, n1 = XS.sources_digest(tmp_path, tmp_path / "data", {"train": spec})
    assert n1 == 6
    (tmp_path / "data" / "derived" / "n.jsonl").write_text("m")
    assert XS.sources_digest(tmp_path, tmp_path / "data", {"train": spec})[0] != d1
    (tmp_path / "artifacts/r/p/summary.json").unlink()
    with pytest.raises(XS.SourceError):
        XS.sources_digest(tmp_path, tmp_path / "data", {"train": spec})


def test_fac_rows_reads_last_answer(tmp_path):
    raw = _write_jsonl(tmp_path / "proto_raw.jsonl", [
        {"row": 0, "phase": "first", "text": json.dumps({"student_final_answer": "1", "final_answer_correct": True})},
        {"row": 0, "phase": "finish", "text": json.dumps({"student_final_answer": "2", "final_answer_correct": False})},
        {"row": 1, "phase": "first", "text": json.dumps({"student_final_answer": "3", "final_answer_correct": True})},
        {"row": 2, "phase": "first", "text": json.dumps({"final_answer_correct": True})}])
    assert XS.fac_rows([raw], 3).tolist() == [0.0, 1.0, 0.0]  # the last answer with student_final_answer counts


# ------------------------------------------------------------------------------------------------ make_submission (additive)

def test_make_submission_extra_columns_path_is_additive():
    src = inspect.getsource(MS.apply_stage2)
    assert inspect.signature(MS.apply_stage2).parameters["xcontrol"].default is None
    assert src.count('if "extra_columns" in s2:') == 1 and src.count("_stage2_extra_columns(") == 1
    assert 'elif xcontrol is not None:' in src  # a control without the field is rejected
    main = inspect.getsource(MS.main)
    assert main.count('extra["xcontrol"] = a.stage2_xcontrol') == 1 and '"extra_columns" not in s2j' in main
    safe = inspect.getsource(MS.apply_stage2_safe)
    assert "extra_columns" not in safe and "xcontrol" not in safe


@pytest.mark.parametrize("patch, match", [({"names": []}, r"extra_columns\.names \[\]"), ({"names": ["x"]}, r"extra_columns\.names \['x'\]"),
                                          ({"version": "old"}, r"'old'"),
                                          ({"features": ["c_vote[pr27+q397]"]}, r"extra_columns\.features"),
                                          ({"guard_rule": {"rel": 0.3, "z_max": 3.0, "floors": {"share_valid": 0.97}}}, r"extra_columns\.guard_rule"),
                                          ({"guard_rule": {"rel": 0.15, "z_max": 3.0, "floors": {}}}, r"extra_columns\.guard_rule")])
def test_extra_columns_refuses_bad_fields(patch, match):
    spec = {"names": list(XS.COLUMNS), "features": list(XS.COLUMNS), "module": "src/stage2_sources.py", "version": XS.VERSION,
            "guard_rule": dict(XS.GUARD_RULE), **patch}
    with pytest.raises(SystemExit, match=match):
        MS._stage2_extra_columns({"extra_columns": spec}, None, None, CLASSES, COMP, np.zeros((1, 5), int), [], [], np.ones(K, bool))


def test_xcontrol_choices_equal_module_controls():
    ap_src = inspect.getsource(MS.main)
    assert '"--stage2-xcontrol", choices=XS.CONTROLS' in ap_src


# ------------------------------------------------------------------------------------------------ c151 and build path

def test_c151_exit_codes_and_seeds():
    assert C151.exit_code_of(C151.Missing("x")) == 3 and C151.exit_code_of(C151.TechnicalError("x")) == 4
    assert C151.exit_code_of(C151.GuardFail("x")) == 5 and C151.exit_code_of(C151.ControlFail("x")) == 6
    assert C151.exit_code_of(XS.SourceError("x")) == 4 and C151.exit_code_of(ValueError("x")) is None
    assert C151.parse_seeds("orig,3000-3002,7") == [None, 3000, 3001, 3002, 7]


def test_c151_normalized_compare():
    base = {"a": 1, "models": {"main": {"features": ["f1", "f2"], "w": [0.5, 0.2], "mu": [0, 0], "sd": [1, 1], "features_sha256": "s", "oof_vs_i1": {"x": 1},
                                        "gate": {"abstain": False}}, "fallback": {"w": [1]}}}
    cp = {"a": 1, "models": {"main": {"features": ["f1", "f2", "c"], "extra_columns": ["c"], "w": [0.5, 0.2, 0.0], "mu": [0, 0, 0.0], "sd": [1, 1, 1e-9],
                                      "features_sha256": "t", "features_sha256_mm": "s", "oof_vs_mm": {"x": 0.0}, "gate": {"abstain": False}},
                             "fallback": {"w": [1]}}, "extra_columns": {}, "cols_build": {}}
    assert C151.normalized(cp) == C151.normalized(base)
    cp["models"]["main"]["w"][1] = 0.3
    assert C151.normalized(cp) != C151.normalized(base)


def _pins(text: str) -> re.Match:
    m = re.search(r'^STAGE2_SHA="([0-9a-f]{64})?"; OUT_SHA="([0-9a-f]{64})?"; TESTMAT_SHA="([0-9a-f]{64})?"', text, re.M)
    assert m, "pin line of the build path"
    return m


def test_build_script_call_and_constants():
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0
    assert subprocess.run(["bash", str(SH), "x"], capture_output=True).returncode == 2
    text = SH.read_text()
    m = _pins(text)
    assert m.group(2) and m.group(3), "file and test matrix are pinned to the submitted values"
    for base, cols in PINNED:
        out = ART / "submissions" / f"sub_{C151.out_name(base, cols)}.csv"
        if out.is_file():  # only the checksum of the file (its content is not read)
            assert _sha(out) == m.group(2), f"{out.name} differs from the pin in the build path"
    assert re.search(r'^SOURCES_SHA="[0-9a-f]{64}"', text, re.M), "checksum of the column sources not pinned"
    assert "tools/builds/primary_m3x4.sh" in text and "--stage2-xcontrol" not in text


def test_build_script_stage2_pin():
    m = _pins(SH.read_text())
    if not m.group(1):
        pytest.skip("STAGE2_SHA empty: the deployment file is written by the code of this repository (first build prints its checksum)")
    for base, cols in PINNED:
        f = ART / f"stage2_{C151.out_name(base, cols)}.json"
        _need(f)
        assert _sha(f) == m.group(1), f"{f.name} differs from the pin in the build path"


# ------------------------------------------------------------------------------------------------ real artifacts (training side only)

@pytest.mark.parametrize("base, cols", PINNED)
def test_deploy_file_against_base(base, cols):
    b = C151.BASES[base]
    f, f0 = ART / f"stage2_{C151.out_name(base, cols)}.json", ART / b["stage2"]
    _need(f, f0)
    e, e0 = json.loads(f.read_text()), json.loads(f0.read_text())
    assert list(e) == [*e0, "extra_columns", "cols_build"]
    assert all(e[k] == e0[k] for k in e0 if k != "models") and e["models"]["fallback"] == e0["models"]["fallback"]
    m, m0 = e["models"]["main"], e0["models"]["main"]
    feats = XS.feature_names(XS.VARIANTS[cols])
    assert m["features"] == m0["features"] + feats and m["extra_columns"] == feats and len(m["w"]) == 64 + len(feats)
    assert m["feature_set"] == S2.MM_SET and m["protect_none"] is True and m["features_sha256_mm"] == m0["features_sha256"]
    x = e["extra_columns"]
    assert x["version"] == XS.VERSION and x["guard_rule"] == XS.GUARD_RULE and x["test"]["spec"] == XS.TEST_SPEC and x["train"]["spec"] == XS.TRAIN_SPEC
    assert x["names"] == list(XS.VARIANTS[cols]) and x["features"] == feats and x["variant"] == cols and x["channels"] == e0["channels"]
    assert set(x["test"]["files_sha256"]) == set(XS.source_files(XS.TEST_SPEC))
    assert e["cols_build"]["kind"] == "deploy" and e["cols_build"]["base"]["sha256"] == _sha(f0) and e["cols_build"]["variant"] == cols


def test_protocol_matrices_reproduce_deploy_file():
    """The six protocol columns from the frozen training sources = checksums of the deployment file (the six columns of the 397B group of the
    deployed variant, without blend channels)."""
    f = ART / f"stage2_{C151.out_name('m3x4', C151.DEFAULT_COLS)}.json"
    _need(f, *(COMP / r for r in XS.TRAIN_SPEC["pr27"]["raw"]), COMP / XS.TRAIN_SPEC["q397"]["labels"])
    import labels
    import pandas as pd
    from vlm import common as VC

    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    P, info = XS.load_side(XS.TRAIN_SPEC, "train", COMP, root, classes, [str(k) for k in train["correct_answer"].tolist()])
    tax = pd.read_csv(root / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)
    keys = S2.task_keys(train["question_id"].to_numpy(), train["topic_name"].to_numpy())
    inp = {"classes": classes, "P": P, "minor": XS.minor_vector(classes, dict(zip(tax["id"], tax["default_severity"]))), "row_keys": keys,
           "ref_keys": keys, "ref_y": labels.encode(train["root_code"], classes)}
    cols = XS.build_columns(XS.VARIANTS["q"], inp)
    x = json.loads(f.read_text())["extra_columns"]
    assert XS.matrices_sha256(cols, len(train), len(classes)) == {k: x["train"]["matrices_sha256"][k] for k in XS.VARIANTS["q"]}
    assert info["files"] == x["train"]["files_sha256"]
