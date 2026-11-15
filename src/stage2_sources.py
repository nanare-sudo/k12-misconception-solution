"""Extra columns of the second stage from frozen sources (intermediate build primary_cols and the primary build that extends it; chosen by the
bundle test of the extra columns: all columns of a bundle tested TOGETHER against a placebo bundle on untouched splits).

One named column source per source, with the interface of the internal bundle measurement (the production columns were verified byte for
byte against it internally):

    obj.cols(rows, cand, ref) -> np.ndarray [len(rows)·Kc, m] (float64)   # row outside, candidate inside (as stage2.pair_features)

rows = table positions of the rows (training or test rows), cand = [len(rows), Kc] class indices of the candidates, ref = positions of the
reference rows in the TRAINING TABLE (training rows: the other folds, for training blocks also without the outer fold; test rows: all 1,500).
Only label-dependent sources read labels, and only from ref.

SOURCES (order SOURCE_ORDER; definitions exactly as in the measurement):
  c_vote[pr27+q397]     number of sources (pr27, q397) whose root_code is the code.
  b_last[pr27+q397]     number of sources whose note (reasoning of the first pass) names the code LAST.
  a_major[pr27+q397]    number of finding lists with the code as major.
  c_formed[pr27+q397]   votes only of the sources with a formed verdict (complete/draft/spoken; hierarchy of Texts.verdict).
  d_minor[pr27+q397]    −minor(code) · number of sources whose read final answer matches the key mechanically.
  k_nov_single[q397]    top code of the 397B answer × code never gold for the task (question_id|topic_name) in ref (NONE never new).
                                                                                                                       (label-dependent)
  fac27_rule            pr27 "final_answer_correct" (last readable JSON answer, answer_match.last_answers) × {default severity of the
                        candidate = minor, 1[candidate = ANS-01]} — 2 columns.
  oth_novelty           × 1[candidate = OTH-00]: entropy, maximum and largest special mass of the mean of the blend channels (probabilities
                        BEFORE the floor, without margin alignment = OOF artifacts or make_submission raw_chans), number of distinct channel
                        top codes and draw top codes (d1–d3; as stage2.Static), number of reference rows of the same task — 6 columns.
  psy_adm_sec           1 if the taxonomy section of the candidate occurred as gold within the topic of the row among the reference rows; the
                        sections NONE, CROSS-STRAND, CATCH-ALL, "NUMBER — Place Value & Operations" always 1.         (label-dependent)
VARIANTS (VARIANTS): q (the six columns of the 397B group), qf, qfo, qfop (all nine sources, 15 columns), qfp (without oth_novelty).

FILES of the protocol answers (the deployment file holds the checksums, field extra_columns):
  training  pr27 = artifacts/runs/pr27_rp/pr27_rp/proto_{a,b}/proto_raw.jsonl (own computation path); q397 = Tinker answers
            artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_labels.jsonl + notes data/derived/sft_q397m3/q397m3_train_notes.jsonl.
  test      pr27 = artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl; q397 = artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_raw.jsonl
            (own computation path) with the parser of the Tinker measurement (tools/open_teacher.parse_answer) — so training and test carry
            the same fields. proto_labels.jsonl of each run is the cross-check of the parser.
The caller provides blend channels, draws, table and taxonomy (training: context of the second stage; test: tools/make_submission.py).
TRANSFER RISK q397 (Tinker -> own computation path): test guard per column and source (guard_stats/guard_check, rule GUARD_RULE) and positive
controls (CONTROLS, in memory only). Pure building blocks except for the reading functions. No output of rows, texts or identifiers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import answer_match as AM
import stage2 as S2

SOURCE_ORDER = ("c_vote[pr27+q397]", "b_last[pr27+q397]", "a_major[pr27+q397]", "c_formed[pr27+q397]", "d_minor[pr27+q397]", "k_nov_single[q397]",
                "fac27_rule", "oth_novelty", "psy_adm_sec")
COLUMNS = SOURCE_ORDER[:6]  # the 397B group q of the bundle test
LABEL_FREE = COLUMNS[:5]
VARIANTS = {"q": SOURCE_ORDER[:6], "qf": SOURCE_ORDER[:7], "qfo": SOURCE_ORDER[:8], "qfop": SOURCE_ORDER,
            "qfp": SOURCE_ORDER[:7] + ("psy_adm_sec",)}  # qfp: without oth_novelty (test guard: its task count does not transfer to the test side)
SUBCOLS = {"fac27_rule": ("minor", "ANS-01"), "oth_novelty": ("ent", "max", "specmax", "ch_ntop", "d_ntop", "task_rows")}
SOURCES = ("pr27", "q397")
NOTE_CHARS = 16000  # vlm/proto.rank_notes, tools/export_proto_notes.py, tools/make_q397m3_notes.py
VERSION = "stage2-cols-2"
GENERAL = ("ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00")
ALWAYS_SECTIONS = ("NONE", "CROSS-STRAND", "CATCH-ALL", "NUMBER — Place Value & Operations")  # always admissible (as in the measurement)
UNKNOWN_TOPIC = "\x00unknown"  # control topic_unknown
# note states: name -> label recorded in the frozen note files (as tools/export_proto_notes.RECORDED_STATE and
# tools/make_q397m3_notes.RECORDED_STATE; the 397B notes of the training side store these labels in the field q397_state)
RECORDED_STATE = {"complete": "fertig", "draft": "entwurf", "open": "offen"}
SPOKEN = "spoken"  # verdict spoken in the reasoning (Texts.verdict; in memory only, never recorded in a file)
DEFINITIONS = {
    "c_vote[pr27+q397]": "number of sources pr27, q397 with root_code = code (all votes)",
    "b_last[pr27+q397]": "number of sources whose note (reasoning of the first pass) names the code last",
    "a_major[pr27+q397]": "number of finding lists with the code as major",
    "c_formed[pr27+q397]": "votes (root_code) only of the sources with a formed verdict (complete/draft/spoken)",
    "d_minor[pr27+q397]": "−minor(code) · number of sources whose read final answer matches the key mechanically",
    "k_nov_single[q397]": "top code of the 397B answer × code never gold for the task (question_id|topic_name) among the reference rows (NONE never new)",
    "fac27_rule": "pr27 final_answer_correct (last readable JSON answer) × {default severity minor, 1[candidate = ANS-01]} (2 columns)",
    "oth_novelty": ("× 1[candidate = OTH-00]: entropy, maximum, largest special mass of the mean of the blend channels (before the floor), number of "
                    "distinct channel and draw top codes, number of reference rows of the same task (6 columns)"),
    "psy_adm_sec": ("1 if the section of the candidate was gold within the topic of the row among the reference rows (NONE, CROSS-STRAND, CATCH-ALL, "
                    "place value always 1)"),
}
TRAIN_SPEC = {
    "pr27": {"kind": "own_json", "raw": ["artifacts/runs/pr27_rp/pr27_rp/proto_a/proto_raw.jsonl", "artifacts/runs/pr27_rp/pr27_rp/proto_b/proto_raw.jsonl"],
             "labels_check": ["artifacts/runs/pr27_rp/pr27_rp/proto_a/proto_labels.jsonl", "artifacts/runs/pr27_rp/pr27_rp/proto_b/proto_labels.jsonl"],
             "state": "pr27"},
    "q397": {"kind": "tinker", "labels": "artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_labels.jsonl",
             "notes": "derived/sft_q397m3/q397m3_train_notes.jsonl"},
}
TEST_SPEC = {
    "pr27": {"kind": "own_json", "raw": ["artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl"],
             "labels_check": ["artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_labels.jsonl"], "state": "pr27", "model": "Qwen/Qwen3.8-27B"},
    "q397": {"kind": "own_tinker", "raw": ["artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_raw.jsonl"],
             "labels_check": ["artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_labels.jsonl"], "state": "q397",
             "model": "Qwen/Qwen3.5-397B-A17B-FP8"},
}
KINDS = ("own_json", "own_tinker", "tinker")
GUARD_RULE = {"rule": "test guard per column and source (pre-registered; floor of valid answers added before the first build)", "rel": 0.15,
              "z_max": 3.0, "floors": {"share_valid": 0.97},
              "text": ("Per column (candidate pairs of the base form, first K = 5): share of rows with a value ≠ 0, share of rows in which the column "
                       "supports the top candidate of the base (value of the top = row maximum and row not constant), mean and spread of the "
                       "values; per source: share of valid answers, \"verdict formed\", \"read final answer = key\" (pr27, q397), share "
                       "\"final_answer_correct\" (fac27_rule), share of rows with OTH-00 among the candidates and mean of the six novelty statistics "
                       "(oth_novelty). Test against training (deployment form of the training rows = evaluated OOF features; only the label-free "
                       "task count of oth_novelty in test form: reference all training rows, because the test side counts all 1,500 by "
                       "definition): passes if |test − training| ≤ 0.15·|training| OR |z| ≤ 3 — shares: two-sample binomial test "
                       "(stage2.share_z, as in the widened rule); means: rows as units; spread: delta method over the rows. ADDENDUM (before "
                       "the first build; tightened after the positive control q397_drop10, which lay inside the ±15 % band): valid answers "
                       "per source additionally ≥ 0.97 (= make_submission --proto-max-missing 0.03). If a statistic is violated: no file "
                       "(abort), no automatic dropping of a column.")}
CONTROLS = ("q397_drop10", "q397_unformed", "mech_zero", "q397_rotate", "fac_flip", "chan_rotate", "topic_unknown")
CONTROL_TARGET = {"q397_drop10": "source q397: 10 % of the rows without a 397B answer", "q397_unformed": "source q397: no verdict formed",
                  "mech_zero": "both sources: no read final answer matches", "q397_rotate": "397B building blocks shifted by one row (row order)",
                  "fac_flip": "pr27 final_answer_correct inverted", "chan_rotate": "first blend channel shifted by one row (row order)",
                  "topic_unknown": "topics of the rows unknown (no reference)"}
CONTROL_NEEDS = {"q397_drop10": "c_vote[pr27+q397]", "q397_unformed": "c_formed[pr27+q397]", "mech_zero": "d_minor[pr27+q397]",
                 "q397_rotate": "c_vote[pr27+q397]", "fac_flip": "fac27_rule", "chan_rotate": "oth_novelty", "topic_unknown": "psy_adm_sec"}
CONTROL_SEED = 107


class SourceError(RuntimeError):
    """A source is missing, does not match its checksum or is inconsistent."""


def feature_names(sources: Sequence[str]) -> list[str]:
    """Feature names of the columns (order of the sources; multi-column sources as <source>:<part>)."""
    out = []
    for s in sources:
        if s not in SOURCE_ORDER:
            raise KeyError(f"unknown column source {s!r}")
        out += [f"{s}:{p}" for p in SUBCOLS[s]] if s in SUBCOLS else [s]
    return out


def variant_of(sources: Sequence[str]) -> str | None:
    for v, s in VARIANTS.items():
        if tuple(sources) == tuple(s):
            return v
    return None


# ------------------------------------------------------------------------------------------------ reading and parsing the answers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def json_obj(text: str) -> dict | None:
    """First "{" to last "}" as a JSON object (as vlm/proto.parse_answer and the measurement)."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        o = json.loads(text[i: j + 1])
    except json.JSONDecodeError:
        return None
    return o if isinstance(o, dict) else None


def clean_answer(o: dict | None, valid: set) -> dict | None:
    """Fields of a protocol answer, checked against the class list (exactly as in the measurement)."""
    if o is None:
        return None
    root = o.get("root_code")
    raw_top = o.get("top3") if isinstance(o.get("top3"), list) else []
    top, probs = [], []
    raw_p = o.get("probabilities") if isinstance(o.get("probabilities"), list) else []
    for k, c in enumerate(raw_top):
        if isinstance(c, str) and c in valid and c not in top:
            top.append(c)
            p = raw_p[k] if k < len(raw_p) and len(raw_p) == len(raw_top) else None
            probs.append(float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) and np.isfinite(float(p)) else None)
            if len(top) == 3:
                break
    if not isinstance(root, str) or root not in valid or not top:
        return None
    fnd = []
    for f in o.get("findings") or []:
        if not isinstance(f, dict):
            continue
        code = f.get("code")
        fnd.append({"step": str(f.get("step") or ""), "code": code if isinstance(code, str) and code in valid else None,
                    "severity": str(f.get("severity") or "").strip().lower(), "root": f.get("root") is True,
                    "text": str(f.get("text") or "")})
    tr = o.get("transcription") if isinstance(o.get("transcription"), list) else []
    return {"root_code": root, "top3": top, "probs": probs, "findings": fnd, "transcription": [str(t) for t in tr],
            "student_final_answer": o.get("student_final_answer") if isinstance(o.get("student_final_answer"), str) else None,
            "final_answer_correct": o.get("final_answer_correct") if isinstance(o.get("final_answer_correct"), bool) else None,
            "readable": str(o.get("readable") or "").strip().lower()[:10],
            "confidence": float(o["confidence"]) if isinstance(o.get("confidence"), (int, float)) and not isinstance(o.get("confidence"), bool) else None}


def _is_number(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        float(v)
    except OverflowError:
        return False
    return True


def tinker_parse(text: str, classes: Sequence[str]) -> dict | None:
    """Parser of the Tinker measurement (exactly the fields of tools/open_teacher.parse_answer): so the test side of the 397B carries the same
    fields as its training side (proto_labels.jsonl of the Tinker measurement)."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a: b + 1])
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
    if isinstance(raw_probs, list) and len(raw_probs) == len(raw_top):
        selected = [raw_probs[i] for i in positions]
        if all(_is_number(v) and np.isfinite(float(v)) for v in selected):
            probs = [float(v) for v in selected]
    return {"root_code": root, "top3": top, "probabilities": probs,
            "confidence": float(obj["confidence"]) if _is_number(obj.get("confidence")) else None,
            "final_answer_correct": obj.get("final_answer_correct") if isinstance(obj.get("final_answer_correct"), bool) else None,
            "student_final_answer": str(obj.get("student_final_answer") or "")[:200],
            "readable": str(obj.get("readable") or "").strip().lower()[:10],
            "transcription": [str(x) for x in obj.get("transcription", [])][:60] if isinstance(obj.get("transcription"), list) else [],
            "findings": [f for f in (obj.get("findings") if isinstance(obj.get("findings"), list) else [])
                         if isinstance(f, dict) and isinstance(f.get("code"), str) and f.get("code") in valid][:12]}


def rank_notes(rec: dict, note_chars: int = NOTE_CHARS) -> str:
    """Note of a row = reasoning of the first pass without think markers and surrounding whitespace, the last note_chars characters (exactly
    vlm/proto.rank_notes)."""
    if rec.get("phase") != "first":
        return ""
    return str(rec.get("thinking") or "").replace("<think>", "").replace("</think>", "").strip()[-int(note_chars):]


def note_state(rec: dict, text: str) -> str:
    """State of the note without label (tools/export_proto_notes.note_state = tools/make_q397m3_notes.q397_state for an existing record), as
    its recorded label (RECORDED_STATE)."""
    if str(rec.get("end", rec.get("stop", "")) or "") == "stop":
        return RECORDED_STATE["complete"]
    return RECORDED_STATE["draft"] if "root_code" in text else RECORDED_STATE["open"]


def _empty_rows(n: int) -> list[dict]:
    return [{"ans": None, "first_end": None, "first_valid": False, "notes": "", "state": None, "has_first": False} for _ in range(n)]


def read_own(paths: Sequence[Path], n: int, classes: Sequence[str], parse: str, state_rule: str) -> tuple[list[dict], dict]:
    """Answers of the own computation path (proto_gen, <tag>_raw.jsonl) in file order: per row the last valid answer (a completion replaces
    the first pass), end and validity of the first pass (as in the measurement), note = rank_notes of the first pass. parse "json": JSON
    object of the answer text (pr27); "tinker": parser of the Tinker measurement (q397 test side). state_rule "pr27": state only for a
    non-empty note (as the export); "q397": state for every first pass. At most one first pass per row."""
    valid = set(classes)
    out = _empty_rows(n)
    cnt = {"records": 0, "first": 0, "phases": {}}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                i = int(r["row"])
                if not 0 <= i < n:
                    raise SourceError(f"{Path(p).name}: row position outside 0…{n - 1}")
                cnt["records"] += 1
                ph = str(r.get("phase"))
                cnt["phases"][ph] = cnt["phases"].get(ph, 0) + 1
                text = r.get("text") or ""
                if parse == "json":
                    a = clean_answer(json_obj(text), valid)
                elif parse == "tinker":
                    pa = tinker_parse(text, classes)
                    a = clean_answer({"ok": True, **pa} if pa is not None else None, valid)
                else:
                    raise ValueError(f"parser {parse!r}")
                if r.get("phase") == "first":
                    if out[i]["has_first"]:
                        raise SourceError(f"{Path(p).name}: a row has more than one first pass")
                    cnt["first"] += 1
                    out[i]["has_first"] = True
                    out[i]["first_end"] = r.get("end")
                    out[i]["first_valid"] = a is not None
                    note = rank_notes(r)
                    out[i]["notes"] = note
                    if state_rule == "pr27":
                        out[i]["state"] = note_state(r, note) if note else None
                    elif state_rule == "q397":
                        out[i]["state"] = note_state(r, note)
                    else:
                        raise ValueError(f"state rule {state_rule!r}")
                if a is not None:
                    out[i]["ans"] = a
    return out, cnt


def own_labels_check(raw_paths: Sequence[Path], labels_paths: Sequence[Path], classes: Sequence[str], parse: str) -> dict:
    """Cross-check of the parser against the answer file of the SAME run (proto_labels.jsonl, vlm/proto.parse_answer): per record (row,
    pass) the same validity, the same root_code, the same top3. -> counts (deviations must be 0)."""
    valid = set(classes)
    lab: dict = {}
    for p in labels_paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    lab.setdefault((int(r["row"]), str(r.get("phase"))), []).append(r)
    raw: dict = {}
    for p in raw_paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    raw.setdefault((int(r["row"]), str(r.get("phase"))), []).append(r)
    mism, n = 0, 0
    if set(lab) != set(raw) or any(len(lab[k]) != len(raw[k]) for k in raw):
        return {"records": len(raw), "mismatch": -1, "same_keys": False}
    for k, rr in raw.items():
        for r, lb in zip(rr, lab[k]):
            n += 1
            text = r.get("text") or ""
            if parse == "json":
                a = clean_answer(json_obj(text), valid)
                got = None if a is None else (a["root_code"], a["top3"])
            else:
                pa = tinker_parse(text, classes)
                got = None if pa is None else (pa["root_code"], pa["top3"])
            want = (lb.get("root_code"), lb.get("top3")) if lb.get("ok") else None
            mism += int(got != want)
    return {"records": n, "mismatch": mism, "same_keys": True}


def read_tinker(labels_path: Path, notes_path: Path, n: int, classes: Sequence[str]) -> tuple[list[dict], dict]:
    """397B of the training side (Tinker): parsed fields from proto_labels.jsonl (last valid record per row; end of the first pass = field
    stop) and notes with the state q397_state (as in the measurement)."""
    valid = set(classes)
    out = _empty_rows(n)
    cnt = {"records": 0, "first": 0}
    with open(labels_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["row"])
            if not 0 <= i < n:
                raise SourceError(f"{Path(labels_path).name}: row position outside 0…{n - 1}")
            cnt["records"] += 1
            a = clean_answer(r if r.get("ok") else None, valid)
            if r.get("phase") == "first":
                cnt["first"] += 1
                out[i]["has_first"] = True
                out[i]["first_end"] = r.get("stop")
                out[i]["first_valid"] = a is not None
            if a is not None:
                out[i]["ans"] = a
    with open(notes_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("table", "train") != "train":
                    raise SourceError(f"{Path(notes_path).name}: table {r.get('table')}")
                i = int(r["row"])
                out[i]["notes"] = r.get("text") or ""
                out[i]["state"] = r.get("q397_state") or r.get("state")
    return out, cnt


def fac_rows(paths: Sequence[Path], n: int) -> np.ndarray:
    """pr27 "final_answer_correct" per row from the last readable JSON answer with student_final_answer (answer_match.last_answers, as in
    the measurement): 1.0 for True, otherwise 0.0."""
    answers = AM.last_answers([Path(p) for p in paths], n)
    out = np.zeros(n)
    for i, a in enumerate(answers):
        if a is not None:
            out[i] = 1.0 if a.get("final_answer_correct") is True else 0.0
    return out


# ------------------------------------------------------------------------------------------------ building blocks per source (as in the measurement)

class Texts:
    """Mentions and verdict in the text (exactly as in the measurement: regex of the class list + NONE, draft, spoken verdict)."""

    def __init__(self, classes: Sequence[str]):
        self.classes = list(classes)
        self.K = len(self.classes)
        self.idx = {c: k for k, c in enumerate(self.classes)}
        codes = sorted({c for c in self.classes if c != "NONE"}, key=len, reverse=True)
        self.pat = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(c) for c in codes) + r"|NONE)(?![A-Za-z0-9])")
        alt = "|".join(re.escape(c) for c in codes) + "|NONE"
        self.draft = re.compile(r'"root_code"\s*:\s*"(' + alt + r')"')
        self.spoken = re.compile(r"(?:\broot(?:\s+(?:code|misconception|cause))?\s*(?:is|=|:|would be|should be|seems to be|will be)\s*\**\s*"
                                 r"|\bgo with\s+\**|\bRoot:\s*\**|\bfinal answer:\s*\**|\bchoose\s+\**|\bpick\s+\**)(" + alt + r")(?![A-Za-z0-9])",
                                 re.IGNORECASE)

    def last(self, text: str) -> int | None:
        found = self.pat.findall(text or "")
        return self.idx[found[-1]] if found else None

    def verdict(self, text: str) -> str:
        """Hierarchy: JSON draft > spoken verdict > open (labels RECORDED_STATE["draft"], SPOKEN, RECORDED_STATE["open"])."""
        if self.draft.findall(text or ""):
            return RECORDED_STATE["draft"]
        s = [x for x in self.spoken.findall(text or "") if x in self.idx or x.upper() == "NONE"]
        return SPOKEN if s else RECORDED_STATE["open"]


def parts(rows: list[dict], classes: Sequence[str], keys: Sequence[str]) -> dict:
    """Per source (as in the measurement, only what the columns need): root [n] (−1 without a valid answer), last [n, K], f_major [n, K],
    formed [n], mech [n] (read final answer = key), has_read [n], valid [n]."""
    T = Texts(classes)
    n, K = len(rows), T.K
    last, f_major = np.zeros((n, K), np.int8), np.zeros((n, K), np.int8)
    root = np.full(n, -1)
    formed, has_read, valid = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    mech = np.zeros(n)
    kv = [AM.parse_value(k)[0] for k in keys]
    for i, r in enumerate(rows):
        notes = r.get("notes") or ""
        lc = T.last(notes)
        if lc is not None:
            last[i, lc] = 1
        st = r.get("state")
        if st in (RECORDED_STATE["complete"], RECORDED_STATE["draft"]):
            vs = st
        else:
            vs = T.verdict(notes)
            if r.get("first_end") == "stop" and r.get("first_valid"):
                vs = RECORDED_STATE["complete"]
        formed[i] = vs in (RECORDED_STATE["complete"], RECORDED_STATE["draft"], SPOKEN)
        a = r.get("ans")
        if a is None:
            continue
        valid[i] = True
        root[i] = T.idx[a["root_code"]]
        for f in a["findings"]:
            if f["code"] is None:
                continue
            if f["severity"] == "major":
                f_major[i, T.idx[f["code"]]] = 1
        sfa = a.get("student_final_answer")
        if sfa not in (None, ""):
            has_read[i] = True
            mech[i] = float(AM.mech_match(sfa, keys[i], kv[i]))
    return {"root": root, "last": last, "f_major": f_major, "formed": formed, "mech": mech, "has_read": has_read, "valid": valid}


def onehot(idx: np.ndarray, K: int) -> np.ndarray:
    M = np.zeros((len(idx), K), np.int8)
    ok = idx >= 0
    M[np.flatnonzero(ok), idx[ok]] = 1
    return M


def minor_vector(classes: Sequence[str], tax_severity: dict) -> np.ndarray:
    """1.0 for codes with default severity minor (as in the measurement: NONE never), otherwise 0.0."""
    return np.array([1.0 if c != "NONE" and str(tax_severity.get(c, "")).strip().lower() == "minor" else 0.0 for c in classes])


def label_free(P: dict, minor: np.ndarray) -> dict:
    """The five label-free matrices [n, K] (float64) from the building blocks of the sources pr27 and q397 — computation exactly as in the
    measurement (same bit patterns, also −0.0 in d_minor)."""
    a_, b_ = "pr27", "q397"
    n, K = P[a_]["last"].shape
    V = np.zeros((n, K))
    Vo = np.zeros((n, K))
    for s in (a_, b_):
        oh = onehot(P[s]["root"], K)
        V += oh * P[s]["formed"][:, None]
        Vo += oh * (~P[s]["formed"])[:, None]
    cons = sum(P[s]["mech"] for s in (a_, b_))
    out = {"c_vote[pr27+q397]": V + Vo, "b_last[pr27+q397]": P[a_]["last"] + P[b_]["last"],
           "a_major[pr27+q397]": P[a_]["f_major"] + P[b_]["f_major"], "c_formed[pr27+q397]": V,
           "d_minor[pr27+q397]": -minor[None, :] * cons[:, None]}
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def single_votes(root: np.ndarray, K: int) -> np.ndarray:
    """Vote matrix of the family k "single" (as in the measurement): 1 at the root_code of the source."""
    W = np.zeros((len(root), K))
    ok = root >= 0
    np.add.at(W, (np.flatnonzero(ok), root[ok]), 1.0)
    return W


def novelty_rows(chans: Sequence[np.ndarray], draws: Sequence[np.ndarray], seen: np.ndarray, general: np.ndarray) -> np.ndarray:
    """The five novelty statistics per row (as in the measurement): entropy, maximum and largest special mass (not NONE, not general, with a
    training example) of the mean of the blend channels BEFORE the floor in the order of the blend; number of distinct top codes of the
    channels and of the draws d1–d3 (ranks as stage2.Static: only classes with a training example, cap RANK_CAP). -> [n, 5]."""
    P = np.mean([np.asarray(c, np.float64) for c in chans], 0)
    seen = np.asarray(seen, bool)
    special = np.ones(len(seen), bool)
    special[0] = False
    special[np.asarray(general, bool)] = False
    special &= seen
    ch_t = np.stack([np.minimum(S2.ranks_of(np.asarray(c, np.float64), seen), S2.RANK_CAP).argmin(1) for c in chans], 1)
    d_t = np.stack([np.minimum(S2.ranks_of(np.asarray(d, np.float64), seen), S2.RANK_CAP).argmin(1) for d in draws], 1)
    return np.column_stack([-(P * np.log(np.clip(P, 1e-12, 1))).sum(1), P.max(1), np.where(special[None, :], P, 0).max(1),
                            np.array([len(set(t)) for t in ch_t], float), np.array([len(set(t)) for t in d_t], float)])


# ------------------------------------------------------------------------------------------------ column sources

def _pairs(rows: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R, Kc = cand.shape
    return np.asarray(rows)[np.repeat(np.arange(R), Kc)], cand.reshape(-1)


class FixedColumn:
    """Label-free: fixed matrix V[row, class] of the table of the rows; ref is not read."""

    labels, guard_ref_all = False, False

    def __init__(self, V: np.ndarray):
        self.V = np.asarray(V, np.float64)

    def cols(self, rows, cand, ref=None) -> np.ndarray:
        gi, cc = _pairs(rows, cand)
        return self.V[gi, cc][:, None]


class NovelColumn:
    """k_nov_single: vote W[row, class] × "code never gold for the task among the reference rows" (NONE never new). ref = positions in the
    training table (ref_keys, ref_y); row_keys = keys of the table of the rows."""

    labels, guard_ref_all = True, False

    def __init__(self, row_keys: Sequence[str], W: np.ndarray, ref_keys: Sequence[str], ref_y: np.ndarray):
        self.W = np.asarray(W, np.float64)
        keys = np.asarray(ref_keys).astype(str)
        uniq = {k: j for j, k in enumerate(sorted(set(keys.tolist())))}
        self.ref_kid = np.array([uniq[k] for k in keys], np.int64)
        self.row_kid = np.array([uniq.get(k, -1) for k in np.asarray(row_keys).astype(str)], np.int64)
        self.ref_y = np.asarray(ref_y, np.int64)
        self.nt, self.K = len(uniq), self.W.shape[1]

    def cols(self, rows, cand, ref) -> np.ndarray:
        ref = np.asarray(ref, np.int64)
        cnt = np.zeros((self.nt + 1, self.K))  # last row: task without a training row (never gold, hence new)
        np.add.at(cnt, (self.ref_kid[ref], self.ref_y[ref]), 1.0)
        gi, cc = _pairs(rows, cand)
        nov = (cnt[self.row_kid[gi], cc] == 0) & (cc != 0)
        return (nov * self.W[gi, cc])[:, None]


class FacColumn:
    """fac27_rule: fac[row] × {minor_raw(candidate), 1[candidate = ANS-01]} (as in the measurement; minor_raw = taxonomy field
    default_severity literally "minor")."""

    labels, guard_ref_all = False, False

    def __init__(self, fac: np.ndarray, sev_minor: np.ndarray, ans_idx: int):
        self.fac = np.asarray(fac, np.float64)
        self.attrs = [np.asarray(sev_minor, np.float64), np.eye(len(sev_minor))[int(ans_idx)]]

    def cols(self, rows, cand, ref=None) -> np.ndarray:
        rows = np.asarray(rows, np.int64)
        Rn, Kc = cand.shape
        ii, cc = np.repeat(np.arange(Rn), Kc), np.asarray(cand).reshape(-1)
        Z = self.fac[rows][:, None]
        return np.stack([Z[ii, j] * a[cc] for j in range(Z.shape[1]) for a in self.attrs], 1)


class NoveltyColumn:
    """oth_novelty: [novelty statistics of the row (novelty_rows), number of reference rows of the same task] × 1[candidate = OTH-00] (6
    columns; as in the measurement). The task count reads keys only (label-free)."""

    labels, guard_ref_all = False, True

    def __init__(self, F5: np.ndarray, row_keys: Sequence[str], ref_keys: Sequence[str], oth_idx: int, K: int):
        self.F = np.asarray(F5, np.float64)
        self.row_keys = np.asarray(row_keys).astype(str)
        self.ref_keys = np.asarray(ref_keys).astype(str)
        self.attr = np.eye(K)[int(oth_idx)]

    def row_values(self, rows, ref) -> np.ndarray:
        cnt = Counter(self.ref_keys[np.asarray(ref, np.int64)].tolist())
        tr = np.array([cnt.get(k, 0) for k in self.row_keys[np.asarray(rows, np.int64)]], np.float64)
        return np.column_stack([self.F[np.asarray(rows, np.int64)], tr])

    def cols(self, rows, cand, ref) -> np.ndarray:
        Rn, Kc = cand.shape
        ii, cc = np.repeat(np.arange(Rn), Kc), np.asarray(cand).reshape(-1)
        Z = self.row_values(rows, ref)
        return np.stack([Z[ii, j] * self.attr[cc] for j in range(Z.shape[1])], 1)


class SectionColumn:
    """psy_adm_sec: 1 if the section of the candidate was gold within the topic of the row among the reference rows or is always admissible
    (as in the measurement). Labels only from ref; topics of rows without a training topic have no reference."""

    labels, guard_ref_all = True, False

    def __init__(self, row_topics: Sequence[str], ref_topics: Sequence[str], ref_y: np.ndarray, sections: Sequence[str]):
        ref_topics = np.asarray(ref_topics).astype(str)
        uniq = {t: j for j, t in enumerate(dict.fromkeys(ref_topics.tolist()))}
        self.t_ref = np.array([uniq[t] for t in ref_topics], np.int64)
        self.t_row = np.array([uniq.get(t, len(uniq)) for t in np.asarray(row_topics).astype(str)], np.int64)
        self.nt = len(uniq)
        secs = np.asarray(sections)
        su, self.s_code = np.unique(secs, return_inverse=True)
        self.ns = len(su)
        self.always = np.isin(secs, ALWAYS_SECTIONS)
        self.ref_y = np.asarray(ref_y, np.int64)

    def cols(self, rows, cand, ref) -> np.ndarray:
        ref, rows = np.asarray(ref, np.int64), np.asarray(rows, np.int64)
        TS = np.zeros((self.nt + 1, self.ns), bool)  # last row: topic without a training row
        TS[self.t_ref[ref], self.s_code[self.ref_y[ref]]] = True
        ok = TS[self.t_row[rows]][:, self.s_code]
        ok |= self.always[None, :]
        return ok[np.arange(len(rows))[:, None], np.asarray(cand)].astype(float).reshape(-1, 1)


class ZeroColumn:
    """Copy null point: 0 everywhere (same number of columns as the source)."""

    labels, guard_ref_all = False, False

    def __init__(self, width: int = 1):
        self.width = int(width)

    def cols(self, rows, cand, ref=None) -> np.ndarray:
        return np.zeros((np.asarray(cand).size, self.width))


class Columns:
    """Group of column sources in a fixed order (names = sources; features = feature names at the end of the main model)."""

    def __init__(self, names: Sequence[str], sources: dict):
        self.names = list(names)
        missing = [nm for nm in self.names if nm not in sources]
        if missing:
            raise KeyError(f"column sources missing: {missing}")
        self.sources = {nm: sources[nm] for nm in self.names}
        self.features = feature_names(self.names)

    def cols(self, rows, cand, ref) -> np.ndarray:
        cand = np.asarray(cand)
        out = np.concatenate([np.asarray(self.sources[nm].cols(rows, cand, ref), np.float64).reshape(cand.size, -1) for nm in self.names], 1)
        if out.shape[1] != len(self.features):
            raise ValueError(f"{out.shape[1]} columns instead of {len(self.features)}")
        return out

    def guard_cols(self, rows, cand, ref, ref_all) -> np.ndarray:
        """As cols, but sources with guard_ref_all (label-free task count) with the reference ref_all — test form of the reference of the
        test guard."""
        cand = np.asarray(cand)
        return np.concatenate([np.asarray(self.sources[nm].cols(rows, cand, ref_all if getattr(self.sources[nm], "guard_ref_all", False) else ref),
                                          np.float64).reshape(cand.size, -1) for nm in self.names], 1)

    def zero(self) -> "Columns":
        return Columns(self.names, {nm: ZeroColumn(len(SUBCOLS.get(nm, (nm,)))) for nm in self.names})


def build_columns(names: Sequence[str], inp: dict) -> Columns:
    """Column group of a table. inp: P {pr27, q397: parts} and minor (q columns), row_keys/ref_keys/ref_y (task knowledge, training table as
    reference), fac, sev_minor, classes (fac27_rule), chans/draws/seen (oth_novelty), row_topics/ref_topics/sections (psy_adm_sec)."""
    names = list(names)
    classes = list(inp["classes"])
    K = len(classes)
    src = {}
    if any(nm in names for nm in COLUMNS):
        V = label_free(inp["P"], inp["minor"])
        for nm in LABEL_FREE:
            src[nm] = FixedColumn(V[nm])
        src["k_nov_single[q397]"] = NovelColumn(inp["row_keys"], single_votes(inp["P"]["q397"]["root"], K), inp["ref_keys"], inp["ref_y"])
    if "fac27_rule" in names:
        src["fac27_rule"] = FacColumn(inp["fac"], inp["sev_minor"], classes.index("ANS-01"))
    if "oth_novelty" in names:
        general = np.isin(np.array(classes), GENERAL)
        F5 = novelty_rows(inp["chans"], inp["draws"], inp["seen"], general)
        src["oth_novelty"] = NoveltyColumn(F5, inp["row_keys"], inp["ref_keys"], classes.index("OTH-00"), K)
    if "psy_adm_sec" in names:
        src["psy_adm_sec"] = SectionColumn(inp["row_topics"], inp["ref_topics"], inp["ref_y"], inp["sections"])
    return Columns(names, src)


def class_sections(classes: Sequence[str], tax_ids: Sequence[str], tax_sections: Sequence[str]) -> np.ndarray:
    """Section per class (NONE: "NONE"), as in the measurement."""
    m = dict(zip(tax_ids, tax_sections))
    return np.array(["NONE"] + [m[c] for c in list(classes)[1:]])


def sev_minor_raw(classes: Sequence[str], tax_ids: Sequence[str], tax_sev: Sequence[str]) -> np.ndarray:
    """default_severity literally "minor" per class (as in the measurement: sev == "minor"; NONE never)."""
    m = dict(zip(tax_ids, tax_sev))
    return np.array([0.0] + [1.0 if m[c] == "minor" else 0.0 for c in list(classes)[1:]])


def matrices_sha256(cols: Columns, n: int, K: int) -> dict:
    """Checksum per source over all rows of the table and all classes as candidates (bytes float64; reference = all rows) — pin of the
    training side (a change of a source changes the deployment file)."""
    allc = np.tile(np.arange(K), (n, 1))
    rows = np.arange(n)
    return {nm: hashlib.sha256(np.ascontiguousarray(cols.sources[nm].cols(rows, allc, rows), np.float64).tobytes()).hexdigest() for nm in cols.names}


# ------------------------------------------------------------------------------------------------ loading the sources (training and test side)

def _resolve(rel: str, comp: Path, data_root: Path) -> Path:
    return (Path(data_root) / rel) if str(rel).startswith("derived/") else (Path(comp) / rel)


def _meta_of(raw: Path) -> dict:
    f = raw.parent / "proto_meta.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def _summary_model(raw: Path) -> str | None:
    f = raw.parent / "summary.json"
    return ((json.loads(f.read_text()) if f.is_file() else {}).get("model") or {}).get("repo")


def source_files(spec: dict) -> list[str]:
    """All files of a source specification (for checksums and the offline package): answers, cross-check files, Tinker answers, notes."""
    out = []
    for s in SOURCES:
        d = spec[s]
        out += list(d.get("raw") or []) + list(d.get("labels_check") or [])
        for k in ("labels", "notes"):
            if d.get(k):
                out.append(d[k])
    return out


def sources_digest(comp: Path, data_root: Path, specs: dict | None = None) -> tuple[str, int]:
    """One checksum over all source files of the protocol columns of both sides (path and SHA-256 per file, fixed order; plus the meta and
    summary files of the runs of the own computation path) — pin of the inputs in the build path. -> (SHA-256, number of files)."""
    specs = specs or {"train": TRAIN_SPEC, "test": TEST_SPEC}
    lines = []
    for side in sorted(specs):
        spec = specs[side]
        rels = list(source_files(spec))
        for s in SOURCES:
            for r in spec[s].get("raw") or []:
                rels += [str(Path(r).parent / "proto_meta.json"), str(Path(r).parent / "summary.json")]
        for rel in rels:
            p = _resolve(rel, comp, data_root)
            if not p.is_file():
                raise SourceError(f"{rel} missing")
            lines.append(f"{side}\t{rel}\t{sha256_file(p)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), len(lines)


def load_side(spec: dict, table: str, comp: Path, data_root: Path, classes: Sequence[str], keys: Sequence[str],
              want_sha: dict | None = None, check_labels: bool = True) -> tuple[dict, dict]:
    """Building blocks of both protocol sources of a table. want_sha {file: SHA-256} (deployment file): every file must carry it. Checks per
    run of the own computation path: meta file (table, number of rows), model (summary.json, if given), exactly one first pass per row for
    ALL rows, parser cross-check against proto_labels.jsonl of the same run. -> ({pr27: parts, q397: parts}, details)."""
    n = len(keys)
    info: dict = {"table": table, "rows": n, "files": {}, "sources": {}}
    for rel in source_files(spec):
        p = _resolve(rel, comp, data_root)
        if not p.is_file():
            raise SourceError(f"{rel} missing")
        sha = sha256_file(p)
        if want_sha is not None and want_sha.get(rel) != sha:
            raise SourceError(f"{rel}: SHA-256 {sha[:16]}… instead of {str(want_sha.get(rel))[:16]}… (deployment file)")
        info["files"][rel] = sha
    P = {}
    for s in SOURCES:
        d = spec[s]
        if d.get("kind") not in KINDS:
            raise SourceError(f"{s}: source kind {d.get('kind')!r} unknown")
        if d["kind"] in ("own_json", "own_tinker"):
            raws = [_resolve(r, comp, data_root) for r in d["raw"]]
            rows_meta = 0
            for r in raws:
                m = _meta_of(r)
                if m.get("table") != table:
                    raise SourceError(f"{s}: the meta file next to {r.name} names the table {m.get('table')!r} instead of {table}")
                rows_meta += int(m.get("rows", -1))
                if d.get("model") and _summary_model(r) != d["model"]:
                    raise SourceError(f"{s}: model {_summary_model(r)!r} instead of {d['model']} (summary.json)")
            if rows_meta != n:
                raise SourceError(f"{s}: the runs computed {rows_meta} rows, the table {table} has {n}")
            parse = "json" if d["kind"] == "own_json" else "tinker"
            rows, cnt = read_own(raws, n, classes, parse, d["state"])
            if cnt["first"] != n:
                raise SourceError(f"{s}: {cnt['first']} first passes for {n} rows (run incomplete)")
            ck = None
            if check_labels:
                labs = [_resolve(r, comp, data_root) for r in d.get("labels_check") or []]
                if len(labs) != len(raws):
                    raise SourceError(f"{s}: the cross-check requires one proto_labels.jsonl per answer file (labels_check)")
                ck = own_labels_check(raws, labs, classes, parse)
                if ck["mismatch"] != 0 or not ck["same_keys"]:
                    raise SourceError(f"{s}: parser cross-check against proto_labels.jsonl failed ({ck})")
            info["sources"][s] = {"kind": d["kind"], "records": cnt["records"], "phases": cnt["phases"], "labels_check": ck}
        else:
            rows, cnt = read_tinker(_resolve(d["labels"], comp, data_root), _resolve(d["notes"], comp, data_root), n, classes)
            if cnt["first"] != n:
                raise SourceError(f"{s}: {cnt['first']} first passes for {n} rows")
            info["sources"][s] = {"kind": d["kind"], "records": cnt["records"]}
        P[s] = parts(rows, classes, keys)
    return P, info


def notes_body(rows_notes: dict[int, str], states: dict[int, str], table: str) -> str:
    """File of the pr27 notes as tools/export_proto_notes.py writes it (only for the checksum cross-check; never printed)."""
    return "".join(json.dumps({"table": table, "row": int(i), "state": states[i], "text": rows_notes[i]}, ensure_ascii=False) + "\n"
                   for i in sorted(rows_notes))


# ------------------------------------------------------------------------------------------------ test guard (aggregates only)

def _se_sd(V: np.ndarray) -> tuple[float, float]:
    """Spread of the values over all pairs (ddof 0) and its standard error (delta method, rows as clusters)."""
    R = V.shape[0]
    a, b = (V * V).mean(1), V.mean(1)
    B = float(b.mean())
    sd = float(V.std())
    if R < 2 or sd == 0.0:
        return sd, 0.0
    S = np.cov(np.stack([a, b]), ddof=1)
    g = np.array([1.0, -2.0 * B])
    return sd, math.sqrt(max(float(g @ S @ g) / R, 0.0)) / (2.0 * sd)


def column_stats(E: np.ndarray, cand: np.ndarray, names: Sequence[str]) -> dict:
    """Statistics per column on the candidate pairs (E [R·Kc, m], cand [R, Kc]; base order: cand[:, 0] = top of the base)."""
    cand = np.asarray(cand)
    R, Kc = cand.shape
    out = {}
    for j, nm in enumerate(names):
        V = np.asarray(E[:, j], np.float64).reshape(R, Kc)
        mx, mn = V.max(1), V.min(1)
        rm = V.mean(1)
        sd, se_sd = _se_sd(V)
        out[nm] = {"share_rows_nonzero": float((V != 0).any(1).mean()), "share_rows_support_top": float(((V[:, 0] >= mx) & (mx > mn)).mean()),
                   "mean": float(V.mean()), "se_mean": float(rm.std(ddof=1) / math.sqrt(R)) if R > 1 else 0.0, "sd": sd, "se_sd": se_sd}
    return {"rows": int(R), "columns": out}


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, np.float64)
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0)


NOVELTY_PARTS = SUBCOLS["oth_novelty"]


def source_stats(cols: Columns, inp: dict, cand: np.ndarray, rows: np.ndarray, ref_all: np.ndarray) -> dict:
    """Per source (aggregates only): pr27/q397 share valid, "verdict formed", "final answer = key"; fac27 share final_answer_correct; oth
    share of the rows with OTH-00 among the candidates and mean of the six novelty statistics (task count with the reference ref_all)."""
    out, se = {}, {}
    if any(nm in cols.names for nm in COLUMNS):
        for s in SOURCES:
            P = inp["P"][s]
            out[f"source {s}"] = {"share_valid": float(np.mean(P["valid"])), "share_formed": float(np.mean(P["formed"])),
                                  "share_mech": float(np.mean(P["mech"]))}
    if "fac27_rule" in cols.names:
        out["fac27_rule"] = {"share_fac": float(np.mean(np.asarray(inp["fac"])[rows]))}
    if "oth_novelty" in cols.names:
        src = cols.sources["oth_novelty"]
        Z = src.row_values(rows, ref_all)
        oth = list(inp["classes"]).index("OTH-00")
        d, e = {"share_rows_oth_candidate": float((np.asarray(cand) == oth).any(1).mean())}, {}
        for j, p in enumerate(NOVELTY_PARTS):
            d[f"mean_{p}"], e[f"mean_{p}"] = _mean_se(Z[:, j])
        out["oth_novelty"], se["oth_novelty"] = d, e
    return {"values": out, "se": se}


def guard_stats(cols: Columns, E: np.ndarray, cand: np.ndarray, inp: dict, rows: np.ndarray, ref_all: np.ndarray) -> dict:
    """Statistics of the test guard of one table: columns (E as passed) and sources."""
    st = column_stats(E, cand, cols.features)
    src = source_stats(cols, inp, cand, rows, ref_all)
    return {"rows": int(len(rows)), "columns": st["columns"], "sources": src["values"], "sources_se": src["se"]}


def _z_mean(r: float, se_r: float, t: float, se_t: float) -> float:
    se = math.sqrt(se_r * se_r + se_t * se_t)
    d = t - r
    if se == 0.0:
        return 0.0 if d == 0.0 else math.copysign(math.inf, d)
    return d / se


def guard_check(ref: dict, test: dict, rel: float = GUARD_RULE["rel"], z_max: float = GUARD_RULE["z_max"],
                floors: dict | None = None) -> dict:
    """Test guard (rule GUARD_RULE): passes inside the band |test − training| ≤ rel·|training| OR |z| ≤ z_max; floors in addition.
    -> {checks, ok, failed}. Aggregates only."""
    floors = GUARD_RULE["floors"] if floors is None else floors
    n_ref, n_test = int(ref["rows"]), int(test["rows"])
    checks = []

    def add(scope: str, stat: str, r: float, t: float, z: float) -> None:
        ok_band = bool(abs(t - r) <= rel * abs(r)) if r != 0.0 else bool(t == 0.0)
        ok_z = bool(abs(z) <= z_max)
        fl = floors.get(stat)
        ok_floor = None if fl is None else bool(t >= fl)
        ok = bool((ok_band or ok_z) and ok_floor is not False)
        checks.append({"scope": scope, "stat": stat, "train": r, "test": t, "z": z, "ok_band": ok_band, "ok_z": ok_z, "floor": fl,
                       "ok_floor": ok_floor, "ok": ok, "passed_by": ("band" if ok_band else ("z" if ok_z else None)) if ok else None})

    for nm, rc in ref["columns"].items():
        tc = test["columns"].get(nm)
        if tc is None:
            raise SourceError(f"test guard: column {nm} missing on the test side")
        for st in ("share_rows_nonzero", "share_rows_support_top"):
            add(nm, st, rc[st], tc[st], S2.share_z(rc[st], n_ref, tc[st], n_test))
        add(nm, "mean", rc["mean"], tc["mean"], _z_mean(rc["mean"], rc["se_mean"], tc["mean"], tc["se_mean"]))
        add(nm, "sd", rc["sd"], tc["sd"], _z_mean(rc["sd"], rc["se_sd"], tc["sd"], tc["se_sd"]))
    for scope, rs in ref["sources"].items():
        ts = test["sources"].get(scope)
        if ts is None:
            raise SourceError(f"test guard: source {scope} missing on the test side")
        for st, rv in rs.items():
            if st.startswith("mean_"):
                z = _z_mean(rv, ref["sources_se"][scope][st], ts[st], test["sources_se"][scope][st])
            else:
                z = S2.share_z(rv, n_ref, ts[st], n_test)
            add(scope, st, rv, ts[st], z)
    failed = [c for c in checks if not c["ok"]]
    return {"rule": GUARD_RULE["rule"], "rel": rel, "z_max": z_max, "floors": floors, "n_ref": n_ref, "n_test": n_test, "checks": checks,
            "ok": not failed, "failed": [{k: c[k] for k in ("scope", "stat", "train", "test", "z", "floor")} for c in failed]}


# ------------------------------------------------------------------------------------------------ positive controls (in memory only)

def apply_control(inp: dict, control: str) -> dict:
    """Planted disturbance of the inputs of the test side (in memory only; the test guard MUST fire). Returns a copy."""
    if control not in CONTROLS:
        raise ValueError(f"control {control!r}")
    out = dict(inp)
    if control in ("q397_drop10", "q397_unformed", "mech_zero", "q397_rotate"):
        Q = {s: {k: np.array(v, copy=True) for k, v in inp["P"][s].items()} for s in SOURCES}
        q = Q["q397"]
        n = len(q["root"])
        if control == "q397_drop10":
            idx = np.random.default_rng(CONTROL_SEED).choice(n, size=int(round(0.10 * n)), replace=False)
            q["root"][idx] = -1
            q["f_major"][idx] = 0
            q["mech"][idx] = 0.0
            q["has_read"][idx] = False
            q["valid"][idx] = False
        elif control == "q397_unformed":
            q["formed"][:] = False
        elif control == "mech_zero":
            for s in SOURCES:
                Q[s]["mech"][:] = 0.0
        else:
            for k in q:
                q[k] = np.roll(q[k], 1, axis=0)
        out["P"] = Q
    elif control == "fac_flip":
        out["fac"] = 1.0 - np.asarray(inp["fac"], np.float64)
    elif control == "chan_rotate":
        out["chans"] = [np.roll(np.asarray(inp["chans"][0]), 1, axis=0)] + list(inp["chans"][1:])
    elif control == "topic_unknown":
        out["row_topics"] = np.full(len(inp["row_topics"]), UNKNOWN_TOPIC)
    return out
