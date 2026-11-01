"""Subcommand `continue` of tools/open_teacher.py — without network, with a rebuilt template and a rebuilt sampler.

Checked: rows are selected only at the token limit without answer text; notes in the finish step are complete (first pass +
continuation, without the 16,000-character cut); output folder ≠ source; the budget check aborts or admits only rows that
can safely be paid for; table train only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import open_teacher as OT
from vlm import proto as PR

CLASSES = ["NONE", "FRA-01", "CAL-01", "ALG-06"]
VALID = '{"root_code": "FRA-01", "top3": ["FRA-01", "CAL-01", "NONE"], "probabilities": [0.6, 0.2, 0.1], "confidence": 0.6}'
ROW = {"grade": "7", "topic_name": "Fractions", "question_image_file": "", "correct_answer": "3/4"}


def first(row: int, text: str, stop: str = "length", thinking: str = "", **kw) -> dict:
    """First pass in the format of raw.jsonl (Tinker: at the limit the raw text is under text)."""
    return {"row": row, "phase": "first", "thinking": thinking, "text": text, "stop": stop, "in_tokens": 4000,
            "cache_tokens": 3000, "out_tokens": 2500, **kw}


# ------------------------------------------------------------------------------------------------ pure logic

def test_eligibility_only_length_without_answer():
    assert OT.eligible_first(first(0, "still thinking"))  # Tinker form: raw text, no end of the thinking block
    assert OT.eligible_first(first(0, "", thinking="all thinking"))  # form of the in-house compute path
    assert not OT.eligible_first(first(0, "<think>\ndone thinking\n</think>\n\n{\"root_code\": \"FRA"))  # JSON cut off
    assert not OT.eligible_first(first(0, VALID, stop="stop", thinking="because"))  # finished regularly
    assert not OT.eligible_first(first(0, "{\"root", thinking="because"))  # answer text present
    assert not OT.eligible_first(dict(first(0, "x"), phase="finish"))
    assert OT.tinker_split(first(0, VALID, stop="stop")) == ("", VALID)  # finished without a thinking part: stays an answer
    assert OT.tinker_split(first(0, "<think>\nbecause\n</think>\n\n{x}")) == ("because", "\n\n{x}")
    firsts = {0: first(0, "a"), 1: first(1, "<think>\nb\n</think>\n\n{"), 2: first(2, VALID, stop="stop", thinking="c"), 3: first(3, "d")}
    plan = OT.continue_plan(firsts, [3, 0, 1, 2, 5], [], [])
    assert plan["todo"] == [(3, "continue"), (0, "continue")]  # order of the list
    assert plan["counts"] == {"rows": 5, "missing_first": 1, "not_eligible": 2, "done": 0, "resume_finish": 0}
    lab = [{"row": 3, "phase": "continue", "ok": True}, {"row": 0, "phase": "continue", "ok": False}]
    raw = [{"row": 3, "phase": "continue", "thinking": "t", "text": VALID}, {"row": 0, "phase": "continue", "thinking": "u", "text": ""}]
    plan = OT.continue_plan(firsts, [3, 0, 1, 2, 5], lab, raw)
    assert plan["todo"] == [(0, "finish")] and plan["counts"]["done"] == 1 and plan["cont_raw"][0]["thinking"] == "u"
    with pytest.raises(SystemExit):  # labels and raw do not match
        OT.continue_plan(firsts, [3, 0], lab, raw[:1])


def test_notes_are_complete_and_uncut_below_hard_limit():
    f, c = first(0, "A" * 20000), {"thinking": "B" * 15000, "text": "", "stop": "length"}
    notes, cut = OT.finish_notes(f, c, 40000)
    assert notes == "A" * 20000 + "\n\n" + "B" * 15000 and not cut  # longer than the old limit of 16,000, nothing is missing
    notes, cut = OT.finish_notes(f, c, 1000)
    assert cut and notes == ("A" * 20000 + "\n\n" + "B" * 15000)[-1000:]
    assert OT.record_notes(first(0, "  raw  ")) == "raw" == PR.full_notes({"thinking": "raw", "text": ""})


def test_continue_text_is_imported_not_copied():
    assert OT.PR.CONTINUE_TEXT is PR.CONTINUE_TEXT
    src = (Path(OT.__file__)).read_text()
    assert "Your earlier analysis was interrupted" not in src


def test_output_dir_differs_from_source(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, "OUT", tmp_path)
    for tag, src in (("", ""), ("x", "x"), ("../x", ""), ("a/b", ""), (" ", "")):
        with pytest.raises(SystemExit):
            OT.continue_dirs("qwen35-397b", src, tag)
    s, o = OT.continue_dirs("qwen35-397b", "", "c5k")
    assert s == tmp_path / "qwen35-397b" and o == tmp_path / "qwen35-397b_c5k"
    s, o = OT.continue_dirs("qwen35-397b", "smoke", "c5k")
    assert s.name == "qwen35-397b_smoke" and o.name == "qwen35-397b_c5k"


def test_budget_check_and_gate():
    assert OT.budget_check([1.0, 1.0], 2.0, False) == 2
    with pytest.raises(SystemExit):
        OT.budget_check([1.0, 1.0, 1.0], 2.5, False)
    assert OT.budget_check([1.0, 1.0, 1.0], 2.5, True) == 2
    with pytest.raises(SystemExit):
        OT.budget_check([3.0, 1.0], 2.5, True)  # not even the first row
    g = OT.BudgetGate(2.0)
    assert g.fits(1.5, 0.5) and not g.fits(1.6, 0.5)
    g.reserve(1.0)
    assert g.fits(0.5, 0.5) and not g.fits(0.6, 0.5)
    g.release(5.0)
    assert g.reserved == 0.0


def test_only_train(tmp_path):
    with pytest.raises(SystemExit):
        OT.check_train_record({"row": 0, "table": "test"}, 10)
    for bad in (-1, 10, True, "3"):
        with pytest.raises(SystemExit):
            OT.check_train_record({"row": bad}, 10)
    OT.check_train_record({"row": 9, "table": "train"}, 10)
    p = tmp_path / "rows.npy"
    np.save(p, np.array([0, 12]))
    with pytest.raises(SystemExit):
        OT.load_rows(str(p), 10)
    np.save(p, np.array([1, 1]))
    with pytest.raises(SystemExit):
        OT.load_rows(str(p), 10)
    d = tmp_path / "competition_data" / "images" / "test"
    d.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (8, 8)).save(d / "t.jpeg")
    with pytest.raises(SystemExit):
        OT.load_image(d / "t.jpeg", 1024)


# ------------------------------------------------------------------------------------------------ flow with rebuilt components

class FakeRenderer:
    """Template without a tokenizer: prompt length from the text length; answers via a table token -> content."""

    def __init__(self, store: dict):
        self.store, self.built = store, []

    def build_generation_prompt(self, msgs, **kw):
        self.built.append(msgs)
        text = "".join(p.get("text", "") for p in msgs[1]["content"] if p["type"] == "text")
        return SimpleNamespace(length=len(text) // 4 + 1000, msgs=msgs)

    def parse_response(self, toks):
        return {"role": "assistant", "content": self.store[toks[0]]}, None

    def get_stop_sequences(self):
        return []


class FakeSampler:
    """Answer per (question text, step): the step is recognised by the appended part (CONTINUE_TEXT or FINISH_TEXT)."""

    def __init__(self, store: dict, script: dict):
        self.store, self.script, self.calls = store, script, []

    async def sample_async(self, prompt, num_samples, sampling_params):
        parts = prompt.msgs[1]["content"]
        q = parts[0]["text"].split("Question:\n", 1)[1].strip()
        tail = parts[-1]["text"]
        step = "continue" if tail.startswith(PR.CONTINUE_TEXT.split("{notes}")[0]) else "finish"
        content, stop, n_out = self.script[(q, step)]
        self.calls.append((q, step, tail))
        key = len(self.store)
        self.store[key] = content
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[key] * n_out, stop_reason=stop)], prompt_cache_hit_tokens=0)


def setup_env(tmp_path, monkeypatch, firsts: list[dict], script: dict, cap: float = 1000.0, rows=None):
    from PIL import Image

    monkeypatch.setattr(OT, "OUT", tmp_path / "open_teacher")
    monkeypatch.setattr(OT, "LEDGER", tmp_path / "ledger.json")
    (tmp_path / "ledger.json").write_text(json.dumps({"cap_usd": cap, "entries": []}))
    src = tmp_path / "open_teacher" / "qwen35-397b"
    src.mkdir(parents=True)
    (src / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in firsts))
    d = tmp_path / "data" / "competition_data" / "images" / "train"
    d.mkdir(parents=True)
    train = []
    for k in range(4):
        Image.new("RGB", (40, 30), (k * 40, 0, 0)).save(d / f"r{k}.jpeg")
        train.append(dict(ROW, trace_id=f"r{k}", image_file=f"r{k}.jpeg", question_text=f"Task {k}"))
    rows_file = tmp_path / "rows.npy"
    np.save(rows_file, np.array(rows if rows is not None else [0, 1, 2], dtype=np.int64))
    store: dict = {}
    renderer, frenderer = FakeRenderer(store), FakeRenderer(store)
    env = {"root": tmp_path / "data", "train": pd.DataFrame(train), "classes": CLASSES, "system": "SYSTEM",
           "renderer": renderer, "finish_renderer": frenderer, "sampler": FakeSampler(store, script),
           "make_params": lambda max_tokens, rend: {"max_tokens": max_tokens}}
    args = argparse.Namespace(model="qwen35-397b", rows=str(rows_file), source_tag="", tag="c5k", max_tokens=5000, finish_tokens=1500,
                              note_chars=40000, temperature=0.0, effort=0.9, workers=2, limit=0, allow_partial=False, dry_run=False,
                              guard_rows=24, guard_min_valid=0.5, note="test")
    return env, args, src


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()] if path.is_file() else []


def test_flow_full_notes_in_finish_new_dir_and_resume(tmp_path, monkeypatch):
    firsts = [first(0, "A" * 20000), first(1, "B" * 20000), first(2, VALID, stop="stop", thinking="done")]
    script = {("Task 0", "continue"): ([{"type": "thinking", "thinking": "more"}, {"type": "text", "text": VALID}], "stop", 900),
              ("Task 1", "continue"): ("C" * 15000, "length", 5000),
              ("Task 1", "finish"): ([{"type": "text", "text": VALID}], "stop", 300)}
    env, args, src = setup_env(tmp_path, monkeypatch, firsts, script)
    before = hashlib.sha256((src / "raw.jsonl").read_bytes()).hexdigest()
    assert asyncio.run(OT.continue_async(args, env)) == 0
    out = OT.OUT / "qwen35-397b_c5k"
    lab, raw = read(out / "labels.jsonl"), read(out / "raw.jsonl")
    assert [(r["row"], r["phase"], r["ok"]) for r in sorted(lab, key=lambda r: (r["row"], r["phase"]))] == [
        (0, "continue", True), (1, "continue", False), (1, "finish", True)]  # row 2 (done) skipped
    assert all(r["table"] == "train" for r in lab + raw)
    fin = next(r for r in lab if r["phase"] == "finish")
    assert fin["notes_input_chars"] == 20000 + 2 + 15000 and fin["notes_cut"] is False and fin["reasoning_budget_exhausted"] is True
    assert fin["root_code"] == "FRA-01" and fin["top3"] == ["FRA-01", "CAL-01", "NONE"]
    calls = env["sampler"].calls
    tail_fin = next(t for q, s, t in calls if s == "finish")
    assert tail_fin == OT.FINISH_TEXT.format(notes="B" * 20000 + "\n\n" + "C" * 15000)  # all notes, nothing cut
    tail_c = next(t for q, s, t in calls if q == "Task 1" and s == "continue")
    assert tail_c == PR.CONTINUE_TEXT.format(notes="B" * 20000)
    # message of the continuation = first pass + appended part (same parts before it)
    m0 = OT.messages_for(env["root"], env["train"].iloc[1], "SYSTEM", "")
    m1 = next(m for m in env["renderer"].built if m[1]["content"][0]["text"].endswith("Task 1\n"))
    assert [p["type"] for p in m1[1]["content"]] == [p["type"] for p in m0[1]["content"]] + ["text"]
    assert hashlib.sha256((src / "raw.jsonl").read_bytes()).hexdigest() == before  # source untouched
    assert len(json.loads(OT.LEDGER.read_text())["entries"]) == 3  # every request booked
    assert json.loads((out / OT.RECIPE_FILE).read_text())["recipe"]["continuation_prompt"] == PR.CONTINUE_TEXT
    # resume: nothing left to do
    env["sampler"].calls.clear()
    asyncio.run(OT.continue_async(args, env))
    assert env["sampler"].calls == []
    # aborted after the continuation: only the finish step is redone, with the saved notes
    for f in ("labels.jsonl", "raw.jsonl"):
        recs = read(out / f)
        (out / f).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs if r["phase"] != "finish"))
    asyncio.run(OT.continue_async(args, env))
    assert [(q, s) for q, s, _ in env["sampler"].calls] == [("Task 1", "finish")]
    assert env["sampler"].calls[0][2] == tail_fin
    # a different recipe in the same folder: abort
    args.max_tokens = 4000
    with pytest.raises(SystemExit):
        asyncio.run(OT.continue_async(args, env))


def test_flow_refuses_foreign_output_dir_and_test_rows(tmp_path, monkeypatch):
    env, args, src = setup_env(tmp_path, monkeypatch, [first(0, "a"), first(1, "b")], {})
    foreign = OT.OUT / "qwen35-397b_c5k"
    foreign.mkdir(parents=True)
    (foreign / "labels.jsonl").write_text(json.dumps({"row": 0, "phase": "first", "ok": False}) + "\n")
    with pytest.raises(SystemExit):
        asyncio.run(OT.continue_async(args, env))
    (foreign / "labels.jsonl").unlink()
    (src / "raw.jsonl").write_text(json.dumps(dict(first(0, "a"), table="test")) + "\n")
    with pytest.raises(SystemExit):
        asyncio.run(OT.continue_async(args, env))


def test_flow_budget_abort_and_partial(tmp_path, monkeypatch):
    firsts = [first(0, "A" * 9000), first(1, "B" * 9000)]
    u = sum(OT.upper_usd("qwen35-397b", firsts[0], 5000, 1500))
    expensive = {("Task 0", "continue"): ("C" * 15000, "length", 5000), ("Task 0", "finish"): ([{"type": "text", "text": VALID}], "stop", 300),
                 ("Task 1", "continue"): ("C" * 15000, "length", 5000), ("Task 1", "finish"): ([{"type": "text", "text": VALID}], "stop", 300)}
    env, args, _ = setup_env(tmp_path, monkeypatch, firsts, expensive, cap=1.5 * u, rows=[0, 1])
    with pytest.raises(SystemExit):  # upper estimate 2u > remainder 1.5u: abort before any call
        asyncio.run(OT.continue_async(args, env))
    assert env["sampler"].calls == [] and json.loads(OT.LEDGER.read_text())["entries"] == []
    args.allow_partial = True
    asyncio.run(OT.continue_async(args, env))
    out = OT.OUT / "qwen35-397b_c5k"
    assert sorted({r["row"] for r in read(out / "labels.jsonl")}) == [0]  # row 1 no longer fits after the expensive row 0
    led = json.loads(OT.LEDGER.read_text())
    assert OT.spent(led) <= 1.5 * u + 1e-12


def test_flow_partial_admits_more_when_cheap(tmp_path, monkeypatch):
    firsts = [first(0, "A" * 9000), first(1, "B" * 9000)]
    u = sum(OT.upper_usd("qwen35-397b", firsts[0], 5000, 1500))
    cheap = {("Task 0", "continue"): ([{"type": "thinking", "thinking": "w"}, {"type": "text", "text": VALID}], "stop", 10),
             ("Task 1", "continue"): ([{"type": "thinking", "thinking": "w"}, {"type": "text", "text": VALID}], "stop", 10)}
    env, args, _ = setup_env(tmp_path, monkeypatch, firsts, cheap, cap=1.5 * u, rows=[0, 1])
    args.allow_partial = True
    asyncio.run(OT.continue_async(args, env))
    lab = read(OT.OUT / "qwen35-397b_c5k" / "labels.jsonl")
    assert sorted(r["row"] for r in lab) == [0, 1] and all(r["ok"] for r in lab)  # reservation released: row 1 moves up
    assert OT.spent(json.loads(OT.LEDGER.read_text())) <= 1.5 * u + 1e-12


# ------------------------------------------------------------------------------------------------ cost check

def test_parse_answer_unhashable_finding_code_does_not_raise():
    t = '{"root_code": "FRA-01", "top3": ["FRA-01"], "findings": [{"code": ["CAL-01"]}, {"code": "CAL-01"}, "x"], "transcription": "abc"}'
    p = OT.parse_answer(t, CLASSES)
    assert p is not None and p["root_code"] == "FRA-01" and p["findings"] == [{"code": "CAL-01"}] and p["transcription"] == []
    assert OT.parse_answer('{"root_code": "FRA-01", "top3": ["FRA-01"], "findings": {"code": "CAL-01"}}', CLASSES)["findings"] == []


def test_ledger_missing_aborts_before_any_call(tmp_path, monkeypatch):
    env, args, _ = setup_env(tmp_path, monkeypatch, [first(0, "a")], {}, rows=[0])
    (tmp_path / "ledger.json").unlink()
    with pytest.raises(SystemExit):
        asyncio.run(OT.continue_async(args, env))
    assert env["sampler"].calls == [] and not (tmp_path / "ledger.json").exists()


def test_second_process_is_refused(tmp_path, monkeypatch):
    import fcntl

    env, args, _ = setup_env(tmp_path, monkeypatch, [first(0, "a")], {}, rows=[0])
    with (OT.OUT / ".continue.lock").open("a") as fh:  # another process holds the lock (flock applies per opened file)
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit):
            asyncio.run(OT.continue_async(args, env))
    assert env["sampler"].calls == []


class ScriptedSampler:
    """One behaviour per request: 'raise' (failed, cost unknown), 'badparse' (answer present, parsing fails), otherwise
    (content, stop, n_out). Records the prompt lengths of failed requests."""

    def __init__(self, store: dict, behave):
        self.store, self.behave, self.calls, self.failed_lengths = store, behave, [], []

    async def sample_async(self, prompt, num_samples, sampling_params):
        await asyncio.sleep(0)
        parts = prompt.msgs[1]["content"]
        step = "continue" if parts[-1]["text"].startswith(PR.CONTINUE_TEXT.split("{notes}")[0]) else "finish"
        b = self.behave(len(self.calls), step, sampling_params["max_tokens"])
        self.calls.append(step)
        if b == "raise":
            self.failed_lengths.append((prompt.length, sampling_params["max_tokens"]))
            raise TimeoutError("simulated")
        key = len(self.store)
        if b == "badparse":
            self.store[key] = None  # FakeRenderer.parse_response returns content=None -> split_message raises
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[key] * 7, stop_reason="stop")], prompt_cache_hit_tokens=0)
        content, stop, n_out = b
        self.store[key] = content
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[key] * n_out, stop_reason=stop)], prompt_cache_hit_tokens=0)


def test_failed_and_unparsable_requests_are_handled_conservatively(tmp_path, monkeypatch):
    firsts = [first(k, "A" * 9000) for k in range(4)]
    env, args, _ = setup_env(tmp_path, monkeypatch, firsts, {}, rows=[0, 1, 2, 3])

    def behave(n, step, max_tokens):
        if n == 0:
            return "badparse"  # continuation booked, not parsable -> invalid -> finish step
        if n == 2:  # call 1 is the finish step of the first row, call 2 the continuation of the second
            return "raise"
        if step == "continue":
            return ([{"type": "thinking", "thinking": "w"}, {"type": "text", "text": "no json"}], "stop", 50)
        return ([{"type": "text", "text": VALID}], "stop", 40)

    env["sampler"] = ScriptedSampler(env["sampler"].store, behave)
    args.workers = 1
    assert asyncio.run(OT.continue_async(args, env)) == 0  # no crash
    out = OT.OUT / "qwen35-397b_c5k"
    lab = read(out / "labels.jsonl")
    led = json.loads(OT.LEDGER.read_text())["entries"]
    assert len(led) == len(env["sampler"].calls) - 1  # every answered request booked, including the unparsable one
    bad = next(r for r in lab if r.get("parse_error"))
    assert bad["phase"] == "continue" and bad["ok"] is False
    assert any(r["row"] == bad["row"] and r["phase"] == "finish" and r["ok"] for r in lab)
    assert len({r["row"] for r in lab}) == 3  # one row failed (nothing written), stays open
    # resume catches up on the failed row
    env["sampler"].behave = lambda n, step, mt: ([{"type": "text", "text": VALID}], "stop", 40) if step == "finish" else (
        [{"type": "thinking", "thinking": "w"}, {"type": "text", "text": VALID}], "stop", 30)
    asyncio.run(OT.continue_async(args, env))
    assert {r["row"] for r in read(out / "labels.jsonl")} == {0, 1, 2, 3}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cap_holds_under_random_load(tmp_path, monkeypatch, seed):
    """8 workers, a tight limit, random lengths, randomly failed and unparsable requests: booked + at most the computed failed
    requests stay below the limit; the source stays untouched."""
    rng = np.random.default_rng(seed)
    n = 40
    firsts = [first(k, "A" * int(rng.integers(2000, 11000)), out_tokens=2500) for k in range(n)]
    u = sum(OT.upper_usd("qwen35-397b", firsts[0], 5000, 1500))
    cap = float(rng.uniform(4, 12)) * u
    env, args, src = setup_env(tmp_path, monkeypatch, firsts, {}, cap=cap, rows=list(rng.permutation(n)))
    env["train"] = pd.DataFrame([dict(ROW, trace_id=f"r{k}", image_file=f"r{k % 4}.jpeg", question_text=f"Task {k}") for k in range(n)])
    before = hashlib.sha256((src / "raw.jsonl").read_bytes()).hexdigest()

    def behave(_n, step, max_tokens):
        x = rng.random()
        if x < 0.05:
            return "raise"
        if x < 0.08:
            return "badparse"
        if step == "continue":
            ok = rng.random() < 0.4
            return ([{"type": "thinking", "thinking": "w"}, {"type": "text", "text": VALID if ok else ""}], "stop" if ok else "length",
                    int(rng.integers(1, max_tokens + 1)))
        return ([{"type": "text", "text": VALID}], "stop", int(rng.integers(1, max_tokens + 1)))

    env["sampler"] = ScriptedSampler(env["sampler"].store, behave)
    args.workers, args.allow_partial, args.guard_rows, args.max_failed = 8, True, 10_000, 10_000
    asyncio.run(OT.continue_async(args, env))
    spent = OT.spent(json.loads(OT.LEDGER.read_text()))
    lost = sum(OT.cost("qwen35-397b", L, 0, mt) for L, mt in env["sampler"].failed_lengths)
    assert spent + lost <= cap + 1e-6
    assert hashlib.sha256((src / "raw.jsonl").read_bytes()).hexdigest() == before
