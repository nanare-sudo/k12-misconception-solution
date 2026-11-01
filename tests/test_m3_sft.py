"""Tests for M3, task proto_sft (vlm/sft.py): training target, joint tokenisation, scoring in sft mode, generation,
merging, combining. Without a model and without competition data; the tokenizer of the smallest model is used only if
it is available locally."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vlm import prompts as P  # noqa: E402
from vlm import proto as PR  # noqa: E402
from vlm import sft as S  # noqa: E402

COMP = Path(__file__).resolve().parents[1]
TOK_DIR = COMP / "data" / "models" / "Qwen3.5-0.8B"
ROW = {"trace_id": "t1", "question_id": "q1", "grade": "7", "topic_name": "Fractions", "question_text": "1/2 + 1/3 = ...",
       "question_image_file": "", "correct_answer": "5/6", "root_code": "FRA-01", "final_answer_correct": "False",
       "image_file": "a.jpeg", "transcription_steps": "L1 | 1/2 + 1/3\nL2 | = 2/5",
       "misconceptions": json.dumps([{"step": "L1", "category": "CAL-01", "severity": "minor", "root": False, "text": "by the way"},
                                     {"step": "L2", "category": "FRA-01", "severity": "major", "root": True, "text": "Added  tops\nand bottoms."}])}
TAX = pd.DataFrame({"id": ["FRA-01", "CAL-01"], "name": ["Adding tops and bottoms", "Calculation slip"], "description": ["d1", "d2"]})


@pytest.fixture(scope="module")
def tok():
    if not (TOK_DIR / "tokenizer.json").is_file():
        pytest.skip("tokenizer of the smallest model is not available locally")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOK_DIR))


def test_system_text_is_protocol_without_json_and_without_labels():
    system = S.system_text(TAX)
    assert system.startswith(PR.INSTRUCTIONS.split(PR.JSON_PART)[0]) and "JSON" not in system
    assert "`Finding:" in system and system.endswith(P.NONE_LINE) and "FRA-01 | Adding tops and bottoms | d1" in system
    assert S.system_text(TAX, "name") != system  # the taxonomy style enters the checksum of the recipe
    with pytest.raises(ValueError, match="no question knowledge"):
        S.refuse_task_knowledge({"topic_codes": 10})
    S.refuse_task_knowledge({"taxonomy": "full", "image": {}})


def test_finding_text_full_explanation_and_none():
    long = json.dumps([{"step": "L1", "category": "FRA-01", "root": True, "text": "x " * 400}])
    assert S.finding_text("FRA-01", long) == ("x " * 400).strip()  # complete, no truncation as in examples (220 characters)
    assert S.finding_text("FRA-01", ROW["misconceptions"]) == "Added tops and bottoms."
    assert S.finding_text("NONE", ROW["misconceptions"]) == "none"
    with pytest.raises(ValueError, match="without an explanation"):
        S.finding_text("FRA-01", "[]")


def test_train_sample_with_and_without_notes_loss():
    rec = {"finding": "Added tops and bottoms.", "code": "FRA-01", "notes_loss": True}
    s = S.train_sample(ROW, None, None, "SYS", rec, " read first.\nThen compute. ")
    assert s.prefix == "" and s.meta["joint_target"] and s.meta["think"] and s.meta["system"] == s.meta["static"] == "SYS"
    assert s.target == [("read first.\nThen compute.", "notes"), ("\n</think>", "notes"), ("\n\nFinding:", "aux"),
                        (" Added tops and bottoms.", "why"), ("\nRoot:", "aux"), (" FRA-01", "code")]
    full = "".join(t for t, _ in s.target)
    assert full == S.rank_prefix("read first.\nThen compute.", "Added tops and bottoms.") + " FRA-01"  # scoring reads the same string
    s2 = S.train_sample(ROW, None, None, "SYS", dict(rec, notes_loss=False), "read first.")
    assert s2.prefix == "read first.\n</think>\n\nFinding:" and [t for _, t in s2.target] == ["why", "aux", "code"]
    assert s2.prefix + "".join(t for t, _ in s2.target) == S.rank_prefix("read first.", "Added tops and bottoms.") + " FRA-01"
    text = " ".join(v for k, v in s.parts if k == "text")
    assert "FRA-01" not in text and "Added tops" not in text  # no labelled row in the prompt
    assert [k for k, _ in s.parts] == [k for k, _ in PR.user_parts(ROW, None, None)]
    with pytest.raises(ValueError, match="without notes"):
        S.train_sample(ROW, None, None, "SYS", rec, " ")


def test_joint_target_tags_and_code_tokens(tok):
    from vlm.engine import LOSS_TAGS, Engine

    rec = {"finding": "Added tops and bottoms.", "code": "FRA-01", "notes_loss": True}
    s = S.train_sample(ROW, None, None, "SYS", rec, "Line one.\nThe student wrote 2/5.")
    ids, tags = Engine.joint_target(SimpleNamespace(tok=tok), s.target)
    full = "".join(t for t, _ in s.target)
    assert ids == tok(full, add_special_tokens=False)["input_ids"]  # exactly the tokenisation of the whole string
    think = tok.convert_tokens_to_ids("</think>")
    assert tags[ids.index(think)] == LOSS_TAGS["notes"]  # closing the thinking block belongs to the note sequence
    code_ids = [i for i, t in zip(ids, tags) if t == LOSS_TAGS["code"]]
    assert code_ids == tok(" FRA-01", add_special_tokens=False)["input_ids"]  # like the candidate in scoring
    assert ids[-len(code_ids):] == code_ids and tags.count(LOSS_TAGS["why"]) >= 5
    # NONE: finding "none" (one token), code "NONE"
    s0 = S.train_sample(ROW, None, None, "SYS", {"finding": "none", "code": "NONE", "notes_loss": False}, "ok")
    ids0, tags0 = Engine.joint_target(SimpleNamespace(tok=tok), s0.target)
    assert [i for i, t in zip(ids0, tags0) if t == LOSS_TAGS["why"]] == tok(" none", add_special_tokens=False)["input_ids"]
    with pytest.raises(ValueError, match="code splits differently"):  # a code that merges with its neighbour is not learned
        Engine.joint_target(SimpleNamespace(tok=tok), [("x", "aux"), ("FRA-01", "code")])


def test_cut_notes_at_token_limit(tok):
    text = " ".join(f"step{i}" for i in range(400))
    same, n, cut = S.cut_notes(text, tok, 10_000)
    assert same == text and not cut and n == len(tok(text, add_special_tokens=False)["input_ids"])
    short, n2, cut2 = S.cut_notes(text, tok, 50)
    assert cut2 and n2 == 50 and text.startswith(short) and len(tok(short, add_special_tokens=False)["input_ids"]) <= 50


def test_parse_generation_states():
    classes = ["NONE", "FRA-01", "CAL-01"]
    p = S.parse_generation("reading line 2\n</think>\n\nFinding: Added tops.\nRoot: FRA-01", classes)
    assert p == {"notes": "reading line 2", "closed": True, "finding": "Added tops.", "root": "FRA-01", "state": "complete"}
    p = S.parse_generation("long notes without end", classes)
    assert p["state"] == "open" and p["finding"] is None and p["notes"] == "long notes without end" and not p["closed"]
    p = S.parse_generation("x\n</think>\n\nFinding: half", classes)  # cut off at the length limit inside the finding
    assert p["state"] == "open" and p["finding"] is None and p["notes"] == "x"
    p = S.parse_generation("x\n</think>\n\nRoot: FRA-01", classes)  # finding skipped
    assert p["state"] == "open"
    p = S.parse_generation("x\n</think>\n\nFinding: none\nRoot: `NONE`.", classes)
    assert p["state"] == "complete" and p["root"] == "NONE" and p["finding"] == "none"
    assert S.parse_generation("x\n</think>\n\nFinding: y\nRoot: XYZ-99", classes)["state"] == "open"  # unknown code
    assert S.parse_generation("", classes)["state"] == "empty"


def test_rank_prefix_and_finish():
    assert S.rank_prefix(" notes \n", "  Added tops. ") == "notes\n</think>\n\nFinding: Added tops.\nRoot:"
    pre = S.rank_prefix("notes", None)
    assert pre == "notes\n</think>\n\nFinding:" and S.needs_finding(SimpleNamespace(prefix=pre))
    assert S.complete_prefix(pre, " The student\nadded tops.\nRoot: FRA") == pre + " The student added tops.\nRoot:"
    assert S.complete_prefix(pre, "Root: X") == pre + "\nRoot:"
    with pytest.raises(ValueError):
        S.complete_prefix("x\nRoot:", "y")


def _targets(tmp_path, recs, fold):
    f = tmp_path / "t.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    f.with_suffix(".meta.json").write_text(json.dumps({"fold": fold, "sha256": S.sha256_file(f)}))
    return f


def test_load_targets_guards(tmp_path):
    folds = np.array([0, 1, 1, 2])
    rec = lambda i: {"table": "train", "row": i, "fold": int(folds[i]), "notes": "n", "notes_loss": i == 3, "finding": "f", "code": "FRA-01"}  # noqa: E731
    recs, info = S.load_targets(_targets(tmp_path, [rec(0), rec(3)], 1), 1, folds, 4)
    assert [r["row"] for r in recs] == [0, 3] and info["notes_loss"] == 1 and info["rows"] == 2
    with pytest.raises(ValueError, match="held-out fold"):  # row of the validation fold in the training target
        S.load_targets(_targets(tmp_path, [rec(0), rec(1)], 1), 1, folds, 4)
    with pytest.raises(ValueError, match="fold 1, requested 2"):
        S.load_targets(_targets(tmp_path, [rec(0)], 1), 2, folds, 4)
    f = _targets(tmp_path, [rec(0), rec(0)], 1)
    with pytest.raises(ValueError, match="appears twice"):
        S.load_targets(f, 1, folds, 4)
    f = _targets(tmp_path, [rec(0)], 1)
    f.write_text(f.read_text().replace('"n"', '"m"'))
    with pytest.raises(ValueError, match="checksum"):
        S.load_targets(f, 1, folds, 4)
    recs, _ = S.load_targets(_targets(tmp_path, [rec(i) for i in range(4)], None), None, folds, 4)  # full fit: all rows
    assert len(recs) == 4


def test_check_recipe_guards():
    system = S.system_text(TAX)
    folds = np.array([0, 1, 1, 2])
    recipe = {"system_sha256": S.sha256_text(system), "fold": 1}
    S.check_recipe(recipe, system, "train", [1, 2], folds)
    S.check_recipe(recipe, system, "test", [0, 5, 9], None)
    with pytest.raises(ValueError, match="outside fold 1"):
        S.check_recipe(recipe, system, "train", [0, 1], folds)
    with pytest.raises(ValueError, match="system text"):
        S.check_recipe(recipe, S.system_text(TAX, "name"), "test", [], None)
    with pytest.raises(ValueError, match="all training rows"):
        S.check_recipe(dict(recipe, fold=None), system, "train", [1], folds)
    with pytest.raises(ValueError, match="merged"):
        S.check_recipe(None, system, "test", [], None)


def test_merge_lora_matches_formula_links_unchanged_and_checks_names(tmp_path):
    from safetensors.torch import load_file, save_file

    base, ad, out = tmp_path / "base", tmp_path / "ad", tmp_path / "models" / "m"
    base.mkdir()
    ad.mkdir()
    g = torch.Generator().manual_seed(0)
    w1 = torch.randn(6, 4, generator=g).to(torch.bfloat16)
    w2 = torch.randn(3, 3, generator=g).to(torch.bfloat16)
    save_file({"model.language_model.layers.1.mlp.up_proj.weight": w1, "model.norm.weight": torch.ones(4)}, str(base / "s1.safetensors"))
    save_file({"model.visual.blocks.0.proj.weight": w2}, str(base / "s2.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.layers.1.mlp.up_proj.weight": "s1.safetensors", "model.norm.weight": "s1.safetensors",
        "model.visual.blocks.0.proj.weight": "s2.safetensors"}}))
    (base / "config.json").write_text("{}")
    (base / "chat_template.jinja").write_text("tpl")
    a, b = torch.randn(2, 4, generator=g), torch.randn(6, 2, generator=g)
    save_file({"base_model.model.model.language_model.layers.1.mlp.up_proj.lora_A.weight": a,
               "base_model.model.model.language_model.layers.1.mlp.up_proj.lora_B.weight": b}, str(ad / "adapter_model.safetensors"))
    (ad / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_rslora": False}))
    info = S.merge_lora(base, ad, out, log=lambda *x: None, recipe={"system_sha256": "abc", "fold": 1})
    got = load_file(str(out / "s1.safetensors"))
    want = (w1.float() + 2.0 * (b @ a)).to(torch.bfloat16)
    assert torch.equal(got["model.language_model.layers.1.mlp.up_proj.weight"], want) and torch.equal(got["model.norm.weight"], torch.ones(4))
    assert got["model.language_model.layers.1.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert (out / "s2.safetensors").is_symlink() and (out / "config.json").read_text() == "{}" and (out / "chat_template.jinja").is_file()
    assert info["merged_tensors"] == 1 and info["shards_rewritten"] == 1 and info["shards_linked"] == 1
    assert S.read_recipe(out) == {"system_sha256": "abc", "fold": 1}
    # name matching: an adapter weight without a partner in the base model aborts
    save_file({"base_model.model.model.language_model.layers.9.mlp.up_proj.lora_A.weight": a,
               "base_model.model.model.language_model.layers.9.mlp.up_proj.lora_B.weight": b}, str(ad / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="without a partner"):
        S.merge_lora(base, ad, out, log=lambda *x: None)
    (ad / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_dora": True}))
    with pytest.raises(ValueError, match="standard LoRA"):
        S.merge_lora(base, ad, out, log=lambda *x: None)


def test_combine_rank_outputs_for_a8(tmp_path):
    runs = tmp_path / "runs"
    for job, rows in (("r1", [5, 1]), ("r2", [3])):
        d = runs / "m3" / "m3" / job
        d.mkdir(parents=True)
        np.save(d / "bench_idx.npy", np.array(rows))
        np.save(d / "bench_proto_rank.npy", np.arange(len(rows) * 3, dtype=float).reshape(len(rows), 3))
        np.save(d / "bench_proto_rank.done.npy", np.ones(len(rows), bool))
        (d / "bench_meta.json").write_text(json.dumps({"table": "train", "n": len(rows)}))
    info = S.combine_rank(runs, "m3", ["r1", "r2"], "bench", "all")
    out = runs / "m3" / "m3" / "all"
    assert info["n"] == 3 and np.load(out / "bench_idx.npy").tolist() == [1, 3, 5]
    assert np.load(out / "bench_proto_rank.npy")[:, 0].tolist() == [3.0, 0.0, 0.0] and np.load(out / "bench_proto_rank.done.npy").all()
    assert json.loads((out / "bench_meta.json").read_text())["table"] == "train"
    with pytest.raises(ValueError, match="overlap"):
        S.combine_rank(runs, "m3", ["r1", "r1"], "bench", "x")


def _root(tmp_path):
    from PIL import Image

    d = tmp_path / "competition_data" / "images" / "train"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("a.jpeg", "b.jpeg"):
        Image.fromarray(np.full((20, 30, 3), 128, np.uint8)).save(d / name)
    return pd.DataFrame([dict(ROW, trace_id="a", image_file="a.jpeg"), dict(ROW, trace_id="b", image_file="b.jpeg", question_text="Question B")])


class _Proc:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_sft_generate_vllm_records_notes_guard_and_resume(tmp_path, monkeypatch):
    from vlm import think_vllm as TV
    from vlm.data import ImageConfig

    df = _root(tmp_path)
    classes = ["NONE", "FRA-01", "CAL-01"]
    model = tmp_path / "merged"
    model.mkdir()
    (model / S.RECIPE_FILE).write_text(json.dumps({"system_sha256": S.sha256_text(S.system_text(TAX)), "fold": 1}))
    calls = []

    def fake_post(url, body, timeout):
        calls.append(body)
        assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["max_tokens"] == 3200
        assert body["messages"][0]["content"][0]["text"] == S.system_text(TAX)
        user = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
        if "Question B" in user[0]:
            return {"choices": [{"message": {"content": "notes too long"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 9, "completion_tokens": 3200}}
        return {"choices": [{"message": {"content": "read\n</think>\n\nFinding: Added tops.\nRoot: FRA-01"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12}}

    monkeypatch.setattr(TV, "start_server", lambda model_dir, spec, log_path: _Proc())
    monkeypatch.setattr(TV, "wait_ready", lambda port, proc, timeout_s: None)
    monkeypatch.setattr(TV, "_post", fake_post)
    folds = np.array([1, 1])
    out = tmp_path / "out"
    stats = S.generate({"concurrency": 2}, model, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), out, "gen",
                       log=lambda *a: None, folds=folds)
    assert len(calls) == 2 and stats["states"] == {"complete": 1, "open": 1, "empty": 0} and stats["ends"] == {"length": 1, "stop": 1}
    table, notes = S.load_notes(out / "gen_notes.jsonl")
    assert table == "train" and notes == {0: ("read", "Added tops."), 1: ("notes too long", None)}
    meta = json.loads((out / "gen_meta.json").read_text())
    assert meta["table"] == "train" and meta["share_none_root"] == 0.0 and "Added" not in json.dumps(meta)  # aggregates only
    # resume: everything exists, no request; a different recipe requires a new tag
    S.generate({"concurrency": 2}, model, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), out, "gen", log=lambda *a: None, folds=folds)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="recipe changed"):
        S.generate({"max_tokens": 100}, model, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), out, "gen", log=lambda *a: None, folds=folds)
    # measurement only on the held-out fold; guard against a missed form
    with pytest.raises(ValueError, match="outside fold 1"):
        S.generate({}, model, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), tmp_path / "o2", "gen", log=lambda *a: None,
                   folds=np.array([1, 2]))
    monkeypatch.setattr(TV, "_post", lambda url, body, timeout: {"choices": [{"message": {"content": "nonsense"}, "finish_reason": "length"}], "usage": {}})
    with pytest.raises(RuntimeError, match="form missed"):
        S.generate({"guard_rows": 2, "concurrency": 1}, model, df, np.array([0, 1]), "train", tmp_path, TAX, classes, ImageConfig(), tmp_path / "o3", "gen",
                   log=lambda *a: None, folds=folds)


def test_proto_rank_sft_mode_builds_canonical_prefix_and_generates_missing_findings(tmp_path):
    from vlm.data import ImageConfig
    from vlm.runner import ProbeRunner

    df = _root(tmp_path)
    system = S.system_text(TAX)
    fake = SimpleNamespace(proto_notes={0: "read", 1: "long"}, sft_findings={0: "Added tops.", 1: None}, proto_table="train", proto_system=system,
                           builder=SimpleNamespace(root=tmp_path, image=ImageConfig()))
    s0 = ProbeRunner._build_proto(fake, df, 0, "train")
    s1 = ProbeRunner._build_proto(fake, df, 1, "train")
    assert s0.prefix == "read\n</think>\n\nFinding: Added tops.\nRoot:" and s0.meta["system"] == system and s0.meta["static"] == system
    assert s1.prefix == "long\n</think>\n\nFinding:" and S.needs_finding(s1)

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return " The student\nadded.\nRoot:"

    seen = {}

    def gen(samples, max_new_tokens, stop_text=None, return_ids=False):
        seen.update(n=len(samples), stop=stop_text, ids=return_ids, limit=max_new_tokens)
        return [[1, 2, 3] for _ in samples]

    fake2 = SimpleNamespace(engine=SimpleNamespace(generate=gen, tok=Tok()), job={"finish_tokens": 99})
    made = ProbeRunner._sft_finish(fake2, [s0, s1])
    assert made == [None, "The student added."] and seen == {"n": 1, "stop": "Root:", "ids": True, "limit": 99}
    assert s1.prefix == "long\n</think>\n\nFinding: The student added.\nRoot:" and s0.prefix.endswith("Added tops.\nRoot:")
    fake.sft_findings = None  # without sft mode: the earlier ranking behind the notes
    assert ProbeRunner._build_proto(fake, df, 0, "train").prefix == "read\n</think>\n\nRoot:"


def test_defaults_keep_existing_training_unchanged():
    from vlm.engine import LOSS_TAGS
    from vlm.train import TrainConfig

    cfg = TrainConfig()
    assert cfg.prefix_with_adapter is False and cfg.frozen_prefix is True and cfg.tasks == ("diag", "full")
    assert {k: LOSS_TAGS[k] for k in ("code", "fac", "aux", "why")} == {"code": 0, "fac": 1, "aux": 2, "why": 3} and LOSS_TAGS["notes"] == 4


@pytest.mark.parametrize("name", ["m3_pilot_rp", "local_m3_test"])
def test_m3_configs_are_consistent(name):
    """Every ranking and generation job measures only the fold its merged model has NOT learned; training jobs compute
    the header with the adapter; the chain refers to existing jobs; pre-runs have must_pass."""
    import yaml

    from vlm.train import TrainConfig

    cfg = yaml.safe_load((COMP / "configs" / "vlm" / f"{name}.yaml").read_text())
    jobs = {j["name"]: j for w in cfg["workers"] for j in w["jobs"]}
    order = [j["name"] for w in cfg["workers"] for j in w["jobs"]]
    fold_of_model = {}
    for n in order:
        j = jobs[n]
        if j["kind"] == "train":
            tcfg = TrainConfig(**{**j["train"], "tasks": tuple(j["train"]["tasks"])})
            assert tcfg.tasks == ("proto_sft",) and tcfg.prefix_with_adapter and tcfg.code_loss == "sum" and tcfg.why_loss == "first+mean"
            assert j["sft_targets"] == f"derived/sft/m3_fold{j['fold']}_targets.jsonl" and not j.get("probes")
            S.refuse_task_knowledge(j)
        elif j["kind"] == "merge":
            src = jobs[j["adapter_from"]]
            assert order.index(j["adapter_from"]) < order.index(n) and src["kind"] == "train"
            merged = [m for m, spec in cfg["models"].items() if spec.get("merged") == j["out_model"]]
            assert len(merged) == 1
            fold_of_model[merged[0]] = src["fold"]
        elif j["kind"] == "sft_gen":
            assert j["rows"]["folds"] == [fold_of_model[j["model"]]] and j.get("split", "train") == "train"
            S.refuse_task_knowledge(j)
        elif j["kind"] == "bench":
            gen = jobs[j["sft_from"].split("/")[0]]
            assert j["probes"] == ["proto_rank"] and gen["model"] == j["model"] and j["rows"] == gen["rows"]
            assert j["rows"]["folds"] == [fold_of_model[j["model"]]] and j.get("drop_model") is True
            assert j.get("taxonomy", "full") == gen.get("taxonomy", "full")
    first = jobs[order[0]]
    assert first["kind"] == "train" and first.get("must_pass")
    if name == "m3_pilot_rp":
        smoke = [n for n in order if n.startswith("m3_smoke")]
        assert smoke == order[:4] and all(jobs[n].get("must_pass") for n in smoke)
        assert all(jobs[n]["beams"] == 74 for n in order if jobs[n]["kind"] == "bench")
        assert sorted(fold_of_model.values()) == [1, 1, 2]


def test_adapter_head_keeps_gradient_under_autocast(monkeypatch):
    """Regression: "element 0 of tensors does not require grad". The header with adapter is computed under no_grad inside
    the autocast region of the training step; autocast (torch up to 2.8) put bf16 copies of the fp32 adapter weights
    WITHOUT grad_fn into its cache, and the rest of the step reused them. Locally the chain ran on the CPU without
    autocast; newer torch versions cast with gradient in the cache and do not show the error.
    Here: one step on the 0.8B model with checkpointing, adapter header and autocast (bf16) must give a loss with grad_fn,
    carry gradients into the adapters, and clear the cache after the header (the fix; verified on torch 2.8: without it
    grad_fn is missing, with it the gradient flows)."""
    model_dir = COMP / "data" / "models" / "Qwen3.5-0.8B"
    if not (model_dir / "config.json").is_file():
        pytest.skip("weights of the smallest model are not available locally")
    from vlm.engine import Engine, EngineConfig
    from vlm.train import TrainConfig, add_lora, sample_loss

    torch.manual_seed(0)
    eng = Engine(EngineConfig(model_path=str(model_dir), dtype="fp32", device="cpu"))
    cfg = TrainConfig(lora_r=4, lora_alpha=8, lora_from_layer=20, grad_checkpointing=True, prefix_with_adapter=True)
    model = add_lora(eng, cfg)
    model.train()
    s = S.train_sample(ROW, None, None, S.system_text(TAX), {"finding": "Added tops and bottoms.", "code": "FRA-01", "notes_loss": True},
                       "Line one: 1/2 + 1/3. The student wrote 2/5.")
    s.parts = [("text", "Grade: 7 | Topic: Fractions\nQuestion:\n1/2 + 1/3 = ...\nAnswer key: 5/6\n\nStudent's working: 1/2 + 1/3 = 2/5")]
    batch = eng.encode_train(s)
    assert batch["n_static"] >= eng.min_static  # the header really comes from the cache (with adapter)
    cleared = []
    real_clear = torch.clear_autocast_cache
    monkeypatch.setattr(torch, "clear_autocast_cache", lambda: (cleared.append(1), real_clear())[1])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, parts = sample_loss(eng, batch, cfg)
    assert cleared, "after the header with adapter the autocast cache must be cleared"
    assert eng._static_mode is True and set(parts) == {"code", "aux", "why", "notes"}
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    grads = [float(p.grad.abs().sum()) for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None]
    assert grads and max(grads) > 0


def test_freeze_m3_texts_fills_generated_findings(tmp_path):
    """Frozen M3 texts carry the finding from the generation or the one generated before scoring."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_m3_texts", Path(__file__).resolve().parent.parent / "tools" / "freeze_m3_texts.py")
    fz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fz)
    gen = tmp_path / "gen_notes.jsonl"
    gen.write_text("\n".join(json.dumps(r) for r in [
        {"table": "train", "row": 3, "text": "n3", "finding": "f3", "state": "complete"},
        {"table": "train", "row": 5, "text": "n5", "finding": None, "state": "open"},
        {"table": "train", "row": 7, "text": "n7", "finding": "", "state": "open"},
    ]) + "\n")
    rank = tmp_path / "bench_proto_rank.jsonl"
    rank.write_text("\n".join(json.dumps(r) for r in [
        {"i": 3, "text": ""}, {"i": 5, "text": "[finding generated] made five"}, {"i": 7, "text": "[finding generated] "},
    ]) + "\n")
    rows, meta = fz.freeze(gen, rank)
    by = {r["row"]: r for r in rows}
    assert by[3]["finding"] == "f3" and by[5]["finding"] == "made five" and by[7]["finding"] is None
    assert meta["finding_from_gen"] == 1 and meta["finding_from_rank"] == 1 and meta["still_open"] == 1
    from vlm import sft as S
    assert S.rank_prefix(by[5]["text"], by[5]["finding"]) == S.complete_prefix(S.rank_prefix("n5", None), "made five")
