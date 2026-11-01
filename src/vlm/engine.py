"""Load the model, encode samples, generate text and score candidates.

Two building blocks carry all tasks:

* `Engine.generate`: batched, deterministic text generation (greedy, no sampling).
* `Engine.score`: exact log-probabilities of a fixed candidate set (all codes, or "yes"/"no") behind a forced
  answer prefix: beam search over the prefix tree of the candidates. The ranking comes from log-probabilities
  instead of sampling, so it is deterministic.

The candidates are tokenised on their own (`CandidateSet`); training appends the same token sequences
(`Engine.encode_train`), so the token split used for training and for scoring is identical.
"""

from __future__ import annotations

import copy
import gc
import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig, LogitsProcessor, LogitsProcessorList

from . import prompts as P

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj")
LOSS_TAGS = {"code": 0, "fac": 1, "aux": 2, "why": 3, "notes": 4}  # notes: own notes as training target (task proto_sft, vlm/sft.py)
# "Read first" variant (vlm/sft.py, reading_block): read = transcription of the working before the notes (weight
# TrainConfig.w_read), ctx = text in the target WITHOUT loss (notes without notes_loss behind the transcription). Earlier
# targets carry neither marker; for them every computation stays bit-identical.
LOSS_TAGS.update({"read": 5, "ctx": 6})


class CandidateSet:
    """Fixed candidate strings as token sequences with a terminator token and a prefix tree."""

    def __init__(self, tokenizer, strings: list[str], terminator: str = "\n"):
        term = tokenizer(terminator, add_special_tokens=False)["input_ids"]
        if len(term) != 1:
            raise ValueError(f"terminator {terminator!r} is not a single token")
        self.term_id = term[0]
        self.strings = list(strings)
        self.token_lists = [tokenizer(s, add_special_tokens=False)["input_ids"] + [self.term_id] for s in self.strings]
        if len({tuple(t) for t in self.token_lists}) != len(self.strings):
            raise ValueError("candidates are not unique after tokenisation")
        self.max_len = max(len(t) for t in self.token_lists)
        self.index = {tuple(t): i for i, t in enumerate(self.token_lists)}
        trie: dict[tuple, set] = {}
        for t in self.token_lists:
            for k in range(len(t)):
                trie.setdefault(tuple(t[:k]), set()).add(t[k])
        self.trie = {k: sorted(v) for k, v in trie.items()}
        for t in self.token_lists:  # no candidate may be a prefix of another one
            if tuple(t) in self.trie:
                raise ValueError("candidate set is not prefix-free")

    def __len__(self) -> int:
        return len(self.strings)


class _TrieProcessor(LogitsProcessor):
    """Allow in every step only tokens that continue a candidate.

    Once a candidate is complete, the terminator token follows with log-probability 0: it only ends the beam.
    The score is therefore the log-probability of the candidate tokens alone (the beam search already passes
    log-softmax values in here).
    """

    def __init__(self, cands: CandidateSet, prompt_len: int):
        self.cands, self.prompt_len = cands, prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        out = torch.full_like(scores, float("-inf"))
        gen = input_ids[:, self.prompt_len :].tolist()
        for i, g in enumerate(gen):
            nxt = self.cands.trie.get(tuple(g))
            if nxt is None or nxt == [self.cands.term_id]:  # candidate complete (or beam already finished)
                out[i, self.cands.term_id] = 0.0
            else:
                out[i, nxt] = scores[i, nxt]
        return out


@dataclass
class EngineConfig:
    model_path: str
    dtype: str = "fp16"  # fp16 | bf16 | fp32
    quant: str | None = None  # None | "4bit" | "8bit"
    device: str = "cuda:0"
    adapter: str | None = None
    attn: str | None = None  # e.g. "sdpa"
    max_beam_tokens: int = 36000  # rows x prefixes x prompt tokens per scoring group (for 15 GB cards)
    max_memory: dict | None = None  # only with device="auto": upper bound per GPU, e.g. {0: "11GiB", 1: "13GiB"}
    prefix_cache: bool = True  # reuse the computed state of the prompt header shared by all rows
    # Compute the header WITH the adapter (proto_sft adapter, trained with prefix_with_adapter), the way the merged model
    # computes it. For the unmerged adapter on a 4-bit base model (Kaggle T4), where merging into the quantised weights is
    # not possible. Default: header from the base model.
    head_with_adapter: bool = False


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def free_gpu_memory() -> None:
    """Really release freed tensors (including those in reference cycles) and empty the allocator cache.

    Only effective outside an except block: a live exception keeps all tensors alive through its traceback.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.dtype = DTYPES[cfg.dtype]
        self.proc = AutoProcessor.from_pretrained(cfg.model_path)
        self.tok = self.proc.tokenizer
        kwargs: dict = {"dtype": self.dtype}
        if cfg.quant == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype
            )
        elif cfg.quant == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif cfg.quant:
            raise ValueError(f"unknown quantisation {cfg.quant!r}")
        if cfg.attn:
            kwargs["attn_implementation"] = cfg.attn
        if cfg.device in ("auto", "sequential"):  # spread over all visible GPUs (large models); "sequential":
            # fill the cards one after another up to max_memory instead of balancing; on two T4 cards, with "auto" card 0 holds only
            # the balanced half minus the largest layer (27B in 4-bit: 5.1 versus 11.2 GiB)
            kwargs["device_map"] = cfg.device
            if cfg.max_memory:
                kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in cfg.max_memory.items()}
        elif cfg.device != "cpu":
            kwargs["device_map"] = {"": cfg.device}
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_path, **kwargs)
        if cfg.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, cfg.adapter)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device  # the inputs go there
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.image_token = getattr(self.proc, "image_token", "<|image_pad|>")
        self.max_beam_tokens = cfg.max_beam_tokens
        self.prefix_cache = cfg.prefix_cache
        self.min_static = 256  # below this the cache does not pay off
        self._static_ids, self._static_cache = None, None
        self._static_mode = False  # True: the stored header was computed WITH the adapter (_static_state, adapter=True)
        self._train_pristine = None  # set only during training (train.py, layer wrappers)
        self.oom_retries = 0  # how often scoring was repeated with fewer prefixes because of memory shortage

    # ------------------------------------------------------------------ Encoding

    def chat_text(self, sample: P.Sample) -> str:
        content = [{"type": "text", "text": v} if k == "text" else {"type": "image"} for k, v in sample.parts]
        messages = [
            # own system text of a sample (teacher protocol: verdict rules and taxonomy live there, vlm/proto.py)
            {"role": "system", "content": [{"type": "text", "text": sample.meta.get("system") or P.SYSTEM}]},
            {"role": "user", "content": content},
        ]
        think = bool(sample.meta.get("think"))  # thinking mode: the answer then starts with "<think>"
        kw = {"reasoning_effort": sample.meta["effort"]} if think and sample.meta.get("effort") else {}  # reasoning effort (Qwen3.8)
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think, **kw)
        return text + sample.prefix

    def _encode_one(self, sample: P.Sample) -> dict:
        images = [v for k, v in sample.parts if k == "image"]
        text = self.chat_text(sample)
        enc = self.proc(text=[text], images=images or None, return_tensors="pt")
        out = {"input_ids": enc["input_ids"][0], "n_static": 0}
        static = sample.meta.get("static")
        if static and self.prefix_cache:  # number of tokens of the header shared by all rows (with a safety margin)
            j = text.find(static)
            if j >= 0:
                lead = self.tok(text[: j + len(static)], add_special_tokens=False)["input_ids"]
                ids = out["input_ids"].tolist()
                k = 0
                while k < min(len(lead), len(ids)) and lead[k] == ids[k]:
                    k += 1
                out["n_static"] = max(0, k - 2)
        # Marks image tokens (1) versus text (0); the model needs it for the position encoding.
        out["mm_token_type_ids"] = enc["mm_token_type_ids"][0] if "mm_token_type_ids" in enc else torch.zeros_like(out["input_ids"])
        if images:
            out["pixel_values"] = enc["pixel_values"]
            out["image_grid_thw"] = enc["image_grid_thw"]
        extra = sample.meta.get("extra_ids")
        if extra:
            out = self._append(out, extra)
        return out

    @staticmethod
    def _append(one: dict, ids: list[int]) -> dict:
        """Append text tokens to an encoded sample."""
        add = torch.tensor(ids, dtype=torch.long)
        one = dict(one)
        one["input_ids"] = torch.cat([one["input_ids"], add])
        one["mm_token_type_ids"] = torch.cat([one["mm_token_type_ids"], torch.zeros_like(add)])
        return one

    def encode(self, samples: list[P.Sample]) -> dict:
        """Batch with left padding (for generating and scoring)."""
        return self._collate([self._encode_one(s) for s in samples])

    def _collate(self, ones: list[dict]) -> dict:
        n = max(len(o["input_ids"]) for o in ones)
        ids = torch.full((len(ones), n), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(ones), n), dtype=torch.long)
        mm = torch.zeros((len(ones), n), dtype=torch.long)
        for i, o in enumerate(ones):
            k = len(o["input_ids"])
            ids[i, n - k :] = o["input_ids"]
            att[i, n - k :] = 1
            mm[i, n - k :] = o["mm_token_type_ids"]
        batch = {"input_ids": ids.to(self.device), "attention_mask": att.to(self.device), "mm_token_type_ids": mm.to(self.device)}
        pix = [o["pixel_values"] for o in ones if "pixel_values" in o]
        if pix:
            batch["pixel_values"] = torch.cat(pix).to(self.device, self.dtype)
            batch["image_grid_thw"] = torch.cat([o["image_grid_thw"] for o in ones if "image_grid_thw" in o]).to(self.device)
        return batch

    def joint_target(self, segments: list) -> tuple[list[int], list[int]]:
        """Training target tokenised as ONE string (meta["joint_target"]); a token's marker is that of the segment in which
        its first character lies. This way the model sees the same tokens during training as scoring later, which
        reads the generated text as one string (at boundaries such as ".\\n" before "Root:" a separate tokenisation
        splits differently). The code segment must split exactly like the candidate on its own (engine.CandidateSet)."""
        full = "".join(text for text, _ in segments)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
        ends = np.cumsum([len(text) for text, _ in segments])
        seg = [min(int(np.searchsorted(ends, int(s), side="right")), len(segments) - 1) for s, _ in offs]
        tags = [LOSS_TAGS[segments[j][1]] for j in seg]
        for j, (text, tag) in enumerate(segments):
            if tag == "code":
                got = [t for t, k in zip(ids, seg) if k == j]
                if got != self.tok(text, add_special_tokens=False)["input_ids"]:
                    raise ValueError("joint_target: the code splits differently in context than as a candidate")
        return ids, tags

    def encode_train(self, sample: P.Sample) -> dict:
        """A sample with a training target: input, target mask and marker per target position (code | fac | aux)."""
        one = self._encode_one(sample)
        n_prompt = len(one["input_ids"])
        tgt, tags = [], []
        if sample.meta.get("joint_target"):
            tgt, tags = self.joint_target(sample.target)
        else:
            for text, tag in sample.target:
                t = self.tok(text, add_special_tokens=False)["input_ids"]
                tgt += t
                tags += [LOSS_TAGS[tag]] * len(t)
        tgt.append(self.eos_id)
        tags.append(LOSS_TAGS["aux"])
        one = self._append(one, tgt)
        ids = one["input_ids"]
        batch = {
            "input_ids": ids[None].to(self.device),
            "attention_mask": torch.ones_like(ids)[None].to(self.device),
            "mm_token_type_ids": one["mm_token_type_ids"][None].to(self.device),
            "n_prompt": n_prompt,
            "n_static": one.get("n_static", 0),
            "target_ids": torch.tensor(tgt, dtype=torch.long, device=self.device),
            "target_tags": torch.tensor(tags, dtype=torch.long, device=self.device),
        }
        if "pixel_values" in one:
            batch["pixel_values"] = one["pixel_values"].to(self.device, self.dtype)
            batch["image_grid_thw"] = one["image_grid_thw"].to(self.device)
        return batch

    # ------------------------------------------------------------------ Generation and scoring

    def _model_inputs(self, batch: dict) -> dict:
        keys = ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")
        return {k: v for k, v in batch.items() if k in keys}

    @torch.inference_mode()
    def generate(self, samples: list[P.Sample], max_new_tokens: int = 384, repetition_penalty: float = 1.0,
                 stop_text: str | None = None, return_ids: bool = False) -> list:
        """Greedy generation. Samples with a shared prompt header run one by one over the same fixed header as training
        and scoring (otherwise a fine-tuned model would see a header during generation that it never learned);
        all others run batched through the library.

        stop_text: generation ends as soon as the generated text contains this string (it belongs to the output).
        return_ids: return the generated token sequences instead of the text; whoever scores behind the own text
        afterwards appends exactly these tokens (`meta["extra_ids"]`) instead of re-tokenising the text.
        """
        ones = [self._encode_one(s) for s in samples]
        if stop_text or return_ids:  # row by row: stop position and token sequence per sample
            ids = [self._generate_cached(o, max_new_tokens, stop_text=stop_text, return_ids=True) for o in ones]
            return ids if return_ids else [self.tok.decode(t, skip_special_tokens=True).strip() for t in ids]
        out: list[str | None] = [None] * len(samples)
        # Without an adapter, the batched path over the whole prompt is equivalent (the header there is the base model itself).
        plain = [i for i, o in enumerate(ones) if o["n_static"] < self.min_static or not self.has_adapter()]
        if plain:
            batch = self._collate([ones[i] for i in plain])
            self._reset_positions()
            gen = GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, repetition_penalty=repetition_penalty,
                eos_token_id=self.eos_id, pad_token_id=self.pad_id,
            )
            res = self.model.generate(**self._model_inputs(batch), generation_config=gen)
            new = res[:, batch["input_ids"].shape[1] :]
            for i, text in zip(plain, self.tok.batch_decode(new, skip_special_tokens=True)):
                out[i] = text.strip()
        for i, o in enumerate(ones):
            if out[i] is None:
                out[i] = self._generate_cached(o, max_new_tokens)
        return out

    def _generate_cached(self, one: dict, max_new_tokens: int, stop_text: str | None = None, return_ids: bool = False):
        """Greedy generation of one sample, token by token with cache; behind the fixed prompt header if the sample has one."""
        batch = self._collate([one])
        inputs = self._model_inputs(batch)
        core = self._core()
        pos_kw = {k: batch[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw") if k in batch}
        pos = core._prepare_position_ids_for_generation(batch["input_ids"], pos_kw)
        k0 = one["n_static"] if one["n_static"] >= self.min_static else 0
        if k0:
            cache = self._static_state(one["input_ids"][:k0], adapter=self.cfg.head_with_adapter)
            rest = {k: (v[:, k0:] if k in ("input_ids", "mm_token_type_ids") else v) for k, v in inputs.items()}
            o = core.model(**rest, position_ids=pos[..., k0:], past_key_values=cache, use_cache=True)
        else:
            o = core.model(**inputs, position_ids=pos, use_cache=True)
        cache, att, last_pos = o.past_key_values, batch["attention_mask"], pos[..., -1:]
        tokens: list[int] = []
        for _ in range(max_new_tokens):
            nxt = int(core.lm_head(o.last_hidden_state[:, -1]).argmax(-1))
            if nxt == self.eos_id:
                break
            tokens.append(nxt)
            if stop_text and stop_text in self.tok.decode(tokens[-12:], skip_special_tokens=True):
                break
            att = torch.cat([att, att.new_ones((1, 1))], dim=1)
            last_pos = last_pos + 1
            o = core.model(input_ids=torch.tensor([[nxt]], device=self.device), attention_mask=att, position_ids=last_pos,
                           past_key_values=cache, use_cache=True)
        return tokens if return_ids else self.tok.decode(tokens, skip_special_tokens=True).strip()

    def _core(self):
        """The base model behind a PEFT wrapper."""
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _reset_positions(self) -> None:
        """The model remembers position offsets of the last image batch; clear them before every new forward pass,
        otherwise text-only inputs in the next step would use foreign offsets."""
        self._core().model.rope_deltas = None

    @torch.inference_mode()
    def score(self, samples: list[P.Sample], cands: CandidateSet, beams: int = 12, return_hidden: bool = False,
              exact_chunk: int = 0, first_top: int = 0):
        """Log-probability of every candidate (without terminator token); -inf = the prefix fell out of the best `beams`.

        With return_hidden=True also the last hidden state at the decision position (float16, [rows, width]),
        as features for probes without training the model.

        One forward pass over the prompt per sample, then the search moves token by token through the prefix tree
        of the candidates: per sample the `beams` best incomplete prefixes are kept, every complete candidate is
        recorded with its exact log-probability. The cache is reordered to the active prefixes in every step (the
        prompt is not read again).

        exact_chunk > 0: ALL candidates exactly, without selecting prefixes; after the forward pass the candidate
        tokens run in groups of `exact_chunk` as a continuation behind the prompt. Memory is bounded by the group
        size (for 15 GB cards); the result does not depend on `beams`.

        first_top > 0 (only with exact_chunk): continuations are computed only for candidates whose first token is
        among the `first_top` most probable first tokens; all others count as dropped (-inf). With very long prompt
        headers every continuation group costs a copy of the whole computed state; the selection saves most of it
        without touching the top ranks.
        """
        ones = [self._encode_one(s) for s in samples]
        if exact_chunk:
            res = np.full((len(samples), len(cands)), -np.inf, dtype=np.float64)
            hidden = [None] * len(samples)
            for i, one in enumerate(ones):
                # group size also capped by the memory limit: one copy of the prompt state per candidate of the group
                c = max(1, min(int(exact_chunk), self.max_beam_tokens // max(1, len(one["input_ids"]))))
                while True:  # if memory is insufficient, repeat with half the group size
                    oom = False
                    try:
                        res[i], hidden[i] = self._score_exact(one, cands, c, first_top)
                    except torch.OutOfMemoryError:
                        oom = True
                    if not oom:
                        break
                    free_gpu_memory()  # outside the except block, see below
                    if c <= 1:
                        raise torch.OutOfMemoryError(f"memory is insufficient even for one candidate per group ({len(one['input_ids'])} prompt tokens)")
                    c = max(1, c // 2)
                    self.oom_retries += 1
            if return_hidden:
                return res, np.stack(hidden).astype(np.float16)
            return res
        # Memory: for every active prefix the cache holds a copy of the prompt keys. Keep the groups small enough that
        # rows x prefixes x prompt length stays below the limit (at least one row).
        res = np.full((len(samples), len(cands)), -np.inf, dtype=np.float64)
        hidden = [None] * len(samples)
        width = min(beams, len(cands))
        start = 0
        while start < len(ones):
            stop = start + 1
            # Rows with a shared header run one by one: the header comes from the cache, and left padding would shift it.
            alone = ones[start]["n_static"] >= self.min_static
            while (not alone and stop < len(ones) and ones[stop]["n_static"] < self.min_static
                   and (stop + 1 - start) * width * max(len(o["input_ids"]) for o in ones[start : stop + 1]) <= self.max_beam_tokens):
                stop += 1
            b = beams
            while True:  # if memory is insufficient, repeat with half as many prefixes instead of losing the row
                oom = False
                try:
                    res[start:stop], h = self._score_group(ones[start:stop], cands, b)
                except torch.OutOfMemoryError:
                    oom = True
                if not oom:
                    break
                # Clean up only AFTER the except block: as long as the exception lives, its traceback keeps the tensors of the
                # failed attempt alive, and empty_cache() would release nothing.
                free_gpu_memory()
                if b <= 2:
                    raise torch.OutOfMemoryError(f"memory is insufficient even with two prefixes ({len(ones[start]['input_ids'])} prompt tokens)")
                b = max(2, b // 2)
                self.oom_retries += 1
            hidden[start:stop] = list(h)
            start = stop
        if return_hidden:
            return res, np.stack(hidden).astype(np.float16)
        return res

    def has_adapter(self) -> bool:
        return hasattr(self.model, "peft_config")

    def _static_state(self, ids: torch.Tensor, copy_state: bool = True, adapter: bool = False):
        """Computed state after the header shared by all rows, computed once and copied per row.

        The header is always computed with the **base model** (adapters disabled, no gradient): it is a fixed cache
        shared by training and inference. The full taxonomy therefore costs nothing per row, and a fine-tuned model
        sees exactly the header in inference that it saw in training.

        adapter=True (only training with TrainConfig.prefix_with_adapter, task proto_sft): header WITH the current
        adapter, without gradient and without dropout, the way a merged model computes it in vLLM and when scoring.
        The cache remembers in which form it was computed; training discards it after every optimiser step.
        """
        mode = bool(adapter) and self.has_adapter()
        if (self._static_ids is None or self._static_mode != mode or len(self._static_ids) != len(ids)
                or not torch.equal(self._static_ids, ids)):
            core = self._core()
            p = ids[None].to(self.device)
            att = torch.ones_like(p)
            with torch.no_grad():
                pos = core._prepare_position_ids_for_generation(p, {"input_ids": p, "attention_mask": att})
                if self.has_adapter() and not mode:
                    with self.model.disable_adapter():
                        out = core.model(input_ids=p, attention_mask=att, position_ids=pos, use_cache=True)
                elif mode and self.model.training:  # header as in inference: without adapter dropout
                    self.model.eval()
                    try:
                        out = core.model(input_ids=p, attention_mask=att, position_ids=pos, use_cache=True)
                    finally:
                        self.model.train()
                else:
                    out = core.model(input_ids=p, attention_mask=att, position_ids=pos, use_cache=True)
            self._static_ids, self._static_cache, self._static_mode = ids.clone(), out.past_key_values, mode
        return copy.deepcopy(self._static_cache) if copy_state else self._static_cache

    def _score_exact(self, one: dict, cands: CandidateSet, chunk: int, first_top: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """All candidates of one sample exactly: one forward pass over the prompt, then the candidates as a continuation
        in groups.

        The log-probability of a candidate is the sum over its tokens (without terminator): the first from the
        forward pass, every further one from a continuation that reads the preceding candidate tokens behind the
        computed state of the prompt. Shorter candidates of a group are right-padded (causally without effect).
        """
        batch = self._collate([one])
        inputs = self._model_inputs(batch)
        core = self._core()
        pos_kw = {k: batch[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw") if k in batch}
        pos = core._prepare_position_ids_for_generation(batch["input_ids"], pos_kw)
        k0 = one["n_static"] if one["n_static"] >= self.min_static else 0
        if k0:
            cache = self._static_state(one["input_ids"][:k0], adapter=self.cfg.head_with_adapter)
            rest = {k: (v[:, k0:] if k in ("input_ids", "mm_token_type_ids") else v) for k, v in inputs.items()}
            out = core.model(**rest, position_ids=pos[..., k0:], past_key_values=cache, use_cache=True)
        else:
            out = core.model(**inputs, position_ids=pos, use_cache=True)
        cache0 = out.past_key_values
        h_last = out.last_hidden_state[:, -1]
        hidden = h_last.float().cpu().numpy()[0]
        lp0 = torch.log_softmax(core.lm_head(h_last).float(), dim=-1)[0]
        del out
        att0, last_pos = batch["attention_mask"], pos[..., -1:]
        body = [t[:-1] for t in cands.token_lists]  # without terminator token
        res = lp0[torch.tensor([t[0] for t in body], device=lp0.device)].cpu().numpy().astype(np.float64)
        multi = sorted((j for j in range(len(body)) if len(body[j]) > 1), key=lambda j: len(body[j]))
        if first_top > 0:  # continue only candidates behind the most probable first tokens
            firsts = sorted({t[0] for t in body}, key=lambda tok: -float(lp0[tok]))[:first_top]
            keep = set(firsts)
            for j in multi:
                if body[j][0] not in keep:
                    res[j] = -np.inf
            multi = [j for j in multi if body[j][0] in keep]
        for s in range(0, len(multi), chunk):
            js = multi[s : s + chunk]
            n, width = len(js), max(len(body[j]) for j in js) - 1
            ids = torch.full((n, width), cands.term_id, dtype=torch.long)
            nxt = torch.zeros((n, width), dtype=torch.long)
            mask = torch.zeros((n, width))
            for r, j in enumerate(js):
                t = body[j]
                ids[r, : len(t) - 1] = torch.tensor(t[:-1])
                nxt[r, : len(t) - 1] = torch.tensor(t[1:])
                mask[r, : len(t) - 1] = 1.0
            cache = copy.deepcopy(cache0)
            cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))
            att = torch.cat([att0.expand(n, -1), att0.new_ones((n, width))], dim=1)
            steps = torch.arange(1, width + 1, device=last_pos.device)
            o = self.model(input_ids=ids.to(self.device), attention_mask=att, position_ids=last_pos.expand(-1, n, -1) + steps,
                           past_key_values=cache, use_cache=True)
            lp = torch.log_softmax(o.logits.float(), dim=-1)
            add = (lp.gather(2, nxt.to(lp.device)[..., None])[..., 0] * mask.to(lp.device)).sum(1).cpu().numpy()
            res[js] += add
            del cache, o, lp
        del cache0
        return res, hidden

    def _score_group(self, ones: list[dict], cands: CandidateSet, beams: int) -> tuple[np.ndarray, np.ndarray]:
        batch = self._collate(ones)
        inputs = self._model_inputs(batch)
        core = self._core()
        # Positions as in the library's generation: [4, n, length] from text position and image grid;
        # every following step counts one further per prefix.
        pos_kw = {k: batch[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw") if k in batch}
        pos = core._prepare_position_ids_for_generation(batch["input_ids"], pos_kw)
        k0 = ones[0]["n_static"] if len(ones) == 1 and ones[0]["n_static"] >= self.min_static else 0
        if k0:  # header from the cache, only the row's own remainder is computed
            cache = self._static_state(ones[0]["input_ids"][:k0], adapter=self.cfg.head_with_adapter)
            rest = {k: (v[:, k0:] if k in ("input_ids", "mm_token_type_ids") else v) for k, v in inputs.items()}
            out = core.model(**rest, position_ids=pos[..., k0:], past_key_values=cache, use_cache=True)
        else:
            out = core.model(**inputs, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        h_last = out.last_hidden_state[:, -1]
        hidden = h_last.float().cpu().numpy()
        logp = torch.log_softmax(core.lm_head(h_last).float(), dim=-1)
        n = len(ones)
        res = np.full((n, len(cands)), -np.inf, dtype=np.float64)
        term = cands.term_id
        att = batch["attention_mask"]
        last_pos = pos[..., -1:]

        def settle(beams_new: list) -> list:
            """Record complete candidates; keep the best incomplete prefixes per sample."""
            keep: dict[int, list] = {}
            for row, toks, sc, parent in beams_new:
                if cands.trie[toks] == [term]:
                    res[row, cands.index[toks + (term,)]] = sc
                else:
                    keep.setdefault(row, []).append((row, toks, sc, parent))
            out_beams = []
            for row in sorted(keep):
                out_beams += sorted(keep[row], key=lambda b: -b[2])[:beams]
            return out_beams

        first = cands.trie[()]
        lp0 = logp[:, first].cpu().numpy()
        active = settle([(r, (tok,), float(lp0[r, j]), r) for r in range(n) for j, tok in enumerate(first)])
        while active:
            parent = torch.tensor([b[3] for b in active], device=self.device)
            cache.reorder_cache(parent)
            att = torch.cat([att.index_select(0, parent), att.new_ones((len(active), 1))], dim=1)
            last_pos = last_pos.index_select(1, parent) + 1
            last = torch.tensor([[b[1][-1]] for b in active], device=self.device)
            out = self.model(input_ids=last, attention_mask=att, position_ids=last_pos, past_key_values=cache, use_cache=True)
            lp = torch.log_softmax(out.logits[:, -1].float(), dim=-1).cpu().numpy()
            active = settle([(row, toks + (tok,), sc + float(lp[i, tok]), i)
                             for i, (row, toks, sc, _) in enumerate(active) for tok in cands.trie[toks]])
        del cache, out
        return res, hidden

    @torch.inference_mode()
    def score_hf(self, samples: list[P.Sample], cands: CandidateSet, beams: int = 12) -> np.ndarray:
        """Cross-check via the library's beam search (re-reads the prompt per beam; slow)."""
        batch = self.encode(samples)
        self._reset_positions()
        n, prompt_len = batch["input_ids"].shape
        k = min(beams, len(cands))
        gen = GenerationConfig(
            max_new_tokens=cands.max_len, do_sample=False, num_beams=k, num_return_sequences=k, length_penalty=0.0,
            early_stopping=True, repetition_penalty=1.0, eos_token_id=cands.term_id, pad_token_id=self.pad_id,
            return_dict_in_generate=True, output_scores=True,
        )
        out = self.model.generate(
            **self._model_inputs(batch), generation_config=gen,
            logits_processor=LogitsProcessorList([_TrieProcessor(cands, prompt_len)]),
        )
        seqs = out.sequences[:, prompt_len:].view(n, k, -1).tolist()
        scores = out.sequences_scores.view(n, k).float().cpu().numpy()
        res = np.full((n, len(cands)), -np.inf, dtype=np.float64)
        for i in range(n):
            for j in range(k):
                s = seqs[i][j]
                if cands.term_id not in s:
                    continue
                idx = cands.index.get(tuple(s[: s.index(cands.term_id) + 1]))
                if idx is not None and math.isfinite(scores[i, j]):
                    res[i, idx] = max(res[i, idx], float(scores[i, j]))
        return res

    @torch.inference_mode()
    def score_bruteforce(self, sample: P.Sample, cands: CandidateSet) -> np.ndarray:
        """Reference for tests: every candidate sequence separately through the model (slow, exact)."""
        one = self._encode_one(sample)
        res = np.zeros(len(cands))
        for j, t in enumerate(cands.token_lists):
            full = self._append(one, t)
            ids = full["input_ids"][None].to(self.device)
            inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": full["mm_token_type_ids"][None].to(self.device)}
            if "pixel_values" in one:
                inp["pixel_values"] = one["pixel_values"].to(self.device, self.dtype)
                inp["image_grid_thw"] = one["image_grid_thw"].to(self.device)
            logits = self.model(**inp, logits_to_keep=len(t) + 1).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1)
            res[j] = float(sum(lp[i, tok] for i, tok in enumerate(t[:-1])))  # without terminator token
        return res


def probs_from_logp(logp: np.ndarray, floor: float = 1e-6, absent: np.ndarray | None = None) -> np.ndarray:
    """Row-normalised probabilities from the log-probabilities of `Engine.score`.

    Candidates that fell out of the best prefixes (-inf) share the remaining mass of the row
    (1 - sum of the returned ones), at least `floor` per candidate; a fixed tiny floor would distort blend
    weights. `absent` marks columns that were never a candidate (classes without a training example as root);
    if not given, columns that are finite in no row count as absent.
    """
    fin = np.isfinite(logp)
    if absent is None:
        absent = ~fin.any(axis=0) if len(logp) >= 20 else np.zeros(logp.shape[1], bool)
    p = np.where(fin, np.exp(np.where(fin, logp, 0.0)), 0.0)
    missing = ~fin & ~absent[None, :]
    n_missing = missing.sum(1, keepdims=True)
    rest = np.maximum(1.0 - p.sum(1, keepdims=True), floor * n_missing)
    # never above the weakest returned candidate: whatever fell out of the search ranks behind it
    low = np.where(fin, p, np.inf).min(1, keepdims=True)
    fill = np.minimum(rest / np.maximum(n_missing, 1), np.where(np.isfinite(low), low, 1.0))
    p = np.where(missing, fill, p)
    s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / logp.shape[1])
