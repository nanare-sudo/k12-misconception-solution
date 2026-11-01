"""LoRA fine-tuning of a vision-language model on the diagnosis tasks.

Loss per sample: weighted sum of the mean cross-entropy over three groups of target positions: the root code,
the final-answer check (yes/no) and auxiliary targets (transcription, findings). The groups are averaged
separately so that the few code tokens are not drowned by the hundreds of transcription tokens.

Memory: the output layer (248k vocabulary) is evaluated only at the target positions, not over the whole prompt.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

import copy

from .engine import LORA_TARGETS, LOSS_TAGS, Engine, free_gpu_memory


@dataclass
class TrainConfig:
    epochs: float = 3.0
    lr: float = 1e-4
    warmup_frac: float = 0.05
    accum: int = 8
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_vision: bool = False
    w_code: float = 1.0
    w_fac: float = 0.5
    w_aux: float = 0.5
    w_why: float = 1.0  # description of the root error before the code (task why)
    # Loss of the description: "first+mean" = first token at full weight, the remaining tokens averaged. The first token
    # decides between "none" and a description; with a plain mean ("mean") it weighed thirty times as much on NONE rows
    # (one token) as on error rows, and the model wrote "none" for 56 % of the error rows.
    why_loss: str = "first+mean"
    grad_clip: float = 1.0
    grad_checkpointing: bool = True
    seed: int = 42
    tasks: tuple = ("diag", "full")  # alternating per sample and epoch
    max_steps: int | None = None  # for trial runs only
    lora_from_layer: int = 0  # adapters only from this layer on; below it the model runs without gradient
    frozen_prefix: bool = True  # compute the shared prompt header once with the base model (engine._static_state)
    # Loss of the code tokens per sample: "sum" = -log P(code); every sample counts equally regardless of how many tokens
    # its code has. "mean" averages over the tokens: NONE (one token) then weighed four to five times as much as an
    # error code (four to five tokens, three of them trivial) and pulled the model towards NONE.
    code_loss: str = "sum"
    log_every: int = 20
    targets: tuple = field(default_factory=lambda: LORA_TARGETS)
    # Initial adapter weights from an existing adapter folder with the SAME architecture (rank, layers, targets), for a
    # second stage after pre-training. Empty = fresh adapters.
    init_adapter: str = ""
    # Only with init_adapter: the file may carry a SUBSET of the adapters (e.g. adapters from layer 40 on, new model with
    # adapters on all layers). Missing adapters keep their initial values (LoRA: B = 0, i.e. initially without effect);
    # unknown keys remain an error. Default off: the architecture must match exactly.
    init_adapter_partial: bool = False
    # Own notes as training target (task proto_sft, marker notes): weight of the loss averaged over the note tokens.
    w_notes: float = 0.5
    # Compute the header (frozen_prefix) WITH the current adapter, without gradient, again after every optimiser step,
    # the way a merged model computes it in vLLM and when scoring (proto_sft). Off: header from the base model.
    prefix_with_adapter: bool = False
    # "Read first" variant (marker read, vlm/sft.py): weight of the loss averaged over the transcription tokens. Training
    # targets without the read marker are unaffected; the ctx marker never enters the loss.
    w_read: float = 0.5


def add_lora(engine: Engine, cfg: TrainConfig):
    """Attach LoRA adapters to the language model (from `lora_from_layer` on) and optionally to the vision tower."""
    from peft import LoraConfig, get_peft_model

    n_layers = len(engine.model.model.language_model.layers)
    first = max(0, min(int(cfg.lora_from_layer), n_layers - 1))
    if cfg.lora_vision and first > 0:
        raise ValueError("lora_vision requires lora_from_layer=0 (no gradient flows below the adapters)")
    names = "|".join(cfg.targets)
    layer_ids = "|".join(str(i) for i in range(first, n_layers))
    pattern = rf".*model\.language_model\.layers\.({layer_ids})\..*\.({names})"
    if cfg.lora_vision:
        pattern = rf"({pattern})|(.*model\.visual\..*\.(qkv|proj|linear_fc1|linear_fc2))"
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=pattern, bias="none")
    # Deliberately without peft.prepare_model_for_kbit_training: it casts all non-quantised weights (embedding with
    # 248k entries, vision tower) to fp32 and overflows a 15 GB card.
    for p_ in engine.model.parameters():
        p_.requires_grad_(False)
    engine.model = get_peft_model(engine.model, lcfg)
    if cfg.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(str(Path(cfg.init_adapter) / "adapter_model.safetensors"))
        loaded, fresh = load_init_adapter(engine.model, state, cfg.init_adapter_partial, set_peft_model_state_dict)
        print(f"init_adapter: {loaded} tensors loaded from {cfg.init_adapter}"
              + (f", {fresh} adapter tensors with initial values (partial transfer)" if fresh else ""), flush=True)
    _wrap_layers(engine, first, cfg.grad_checkpointing, no_grad_below=not cfg.lora_vision)
    return engine.model


def load_init_adapter(model, state: dict, partial: bool, setter) -> tuple[int, int]:
    """Load initial adapter weights. -> (loaded tensors, adapter tensors with initial values).

    Without `partial` the file must carry exactly as many adapter tensors as the model (same architecture). With
    `partial` it may carry a subset; every key of the file must exist in the model and all tensors must match in
    shape. Checked by comparing the values after loading (the library does not reliably report missing keys for
    adapters)."""
    import torch

    names = [n_ for n_, _ in model.named_parameters() if "lora_" in n_]
    res = setter(model, state)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise ValueError(f"init_adapter: {len(unexpected)} unknown keys, e.g. {unexpected[:2]}")
    if not partial and len(state) != len(names):
        raise ValueError(f"init_adapter does not match the architecture: {len(state)} tensors in the file, {len(names)} expected, "
                         f"0 unknown; rank, layers and targets must agree (or use init_adapter_partial)")
    if len(state) > len(names):
        raise ValueError(f"init_adapter: file carries {len(state)} tensors, the model only {len(names)}")
    # How many adapter tensors take their value from the file? (A B tensor with initial value 0 can stay unchanged if
    # the file also holds 0 there; therefore counting goes over the keys of the file, matched against the model names.)
    def norm(k: str) -> str:
        return k.replace(".default", "")
    model_keys = {norm(n_): n_ for n_ in names}
    hit = [k for k in state if norm(k) in model_keys]
    if len(hit) != len(state):
        raise ValueError(f"init_adapter: {len(state) - len(hit)} keys of the file without a matching adapter in the model")
    params = dict(model.named_parameters())
    for k in hit:
        got = params[model_keys[norm(k)]]
        if tuple(got.shape) != tuple(state[k].shape) or not torch.equal(got.detach().to(state[k].dtype).cpu(), state[k].cpu()):
            raise ValueError(f"init_adapter: tensor {k} not transferred")
    return len(hit), len(names) - len(hit)


_FROZEN_TYPES: dict[type, type] = {}


def _frozen_view(layer_cache):
    """View on the cache entry of a layer that never changes the state of the prompt header.

    The library updates the states of the linear layers in place; during training this destroys values the
    backward pass still needs. The view only returns the new states. Full-attention entries append anyway
    (new tensors); a shallow copy suffices there.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    view = copy.copy(layer_cache)
    if isinstance(layer_cache, LinearAttentionCacheLayerMixin):
        base = type(layer_cache)
        if base not in _FROZEN_TYPES:
            def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
                return recurrent_states

            def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
                return torch.cat([self.conv_states[state_idx], conv_states], dim=-1)

            _FROZEN_TYPES[base] = type(f"Frozen{base.__name__}", (base,), {
                "update_recurrent_state": update_recurrent_state, "update_conv_state": update_conv_state})
        view.__class__ = _FROZEN_TYPES[base]
    return view


def _wrap_layers(engine: Engine, first: int, checkpointing: bool, no_grad_below: bool) -> None:
    """Own wrappers around the language-model layers (replace the library's checkpointing).

    * Layers below `first` carry no adapters: they run without gradient.
    * Layers from `first` on run under torch.utils.checkpoint during training. Before every execution, including the
      recomputation in the backward pass, their cache entry is reset to the state of the shared prompt header.
      The library's checkpointing cannot do this: it discards the cache, because the layers would update it a
      second time during the second pass.
    """
    from torch.utils.checkpoint import checkpoint

    engine._train_ckpt = bool(checkpointing)
    layers = _core(engine.model).model.language_model.layers
    for i, layer in enumerate(layers):
        if getattr(layer, "_k12_wrapped", False):
            continue
        orig = layer.forward

        def wrapped(hidden_states, *args, _i=i, _orig=orig, _layer=layer, **kwargs):
            def run(h):
                cache = kwargs.get("past_key_values")
                if _layer.training and cache is not None and engine._train_pristine is not None:
                    cache.layers[_i] = _frozen_view(engine._train_pristine.layers[_i])
                return _orig(h, *args, **kwargs)

            if _i < first and no_grad_below:
                with torch.no_grad():
                    return run(hidden_states)
            if engine._train_ckpt and _layer.training and torch.is_grad_enabled():
                return checkpoint(run, hidden_states, use_reentrant=False)
            return run(hidden_states)

        layer.forward = wrapped
        layer._k12_wrapped = True


def _core(model):
    """The base model behind a PEFT wrapper (with attached adapters)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def sample_loss(engine: Engine, batch: dict, cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    """Loss of one sample; evaluates the output layer only at the target positions."""
    core = _core(engine.model)
    keys = ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")
    inputs = {k: batch[k] for k in keys if k in batch}
    pos_kw = {k: batch[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw") if k in batch}
    with torch.no_grad():
        pos = core._prepare_position_ids_for_generation(batch["input_ids"], pos_kw)
    k0 = int(batch.get("n_static", 0)) if cfg.frozen_prefix else 0
    if k0 >= engine.min_static:  # header from the fixed cache of the base model, only the rest is computed
        engine._train_pristine = None
        if cfg.prefix_with_adapter:  # header with the current adapter (without gradient), as in the merged model
            pristine = engine._static_state(batch["input_ids"][0, :k0].cpu(), copy_state=False, adapter=True)
            # While computing the header, autocast puts bf16 copies of the fp32 adapter weights into its cache, under no_grad,
            # i.e. without grad_fn. The rest in the same autocast region would reuse these copies and the loss would depend on no
            # trainable parameter ("element 0 of tensors does not require grad"). Clear it.
            torch.clear_autocast_cache()
        else:
            pristine = engine._static_state(batch["input_ids"][0, :k0].cpu(), copy_state=False)
        engine._train_pristine = pristine
        rest = {k: (v[:, k0:] if k in ("input_ids", "mm_token_type_ids") else v) for k, v in inputs.items()}
        cache = copy.copy(pristine)
        cache.layers = [_frozen_view(layer) for layer in pristine.layers]
        out = core.model(**rest, position_ids=pos[..., k0:], past_key_values=cache, use_cache=True)
    else:
        k0 = 0
        engine._train_pristine = None
        out = core.model(**inputs, position_ids=pos, use_cache=False)
    hidden = out.last_hidden_state[0]
    n_prompt, n_tgt = batch["n_prompt"] - k0, len(batch["target_ids"])
    h = hidden[n_prompt - 1 : n_prompt - 1 + n_tgt]  # position t predicts token t+1
    logits = core.lm_head(h).float()
    ce = torch.nn.functional.cross_entropy(logits, batch["target_ids"], reduction="none")
    tags = batch["target_tags"]
    parts, total = {}, logits.new_zeros(())
    for name, w in (("code", cfg.w_code), ("fac", cfg.w_fac), ("aux", cfg.w_aux), ("why", cfg.w_why), ("notes", cfg.w_notes), ("read", cfg.w_read)):
        m = tags == LOSS_TAGS[name]
        if m.any():
            if name == "code" and cfg.code_loss == "sum":
                parts[name] = ce[m].sum()
            elif name == "why" and cfg.why_loss == "first+mean":
                seg = ce[m]  # one contiguous segment: seg[0] is the first token of the description
                parts[name] = seg[0] + (seg[1:].mean() if len(seg) > 1 else 0.0)
            else:
                parts[name] = ce[m].mean()
            total = total + w * parts[name]
    return total, {k: float(v.detach()) for k, v in parts.items()}


def train(engine: Engine, make_sample, n_items: int, cfg: TrainConfig, out_dir: Path, budget_s: float | None = None,
          on_epoch_end=None) -> dict:
    """Train LoRA adapters. `make_sample(i, task)` returns the sample for position i with its training target.

    Writes the adapter to out_dir/adapter after every epoch and a log to out_dir/train_log.json.
    `on_epoch_end(epoch)` runs in evaluation mode (for probes on the validation fold: a real learning curve
    instead of an extrapolation). Samples that run out of memory are skipped and counted instead of ending the run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = add_lora(engine, cfg)
    model.train()
    engine._static_ids, engine._static_cache = None, None  # recompute the header with the base model
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_total = math.ceil(cfg.epochs * n_items)
    if cfg.max_steps:
        steps_total = min(steps_total, cfg.max_steps)
    updates_total = max(1, steps_total // cfg.accum)
    warm = max(1, int(cfg.warmup_frac * updates_total))

    def lr_at(u: int) -> float:
        if u < warm:
            return cfg.lr * (u + 1) / warm
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, updates_total - warm)))

    use_amp = engine.device.type == "cuda"
    amp_dtype = engine.dtype if engine.dtype != torch.float32 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    order: list[tuple[int, str]] = []
    epoch = 0
    while len(order) < steps_total:
        idx = list(range(n_items))
        random.shuffle(idx)
        order += [(i, cfg.tasks[(i + epoch) % len(cfg.tasks)]) for i in idx]
        epoch += 1
    order = order[:steps_total]

    log, t0, tok_seen, run = [], time.time(), 0, {}
    opt.zero_grad(set_to_none=True)
    update, skipped, nonfinite, pending = 0, 0, 0, 0  # pending: samples with gradient since the last step

    def epoch_end(ep: int) -> None:
        model.eval()
        engine._train_pristine = None
        model.save_pretrained(out_dir / "adapter")
        if on_epoch_end is not None:
            try:
                on_epoch_end(ep)
            except Exception as exc:  # noqa: BLE001 - a failed intermediate evaluation does not end the training
                print(f"train: intermediate evaluation after epoch {ep} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        model.train()

    for step, (i, task) in enumerate(order):
        # A failed sample (memory, non-finite loss) is counted; the step, log and epoch boundaries below still apply.
        oom = False
        try:
            sample = make_sample(i, task)
            batch = engine.encode_train(sample)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, parts = sample_loss(engine, batch, cfg)
            loss = loss * float(sample.meta.get("weight", 1.0))  # e.g. teacher rows of the pool weighted lower than labelled rows
            if torch.isfinite(loss):
                scaler.scale(loss / cfg.accum).backward()
                pending += 1
                tok_seen += batch["input_ids"].shape[1]
                for k, v in parts.items():
                    run.setdefault(k, []).append(v)
            else:
                nonfinite += 1
                if nonfinite > 20:
                    raise FloatingPointError(f"loss repeatedly not finite (last step {step}, sample {i})")
        except torch.OutOfMemoryError:
            oom = True
        if oom:  # clean up outside the except block: the live exception would keep the sample's tensors alive
            skipped += 1
            engine._train_pristine = None
            batch = loss = None
            free_gpu_memory()
            print(f"train: sample {i} skipped (memory), {skipped} so far", flush=True)
            if skipped > max(20, steps_total // 20):
                raise torch.OutOfMemoryError(f"too many samples run out of memory ({skipped})")
        if (step + 1) % cfg.accum == 0 or step + 1 == steps_total:
            for g in opt.param_groups:
                g["lr"] = lr_at(update)
            if pending:  # if every sample of the group failed, there is nothing to scale (GradScaler would abort otherwise)
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                if not torch.isfinite(norm):  # without GradScaler (bf16) there would otherwise be no protection against broken gradients
                    nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                scaler.step(opt)
                scaler.update()
                if cfg.prefix_with_adapter:  # new adapter weights: recompute the header that was computed with the adapter
                    engine._static_ids, engine._static_cache = None, None
            opt.zero_grad(set_to_none=True)
            update += 1
            pending = 0
        if (step + 1) % cfg.log_every == 0 or step + 1 == steps_total:
            el = time.time() - t0
            rec = {
                "step": step + 1, "of": steps_total, "elapsed_s": round(el, 1), "tok_per_s": round(tok_seen / el, 1),
                "s_per_step": round(el / (step + 1), 3), "lr": lr_at(max(0, update - 1)),
                **{f"loss_{k}": round(sum(v) / len(v), 4) for k, v in run.items()},
            }
            if use_amp:
                rec["gpu_gb"] = round(torch.cuda.max_memory_allocated(engine.device) / 2**30, 2)
            log.append(rec)
            print("train", json.dumps(rec), flush=True)
            run = {}
        if (step + 1) % n_items == 0 and step + 1 < steps_total:
            epoch_end((step + 1) // n_items)
        if budget_s is not None and time.time() - t0 > budget_s:
            print(f"train: time budget reached after {step + 1} of {steps_total} steps", flush=True)
            break
    model.eval()
    engine._train_pristine = None
    model.save_pretrained(out_dir / "adapter")
    summary = {
        "config": {**asdict(cfg), "targets": list(cfg.targets), "tasks": list(cfg.tasks)},
        "trainable_params": n_train, "steps_done": step + 1, "steps_total": steps_total, "seconds": round(time.time() - t0, 1),
        "skipped_oom": skipped, "nonfinite": nonfinite, "log": log,
    }
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=1))
    return summary
