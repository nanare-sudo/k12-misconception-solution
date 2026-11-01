"""Partial transfer of an adapter (train.load_init_adapter): load adapters from one layer on into a model with adapters
on all layers; new adapters keep B = 0, architecture errors remain errors."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
peft = pytest.importorskip("peft")
from vlm.train import load_init_adapter  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"proj": torch.nn.Linear(8, 8)}) for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer["proj"](x)
        return x


def lora(pattern: str):
    torch.manual_seed(0)
    return peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, lora_alpha=4, target_modules=pattern))


def trained_upper_state():
    m = lora(r"layers\.1\.proj")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.add_(0.5)
    return {k: v.clone() for k, v in peft.get_peft_model_state_dict(m).items()}


def test_partial_loads_subset_and_keeps_new_adapters_inert():
    state = trained_upper_state()
    full = lora(r"layers\.[01]\.proj")
    loaded, fresh = load_init_adapter(full, state, True, peft.set_peft_model_state_dict)
    assert (loaded, fresh) == (2, 2)
    params = dict(full.named_parameters())
    b0 = next(v for n, v in params.items() if "layers.0" in n and "lora_B" in n)
    assert torch.count_nonzero(b0) == 0  # new adapters without effect at the start
    a1 = next(v for n, v in params.items() if "layers.1" in n and "lora_A" in n)
    want = next(v for k, v in state.items() if "layers.1" in k and "lora_A" in k)
    assert torch.equal(a1.detach(), want)


def test_exact_mode_rejects_subset():
    with pytest.raises(ValueError, match="architecture"):
        load_init_adapter(lora(r"layers\.[01]\.proj"), trained_upper_state(), False, peft.set_peft_model_state_dict)


def test_exact_mode_accepts_same_architecture():
    state = trained_upper_state()
    assert load_init_adapter(lora(r"layers\.1\.proj"), state, False, peft.set_peft_model_state_dict) == (2, 0)


def test_unknown_key_is_an_error():
    state = trained_upper_state()
    lower = lora(r"layers\.0\.proj")  # file carries layer 1, the model only layer 0
    with pytest.raises(ValueError):
        load_init_adapter(lower, state, True, peft.set_peft_model_state_dict)
