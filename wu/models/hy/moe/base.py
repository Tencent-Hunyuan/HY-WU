from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN


def resolve_layer_value(value, layer_idx: Optional[int]):
    if isinstance(value, int):
        return value
    if layer_idx is None:
        raise ValueError("layer_idx is required when config field is a list.")
    return value[layer_idx]


class BaseHunyuanMoE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = resolve_layer_value(config.num_experts, layer_idx)
        self.moe_topk = resolve_layer_value(config.moe_topk, layer_idx)
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}")
        if self.moe_topk <= 0:
            raise ValueError(f"moe_topk must be > 0, got {self.moe_topk}")
        if self.moe_topk > self.num_experts:
            raise ValueError(
                f"moe_topk ({self.moe_topk}) must be <= num_experts ({self.num_experts}) for layer {layer_idx}."
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HunyuanMLP(nn.Module):
    def __init__(self, config, layer_idx=None, is_shared_mlp=False, is_moe=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act

        self.intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            if config.moe_intermediate_size is not None:
                self.intermediate_size = resolve_layer_value(config.moe_intermediate_size, layer_idx)
            if is_shared_mlp:
                num_shared_expert = resolve_layer_value(config.num_shared_expert, layer_idx)
                self.intermediate_size *= num_shared_expert

        self.act_fn = ACT2FN[config.hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=config.mlp_bias)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        else:
            raise AssertionError("other hidden_act are not supported")

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=-1)
            return self.down_proj(x1 * self.act_fn(x2))
        if self.hidden_act == "gelu":
            intermediate = self.act_fn(self.gate_and_up_proj(x))
            return self.down_proj(intermediate)
        raise AssertionError("other hidden_act are not supported")
