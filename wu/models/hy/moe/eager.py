from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseHunyuanMoE, HunyuanMLP, resolve_layer_value


class HunyuanTopKGate(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.moe_topk = resolve_layer_value(config.moe_topk, layer_idx)
        num_experts = resolve_layer_value(config.num_experts, layer_idx)
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        gates = F.softmax(logits, dim=1)
        topk_weight, topk_idx = torch.topk(gates, self.moe_topk, dim=1)
        topk_weight = topk_weight / topk_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return topk_weight, topk_idx


class EagerExperts(nn.ModuleList):
    """Drop-in replacement for nn.ModuleList that adds forward() and pg_lora_forward().

    This allows PGMixin to wrap the experts module and route LoRA parameters
    in the torchtitan w1/w2/w3 format to individual eager experts.

    State-dict layout is identical to nn.ModuleList (e.g. ``experts.0.gate_and_up_proj.weight``).
    """

    def __init__(self, modules: List[nn.Module], gate: HunyuanTopKGate, num_experts: int):
        super().__init__(modules)
        self._num_experts = num_experts
        # Store gate reference WITHOUT registering as a child module so that
        # the gate parameters are not duplicated in state_dict.
        object.__setattr__(self, "_gate", gate)

    def _route(self, hidden_states: torch.Tensor):
        """Shared routing logic used by both forward and pg_lora_forward."""
        bsz, seq_len, dim = hidden_states.shape
        x = hidden_states.view(bsz * seq_len, dim)

        topk_weight, topk_idx = self._gate(hidden_states)
        k = topk_idx.shape[1]

        flat_expert_idx = topk_idx.reshape(-1)
        flat_weight = topk_weight.reshape(-1).to(x.dtype)
        token_idx = torch.arange(bsz * seq_len, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)

        _, perm = flat_expert_idx.sort(stable=True)
        sorted_token_idx = token_idx[perm]
        sorted_weight = flat_weight[perm]

        expert_counts = torch.bincount(flat_expert_idx, minlength=self._num_experts).tolist()
        sorted_x = x[sorted_token_idx]
        chunks = sorted_x.split(expert_counts)

        return x, bsz, seq_len, dim, sorted_token_idx, sorted_weight, chunks

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard eager expert forward with routing."""
        x, bsz, seq_len, dim, sorted_token_idx, sorted_weight, chunks = self._route(hidden_states)

        y = torch.cat(
            [self[i](chunk) if chunk.numel() > 0 else chunk for i, chunk in enumerate(chunks)],
            dim=0,
        )

        y = y * sorted_weight.unsqueeze(-1)
        out = x.new_zeros(bsz * seq_len, dim)
        out.scatter_add_(0, sorted_token_idx.unsqueeze(-1).expand_as(y), y)
        return out.view(bsz, seq_len, dim)

    def pg_lora_forward(
        self,
        hidden_states: torch.Tensor,
        lora_dict: dict,
        scale: float,
    ) -> torch.Tensor:
        """Forward with PG-generated LoRA (torchtitan w1/w2/w3 format) applied per expert.

        The base-weight layout of each eager expert's ``gate_and_up_proj`` is ``[w3, w1]``
        (first half = w3/up, second half = w1/gate), matching ``_copy_torchtitan_to_eager``.

        Args:
            hidden_states: (bsz, seq_len, dim)
            lora_dict: keys ``w1.lora_A``, ``w1.lora_B``, ``w2.lora_A``, ``w2.lora_B``,
                        ``w3.lora_A``, ``w3.lora_B``.  Each has shape ``(batch, E, ...)``.
            scale: LoRA scaling factor (alpha / rank).
        """
        x, bsz, seq_len, dim, sorted_token_idx, sorted_weight, chunks = self._route(hidden_states)

        # Extract and squeeze batch dimension (PG batch_size is always 1 at inference)
        lora_A_w1 = lora_dict["w1.lora_A"].squeeze(0)  # (E, R, D)
        lora_B_w1 = lora_dict["w1.lora_B"].squeeze(0)  # (E, H, R)
        lora_A_w2 = lora_dict["w2.lora_A"].squeeze(0)  # (E, R, H)
        lora_B_w2 = lora_dict["w2.lora_B"].squeeze(0)  # (E, D, R)
        lora_A_w3 = lora_dict["w3.lora_A"].squeeze(0)  # (E, R, D)
        lora_B_w3 = lora_dict["w3.lora_B"].squeeze(0)  # (E, H, R)

        out_splits = []
        for expert_idx, chunk in enumerate(chunks):
            if chunk.numel() == 0:
                out_splits.append(chunk)
                continue

            expert = self[expert_idx]

            # --- gate_and_up_proj + LoRA ---
            gate_and_up = expert.gate_and_up_proj(chunk)  # (N, 2*H)
            hidden_dim = gate_and_up.shape[-1] // 2

            # w3 (up) LoRA correction → first half of gate_and_up
            w3_corr = torch.matmul(
                torch.matmul(chunk, lora_A_w3[expert_idx].transpose(-2, -1)),
                lora_B_w3[expert_idx].transpose(-2, -1),
            )
            # w1 (gate) LoRA correction → second half of gate_and_up
            w1_corr = torch.matmul(
                torch.matmul(chunk, lora_A_w1[expert_idx].transpose(-2, -1)),
                lora_B_w1[expert_idx].transpose(-2, -1),
            )
            gate_and_up = torch.cat(
                [gate_and_up[..., :hidden_dim] + w3_corr * scale, gate_and_up[..., hidden_dim:] + w1_corr * scale],
                dim=-1,
            )

            # SwiGLU
            x1, x2 = gate_and_up.chunk(2, dim=-1)
            h = x1 * F.silu(x2)

            # --- down_proj + LoRA ---
            out_h = expert.down_proj(h)
            w2_corr = torch.matmul(
                torch.matmul(h, lora_A_w2[expert_idx].transpose(-2, -1)),
                lora_B_w2[expert_idx].transpose(-2, -1),
            )
            out_h = out_h + w2_corr * scale

            out_splits.append(out_h)

        y = torch.cat(out_splits, dim=0)
        y = y * sorted_weight.unsqueeze(-1)
        out = x.new_zeros(bsz * seq_len, dim)
        out.scatter_add_(0, sorted_token_idx.unsqueeze(-1).expand_as(y), y)
        return out.view(bsz, seq_len, dim)


class HunyuanMoEEager(BaseHunyuanMoE):
    def __init__(self, config, layer_idx: Optional[int]):
        super().__init__(config, layer_idx)
        self.shared_mlp = None
        if config.use_mixed_mlp_moe:
            self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True)
        self.gate = HunyuanTopKGate(config, layer_idx=layer_idx)
        self.experts = EagerExperts(
            [
                HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True)
                for _ in range(self.num_experts)
            ],
            gate=self.gate,
            num_experts=self.num_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_mlp(hidden_states) if self.shared_mlp is not None else None
        out = self.experts(hidden_states)
        if shared_out is not None:
            out = shared_out + out
        return out
