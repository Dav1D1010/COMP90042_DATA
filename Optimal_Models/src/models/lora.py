"""
LoRA (Low-Rank Adaptation) for efficient fine-tuning of Climatron.

LoRALinear wraps a frozen nn.Linear with trainable low-rank matrices A and B:
    h = W_0 x + (α/r) · dropout(x @ A @ B)

The forward pass adds a rank-r correction to the frozen base weights. Only
A (in_features × r) and B (r × out_features) are trained — reducing trainable
params by orders of magnitude (e.g. from 384×384=147K to 384×8+8×384=6K per
linear layer).

apply_lora() replaces targeted Linear layers with LoRALinear wrappers.
Target set: {"W_q", "W_k", "W_v", "W_o", "gate_proj", "up_proj", "down_proj"}
covers all attention projections and SwiGLU gates — the full transformer core.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-rank adaptation wrapper for a frozen nn.Linear layer.

    Initialization:
      - A: Kaiming uniform (fan_in with gain √5) — ensures balanced gradient
        flow from the first forward pass.
      - B: zeros — so the LoRA correction starts at zero, preserving the
        pretrained model's outputs exactly at initialization.
      - scale = α/r: normalizes the correction so α controls total magnitude
        independently of the rank.

    Args:
        base: nn.Linear — the pretrained weight (frozen).
        r: int — rank of the low-rank decomposition.
        alpha: int — scaling factor for the LoRA correction.
        dropout: float — dropout rate on the LoRA path.
    """
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(base.in_features, r,
            device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(r, base.out_features,
            device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_dropout(x) @ self.A @ self.B


def apply_lora(model: nn.Module, r=8, alpha=16, dropout=0.1) -> nn.Module:
    """
    Recursively replace targeted Linear layers with LoRALinear wrappers.

    Target modules (by attribute name):
        W_q, W_k, W_v, W_o       — attention projections
        gate_proj, up_proj, down_proj — SwiGLU FFN gates

    These are the layers that learn task-specific adaptations; the embedding,
    norms, and lm_head / classification_head are left untouched.

    Mutation is in-place via setattr on the parent module.

    Args:
        model: nn.Module — the model (or submodule) to apply LoRA to.
        r: rank of the low-rank decomposition.
        alpha: scaling factor.
        dropout: dropout probability on the LoRA path.

    Returns:
        The same model object (mutated in-place).
    """
    targets = {"W_q", "W_k", "W_v", "W_o", "gate_proj", "up_proj", "down_proj"}
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(model, name, LoRALinear(child, r, alpha, dropout))
        else:
            apply_lora(child, r, alpha, dropout)
    return model
