"""
Shared model components for all Climatron variants.
ModernBERT-style bidirectional encoder with configurable depth/width.
No DRW — uses LDAM + CB weights from epoch 1.
Label smoothing to prevent overconfidence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    d_intermediate: int = 1024   # (8/3)*d_model rounded to 256 multiple
    max_seq_len: int = 1024
    dropout: float = 0.0
    mask_token_id: int = 32768
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    bidirectional: bool = True
    pooling_type: str = "mean"

    def __post_init__(self):
        if self.d_intermediate is None:
            raw = (8 * self.d_model) // 3
            self.d_intermediate = ((raw + 255) // 256) * 256


# ═══════════════════════════════════════════════════════════════════════════
# Embedding & Positional
# ═══════════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, extended_size=None):
        super().__init__()
        self.embedding = nn.Embedding(extended_size or vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: q_m · k_n = f(m-n) — relative position in dot product."""
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))

    def apply_rotary(self, xq, xk, start_pos=0):
        S = xq.shape[2]
        freqs = self.freqs_cis[start_pos:start_pos+S].to(xq.device)
        def rotate(x):
            xr = x.float().reshape(*x.shape[:-1], -1, 2)
            xc = torch.view_as_complex(xr)
            f = freqs.view([1]*(x.ndim-2) + [S, self.head_dim//2])
            return torch.view_as_real(xc * f).flatten(-2).type_as(x)
        return rotate(xq), rotate(xk)


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# Attention & FFN
# ═══════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h, kv = config.d_model, config.n_heads, config.n_kv_heads
        self.head_dim = d // h
        self.n_groups = h // kv
        self.dropout = config.dropout
        self.W_q = nn.Linear(d, h * self.head_dim)
        self.W_k = nn.Linear(d, kv * self.head_dim)
        self.W_v = nn.Linear(d, kv * self.head_dim)
        self.W_o = nn.Linear(d, d)
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x, causal=False):
        B, S, D = x.shape
        q = self.W_q(x).view(B, S, -1, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, -1, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, -1, self.head_dim).transpose(1, 2)
        q, k = self.rope.apply_rotary(q, k)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0)
        return self.W_o(out.transpose(1,2).contiguous().view(B,S,D))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_intermediate, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_intermediate)
        self.up_proj = nn.Linear(d_model, d_intermediate)
        self.down_proj = nn.Linear(d_intermediate, d_model)
        self.dropout = dropout

    def forward(self, x):
        out = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)
        return self.down_proj(out)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block & Full Model
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config.d_model, config.d_intermediate, config.dropout)

    def forward(self, x, causal=False):
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class ClimatronForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ext_vocab = max(config.vocab_size, config.mask_token_id + 1)
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model, ext_vocab)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, ext_vocab)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, causal=not self.config.bidirectional)
        x = self.final_norm(x)
        return self.lm_head(x), {}


class ClimatronForClassification(nn.Module):
    def __init__(self, config, pretrained_model):
        super().__init__()
        self.config = config
        self.token_embedding = pretrained_model.token_embedding
        self.blocks = pretrained_model.blocks
        self.final_norm = pretrained_model.final_norm
        self.classification_head = nn.Linear(config.d_model, 4)

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, causal=False)
        x = self.final_norm(x)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.classification_head(x)


# ═══════════════════════════════════════════════════════════════════════════
# LoRA
# ═══════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(base.in_features, r, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(r, base.out_features, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.scale = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_dropout(x) @ self.A @ self.B


def apply_lora(model, r=8, alpha=16, dropout=0.1):
    targets = {"W_q", "W_k", "W_v", "W_o", "gate_proj", "up_proj", "down_proj"}
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(model, name, LoRALinear(child, r, alpha, dropout))
        else:
            apply_lora(child, r, alpha, dropout)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Class Imbalance Loss (NO DRW — fixed for stability)
# ═══════════════════════════════════════════════════════════════════════════

class StableImbalancedLoss(nn.Module):
    """LDAM margins + Class-Balanced weights from epoch 1.
    NO deferred re-weighting — avoids the epoch 2 degradation we observed.
    Adds label smoothing (ε=0.1) to prevent overconfidence on small classes."""
    
    def __init__(self, class_counts, ldam_margin=0.3, cb_beta=0.999, label_smoothing=0.1):
        super().__init__()
        self.margins = ldam_margin / (class_counts.float() ** 0.25)
        effective = (1 - cb_beta ** class_counts) / (1 - cb_beta)
        self.cb_weights = effective.sum() / (len(class_counts) * effective)
        self.label_smoothing = label_smoothing
        self.num_classes = len(class_counts)

    def forward(self, logits, targets):
        # LDAM: subtract margin from true class logit
        batch_margins = self.margins.to(logits.device)[targets]
        logits_adj = logits.clone()
        logits_adj.scatter_add_(1, targets.unsqueeze(1), -batch_margins.unsqueeze(1))
        
        # Label smoothing: soften targets from one-hot to (1-ε) one-hot + ε/K uniform
        smooth_targets = torch.full_like(logits_adj, self.label_smoothing / self.num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes)
        
        # CB-weighted cross-entropy with soft targets
        log_probs = F.log_softmax(logits_adj, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        weight = self.cb_weights.to(logits.device)[targets]
        return (weight * loss).mean()
