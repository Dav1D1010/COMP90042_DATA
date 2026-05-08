"""
Building-block modules for the Climatron architecture.
ModernBERT-style bidirectional encoder components.

Design rationale summary:
  - RoPE: Relative positional encoding via complex rotation. Length-generalizable
    (no max position hard-cap), parameter-free, and efficient -- the dot product
    q_m * k_n = f(m-n) naturally encodes relative distances without learned embeddings.
  - GQA (Grouped-Query Attention, 3:1 ratio): Shares K,V heads across query groups.
    Reduces KV cache by 3x at <0.1% quality loss vs full multi-head.
    Critical for memory-constrained training (Colab T4).
  - RMSNorm: x / RMS(x) * weight -- same quality as LayerNorm but removes the
    mean-subtraction step, saving ~10% compute. No bias parameters needed.
  - SwiGLU FFN: SiLU(gate_proj(x)) * up_proj(x) -> down_proj. Gated activation
    with a learnable multiplicative gate -- outperforms ReLU/GELU in transformers.
  - Pre-norm (norm->sublayer->add): Better gradient flow through residual paths,
    enables stable training without warmup for moderate-depth models.
  - No dropout in pretraining: Modern practice -- mini-batch stochasticity + weight
    decay act as implicit regularizer. Dropout reserved for fine-tuning if needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.config import ModelConfig


# ============================================================================
# Embedding and Positional
# ============================================================================

class TokenEmbedding(nn.Module):
    """Scaled embedding lookup with optional extended vocabulary to accommodate
    special token IDs beyond vocab_size (e.g. mask_token_id in MLM pretraining).
    Scale = sqrt(d_model) prevents embeddings from dominating residual variance."""

    def __init__(self, vocab_size, d_model, extended_size=None):
        super().__init__()
        self.embedding = nn.Embedding(extended_size or vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale


class RotaryPositionalEmbedding(nn.Module):
    """RoPE encodes relative position via complex rotation of Q/K vectors.
    The inner product q_m * k_n becomes a function of (m-n).
    
    Why over alternatives:
      - Learned absolute: fails on sequences longer than training
      - Sinusoidal absolute: no relative-distance signal
      - AliBi: static bias, no learned flexibility
      - RoPE: relative + automatic length generalization via complex rotation"""

    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))

    def apply_rotary(self, xq, xk, start_pos=0):
        """Rotate query and key tensors by position-dependent frequencies.
        Args: xq,xk = [B, H, S, head_dim]; start_pos for KV-cache offset."""
        S = xq.shape[2]
        freqs = self.freqs_cis[start_pos:start_pos + S].to(xq.device)

        def rotate(x):
            xr = x.float().reshape(*x.shape[:-1], -1, 2)
            xc = torch.view_as_complex(xr)
            f = freqs.view([1] * (x.ndim - 2) + [S, self.head_dim // 2])
            return torch.view_as_real(xc * f).flatten(-2).type_as(x)

        return rotate(xq), rotate(xk)


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization: x / RMS(x) * weight.
    
    Why over LayerNorm: drops the mean-subtraction step (empirically unnecessary
    for transformer stability), saves ~10% compute with equal quality.
    No bias parameter -- fewer params, simpler gradient paths."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ============================================================================
# Attention and FFN
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """Multi-Head Attention with Grouped Query (GQA).
    
    Projects to Q (n_heads groups), K/V (n_kv_heads groups), then repeats K/V
    across n_groups = n_heads/n_kv_heads to match Q head count.
    
    GQA 3:1 ratio: KV cache reduced 3x vs full MHA, <0.1% quality loss.
    RoPE applied to Q,K before attention. Uses F.scaled_dot_product_attention
    for memory-efficient Flash Attention under the hood."""

    def __init__(self, config: "ModelConfig"):
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
        return self.W_o(out.transpose(1, 2).contiguous().view(B, S, D))


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(SiLU(gate_proj(x)) * up_proj(x)).
    
    Why over alternatives:
      - ReLU FFN: wastes capacity on dead neurons
      - GELU FFN: no learned gating
      - SwiGLU: multiplicative gate learns which dims to suppress/amplify
      - d_intermediate ~ (8/3)*d_model is the optimal expansion ratio"""

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


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> GQA -> add, RMSNorm -> SwiGLU -> add.
    
    Why pre-norm: gradient flows directly through residual path without passing
    through normalization, enabling stable training without LR warmup."""

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config.d_model, config.d_intermediate, config.dropout)

    def forward(self, x, causal=False):
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x
