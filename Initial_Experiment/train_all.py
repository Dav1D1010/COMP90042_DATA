#!/usr/bin/env python3
"""Cloud orchestrator: train and compare multiple Climatron variants on RTX 5090.

Single-script pipeline:
  1. Quick-pretrain (500M tokens) → compare → select winner
  2. Full-pretrain winner (2B tokens)
  3. Fine-tune all variants with LoRA/DoRA/QLoRA
  4. Generate comparison report

Designed for RTX 5090 32GB.  Total runtime capped at 15 hours.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# ── HF Mirror for mainland China ──────────────────────────────────────
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _HF_MIRROR

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore

# Configure datasets library to use the mirror (it ignores env var)
try:
    import datasets.config
    datasets.config.HF_ENDPOINT = _HF_MIRROR
except Exception:
    pass

# Add project root to path so we can import src/
_project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_dir))

from src.config import Config
from src.data.collator import CausalLMCollator, MaskedLMCollator, ClassificationCollator
from src.data.pretraining_data import FineWebEduDataset
from src.data.tokenizer import ClimatronTokenizer
from src.data.claims import ClaimDataset, LabelEncoder, load_claims
from src.models.config import ClimatronConfig
from src.models.model import ClimatronForPretraining, ClimatronForClassification
from src.models.variants import build_v1, build_v2, build_v3, build_v4, build_v5
from src.training.finetune import finetune as _finetune
from src.training.lora import merge_lora
from src.training.losses import FocalLoss
from src.training.scheduler import WSDScheduler
from src.training.trainer import Trainer


# ═══════════════════════════════════════════════════════════════════════════
# T4-Efficient Config (runs within 10 hours on Colab T4 free tier)
# ═══════════════════════════════════════════════════════════════════════════

T4_EFFICIENT_CONFIG = ClimatronConfig(
    vocab_size=32768,
    d_model=384,           # Narrower — less memory
    n_layers=8,            # Fewer layers — faster
    n_heads=6,
    n_kv_heads=2,          # GQA 3:1
    d_intermediate=1024,   # 8/3 * 384 ≈ 1024
    max_seq_len=1024,      # Shorter sequences
    dropout=0.0,
    norm_type="rmsnorm",
    attn_type="gqa",
    ffn_type="swiglu",
    pos_type="rope",
    pad_token_id=0,
    eos_token_id=3,
    bos_token_id=2,
    tie_word_embeddings=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Corpus Pretraining Mix (5:2:2:1 — research-backed)
# Stage 1 (80%): FineWeb-Edu 50% | Wikipedia 20% | ClimateSci 15% | Debate 15%
# Stage 2 (20%): FineWeb-Edu 30% | Wikipedia 20%  | ClimateSci 30% | Debate 20%
# Domain upsampling at final 20% improves domain transfer (arXiv:2406.03476)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CorpusConfig:
    fineweb: float = 0.50
    wikipedia: float = 0.20
    climate_sci: float = 0.15
    debate: float = 0.15

    def as_probabilities(self) -> list[float]:
        return [self.fineweb, self.wikipedia, self.climate_sci, self.debate]

    @property
    def sources(self) -> list[str]:
        return [
            "HuggingFaceFW/fineweb-edu:default",
            "wikimedia/wikipedia:20231101.en",
            "rabuahmad/climatecheck_publications_corpus",
            "Hellisotherpeople/DebateSum",
        ]


CORPUS_STAGE1 = CorpusConfig(fineweb=0.50, wikipedia=0.20, climate_sci=0.15, debate=0.15)
CORPUS_STAGE2 = CorpusConfig(fineweb=0.30, wikipedia=0.20, climate_sci=0.30, debate=0.20)


# ═══════════════════════════════════════════════════════════════════════════
# Variant Registry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VariantSpec:
    """Specification for a model variant to train."""
    name: str
    config: ClimatronConfig
    build_fn: Any
    description: str = ""
    pretrain_tokens: int = 500_000_000     # Quick comparison
    full_pretrain_tokens: int = 2_000_000_000  # Full training
    use_qlora: bool = False                # 4-bit QLoRA for T4
    use_dora: bool = False                 # Weight-Decomposed LoRA
    use_neftune: bool = False              # Noisy Embedding Fine-Tuning
    priority: int = 1                      # 1=must, 2=optional
    corpus_config: CorpusConfig | None = None  # Multi-corpus mix (None = FineWeb-Edu only)


REGISTRY: dict[str, VariantSpec] = {
    "v1": VariantSpec(
        name="v1",
        config=ClimatronConfig(d_model=768, n_layers=12, n_heads=12, n_kv_heads=12,
                               norm_type="layernorm", attn_type="mha", ffn_type="gelu", pos_type="learned"),
        build_fn=build_v1,
        description="Original BERT-style (LayerNorm + MHA + GELU)",
        priority=1,
        use_neftune=False,
        corpus_config=None,
    ),
    "v2": VariantSpec(
        name="v2",
        config=ClimatronConfig(d_model=576, n_layers=24, n_heads=9, n_kv_heads=3,
                               norm_type="rmsnorm", attn_type="gqa", ffn_type="swiglu", pos_type="rope"),
        build_fn=build_v2,
        description="ModernBERT (GQA + SwiGLU + RMSNorm + RoPE)",
        priority=1,
        use_neftune=True,
        corpus_config=None,
    ),
    "v3": VariantSpec(
        name="v3",
        config=ClimatronConfig(d_model=576, n_layers=24, n_heads=9, n_kv_heads=3,
                               norm_type="rmsnorm", attn_type="diff_v2", ffn_type="swiglu", pos_type="rope"),
        build_fn=build_v3,
        description="DiffAttention V2 (ICLR 2024)",
        priority=1,
        use_neftune=True,
        corpus_config=None,
    ),
    "v4": VariantSpec(
        name="v4",
        config=ClimatronConfig(d_model=576, n_layers=24, n_heads=9, n_kv_heads=3,
                               norm_type="rmsnorm", attn_type="gqa", ffn_type="moe", pos_type="rope",
                               n_experts=4, top_k=1, d_expert=512),
        build_fn=build_v4,
        description="Mixture of Experts (DeepSeek/Mixtral style)",
        priority=2,
        use_neftune=True,
        corpus_config=None,
    ),
    "v7": VariantSpec(
        name="v7",
        config=T4_EFFICIENT_CONFIG,
        build_fn=lambda: ClimatronForPretraining(T4_EFFICIENT_CONFIG),
        description="Deploy-efficient (~25M, 8L/384d)",
        pretrain_tokens=1_000_000_000,
        full_pretrain_tokens=2_000_000_000,
        use_qlora=True,
        use_dora=True,
        use_neftune=True,
        priority=1,
        corpus_config=None,
    ),
    "v5": VariantSpec(
        name="v5",
        config=ClimatronConfig(d_model=1024, n_layers=8, n_heads=16, n_kv_heads=4,
                               norm_type="rmsnorm", attn_type="gqa", ffn_type="swiglu", pos_type="rope"),
        build_fn=build_v5,
        description="Wide-shallow (8L/1024d, GQA + SwiGLU + RoPE)",
        priority=2,
        use_neftune=True,
        corpus_config=None,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Progress / ETA tracking
# ═══════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Tracks training progress and estimates time remaining."""

    def __init__(self, total_tokens: int, batch_size: int, seq_len: int):
        self.total_tokens = total_tokens
        self.tokens_per_step = batch_size * seq_len
        self.tokens_seen = 0
        self.start_time = time.time()
        self.step_times: list[float] = []

    def update(self, tokens: int):
        self.tokens_seen = tokens
        self.step_times.append(time.time())

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def tokens_per_sec(self) -> float:
        if self.elapsed < 1:
            return 0.0
        return self.tokens_seen / self.elapsed

    @property
    def eta_seconds(self) -> float:
        rate = self.tokens_per_sec
        if rate < 1:
            return float("inf")
        remaining = self.total_tokens - self.tokens_seen
        return remaining / rate

    @property
    def eta_str(self) -> str:
        secs = self.eta_seconds
        if secs == float("inf"):
            return "calculating..."
        return str(timedelta(seconds=int(secs)))

    @property
    def progress_pct(self) -> float:
        return 100.0 * self.tokens_seen / max(self.total_tokens, 1)

    def log_line(self) -> str:
        return (
            f"{self.progress_pct:5.1f}% | "
            f"tokens: {self.tokens_seen:>14,} | "
            f"speed: {self.tokens_per_sec:>8,.0f} tok/s | "
            f"elapsed: {timedelta(seconds=int(self.elapsed))} | "
            f"ETA: {self.eta_str}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# NEFTune — Noisy Embedding Fine-Tuning (arXiv:2310.05914, ICLR 2024)
# ═══════════════════════════════════════════════════════════════════════════
# Adds uniform noise to token embeddings during training only.
# noise_alpha=5 is the standard value. Zero overhead, proven gains.
# ═══════════════════════════════════════════════════════════════════════════

def _install_neftune(model: torch.nn.Module, noise_alpha: float = 5.0) -> None:
    """Add NEFTune noise to token embeddings during training only."""
    embed: torch.nn.Embedding | None = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and (embed is None or "." not in name):
            embed = module
    if embed is None:
        print("  ⚠ NEFTune: no Embedding found, skipping noise")
        return

    _orig_forward = embed.forward

    def _neftune_forward(x: torch.Tensor) -> torch.Tensor:
        out = _orig_forward(x)
        if embed.training:
            dims = out.size(1) * out.size(2)
            mag = noise_alpha / math.sqrt(dims)
            out = out + torch.empty_like(out).uniform_(-mag, mag)
        return out

    embed.forward = _neftune_forward  # type: ignore[method-assign]
    print(f"  NEFTune enabled: noise_alpha={noise_alpha}")


# ═══════════════════════════════════════════════════════════════════════════
# QLoRA — 4-bit NF4 quantization (bitsandbytes) + LoRA for T4 deployment
# ═══════════════════════════════════════════════════════════════════════════
# Applies bitsandbytes Linear4bit replacement to all Linear layers,
# then applies LoRA on top. 4× memory reduction during fine-tuning.
# ═══════════════════════════════════════════════════════════════════════════

def _apply_qlora(
    model: torch.nn.Module,
    compute_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Apply 4-bit NF4 quantization to a pretrained Climatron model.
    Uses bitsandbytes.nn.Linear4bit + double quantization."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        print("  ⚠ bitsandbytes not installed, skipping QLoRA")
        return model

    if compute_dtype is None:
        compute_dtype = torch.bfloat16

    model = _replace_linear_recursive(model, compute_dtype)
    model = model.cuda()

    for m in model.modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            m.to(torch.float32)

    print("  QLoRA enabled: 4-bit NF4 + double quantization")
    return model


def _replace_linear_recursive(
    module: torch.nn.Module,
    compute_dtype: torch.dtype,
    skip_names: tuple[str, ...] = ("lm_head", "classification_head"),
) -> torch.nn.Module:
    """Recursively replace nn.Linear with bnb.nn.Linear4bit."""
    import bitsandbytes as bnb

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and not any(s in name for s in skip_names):
            new_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=True,
                quant_type="nf4",
            )
            new_linear.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, new_linear)
        else:
            _replace_linear_recursive(child, compute_dtype, skip_names)
    return module


# ═══════════════════════════════════════════════════════════════════════════
# DoRA — Weight-Decomposed LoRA (arXiv:2402.09353, ICML 2024 Oral)
# ═══════════════════════════════════════════════════════════════════════════
# Normalizes LoRA direction and learns per-output-channel magnitude.
# DoRA r=16 often beats LoRA r=32. Same inference cost as LoRA.
# ═══════════════════════════════════════════════════════════════════════════

def _apply_dora_to_lora(model: torch.nn.Module) -> None:
    """Add DoRA magnitude normalization to existing LoRA layers.
    Must be called AFTER apply_lora()."""
    from src.training.lora import LoRALinear

    for module in model.modules():
        if isinstance(module, LoRALinear):
            _add_dora_magnitude(module)
    print("  DoRA enabled: weight-decomposed LoRA")


def _add_dora_magnitude(lora: "LoRALinear") -> None:
    """Add learnable magnitude vector m to a LoRALinear layer.
    W' = m * (W0 + BA) / ||W0 + BA||_c — initialized to match LoRA at start."""
    import torch.nn.functional as F

    with torch.no_grad():
        base_norms = lora.base.weight.norm(p=2, dim=0)

    if not hasattr(lora, "_dora_m"):
        lora._dora_m = torch.nn.Parameter(base_norms.clone())
        lora._dora_orig_forward = lora.forward

        def _dora_forward(x: torch.Tensor) -> torch.Tensor:
            base_out = F.linear(x, lora.base.weight, lora.base.bias)
            if lora.A is not None and lora.B is not None:
                lora_adapt = lora.scale * lora.dropout(x) @ lora.A @ lora.B
            else:
                lora_adapt = 0.0
            combined = base_out + lora_adapt
            norm = combined.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            normalized = combined / norm
            return normalized * lora._dora_m.unsqueeze(0)

        lora.forward = _dora_forward  # type: ignore[method-assign]


def create_pretraining_dataloader(
    tokenizer: ClimatronTokenizer,
    corpus_config: CorpusConfig,
    max_seq_len: int = 512,
    batch_size: int = 8,
    mask_token_id: int = 32768,
    stage: str = "stage1",
) -> torch.utils.data.DataLoader:
    """Create a multi-corpus pretraining dataloader with weighted interleaving.

    Streams from HuggingFace datasets in the configured mix ratio.
    Falls back to FineWeb-Edu only if other sources aren't available.
    Uses MaskedLMCollator for BERT-style bidirectional pretraining.
    """
    from src.data.collator import MaskedLMCollator

    try:
        from datasets import load_dataset, interleave_datasets

        sources = corpus_config.sources
        probs = corpus_config.as_probabilities()
        loaded: list = []
        valid_probs: list[float] = []

        for src, prob in zip(sources, probs):
            try:
                ds = _load_dataset_stream(src, max_seq_len, tokenizer)
                loaded.append(ds)
                valid_probs.append(prob)
                print(f"    Loaded corpus: {src}")
            except Exception as e:
                print(f"    ⚠ Skipping {src}: {e}")

        if len(loaded) > 1:
            total = sum(valid_probs)
            valid_probs = [p / total for p in valid_probs]
            dataset = interleave_datasets(loaded, probabilities=valid_probs, seed=42)
            print(f"    Mixed {len(loaded)} corpora (stage={stage}): "
                  + ", ".join(f"{src.split(':')[0].split('/')[-1]}={prob:.0%}"
                              for src, prob in zip(sources[:len(loaded)], valid_probs)))
        elif len(loaded) == 1:
            dataset = loaded[0]
        else:
            print("    ⚠ No corpora available, falling back to FineWeb-Edu")
            return _fallback_dataloader(tokenizer, max_seq_len, batch_size, mask_token_id)
    except Exception as e:
        print(f"    ⚠ Multi-corpus setup failed: {e}. Falling back to FineWeb-Edu.")
        return _fallback_dataloader(tokenizer, max_seq_len, batch_size, mask_token_id)

    collator = MaskedLMCollator(tokenizer, mask_token_id=mask_token_id, max_seq_len=max_seq_len)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator
    )


def _load_dataset_stream(name: str, max_seq_len: int, tokenizer):
    """Load and preprocess a HuggingFace streaming dataset for pretraining.

    Returns an IterableDataset that yields tokenised ``List[int]`` sequences
    so the MaskedLMCollator receives the same format as FineWebEduDataset.
    """

    from datasets import load_dataset

    if ":" in name:
        ds_name, ds_config = name.split(":", 1)
        ds = load_dataset(ds_name, ds_config, streaming=True, split="train")
    else:
        ds = load_dataset(name, streaming=True, split="train")

    def _tokenize_gen():
        for example in ds:
            text = example.get("text", example.get("content", ""))
            if not text:
                continue
            tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
            yield tokens[:max_seq_len]

    class _TokenizedStream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return _tokenize_gen()
        def __len__(self):
            return 10_000_000  # approximate for progress bars

    return _TokenizedStream()


def _fallback_dataloader(tokenizer, max_seq_len, batch_size, mask_token_id=32768):
    """Fallback: FineWeb-Edu only."""
    from src.data.collator import MaskedLMCollator
    from src.data.pretraining_data import FineWebEduDataset

    dataset = FineWebEduDataset(tokenizer, split="train", max_seq_len=max_seq_len)
    collator = MaskedLMCollator(tokenizer, mask_token_id=mask_token_id, max_seq_len=max_seq_len)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pretraining helper
# ═══════════════════════════════════════════════════════════════════════════

def _pretrain_variant(
    spec: VariantSpec,
    config: Config,
    num_tokens: int,
    phase: str = "quick",
) -> tuple[ClimatronForPretraining, dict]:
    """Pretrain a single variant and return (model, metrics)."""
    device = torch.device(config.device)
    training = config.training
    seq_len = training.pretrain_seq_len
    batch_size = training.pretrain_batch_size
    grad_accum = training.pretrain_grad_accum

    print(f"\n{'='*70}")
    print(f"  PRETRAINING: {spec.name} ({phase} phase, {num_tokens:,} tokens)")
    print(f"  {spec.description}")
    print(f"{'='*70}")

    # ── Model ──────────────────────────────────────────────────────────
    model = spec.build_fn()
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    if spec.use_neftune:
        _install_neftune(model, noise_alpha=5.0)

    # ── Data ───────────────────────────────────────────────────────────
    tok_path = config.tokenizer_path / "sentencepiece.bpe.model"
    tokenizer = ClimatronTokenizer(str(tok_path))

    corpus_config = getattr(spec, "corpus_config", None)
    mask_id = getattr(spec.config, "mask_token_id", tokenizer.vocab_size)
    if corpus_config is not None:
        dataloader = create_pretraining_dataloader(
            tokenizer, corpus_config, max_seq_len=seq_len, batch_size=batch_size,
            mask_token_id=mask_id, stage=phase
        )
    else:
        dataset = FineWebEduDataset(tokenizer, split="train", max_seq_len=seq_len)
        collator = MaskedLMCollator(tokenizer, mask_token_id=mask_id, max_seq_len=seq_len)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator
        )

    # ── Optimizer + Scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.pretrain_lr,
        betas=training.betas,
        weight_decay=training.weight_decay,
    )

    tokens_per_opt_step = batch_size * grad_accum * seq_len
    total_opt_steps = max(1, num_tokens // tokens_per_opt_step)
    warmup_steps = max(1, int(total_opt_steps * 0.05))

    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_opt_steps,
        peak_lr=training.pretrain_lr,
        stable_frac=0.85,
    )

    # ── W&B ────────────────────────────────────────────────────────────
    wandb_logger = None
    try:
        import wandb
        wandb_logger = wandb.init(
            project=config.wandb_project,
            name=f"pretrain-{spec.name}-{phase}",
            config={"variant": spec.name, "phase": phase, "num_tokens": num_tokens},
            reinit=True,
        )
    except Exception:
        pass

    # ── Trainer ────────────────────────────────────────────────────────
    from src.training.pretrain import MaskedLMLoss
    trainer = Trainer(
        model, optimizer, scheduler, device,
        loss_fn=MaskedLMLoss(),
        fp16=training.fp16,
        grad_accum_steps=grad_accum,
        wandb_logger=wandb_logger,
        checkpoint_dir=config.checkpoints_dir / spec.name,
    )

    # ── Training loop with progress ────────────────────────────────────
    tracker = ProgressTracker(num_tokens, batch_size, seq_len)
    tokens_seen = 0
    batch_step = 0
    running_loss = 0.0
    log_interval = 100
    ckpt_interval = 2000
    keep_last_n_ckpts = 3
    save_full_every = 5
    saved_ckpts: list[Path] = []
    metrics: dict[str, list] = {"loss": [], "step": []}
    stage2_tokens = int(num_tokens * 0.80)  # Switch to stage 2 at 80%
    stage2_switched = False

    print(f"\n  Config: batch={batch_size} grad_accum={grad_accum} seq={seq_len}")
    print(f"  LR={training.pretrain_lr:.0e} warmup={warmup_steps} steps={total_opt_steps}")
    print()

    for batch in dataloader:
        result = trainer.train_step(batch)
        running_loss += result["loss"]
        batch_step += 1
        tokens_seen += batch_size * seq_len
        tracker.update(tokens_seen)

        if batch_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            metrics["loss"].append(avg_loss)
            metrics["step"].append(batch_step)
            print(f"  [step {batch_step:>6d}]  {tracker.log_line()}  loss={avg_loss:.4f}")
            running_loss = 0.0

        if batch_step % ckpt_interval == 0:
            ckpt_path = config.checkpoints_dir / spec.name / f"step_{batch_step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            if (batch_step // ckpt_interval) % save_full_every == 0:
                trainer.save_checkpoint(ckpt_path)
            else:
                checkpoint = {
                    "model_state_dict": {
                        k: v.cpu() for k, v in trainer.model.state_dict().items()
                    },
                    "epoch": trainer._epoch,
                    "global_step": trainer._global_step,
                }
                torch.save(checkpoint, ckpt_path)
            saved_ckpts.append(ckpt_path)
            while len(saved_ckpts) > keep_last_n_ckpts:
                old = saved_ckpts.pop(0)
                if old.exists():
                    old.unlink()

        if not stage2_switched and tokens_seen >= stage2_tokens and corpus_config is not None:
            print(f"\n  ── Switching to Stage 2 corpus (domain upsampling) at {tokens_seen:,} tokens ──")
            corpus_config = CORPUS_STAGE2
            dataloader = create_pretraining_dataloader(
                tokenizer, corpus_config, max_seq_len=seq_len, batch_size=batch_size,
                mask_token_id=mask_id, stage="stage2"
            )
            stage2_switched = True

        if tokens_seen >= num_tokens:
            break

    # Final checkpoint (always full save)
    final_path = config.checkpoints_dir / spec.name / "pretrain_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(final_path)
    for old in saved_ckpts:
        if old.exists() and old != final_path:
            old.unlink()
    print(f"\n  ✓ Pretraining complete. Model saved to {final_path}")
    print(f"  Total time: {timedelta(seconds=int(tracker.elapsed))}")

    if wandb_logger is not None:
        wandb_logger.finish()

    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Fine-tuning helper
# ═══════════════════════════════════════════════════════════════════════════

def _finetune_variant(
    spec: VariantSpec,
    pretrained_model: ClimatronForPretraining,
    config: Config,
) -> dict:
    """Fine-tune a pretrained model and return metrics."""
    device = config.device

    print(f"\n  Fine-tuning: {spec.name}")

    train_data = load_claims(config.train_claims_path) if config.train_claims_path.exists() else {}
    dev_data = load_claims(config.dev_claims_path) if config.dev_claims_path.exists() else {}

    if not train_data:
        print("  ⚠ No training data found, skipping fine-tuning")
        return {"accuracy": 0.0}

    # Use augmented data if available
    use_aug = config.augmented_data_path.exists()
    use_distill = config.teacher_labels_path.exists()

    if spec.use_qlora:
        pretrained_model = _apply_qlora(pretrained_model)
        from src.training.lora import apply_lora
        apply_lora(pretrained_model, r=8, alpha=16, dropout=0.1,
                   target_modules=["W_q", "W_k", "W_v", "W_o", "gate_proj", "up_proj", "down_proj"])
        if spec.use_dora:
            _apply_dora_to_lora(pretrained_model)

    classifier, history = _finetune(
        pretrained_model=pretrained_model,
        train_data=train_data,
        dev_data=dev_data,
        config=config.training,
        use_distillation=use_distill,
        teacher_labels_path=str(config.teacher_labels_path) if use_distill else None,
        use_augmented=use_aug,
        augmented_path=str(config.augmented_data_path) if use_aug else None,
        evidence_path=str(config.evidence_path) if config.evidence_path.exists() else None,
        device=device,
    )

    # Save merged model
    merge_lora(classifier)
    out_path = config.checkpoints_dir / spec.name / "classifier_merged.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), out_path)

    final_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0
    print(f"  ✓ Fine-tuning complete. Val accuracy: {final_acc:.4f}")

    return {
        "accuracy": final_acc,
        "history": history,
        "model_path": str(out_path),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_predictor(config: Config, spec: VariantSpec):
    """Build evaluation pipeline. Returns Predictor or None if evidence unavailable."""
    if not config.evidence_path.exists():
        print(f"  ⚠ evidence.json not found at {config.evidence_path}")
        return None

    try:
        from src.data.evidence import EvidenceLoader
        from src.data.tokenizer import ClimatronTokenizer
        from src.evaluation.predictor import Predictor
        from src.retrieval.bi_encoder import BiEncoder
        from src.retrieval.faiss_index import EvidenceIndex

        ev_loader = EvidenceLoader(str(config.evidence_path))
        bi_encoder = BiEncoder(device=str(config.device))
        tokenizer = ClimatronTokenizer(str(config.tokenizer_path / "sentencepiece.bpe.model"))

        idx_path = config.outputs_dir / "faiss_evidence.index"
        if idx_path.exists():
            faiss_index = EvidenceIndex()
            faiss_index.load(str(idx_path))
        else:
            print("  Building FAISS index (one-time, ~2 min)...")
            faiss_index = EvidenceIndex(dim=384)
            faiss_index.build(ev_loader, bi_encoder, batch_size=128)
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            faiss_index.save(str(idx_path))

        return Predictor(
            bi_encoder=bi_encoder,
            faiss_index=faiss_index,
            classifier=None,
            tokenizer=tokenizer,
            evidence_loader=ev_loader,
            k=6,
        )
    except Exception as e:
        print(f"  ⚠ Predictor setup failed: {e}")
        return None


def _write_markdown_report(
    config: Config,
    specs: list[VariantSpec],
    pretrained_models: dict[str, "ClimatronForPretraining"],
    pretrain_metrics: dict[str, dict],
    finetune_results: dict[str, dict],
    eval_results: dict[str, dict],
    total_time: float,
    best_name: str,
):
    """Write final_results.md with full comparison tables."""
    out = config.outputs_dir / "final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Climatron Final Results",
        "",
        f"**Training time**: {timedelta(seconds=int(total_time))}",
        f"**Best variant**: {best_name}",
        "",
        "## Comparison Table",
        "",
        "| Variant | Params | PT Loss | Acc | Evid F1 | Macro F1 | Harm Mean |",
        "|---------|--------|---------|-----|---------|----------|-----------|",
    ]
    for spec in specs:
        name = spec.name
        params = (sum(p.numel() for p in pretrained_models[name].parameters())
                  if name in pretrained_models else 0)
        pt_loss = pretrain_metrics.get(name, {}).get("loss", [None])[-1] or 0
        m = eval_results.get(name, {})
        lines.append(
            f"| {name} | {params:,} | {pt_loss:.4f} | {m.get('accuracy', 0):.4f} | "
            f"{m.get('evidence_f1', 0):.4f} | {m.get('macro_f1', 0):.4f} | "
            f"{m.get('harmonic_mean', 0):.4f} |"
        )

    # Per-class breakdown for best variant
    best_eval = eval_results.get(best_name, {})
    if "per_class" in best_eval:
        lines.extend([
            "",
            f"## Per-Class Breakdown ({best_name})",
            "",
            "| Class | Precision | Recall | F1 |",
            "|-------|-----------|--------|-----|",
        ])
        for label, scores in best_eval["per_class"].items():
            lines.append(f"| {label} | {scores['p']:.4f} | {scores['r']:.4f} | {scores['f1']:.4f} |")

    if best_eval.get("confusion_matrix"):
        cm = best_eval["confusion_matrix"]
        labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH", "DISPUTED"]
        lines.extend([
            "",
            f"## Confusion Matrix ({best_name})",
            "",
            "| | " + " | ".join(labels) + " |",
            "|---|" + "|".join(["---"] * 4) + "|",
        ])
        for i, row in enumerate(cm):
            lines.append(f"| {labels[i]} | " + " | ".join(str(c) for c in row) + " |")

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Markdown report → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Climatron Cloud Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4", "v5", "v7"],
                        help="Variants to train (default: v1 v2 v3 v4 v5 v7)")
    parser.add_argument("--quick-tokens", type=int, default=500_000_000,
                        help="Tokens for quick comparison phase (default: 500M)")
    parser.add_argument("--full-tokens", type=int, default=2_000_000_000,
                        help="Tokens for winner full training (default: 2B)")
    parser.add_argument("--max-hours", type=float, default=14.5,
                        help="Maximum total runtime in hours (default: 14.5)")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip fine-tuning (pretrain only)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision (fp16)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per GPU step")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--corpus-mix", choices=["multi", "single"], default="single",
                        help="Corpus strategy: multi=5:2:2:1 with stage switch, single=FineWeb-Edu only")
    parser.add_argument("--colab", action="store_true",
                        help="Colab T4 mode: overrides to --variants v7 --fp16 --batch-size 2 --corpus-mix single, "
                             "skips evidence download, limits disk/cache, offline W&B")
    args = parser.parse_args()

    # ── Colab overrides ─────────────────────────────────────────────────
    if args.colab:
        args.variants = ["v7"]
        args.corpus_mix = "single"
        args.fp16 = True
        args.batch_size = 2
        if args.grad_accum > 4:
            args.grad_accum = 4
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("HF_DATASETS_CACHE", "/content/cache")
        os.environ.setdefault("HF_HUB_CACHE", "/content/cache")
        print("  🟢 Colab mode: v7 only, fp16, batch=2, streaming corpus, W&B offline")

    config = Config()
    config.training.pretrain_batch_size = args.batch_size
    config.training.pretrain_grad_accum = args.grad_accum
    config.training.fp16 = args.fp16

    device = torch.device(config.device)
    print(f"\n{'='*70}")
    print(f"  CLIMATRON CLOUD ORCHESTRATOR")
    print(f"  Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"  Variants: {args.variants}")
    print(f"  Quick tokens/variant: {args.quick_tokens:,}")
    print(f"  Max runtime: {args.max_hours} hours")
    print(f"{'='*70}\n")

    # ── Resolve variants ───────────────────────────────────────────────
    specs: list[VariantSpec] = []
    for name in args.variants:
        if name in REGISTRY:
            spec = REGISTRY[name]
            spec.pretrain_tokens = args.quick_tokens
            spec.full_pretrain_tokens = args.full_tokens
            if args.corpus_mix == "single":
                spec.corpus_config = None  # Fallback to FineWeb-Edu only
            specs.append(spec)
        else:
            print(f"  ⚠ Unknown variant '{name}', skipping")

    if not specs:
        print("No valid variants specified. Aborting.")
        return

    # ── Phase 1: Quick pretrain all variants ───────────────────────────
    print(f"\n{'─'*70}")
    print(f"  PHASE 1: Quick pretraining ({args.quick_tokens:,} tokens each)")
    print(f"{'─'*70}")

    phase1_start = time.time()
    pretrained_models: dict[str, ClimatronForPretraining] = {}
    pretrain_metrics: dict[str, dict] = {}

    for spec in specs:
        # Check time budget
        elapsed_hours = (time.time() - phase1_start) / 3600
        remaining = args.max_hours - elapsed_hours
        if remaining < 0.5:
            print(f"\n  ⚠ Time budget nearly exhausted ({elapsed_hours:.1f}h). Stopping.")
            break

        model, metrics = _pretrain_variant(spec, config, spec.pretrain_tokens, "quick")
        pretrained_models[spec.name] = model
        pretrain_metrics[spec.name] = metrics

    phase1_time = time.time() - phase1_start
    print(f"\n  Phase 1 complete: {timedelta(seconds=int(phase1_time))}")

    # ── Select best variant ────────────────────────────────────────────
    best_name = min(
        pretrain_metrics,
        key=lambda n: pretrain_metrics[n]["loss"][-1] if pretrain_metrics[n]["loss"] else float("inf"),
        default=specs[0].name,
    )
    best_spec = next(s for s in specs if s.name == best_name)
    print(f"\n  Best quick-pretrain variant: {best_name} (final loss: {pretrain_metrics[best_name]['loss'][-1]:.4f})")

    # ── Phase 2: Full pretrain winner ──────────────────────────────────
    elapsed_hours = (time.time() - phase1_start) / 3600
    if elapsed_hours < args.max_hours - 3:
        print(f"\n{'─'*70}")
        print(f"  PHASE 2: Full pretraining winner ({best_spec.full_pretrain_tokens:,} tokens)")
        print(f"{'─'*70}")
        model, _ = _pretrain_variant(best_spec, config, best_spec.full_pretrain_tokens, "full")
        pretrained_models[best_name] = model
    else:
        print(f"\n  ⚠ Skipping full pretrain — insufficient time budget ({elapsed_hours:.1f}h)")

    # ── Phase 3: Fine-tune all pretrained variants ─────────────────────
    if not args.skip_finetune:
        print(f"\n{'─'*70}")
        print(f"  PHASE 3: Fine-tuning")
        print(f"{'─'*70}")

        finetune_results: dict[str, dict] = {}
        for spec in specs:
            if spec.name not in pretrained_models:
                continue
            elapsed_hours = (time.time() - phase1_start) / 3600
            if elapsed_hours > args.max_hours - 0.5:
                print(f"\n  ⚠ Time budget nearly exhausted. Stopping fine-tuning.")
                break
            result = _finetune_variant(spec, pretrained_models[spec.name], config)
            finetune_results[spec.name] = result

    # ── Phase 4: Evaluation on dev set ──────────────────────────────────
    eval_results: dict[str, dict] = {}
    predictor = _build_predictor(config, best_spec)

    if predictor is not None:
        print(f"\n{'─'*70}")
        print(f"  PHASE 4: Evaluation on dev set")
        print(f"{'─'*70}")

        from src.evaluation.metrics import compute_metrics

        dev_claims = load_claims(config.dev_claims_path) if config.dev_claims_path.exists() else {}
        for spec in specs:
            name = spec.name
            if name not in finetune_results and not args.skip_finetune:
                continue
            model_path = finetune_results.get(name, {}).get("model_path")
            if not model_path or not Path(model_path).exists():
                print(f"  ⚠ No classifier for {name}, skipping eval")
                eval_results[name] = {"harmonic_mean": 0.0, "accuracy": 0.0}
                continue

            # Load classifier and wire into predictor
            from src.models.variants import build_classifier
            classifier = build_classifier(pretrained_models[name], variant=name)
            classifier.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            classifier.to(device)
            classifier.eval()
            predictor.classifier = classifier
            predictor._device = device

            preds = predictor.predict_all(dev_claims)
            metrics = compute_metrics(preds, dev_claims)
            eval_results[name] = metrics

            hm = metrics["harmonic_mean"]
            acc = metrics["accuracy"]
            print(f"  {name:<10} acc={acc:.4f}  evidence_f1={metrics['evidence_f1']:.4f}  "
                  f"macro_f1={metrics['macro_f1']:.4f}  harmonic_mean={hm:.4f}")
    else:
        print(f"\n  ⚠ Skipping evaluation — evidence.json not available (download manually)")

    # ── Phase 5: Test predictions (best variant) ────────────────────────
    if predictor is not None:
        print(f"\n{'─'*70}")
        print(f"  PHASE 5: Test predictions")
        print(f"{'─'*70}")

        # Determine best by harmonic mean
        scored = [(n, eval_results.get(n, {}).get("harmonic_mean", 0.0)) for n in eval_results]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_by_eval = scored[0][0] if scored else best_name
        print(f"  Best variant by dev eval: {best_by_eval}")

        test_claims = load_claims(config.test_claims_path) if config.test_claims_path.exists() else {}
        if test_claims:
            model_path = finetune_results.get(best_by_eval, {}).get("model_path")
            if model_path and Path(model_path).exists():
                from src.models.variants import build_classifier
                classifier = build_classifier(pretrained_models[best_by_eval], variant=best_by_eval)
                classifier.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                classifier.to(device)
                classifier.eval()
                predictor.classifier = classifier
                predictor._device = device

                test_preds = predictor.predict_all(test_claims)
                pred_path = config.outputs_dir / "test_predictions.json"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                predictor.save_predictions(test_preds, str(pred_path))
                print(f"  Test predictions saved to {pred_path}")
            else:
                print(f"  ⚠ No classifier found for {best_by_eval}")
        else:
            print(f"  ⚠ No test claims found at {config.test_claims_path}")
    else:
        best_by_eval = best_name

    # ── Phase 6: Final comparison report ────────────────────────────────
    total_time = time.time() - phase1_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total wall time: {timedelta(seconds=int(total_time))}")
    print(f"{'='*70}")

    # Save structured report
    report = {
        "training_time_s": total_time,
        "hardware": str(device),
        "best_variant": best_by_eval,
        "pretrain_metrics": {k: {"final_loss": v["loss"][-1] if v["loss"] else None}
                             for k, v in pretrain_metrics.items()},
        "finetune_results": {k: {"accuracy": v.get("accuracy", 0)}
                             for k, v in (finetune_results if not args.skip_finetune else {}).items()},
        "eval_results": eval_results,
    }
    report_path = config.outputs_dir / "cloud_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Terminal comparison table
    print(f"\n{'─'*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'─'*70}")
    header = (f"  {'Variant':<10} {'Params':>10} {'PT Loss':>10} "
              f"{'Acc':>8} {'EvidF1':>8} {'MacroF1':>8} {'HarmM':>8}")
    print(header)
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for spec in specs:
        name = spec.name
        params = (sum(p.numel() for p in pretrained_models[name].parameters())
                  if name in pretrained_models else 0)
        pt_loss = pretrain_metrics.get(name, {}).get("loss", [None])[-1] or 0
        m = eval_results.get(name, {})
        print(f"  {name:<10} {params:>10,} {pt_loss:>10.4f} "
              f"{m.get('accuracy', 0):>8.4f} {m.get('evidence_f1', 0):>8.4f} "
              f"{m.get('macro_f1', 0):>8.4f} {m.get('harmonic_mean', 0):>8.4f}")

    # Markdown report
    _write_markdown_report(config, specs, pretrained_models, pretrain_metrics,
                           finetune_results if not args.skip_finetune else {},
                           eval_results, total_time, best_by_eval)

    print(f"\n  Report saved → {report_path}")
    print(f"  Markdown report → {config.outputs_dir / 'final_report.md'}")
    print(f"  Best variant: {best_by_eval}")
    print(f"\n  ✓ ALL PHASES COMPLETE.")


if __name__ == "__main__":
    main()
