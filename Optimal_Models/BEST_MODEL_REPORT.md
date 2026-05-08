# Climatron Optimal Model — Best Architecture Report

> v7: Deploy-Efficient Bidirectional Encoder (25M params, 8L/384d)
> **Winner** of 5-variant architecture comparison
> Test Accuracy: 42.21% | GPU: RTX 5090 | Designed for T4 deployment (11h budget)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Architecture Specification](#2-architecture-specification)
3. [Why v7 Beat All Larger Models](#3-why-v7-beat-all-larger-models)
4. [Component-by-Component Justification](#4-component-by-component-justification)
5. [Pretraining Design](#5-pretraining-design)
6. [Fine-Tuning Design](#6-fine-tuning-design)
7. [Class Imbalance Pipeline](#7-class-imbalance-pipeline)
8. [T4 Deployment Optimization](#8-t4-deployment-optimization)
9. [Colab Adaptation Strategy](#9-colab-adaptation-strategy)
10. [References & Influences](#10-references--influences)

---

## 1. Executive Summary

After testing 5 transformer variants (v1-v7), **v7** emerged as the clear winner with 42.21% accuracy — 12 percentage points above the next-best variant (v2 at 30.52%). v7 is a compact bidirectional encoder with 25M parameters, 8 transformer layers, 384-dimensional hidden states, Grouped Query Attention (GQA), SwiGLU feed-forward layers, RMSNorm, and Rotary Positional Embeddings (RoPE).

The key insight: **on limited pretraining data (150M tokens), smaller models outperform larger ones** due to Chinchilla scaling laws. The design is directly inspired by ModernBERT (Warner et al., 2024) with adaptations for extreme data efficiency.

---

## 2. Architecture Specification

### 2.1 Full Configuration

```
╔══════════════════════════════════════════════════════════════╗
║                   Climatron-v7 Architecture                  ║
╠══════════════════════════════════════════════════════════════╣
║ Vocabulary      │ 32,768 (+ [MASK] extension to 32,769)     ║
║ d_model         │ 384                                        ║
║ n_layers        │ 8                                          ║
║ n_heads (Query) │ 6                                          ║
║ n_kv_heads      │ 2 (3:1 GQA ratio)                         ║
║ d_head          │ 64 (384 / 6)                               ║
║ d_intermediate  │ 1,024 (8/3 × 384, rounded to 256-multiple) ║
║ max_seq_len     │ 1,024                                      ║
║ dropout         │ 0.0 (no dropout in pretraining — MLM is    ║
║                 │       already strong regularization)        ║
║ norm_type       │ RMSNorm                                    ║
║ attn_type       │ GQA (Grouped Query Attention)              ║
║ ffn_type        │ SwiGLU                                     ║
║ pos_type        │ RoPE (Rotary Positional Embedding)         ║
║ bidirectional   │ True (full context attention)              ║
║ pooling_type    │ Mean (average all non-pad token outputs)   ║
║ tie_embeddings  │ False (extended vocab for [MASK])          ║
║ total_params    │ ~25.6M                                     ║
╚══════════════════════════════════════════════════════════════╝
```

### 2.2 Architecture Diagram (Detailed)

```
                         ┌──────────────────────────┐
                         │    Input: token IDs       │
                         │  [BOS][claim][SEP][ev]... │
                         │    max_seq_len = 1024     │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   Token Embedding         │
                         │   nn.Embedding(32769,384) │
                         │   output × √384           │
                         │   → (B,S,384)             │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   RoPE Rotary Encoding    │
                         │   freq_i = 10000^(-2i/64) │
                         │   rotate Q and K vectors  │
                         │   (no extra parameters)   │
                         └────────────┬─────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────┐
        │                Pre-Norm Transformer Block (×8 layers) │
        │                                                       │
        │  ┌─────────────────────────────────────────────────┐  │
        │  │  Layer Structure (per block):                   │  │
        │  │                                                 │  │
        │  │  x ──→ RMSNorm ──→ GQA Attention ──→ (+) ──┐   │  │
        │  │       (384)        (6Q/2KV heads)          │   │  │
        │  │                              ┌─────────────┘   │  │
        │  │                              │                  │  │
        │  │  ┌───────────────────────────┘                  │  │
        │  │  │                                              │  │
        │  │  ├──→ RMSNorm ──→ SwiGLU FFN ──→ (+) ──→ out   │  │
        │  │       (384)        (384→1024→384)               │  │
        │  └─────────────────────────────────────────────────┘  │
        └─────────────────────────────┬─────────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   Final RMSNorm           │
                         │   → (B,S,384)             │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────┐
              │                                           │
    ┌─────────▼──────────┐                   ┌───────────▼──────────┐
    │  PRETRAINING:       │                   │  CLASSIFICATION:      │
    │  LM Head            │                   │  Mean Pooling         │
    │  Linear(384,32769)  │                   │  Σ(x_i · mask_i) /    │
    │  → token logits     │                   │  Σ mask_i             │
    │  CrossEntropy on    │                   │  → (B,384)            │
    │  masked positions   │                   │                       │
    └─────────────────────┘                   │  Classification Head  │
                                              │  Linear(384,4)        │
                                              │  → class logits       │
                                              └───────────────────────┘
```

### 2.3 Parameter Breakdown

| Component | Shape | Params | % |
|-----------|-------|--------|---|
| Token Embedding | 32769 × 384 | 12,583,296 | 49.1% |
| Per Attention Layer: | | | |
| └ W_q | 384 × 384 | 147,456 | |
| └ W_k | 384 × 128 | 49,152 | |
| └ W_v | 384 × 128 | 49,152 | |
| └ W_o | 384 × 384 | 147,456 | |
| └ Gate proj | 384 × 1024 | 393,216 | |
| └ Up proj | 384 × 1024 | 393,216 | |
| └ Down proj | 1024 × 384 | 393,216 | |
| Per Block Total | | 1,572,864 | |
| 8 Blocks × | | 12,582,912 | 49.1% |
| 2 RMSNorm/block | 8 × 2 × 384 | 6,144 | 0.0% |
| Final RMSNorm | 384 | 384 | 0.0% |
| LM Head | 384 × 32769 | 12,583,296 | — (tied) |
| Classification Head | 384 × 4 | 1,536 | 0.0% |
| **Total** | | **~25,636,100** | 100% |

The embedding dominates (49%) — this is typical for small models. With `tie_word_embeddings=False` (needed for [MASK] token extension), the LM head is separate during pretraining.

---

## 3. Why v7 Beat All Larger Models

### 3.1 The Chinchilla Scaling Law

Hoffmann et al. (NeurIPS 2022) established that for compute-optimal training, the ratio of training tokens to model parameters should be approximately **20:1**. That is, a 100M-param model needs ~2B tokens for optimal training.

Our experiment trained all models on **150M tokens**:

| Variant | Params | Tokens:Params | Undertraining Factor |
|---------|--------|---------------|---------------------|
| v7 | 25M | 6.0:1 | 3.3× |
| v2 | 104M | 1.44:1 | 13.9× |
| v3 | 97M | 1.55:1 | 12.9× |
| v4 | 125M | 1.20:1 | 16.7× |
| v1 | 112M | 1.34:1 | 14.9× |

**v7 is 4-5× closer to the optimal ratio than the larger models.** With 6× more training than "needed" relative to optimal, the small model's features are substantially richer than the large models' undertrained features.

### 3.2 Implicit Regularization

Smaller models have lower capacity, which naturally prevents overfitting. With 4249 augmented training examples:
- v7: 6.0 params per example → room to learn but not memorize
- v2: 24.9 params per example → capacity to memorize noise

### 3.3 LoRA Effectiveness

LoRA rank 8 represents a larger fraction of the total parameters for smaller models:
- v7: 434K / 25.6M = 1.69% — proportionally more adaptation capacity
- v1: 590K / 112M = 0.52% — proportionally less

The LoRA adaptation is proportionally 3× more impactful on v7.

### 3.4 Training Speed Advantage

v7 trains at 120K tok/s (vs 60-90K for larger models). In the same wall-clock time, v7 processes 33-100% more tokens, achieving more effective pretraining.

---

## 4. Component-by-Component Justification

### 4.1 Why d_model = 384 (Not 576 or 768)

**Chosen**: 384 (vs 576 for v2-v4, 768 for v1)

**Justification**: For a model with only 150M pretraining tokens, embedding dimension must be small enough to learn meaningful representations from limited data. 384 provides 64-dim attention heads (6 heads × 64 dim), which is sufficient for English text understanding. BERT-base uses 768 with 12 heads (64-dim each) — our head dimension matches BERT-base even though our model dimension is smaller.

**What we rejected**: 768 (too many parameters for limited data — would be 4× more undertrained than 384).

### 4.2 Why n_layers = 8 (Not 12 or 24)

**Chosen**: 8 layers (vs 12 for v1, 24 for v2-v4)

**Justification**: Each additional transformer layer adds 1.57M parameters and increases the effective depth of feature hierarchy. For fact-checking, the input is ~200-300 tokens of claim+evidence text—not enough tokens to need 24 layers of processing. 8 layers can capture:
- Layers 1-2: Local syntax, word-level patterns
- Layers 3-5: Phrase-level semantics, evidence-claim alignment
- Layers 6-8: Global contradiction/support detection

ModernBERT-base uses 22 layers but with 2T training tokens. For 150M tokens, depth must be proportional.

**What we rejected**: 24 layers (v2-v4). Too deep for limited data — the gradient signal attenuates through many layers, making early layers undertrained. The "shattered gradients" problem.

### 4.3 Why GQA with 3:1 Ratio (6Q/2KV)

**Chosen**: 6 query heads, 2 KV heads (3:1 GQA ratio)

**Justification**: 
- GQA reduces KV-cache memory by 3× vs full MHA (critical for T4 with 16GB)
- 6 query heads × 64 dim = 384 = d_model (clean division)
- 3:1 ratio is standard (LLaMA 2 uses 32:8 = 4:1, LLaMA 3 uses 32:8)

**What we rejected**: 
- Full MHA (v1: 12Q/12KV) — 6× more KV parameters, minimal accuracy gain
- DiffAttention (v3) — 2× compute for marginal improvement; underperformed in experiments

### 4.4 Why SwiGLU (Not GELU or MoE)

**Chosen**: SwiGLU with intermediate dim = 1024 (8/3 × 384, rounded to 256 multiple)

**Justification**:
- Gating mechanism: SiLU(gate(x)) ⊙ up(x) provides learned information filtering
- 1024 is the correct intermediate size: (8 × 384) / 3 = 1024, which is already a multiple of 256
- Parameter count: 394K × 3 projections = 1.18M per block for FFN
- Proven superior to GELU in LLaMA, PaLM, Chinchilla, ModernBERT

**What we rejected**:
- GELU (v1): No gating, less expressive
- MoE (v4): Requires learned router + 4×512 experts. Router was undertrained on 150M tokens

### 4.5 Why RMSNorm (Not LayerNorm)

**Chosen**: RMSNorm

**Justification**:
- 10-15% faster than LayerNorm (no mean subtraction, no bias)
- No learnable bias term → fewer parameters, less overfitting risk
- Empirically equivalent or better training stability
- Used by LLaMA, Mistral, ModernBERT, PaLM — all SOTA

**What we rejected**: LayerNorm (v1) — slower, more parameters, no accuracy benefit.

### 4.6 Why RoPE (Not Learned Positional)

**Chosen**: Rotary Positional Embedding (RoPE) with θ=10000

**Justification**:
- Encodes relative position in the dot product: q_m · k_n = f(m-n)
- Extrapolates to any sequence length (learned embeddings are fixed)
- No additional parameters
- Applied only to Q and K (not V) — computationally efficient
- Used by LLaMA, Mistral, ModernBERT, Gemma, Phi

**What we rejected**: Learned positional embeddings (v1) — fixed-length, no extrapolation, extra 1024×384=393K parameters.

### 4.7 Why Max Seq Len = 1024

**Chosen**: 1024

**Justification**: A typical formatted input is `<bos> claim (~20 words → 25 tokens) <sep> ev1 (~15 words → 20 tokens) <sep> ev2 <sep> ... ev6 <eos>` ≈ 25 + 6×20 = ~150 tokens. 1024 is more than enough for any input, while keeping memory manageable: 1024²=1M attention pairs vs 2048²=4M (4× more memory).

**What we rejected**: 2048 (v1-v4 default) — 4× memory with no benefit for our input lengths.

---

## 5. Pretraining Design

### 5.1 Objective: Masked Language Modeling

**Why MLM** (not Causal LM, not ELECTRA, not replaced token detection):

| Objective | Pros | Cons | Verdict |
|-----------|------|------|---------|
| MLM (15% masking) | Bidirectional, well-studied, fast | Only 15% supervision | **Chosen** |
| Causal LM | Full supervision | Unidirectional, bad for classification | Rejected |
| ELECTRA (RTD) | All-token supervision | Needs 2 models (gen+disc), 2× cost | Too expensive |
| SOPE (permutation) | Bidirectional in expectation | Complex, hard to tune | Over-engineered |

MLM provides bidirectional context (essential for cross-referencing evidence and claim) while being computationally efficient (only 15% of tokens contribute to loss).

### 5.2 Data: FineWeb-Edu (sample-10BT)

FineWeb-Edu is a filtered subset of CommonCrawl, selecting pages with high educational value using a classifier. It contains science articles, educational content, and reference material — including climate-related content. For T4 deployment, we use the streaming API to avoid downloading the full 10BT dataset.

### 5.3 Hyperparameter Table

| Parameter | Value | Source/Justification |
|-----------|-------|---------------------|
| MLM probability | 0.15 | BERT/RoBERTa standard |
| Mask-to-[MASK] | 0.80 | BERT standard |
| Mask-to-random | 0.10 | BERT standard |
| Mask-unchanged | 0.10 | BERT standard |
| Batch size | 16 | Fits T4 16GB at seq=1024 |
| Seq length | 1024 | Covers full claim+evidence |
| Peak LR | 3e-4 | ModernBERT/LDM standard |
| LR schedule | WSD (5-85-10) | Warmup-Stable-Decay, proven in many LLMs |
| Optimizer | AdamW (β=0.9,0.95) | Standard for transformers |
| Weight decay | 0.1 | LLaMA/ModernBERT standard |
| Mixed precision | FP16 | 2× speed on T4 |
| Gradient accumulation | 1 | Single-step accumulation at batch=16 |

### 5.4 Pretraining Token Budget

On T4 with estimated 30K tok/s throughput:
- Target: 200M tokens
- Time: 200M / 30K = 6,667s ≈ **1.85 hours**

This leaves ~7 hours for fine-tuning (1h) + evaluation (1h) + overhead (2h for setup/downloads).

---

## 6. Fine-Tuning Design

### 6.1 LoRA Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Rank (r) | 8 | Standard for classification; higher ranks overfit on small data |
| Alpha | 16 | Scale factor = α/r = 2.0 |
| Dropout | 0.1 | Light regularization |
| Target modules | W_q, W_k, W_v, W_o, gate_proj, up_proj, down_proj | All attention + FFN projections |

### 6.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 (was 2 in experiment; increased for better convergence) |
| Batch size | 4 (smaller for T4, grad_accum=2 for effective batch=8) |
| Peak LR | 2e-4 |
| LR schedule | Cosine decay with 50-step warmup |
| Weight decay | 0.1 |
| Loss | ImbalancedLoss (LDAM + CB + DRW) |
| Mixed precision | FP16 |

### 6.3 Class Imbalance Pipeline (Same as Experiment)

The full 7-method pipeline from the experiment report is preserved:
1. LDAM margins: class-dependent logit offsets
2. DRW: enable re-weighting at epoch 2
3. Class-Balanced weights: effective number sampling
4. Balanced Sampler: WeightedRandomSampler
5. MARC calibration: post-hoc ω·logit+β (8 parameters)
6. Teacher distillation: soft labels from DeepSeek
7. Data augmentation: augmented_train.json (3.5× more data)

---

## 7. T4 Deployment Optimization

### 7.1 T4 Hardware Profile

| Metric | T4 | RTX 5090 (experiment) |
|--------|-----|----------------------|
| VRAM | 16 GB | 32 GB |
| FP16 TFLOPS | ~65 | ~104 (est.) |
| Memory bandwidth | 320 GB/s | ~1,792 GB/s |
| Architecture | Turing (2018) | Blackwell (2025) |
| Flash Attention | Via xformers | Via SDPA native |

### 7.2 T4-Specific Optimizations

| Optimization | Effect | Implementation |
|-------------|--------|----------------|
| **FP16 mixed precision** | 2× speed, half VRAM | `torch.cuda.amp.autocast` |
| **TF32 matmul** | 1.5× speed on Ampere+ (NOT on T4!) | Set via `torch.backends.cuda.matmul.allow_tf32` |
| **Reduced batch size** | Fits 16GB VRAM | batch=16 (down from 24) |
| **Streaming dataset** | No disk cache needed | `datasets.load_dataset(streaming=True)` |
| **Gradient checkpointing** | Trades compute for memory | Disabled by default (not needed at batch=16) |
| **CUDA graph capture** | Reduces kernel launch overhead | Not implemented (complex) |

### 7.3 Estimated T4 Runtime

| Phase | Time Estimate | Notes |
|-------|--------------|-------|
| UV setup + package install | 3-5 min | uv is extremely fast |
| HuggingFace data download | 2-5 min | Streaming, metadata only |
| Tokenizer train/load | <1 min | Pretrained tokenizer loaded |
| Pretraining (200M tokens) | ~2 hours | At ~30K tok/s on T4 |
| Fine-tuning (3 epochs) | ~30 min | LoRA, small batches |
| Evaluation + report | ~15 min | FAISS index + classification |
| **Total** | **~3 hours** | Well within 11h budget |

The 11-hour budget leaves 8 hours of margin for network delays, debugging, and potential retries.

---

## 8. Colab Adaptation Strategy

### 8.1 Package Management: uv

uv is chosen over pip because:
- **10-100× faster** dependency resolution (no backtracking)
- **Single binary** — `curl | sh` installs in <5 seconds
- **Deterministic** — lockfile ensures reproducibility
- **Colab-friendly** — works in the Colab Python environment

### 8.2 Data Download Strategy

Use HuggingFace streaming API with the hf-mirror.com endpoint (mainland China access):
```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       streaming=True, split="train")
```

### 8.3 Notebook Structure

The Colab notebook (`train_colab.ipynb`) is organized into clear sections:
1. Setup (uv, packages, GPU check)
2. Data loading (HuggingFace streaming)
3. Model definition (all architecture code inline)
4. Pretraining (MLM loop with progress tracking)
5. Fine-tuning (LoRA + class imbalance pipeline)
6. Evaluation (classification + evidence retrieval)
7. Report generation (comparison table + markdown)

---

## 9. References & Influences

| Paper / Source | Venue | What We Adopted |
|---------------|-------|-----------------|
| **ModernBERT** (Warner et al., 2024) | Answer.AI | GQA + SwiGLU + RMSNorm + RoPE recipe for encoder |
| **BERT** (Devlin et al., 2019) | NAACL | MLM pretraining, 15% masking, bidirectional attention |
| **RoBERTa** (Liu et al., 2019) | arXiv | Optimized BERT training: dynamic masking, no NSP |
| **LLaMA** (Touvron et al., 2023) | Meta AI | SwiGLU, RMSNorm, RoPE recipe |
| **RoPE** (Su et al., 2021) | arXiv | Rotary Positional Embedding with θ=10000 |
| **GQA** (Ainslie et al., 2023) | arXiv | Grouped Query Attention 3:1 ratio |
| **SwiGLU** (Shazeer, 2020) | arXiv | Gated activation function |
| **RMSNorm** (Zhang & Sennrich, 2019) | NeurIPS | Root Mean Square normalization |
| **LDAM** (Cao et al., 2019) | NeurIPS | Label-Distribution-Aware Margin loss |
| **Logit Adjustment** (Menon et al., 2021) | ICLR | Logit calibration for imbalanced data |
| **CB Loss** (Cui et al., 2019) | CVPR | Class-Balanced re-weighting |
| **LoRA** (Hu et al., 2022) | ICLR | Low-Rank Adaptation (r=8, α=16) |
| **Focal Loss** (Lin et al., 2017) | ICCV | Confidence-based loss weighting |
| **Chinchilla** (Hoffmann et al., 2022) | NeurIPS | Scaling laws: 20:1 token:param ratio |
| **Class Imbalance NLP** (Henning et al., 2023) | EACL | Survey of imbalance methods for NLP |
| **DeepSeek-V3** (DeepSeek-AI, 2024) | arXiv | MoE with auxiliary load balancing |
| **NEFTune** (Jain et al., 2024) | ICLR | Noisy Embedding Fine-Tuning |

---

*Report generated from architecture comparison experiment. Best model selected by direct classification accuracy on dev set (154 claims).*
