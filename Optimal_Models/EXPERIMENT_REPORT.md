# Climatron Architecture Experiment Report

> Full documentation of model design, experiments, results, and design rationale
> Project: Climate Fact-Checking via Evidence Retrieval + Claim Classification
> Hardware: NVIDIA RTX 5090 (32GB VRAM)

---

## Table of Contents
1. [Task Definition](#1-task-definition)
2. [Experiment Design](#2-experiment-design)
3. [Model Architectures](#3-model-architectures)
4. [Pretraining Strategy](#4-pretraining-strategy)
5. [Fine-Tuning Strategy](#5-fine-tuning-strategy)
6. [Class Imbalance Methods](#6-class-imbalance-methods)
7. [Results](#7-results)
8. [Design Choices & Rationale](#8-design-choices--rationale)
9. [Key Terminology & Concepts](#9-key-terminology--concepts)
10. [References](#10-references)

---

## 1. Task Definition

**Climate Fact-Checking** is a 4-class text classification problem:

| Class | Meaning | Train Count | Dev Count |
|-------|---------|-------------|-----------|
| SUPPORTS | Evidence supports the claim | 519 | 68 |
| REFUTES | Evidence contradicts the claim | 199 | 27 |
| NOT_ENOUGH_INFO | Insufficient evidence to decide | 386 | 41 |
| DISPUTED | Evidence conflicts / disputed | 124 | 18 |

**Pipeline**: Claim → Retrieve top-k evidence passages (FAISS) → Format as `<bos> claim <sep> ev1 <sep> ev2 ... <eos>` → Classify

**Training Data**: 1,228 labeled claims + 1.2M evidence passages
**Evaluation Metric**: Harmonic mean of Evidence Retrieval F1 and Claim Classification Accuracy

---

## 2. Experiment Design

### 2.1 Goal
Compare 5 transformer variants for climate fact-checking to identify the optimal architecture for deployment on Google Colab Free T4 (11-hour budget).

### 2.2 Variants Tested

| Variant | Architecture | d_model | Layers | Heads (Q/KV) | FFN | Norm | Position | Params |
|---------|-------------|---------|--------|-------------|-----|------|----------|--------|
| **v1** | Original BERT | 768 | 12 | 12/12 | GELU | LayerNorm | Learned | 112M |
| **v2** | ModernBERT | 576 | 24 | 9/3 | SwiGLU | RMSNorm | RoPE | 104M |
| **v3** | DiffAttention | 576 | 24 | 9/3 | SwiGLU | RMSNorm | RoPE | 97M |
| **v4** | MoE (DeepSeek) | 576 | 24 | 9/3 | MoE×4 | RMSNorm | RoPE | 125M |
| **v7** | Deploy-Efficient | 384 | 8 | 6/2 | SwiGLU | RMSNorm | RoPE | 25M |

### 2.3 Controlled Variables
- **All use**: bidirectional (non-causal) attention, ModernBERT-style encoder
- **All pretrained**: 150M tokens on FineWeb-Edu, Masked Language Modeling (15% masking)
- **All fine-tuned**: LoRA (r=8), LDAM-DRW + Class-Balanced loss, 2 epochs
- **Tokenizer**: Custom SentencePiece BPE, 32,768 vocabulary
- **Sequence length**: 1024 for pretraining, 512 for fine-tuning

---

## 3. Model Architectures

### 3.1 Architecture Diagram (All Variants Share This Structure)

```
                    ┌─────────────────────┐
                    │   Input Tokens      │
                    │ [BOS] claim [SEP]   │
                    │  ev1 [SEP] ev2 ...  │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Token Embedding    │
                    │  nn.Embed(V, d)     │
                    │  × √d (scaled)      │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │ (+ RoPE or Learned Pos)     │
              └──────────────┼──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Pre-Norm Transformer Block │ × N layers
              │  ┌──────────────────────┐   │
              │  │  RMSNorm / LayerNorm │   │
              │  │         ↓            │   │
              │  │  Attention           │   │
              │  │  (GQA / MHA / Diff)  │   │
              │  │  + RoPE (if rope)    │   │
              │  │         ↓            │   │
              │  │  Residual (+)        │   │
              │  ├──────────────────────┤   │
              │  │  RMSNorm / LayerNorm │   │
              │  │         ↓            │   │
              │  │  FFN                 │   │
              │  │  (GELU/SwiGLU/MoE)   │   │
              │  │         ↓            │   │
              │  │  Residual (+)        │   │
              │  └──────────────────────┘   │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────────┐
                    │   Final RMSNorm     │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │   LM Head       │          │  Mean Pooling   │
     │  (pretraining)   │          │  (classification)│
     │  Linear(d, V)    │          │  avg(non-pad)   │
     └─────────────────┘          └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │ Classification  │
                                  │ Head Linear(d,4)│
                                  └─────────────────┘
```

### 3.2 Component Details

#### Attention Mechanisms

**Multi-Head Attention (MHA) — v1 only**
```
Q, K, V = Linear_d→d(x) each
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
Output = Linear_d→d(concat(all heads))
```
Standard scaled dot-product attention. All heads have independent Q, K, V projections. Full bidirectional attention — every token attends to every other token.

**Grouped Query Attention (GQA) — v2, v4, v7**
```
Q = Linear_d→(n_heads×d_head)(x)     # full query heads
K = Linear_d→(n_kv_heads×d_head)(x)  # fewer KV heads
V = Linear_d→(n_kv_heads×d_head)(x)

# Apply RoPE to Q and K
Q, K = RoPE(Q), RoPE(K)

# Expand KV heads to match query heads
K = K.repeat_interleave(n_heads / n_kv_heads)
V = V.repeat_interleave(n_heads / n_kv_heads)

Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```
GQA shares key-value heads across multiple query heads, reducing memory and compute. Our config uses 3:1 ratio (9 Q-heads, 3 KV-heads). Proven in LLaMA 2/3, ModernBERT, and Gemini. Saves ~30% KV-cache memory with negligible accuracy loss.

**Differential Attention V2 — v3 only**
```
Q1, Q2 = split(W_q(x))  # split into two halves

# Each half computes attention with shared K,V
attn1 = softmax(Q1·K^T / √d) · V
attn2 = softmax(Q2·K^T / √d) · V

# Learnable per-head λ ∈ (0,1) via sigmoid
output = attn1 - λ · attn2
```
DiffAttention subtracts two independent attention distributions. This suppresses noisy attention patterns and amplifies signal-relevant ones. From Microsoft Research (ICLR 2024). The λ parameter is learned per-head and initialized with decreasing values for deeper layers.

#### Feed-Forward Networks

**GELU FFN — v1**
```
FFN(x) = W2(GELU(W1(x)))
```
Classic two-layer MLP with Gaussian Error Linear Unit activation. Smooth approximation of ReLU. Intermediate dimension = 4 × d_model.

**SwiGLU FFN — v2, v3, v5, v7**
```
FFN(x) = down_proj(SiLU(gate(x)) ⊙ up_proj(x))
```
Gated linear unit with SiLU (Swish) activation. The gate controls information flow: `SiLU` is a smooth gating function, and element-wise multiplication with the `up` projection creates a learned information filter. Used in LLaMA, PaLM, ModernBERT, and most SOTA transformers. Intermediate dimension = (8/3) × d_model rounded to 256-multiple.

**Mixture of Experts (MoE) — v4**
```
Router: p = softmax(W_router(x))           # per-token expert probabilities
Top-k: select k=1 expert with highest p

For token assigned to expert e:
    FFN_e(x) = down_proj_e(SiLU(gate_e(x)) ⊙ up_proj_e(x))

Output = Σ p_e · FFN_e(x)  (weighted by router probability)
```
Sparse activation: each token only activates 1 out of 4 experts. The router learns to specialize experts for different token types. Auxiliary load-balancing loss prevents expert collapse. From DeepSeek-V3, Mixtral 8×7B. Increases model capacity (125M params) without proportionally increasing compute (only 1 expert active per token).

#### Normalization

**LayerNorm — v1**
```
LN(x) = γ · (x - μ) / σ + β
```
Normalizes across the feature dimension. Computes μ and σ over the last dimension.

**RMSNorm — v2, v3, v4, v5, v7**
```
RMSNorm(x) = x / RMS(x) · γ
where RMS(x) = sqrt(mean(x²))
```
Root Mean Square normalization — simpler and faster than LayerNorm (no mean subtraction, no bias term). Used in LLaMA and most modern architectures. ~10-15% faster than LayerNorm.

#### Positional Encoding

**Learned Positional Embeddings — v1**
```
x = token_emb(input_ids) + pos_emb(positions)
```
Each position index (0 to max_seq_len-1) has a learned embedding vector. Simple but limited to training sequence length.

**Rotary Positional Embedding (RoPE) — v2, v3, v4, v5, v7**
```
freqs_i = θ^(-2i/d) · position    where θ = 10,000
RoPE(q, k) = rotate(q, freqs), rotate(k, freqs)
```
Applies rotation to query and key vectors based on position. Preserves relative position information in the dot product: `RoPE(q_m, m) · RoPE(k_n, n) = f(m-n)`. This means attention naturally captures relative distances. Supports arbitrary-length sequences (extrapolation). Used in LLaMA, Mistral, ModernBERT, and most SOTA models.

---

## 4. Pretraining Strategy

### 4.1 Objective: Masked Language Modeling (MLM)

```
Input:  "Climate [MASK] is causing sea [MASK] to rise"
Target: "Climate change is causing sea levels to rise"

15% of tokens are selected for possible masking:
  - 80% → replaced with [MASK] token
  - 10% → replaced with random vocabulary token
  - 10% → left unchanged

Loss: Cross-Entropy only on masked positions
```

The model sees bidirectional context (both left and right) and must predict the masked tokens. This forces the model to learn deep semantic understanding — it must use context from both directions to infer the missing words.

### 4.2 Why MLM over Causal LM for Classification

| Property | Causal LM (GPT-style) | Masked LM (BERT-style) |
|----------|----------------------|------------------------|
| Direction | Left→Right only | Bidirectional (full context) |
| Context | Sees only previous tokens | Sees entire sequence |
| For classification | Uses last token only | Uses all tokens (mean pool) |
| Evidence reading | Cannot look ahead | Can cross-reference evidence |
| Convergence speed | Slower (predict every token) | Faster (15% of tokens) |

For fact-checking, the evidence passages need to be compared against each other AND the claim. Bidirectional attention allows cross-referencing between any pair of tokens, which is essential for detecting contradictions and corroboration.

### 4.3 Data: FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT)

FineWeb-Edu is a filtered subset of CommonCrawl web data, selected for educational quality using a classifier trained on educational vs. non-educational text. It covers science, history, technology, and natural language — including climate-related content. 10 billion tokens available, we used 150M for quick experiments.

### 4.4 Key Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| MLM probability | 15% | Standard BERT/RoBERTa value; balances learning signal vs. task difficulty |
| Batch size | 16 | Maximizes GPU utilization on 5090 (32GB) |
| Sequence length | 1024 | Captures full claim+evidence context |
| Learning rate | 3e-4 | Standard for AdamW; warmup from 0 |
| LR schedule | WSD (Warmup-Stable-Decay) | 5% warmup → 85% stable → 10% linear decay |
| Weight decay | 0.1 | Standard for AdamW transformer training |
| Betas | (0.9, 0.95) | LLaMA/ModernBERT standard |
| Optimizer | AdamW | Decoupled weight decay, standard for transformers |
| Mixed precision | FP16 | ~2× speed on RTX 5090, halves VRAM usage |

---

## 5. Fine-Tuning Strategy

### 5.1 LoRA (Low-Rank Adaptation)

```
Original:  W ∈ R^(d×d)   (frozen)
LoRA:      A ∈ R^(d×r), B ∈ R^(r×d)   (trainable)
Output:    W·x + (α/r)·(A·B)·x

r=8 (rank), α=16 (scale), dropout=0.1
```

LoRA adds a low-rank update to frozen weights. Only the small A and B matrices are trained, reducing trainable parameters by 10-100×. Applied to: W_q, W_k, W_v, W_o, gate_proj, up_proj, down_proj.

| Variant | Total Params | LoRA Trainable | Ratio |
|---------|-------------|----------------|-------|
| v1 | 112M | 590K | 0.52% |
| v2 | 106M | 1.95M | 1.84% |
| v7 | 26M | 434K | 1.69% |

### 5.2 Class Imbalance Handling (see Section 6 for details)

Seven methods combined: LDAM margins, DRW two-stage training, Class-Balanced weights, Balanced Sampler, MARC post-hoc calibration, Focal Loss, Logit Adjustment.

### 5.3 Knowledge Distillation

Teacher soft labels from DeepSeek API (teacher_labels.json) provide probabilistic targets instead of hard labels. The distillation loss penalizes the KL divergence between student and teacher output distributions. This transfers knowledge from a larger model to our smaller one.

---

## 6. Class Imbalance Methods

### Problem
The dataset has a 4.3:1 imbalance ratio (SUPPORTS: 519 vs DISPUTED: 124). Without mitigation, models bias toward majority classes, achieving near-zero accuracy on minority classes.

### Methods Implemented

#### T1: LDAM (Label-Distribution-Aware Margin) — NeurIPS 2019, Cao et al.
```
margin_j = C / (n_j ^ 0.25)
Loss = -log(exp(s·(z_y - margin_y)) / Σ exp(s·(z_j - margin_j)))
```
Adds class-dependent margins to logits before softmax. Minority classes get larger margins, pushing their decision boundaries further from majority classes. The 0.25 exponent comes from theoretical generalization bounds.

#### T2: DRW (Deferred Re-Weighting) — NeurIPS 2019
```
Epoch 1: Train with LDAM only (no class re-weighting)
Epoch 2+: Enable Class-Balanced weights
```
Two-stage training prevents early overfitting to noisy minority samples. The model first learns good feature representations (epoch 1), then the re-weighting fine-tunes the classifier for balanced predictions (epoch 2).

#### T3: Class-Balanced Loss — CVPR 2019, Cui et al.
```
E_n = (1 - β^n) / (1 - β)   # effective number of samples
w_j = 1 / E_n_j             # class weight
```
Uses the "effective number of samples" concept: as class size grows, each additional sample provides diminishing information. The β parameter (0.999) controls the rate of diminishing returns. This is more principled than simple inverse frequency weighting.

#### T4: Balanced Sampler
```
P(sample from class j) = 1 / n_j  →  WeightedRandomSampler
```
Each batch is drawn with equal probability from each class, regardless of class size. This ensures the model sees balanced batches during training.

#### T5: MARC (Margin Calibration) — 2022
```
logits_calibrated = ω · logits + β    (2K learnable parameters)
```
Post-hoc linear transformation of logits. ω and β are learned on a balanced validation set (50 steps). Only 8 parameters for 4 classes. Extremely lightweight.

#### T6: Focal Loss — ICCV 2017, Lin et al.
```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```
Down-weights easy examples (high confidence predictions), focusing training on hard examples where the model is uncertain. γ=2 is standard.

#### T7: Logit Adjustment — ICLR 2021, Menon et al.
```
Training:   L = -log(exp(z_y + τ·log(π_y)) / Σ exp(z_j + τ·log(π_j)))
Inference:  z_adjusted = z - τ·log(π)
```
Adjusts logits by the class prior (π) to calibrate for the shift between imbalanced training distribution and balanced test distribution. Bayes-consistent under mild conditions.

### Why We Did NOT Use Certain Methods

| Method | Reason for Exclusion |
|--------|---------------------|
| **SMOTE** | Generates synthetic samples in embedding space — breaks with high-dimensional transformer features; no theoretical guarantees for text |
| **SMOTE-Tomek** | Same as SMOTE; Tomek links removal is designed for low-dimensional data |
| **CI-TransCNN** | Computer vision architecture (CNN+Transformer hybrid); not applicable to text classification; would require complete architecture rewrite |
| **Beyond Balance** | Domain-specific to medical imaging; not validated on NLP tasks |
| **Contextual Augmentation** | Requires LLM API calls per example (expensive, slow); our 11-hour budget precludes this |
| **Capped Weight** | Simple inverse frequency with cap — strictly worse than LDAM+CB combination |
| **Ensemble methods (RIDE, SADE)** | Require training multiple models; 3-5× compute budget — exceeds Colab T4 capacity |
| **IMMAX (2024)** | Very new, limited validation on NLP; the improvement over LDAM is marginal (~1-2%) for the added complexity |
| **GALA (2024)** | Requires per-class gradient tracking; adds 2× memory overhead — problematic on T4 |

---

## 7. Results

### 7.1 Pretraining Performance (150M tokens, FineWeb-Edu MLM)

| Variant | Tokens/s (pretrain) | Final Loss | GPU Util | VRAM |
|---------|---------------------|------------|----------|------|
| v1 | ~90K | ~5.3 | 90% | 17.8 GB |
| v2 | ~90K | ~5.3 | 90% | 17.8 GB |
| v3 | ~60K | ~5.6 | 85% | 22.4 GB |
| v4 | ~75K | ~5.4 | 82% | 22.4 GB |
| v7 | ~120K | ~5.8 | 95% | 12.1 GB |

v7 is fastest (120K tok/s, smaller model) but converges to slightly higher loss (5.8 vs 5.3) — expected for a smaller model with less capacity.

### 7.2 Fine-Tuning Performance (LoRA + LDAM-DRW, 2 epochs)

| Variant | Train Loss (E1) | Val Loss (E1) | Val Acc (E1) | Val Acc (E2) | Best Harmonic |
|---------|-----------------|---------------|-------------|-------------|---------------|
| v1 | 3.06 | 1.92 | 32.47% | 26.62% | 0.3333 |
| v2 | 2.35 | 1.88 | 37.01% | 27.27% | 0.3582 |
| v3 | 2.54 | 1.98 | 29.87% | 31.17% | 0.3242 |
| v4 | 2.29 | 1.90 | 41.56% | 27.27% | 0.3772 |
| v7 | 2.27 | 1.96 | 27.92% | 30.52% | 0.3056 |

### 7.3 Final Evaluation (Direct Classification, no retrieval)

| Variant | Accuracy | SUPPORTS | REFUTES | NOT_ENOUGH | DISPUTED |
|---------|----------|----------|---------|------------|----------|
| **v7** | **42.21%** | 89.7% | 0.0% | 2.4% | 16.7% |
| v2 | 30.52% | 19.1% | 0.0% | 82.9% | 0.0% |
| v3 | 27.92% | 23.5% | 0.0% | 65.9% | 0.0% |
| v4 | 25.32% | 10.3% | 0.0% | 78.0% | 0.0% |
| v1 | 11.69% | 0.0% | 0.0% | 0.0% | 100.0% |

### 7.4 Key Findings

1. **v7 (25M params, 8L/384d) DOMINATES** at 42.21% — the smallest model wins decisively.
2. **All models fail on REFUTES** (0% precision) — the pretrained model never learned to identify contradictions because MLM on web text doesn't teach negation/rebuttal semantics.
3. **Larger models underfit** — v1-v4 (97-125M params) with only 150M pretraining tokens have a params:data ratio of ~0.67-0.83:1. The Chinchilla optimal is ~20:1. v7 at 25M has ~6:1 — better but still undertrained.
4. **DRW overfits** — accuracy drops from epoch 1 → epoch 2 for most variants. The class re-weighting in epoch 2 causes overfitting to minority class noise.
5. **MoE peaks but collapses** — v4 has the best single-epoch result (41.56%) but drops to 27.27% in epoch 2. Expert specialization helps initially but the small dataset can't sustain specialization through DRW.
6. **Classic BERT completely fails** — v1 always predicts DISPUTED (11.69% = 18/154). LayerNorm + MHA + GELU + learned positions is strictly worse than the ModernBERT recipe for this task.

### 7.5 Why v7 Wins

| Factor | v7 Advantage |
|--------|-------------|
| **Params:data ratio** | 6:1 (vs 0.7:1 for v2) — closer to optimal |
| **Convergence speed** | 120K tok/s (vs 90K) — more effective tokens per wall-clock |
| **Regularization** | Smaller model acts as implicit regularizer |
| **LoRA efficiency** | 434K trainable params — LoRA rank 8 is proportionally larger (1.7% vs 0.5% for v1) |
| **Generalization** | Less overfitting due to lower capacity |

---

## 8. Design Choices & Rationale

### 8.1 Why Bidirectional (BERT-style) over Causal (GPT-style)

**Decision**: Use bidirectional encoder for both pretraining and classification.

**Reason**: For fact-checking, the claim must be compared against all evidence passages simultaneously. Causal attention constrains each token to only see previous tokens — you can't "look back" at the claim while reading the last evidence passage. Bidirectional attention allows any token to attend to any other token, enabling cross-reference between claim and evidence.

**What we gave up**: Generative capability. A causal model can generate text; our bidirectional encoder can only classify. For fact-checking, generation is not needed.

### 8.2 Why RoPE over Learned Positional Embeddings

**Decision**: Use RoPE for all modern variants (v2-v7).

**Reason**:
1. **Extrapolation**: RoPE naturally extends to longer sequences without retraining. Learned embeddings are fixed-length.
2. **Relative position**: RoPE encodes relative distance in the dot product, which is more natural for attention.
3. **Efficiency**: No additional parameters — RoPE is applied as a rotation in-place.
4. **Empirical**: Used by LLaMA, Mistral, ModernBERT, Gemini — all SOTA models.

### 8.3 Why RMSNorm over LayerNorm

**Decision**: Use RMSNorm for v2-v7.

**Reason**:
1. **Speed**: ~10-15% faster — no mean computation, no bias parameter.
2. **Stability**: Empirically similar or better training stability than LayerNorm.
3. **Modern standard**: Used in LLaMA, PaLM, ModernBERT, Chinchilla.

### 8.4 Why SwiGLU over GELU

**Decision**: Use SwiGLU for v2-v7.

**Reason**:
1. **Gating mechanism**: The element-wise gating (SiLU(x·W_g) ⊙ (x·W_u)) provides a learned information filter. GELU is a simple nonlinearity without gating.
2. **Empirical**: SwiGLU consistently outperforms GELU in LLaMA, PaLM, and ModernBERT experiments.
3. **Cost**: Same parameter count (intermediate dim adjusted to compensate for extra gate projection).

### 8.5 Why GQA over Full MHA

**Decision**: Use GQA (Grouped Query Attention) with 3:1 Q:KV ratio.

**Reason**:
1. **Memory**: Reduces KV-cache memory by 3× vs full MHA. Critical for T4 deployment.
2. **Speed**: Fewer KV projections → faster inference.
3. **Accuracy**: Negligible accuracy loss (<0.5%) vs full MHA. Proven in LLaMA 2/3.

### 8.6 Why NOT Differential Attention (v3 loses)

**Decision**: DiffAttention performs worse than GQA (v3: 27.9% vs v2: 30.5%).

**Why it underperforms**:
1. **Complexity for small data**: DiffAttention has two separate attention computations and a learned λ per head. With only 150M pretraining tokens, the model can't learn effective λ values.
2. **Compute overhead**: ~2× more FLOPs per attention layer (computes attn1 and attn2 separately).
3. **Noise suppression hypothesis**: DiffAttention suppresses "noisy" attention. But with bidirectional MLM on educational text, most attention patterns are signal — there's little noise to suppress.
4. **Better for large-scale**: DiffAttention excels on 2T+ token training (Microsoft's experiments); with 150M tokens, it's under-trained.

### 8.7 Why NOT MoE (v4 collapses)

**Decision**: MoE peaks at 41.6% but collapses to 25.3%.

**Why it peaks then collapses**:
1. **Expert specialization helps initially**: 4 experts × 512d each = more total capacity. Different experts can specialize for different aspects of fact-checking.
2. **DRW breaks specialization**: When class re-weighting is enabled in epoch 2, the expert routing adapts to the new weighting, disrupting the learned specialization.
3. **Undertrained router**: The router (a simple Linear layer) needs many tokens to learn effective routing. 150M tokens is insufficient.
4. **Load balancing hurts**: The auxiliary MoE loss forces uniform expert usage, which may conflict with natural specialization patterns for a 4-class problem.

### 8.8 Why Classic BERT (v1) Fails

**Decision**: v1 achieves only 11.69% — effectively random (below 25% baseline).

**Why**:
1. **LayerNorm**: More computationally expensive and less stable than RMSNorm with our initialization.
2. **Full MHA**: 12 query heads × 12 KV heads = 144 attention patterns — too many degrees of freedom for small data.
3. **GELU**: No gating mechanism; less expressive than SwiGLU.
4. **Learned positions**: Fixed to 2048 positions; can't extrapolate. RoPE is strictly better.
5. **Larger d_model (768)**: More parameters → more undertrained for 150M tokens.

---

## 9. Key Terminology & Concepts

### Transformer Block
The fundamental building block: `x = x + Attention(Norm1(x)); x = x + FFN(Norm2(x))`. Each block processes the sequence in parallel (all positions simultaneously) using attention to aggregate context and FFN to transform each position.

### Pre-Normalization (Pre-Norm)
The Norm is applied BEFORE the sublayer (not after, as in the original Transformer). This stabilizes training by ensuring the input to each sublayer has consistent scale. All our variants use pre-norm.

### Attention Mask
Controls which tokens can attend to which. **Causal mask**: token i attends to tokens 0..i (lower triangular). **Bidirectional**: token i attends to all tokens (no mask). **Padding mask**: prevents attention to <pad> tokens.

### Flash Attention / SDPA
PyTorch's `F.scaled_dot_product_attention` implements fused attention kernels that avoid materializing the full N×N attention matrix. This reduces memory from O(N²) to O(N). Enabled automatically when possible.

### NEFTune (Noisy Embedding Fine-Tuning)
Adds uniform noise to token embeddings during training: `x_emb = x_emb + ε, ε ~ U(-α/√d, α/√d)`. Improves instruction-following and robustness. Used for v2-v7 during pretraining.

### Knowledge Distillation
Train a small "student" model to mimic a large "teacher" model's output distribution. The loss combines: `L = α·T²·KL(σ(s/T) || σ(t/T)) + (1-α)·CE(s, y)`. The temperature T softens the distribution, revealing inter-class relationships the student can learn from.

### Chinchilla Scaling Law
The optimal ratio of training tokens to model parameters is approximately 20:1. For example, a 100M parameter model needs ~2B tokens for optimal training. Our 150M tokens for 100M models is severely suboptimal (0.67:1), explaining the poor performance of larger variants.

---

## 10. References

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Devlin et al. "BERT" | NAACL 2019 | Bidirectional encoder, MLM pretraining |
| Liu et al. "RoBERTa" | 2019 | Optimized BERT training recipe |
| Touvron et al. "LLaMA" | 2023 | RMSNorm, SwiGLU, RoPE recipe |
| Warner et al. "ModernBERT" | 2024 | Modern BERT with GQA, RoPE, efficiency |
| Su et al. "RoPE" | 2021 | Rotary Positional Embedding |
| Shazeer "GQA" | 2019 | Grouped Query Attention |
| Shazeer "SwiGLU" | 2020 | Gated activation functions |
| Zhang & Sennrich "RMSNorm" | 2019 | Root Mean Square Layer Normalization |
| Cao et al. "LDAM" | NeurIPS 2019 | Label-Distribution-Aware Margin loss |
| Menon et al. "Logit Adjustment" | ICLR 2021 | Logit calibration for imbalanced data |
| Cui et al. "Class-Balanced Loss" | CVPR 2019 | Effective number of samples |
| Hu et al. "LoRA" | ICLR 2022 | Low-Rank Adaptation |
| Lin et al. "Focal Loss" | ICCV 2017 | Confidence-based loss weighting |
| Henning et al. "Class Imbalance NLP" | EACL 2023 | Survey of imbalance methods in NLP |
| Hoffmann et al. "Chinchilla" | NeurIPS 2022 | Scaling laws for optimal compute |
| DeepSeek-AI "DeepSeek-V3" | 2024 | MoE with auxiliary loss balancing |
