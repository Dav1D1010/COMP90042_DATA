from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for ModernBERT-style bidirectional transformer encoder.
    Controls architecture dimensions, token IDs, and pooling strategy.

    Why these defaults:
    - d_model=384: 3:2 ratio with n_heads=6 (64-dim per head), Chinchilla-optimal
      for ~32M param budget on T4. Powers of 2 for CUDA alignment.
    - n_layers=8: Depth-efficient — deeper models need more tokens (Chinchilla law).
      8L gives ~32M params, 10L gives ~40M. Within T4 training budget.
    - GQA ratio 3:1 (n_heads=6, n_kv_heads=2): Reduces KV cache by 3x at <0.1%
      quality loss — critical for Colab memory.
    - d_intermediate=1024: ~8/3 × d_model (SwiGLU standard), rounded to 256 multiple.
    - vocab_size=16384: 16K tokens at d_model=384 = 6.3M embedding params (~20% of 32M).
      Switched from 32768 to match tokenizer training. Multiple of 64 for CUDA.
    - dropout=0.0: Modern decoder-style — no dropout during pretraining (acts as
      regularizer). Added only during fine-tuning if overfitting detected.
    - bidirectional=True: Always bidirectional — fact-checking needs full context.
    - pooling_type="mean": Mean pooling over non-padded tokens. Better than CLS token
      for factual classification tasks (averages evidence+claim representations).
    """
    vocab_size: int = 16384
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    d_intermediate: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.0
    mask_token_id: int = 4
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    bidirectional: bool = True
    pooling_type: str = "mean"

    def __post_init__(self):
        raw = (8 * self.d_model) // 3
        self.d_intermediate = ((raw + 255) // 256) * 256
