"""
Climatron model: bidirectional transformer encoder for pretraining (MLM)
and fine-tuning (claim classification).

Two model heads:
  - ClimatronForPretraining: masked language modeling — predicts token IDs
    from [MASK] positions. The lm_head projects to ext_vocab ≥ max(vocab_size,
    mask_token_id + 1) to handle special tokens with IDs outside the base vocab.
  - ClimatronForClassification: 4-class claim classification (supported,
    refuted, not-enough-evidence, conflicting-evidence). Reuses pretrained
    embedding/blocks/norm and adds a classification head. Mean-pools token
    representations (with optional attention mask for padding).
"""

import torch.nn as nn
from src.models.config import ModelConfig
from src.models.layers import TokenEmbedding, TransformerBlock, RMSNorm


class ClimatronForPretraining(nn.Module):
    """
    Bidirectional transformer encoder for masked language modeling.

    ext_vocab = max(vocab_size, mask_token_id + 1) ensures the embedding table
    and lm_head can represent special tokens (e.g. [MASK]) whose IDs may exceed
    the base vocabulary size — critical when vocabulary is compact (16K) but
    tokenizer assigns special tokens higher IDs.
    """
    def __init__(self, config: ModelConfig):
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
    """
    Sequence classifier built on a frozen/unfrozen pretrained encoder.

    Uses mean pooling over the token dimension (with optional attention mask
    to exclude padding tokens) followed by a linear classification head
    projecting d_model → 4 classes.

    Args:
        config: ModelConfig (defines d_model, pooling strategy, etc.)
        pretrained_model: ClimatronForPretraining — its embedding, blocks,
                          and final_norm are reused by reference.
    """
    def __init__(self, config: ModelConfig, pretrained_model: ClimatronForPretraining):
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
