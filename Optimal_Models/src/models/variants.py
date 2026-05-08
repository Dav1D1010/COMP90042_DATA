"""
Model variant factory functions for Climatron.

Each variant explores a point on the depth-width-token tradeoff per Chinchilla
(optimal tokens ≈ 20× params). Deeper/wider architectures need proportionally
more pretraining tokens to avoid diminishing returns.

  v7_baseline  (8L/384d, 150M tokens): ~32M params, minimal baseline
  v7_extended  (8L/384d, 800M tokens): same arch, 5.3× tokens → isolates data axis
  v8_optimal   (10L/384d, 650M tokens): ~40M params, deeper → better classification
  v9_wide      (512d/8L, 500M tokens): ~49M params, wider → fine-grained semantics
"""

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import VARIANT_CONFIGS                                          # noqa: E402
from src.models.config import ModelConfig                                      # noqa: E402
from variants.shared_model import (                                             # noqa: E402
    ClimatronForPretraining,
    ClimatronForClassification,
)


def build_variant(label: str, vocab_size: int = 16384):
    """Build a pretraining model variant with architecture config from VARIANT_CONFIGS.

    Args:
        label: "v7_baseline" | "v7_extended" | "v8_optimal" | "v9_wide"
        vocab_size: Tokenizer vocabulary size (default 16384)

    Returns:
        (ClimatronForPretraining, ModelConfig)

    Raises:
        ValueError: Unknown variant label
    """
    if label not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{label}'. Available: {list(VARIANT_CONFIGS.keys())}"
        )

    overrides = VARIANT_CONFIGS[label]
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=overrides["d_model"],
        n_layers=overrides["n_layers"],
        n_heads=overrides["n_heads"],
        n_kv_heads=2,
        d_intermediate=overrides["d_intermediate"],
    )
    return ClimatronForPretraining(config), config


def create_classifier_from_pretrained(pretrained_model, variant_label=None):
    """Build a ClimatronForClassification reusing a pretrained encoder backbone.

    Transfers token_embedding, transformer blocks, and final_norm, then adds
    a fresh Linear→4 classification head.

    Args:
        pretrained_model: ClimatronForPretraining instance
        variant_label: Optional identifier for checkpoint tracking

    Returns:
        ClimatronForClassification ready for fine-tuning
    """
    classifier = ClimatronForClassification(
        config=pretrained_model.config,
        pretrained_model=pretrained_model,
    )
    if variant_label is not None:
        classifier._variant_label = variant_label
    return classifier
