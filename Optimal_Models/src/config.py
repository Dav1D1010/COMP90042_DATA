import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class ProjectConfig:
    """
    Central configuration for the Climatron fact-checking system.
    Controls paths, training parameters, model variant selection, and hardware settings.
    """
    data_dir: Path = Path(__file__).parent.parent.parent / "data"
    shared_checkpoint_dir: Path = Path(__file__).parent.parent / "shared_checkpoints"
    variant_checkpoint_dir: Path = Path(__file__).parent.parent / "variant_checkpoints"

    # Model variant (set via CLI/env: "v7_baseline", "v7_extended", "v8_optimal", "v9_wide", "all")
    variant: str = "v8_optimal"

    # Pretraining
    pretrain_tokens: int = 650_000_000
    pretrain_batch_size: int = 16
    pretrain_epochs: int = 1
    pretrain_learning_rate: float = 1e-3
    mlm_probability: float = 0.15

    # Fine-tuning
    finetune_epochs: int = 5
    finetune_batch_size: int = 16
    finetune_learning_rate: float = 2e-4

    # LoRA
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Loss
    ldam_margin: float = 0.3
    cb_beta: float = 0.999
    label_smoothing: float = 0.1

    # Training infrastructure
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    fp16: bool = True
    grad_clip: float = 1.0
    use_wandb: bool = True
    wandb_project: str = "climatron-fact-check"

    # LR scheduler (WSD: Warmup-Stable-Decay)
    warmup_steps: int = 2000
    decay_steps: int = 2000
    min_lr: float = 1e-5

    # Logging
    log_interval: int = 100

    # Checkpointing
    save_every_steps: int = 2000
    checkpoint_interval_tokens: int = 50_000_000
    keep_last_n: int = 3

    # Tokenizer
    tokenizer_vocab_size: int = 16384
    tokenizer_name: str = "climatron_16k"

    # Retrieval
    top_k_retrieval: int = 5
    faiss_index_path: Path = Path(__file__).parent.parent / "shared_checkpoints" / "evidence.index"

    # Cross-validation
    cv_folds: int = 5

    def __post_init__(self):
        for attr in ('data_dir', 'shared_checkpoint_dir', 'variant_checkpoint_dir', 'faiss_index_path'):
            val = getattr(self, attr)
            if isinstance(val, (str, Path)):
                setattr(self, attr, Path(val))

        os.makedirs(self.shared_checkpoint_dir, exist_ok=True)
        os.makedirs(self.variant_checkpoint_dir, exist_ok=True)

    @property
    def tokenizer_path(self):
        return self.shared_checkpoint_dir / f"{self.tokenizer_name}.json"

    @property
    def pretrained_model_path(self):
        return self.shared_checkpoint_dir / "pretrained.pt"

    def variant_model_path(self, variant_label):
        return self.variant_checkpoint_dir / f"{variant_label}_finetuned.pt"


VARIANT_CONFIGS = {
    "v7_baseline":  dict(d_model=384, n_layers=8,  n_heads=6, d_intermediate=1024, pretrain_tokens=150_000_000, batch_size=16),
    "v7_extended":  dict(d_model=384, n_layers=8,  n_heads=6, d_intermediate=1024, pretrain_tokens=800_000_000, batch_size=16),
    "v8_optimal":   dict(d_model=384, n_layers=10, n_heads=6, d_intermediate=1024, pretrain_tokens=650_000_000, batch_size=16),
    "v9_wide":      dict(d_model=512, n_layers=8,  n_heads=8, d_intermediate=1365, pretrain_tokens=500_000_000, batch_size=14),
}

LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
NUM_CLASSES = len(LABEL_NAMES)

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would',
    'shall', 'should', 'may', 'might', 'must', 'can', 'could'}
