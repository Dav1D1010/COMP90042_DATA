"""
Pretrainer and Finetuner classes for Climatron fact-checking system.

Pretrainer: MLM pretraining loop with WSD scheduling, FP16 autocast,
  streaming data, and token-based checkpointing.

Finetuner: LoRA-based fine-tuning with LDAM+CB loss for imbalanced
  4-way claim classification. Tracks best model by harmonic mean
  of macro-F1 and overall accuracy.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ProjectConfig, LABEL_NAMES
from src.models.lora import apply_lora
from src.models.variants import create_classifier_from_pretrained
from src.data.collator import MLMCollator, ClassificationCollator
from src.data.claims import ClaimDataset
from src.training.losses import StableImbalancedLoss
from src.training.scheduler import WSDScheduler


# ── Pretrainer ────────────────────────────────────────────────────────────

class Pretrainer:
    """MLM pretraining loop for ClimatronForPretraining.

    Constructs a DataLoader over StreamingPretrainDataset, runs the WSD
    schedule, and logs loss/throughput/ETA at configurable intervals.
    Saves checkpoints every N tokens to variant_checkpoints/{variant_label}/.

    Args:
        model:        ClimatronForPretraining instance.
        tokenizer:    Tokenizer with encode(), pad_token_id, bos_token_id, etc.
        train_dataset: StreamingPretrainDataset (or compatible iterable).
        config:       ProjectConfig controlling lr, batch, fp16, intervals.
        variant_label: Identifier string for per-variant checkpoint subdirectory.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        config: ProjectConfig,
        variant_label: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.config = config
        self.variant_label = variant_label

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self._wandb = None
        self._wandb_project = None
        self._init_optimizer = None  # stored for checkpoint resume

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Pretrainer] device={self.device}  "
              f"params={n_params/1e6:.1f}M  "
              f"batch={config.pretrain_batch_size}  "
              f"fp16={config.fp16}")

    def set_wandb(self, project_name: str):
        """Enable optional Weights & Biases logging.

        Call once before train() to log loss, lr, tokens/sec each step.
        """
        try:
            import wandb
            self._wandb = wandb
            self._wandb_project = project_name
        except ImportError:
            print("[Pretrainer] wandb not installed — skipping W&B logging.")

    def train(self, num_tokens_target: int):
        """Run the full pretraining loop.

        Iterates over DataLoader, consuming tokens until num_tokens_target
        is reached.  Applies WSD scheduling, FP16 autocast, and periodic
        checkpointing.

        Args:
            num_tokens_target:  Total tokens to consume before stopping.
        """
        cfg = self.config
        collator = MLMCollator(self.tokenizer, mlm_probability=cfg.mlm_probability)

        loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.pretrain_batch_size,
            collate_fn=collator,
            pin_memory=(self.device.type == "cuda"),
        )

        # ── Optimizer & scheduler ──
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.pretrain_learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01,
        )
        self._init_optimizer = optimizer

        # Estimate total steps from target tokens and average seq length.
        # StreamingPretrainDataset caps at max_seq_len (1024).  Real average
        # is ~512 tokens per example after truncation.
        avg_tokens_per_batch = cfg.pretrain_batch_size * 512
        total_steps = max(1, num_tokens_target // avg_tokens_per_batch)

        stable_steps = max(0, total_steps - cfg.warmup_steps - cfg.decay_steps)
        scheduler = WSDScheduler(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            stable_steps=stable_steps,
            decay_steps=cfg.decay_steps,
            peak_lr=cfg.pretrain_learning_rate,
            min_lr=cfg.min_lr,
        )

        # ── WandB init ──
        if self._wandb is not None:
            self._wandb.init(
                project=self._wandb_project,
                name=f"pretrain_{self.variant_label}",
                config={
                    "variant": self.variant_label,
                    "lr": cfg.pretrain_learning_rate,
                    "batch_size": cfg.pretrain_batch_size,
                    "target_tokens": num_tokens_target,
                    "warmup": cfg.warmup_steps,
                    "stable": stable_steps,
                    "decay": cfg.decay_steps,
                },
            )

        # ── Checkpoint directory ──
        ckpt_dir = cfg.variant_checkpoint_dir / self.variant_label
        os.makedirs(ckpt_dir, exist_ok=True)

        # ── Main loop ──
        self.model.train()
        tokens_consumed = 0
        step = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        scaler = torch.amp.GradScaler("cuda") if cfg.fp16 and self.device.type == "cuda" else None
        t0 = time.time()
        next_ckpt_tokens = cfg.checkpoint_interval_tokens

        for batch in loader:
            if tokens_consumed >= num_tokens_target:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg.fp16 and self.device.type == "cuda"):
                logits, _ = self.model(input_ids)
                loss = loss_fn(logits.transpose(1, 2), labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            batch_tokens = batch["attention_mask"].sum().item()
            tokens_consumed += batch_tokens
            step += 1

            # ── Logging ──
            if step % cfg.log_interval == 0 or tokens_consumed >= num_tokens_target:
                elapsed = time.time() - t0
                tps = tokens_consumed / elapsed if elapsed > 0 else 0
                eta_sec = ((num_tokens_target - tokens_consumed) / tps) if tps > 0 else 0
                eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 0 else "—"

                print(
                    f"step {step:>6d}  "
                    f"loss {loss.item():.4f}  "
                    f"lr {scheduler.last_lr:.2e}  "
                    f"tokens {tokens_consumed/1e6:.0f}M/{num_tokens_target/1e6:.0f}M  "
                    f"speed {tps/1e3:.1f}K tok/s  "
                    f"ETA {eta_str}"
                )

                if self._wandb is not None:
                    self._wandb.log({
                        "pretrain/loss": loss.item(),
                        "pretrain/lr": scheduler.last_lr,
                        "pretrain/tokens_consumed": tokens_consumed,
                        "pretrain/tokens_per_sec": tps,
                        "pretrain/step": step,
                    })

            # ── Checkpointing ──
            if tokens_consumed >= next_ckpt_tokens:
                self.save_checkpoint(step, ckpt_dir, optimizer, tokens_consumed)
                next_ckpt_tokens += cfg.checkpoint_interval_tokens

        # Final checkpoint at completion
        self.save_checkpoint(step, ckpt_dir, optimizer, tokens_consumed)
        print(f"[Pretrainer] done — {tokens_consumed/1e6:.0f}M tokens in {step} steps")

        if self._wandb is not None:
            self._wandb.finish()

    def save_checkpoint(self, step: int, ckpt_dir: Path, optimizer, tokens: int):
        """Save model weights and optimizer state to a checkpoint file.

        Args:
            step:      Current training step (used in filename).
            ckpt_dir:  Directory to write the checkpoint.
            optimizer: The optimizer whose state dict to save.
            tokens:    Tokens consumed so far (embedded in metadata).
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        path = ckpt_dir / f"pretrain_step{step}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "tokens_consumed": tokens,
                "variant_label": self.variant_label,
            },
            path,
        )
        # Prune old checkpoints
        all_ckpts = sorted(
            ckpt_dir.glob("pretrain_step*.pt"), key=os.path.getmtime
        )
        keep = max(1, self.config.keep_last_n)
        for old in all_ckpts[:-keep]:
            old.unlink()

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load a saved checkpoint, restoring model and optimizer state.

        Args:
            path:  Path to a .pt checkpoint file.

        Returns:
            dict with keys: step, tokens_consumed, variant_label.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self._init_optimizer is not None:
            self._init_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return {k: ckpt[k] for k in ("step", "tokens_consumed", "variant_label")}


# ── Finetuner ─────────────────────────────────────────────────────────────

class Finetuner:
    """LoRA-based fine-tuning for 4-way claim classification.

    Wraps a pretrained ClimatronForPretraining backbone, applies LoRA to
    attention + FFN projections, then trains with LDAM+CB loss for the
    imbalanced SUPPORTS/REFUTES/NEI/DISPUTED label distribution.

    Tracks best model by harmonic mean of macro-F1 and overall accuracy,
    saving the best checkpoint to variant_checkpoints/{variant_label}/.

    Args:
        pretrained_model:  ClimatronForPretraining backbone (encoder + lm_head).
        tokenizer:         Tokenizer for encoding claim+evidence text.
        train_claims:      Dict[str, dict] from load_claims() for training.
        dev_claims:        Dict[str, dict] from load_claims() for validation.
        evidence:          Dict[str, str] mapping evidence_id → text.
        config:            ProjectConfig controlling lr, epochs, LoRA dims.
        variant_label:     Identifier for per-variant checkpoint directory.
    """

    def __init__(
        self,
        pretrained_model,
        tokenizer,
        train_claims: dict[str, dict],
        dev_claims: dict[str, dict],
        evidence: dict[str, str],
        config: ProjectConfig,
        variant_label: str,
    ):
        self.config = config
        self.variant_label = variant_label
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)

        # Build classification model from pretrained backbone
        self.model = create_classifier_from_pretrained(
            pretrained_model, variant_label
        )
        self.model.to(self.device)

        # Apply LoRA to attention + FFN projections
        apply_lora(
            self.model,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

        # Datasets
        self.train_ds = ClaimDataset(
            train_claims, evidence, tokenizer, split="train",
        )
        self.dev_ds = ClaimDataset(
            dev_claims, evidence, tokenizer, split="dev",
        )

        # Loss: LDAM + CB weights from actual class counts
        counts_dict = self.train_ds.class_counts
        class_counts = torch.tensor(
            [counts_dict.get(name, 1) for name in LABEL_NAMES],
            dtype=torch.float32,
        )
        self.loss_fn = StableImbalancedLoss(
            class_counts,
            ldam_margin=config.ldam_margin,
            cb_beta=config.cb_beta,
            label_smoothing=config.label_smoothing,
        )

        n_trainable = sum(p.numel() for p in self.model.parameters()
                         if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[Finetuner] device={self.device}  "
            f"trainable={n_trainable/1e3:.1f}K / {n_total/1e6:.1f}M  "
            f"({100*n_trainable/n_total:.1f}%)  "
            f"train={len(self.train_ds)}  dev={len(self.dev_ds)}"
        )

    def finetune(self, num_epochs: int = 5):
        """Full fine-tuning loop: train + evaluate each epoch.

        Monitors best model by harmonic mean of macro-F1 and accuracy,
        saving the best checkpoint under variant_checkpoints/{variant_label}/.

        Args:
            num_epochs: Number of passes over the training set.
        """
        cfg = self.config

        train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.finetune_batch_size,
            shuffle=True,
            collate_fn=ClassificationCollator(self.tokenizer, max_length=1024),
            pin_memory=(self.device.type == "cuda"),
        )
        dev_loader = DataLoader(
            self.dev_ds,
            batch_size=cfg.finetune_batch_size,
            shuffle=False,
            collate_fn=ClassificationCollator(self.tokenizer, max_length=1024),
            pin_memory=(self.device.type == "cuda"),
        )

        optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.finetune_learning_rate,
            weight_decay=0.01,
        )

        best_hmean = 0.0
        best_state = None
        ckpt_dir = cfg.variant_checkpoint_dir / self.variant_label
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, self.loss_fn)

            dev_metrics = self.eval_epoch(dev_loader)
            hmean = dev_metrics["harmonic_mean"]

            status = "★ BEST" if hmean > best_hmean else ""
            print(
                f"epoch {epoch:>2d}  "
                f"train_loss {train_loss:.4f}  "
                f"acc {dev_metrics['accuracy']:.4f}  "
                f"macro_f1 {dev_metrics['macro_f1']:.4f}  "
                f"hmean {hmean:.4f}  {status}"
            )

            if hmean > best_hmean:
                best_hmean = hmean
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                torch.save(best_state, ckpt_dir / f"{self.variant_label}_finetuned.pt")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"[Finetuner] best harmonic mean = {best_hmean:.4f}")

    def train_epoch(self, dataloader, optimizer, loss_fn) -> float:
        """Run one training epoch.  Returns average loss.

        Applies gradient clipping at config.grad_clip.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def eval_epoch(self, dataloader) -> dict:
        """Evaluate on the validation set.  Returns a dict with:

        accuracy, macro_f1, harmonic_mean, per_class_accuracy (dict),
        predictions (tensor), targets (tensor).
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                logits = self.model(input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=-1)

                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        n_classes = len(LABEL_NAMES)

        # Per-class accuracy
        per_class = {}
        for c in range(n_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                per_class[LABEL_NAMES[c]] = (preds[mask] == c).float().mean().item()
            else:
                per_class[LABEL_NAMES[c]] = 0.0

        # Overall accuracy
        acc = (preds == targets).float().mean().item()

        # Macro-F1
        f1s = []
        for c in range(n_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

        macro_f1 = sum(f1s) / len(f1s)

        # Harmonic mean of macro-F1 and accuracy
        hmean = (2 * macro_f1 * acc / (macro_f1 + acc)
                 if (macro_f1 + acc) > 0 else 0.0)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "harmonic_mean": hmean,
            "per_class_accuracy": per_class,
            "predictions": preds,
            "targets": targets,
        }

    def predict(
        self,
        claim_texts: str,
        evidence_texts: str,
        tokenizer=None,
    ) -> dict:
        """Run inference on a single claim+evidence pair.

        Args:
            claim_texts:    Raw claim string.
            evidence_texts: Raw evidence string (or space/SEP-separated).
            tokenizer:      Optional tokenizer override (uses self.tokenizer
                            if None).

        Returns:
            dict with keys: predicted_class (int), predicted_label (str),
            probabilities (torch.Tensor of shape [1, 4]).
        """
        tokenizer = tokenizer or self.tokenizer
        self.model.eval()

        text = f"{claim_texts} <sep> {evidence_texts}"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            probs = logits.softmax(dim=-1)
            pred = logits.argmax(dim=-1).item()

        return {
            "predicted_class": pred,
            "predicted_label": LABEL_NAMES[pred],
            "probabilities": probs.cpu(),
        }
