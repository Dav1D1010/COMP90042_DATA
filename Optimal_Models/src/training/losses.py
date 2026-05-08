"""
Loss functions for imbalanced multi-class fact-check classification.

Class distribution: SUPPORTS=519, REFUTES=199, NOT_ENOUGH_INFO=386, DISPUTED=124
DISPUTED at ~10% → model sees SUPPORTS 4.2× more often.

Key strategies:
  LDAM: enforces larger classification margins for minority classes.
    DISPUTED gets margin ∝ 1/124^0.25 vs SUPPORTS ∝ 1/519^0.25.
  CB weights: effective-number re-weighting (β=0.999). SUPPORTS has 519 effective
    samples, DISPUTED has 124 effective → up-weights minority classes.
  Label smoothing: ε=0.1 prevents overconfidence on small classes where the model
    might memorize instead of generalise.
  AUFL over pure Focal Loss: asymmetric γ gives different treatment to positive
    vs negative per class. γ_pos=1 (lower) retains signal from well-classified
    SUPPORTS; γ_neg=4 (higher) aggressively down-weights well-classified DISPUTED.
  NO DRW: deferred re-weighting caused degradation at epoch transition.
    Stable approach (imbalanced treatment from epoch 1) proved better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableImbalancedLoss(nn.Module):
    """LDAM margins + Class-Balanced weights from epoch 1.

    NO deferred re-weighting — avoids the epoch 2 degradation we observed.
    Adds label smoothing (ε=0.1) to prevent overconfidence on small classes.

    LDAM rationale: minority classes need larger decision margins.
    DISPUTED (124 samples) gets margin = 0.3 / 124^0.25 ≈ 0.089,
    SUPPORTS (519 samples) gets margin = 0.3 / 519^0.25 ≈ 0.062.
    This forces the model to be more confident before predicting minority classes.

    CB rationale: β=0.999 gives effective sample count (1-β^N)/(1-β).
    Re-weights to balance the contribution of each class to the loss.

    Args:
        class_counts: torch.Tensor of per-class sample counts (e.g. [519, 199, 386, 124])
        ldam_margin: LDAM margin scalar (default 0.3)
        cb_beta: Class-balanced beta, controls effective sample scaling (default 0.999)
        label_smoothing: epsilon for label smoothing (default 0.1)
    """

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


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Asymmetric Unified Focal Loss for extreme class imbalance.

    Combines asymmetric focal loss (different gamma for pos/neg) with unified
    focal loss (probability shift delta). Treats multi-class as C independent
    binary problems via binary_cross_entropy_with_logits.

    Why asymmetric γ:
      γ_pos=1.0 (lower) → mildly down-weights well-classified SUPPORTS.
        SUPPORTS is the majority class; aggressive down-weighting would discard
        too much signal.
      γ_neg=4.0 (higher) → aggressively down-weights well-classified DISPUTED.
        DISPUTED negatives dominate the loss; heavy down-weighting prevents
        them from overwhelming minority class gradients.

    Unified shift (δ=0.1): shifts p_t → (1-δ)p_t + δ, smoothing the modulating
    factor near p_t=1. This prevents vanishing gradients on already-correct
    minority predictions, helping DISPUTED escape the "overwhelmed by negatives"
    trap.

    Args:
        gamma_pos: Focal gamma for positive examples (lower retains SUPPORTS signal)
        gamma_neg: Focal gamma for negative examples (higher suppresses DISPUTED noise)
        cls_weights: Per-class weights. If None, defaults to inverse-frequency
                     from [519, 199, 386, 124] (SUPPORTS, REFUTES, NEI, DISPUTED)
        delta: Unified focal shift parameter (default 0.1)
        lamb: Interpolation weight between focal and plain BCE (default 0.5)
    """

    _default_class_counts = torch.tensor([519, 199, 386, 124], dtype=torch.float32)

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, cls_weights=None, delta=0.1, lamb=0.5):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.delta = delta
        self.lamb = lamb

        if cls_weights is not None:
            self.register_buffer("cls_weights", cls_weights.float())
        else:
            counts = self._default_class_counts
            raw = counts.sum() / (len(counts) * counts)
            self.register_buffer("cls_weights", raw / raw.sum() * len(counts))

        self.num_classes = len(self.cls_weights)

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_one_hot, reduction="none"
        )

        p_t = torch.exp(-bce_loss)

        # Unified shift + asymmetric gamma
        p_t_shifted = (1 - self.delta) * p_t + self.delta
        gamma = targets_one_hot * self.gamma_pos + (1 - targets_one_hot) * self.gamma_neg

        # Focal modulation: (1-p_t)^gamma * BCE
        focal_loss = (1 - p_t_shifted).pow(gamma) * bce_loss

        # Interpolate λ·focal + (1-λ)·BCE
        loss = self.lamb * focal_loss + (1 - self.lamb) * bce_loss

        # Per-sample class-weighted aggregation
        weighted = loss * self.cls_weights.to(logits.device)
        return weighted.sum(dim=1).mean()
