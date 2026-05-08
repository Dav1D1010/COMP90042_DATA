"""
Warmup-Stable-Decay (WSD) learning rate scheduler.

Design rationale (WSD over cosine/step):

  Warmup → linearly ramps lr from 0→peak over warmup_steps.
    Prevents early training instability from large gradients before
    optimizer momentum/variance statistics (AdamW) have stabilised.
    Without warmup, the first few steps can produce gradient spikes
    that corrupt the moving averages and require many steps to recover.

  Stable → holds lr at peak_lr for stable_steps.
    Unlike cosine which starts decaying immediately, the flat plateau
    allows deep exploration of the loss basin at full learning rate.
    Critical for language model pretraining where the loss landscape
    is highly non-convex and the model needs sustained high-lr
    exploration to escape poor local minima.

  Decay → linearly anneals from peak_lr→min_lr over decay_steps.
    The gradual noise reduction refines the solution, analogous to
    simulated annealing. Linear decay (vs cosine) gives more time at
    intermediate learning rates where the bulk of optimisation
    progress occurs.

Compared to cosine: WSD is ~1.5× slower to converge in early
training (stable phase doesn't reduce lr) but reaches lower final
loss — the sustained exploration finds better basins than premature
annealing.
"""


class WSDScheduler:
    """Warmup-Stable-Decay learning rate schedule.

    Three-phase schedule applied to an optimizer's param groups:

    - Phase 0 (warmup):  step < warmup_steps
        lr = min_lr + (peak_lr - min_lr) * step / warmup_steps
        (effectively 0→peak since min_lr ≈ 0 relative to peak)

    - Phase 1 (stable):  warmup_steps <= step < warmup_steps + stable_steps
        lr = peak_lr

    - Phase 2 (decay):   step >= warmup_steps + stable_steps
        lr = peak_lr - (peak_lr - min_lr) * (step - warmup - stable) / decay_steps

    Args:
        optimizer:  torch.optim.Optimizer whose param groups will be updated.
        warmup_steps:  Number of warmup steps (0→peak).
        stable_steps:  Number of steps at peak learning rate.
        decay_steps:   Number of decay steps (peak→min).
        peak_lr:       Maximum learning rate (held during stable phase).
        min_lr:        Minimum learning rate (decay floor, default 0.0).
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self._step = 0

    def get_lr(self, step: int | None = None) -> float:
        """Return the scheduled learning rate for the given step.

        If step is None, uses the internal step counter (advanced by step()).
        """
        s = step if step is not None else self._step

        if s < self.warmup_steps:
            progress = s / max(1, self.warmup_steps)
            return self.min_lr + (self.peak_lr - self.min_lr) * progress

        stable_end = self.warmup_steps + self.stable_steps
        if s < stable_end:
            return self.peak_lr

        decay_progress = (s - stable_end) / max(1, self.decay_steps)
        decay_progress = min(decay_progress, 1.0)
        return self.peak_lr - (self.peak_lr - self.min_lr) * decay_progress

    def step(self):
        """Advance one step and update all optimizer param groups to current lr."""
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._step += 1

    @property
    def last_lr(self) -> float:
        """The learning rate from the most recent step()."""
        return self.get_lr(self._step - 1) if self._step > 0 else self.min_lr
