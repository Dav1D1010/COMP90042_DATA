# Climatron Optimal Model — Variant Comparison Suite

Four self-contained variants to test on T4 tonight. Each in its own subdirectory.

## Variants

| Folder | Architecture | Params | Pretrain Tokens | Ratio | T4 Time | Tests |
|--------|-------------|--------|-----------------|-------|---------|-------|
| `v7_baseline/` | 8L/384d | 25M | 150M | 6:1 | ~2h | Reproduce experiment |
| `v7_extended/` | 8L/384d | 25M | 800M | 32:1 | ~8h | More data = better? |
| `v8_optimal/` | 10L/384d | 32M | 650M | 20:1 | ~6.5h | **RECOMMENDED** |
| `v9_wide/` | 8L/512d | 50M | 500M | 10:1 | ~5.5h | Width vs depth |

## Shared Architecture
All variants use: GQA (3:1), SwiGLU, RMSNorm, RoPE, bidirectional attention, mean pooling

## Key Fixes (vs experiment)
1. **NO DRW** — StableImbalancedLoss uses LDAM+CB+label smoothing from epoch 1
2. **Label smoothing** (ε=0.1) — prevents overconfidence on minority classes
3. **5 epochs** fine-tuning — more training without DRW degradation
4. **Gradient clipping** (max_norm=1.0) — prevents gradient explosion
5. **Data leakage verified** — train/dev/test splits clean, teacher labels don't leak

## How to Run
```bash
cd v7_baseline && python train.py   # or any variant
```

## Expected Behavior
- Performance should improve monotonically through epochs (no DRW regression)
- Larger token counts → better performance (test Chinchilla scaling)
- v8_optimal should be best due to optimal token:param ratio

## Monitoring
- Each variant prints progress every 100 steps
- Final model saved as `{variant}_classifier.pt`
