# Project: Patch Transformer Evaluation

## Architecture
- `patch_transformer.py`: Target script to be analyzed. Contains model architecture, data processing logic, and metric calculations.
- `data/`: Directory containing ETTh1, exchange, weather datasets.
- `agent/`: Destination directory for the final walkthrough report.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Code Review | Analyze architecture, data logic, metric implementation (MSE, MAE, R2), and evaluate mathematical sufficiency. | none | DONE |
| 2 | Exec Validation | Run script on 3 datasets, capture logs. | M4 | DONE |
| 3 | Final Report | Compile findings and logs into final walkthrough report in `agent/` folder. | M1, M2, M4 | DONE |
| 4 | Fix Determinism | Add `torch.manual_seed(42)`, `np.random.seed(42)`, and `random.seed(42)` to `patch_transformer.py` to ensure reproducible numerical results. | none | DONE |
| 5 | Update Report | Rewrite the final report with deterministic metrics. | M4 | DONE |
