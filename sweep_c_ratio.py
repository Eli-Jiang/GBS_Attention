"""
sweep_c_ratio.py
----------------
Sweep c_ratio in {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} on the exchange dataset.
Uses 10 epochs (early stopping patience=3) for speed.
Results appended to results.txt and printed as a summary table.
"""
import subprocess, sys, json
import argparse

C_RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DATASET = 'exchange'
EPOCHS = 10
PATIENCE = 3

results = []

for c in C_RATIOS:
    print(f"\n{'='*50}")
    print(f"Running c_ratio={c}")
    print('='*50)
    cmd = [
        sys.executable, 'gbs_patch_transformer.py',
        '--dataset', DATASET,
        '--epochs', str(EPOCHS),
        '--patience', str(PATIENCE),
        '--c_ratio', str(c),
        '--batch_size', '32',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    output = proc.stdout + proc.stderr

    # Parse final test MSE/MAE/R2 from output
    mse = mae = r2 = None
    for line in output.splitlines():
        if 'Test MSE' in line:
            try: mse = float(line.split(':')[1].strip())
            except: pass
        if 'Test MAE' in line:
            try: mae = float(line.split(':')[1].strip())
            except: pass
        if 'Test R2' in line:
            try: r2 = float(line.split(':')[1].strip())
            except: pass

    results.append({'c_ratio': c, 'mse': mse, 'mae': mae, 'r2': r2})
    print(f"  c_ratio={c}: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")

print("\n" + "="*55)
print(f"c_ratio SWEEP SUMMARY — dataset={DATASET}")
print("="*55)
print(f"{'c_ratio':>10}  {'Test MSE':>12}  {'Test MAE':>12}  {'Test R2':>10}")
print("-"*55)
for r in results:
    print(f"{r['c_ratio']:>10.1f}  {r['mse']:>12.6f}  {r['mae']:>12.6f}  {r['r2']:>10.6f}")
best = max(results, key=lambda x: x['r2'])
print(f"\nBest: c_ratio={best['c_ratio']}, R2={best['r2']:.6f}, MSE={best['mse']:.6f}")
