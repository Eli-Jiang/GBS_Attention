"""Run softmax + GBS fixed + GBS learnable comparison."""
import subprocess, sys

results = []
for label, cmd in [
    ("1. Softmax baseline",
     [sys.executable, "patch_transformer.py", "--dataset", "exchange",
      "--epochs", "10", "--patience", "5"]),
    ("2. GBS fixed c_ratio=0.1",
     [sys.executable, "gbs_patch_transformer.py", "--dataset", "exchange",
      "--epochs", "10", "--patience", "5", "--c_ratio", "0.1"]),
    ("3. GBS learnable init=0.3",
     [sys.executable, "gbs_patch_transformer.py", "--dataset", "exchange",
      "--epochs", "10", "--patience", "5", "--c_ratio_mode", "learnable"]),
]:
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="d:/University/Sophomore/2607")
    output = proc.stdout + proc.stderr
    print(output[-2000:])

    mse = mae = r2 = None
    for line in output.splitlines():
        if "Test MSE" in line:
            try: mse = float(line.split(":")[1].strip())
            except: pass
        if "Test MAE" in line:
            try: mae = float(line.split(":")[1].strip())
            except: pass
        if "Test R2" in line and "N_patches" not in line:
            try: r2 = float(line.split(":")[1].strip())
            except: pass
        if "Test R²" in line and "N_patches" not in line:
            try: r2 = float(line.split(":")[1].strip())
            except: pass
    results.append((label, mse, mae, r2))

print("\n\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"{'Mode':>30}  {'MSE':>10}  {'MAE':>10}  {'R²':>8}")
print("-"*55)
for label, mse, mae, r2 in results:
    mse_s = f"{mse:.6f}" if mse else "FAIL"
    mae_s = f"{mae:.6f}" if mae else "FAIL"
    r2_s = f"{r2:.4f}" if r2 else "FAIL"
    print(f"{label:>30}  {mse_s:>10}  {mae_s:>10}  {r2_s:>8}")
