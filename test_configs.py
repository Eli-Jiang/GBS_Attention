"""
test_configs.py — seq_len × pred_len 配置网格测试
====================================================

对 GBSPatchTransformer（量子注意力）和 PatchTransformer（Softmax 注意力）
在不同 (seq_len, pred_len) 组合下进行并排对比训练与评估。

输出三张表：
    1. Softmax 注意力结果
    2. GBS 注意力结果
    3. GBS 相对 Softmax 的指标差值（Δ）

可扩展性说明：
    - 修改下方 CONFIG 区域即可更换测试范围、数据集、训练轮数
    - 支持 --model gbs / softmax / both 三种模式
    - 支持 --quick 快速 2×2 网格（调试用）
    - 所有子进程调用使用 encoding='utf-8' 避免中文 Windows 编码错误

用法示例：
    python test_configs.py --quick --dataset ETTh1 --epochs 5 --model both
    python test_configs.py --dataset exchange --model gbs --epochs 10
    python test_configs.py --dataset weather --model both --epochs 5
"""

import subprocess
import sys
import argparse
import time
from itertools import product

# ============================================================
# 全局配置（修改这里来改变测试范围）
# ============================================================

# 完整网格配置
FULL_SEQ_LENS  = [48, 96, 192, 336]    # 回望窗口长度（时间步）
FULL_PRED_LENS = [24, 48, 96]          # 预测步长（时间步）

# 快速网格配置（--quick 时使用，调试/CI 友好）
QUICK_SEQ_LENS  = [96, 192]
QUICK_PRED_LENS = [24, 96]

# 各数据集的模型超参数（patch_size 和 stride 决定了 N_patches = GBS 模式数）
DATASET_CFG = {
    'ETTh1':    {'patch_size': 16, 'stride': 16, 'd_model': 32},
    'exchange': {'patch_size': 16, 'stride': 16, 'd_model': 32},
    'weather':  {'patch_size': 16, 'stride': 16, 'd_model': 32},
}

# 默认 GBS 挤压强度（c_ratio 扫描结果：exchange 最优 0.1，ETTh1/weather 不敏感）
DEFAULT_C_RATIO = 0.3


# ============================================================
# 辅助函数
# ============================================================

def patch_count(seq_len: int, patch_size: int, stride: int) -> int:
    """计算给定配置下的 patch 数量（= GBS 模式数 nmode）。"""
    return (seq_len - patch_size) // stride + 1


def run_one(
    script: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    patch_size: int,
    stride: int,
    d_model: int,
    c_ratio: float,
    epochs: int,
    patience: int,
) -> tuple[float | None, float | None, float | None, str]:
    """
    运行单个训练配置（子进程），返回 (mse, mae, r2, error_msg)。
    子进程以 --no_save 运行，不写入 results.txt，不保留 checkpoint。

    Returns:
        (mse, mae, r2) : 解析成功时为浮点数，失败时为 None
        error_msg      : 失败时包含详细错误信息，成功时为空字符串
    """
    cmd = [
        sys.executable, script,
        '--dataset',    dataset,
        '--seq_len',    str(seq_len),
        '--pred_len',   str(pred_len),
        '--patch_size', str(patch_size),
        '--stride',     str(stride),
        '--d_model',    str(d_model),
        '--epochs',     str(epochs),
        '--patience',   str(patience),
        '--batch_size', '32',
        '--no_save',
    ]
    if 'gbs' in script:
        cmd += ['--c_ratio', str(c_ratio)]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',      # 显式指定 UTF-8，避免中文 Windows 系统默认 GBK 导致解码错误
        errors='replace',
    )

    mse = mae = r2 = None
    for line in (proc.stdout + proc.stderr).splitlines():
        if 'Test MSE' in line or '测试 MSE' in line:
            try: mse = float(line.split(':')[1].strip())
            except: pass
        if 'Test MAE' in line or '测试 MAE' in line:
            try: mae = float(line.split(':')[1].strip())
            except: pass
        if ('Test R2' in line or '测试 R²' in line) and 'N_patches' not in line:
            try: r2 = float(line.split(':')[1].strip())
            except: pass

    error_msg = ''
    if proc.returncode != 0 or mse is None:
        # 记录完整错误供排查
        error_msg = (
            f"返回码: {proc.returncode}\n"
            f"STDERR: {proc.stderr[-500:] if proc.stderr else '（空）'}"
        )
    return mse, mae, r2, error_msg


def print_table_header(title: str):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print('='*68)
    print(f"{'seq_len':>10} {'pred_len':>10} {'N_patches':>12} "
          f"{'MSE':>10} {'MAE':>10} {'R²':>8}")
    print('-'*68)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='配置网格测试：GBS vs Softmax 注意力对比'
    )
    parser.add_argument('--model',    type=str, default='both',
                        choices=['gbs', 'softmax', 'both'],
                        help='运行哪种模型（gbs / softmax / both）')
    parser.add_argument('--dataset',  type=str, default='ETTh1',
                        choices=['ETTh1', 'exchange', 'weather'])
    parser.add_argument('--epochs',   type=int, default=5,
                        help='每个配置的训练轮数（推荐 5~10）')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心轮数')
    parser.add_argument('--c_ratio',  type=float, default=DEFAULT_C_RATIO,
                        help='GBS 挤压强度（仅对 GBS 模型生效）')
    parser.add_argument('--quick',    action='store_true',
                        help='使用 2×2 快速网格（调试用）')
    args = parser.parse_args()

    cfg       = DATASET_CFG[args.dataset]
    seq_lens  = QUICK_SEQ_LENS  if args.quick else FULL_SEQ_LENS
    pred_lens = QUICK_PRED_LENS if args.quick else FULL_PRED_LENS
    combos    = list(product(seq_lens, pred_lens))

    run_gbs     = args.model in ('gbs', 'both')
    run_softmax = args.model in ('softmax', 'both')

    total = len(combos) * (int(run_gbs) + int(run_softmax))

    print(f"\n{'='*68}")
    print(f"  配置网格测试 | 数据集: {args.dataset} | 模型: {args.model}")
    print(f"{'='*68}")
    print(f"  seq_lens  = {seq_lens}")
    print(f"  pred_lens = {pred_lens}")
    print(f"  训练轮数  = {args.epochs}（早停耐心 = {args.patience}）")
    print(f"  总运行次数 = {total}")
    print(f"{'='*68}")

    results: dict = {}   # (seq_len, pred_len, model) → (mse, mae, r2) or None
    done = 0

    for seq_len, pred_len in combos:
        N = patch_count(seq_len, cfg['patch_size'], cfg['stride'])

        for model_name, script in [
            ('softmax', 'patch_transformer.py'),
            ('gbs',     'gbs_patch_transformer.py'),
        ]:
            if (model_name == 'softmax' and not run_softmax) or \
               (model_name == 'gbs'     and not run_gbs):
                continue

            done += 1
            label = 'GBS' if model_name == 'gbs' else 'Softmax'
            print(f"\n[{done:02d}/{total}] {label:>7} | "
                  f"seq={seq_len}, pred={pred_len}, N={N}")

            t0 = time.time()
            mse, mae, r2, err = run_one(
                script, args.dataset,
                seq_len, pred_len,
                cfg['patch_size'], cfg['stride'],
                cfg['d_model'], args.c_ratio,
                args.epochs, args.patience,
            )
            elapsed = time.time() - t0

            if mse is not None:
                results[(seq_len, pred_len, model_name)] = (mse, mae, r2)
                print(f"         MSE={mse:.6f}  MAE={mae:.6f}  R²={r2:.4f}  "
                      f"（{elapsed:.0f}s）")
            else:
                results[(seq_len, pred_len, model_name)] = None
                print(f"         ⚠ 运行失败（{elapsed:.0f}s）")
                print(f"         错误信息：{err[:200]}")

    # ---- 结果汇总表 ----

    if run_softmax:
        print_table_header(f"Softmax 注意力结果 — {args.dataset}")
        for s, p in combos:
            N = patch_count(s, cfg['patch_size'], cfg['stride'])
            r = results.get((s, p, 'softmax'))
            if r:
                print(f"{s:>10} {p:>10} {N:>12} {r[0]:>10.6f} {r[1]:>10.6f} {r[2]:>8.4f}")
            else:
                print(f"{s:>10} {p:>10} {N:>12} {'—':>10} {'—':>10} {'—':>8}  ⚠失败")

    if run_gbs:
        print_table_header(f"GBS 注意力结果 — {args.dataset}")
        for s, p in combos:
            N = patch_count(s, cfg['patch_size'], cfg['stride'])
            r = results.get((s, p, 'gbs'))
            if r:
                print(f"{s:>10} {p:>10} {N:>12} {r[0]:>10.6f} {r[1]:>10.6f} {r[2]:>8.4f}")
            else:
                print(f"{s:>10} {p:>10} {N:>12} {'—':>10} {'—':>10} {'—':>8}  ⚠失败")

    if run_gbs and run_softmax:
        print_table_header(f"GBS 相对 Softmax 优势（Δ值，负 MSE = GBS 更好）— {args.dataset}")
        print(f"{'seq_len':>10} {'pred_len':>10} {'N':>12} "
              f"{'Δ MSE':>12} {'Δ R²':>10}  {'胜者':>8}")
        print('-'*68)
        for s, p in combos:
            N   = patch_count(s, cfg['patch_size'], cfg['stride'])
            gbs = results.get((s, p, 'gbs'))
            sft = results.get((s, p, 'softmax'))
            if gbs and sft:
                dmse = gbs[0] - sft[0]   # 负数 = GBS MSE 更小 = GBS 更好
                dr2  = gbs[2] - sft[2]   # 正数 = GBS R² 更高 = GBS 更好
                winner = '✓ GBS' if dr2 > 0 else '✓ Sft'
                print(f"{s:>10} {p:>10} {N:>12} "
                      f"{dmse:>+12.6f} {dr2:>+10.4f}  {winner:>8}")
            else:
                print(f"{s:>10} {p:>10} {N:>12} "
                      f"{'—':>12} {'—':>10}  {'⚠':>8}")

    print()


if __name__ == '__main__':
    main()
