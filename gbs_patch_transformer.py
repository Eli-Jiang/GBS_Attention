"""
gbs_patch_transformer.py — 量子 GBS 注意力 PatchTransformer
=============================================================

将 PatchTransformer 中的标准 Softmax 自注意力替换为 GBS（高斯玻色采样）注意力。

模型结构（通道独立）：
    输入 (B, L, C)
    → Patch: (B*C, N, patch_size)   [N = (L - patch_size) / stride + 1]
    → Linear embed: (B*C, N, d_model)
    → GBSAttentionLayer            [B 矩阵解析概率，全可微]
    → FFN + LayerNorm
    → Linear head → (B, pred_len, C)

可扩展性说明：
    - 所有默认超参数集中在下方 DEFAULT_* 区域，方便迁移到新数据集
    - N 由 seq_len/patch_size 自动推导，GBSAttentionLayer 自适应任意 N
    - 支持 --no_save 参数（用于 test_configs.py 网格测试，不污染 results.txt）

用法示例：
    python gbs_patch_transformer.py --dataset ETTh1 --epochs 20
    python gbs_patch_transformer.py --dataset exchange --c_ratio 0.1 --epochs 20
    python gbs_patch_transformer.py --dataset weather --seq_len 192 --d_model 64
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloader
from gbs_attention import GBSAttentionLayer, GBS_CFG

# ============================================================
# 全局默认配置（迁移到新数据集时在此处修改）
# ============================================================
DEFAULT_DATA_DIR      = './data'
DEFAULT_DATASET       = 'ETTh1'        # 数据集名称：ETTh1 / exchange / weather
DEFAULT_SEQ_LEN       = 96             # 回望窗口长度（时间步）
DEFAULT_PRED_LEN      = 24             # 预测步长（时间步）
DEFAULT_PATCH_SIZE    = 16             # 每个 patch 覆盖的时间步数
DEFAULT_STRIDE        = 16             # patch 滑动步长（=PATCH_SIZE 时为非重叠）
DEFAULT_D_MODEL       = 32             # Transformer 特征维度
DEFAULT_C_RATIO       = 0.3           # GBS 挤压强度（0~1，越小越接近线性注意力）
DEFAULT_DROPOUT       = 0.1
DEFAULT_BATCH_SIZE    = 32
DEFAULT_EPOCHS        = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE      = 5              # 早停耐心轮数


# ============================================================
# 模型定义
# ============================================================

class GBSPatchTransformer(nn.Module):
    """
    量子注意力 PatchTransformer：用 GBSAttentionLayer 替换标准 MHA。

    关键设计：
        - 通道独立（Channel-Independent）：每个特征通道独立编码，与经典 PatchTST 一致
        - N = num_patches 自动由 seq_len / patch_size 推导，GBSAttentionLayer.nmode 随之自适应
        - Pre-Norm 架构（先 LayerNorm 再 Attention/FFN），训练更稳定

    Args:
        seq_len     : 输入序列长度（时间步数）。
        pred_len    : 预测步长。
        num_features: 特征变量数 C。
        patch_size  : 单个 patch 的时间步数 P。
        stride      : patch 滑动步长（= patch_size 时为非重叠）。
        d_model     : 特征维度。
        c_ratio     : GBS 挤压强度，控制量子注意力的非线性程度。
        dropout     : Dropout 概率。
    """

    def __init__(
        self,
        seq_len: int      = DEFAULT_SEQ_LEN,
        pred_len: int     = DEFAULT_PRED_LEN,
        num_features: int = 7,
        patch_size: int   = DEFAULT_PATCH_SIZE,
        stride: int       = DEFAULT_STRIDE,
        d_model: int      = DEFAULT_D_MODEL,
        c_ratio: float    = DEFAULT_C_RATIO,
        c_ratio_mode: str | None = None,
        dropout: float    = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.patch_size = patch_size
        self.stride     = stride
        self.d_model    = d_model
        self.c_ratio_mode = c_ratio_mode

        # 自动推导 patch 数量 N（= GBS 模式数 nmode）
        self.num_patches = (seq_len - patch_size) // stride + 1

        # ---- 编码层 ----
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- GBS 注意力（替换标准 MHA）----
        # nmode 自动跟随 num_patches，支持任意 seq_len
        self.attn  = GBSAttentionLayer(
            nmode=self.num_patches,
            d_model=d_model,
            c_ratio=c_ratio,
            c_ratio_mode=c_ratio_mode,
        )

        # ---- FFN 子层 ----
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        # ---- 预测头 ----
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, L, C)  B=批次大小，L=seq_len，C=特征数

        Returns:
            out : (B, pred_len, C)
        """
        B, L, C = x.shape

        # 通道独立：(B, L, C) → (B*C, L)
        x_ci = x.transpose(1, 2).reshape(B * C, L)

        # Patching：(B*C, L) → (B*C, N, P)
        patches = x_ci.unfold(-1, self.patch_size, self.stride)

        # Patch 嵌入 + 位置编码：(B*C, N, d_model)
        enc = self.patch_proj(patches) + self.pos_embed

        # GBS 注意力块（Pre-Norm）
        enc = enc + self.drop(self.attn(self.norm1(enc)))

        # FFN 块（Pre-Norm）
        enc = enc + self.drop(self.ffn(self.norm2(enc)))

        # 展平后预测：(B*C, N*d_model) → (B*C, pred_len)
        enc = enc.reshape(B * C, -1)
        out = self.head(enc)

        # 恢复通道维度：(B*C, pred_len) → (B, pred_len, C)
        out = out.reshape(B, C, -1).transpose(1, 2)
        return out


# ============================================================
# 评估函数
# ============================================================

def evaluate(model, dataloader, criterion, device):
    """
    计算验证/测试集上的 Loss、MSE、MAE、R²、MAPE，以及 Naive Baseline 对比。

    Naive Baseline = 把输入序列最后一个值重复 pred_len 次（持久性模型）。
    若模型 MSE/MAE 不低于 Naive，说明模型没有学到任何时序动态。
    MASE < 1 表示模型比"复制最后一步"好。
    """
    model.eval()
    total_loss = 0.0
    preds, trues, last_vals = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out   = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
            # 保存输入序列最后一个值（用于 Naive Baseline）
            last_vals.append(x[:, -1:, :].cpu().numpy())  # (B, 1, C)

    preds = np.concatenate(preds)            # (N, pred_len, C)
    trues = np.concatenate(trues)
    last_vals = np.concatenate(last_vals)    # (N, 1, C)

    # ── 模型指标 ──
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    r2  = 1.0 - np.sum((trues - preds) ** 2) / (
        np.sum((trues - trues.mean(axis=0)) ** 2) + 1e-9
    )
    # MAPE（百分比误差，加 epsilon 防除零）
    mape = np.mean(np.abs((trues - preds) / (np.abs(trues) + 1e-9))) * 100

    # ── Naive Baseline（持久性模型）──
    naive_preds = np.repeat(last_vals, trues.shape[1], axis=1)  # (N, pred_len, C)
    naive_mse   = np.mean((naive_preds - trues) ** 2)
    naive_mae   = np.mean(np.abs(naive_preds - trues))

    # ── MASE：模型 MAE 相对 Naive MAE 的比值 ──
    mase = mae / (naive_mae + 1e-9)

    return total_loss / len(dataloader.dataset), mse, mae, r2, mape, naive_mse, naive_mae, mase


# ============================================================
# 训练函数
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"[GBSPatchTransformer] 开始训练")
    print(f"{'='*55}")
    print(f"设备        : {device}")
    print(f"数据集      : {args.dataset}")
    print(f"seq_len     : {args.seq_len}  →  pred_len: {args.pred_len}")
    print(f"patch_size  : {args.patch_size},  stride: {args.stride}")
    print(f"d_model     : {args.d_model},  c_ratio: {args.c_ratio}")
    c_ratio_mode = getattr(args, 'c_ratio_mode', None) or GBS_CFG.get("c_ratio_mode", "fixed")
    print(f"c_ratio_mode: {c_ratio_mode}")

    # ---- 加载数据 ----
    train_loader, _, _ = get_dataloader(
        args.data_dir, args.dataset, 'train',
        args.batch_size, args.seq_len, args.pred_len
    )
    val_loader,   _, _ = get_dataloader(
        args.data_dir, args.dataset, 'val',
        args.batch_size, args.seq_len, args.pred_len
    )
    test_loader,  _, _ = get_dataloader(
        args.data_dir, args.dataset, 'test',
        args.batch_size, args.seq_len, args.pred_len
    )

    dummy_x, _ = next(iter(train_loader))
    num_features = dummy_x.shape[2]
    num_patches  = (args.seq_len - args.patch_size) // args.stride + 1
    print(f"特征维度    : {num_features}  →  N_patches (GBS 模式数): {num_patches}")
    print(f"{'='*55}\n")

    # ---- 初始化模型 ----
    model = GBSPatchTransformer(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_features=num_features,
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        c_ratio=args.c_ratio,
        c_ratio_mode=getattr(args, 'c_ratio_mode', None) or GBS_CFG.get("c_ratio_mode", "fixed"),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 模型保存路径（包含 seq_len 和 pred_len 以避免多配置运行时覆盖）
    save_path = (
        f"{args.dataset}_gbs_patch_transformer"
        f"_s{args.seq_len}_p{args.pred_len}_c{args.c_ratio}.pt"
    )

    best_val_loss = float('inf')
    patience_cnt  = 0

    print("轮次   耗时    训练Loss   验证Loss   验证R²")
    print("-" * 50)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]
        val_loss, _, _, val_r2, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Ep {epoch+1:02d}/{args.epochs}  {elapsed:.1f}s  "
              f"{train_loss:.4f}    {val_loss:.4f}    {val_r2:.4f}", end="")

        # learnable 模式：打印当前 c_ratio 变化
        if model.c_ratio_mode == "learnable":
            cr_val = model.attn.c_ratio
            if isinstance(cr_val, torch.Tensor):
                print(f"   c_ratio={cr_val.item():.4f}", end="")
        print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[早停] 第 {epoch+1} 轮触发，验证集 Loss 已连续 {args.patience} 轮未改善。")
                break

    # ---- 测试评估 ----
    model.load_state_dict(torch.load(save_path, weights_only=True))
    (_, test_mse, test_mae, test_r2,
     test_mape, naive_mse, naive_mae, test_mase) = evaluate(
        model, test_loader, criterion, device
    )

    print("\n" + "=" * 55)
    print(f"最终测试结果 — {args.dataset}（GBSPatchTransformer）")
    print("=" * 55)
    print(f"测试 MSE  : {test_mse:.6f}   Naive MSE: {naive_mse:.6f}")
    print(f"测试 MAE  : {test_mae:.6f}   Naive MAE: {naive_mae:.6f}")
    print(f"测试 MAPE : {test_mape:.2f}%")
    print(f"测试 MASE : {test_mase:.4f}  {'✅ < 1 好于 Naive' if test_mase < 1 else '⚠ ≥ 1 差于 Naive'}")
    print(f"测试 R²   : {test_r2:.6f}")
    cr_mode = getattr(args, 'c_ratio_mode', None) or GBS_CFG.get("c_ratio_mode", "fixed")
    if cr_mode == 'learnable':
        print(f"最终 {model.attn.get_c_ratio_info()}    N_patches: {num_patches}")
    else:
        print(f"c_ratio(fixed)={args.c_ratio}    N_patches: {num_patches}")
    print("=" * 55 + "\n")

    # ---- 结果持久化（--no_save 时跳过） ----
    if not args.no_save:
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"Dataset: {args.dataset} | GBSPatchTransformer\n")
            f.write(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f} | Test R2: {test_r2:.6f}\n")
            f.write(f"Naive MSE: {naive_mse:.6f} | Naive MAE: {naive_mae:.6f}\n")
            f.write(f"MAPE: {test_mape:.2f}% | MASE: {test_mase:.4f}\n")
            f.write(
                f"Config: seq_len={args.seq_len}, pred_len={args.pred_len}, "
                f"patch_size={args.patch_size}, d_model={args.d_model}, c_ratio={args.c_ratio}\n"
            )
            f.write("-" * 55 + "\n")
    else:
        # 网格测试模式：清理 checkpoint 文件，不写入 results.txt
        if os.path.exists(save_path):
            os.remove(save_path)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GBS PatchTransformer — 量子注意力时序预测'
    )
    # 数据参数
    parser.add_argument('--dataset',       type=str,   default=DEFAULT_DATASET,
                        choices=['ETTh1', 'exchange', 'weather'])
    parser.add_argument('--data_dir',      type=str,   default=DEFAULT_DATA_DIR)
    parser.add_argument('--seq_len',       type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument('--pred_len',      type=int,   default=DEFAULT_PRED_LEN)
    # 模型参数
    parser.add_argument('--patch_size',    type=int,   default=DEFAULT_PATCH_SIZE)
    parser.add_argument('--stride',        type=int,   default=DEFAULT_STRIDE)
    parser.add_argument('--d_model',       type=int,   default=DEFAULT_D_MODEL)
    parser.add_argument('--c_ratio',       type=float, default=DEFAULT_C_RATIO)
    parser.add_argument('--c_ratio_mode',  type=str,   default=None,
                        choices=['fixed', 'learnable'],
                        help='c_ratio 模式: fixed（固定值）或 learnable（自动学习，默认取 GBS_CFG 值）')
    parser.add_argument('--dropout',       type=float, default=DEFAULT_DROPOUT)
    # 训练参数
    parser.add_argument('--batch_size',    type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs',        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--patience',      type=int,   default=DEFAULT_PATIENCE)
    # 工具参数
    parser.add_argument('--no_save',       action='store_true',
                        help='跳过 results.txt 写入并删除 checkpoint（用于网格测试）')

    args = parser.parse_args()
    train(args)
