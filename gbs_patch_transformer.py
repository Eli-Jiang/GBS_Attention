"""
gbs_patch_transformer.py
------------------------
PatchTransformer with GBS Attention (replaces softmax self-attention).

Architecture (channel-independent):
    Input (B, L, C)
    → Patch: (B*C, N, patch_size)  [N = L/patch_size = 6]
    → Linear embed: (B*C, N, d_model)
    → GBSAttentionLayer (B-matrix physics, fully differentiable)
    → FFN + LayerNorm
    → Linear head → (B, pred_len, C)

Usage:
    python gbs_patch_transformer.py --dataset ETTh1 --epochs 20
    python gbs_patch_transformer.py --dataset exchange --c_ratio 0.5
    python gbs_patch_transformer.py --dataset weather --d_model 64
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloader
from gbs_attention import GBSAttentionLayer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GBSPatchTransformer(nn.Module):
    """
    PatchTransformer with GBS Attention replacing standard softmax attention.

    Args:
        seq_len   : Input sequence length (default 96).
        pred_len  : Prediction horizon (default 24).
        num_features: Number of variables C.
        patch_size: Size of each patch P (default 16, giving N=6 patches).
        stride    : Stride for patching (default 16, non-overlapping).
        d_model   : Feature dimension (default 32).
        c_ratio   : GBS squeezing strength (default 0.3, matches precode).
        dropout   : Dropout rate (default 0.1).
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        num_features: int = 7,
        patch_size: int = 16,
        stride: int = 16,
        d_model: int = 32,
        c_ratio: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.num_patches = (seq_len - patch_size) // stride + 1  # N = 6

        # Patch embedding
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # GBS Attention (replaces nn.TransformerEncoderLayer's MHA)
        self.attn = GBSAttentionLayer(
            nmode=self.num_patches,
            d_model=d_model,
            c_ratio=c_ratio
        )

        # Feed-forward sublayer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Prediction head
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

        # Init positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)

        Returns:
            out: (B, pred_len, C)
        """
        B, L, C = x.shape

        # Channel-independent: merge B and C into batch
        x_ci = x.transpose(1, 2).reshape(B * C, L)

        # Patching: (B*C, N, patch_size)
        patches = x_ci.unfold(-1, self.patch_size, self.stride)

        # Embedding: (B*C, N, d_model)
        enc = self.patch_proj(patches) + self.pos_embed

        # GBS Attention block (pre-norm)
        enc = enc + self.drop(self.attn(self.norm1(enc)))

        # FFN block (pre-norm)
        enc = enc + self.drop(self.ffn(self.norm2(enc)))

        # Flatten + predict
        enc = enc.reshape(B * C, -1)      # (B*C, N*d_model)
        out = self.head(enc)              # (B*C, pred_len)
        out = out.reshape(B, C, -1).transpose(1, 2)  # (B, pred_len, C)

        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    r2 = 1.0 - np.sum((trues - preds) ** 2) / (
        np.sum((trues - trues.mean(axis=0)) ** 2) + 1e-9
    )
    return total_loss / len(dataloader.dataset), mse, mae, r2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataloaders
    train_loader, _, _ = get_dataloader(
        args.data_dir, args.dataset, 'train', args.batch_size, args.seq_len, args.pred_len
    )
    val_loader, _, _ = get_dataloader(
        args.data_dir, args.dataset, 'val', args.batch_size, args.seq_len, args.pred_len
    )
    test_loader, _, _ = get_dataloader(
        args.data_dir, args.dataset, 'test', args.batch_size, args.seq_len, args.pred_len
    )

    dummy_x, _ = next(iter(train_loader))
    num_features = dummy_x.shape[2]
    print(f"Dataset: {args.dataset} | Features: {num_features} | "
          f"Patches: N={(args.seq_len - args.patch_size) // args.stride + 1} | "
          f"c_ratio: {args.c_ratio}")

    model = GBSPatchTransformer(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_features=num_features,
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        c_ratio=args.c_ratio,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_cnt = 0
    save_path = f"{args.dataset}_gbs_patch_transformer_c{args.c_ratio}.pt"

    print("Training...")
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
        train_loss /= len(train_loader.dataset)

        val_loss, val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)
        t_epoch = time.time() - t0

        print(f"Epoch {epoch+1:02d}/{args.epochs} | {t_epoch:.1f}s | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val R2: {val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    model.load_state_dict(torch.load(save_path))
    _, test_mse, test_mae, test_r2 = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*55)
    print(f"FINAL TEST RESULTS — {args.dataset} (GBSPatchTransformer)")
    print("="*55)
    print(f"Test MSE : {test_mse:.6f}")
    print(f"Test MAE : {test_mae:.6f}")
    print(f"Test R2  : {test_r2:.6f}")
    print(f"c_ratio  : {args.c_ratio}")
    print("="*55 + "\n")

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset} | GBSPatchTransformer\n")
        f.write(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f} | Test R2: {test_r2:.6f}\n")
        f.write(f"Config: seq_len={args.seq_len}, pred_len={args.pred_len}, "
                f"patch_size={args.patch_size}, d_model={args.d_model}, c_ratio={args.c_ratio}\n")
        f.write("-"*55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GBS PatchTransformer training')
    parser.add_argument('--dataset',       type=str,   default='ETTh1',
                        choices=['ETTh1', 'exchange', 'weather'])
    parser.add_argument('--data_dir',      type=str,   default='./data')
    parser.add_argument('--seq_len',       type=int,   default=96)
    parser.add_argument('--pred_len',      type=int,   default=24)
    parser.add_argument('--patch_size',    type=int,   default=16)
    parser.add_argument('--stride',        type=int,   default=16)
    parser.add_argument('--d_model',       type=int,   default=32)
    parser.add_argument('--c_ratio',       type=float, default=0.3)
    parser.add_argument('--dropout',       type=float, default=0.1)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience',      type=int,   default=5)

    args = parser.parse_args()
    train(args)
