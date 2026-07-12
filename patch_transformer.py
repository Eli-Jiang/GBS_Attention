"""
PatchTransformer 模型训练与评估脚本

本代码实现了基于 PatchTransformer 的多变量时间序列预测模型，并提供了训练、验证和测试的完整流程。
PatchTransformer 通过将时间序列划分成多个 patch，利用 Transformer 架构捕捉局部特征和全局依赖关系。

使用说明：
- 运行本脚本需要提供对应的数据集，默认数据集为 ETTh1。
- 可以通过命令行参数指定不同的超参数，如：--seq_len (输入序列长度)、--pred_len (预测序列长度) 等。
- 更多详细用法和参数说明，请使用 `python patch_transformer.py -h` 查看。
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# 全局默认配置 (Global Default Configurations)
# ==========================================
# 方便其他人在此处统一修改本地数据路径或默认参数
DEFAULT_DATA_DIR = './data'
DEFAULT_DATASET = 'ETTh1'
DEFAULT_SEQ_LEN = 6
DEFAULT_PRED_LEN = 6
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
# ==========================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from data_loader import get_dataloader

class PatchTransformer(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, num_features=7, patch_size=16, stride=16, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        
        # Number of patches
        self.num_patches = (seq_len - patch_size) // stride + 1
        
        # Projection layer: patch_size -> d_model
        self.patch_proj = nn.Linear(patch_size, d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head: predict pred_len
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        
    def forward(self, x):
        # x shape: (B, L, C)
        B, L, C = x.shape
        
        # Reshape to channel-independent: (B * C, L)
        x_ci = x.transpose(1, 2).reshape(B * C, L)
        
        # Patching: (B * C, num_patches, patch_size)
        patches = x_ci.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        # Linear projection
        enc_in = self.patch_proj(patches)  # (B * C, num_patches, d_model)
        
        # Add positional embedding
        enc_in = enc_in + self.pos_embed
        
        # Transformer
        enc_out = self.transformer(enc_in)  # (B * C, num_patches, d_model)
        
        # Flatten
        enc_out = enc_out.reshape(B * C, -1)  # (B * C, num_patches * d_model)
        
        # Predict
        out = self.head(enc_out)  # (B * C, pred_len)
        
        # Reshape back to (B, pred_len, C)
        out = out.reshape(B, C, self.pred_len).transpose(1, 2)
        
        return out

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    preds = []
    trues = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            preds.append(outputs.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Calculate metrics
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    
    # R2 score
    unexplained_variance = np.sum((trues - preds) ** 2)
    total_variance = np.sum((trues - np.mean(trues, axis=0)) ** 2)
    r2 = 1 - unexplained_variance / (total_variance + 1e-9)
    
    return total_loss / total_samples, mse, mae, r2

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load data
    train_loader, mean, std = get_dataloader(args.data_dir, args.dataset, flag='train', batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len)
    val_loader, _, _ = get_dataloader(args.data_dir, args.dataset, flag='val', batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len)
    test_loader, _, _ = get_dataloader(args.data_dir, args.dataset, flag='test', batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len)
    
    # Get feature count
    dummy_x, _ = next(iter(train_loader))
    num_features = dummy_x.shape[2]
    print(f"Dataset: {args.dataset} | Features: {num_features}")
    
    # Initialize model
    model = PatchTransformer(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_features=num_features,
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    early_stop_cnt = 0
    
    print("Start training...")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0
        train_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
        train_loss /= train_samples
        val_loss, val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)
        
        t_epoch = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Time: {t_epoch:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R2: {val_r2:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.dataset}_patch_transformer.pt")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # Load best model for testing
    model.load_state_dict(torch.load(f"{args.dataset}_patch_transformer.pt"))
    test_loss, test_mse, test_mae, test_r2 = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*50)
    print(f"FINAL TEST RESULTS FOR DATASET: {args.dataset}")
    print("="*50)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R2 : {test_r2:.6f}")
    print("="*50 + "\n")
    
    if not args.no_save:
        # Save test results to txt file
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"Dataset: {args.dataset} | PatchTransformer\n")
            f.write(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f} | Test R2: {test_r2:.6f}\n")
            f.write(f"Configuration: seq_len={args.seq_len}, pred_len={args.pred_len}, patch_size={args.patch_size}, d_model={args.d_model}\n")
            f.write("-"*50 + "\n")
    else:
        # Clean up the checkpoint file
        checkpoint_path = f"{args.dataset}_patch_transformer.pt"
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Cleaned up checkpoint: {checkpoint_path}")
            
    print("\n[提示] 运行结束！")
    print("相关用法说明：")
    print("1. 你可以通过追加参数来测试不同的序列长度，例如：python patch_transformer.py --seq_len 192 --pred_len 96")
    print("2. 增加 d_model 或 nhead 可能会提升效果，但同时也会增加计算量，例如：python patch_transformer.py --d_model 64 --nhead 4")
    print("3. 如需快速调试而不保存模型或写入文件，可以使用 --no_save 参数。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, choices=['ETTh1', 'exchange', 'weather'])
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument('--pred_len', type=int, default=DEFAULT_PRED_LEN)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--no_save', action='store_true', help='Do not save results.txt and delete the generated .pt checkpoint file after testing')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("🚀 运行参数配置 (Run Configurations):")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("="*50 + "\n")
    
    train(args)
