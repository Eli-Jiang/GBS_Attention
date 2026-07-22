import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from config import Config

from src.model.transformer import TimeSeriesTransformer
from src.data.dataset import get_dataloader
from sklearn.preprocessing import MinMaxScaler


def fix_column_names(df):
    """修复列名中的乱码字符"""
    new_columns = []
    for col in df.columns:
        # 替换常见的乱码字符
        col = col.replace('\ufffd', '2')  # 替换�为2 (表示平方)
        col = col.replace('µ', 'u')       # 微符号替换为u
        new_columns.append(col)
    df.columns = new_columns
    return df


def prepare_data(cfg):
    """准备数据"""
    # 加载数据
    df = pd.read_csv(cfg.data_path)
    df = fix_column_names(df)  # 修复乱码列名
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    raw_data = df.values.astype(np.float32)
    cfg.c_in = raw_data.shape[1]

    # 处理缺失值
    raw_data = np.where(raw_data < -9000, np.nan, raw_data)
    col_means = np.nanmean(raw_data, axis=0)
    for i in range(raw_data.shape[1]):
        raw_data[np.isnan(raw_data[:, i]), i] = col_means[i]

    # 归一化
    scaler = MinMaxScaler()
    scaler.fit(raw_data[cfg.train_start:cfg.train_end])
    train_data = scaler.transform(raw_data[cfg.train_start:cfg.train_end])
    val_data = scaler.transform(raw_data[cfg.val_start:cfg.val_end])

    # 根据配置决定是否使用差分
    if cfg.use_diff:
        train_data = np.diff(train_data, axis=0)
        val_data = np.diff(val_data, axis=0)
        print("使用一阶差分")

    return cfg, train_data, val_data, scaler


def train_model(cfg, train_data, val_data, use_gbs, model_path):
    """训练单个模型"""
    # 创建模型
    model = TimeSeriesTransformer(
        c_in=cfg.c_in,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        shots=cfg.shots,
        use_gbs=use_gbs
    ).to(cfg.device)

    gbs_label = "gbs(dq)" if use_gbs is True else "gbs(theory)" if use_gbs is False else "standard"
    print(f"\n=== 训练 {gbs_label} 模型 ===")

    # DataLoader
    train_loader = get_dataloader(train_data, cfg.seq_len, cfg.pred_len, cfg.batch_size, shuffle=True)
    val_loader = get_dataloader(val_data, cfg.seq_len, cfg.pred_len, cfg.batch_size, shuffle=False) if len(val_data) > cfg.seq_len + cfg.pred_len else None

    # 优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 训练
    model.train()
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(cfg.device)
                    batch_y = batch_y.to(cfg.device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Epoch [{epoch+1}/{cfg.epochs}] | Train: {avg_loss:.6f} | Val: {avg_val_loss:.6f} | 保存最佳")
            else:
                print(f"Epoch [{epoch+1}/{cfg.epochs}] | Train: {avg_loss:.6f} | Val: {avg_val_loss:.6f}")

            scheduler.step(avg_val_loss)
            model.train()

    return loss_history, val_loss_history


def main():
    # 1. 加载配置
    cfg = Config()
    print(f"正在使用设备: {cfg.device}")

    # 准备数据（只需一次）
    cfg, train_data, val_data, scaler = prepare_data(cfg)

    # 保存scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"数据归一化完成，scaler已保存")
    print(f"训练数据形状: {train_data.shape}")
    print(f"验证数据形状: {val_data.shape}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # 根据配置决定训练哪些模型
    current_use_gbs = cfg.use_gbs
    current_path = cfg.model_path

    # 训练当前配置的模型
    if current_use_gbs is not "standard":
        train_model(cfg, train_data, val_data, current_use_gbs, current_path)

    # 如果需要对比standard
    if cfg.compare_standard:
        # 训练standard模型
        print("\n" + "="*50)
        train_model(cfg, train_data, val_data, "standard", "best_standard_transformer.pth")

    print("\n---  全部训练完成！ ---")


if __name__ == "__main__":
    main()
