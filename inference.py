# inference.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

# 设置字体以支持Unicode字符（如µ²等）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from config import Config
from src.model.transformer import TimeSeriesTransformer
from src.utils.visualization import plot_prediction_comparison


def fix_column_names(df):
    """修复列名中的乱码字符"""
    new_columns = []
    for col in df.columns:
        col = col.replace('\ufffd', '2')  # 替换�为2
        col = col.replace('µ', 'u')       # 微符号替换为u
        new_columns.append(col)
    df.columns = new_columns
    return df


def calculate_metrics(true_vals, pred_vals):
    """计算评估指标"""
    mse = np.mean((true_vals - pred_vals) ** 2)
    mae = np.mean(np.abs(true_vals - pred_vals))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_vals - pred_vals) / (np.abs(true_vals) + 1e-8))) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def run_inference(cfg, model_path, use_gbs_mode):
    """运行推理并返回预测结果"""

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
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    raw_data = scaler.transform(raw_data)

    # 切分数据
    test_data = raw_data[cfg.test_start:cfg.test_end]
    true_y_raw = test_data[-cfg.pred_len:]
    test_x_with_prev = test_data[-cfg.pred_len - cfg.seq_len - 1: -cfg.pred_len]

    # 根据配置决定是否使用差分
    if cfg.use_diff:
        # 计算差分作为输入
        test_x_diff = np.diff(test_x_with_prev, axis=0)
        input_tensor = torch.tensor(test_x_diff).unsqueeze(0).to(cfg.device)
    else:
        # 直接使用原始数据
        input_tensor = torch.tensor(test_x_with_prev[-cfg.seq_len:]).unsqueeze(0).to(cfg.device)

    # 根据模型路径判断使用哪个d_ff
    # standard模型用d_ff=88（当前配置）, 其他用当前配置
    d_ff = cfg.d_ff

    # 加载模型
    model = TimeSeriesTransformer(
        c_in=cfg.c_in, seq_len=cfg.seq_len, pred_len=cfg.pred_len,
        d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=d_ff,
        n_layers=cfg.n_layers, dropout=cfg.dropout, shots=cfg.shots, use_gbs=use_gbs_mode
    ).to(cfg.device)

    model.load_state_dict(torch.load(model_path, map_location=cfg.device), strict=False)
    model.eval()

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        pred_y = output.squeeze(0).cpu().numpy()

    # 根据配置进行逆变换
    if cfg.use_diff:
        # 逆差分还原
        pred_y_absolute = np.zeros_like(pred_y)
        for i in range(pred_y.shape[1]):
            last_value = test_x_with_prev[-1, i]
            pred_y_absolute[:, i] = last_value + np.cumsum(pred_y[:, i])
        test_x_raw = test_x_with_prev[-cfg.seq_len:]
        last_hist = test_x_with_prev[-1, :]
    else:
        # 直接使用预测值
        pred_y_absolute = pred_y
        test_x_raw = test_x_with_prev[-cfg.seq_len:]
        last_hist = test_x_with_prev[-1, :]

    # 逆归一化
    pred_y_absolute = scaler.inverse_transform(pred_y_absolute)
    test_x_raw = scaler.inverse_transform(test_x_raw)
    true_y_raw_orig = scaler.inverse_transform(true_y_raw)
    last_hist_orig = scaler.inverse_transform(last_hist.reshape(1, -1))[0, :]

    return test_x_raw, true_y_raw_orig, pred_y_absolute, last_hist_orig, numeric_cols

def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"测试数据范围: [{cfg.test_start}:{cfg.test_end}]")

    # 当前配置的模型
    use_gbs = cfg.use_gbs
    # 标签: True=gbs(dq), False=gbs(theory)
    if use_gbs is True:
        current_mode = "gbs(dq)"
        model_label = "gbs(dq)"
    else:
        current_mode = "gbs(theory)"
        model_label = "gbs(theory)"
    current_path = "best_quantum_transformer.pth"

    if cfg.compare_standard:
        # 对比模式
        print(f"\n=== 运行 {model_label} 模型 ===")
        test_x_raw, true_y_orig, pred_y_abs, last_hist, cols = run_inference(cfg, current_path, use_gbs)

        print(f"\n=== 运行 standard 模型 ===")
        test_x_raw2, true_y_orig2, pred_y_std, _, _ = run_inference(cfg, "best_standard_transformer.pth", "standard")

        # 计算指标
        print("\n" + "="*60)
        print("评估指标汇总 (所有特征平均)")
        print("="*60)

        avg_theory = {'MSE': 0, 'MAE': 0, 'RMSE': 0, 'MAPE': 0}
        avg_standard = {'MSE': 0, 'MAE': 0, 'RMSE': 0, 'MAPE': 0}

        for i, col in enumerate(cols):
            m1 = calculate_metrics(true_y_orig[:, i], pred_y_abs[:, i])
            m2 = calculate_metrics(true_y_orig2[:, i], pred_y_std[:, i])
            for k in avg_theory:
                avg_theory[k] += m1[k]
                avg_standard[k] += m2[k]
            print(f"{col}: {current_mode} MSE={m1['MSE']:.6f}, standard MSE={m2['MSE']:.6f}")

        for k in avg_theory:
            avg_theory[k] /= len(cols)
            avg_standard[k] /= len(cols)

        print(f"\n{model_label} 平均: MSE={avg_theory['MSE']:.6f}, MAE={avg_theory['MAE']:.6f}")
        print(f"STANDARD 平均: MSE={avg_standard['MSE']:.6f}, MAE={avg_standard['MAE']:.6f}")

        # 保存详细结果到CSV
        rows = []
        for i, col in enumerate(cols):
            m1 = calculate_metrics(true_y_orig[:, i], pred_y_abs[:, i])
            m2 = calculate_metrics(true_y_orig2[:, i], pred_y_std[:, i])
            rows.append({
                'feature': col,
                f'{current_mode}_MSE': m1['MSE'],
                f'{current_mode}_MAE': m1['MAE'],
                f'{current_mode}_RMSE': m1['RMSE'],
                f'{current_mode}_MAPE': m1['MAPE'],
                'standard_MSE': m2['MSE'],
                'standard_MAE': m2['MAE'],
                'standard_RMSE': m2['RMSE'],
                'standard_MAPE': m2['MAPE']
            })
        df_results = pd.DataFrame(rows)
        df_results.to_csv(f"{cfg.output_dir}/metrics_comparison.csv", index=False)
        print(f"\n指标已保存至: {cfg.output_dir}/metrics_comparison.csv")

        # 绘制全部特征
        n_show = len(cols)
        n_cols = 3
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
        if n_show == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        history_len = len(test_x_raw)
        future_range = range(history_len - 1, history_len - 1 + cfg.pred_len + 1)

        for i in range(n_show):
            row, col_idx = i // n_cols, i % n_cols
            ax = axes[row, col_idx]
            col = cols[i]

            ax.plot(range(history_len), test_x_raw[:, i], label='History', color='black', marker='o', markersize=2, alpha=0.7)
            aligned_true = np.insert(true_y_orig[:, i], 0, last_hist[i])
            ax.plot(future_range, aligned_true, label='True', color='green', marker='o', markersize=3)
            aligned_pred_cur = np.insert(pred_y_abs[:, i], 0, last_hist[i])
            ax.plot(future_range, aligned_pred_cur, label=current_mode, color='red', linestyle='--', marker='x', markersize=3)
            aligned_pred_std = np.insert(pred_y_std[:, i], 0, last_hist[i])
            ax.plot(future_range, aligned_pred_std, label='standard', color='blue', linestyle=':', marker='+', markersize=3)

            m1 = calculate_metrics(true_y_orig[:, i], pred_y_abs[:, i])
            m2 = calculate_metrics(true_y_orig2[:, i], pred_y_std[:, i])
            ax.set_title(f'{col} - {current_mode} MSE:{m1["MSE"]:.4f} vs standard MSE:{m2["MSE"]:.4f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=history_len-1, color='gray', linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{cfg.output_dir}/output.png", dpi=150)
        plt.close()
        print(f"对比图已保存至: {cfg.output_dir}/output.png")

    else:
        # 单模型模式
        print(f"\n=== 运行 {model_label} 模型 ===")
        test_x_raw, true_y_orig, pred_y_abs, last_hist, cols = run_inference(cfg, current_path, use_gbs)

        # 计算并打印指标
        print("\n评估指标:")
        rows = []
        for i, col in enumerate(cols):
            m = calculate_metrics(true_y_orig[:, i], pred_y_abs[:, i])
            print(f"  {col}: MSE={m['MSE']:.6f}, MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, MAPE={m['MAPE']:.2f}%")
            rows.append({'feature': col, **m})

        df_results = pd.DataFrame(rows)
        df_results.to_csv(f"{cfg.output_dir}/metrics_{current_mode}.csv", index=False)
        print(f"\n指标已保存至: {cfg.output_dir}/metrics_{current_mode}.csv")

        # 绘图
        save_path = f"{cfg.output_dir}/output.png"
        plot_prediction_comparison(test_x_raw, true_y_orig, pred_y_abs, cols, cfg, save_path)

if __name__ == "__main__":
    main()
