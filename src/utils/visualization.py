import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# 设置字体以支持Unicode字符（如µ²等）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_prediction_comparison(test_x_raw, true_y_raw_orig, pred_y_absolute, numeric_cols, cfg, save_path=None):
    """
    绘制预测对比图（支持多特征）

    Args:
        test_x_raw: 历史数据（已逆归一化）
        true_y_raw_orig: 真实未来值（已逆归一化）
        pred_y_absolute: 预测值（已逆归一化）
        numeric_cols: 特征列名
        cfg: 配置对象
        save_path: 保存路径
    """
    n_features = pred_y_absolute.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    history_len = len(test_x_raw)
    future_range = range(history_len - 1, history_len - 1 + cfg.pred_len + 1)
    last_hist = test_x_raw[-1, :]

    for i in range(n_features):
        ax = axes[i]

        # 绘制历史
        ax.plot(range(history_len), test_x_raw[:, i], label='History', color='black', marker='o', markersize=3)

        # 对齐点
        aligned_true = np.insert(true_y_raw_orig[:, i], 0, last_hist[i])
        aligned_pred = np.insert(pred_y_absolute[:, i], 0, last_hist[i])

        # 绘制真实和预测
        ax.plot(future_range, aligned_true, label='True', color='green', marker='o', markersize=3)
        ax.plot(future_range, aligned_pred, label='Pred', color='red', linestyle='--', marker='x', markersize=3)

        ax.axvline(x=history_len - 1, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(f'Feature {i}: {numeric_cols[i]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存至: {save_path}")

    plt.close()

def plot_contrast(true_vals, pred_vals, labels, title, save_path):
    """
    绘制单个特征的对比图

    Args:
        true_vals: 真实值数组
        pred_vals: 预测值数组
        labels: [历史标签, 真实标签, 预测标签]
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))

    history_len = len(labels[0])
    future_range = range(history_len - 1, history_len - 1 + len(labels[1]) + 1)

    plt.plot(range(history_len), labels[0], label=labels[0], color='black', marker='o', markersize=3)
    plt.plot(future_range, np.insert(labels[1], 0, labels[0][-1]), label=labels[1], color='green', marker='o', markersize=3)
    plt.plot(future_range, np.insert(labels[2], 0, labels[0][-1]), label=labels[2], color='red', linestyle='--', marker='x', markersize=3)

    plt.axvline(x=history_len - 1, color='gray', linestyle=':', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"对比图已保存至: {save_path}")
