# inference.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from src.model.transformer import TimeSeriesTransformer

def main():
    cfg = Config()
    
    # 1. 加载真实原始绝对值数据
    df = pd.read_csv("data/numbers.csv")
    raw_data = df.values.astype(np.float32)  # [总长度, 特征数]
    
    # 因为差分会使长度减 1，为了让差分后的输入长度依然是完整的 seq_len
    # 我们切取 X 时必须在前面“多拿一个点”来做差分
    total_needed = cfg.seq_len + cfg.pred_len
    if len(raw_data) < total_needed + 5:
        print(f"数据太短了！至少需要 {total_needed + 5} 行数据。")
        return
        
    # 2. 抽出测试滑窗数据（保持和你之前的 -5 偏移一致）
    # 真实未来绝对值 Y（用于最后对比）
    true_y_raw = raw_data[-cfg.pred_len-5 : -5] 
    
    # 历史绝对值 X（为了多留一个点做差分，前面变成 -total_needed-6）
    test_x_with_prev = raw_data[-total_needed-6 : -cfg.pred_len-5] 
    
    # 3. 🚨【核心步骤】计算一阶差分，作为模型的输入
    # np.diff 之后，长度会从 seq_len + 1 缩减回标准的 seq_len
    test_x_diff = np.diff(test_x_with_prev, axis=0)  
    
    # 4. 转换成 PyTorch Tensor 格式并加上 Batch 维度 -> [1, seq_len, c_in]
    # 💡 提示：因为 train.py 没用 scaler，这里直接转为 tensor 送进模型
    input_tensor = torch.tensor(test_x_diff).unsqueeze(0).to(cfg.device)
    
    # 5. 初始化模型并加载权重
    model = TimeSeriesTransformer(
        c_in=cfg.c_in, c_out=cfg.c_out, seq_len=cfg.seq_len, pred_len=cfg.pred_len,
        d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff, n_layers=cfg.n_layers, dropout=cfg.dropout
    ).to(cfg.device)
    
    model.load_state_dict(torch.load("best_quantum_transformer.pth", map_location=cfg.device))
    model.eval()  # 切换到测试模式
    
    # 6. 模型推理（吐出的是未来的变化量）
    with torch.no_grad():
        output = model(input_tensor)  # [1, pred_len, c_out]
        pred_y_diff = output.squeeze(0).cpu().numpy()  # [pred_len, c_out]
        
    # 7. 🚨【绝对值还原】逆差分核心
    # 拿到历史输入数据的最后一天绝对值（注意：要从原本的 raw_data 里拿真正的最后一步）
    last_historical_value = raw_data[-cfg.pred_len-6, 0]  # 这就是 Forecast Start 分割线上的那个绝对值
    
    # 利用 np.cumsum 逐步累加模型预测出来的变化量
    pred_y_absolute = np.zeros_like(pred_y_diff)
    pred_y_absolute[:, 0] = last_historical_value + np.cumsum(pred_y_diff[:, 0])

    # 为了画历史线，我们取出历史对应的真实绝对值
    test_x_raw = raw_data[-total_needed-5 : -cfg.pred_len-5]

    # 8. 绘图对比
    plt.figure(figsize=(12, 6))
    
    history_len = len(test_x_raw)
    
    # 绘制历史绝对值
    plt.plot(range(history_len), test_x_raw[:, 0], label='History (Input X)', color='black', marker='o')
    
    # 时间轴完美衔接：从历史的最后一个索引延伸出去
    future_range = range(history_len - 1, history_len - 1 + cfg.pred_len + 1)
    
    # 把连接点拼到头部，防止线条断裂
    aligned_true_y = np.insert(true_y_raw[:, 0], 0, last_historical_value)
    aligned_pred_y = np.insert(pred_y_absolute[:, 0], 0, last_historical_value)
    
    # 绘制真实曲线和还原后的预测曲线
    plt.plot(future_range, aligned_true_y, label='True Future (Label Y)', color='green', marker='o')
    plt.plot(future_range, aligned_pred_y, label='Predicted Future (Restored Output)', color='red', linestyle='--', marker='x')
    
    # 加上分割线
    plt.axvline(x=history_len - 1, color='gray', linestyle=':', label='Forecast Start Split')
    
    plt.title('Quantum Attention Transformer - Differenced Forecasting')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 保存结果图
    plt.savefig('prediction_contrast.png')
    plt.close()
    print("📈 还原后的预测对比图已重新保存为 'prediction_contrast.png'")

if __name__ == "__main__":
    main()