# inference.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from src.model.transformer import TimeSeriesTransformer
from sklearn.preprocessing import MinMaxScaler

def main():
    cfg = Config()
    
    # 1. 加载你的真实数据（这里假设你依然用 numbers.csv 测试，实际可换成别的数据）
    df = pd.read_csv("data/numbers.csv")
    raw_data = df.values.astype(np.float32)  # [总长度, 1]
    
    # ⚠️ 提示：如果你在 train.py 里用了归一化（如 StandardScaler），
    # 在这里也必须用同样的 scaler 进行 fit_transform！
    scaler = MinMaxScaler()
    raw_data = scaler.fit_transform(raw_data)
    
    # 2. 抽出最后一组完整的滑窗数据来进行推理测试
    # 我们需要 seq_len 个历史数据，后面紧跟 pred_len 个未来的真实标签数据用来做对比
    total_needed = cfg.seq_len + cfg.pred_len
    if len(raw_data) < total_needed:
        print(f"数据太短了！至少需要 {total_needed} 行数据。")
        return
        
    # 取出倒数第 total_needed 到 倒数第 pred_len 之间的作为输入 X
    # 比如总长15，seq_len=6, pred_len=2。输入就是索引 [7:13] 的 6 个数
    test_x = raw_data[-total_needed : -cfg.pred_len] 
    # 最后的 pred_len 个数作为真实的未来标签 Y
    true_y = raw_data[-cfg.pred_len :] 
    
    # 3. 转换成 PyTorch Tensor 格式并加上 Batch 维度 -> [1, seq_len, c_in]
    input_tensor = torch.tensor(test_x).unsqueeze(0).to(cfg.device)
    
    # 4. 初始化模型并加载权重
    model = TimeSeriesTransformer(
        c_in=cfg.c_in, c_out=cfg.c_out, seq_len=cfg.seq_len, pred_len=cfg.pred_len,
        d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff, n_layers=cfg.n_layers, dropout=cfg.dropout
    ).to(cfg.device)
    
    model.load_state_dict(torch.load("best_quantum_transformer.pth", map_location=cfg.device))
    model.eval()  # 💡 切换到测试模式（关闭 Dropout）
    
    # 5. 模型推理（不追踪梯度）
    with torch.no_grad():
        # output shape: [1, pred_len, c_out]
        output = model(input_tensor)
        # 去掉 batch 维度，转回 NumPy array -> [pred_len, c_out]
        pred_y = output.squeeze(0).cpu().numpy()
        
    # 💡 提示：如果你之前用了归一化，请在这里执行：
    pred_y = scaler.inverse_transform(pred_y)
    true_y = scaler.inverse_transform(true_y)
    test_x = scaler.inverse_transform(test_x)

    # 6. 绘图对比：历史数据 + 真实未来 + 预测未来
    plt.figure(figsize=(12, 6))
    
    # 绘制历史输入数据（用实线表示）
    history_len = len(test_x)
    plt.plot(range(history_len), test_x.flatten(), label='History (Input X)', color='black', marker='o')
    
    # 绘制真实未来数据（用绿色实线）
    future_range = range(history_len, history_len + cfg.pred_len)
    plt.plot(future_range, true_y.flatten(), label='True Future (Label Y)', color='green', marker='o')
    
    # 绘制模型预测数据（用红色虚线，方便对比）
    plt.plot(future_range, pred_y.flatten(), label='Predicted Future (Output)', color='red', linestyle='--', marker='x')
    
    # 加上分割线区分历史和未来
    plt.axvline(x=history_len - 1, color='gray', linestyle=':', label='Forecast Start Split')
    
    plt.title('Quantum Attention Transformer - Forecasting Result')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 保存结果图
    plt.savefig('prediction_contrast.png')
    plt.close()
    print("📈 预测对比图已保存为 'prediction_contrast.png'，快去打开看看准不准！")

if __name__ == "__main__":
    main()