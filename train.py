import torch
import torch.nn as nn
import numpy as np
from config import Config
# 💡 注意看，全面加上了 src.
from src.model.transformer import TimeSeriesTransformer
from src.data.dataset import get_dataloader  # 你的 dataset 好像在 src/data/ 目录下

def main():
    # 1. 加载配置
    cfg = Config()
    print(f"正在使用设备: {cfg.device}")

    # 2. 准备你的真实数据
    # 💡 提示：这里假设你使用的是 Numpy 数组 [总时间步, 特征数]。
    # 如果你的数据在 csv 里，可以用 np.loadtxt() 或 pandas 读进来，确保 shape 是 [N, c_in]
    # 我们这里先用随机数模拟真实数据，换成你的数据即可：
    raw_data = np.random.randn(12, cfg.c_in) 
    
    # 3. 实例化 DataLoader
    train_loader = get_dataloader(
        data=raw_data, 
        seq_len=cfg.seq_len, 
        pred_len=cfg.pred_len, 
        batch_size=cfg.batch_size, 
        shuffle=True
    )

    # 4. 实例化你的新型 Transformer 模型
    model = TimeSeriesTransformer(
        c_in=cfg.c_in,
        c_out=cfg.c_out,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout
    ).to(cfg.device)

    # 5. 定义损失函数和优化器
    criterion = nn.MSELoss() # 时序预测的标准损失：均方误差
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # 6. 正式进入训练循环 (Training Loop)
    print("--- 🦾 炼丹炉正式点火 ---")
    model.train() # 将模型设置为训练模式（启用 Dropout 等）
    
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # batch_x: [Batch_Size, seq_len, c_in]
            # batch_y: [Batch_Size, pred_len, c_out]
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            
            # 1. 前向传播
            outputs = model(batch_x)
            
            # 2. 计算损失值
            loss = criterion(outputs, batch_y)
            
            # 3. 反向传播与优化
            optimizer.zero_grad() # 清空上一轮的梯度
            loss.backward()       # 反向传播算梯度
            optimizer.step()      # 更新模型权重
            
            epoch_loss += loss.item()
            
        # 打印当前 Epoch 的平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | Train MSE Loss: {avg_loss:.6f}")

    print("--- 🎉 训练完成！ ---")
    
    # 7. 保存模型权重
    torch.save(model.state_dict(), "best_quantum_transformer.pth")
    print("模型权重已保存至: best_quantum_transformer.pth")

if __name__ == "__main__":
    main()