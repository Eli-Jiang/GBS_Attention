# config.py
import torch

class Config:
    # 数据集相关参数
    c_in = 1          # 输入特征数
    c_out = 1         # 输出特征数
    seq_len = 6      # 历史滑动窗口长度
    pred_len = 2     # 预测未来长度

    # 模型架构参数
    d_model = 6      # 隐藏层维度
    n_heads = 1      # 注意力头数（确保 d_model 能被 n_heads 整除）
    d_ff = 128        # FFN 中间层维度
    n_layers = 2      # TransformerBlock 的堆叠层数
    dropout = 0.1     # 防止过拟合的随机失活率

    # 训练超参数
    batch_size = 1
    lr = 3e-4         # 经典的 AdamW 初始学习率
    epochs = 10        # 先跑 5 个 Epoch 试试水
    device = "cuda" if torch.cuda.is_available() else "cpu"