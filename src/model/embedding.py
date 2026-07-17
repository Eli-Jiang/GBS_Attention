import torch
import torch.nn as nn
import math

# PositionalEncoding 保持不变（依然可以用正弦余弦表示时间先后顺序）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 💡 时序专用的 Token Embedding
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len=5000, dropout=0.1):
        """
        c_in: 输入的时序特征数 (比如只有一维时间序列就是 1，如果是多变量预测就是变数个数)
        d_model: 映射到的 Transformer 特征维度
        """
        super().__init__()
        # 用 nn.Linear 或 nn.Conv1d 把连续的特征值映射到 d_model 空间
        # 这里用 TokenEmbedding 的标准时序做法（一维卷积）
        self.value_embedding = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                         kernel_size=3, padding=1, padding_mode='circular')
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入 x 维度: [Batch_Size, Seq_Len, c_in]
        # Conv1d 期待的维度是 [Batch_Size, Channels, Seq_Len]，所以要转置一下
        x = self.value_embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 此时 x 维度变为: [Batch_Size, Seq_Len, d_model]
        out = self.pos_embedding(x)
        return self.dropout(out)