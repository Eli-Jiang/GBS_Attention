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

#  时序专用的 Token Embedding
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        #  改用 nn.Linear，它只作用于最后一个维度（特征维），对序列长度完全不敏感
        self.value_embedding = nn.Linear(c_in, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入 x 维度: [Batch_Size, Seq_Len, c_in]
        
        # 直接做线性映射，不再需要 permute 转置，直接变成 [Batch_Size, Seq_Len, d_model]
        x = self.value_embedding(x)
        
        out = self.pos_embedding(x)
        return self.dropout(out)