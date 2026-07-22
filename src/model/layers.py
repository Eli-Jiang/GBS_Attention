import torch
import torch.nn as nn
from .gbs_attention import GBSAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 第一层：把维度从 d_model 升到 d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        # 激活函数
        self.activation = nn.GELU()
        # 第二层：把维度降回 d_model
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear_2(self.activation(self.linear_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, shots=4096, use_gbs=False):
        super().__init__()
        # use_gbs: True=GBS量子采样, False=GBS理论计算, "standard"=标准Attention
        self.attention = GBSAttention(d_model=d_model, shots=shots, use_gbs=use_gbs)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 两层 LayerNorm，分别用于 Attention 和 FFN 之前
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Attention 子层 (Pre-LN + 残差连接)
        attn_out = self.attention(self.norm_1(x))
        x = x + self.dropout(attn_out)

        # 2. FFN 子层 (Pre-LN + 残差连接)
        ffn_out = self.feed_forward(self.norm_2(x))
        x = x + self.dropout(ffn_out)

        return x
