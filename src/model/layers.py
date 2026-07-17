import torch
import torch.nn as nn
from .gbs_attention import GBSAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 第一层：把维度从 d_model（比如 512）升到 d_ff（通常是 4 * d_model，比如 2048）
        self.linear_1 = nn.Linear(d_model, d_ff)
        # 激活函数，现代 Transformer 常用 GELU，经典的是 ReLU
        self.activation = nn.GELU()
        # 第二层：把维度降回 d_model
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x 的维度通常是: [Batch_Size, Seq_Len, d_model]
        return self.dropout(self.linear_2(self.activation(self.linear_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 引入你之前写好的 Attention（这里假设你的类名叫 MultiHeadAttention）
        self.attention = GBSAttention()
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 两层 LayerNorm，分别用于 Attention 和 FFN 之前
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 1. Attention 子层 (Pre-LN + 残差连接)
        # 先对输入 x 做 LayerNorm，再进 Attention
        attn_out = self.attention(self.norm_1(x))
        # 残差连接：把原始的 x 加回来
        x = x + self.dropout(attn_out)
        
        # 2. FFN 子层 (Pre-LN + 残差连接)
        ffn_out = self.feed_forward(self.norm_2(x))
        # 再次残差连接
        x = x + self.dropout(ffn_out)
        
        return x