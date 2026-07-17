import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepquantum as dq
from .utils import encode_graph_to_unitary, gbs_sample

class GBSAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(6, 6)
        self.key_proj = nn.Linear(6, 6)
        self.value_proj = nn.Linear(6, 6)

    def gbs(self, A):
        T, r = encode_graph_to_unitary(A)
        counts = gbs_sample(T, r)
        return counts

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        query = self.query_proj(x)  # (batch_size, seq_len, input_dim)
        key = self.key_proj(x)      # (batch_size, seq_len, input_dim)
        value = self.value_proj(x)  # (batch_size, seq_len, input_dim)
        
        # Compute attention scores (带梯度的 PyTorch Tensor)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
        
        # =================【最简洁的修改点】=================
        # 1. 降维到 2D 并解绑梯度转成 NumPy
        s_np = scores.squeeze(0).detach().cpu().numpy()
        # 2. 一行搞定：强制对称，且对角线置 0（满足你的 assert 要求）
        scores_for_gbs = (s_np + s_np.T) / 2.0
        np.fill_diagonal(scores_for_gbs, 0)
        
        # 3. 传入处理好的 NumPy 数组
        attention_weights = self.gbs(scores_for_gbs) # (seq_len, seq_len)
        # ===================================================
        
        # Compute the output
        output = torch.matmul(attention_weights, value)  # (batch_size, seq_len, input_dim)
        return output