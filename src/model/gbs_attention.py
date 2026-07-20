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
            batch_size, seq_len, _ = x.shape
            
            query = self.query_proj(x)  # (batch_size, seq_len, input_dim)
            key = self.key_proj(x)      # (batch_size, seq_len, input_dim)
            value = self.value_proj(x)  # (batch_size, seq_len, input_dim)
            
            # 1. 计算初始注意力分数 (带梯度的 PyTorch Tensor)
            scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
            
            # 2. 循环处理每一个 Batch，确保传入 GBS 的一定是 2D 矩阵
            attn_weights_list = []
            scores_np = scores.detach().cpu().numpy() # 先转成 NumPy 形式
            
            for b in range(batch_size):
                s_np = scores_np[b] # 💡 显式取出第 b 个样本，这 100% 是一个 (seq_len, seq_len) 的 2D 矩阵
                
                # 满足你的 assert 要求：强制对称且对角线置 0
                scores_for_gbs = (s_np + s_np.T) / 2.0
                np.fill_diagonal(scores_for_gbs, 0)
                
                # 调用你的量子采样，得到 (seq_len, seq_len) 的矩阵
                gbs_weight = self.gbs(scores_for_gbs) 
                attn_weights_list.append(gbs_weight)
                
            # 将整个 Batch 的量子权重重新打包回 PyTorch Tensor
            # 此时形式为无梯度的常量 Tensor: (batch_size, seq_len, seq_len)
            attention_weights_gbs = torch.tensor(np.array(attn_weights_list), dtype=scores.dtype, device=scores.device)
            
            # =================【🔥 核心魔法：STE 直通估计器救活梯度】=================
            # 前向传播用完全精准的量子采样权重 (attention_weights_gbs)
            # 反向传播时，把梯度无损地传递给前面的 scores，借鸡生蛋更新 Query 和 Key！
            attention_weights = attention_weights_gbs + (scores - scores.detach())
            # =====================================================================
            
            # 3. 计算最终输出
            output = torch.matmul(attention_weights, value)  # (batch_size, seq_len, input_dim)
            return output