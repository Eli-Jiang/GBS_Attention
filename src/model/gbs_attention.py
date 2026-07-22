import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GBSAttention(nn.Module):
    def __init__(self, d_model, shots, use_gbs=False):
        """
        use_gbs: True = GBS量子采样(dq), False = GBS理论计算, "standard" = 标准Attention
        """
        super().__init__()
        self.d_model = d_model
        self.shots = shots
        self.use_gbs = use_gbs
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # 导入GBS相关函数
        from .utils import encode_graph_to_unitary, gbs_sample, gbs_theory
        self.encode_graph_to_unitary = encode_graph_to_unitary
        self.gbs_sample_func = gbs_sample
        self.gbs_theory_func = gbs_theory

    def gbs(self, A):
        """GBS量子采样模式"""
        mode = A.shape[0]
        T, r = self.encode_graph_to_unitary(A)
        counts = self.gbs_sample_func(T, r, mode, self.shots)
        return counts

    def theory(self, A):
        """GBS理论计算模式"""
        mode = A.shape[0]
        T, r = self.encode_graph_to_unitary(A)
        adj_matrix = self.gbs_theory_func(T, r)
        return torch.tensor(adj_matrix, dtype=torch.float32, device=A.device)

    def forward(self, x):
            batch_size, seq_len, _ = x.shape

            query = self.query_proj(x)
            key = self.key_proj(x)
            value = self.value_proj(x)

            scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)

            if self.use_gbs == "standard":
                # 标准Attention
                attention_weights = F.softmax(scores, dim=-1)
            else:
                # 强制对称且对角线置 0
                scores_sym = (scores + scores.transpose(-2, -1)) / 2.0
                eye_mask = torch.eye(seq_len, device=scores.device, dtype=scores.dtype)
                scores_sym = scores_sym * (1 - eye_mask)

                if self.use_gbs is True:
                    # GBS量子采样模式
                    attn_weights_list = []
                    scores_np = scores_sym.detach().cpu().numpy()
                    for b in range(batch_size):
                        gbs_weight = self.gbs(scores_np[b])
                        attn_weights_list.append(gbs_weight)
                    attention_weights_gbs = torch.tensor(np.array(attn_weights_list), dtype=scores.dtype, device=scores.device)
                    attention_weights = attention_weights_gbs + (scores - scores.detach())

                else:
                    # GBS理论计算模式 (use_gbs = False)
                    attn_weights_list = []
                    scores_np = scores_sym.detach().cpu().numpy()
                    for b in range(batch_size):
                        theory_weight = self.theory(scores_np[b])
                        attn_weights_list.append(theory_weight)
                    attention_weights_theory = torch.stack(attn_weights_list)
                    attention_weights = attention_weights_theory + (scores - scores.detach())

            output = torch.matmul(attention_weights, value)
            return output
