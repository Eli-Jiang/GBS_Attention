import torch
import torch.nn as nn
from .embedding import TimeSeriesEmbedding
from .layers import TransformerBlock

class TimeSeriesTransformer(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, n_layers=3, dropout=0.1, shots=4096, use_gbs=False):
        """
        c_in:     输入特征数（例如单变量预测就是 1）
        seq_len:  输入的历史序列长度（看过去多少步）
        pred_len: 预测的未来序列长度（看未来多少步）
        use_gbs:  是否使用GBS量子采样，False则使用理论计算（更快）
        """
        super().__init__()

        # 1. 进站：时序特征映射 + 位置编码
        self.embedding = TimeSeriesEmbedding(c_in=c_in, d_model=d_model, max_len=seq_len, dropout=dropout)

        # 2. 核心车厢：堆叠多层 Transformer Block
        # 使用 nn.ModuleList 优雅地把多层串联起来
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, shots=shots, use_gbs=use_gbs)
            for _ in range(n_layers)
        ])

        # 3. 出站：预测头（Forecasting Head）
        # 最快跑通的做法：把时序长度和特征维度展平，直接用线性层映射到 [pred_len * c_in]
        self.head = nn.Linear(seq_len * d_model, pred_len * c_in)

        self.pred_len = pred_len
        self.c_in = c_in

    def forward(self, x):
        # 输入 x 维度: [Batch_Size, seq_len, c_in]
        batch_size = x.size(0)
        
        # 1. 经过 Embedding 层
        x = self.embedding(x)  # 输出: [Batch_Size, seq_len, d_model]
        
        # 2. 穿过每一层 Transformer Block
        for layer in self.layers:
            x = layer(x)  # 输出: [Batch_Size, seq_len, d_model]
            
        # 3. 展平并输出预测值
        # 把 [seq_len, d_model] 拉直成一维向量
        x = x.reshape(batch_size, -1)  # 输出: [Batch_Size, seq_len * d_model]
        
        # 映射到未来的预测空间
        output = self.head(x)  # 输出: [Batch_Size, pred_len * c_in]

        # 重新整理成标准的时序维度: [Batch_Size, pred_len, c_in]
        output = output.reshape(batch_size, self.pred_len, self.c_in)
        
        return output