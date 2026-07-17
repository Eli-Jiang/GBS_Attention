import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: 原始的 Numpy 数组，形状为 [总时间步数, 特征数 c_in]
              比如一整年 365 天的数据，特征是 1
        seq_len: 历史窗口长度 (96)
        pred_len: 预测未来长度 (24)
        """
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # 可滑动的最大范围，防止滑窗越界
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # 1. 历史切片 X
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data[s_begin:s_end]
        
        # 2. 未来切片 Y (标签)
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_y = self.data[r_begin:r_end]
        
        # 返回 Tensor
        return torch.tensor(seq_x), torch.tensor(seq_y)

# 快捷获取 DataLoader 的函数
def get_dataloader(data, seq_len, pred_len, batch_size, shuffle=True):
    dataset = TimeSeriesDataset(data, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader