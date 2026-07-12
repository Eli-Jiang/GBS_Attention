import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATASET_URLS = {
    'ETTh1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/master/ETT-small/ETTh1.csv',
    'exchange': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/raw/main/exchange_rate.csv',
    'weather': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/raw/main/weather.csv'
}

class TSDataset(Dataset):
    def __init__(self, data_path, flag='train', size=(96, 24), scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.flag = flag
        self.scale = scale
        self.__read_data__(data_path)

    def __read_data__(self, data_path):
        df_raw = pd.read_csv(data_path)
        
        # We assume the first column is the date/time column, drop it
        data = df_raw.iloc[:, 1:].values
        
        # Train/Val/Test splits (70% / 10% / 20%)
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        
        if self.flag == 'train':
            border1 = border1s[0]
            border2 = border2s[0]
        elif self.flag == 'val':
            border1 = border1s[1]
            border2 = border2s[1]
        else: # test
            border1 = border1s[2]
            border2 = border2s[2]
            
        # Fit scaler on train split
        self.mean = data[border1s[0]:border2s[0]].mean(axis=0)
        self.std = data[border1s[0]:border2s[0]].std(axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
        if self.scale:
            self.data = (data - self.mean) / self.std
        else:
            self.data = data
            
        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

def get_dataloader(data_dir, dataset_name, flag='train', batch_size=32, seq_len=96, pred_len=24):
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_URLS.keys())}")
        
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{dataset_name}.csv"
    data_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(data_path):
        url = DATASET_URLS[dataset_name]
        print(f"Downloading {dataset_name} dataset from {url}...")
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")
    
    dataset = TSDataset(data_path, flag=flag, size=(seq_len, pred_len))
    shuffle = True if flag == 'train' else False
    drop_last = True if flag == 'train' else False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last
    )
    return dataloader, dataset.mean, dataset.std

if __name__ == '__main__':
    # Quick test
    for name in DATASET_URLS.keys():
        print(f"Testing loader for {name}...")
        dl, mean, std = get_dataloader('./data', name, flag='train', batch_size=4)
        x, y = next(iter(dl))
        print(f"Dataset: {name} | Train batch X shape: {x.shape} | Y shape: {y.shape}")
        print(f"Mean shape: {mean.shape} | Std shape: {std.shape}")
