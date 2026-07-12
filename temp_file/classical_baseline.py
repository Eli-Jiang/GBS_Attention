import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import time
from data_loader import get_dataloader

class LSTMModel(nn.Module):
    def __init__(self, in_features, hidden_dim, pred_len, out_features, num_layers=1):
        super().__init__()
        self.pred_len = pred_len
        self.out_features = out_features
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len * out_features)

    def forward(self, x):
        # x shape: (B, seq_len, in_features)
        out, _ = self.lstm(x) # out shape: (B, seq_len, hidden_dim)
        # Use the last step output to predict
        out_last = out[:, -1, :] # shape: (B, hidden_dim)
        preds = self.fc(out_last) # shape: (B, pred_len * out_features)
        return preds.view(-1, self.pred_len, self.out_features)

class LinearModel(nn.Module):
    def __init__(self, seq_len, in_features, pred_len, out_features):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(seq_len * in_features, pred_len * out_features)

    def forward(self, x):
        # x shape: (B, seq_len, in_features)
        x_flat = x.view(-1, self.seq_len * self.in_features)
        preds = self.linear(x_flat) # shape: (B, pred_len * out_features)
        return preds.view(-1, self.pred_len, self.out_features)

def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load dataloaders
    train_loader = get_dataloader(args.data_dir, flag='train', batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len)
    val_loader = get_dataloader(args.data_dir, flag='val', batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len)
    test_loader = get_dataloader(args.data_dir, flag='test', batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len)

    # Initialize model
    if args.model == 'lstm':
        model = LSTMModel(in_features=7, hidden_dim=args.hidden_dim, pred_len=args.pred_len, out_features=7).to(device)
    else:
        model = LinearModel(seq_len=args.seq_len, in_features=7, pred_len=args.pred_len, out_features=7).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    print(f"Start training {args.model.upper()}...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        start_time = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        epoch_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss.append(loss.item())
                
        train_loss_mean = np.mean(train_loss)
        val_loss_mean = np.mean(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f} | Time: {epoch_time:.2f}s")
        
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save(model.state_dict(), f"{args.model}_best.pt")

    # Evaluate on test set
    model.load_state_dict(torch.load(f"{args.model}_best.pt"))
    model.eval()
    test_loss_mse = []
    test_loss_mae = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            # MSE
            mse = criterion(preds, y).item()
            test_loss_mse.append(mse)
            
            # MAE
            mae = torch.mean(torch.abs(preds - y)).item()
            test_loss_mae.append(mae)

    print("\n================ EVALUATION ================")
    print(f"Model: {args.model.upper()}")
    print(f"Test MSE: {np.mean(test_loss_mse):.4f}")
    print(f"Test MAE: {np.mean(test_loss_mae):.4f}")
    print("============================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'lstm'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='./data')
    
    args = parser.parse_args()
    train_and_eval(args)
