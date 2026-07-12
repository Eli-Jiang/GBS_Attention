import torch
import numpy as np
import pandas as pd
from data_loader import get_dataloader
from classical_baseline import LSTMModel, LinearModel

def calculate_r2(preds, targets):
    # preds, targets: (N, pred_len, features) or flattened
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    target_mean = np.mean(targets_flat)
    ss_tot = np.sum((targets_flat - target_mean) ** 2)
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def calculate_feature_metrics(preds, targets):
    # preds, targets shape: (num_samples, pred_len, num_features)
    num_features = targets.shape[-1]
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    results = []
    for f in range(num_features):
        p_f = preds[:, :, f]
        t_f = targets[:, :, f]
        
        mse = np.mean((p_f - t_f) ** 2)
        mae = np.mean(np.abs(p_f - t_f))
        r2 = calculate_r2(p_f, t_f)
        std_err = np.std(p_f - t_f)
        
        results.append({
            'Feature': feature_names[f],
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'StdErr': std_err
        })
        
    return pd.DataFrame(results)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 96
    pred_len = 24
    batch_size = 32
    
    test_loader = get_dataloader('./data', flag='test', batch_size=batch_size, seq_len=seq_len, pred_len=pred_len)
    
    # Extract all test targets for comparison
    all_targets = []
    for _, y in test_loader:
        all_targets.append(y.numpy())
    all_targets = np.concatenate(all_targets, axis=0) # shape: (N, pred_len, 7)
    
    models = {
        'linear': LinearModel(seq_len=seq_len, in_features=7, pred_len=pred_len, out_features=7),
        'lstm': LSTMModel(in_features=7, hidden_dim=64, pred_len=pred_len, out_features=7)
    }
    
    for name, model in models.items():
        print(f"\n======================================")
        print(f" Rich Evaluation for {name.upper()} Model")
        print(f"======================================")
        
        checkpoint_path = f"{name}_best.pt"
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue
            
        model = model.to(device)
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                preds = model(x)
                all_preds.append(preds.cpu().numpy())
                
        all_preds = np.concatenate(all_preds, axis=0) # shape: (N, pred_len, 7)
        
        # Calculate general metrics
        overall_mse = np.mean((all_preds - all_targets) ** 2)
        overall_mae = np.mean(np.abs(all_preds - all_targets))
        overall_r2 = calculate_r2(all_preds, all_targets)
        overall_std = np.std(all_preds - all_targets)
        
        print(f"Overall Metrics:")
        print(f"  MSE:      {overall_mse:.4f}")
        print(f"  MAE:      {overall_mae:.4f}")
        print(f"  R2 Score: {overall_r2:.4f}")
        print(f"  Std Err:  {overall_std:.4f}")
        print("\nFeature-wise Detail Metrics:")
        
        df_features = calculate_feature_metrics(all_preds, all_targets)
        print(df_features.to_string(index=False))
        
if __name__ == '__main__':
    main()
