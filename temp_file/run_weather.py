import argparse
from patch_transformer import train

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weather', choices=['ETTh1', 'exchange', 'weather'])
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--seq_len', type=int, default=6)
parser.add_argument('--pred_len', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=2)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=5)

args = parser.parse_args([])
train(args)
