from bhive_dataloader import BHiveDataset, collate_fn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import data_cost as dt
import utilities as ut
from model import IthemalRNN
TARGET = os.environ.get("ITHEMAL_TARGET", "skl")
throughput_file_path = os.path.join(
    # Example for Skylake
    'bhive', 'benchmark', 'throughput', f'{TARGET}.csv')
dataset = BHiveDataset(
    throughput_file=throughput_file_path, token_idx_map_ref=dt.load_token_idx_map(f'vocab_map_{TARGET}.pkl'))
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn,
                        shuffle=True, num_workers=8)
vocab_size = len(dataset.token_idx_map.token_to_hot_idx)

model = IthemalRNN(vocab_size, 256, 256)
model.eval()  # Set the model to evaluation mode
sample_data = dataset[0]  # Get a sample data item
print(f"Sample data: {sample_data.x}, {sample_data.y.item():.4f}")

model_output = model(sample_data)

print(f"Model output for sample data: {model_output.item():.4f}")
