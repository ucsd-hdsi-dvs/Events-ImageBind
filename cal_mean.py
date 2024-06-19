import torch
from torch.utils.data import DataLoader
from datasets.MVSCEDataset import MVSCEDataset

# Directory containing the dataset
data_dir = '/tsukimi/datasets/MVSEC/event_chunks_10t'

# Create dataset instance
dataset = MVSCEDataset(mode='train', data_dir=data_dir)

# Create DataLoader with num_workers=4
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

sum = torch.zeros(3)
sum_of_squares = torch.zeros(3)
n_pixels = 0

# Iterate through the dataloader and collect all images
for i, (img, _, event_frame, _) in enumerate(dataloader):
    # print progress
    print(f"\r{i}/{len(dataloader)}", end="")
    for ef in event_frame:
        sum += ef.sum(dim=[1, 2])
        sum_of_squares += (event_frame ** 2).sum(dim=[1, 2])
        n_pixels+=ef.numel()// 3
    
    mean = sum / n_pixels
    std = torch.sqrt(sum_of_squares / n_pixels - mean ** 2)
    print("\nMean:", mean)
    print("Std:", std)
