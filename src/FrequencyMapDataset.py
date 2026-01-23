import torch
from torch.utils.data import Dataset
import numpy as np

class FrequencyMapDataset(Dataset):
    def __init__(self, npy_path, transform=None):
        self.npy_path = npy_path
        self.transform = transform
        
        # Open in mmap mode 'r' (read-only)
        # This reads metadata but doesn't load the array to RAM
        self.data = np.load(npy_path, mmap_mode='r')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        sample = torch.from_numpy(sample).float()

        if self.transform:
            sample = self.transform(sample)

        return sample
