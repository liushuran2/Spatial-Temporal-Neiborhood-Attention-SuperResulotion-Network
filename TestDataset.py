
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class TestDataset(Dataset):
    def __init__(self, h5_file):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            video_id = idx
            lr = f['lr'][str(video_id)]
            lr = np.array(lr)
            hr = f['hr'][str(video_id)]
            gt = hr[3, :, :, :]
            return lr.astype(np.float32), gt.astype(np.float32)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])