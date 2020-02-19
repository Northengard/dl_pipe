import torch
from torch.utils.data import Dataset
from torch.nn import LocalResponseNorm


class Dummy(Dataset):
    def __init__(self, len=100):
        self.len = len
        self.data = torch.rand(self.len, 3, 540, 640)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        lables = torch.tensor([0, 1]) if torch.rand(1) > 0.5 else torch.tensor([1, 0])
        lables = lables.to(torch.float32)
        sample = {'images': self.data[index], 'lables': lables}
        return sample


class DFCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.len = 0
