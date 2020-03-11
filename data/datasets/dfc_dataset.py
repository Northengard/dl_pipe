from os import path
from glob import glob
import torch
from json import load
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.storage import load_image


def get_dataset(config, get_dummy=False):
    if get_dummy:
        loader = DataLoader(Dummy(len=100), batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'])
    else:
        loader = DataLoader(DFCDataset(data_dir=config['data_dir']), batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'])
    return loader


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
        self._data = glob(path.join(data_dir, '*.jpg'))
        self._len = len(self._data)
        self._transform = transform
        self._annotations = glob(path.join(data_dir, '*.json'))[0]
        with open(self._annotations) as anf:
            self._annotations = load(anf)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        sample = dict()
        img = load_image(self._data[index])
        img_name = path.basename(self._data[index])
        label = self._annotations[img_name]
        sample['images'] = img
        sample['labels'] = label
        if self._transform:
            sample = self._transform(sample)
        return sample
