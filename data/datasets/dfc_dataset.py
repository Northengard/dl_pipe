from os import path
from os import listdir
from glob import glob
from json import load

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imageio import get_reader

from utils.storage import load_image
from data.transformations import Transforms


def get_dfc_dataset(config, get_dummy=False):
    """
    Returns deep fake detection dataset according to given config
    :param config: dict with params
    :param get_dummy: bool, flag to get dummy dataset
    :return: DFC dataset
    """
    if get_dummy:
        train_loader = DataLoader(Dummy(length=100), batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])
        test_loader = train_loader
    else:
        train_loader = DataLoader(DFCDataset(data_dir=config['train']['data_dir'],
                                             transform=Transforms(config['input_size'])),
                                  batch_size=config['train']['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])

        test_loader = DataLoader(DFCDataset(data_dir=config['validation']['data_dir'],
                                            transform=Transforms(config['input_size'],
                                                                 train=False)),
                                 batch_size=config['validation']['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'])
    return train_loader, test_loader


def get_dfc_video_dataset(config, ):
    test_loader = DataLoader(DFCVideoDataset(data_dir=config['test']['data_dir'],
                                             transform=Transforms(config['input_size'],
                                                                  train=False)),
                             batch_size=config['validation']['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'])
    return test_loader


class Dummy(Dataset):
    def __init__(self, length=100):
        self.len = length
        self.data = torch.rand(self.len, 3, 480, 270)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        labels = torch.tensor([0, 1]) if torch.rand(1) > 0.5 else torch.tensor([1, 0])
        labels = labels.to(torch.float32)
        sample = {'images': self.data[index], 'labels': labels}
        return sample


class DFCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self._data = glob(path.join(data_dir, '*.jpg'))
        self._len = len(self._data)
        self._transform = transform
        self._annotations = path.join(data_dir, 'labels.json')
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
        sample['labels'] = [0, label] if label else [1, 0]
        if self._transform:
            sample = self._transform(sample)
        return sample


class DFCVideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self._data = list(map(lambda x: path.join(data_dir, x), listdir(data_dir)))
        self._len = len(self._data)
        self._transform = transform

    def apply_transform(self, frame):
        sample = {'images': frame, 'labels': [0, 1]}
        sample = self._transform(sample)
        return sample['images']

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        reader = get_reader(self._data[index])
        return reader
