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


def dfc_dataset(config, is_train=True):
    """
    Returns deep fake detection dataset according to given config
    :param config: dict with params
    :param is_train: bool, flag to get loader for trainig
    :return: DFC dataset
    """
    if is_train:
        data_dir = config['train']['data_dir']
        batch_size = config['train']['batch_size']
        shuffle = True
    else:
        data_dir = config['validation']['data_dir']
        batch_size = config['validation']['batch_size']
        shuffle = True

    dataset = DFCDataset(data_dir=data_dir,
                         transform=Transforms(config['input_size'], train=is_train))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=config['num_workers'])
    return loader


def get_dfc_video_dataset(config):
    test_loader = DFCVideoDataset(data_dir=config['test']['data_dir'],
                                  transform=Transforms(config['input_size'], train=False))
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
        self.last_index = 0
        self._len = len(self._data)
        self._transform = transform

    def apply_transform(self, frame):
        sample = {'images': frame, 'labels': [0, 1]}
        sample = self._transform(sample)
        return torch.unsqueeze(sample['images'], dim=0)

    def __len__(self):
        return self._len

    def get_last_vid_name(self):
        return path.basename(self._data[self.last_index])

    def __getitem__(self, index):
        self.last_index = index
        reader = get_reader(self._data[index])
        return reader
