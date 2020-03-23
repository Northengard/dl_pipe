import numpy as np
from torchvision.transforms import Compose
from .normalization import LocalRespNorm, ToTensor
from .rescale import Rescale


class OneOf(object):
    def __init__(self, transformations):
        """
        Apply one of given transformation.
        :param transformations: (list of transformations), iterable object with transformations to apply.
        """
        self.transforms = transformations

    def __call__(self, sample):
        transform = np.random.choice(self.transforms)
        sample = transform(sample)
        return sample


class RandomApply(object):
    def __init__(self, transformations, prob=0.5):
        """
        Creates pipeline of transformations to apply it one-by-one with given probability.
        :param transformations: (list of transformations), iterable object with transformations to apply.
        """
        self.transforms = transformations
        self.prob = prob

    def __call__(self, sample):
        for t in self.transforms:
            if np.random.rand() > self.prob:
                sample = t(sample)
        return sample


# GENERAL
class Transforms(object):
    def __init__(self, input_size, train=False):
        """
        Class to combine all Transformations together.
        :param input_size: tuple, network image input size (w, h)
        :param train: bool, train flag to apply train transformations
        """
        self.train = train
        self.input_size = input_size

        if self.train:
            self.transforms_train = list()
            self.transforms_train = RandomApply(self.transforms_train)

        self.normalize = Compose([
            Rescale(tuple(input_size)),
            LocalRespNorm(size=7),
            ToTensor(),
        ])

    def __call__(self, image):
        if self.train:
            image = self.transforms_train(image)

        image = self.normalize(image)
        return image
