import cv2
import numpy as np


class GaussianBlur(object):
    def __init__(self, max_blur_kernel=12, min_blur_kernel=5):
        """
        Gaussian filter with different kernel size.\n
        Init parameters:
        :param max_blur_kernel: (int), max size of kernel
        :param min_blur_kernel: (int), min size of kernel
        """
        self.kernels = list(range(min_blur_kernel, max_blur_kernel, 2))

    def __call__(self, sample):
        image = sample['images']
        # kernel size
        k_size = np.random.choice(self.kernels)
        # 0 means that sigmaX and sigmaY calculates from kernel
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)
        sample['images'] = image
        return sample


class CompressionArtifacts(object):
    def __init__(self, downscale=5):
        """
        Creates compression artifacts on image connected with strong compression and upscaling
        Init parameters:
        :param downscale: (int), coefficient of downscaling
        """
        self.downscale = downscale

    def __call__(self, sample):
        image = sample['images']
        h, w = image.shape[:2]
        image = cv2.resize(image, (w // self.downscale, h // self.downscale))
        image = cv2.resize(image, (w, h))
        sample['images'] = image
        return sample
