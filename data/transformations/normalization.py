import cv2
import torch
import numpy as np
from torch.nn.functional import conv2d
from torch.nn import LocalResponseNorm


class Normalize(object):
    def __call__(self, sample):
        image = sample['images']
        image = image.astype('uint8')
        real_img = image.copy()
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = ((image * 2) / 255) - 1
        sample['images'] = image
        sample['real_image'] = real_img
        return sample


class LocalContrastNorm(object):
    def __init__(self, radius=9):
        """
        Local Contrast Normalisation.\n
        https://medium.com/@dibyadas/visualizing-different-normalization-techniques-84ea5cc8c378\n
        :param radius: (int), filter kernel size
        """
        if (radius % 2) == 0:
            radius += 1
        self.radius = radius

    @staticmethod
    def _get_gaussian_filter(kernel_shape):
        def gauss(x, y, sigma=2.0):
            z = 2 * np.pi * sigma ** 2
            return 1. / z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

        gaussian = np.zeros(kernel_shape, dtype='float64')
        mid = np.floor(kernel_shape[-1] / 2.)
        for kernel_idx in range(0, kernel_shape[1]):
            for i in range(0, kernel_shape[2]):
                for j in range(0, kernel_shape[3]):
                    gaussian[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
        return gaussian / np.sum(gaussian)

    def __call__(self, sample):
        image = sample['images']
        real_image = image.copy()
        image_tensor = torch.from_numpy(
            np.expand_dims(
                (image.transpose((2, 0, 1))), 0
            )
        )
        c, h, w = image_tensor[0].shape
        gaussian_filter = self._get_gaussian_filter((1, c, self.radius, self.radius))
        gaussian_filter = torch.from_numpy(gaussian_filter)
        filtered_out = conv2d(image_tensor, gaussian_filter, padding=self.radius - 1)
        mid = int(np.floor(gaussian_filter.shape[2] / 2.))
        # Subtractive Normalization
        centered_image = image_tensor - filtered_out[:, :, mid:-mid, mid:-mid]

        # Variance Calc
        sum_sqr_image = conv2d(centered_image.pow(2), gaussian_filter, padding=self.radius - 1)
        s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
        per_img_mean = s_deviation.mean()

        # Divisive Normalization
        divisor = np.maximum(per_img_mean.numpy(), s_deviation.numpy())
        divisor = np.maximum(divisor, 1e-4)
        transformed_img = centered_image / torch.from_numpy(divisor)
        transformed_img = transformed_img[0].numpy().transpose((1, 2, 0))

        numerator = (transformed_img - transformed_img.min())
        denominator = (transformed_img.max() - transformed_img.min()) + 1e-6  # to avoid zero division
        scaled_transformed_img = numerator / denominator

        if len(scaled_transformed_img.shape) > 2:
            scaled_transformed_img = cv2.cvtColor(scaled_transformed_img, cv2.COLOR_BGR2GRAY)
        sample['images'] = scaled_transformed_img
        sample['real_image'] = real_image

        return sample


class LocalRespNorm(object):
    def __init__(self, size=3, alpha=1e-4, beta=0.75, k=1):
        """
        Local Response Normalisation.\n
        https://medium.com/@dibyadas/visualizing-different-normalization-techniques-84ea5cc8c378\n
        This class is wrapper over torch.nn.LocalResponseNorm\n
        :param size: (int), amount of neighbouring channels used for normalization (kernel size)
        :param alpha: (float), multiplicative factor. Default: 0.0001
        :param beta: (float), exponent. Default: 0.75
        :param k: (float), additive factor. Default: 1
        """
        self.LocalResponseNorm = LocalResponseNorm(size, alpha, beta, k)

    def __call__(self, sample):
        image = sample['images']
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)
        real_image = image.copy()

        if len(image.shape) > 2:
            image_tensor = torch.from_numpy(np.expand_dims(image.transpose((2, 0, 1)), 0)).to(torch.float32)
        else:
            image_tensor = torch.from_numpy(np.expand_dims(image.transpose((1, 0)), 0)).to(torch.float32)

        lrn_image = self.LocalResponseNorm(image_tensor)
        transformed_img = lrn_image[0].numpy().transpose((1, 2, 0))

        numerator = (transformed_img - transformed_img.min())
        denominator = (transformed_img.max() - transformed_img.min()) + 1e-6  # to avoid zero division
        scaled_transformed_img = numerator / denominator

        if scaled_transformed_img.shape[2] == 1:
            scaled_transformed_img = np.squeeze(scaled_transformed_img)
        if len(scaled_transformed_img.shape) > 2:
            scaled_transformed_img = cv2.cvtColor(scaled_transformed_img, cv2.COLOR_BGR2GRAY)

        sample['images'] = scaled_transformed_img
        sample['real_image'] = real_image
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    # def __init__(self, with_angles=False):
    #     self.with_angles = with_angles

    def __call__(self, sample):
        image, segmentation_maps = sample['images'], sample['segmentation_maps']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        # if not self.with_angles:
        #     gaze = gaze[:-1]
        image = torch.from_numpy(image)
        image = image.to(torch.float32)

        segmentation_maps = segmentation_maps.transpose(2, 0, 1)
        segmentation_maps = torch.from_numpy(segmentation_maps)
        segmentation_maps = segmentation_maps.to(torch.float32)

        sample['images'] = image
        return sample
