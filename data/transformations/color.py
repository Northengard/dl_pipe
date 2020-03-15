import numpy as np
from PIL import Image
from torchvision import transforms


class ColorTransform(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Color transform to deal with image brightness, contrast, saturation and hue.
        Wraper over torchvision.transforms.ColorJitter\n
        :param brightness: (tuple of float), set range of positive values for brightness changes
        :param contrast:(tuple of float), set range of positive values for contrast changes
        :param saturation: (tuple of float), set range of positive values for saturation changes
        :param hue: (tuple of float), set range of hue changes. Values must be between (-0.5, 0.5)
        """

        def _check_param(param_name, param, min_val, max_val):
            if param != 0:
                try:
                    assert type(param) == tuple
                    assert len(param) == 2
                    assert min(param) >= min_val
                    assert max(param) <= max_val
                    assert param[0] <= param[1]
                    return True
                except AssertionError(param_name +
                                      ' must be a tuple of float (min>={}, max<={})'.format(min_val, max_val)):
                    return False
            else:
                return True

        if _check_param('brightness', brightness, 0, np.inf):
            self.brightness = brightness
        if _check_param('contrast', contrast, 0, np.inf):
            self.contrast = contrast
        if _check_param('saturation', saturation, 0, np.inf):
            self.saturation = saturation
        if _check_param('hue', hue, -0.5, 0.5):
            self.hue = hue
        self.cj = transforms.ColorJitter(brightness=self.brightness,
                                         contrast=self.contrast,
                                         saturation=self.saturation,
                                         hue=self.hue)

    def __call__(self, sample):
        image = sample['images']
        pil_img = Image.fromarray(image)
        pil_img = self.cj(pil_img)
        image = np.array(pil_img)
        sample['images'] = image
        return sample
