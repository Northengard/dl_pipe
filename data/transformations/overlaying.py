import cv2
import glob
import numpy as np


class ImageOverlaying(object):
    def __init__(self, root_dir=None, paths=None, overlay_coef=None, change_background=False, image_num=500):
        """
        Overlay one of the images from given directory on sample image with replacing it's background.
        Background replacing has the same logic as BackgroundShifter do.
        :param root_dir: (str), directory path to images for overlaying (has priority)
        :param paths: (list of str), path list of images for overlaying
        :param overlay_coef: (float) coefficient of overlaying. (weight of overlaid image)
        :param change_background: (bool) change background flag. If True it will replace gray background
        (90-110 pxs, resp to synthetic) with one of overlay images.
        :param image_num: (int), number of image to get from the given directory
        """
        if root_dir:
            self.path_list = glob.glob(root_dir + '/**/*.*p*g', recursive=True)
        else:
            self.path_list = paths
        self.images = list()
        for i in range(image_num):
            path = self.path_list[i]
            self.images.append(cv2.imread(path))
        self.overlay_coef = overlay_coef
        self.change_background = change_background

    @staticmethod
    def _change_background(img, background_image, background_range):
        """
        Changes gray background of an image by replacing it with another image from given list by mask.
        :param img: source image
        :param background_image: image to set as background
        :param background_range: tuple with min and max pix values of background
        :return: copy of source image with replaced background
        """
        target = img.copy()
        overlay_img = background_image.copy()
        overlay_img = cv2.resize(overlay_img, img.shape[:-1][::-1])
        back_mask = np.ones(img.shape[:-1])

        for channel in range(img.shape[-1]):
            channel_mask = (img[:, :, channel] >= background_range[0]) & (img[:, :, channel] <= background_range[1])
            back_mask = np.logical_and(back_mask, channel_mask)

        overlay_img[~back_mask] = 0
        target[back_mask] = 0
        target = target + overlay_img
        return target

    def __call__(self, sample):
        image = sample['image']
        if self.change_background:
            overlay_img, back_img = np.random.choice(self.images, 2)
            image = self._change_background(image, back_img)
        else:
            overlay_img = np.random.choice(self.images)
        overlay_img = cv2.resize(overlay_img, image.shape[1::-1])
        if self.overlay_coef:
            ovc = self.overlay_coef
        else:
            ovc = np.random.uniform(0.12, 0.2)
        image = cv2.addWeighted(image, (1 - ovc), overlay_img, ovc, 0)
        sample['image'] = image
        return sample


class BackgroundShifter(object):
    def __init__(self, root_dir=None, paths=None, image_num=500):
        """
        Changes gray background of an image by replacing it with another image from given list by mask.
        :param root_dir: (str), directory path to background images (has priority)
        :param paths: (list of str), path list of background images
        :param image_num: (int), number of image to get from the given directory
        """
        if root_dir:
            self.path_list = glob.glob(root_dir + '/**/*.*p*g', recursive=True)
        else:
            self.path_list = paths
        self.images = list()
        for i in range(image_num):
            path = self.path_list[i]
            self.images.append(cv2.imread(path))

    def __call__(self, sample):
        img = sample['image']
        back_img = np.random.choice(self.images)
        back_img = cv2.resize(back_img, img.shape[:-1][::-1])
        back_mask = np.ones(img.shape[:-1])

        for channel in range(img.shape[-1]):
            channel_mask = (img[:, :, channel] >= 90) & (img[:, :, channel] <= 110)
            back_mask = np.logical_and(back_mask, channel_mask)

        back_img[~back_mask] = 0
        img[back_mask] = 0
        img = img + back_img
        sample['image'] = img
        return sample


class DrawLines(object):
    """
    Draws vertical and horizontal random lines on the sample's image
    """
    @staticmethod
    def _draw_line(img):
        image = img.copy()
        is_horizontal = np.random.rand() > 0.5
        if is_horizontal:
            start_x, start_y = 0, np.random.randint(0, image.shape[0])
            end_x, end_y = image.shape[1] - 1, np.random.randint(0, image.shape[0])
        else:
            start_x, start_y = np.random.randint(0, image.shape[1]), 0
            end_x, end_y = np.random.randint(0, image.shape[1]), image.shape[0] - 1

        line_thickness = np.random.randint(1, 5)
        color = tuple([np.random.randint(0, 255) for _ in range(3)])
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, line_thickness)
        return image

    def __call__(self, sample):
        line_number = 1 if np.random.rand() > 0.5 else 2
        image = sample['image']
        for num in range(line_number):
            image = self._draw_line(image)

        sample['image'] = image
        return sample
