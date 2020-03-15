import cv2


class HorizontalFlip(object):
    def __init__(self):
        """
        Horizontal Flip of sample (Mirroring)
        """
        self.rot_axis = 1

    def __call__(self, sample):
        if sample['is_right']:
            image = sample['image']
            image = cv2.flip(image, self.rot_axis)
            sample['image'] = image
        return sample
