import cv2


class Rescale(object):
    def __init__(self, output_size):
        """
        Rescale the image in a sample to a given size.
        :param output_size: (tuple or int): Desired output size. If tuple, output is
                                            matched to output_size. If int, smaller of image edges is matched
                                            to output_size keeping aspect ratio the same.
        :param lmrks_only: (bool)
        :param angles_only: (bool)
        """
        assert isinstance(output_size, (int, tuple))
        if type(output_size) == int:
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation_maps = sample['image'], sample['segmentation_maps']

        new_w, new_h = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        seg_maps = cv2.resize(segmentation_maps, (new_w, new_h), interpolation=cv2.INTER_AREA)

        sample['image'] = img
        sample['segmentation_maps'] = seg_maps

        return sample
