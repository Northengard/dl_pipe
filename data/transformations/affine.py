import cv2


class HorizontalFlip(object):
    def __init__(self):
        """
        Horizontal Flip of sample (Mirroring)
        """
        self.rot_axis = 1

    @staticmethod
    def _flip_gaze(annot, rot_axis):
        if rot_axis == 0:
            annot[1] *= -1
        elif rot_axis == 1:
            annot[0] *= -1
        elif rot_axis == 2:
            annot = -annot
        return annot

    def __call__(self, sample):
        if sample['is_right']:
            image, segmentation_maps = sample['image'], sample['segmentation_maps']
            gaze = sample['gaze_vector']
            image = cv2.flip(image, self.rot_axis)
            segmentation_maps = cv2.flip(segmentation_maps, self.rot_axis)
            gaze = self._flip_gaze(gaze, self.rot_axis)
            sample['image'] = image
            sample['segmentation_maps'] = segmentation_maps
            sample['gaze_vector'] = gaze
        return sample
