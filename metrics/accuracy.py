import numpy as np


def accuracy(num_classes=2, ignore=-1):
    return Accuracy(num_classes, ignore=ignore)


class Accuracy(object):
    def __init__(self, num_classes, ignore):
        self.num_class = num_classes
        self.ignore = ignore
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def get_confusion_matrix(self, pred, label):
        """
        Calcute the confusion matrix by given label and pred
        """
        seg_pred = np.asarray(np.argmax(pred.cpu().numpy(), axis=1), dtype=np.int)
        seg_gt = np.asarray(np.argmax(label.cpu().numpy(), axis=1), dtype=np.int)

        ignore_index = seg_gt != self.ignore
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        index = (seg_gt * self.num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((self.num_class, self.num_class))

        for i_label in range(self.num_class):
            for i_pred in range(self.num_class):
                cur_index = i_label * self.num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label,
                                     i_pred] = label_count[cur_index]
        return confusion_matrix

    def __call__(self, pred, label):
        self.confusion_matrix += self.get_confusion_matrix(label, pred)
        total = self.confusion_matrix.sum()
        tp = np.diag(self.confusion_matrix).sum()
        return tp / total
