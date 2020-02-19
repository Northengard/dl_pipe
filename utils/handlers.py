from numpy import median, std


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    def __init__(self, is_validation=False):
        self.reset()
        self.is_validation = is_validation

    def reset(self):
        self.vals = []
        self.val = 0
        self.avg = 0
        self.median = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def update(self, val):
        if self.is_validation:
            self.vals.append(val)
            self.median = median(self.vals)
            self.std = std(self.vals)
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
