from torch.nn import BCELoss, BCEWithLogitsLoss


def bce(reduction='mean'):
    return BCELoss(reduction=reduction)


def bce_with_logits(reduction='mean'):
    return BCEWithLogitsLoss(reduction=reduction)
