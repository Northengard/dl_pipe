import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, Module


class FocalLoss(Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        """
        :param gamma: exponential parameter
        :param alpha: balance variable to make trade off between values of two classes
        :param reduction: reduction (aggregation) method must be equal to 'sum' or 'mean'
        """
        super(FocalLoss, self).__init__()
        if reduction not in ['mean', 'sum']:
            raise AttributeError("reduction must be 'mean' or 'sum'")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    @staticmethod
    def _focal_loss(pred_probs, targets, gamma, alpha):
        gamma = gamma
        alpha = alpha

        p = pred_probs
        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        left = targets * term1 * alpha
        right = (1 - targets) * term2 * (1 - alpha)
        f_loss = -(left + right)
        return f_loss

    def forward(self, logits, targets):
        loss = self._focal_loss(logits, targets, self.gamma, self.alpha)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

    def __repr__(self):
        tmp_repr = self.__class__.__name__ + "("
        tmp_repr += "gamma=" + str(self.gamma)
        tmp_repr += ", alpha=" + str(self.alpha)
        tmp_repr += ")"
        return tmp_repr


def bce(reduction='mean'):
    return BCELoss(reduction=reduction)


def bce_with_logits(reduction='mean', pos_weight=(1,)):
    return BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.tensor(pos_weight))


def focal_loss(gamma=2, alpha=0.1):
    return FocalLoss(gamma=gamma, alpha=alpha)
