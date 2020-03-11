from sklearn.metrics import accuracy_score
from torch import sigmoid, max

def accuracy(y_pred, y_true):
    y_pred = sigmoid(y_pred)
    y_pred = max(y_pred, 1)[1]
    y_true = max(y_true, 1)[1]
    acc = accuracy_score(y_pred, y_true)
    return acc