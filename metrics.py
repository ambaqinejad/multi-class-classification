import torch


def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred.float()).sum().item()
    acc = correct / len(y_true) * 100
    return acc
