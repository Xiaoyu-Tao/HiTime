import torch
import torch.nn as nn


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)
              
