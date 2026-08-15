"""
Charbonnier Loss: Smooth L1 approximation for robust image restoration.
L(x, y) = sqrt((x - y)^2 + eps^2)
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)
