"""
Edge-Aware Loss using Sobel / Laplacian operators for high-frequency semiconductor detail restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """
    Computes L1 difference between edge filter responses of prediction and target.
    Supports Sobel (horizontal + vertical) and Laplacian filters.
    """
    def __init__(self, mode="sobel", eps=1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

        if mode == "sobel":
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_x", sobel_x)
            self.register_buffer("kernel_y", sobel_y)
        elif mode == "laplacian":
            laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_laplacian", laplacian)
        else:
            raise ValueError(f"Unknown edge loss mode: {mode}")

    def forward(self, pred, target):
        if self.mode == "sobel":
            pred_x = F.conv2d(pred, self.kernel_x.type_as(pred), padding=1)
            pred_y = F.conv2d(pred, self.kernel_y.type_as(pred), padding=1)
            target_x = F.conv2d(target, self.kernel_x.type_as(target), padding=1)
            target_y = F.conv2d(target, self.kernel_y.type_as(target), padding=1)

            loss_x = torch.sqrt((pred_x - target_x) ** 2 + self.eps ** 2).mean()
            loss_y = torch.sqrt((pred_y - target_y) ** 2 + self.eps ** 2).mean()
            return loss_x + loss_y

        elif self.mode == "laplacian":
            pred_lap = F.conv2d(pred, self.kernel_laplacian.type_as(pred), padding=1)
            target_lap = F.conv2d(target, self.kernel_laplacian.type_as(target), padding=1)
            return torch.sqrt((pred_lap - target_lap) ** 2 + self.eps ** 2).mean()
