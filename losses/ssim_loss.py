"""
Differentiable SSIM (Structural Similarity) Loss for PyTorch.

Handles arbitrary dynamic range with configurable data_range or dynamic data_range estimation.
SSIM Loss = 1.0 - SSIM(pred, target)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, data_range=None):
    """
    Computes SSIM between img1 and img2.
    """
    (_, channel, _, _) = img1.size()

    if window is None:
        window = create_window(window_size, channel).to(img1.device).type(img1.dtype)
    else:
        window = window.to(img1.device).type(img1.dtype)

    if data_range is None:
        # Dynamic estimation of data range from target
        data_range = (img2.max() - img2.min()).clamp(min=1e-3).item()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Clamp variances for numerical stability
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1, data_range=None, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.size_average = size_average
        self.register_buffer("window", create_window(window_size, channel))

    def forward(self, pred, target):
        return 1.0 - ssim(
            pred,
            target,
            window_size=self.window_size,
            window=self.window,
            size_average=self.size_average,
            data_range=self.data_range,
        )
