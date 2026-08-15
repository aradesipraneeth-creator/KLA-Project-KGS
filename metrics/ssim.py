"""
SSIM (Structural Similarity Index) evaluation metric for image restoration.
Supports PyTorch tensors and NumPy arrays with standard 11x11 Gaussian window.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(window_size=11, sigma=1.5):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def calculate_ssim(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    """
    Computes SSIM between pred and target.
    
    Args:
        pred: [B, 1, H, W] or [1, H, W] or [H, W] Tensor or ndarray
        target: [B, 1, H, W] or [1, H, W] or [H, W] Tensor or ndarray
        data_range: Dynamic range of values (default 1.0)
        
    Returns:
        float: Mean SSIM value
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(np.asarray(pred, dtype=np.float32))
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(np.asarray(target, dtype=np.float32))

    pred = pred.detach().float()
    target = target.detach().float()

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)

    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)

    channel = pred.size(1)
    kernel = _gaussian_kernel_2d(window_size, sigma).to(pred.device).type_as(pred)
    kernel = kernel.repeat(channel, 1, 1, 1)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channel) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def calculate_batch_ssim(preds, targets, data_range=1.0):
    return calculate_ssim(preds, targets, data_range=data_range)
