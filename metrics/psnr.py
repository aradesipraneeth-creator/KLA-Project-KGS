"""
PSNR (Peak Signal-to-Noise Ratio) calculation for image restoration.
Supports both PyTorch tensors and NumPy arrays.
"""

import math
import numpy as np
import torch


def calculate_psnr(pred, target, data_range=1.0, eps=1e-10):
    """
    Computes PSNR between prediction and target.
    
    Args:
        pred: Predicted image [B, 1, H, W] or [H, W] (Tensor or ndarray)
        target: Ground truth image [B, 1, H, W] or [H, W] (Tensor or ndarray)
        data_range: Dynamic range of the images (default 1.0)
        eps: Small epsilon to avoid divide by zero
        
    Returns:
        float: PSNR in dB
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().float()

    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        mse = torch.mean((pred - target) ** 2).item()
    else:
        pred_np = np.asarray(pred, dtype=np.float32)
        target_np = np.asarray(target, dtype=np.float32)
        mse = float(np.mean((pred_np - target_np) ** 2))

    if mse < eps:
        return 100.0  # Identical images

    return 10.0 * math.log10((data_range ** 2) / mse)


def calculate_batch_psnr(preds, targets, data_range=1.0):
    """
    Computes mean PSNR across a batch.
    """
    if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
        b = preds.size(0)
        psnrs = []
        for i in range(b):
            psnrs.append(calculate_psnr(preds[i], targets[i], data_range=data_range))
        return float(np.mean(psnrs))
    else:
        return calculate_psnr(preds, targets, data_range=data_range)
