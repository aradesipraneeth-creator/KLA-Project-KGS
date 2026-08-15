"""
LPIPS (Learned Perceptual Image Patch Similarity) Metric Module.

Handles grayscale input (expands 1 to 3 channels for perceptual backbone)
and includes graceful fallback if lpips is unavailable.
"""

import torch
import torch.nn as nn
import numpy as np


class LPIPSMetric(nn.Module):
    def __init__(self, net="alex", device="cpu"):
        super().__init__()
        self.device = device
        self.available = False
        self.loss_fn = None

        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
            self.loss_fn.eval()
            self.available = True
        except Exception:
            self.available = False

    def forward(self, pred, target):
        """
        Computes LPIPS between pred and target.
        Expects inputs normalized to [-1, 1] or raw grayscale.
        """
        if not self.available or self.loss_fn is None:
            return float("nan")

        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(np.asarray(pred, dtype=np.float32))
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(np.asarray(target, dtype=np.float32))

        pred = pred.to(self.device).float()
        target = target.to(self.device).float()

        if pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 3:
            pred = pred.unsqueeze(0)

        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(0)

        # Grayscale (1-channel) to 3-channel
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        # Scale to [-1, 1] range expected by LPIPS
        pred_scaled = torch.clamp(pred * 2.0 - 1.0, -1.0, 1.0)
        target_scaled = torch.clamp(target * 2.0 - 1.0, -1.0, 1.0)

        with torch.no_grad():
            lpips_val = self.loss_fn(pred_scaled, target_scaled)
            return float(lpips_val.mean().item())


def calculate_lpips(pred, target, device="cpu"):
    metric = LPIPSMetric(device=device)
    return metric(pred, target)
