"""
Restoration-Aware Composite Loss Function.

Total Loss = lambda1 * L_recon + lambda2 * L_structural + lambda3 * L_edge
"""

import torch
import torch.nn as nn

try:
    from .charbonnier_loss import CharbonnierLoss
except ImportError:
    from losses.charbonnier_loss import CharbonnierLoss

try:
    from .ssim_loss import SSIMLoss
except ImportError:
    from losses.ssim_loss import SSIMLoss

try:
    from .edge_loss import EdgeLoss
except ImportError:
    from losses.edge_loss import EdgeLoss


class RestorationLoss(nn.Module):
    def __init__(
        self,
        recon_type="charbonnier",
        lambda_recon=1.0,
        lambda_ssim=0.5,
        lambda_edge=0.05,
        edge_mode="sobel",
        ssim_data_range=None,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

        # 1. Reconstruction loss
        if recon_type.lower() == "charbonnier":
            self.recon_loss = CharbonnierLoss()
        elif recon_type.lower() == "l1":
            self.recon_loss = nn.L1Loss()
        elif recon_type.lower() == "l2" or recon_type.lower() == "mse":
            self.recon_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported reconstruction loss: {recon_type}")

        # 2. Structural loss (SSIM)
        self.ssim_loss = SSIMLoss(channel=1, data_range=ssim_data_range)

        # 3. Edge loss
        self.edge_loss = EdgeLoss(mode=edge_mode)

    def forward(self, pred, target):
        loss_dict = {}

        l_recon = self.recon_loss(pred, target)
        loss_dict["loss_recon"] = l_recon.item()

        total = self.lambda_recon * l_recon

        if self.lambda_ssim > 0:
            l_ssim = self.ssim_loss(pred, target)
            loss_dict["loss_ssim"] = l_ssim.item()
            total = total + self.lambda_ssim * l_ssim

        if self.lambda_edge > 0:
            l_edge = self.edge_loss(pred, target)
            loss_dict["loss_edge"] = l_edge.item()
            total = total + self.lambda_edge * l_edge

        loss_dict["loss_total"] = total.item()
        return total, loss_dict
