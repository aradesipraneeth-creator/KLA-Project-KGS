"""
SSIM Loss module alias matching standard repository structure.
"""

from losses.ssim_loss import SSIMLoss, ssim, gaussian, create_window

__all__ = ["SSIMLoss", "ssim", "gaussian", "create_window"]
