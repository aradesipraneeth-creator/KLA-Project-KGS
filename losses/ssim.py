"""
SSIM Loss module alias matching standard repository structure.
"""

try:
    from .ssim_loss import SSIMLoss, ssim, gaussian, create_window
except ImportError:
    from losses.ssim_loss import SSIMLoss, ssim, gaussian, create_window

__all__ = ["SSIMLoss", "ssim", "gaussian", "create_window"]

