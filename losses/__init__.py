try:
    from .charbonnier_loss import CharbonnierLoss
except ImportError:
    from losses.charbonnier_loss import CharbonnierLoss

try:
    from .ssim_loss import SSIMLoss, ssim
except ImportError:
    from losses.ssim_loss import SSIMLoss, ssim

try:
    from .edge_loss import EdgeLoss
except ImportError:
    from losses.edge_loss import EdgeLoss

try:
    from .combined_loss import RestorationLoss
except ImportError:
    from losses.combined_loss import RestorationLoss

__all__ = [
    "CharbonnierLoss",
    "SSIMLoss",
    "ssim",
    "EdgeLoss",
    "RestorationLoss",
]
