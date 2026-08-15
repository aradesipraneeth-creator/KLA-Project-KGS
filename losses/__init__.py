try:
    from losses.charbonnier import CharbonnierLoss
except ImportError:
    try:
        from .charbonnier import CharbonnierLoss
    except ImportError:
        from .charbonnier_loss import CharbonnierLoss

try:
    from losses.ssim_loss import SSIMLoss, ssim
except ImportError:
    from .ssim_loss import SSIMLoss, ssim

try:
    from losses.edge_loss import EdgeLoss
except ImportError:
    from .edge_loss import EdgeLoss

try:
    from losses.combined_loss import RestorationLoss
except ImportError:
    from .combined_loss import RestorationLoss

__all__ = [
    "CharbonnierLoss",
    "SSIMLoss",
    "ssim",
    "EdgeLoss",
    "RestorationLoss",
]
