"""
Charbonnier Loss alias module matching standard repository structure.
"""

try:
    from losses.charbonnier import CharbonnierLoss
except ImportError:
    from .charbonnier import CharbonnierLoss

__all__ = ["CharbonnierLoss"]
