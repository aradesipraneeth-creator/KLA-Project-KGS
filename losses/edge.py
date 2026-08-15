"""
Edge-Aware Loss module alias matching standard repository structure.
"""

try:
    from .edge_loss import EdgeLoss
except ImportError:
    from losses.edge_loss import EdgeLoss

__all__ = ["EdgeLoss"]

