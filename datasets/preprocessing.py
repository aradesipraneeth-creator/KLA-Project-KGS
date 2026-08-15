"""
Centralized Preprocessing & Normalization Module for KLA Semiconductor Restoration.

Uses fixed dataset statistics:
- Mean: 0.43353602
- Std:  0.28478748
"""

import numpy as np
import torch

# Global dataset statistics
DATASET_MEAN = 0.43353602
DATASET_STD = 0.28478748


class DataNormalizer:
    """
    Standard Z-score normalization and denormalization using global dataset statistics.
    Works seamlessly on float32 NumPy arrays and PyTorch tensors.
    """
    def __init__(self, mean=DATASET_MEAN, std=DATASET_STD):
        self.mean = float(mean)
        self.std = float(std)

    def normalize(self, x):
        """
        Normalized space: (x - mean) / std
        """
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.std
        elif isinstance(x, np.ndarray):
            return (x - self.mean) / self.std
        else:
            return (float(x) - self.mean) / self.std

    def denormalize(self, x):
        """
        Restores back to original dynamic range: x * std + mean
        """
        if isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        elif isinstance(x, np.ndarray):
            return x * self.std + self.mean
        else:
            return float(x) * self.std + self.mean


# Shared global instance
normalizer = DataNormalizer()


def normalize(x):
    return normalizer.normalize(x)


def denormalize(x):
    return normalizer.denormalize(x)
