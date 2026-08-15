"""
NAFNet-style Restoration Blocks for FastNAF-SR V5.

Reference: Simple Baselines for Image Restoration (NAFNet)
Components:
- LayerNorm2d (Channel-first 2D Layer Normalization)
- SimpleGate (x1 * x2 elementwise multiplication)
- Simplified Channel Attention (SCA)
- Depthwise Convolution
- Learnable residual scaling (beta / gamma)
- Residual skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    Channel-wise Layer Normalization for 2D feature maps [B, C, H, W].
    Uses PyTorch's native vectorized C++/CUDA kernel for maximum throughput.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, H, W] -> permute to [B, H, W, C] for native LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """
    SimpleGate: splits feature map along channel dimension into two halves
    and computes element-wise multiplication: f(x1, x2) = x1 * x2.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA):
    Global Average Pooling -> 1x1 Conv -> Elementwise scaling
    """
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B, C, H, W]
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFBlock(nn.Module):
    """
    NAFBlock for FastNAF-SR V5.
    
    Structure:
    Input x
      ↓ (Branch 1: Spatial & Channel Attention)
    LayerNorm2d
      ↓
    1x1 Conv (C -> 2C)
      ↓
    3x3 Depthwise Conv (2C -> 2C, groups=2C)
      ↓
    SimpleGate (2C -> C)
      ↓
    Simplified Channel Attention (SCA)
      ↓
    1x1 Conv (C -> C)
      ↓
    * beta + x
      ↓ (Branch 2: Feed-Forward Network)
    LayerNorm2d
      ↓
    1x1 Conv (C -> 2C)
      ↓
    SimpleGate (2C -> C)
      ↓
    1x1 Conv (C -> C)
      ↓
    * gamma + residual
    """
    def __init__(self, channels=48, expansion=2):
        super().__init__()
        expanded_c = channels * expansion

        # Spatial / Attention Branch
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded_c, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(expanded_c, expanded_c, kernel_size=3, padding=1, groups=expanded_c, bias=True)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

        # Feed-Forward Network Branch
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, expanded_c, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        # Spatial / Attention Branch
        res = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = res + y * self.beta

        # Feed-Forward Branch
        res = x
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = res + y * self.gamma

        return x
