"""
FastNAF-SR V5 Architecture for Semiconductor Image Restoration.

Specification:
1. Input: 1-channel Grayscale [B, 1, H, W]
2. Shallow Feature Extraction: 3x3 Conv (1 -> 48)
3. Low-Resolution Restoration: 8 NAF Blocks (channels=48)
4. Residual Feature Fusion: Skip connection adding shallow features to LR restored features
5. Upsampling: 3x3 Conv (48 -> 192) + PixelShuffle(2) -> [B, 48, 2H, 2W]
6. High-Resolution Refinement: 4 NAF Blocks (channels=48)
7. Output Reconstruction: 3x3 Conv (48 -> 1) -> [B, 1, 2H, 2W]
"""

import torch
import torch.nn as nn
from models.naf_block import NAFBlock


class FastNAFSR_V5(nn.Module):
    """
    FastNAF-SR V5 Model.
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=48,
        num_lr_blocks=8,
        num_hr_blocks=4,
        upscale_factor=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.upscale_factor = upscale_factor

        # 1. Shallow Feature Extraction
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True)

        # 2. Low-Resolution Restoration (8 NAF Blocks)
        self.lr_restoration = nn.Sequential(
            *[NAFBlock(channels=channels) for _ in range(num_lr_blocks)]
        )

        # 3. Upsampling (48 -> 192 -> PixelShuffle(2) -> 48)
        upsample_channels = channels * (upscale_factor ** 2)
        self.upconv = nn.Conv2d(channels, upsample_channels, kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # 4. High-Resolution Refinement (4 NAF Blocks)
        self.hr_refinement = nn.Sequential(
            *[NAFBlock(channels=channels) for _ in range(num_hr_blocks)]
        )

        # 5. Output Reconstruction (48 -> 1)
        self.tail = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        """
        Forward pass:
        Input: [B, 1, H, W]
        Output: [B, 1, 2H, 2W]
        """
        # Shallow features
        feat_shallow = self.stem(x)

        # LR restoration
        feat_lr = self.lr_restoration(feat_shallow)

        # Residual Feature Fusion
        feat_fused = feat_lr + feat_shallow

        # PixelShuffle x2 Upsampling
        feat_up = self.upconv(feat_fused)
        feat_up = self.pixel_shuffle(feat_up)

        # HR refinement
        feat_hr = self.hr_refinement(feat_up)

        # Final output reconstruction
        out = self.tail(feat_hr)

        return out


def build_model(config=None):
    """
    Helper function to instantiate FastNAF-SR V5 model.
    """
    if config is None:
        config = {}
    return FastNAFSR_V5(
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        channels=config.get("channels", 48),
        num_lr_blocks=config.get("num_lr_blocks", 8),
        num_hr_blocks=config.get("num_hr_blocks", 4),
        upscale_factor=config.get("upscale_factor", 2),
    )
