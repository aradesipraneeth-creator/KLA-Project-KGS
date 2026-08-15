"""
Tiled Inference Module for FastNAF-SR V5.

Allows high-resolution or arbitrarily large images to be processed in overlapping tiles
with smooth 2D linear/Hann blending to eliminate boundary artifacts.
For standard test images (e.g., 128x128), it automatically performs direct inference.
"""

import math
import numpy as np
import torch


def _create_2d_window(h, w, device="cpu"):
    """
    Creates a 2D Hann window for tile blending.
    """
    def _1d_window(size):
        if size == 1:
            return torch.ones(1, device=device)
        n = torch.arange(size, device=device).float()
        return 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n + 0.5) / size)

    wy = _1d_window(h).unsqueeze(1)
    wx = _1d_window(w).unsqueeze(0)
    w2d = wy * wx
    return w2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


@torch.no_grad()
def tiled_inference(
    model,
    img_tensor,
    tile_size=128,
    tile_overlap=16,
    scale=2,
    device="cpu",
):
    """
    Runs model inference on img_tensor [B, 1, H, W] with optional tiling.
    
    Args:
        model: PyTorch restoration model
        img_tensor: Input tensor [B, 1, H, W]
        tile_size: LR tile size (default: 128)
        tile_overlap: Overlap in LR pixels (default: 16)
        scale: Upscale factor (default: 2)
        device: Torch device
        
    Returns:
        Tensor [B, 1, H*scale, W*scale]
    """
    model.eval()
    b, c, h, w = img_tensor.shape

    # Direct inference if image fits within tile size
    if h <= tile_size and w <= tile_size:
        img_in = img_tensor.to(device)
        out = model(img_in)
        return out

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, max(1, h - tile_size + stride), stride))
    w_idx_list = list(range(0, max(1, w - tile_size + stride), stride))

    # Adjust last tile to touch image boundary
    if h_idx_list[-1] + tile_size < h:
        h_idx_list.append(h - tile_size)
    if w_idx_list[-1] + tile_size < w:
        w_idx_list.append(w - tile_size)

    out_h, out_w = h * scale, w * scale
    tile_out_size = tile_size * scale

    output = torch.zeros((b, c, out_h, out_w), device=device)
    weights = torch.zeros((b, c, out_h, out_w), device=device)

    # 2D Hann blending window for output tile
    window = _create_2d_window(tile_out_size, tile_out_size, device=device)

    for h_idx in h_idx_list:
        h_end = min(h_idx + tile_size, h)
        h_start = max(0, h_end - tile_size)

        for w_idx in w_idx_list:
            w_end = min(w_idx + tile_size, w)
            w_start = max(0, w_end - tile_size)

            tile_in = img_tensor[:, :, h_start:h_end, w_start:w_end].to(device)
            tile_out = model(tile_in)

            out_h_start, out_h_end = h_start * scale, h_end * scale
            out_w_start, out_w_end = w_start * scale, w_end * scale

            output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_out * window
            weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += window

    # Normalize by accumulated weights
    weights = torch.clamp(weights, min=1e-8)
    output = output / weights
    return output
