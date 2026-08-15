"""
Quick Single-Image or Batch Inference Tool for FastNAF-SR V5.

Usage:
python infer.py --input path/to/sample.npy --output path/to/restored.npy [--checkpoint checkpoints/best_overall.pth]
"""

import os
import argparse
import numpy as np
import torch

from models.fastnaf_sr import FastNAFSR_V5
from datasets.preprocessing import normalize, denormalize
from utils.tiled_inference import tiled_inference
from evaluate import find_default_checkpoint


def infer_single(input_path, output_path, checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_default_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found.")

    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ckpt.get("config", {}).get("model", {})
    model = FastNAFSR_V5(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        channels=model_cfg.get("channels", 48),
        num_lr_blocks=model_cfg.get("num_lr_blocks", 8),
        num_hr_blocks=model_cfg.get("num_hr_blocks", 4),
        upscale_factor=model_cfg.get("upscale_factor", 2),
    ).to(device)

    if "ema_state" in ckpt and ckpt["ema_state"] is not None:
        model.load_state_dict(ckpt["ema_state"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    raw_lr = np.load(input_path).astype(np.float32)
    norm_lr = normalize(raw_lr)
    tensor_lr = torch.from_numpy(norm_lr).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        if raw_lr.shape[0] <= 128 and raw_lr.shape[1] <= 128:
            pred_tensor = model(tensor_lr)
        else:
            pred_tensor = tiled_inference(model, tensor_lr, tile_size=128, tile_overlap=16, scale=2, device=device)

    pred_raw = denormalize(pred_tensor).squeeze().cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, pred_raw)
    print(f"Restored: {input_path} (shape {raw_lr.shape}) -> {output_path} (shape {pred_raw.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Inference")
    parser.add_argument("--input", required=True, help="Input .npy image file")
    parser.add_argument("--output", required=True, help="Output .npy destination file")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    infer_single(args.input, args.output, args.checkpoint, args.device)
