"""
Standalone Evaluation & Inference Script for KLA Semiconductor Restoration.

Evaluator Usage:
python evaluate.py --input_dir PATH_TO_TEST_IMAGES --output_dir PATH_TO_OUTPUT

Features:
- Completely standalone and self-contained
- Automatic checkpoint resolution (best_overall.pth -> best_psnr.pth -> latest.pth)
- Automatic CUDA detection
- Clean filtering of __MACOSX and ._* metadata files
- Exact filename preservation
- Identical normalization and denormalization with global dataset statistics
- Direct batched inference for uniform 128x128 images; seamless tiled inference for large images
- Saves restored float32 .npy files to output_dir
- Generates timing and performance summary
"""

import os
import sys

# Ensure repository root is in python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

from models.fastnaf_sr import FastNAFSR_V5
from datasets.preprocessing import normalize, denormalize
from datasets.kla_dataset import get_clean_npy_filelist
from utils.tiled_inference import tiled_inference
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim


def find_default_checkpoint():
    """
    Finds the best available checkpoint in checkpoints/ or experiments/.
    """
    candidates = [
        "checkpoints/best_overall.pth",
        "checkpoints/best_psnr.pth",
        "checkpoints/best_ssim.pth",
        "checkpoints/latest.pth",
        "fastnaf_sr.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Search in experiments directory
    if os.path.exists("experiments"):
        for root, _, files in os.walk("experiments"):
            if "best.pth" in files:
                return os.path.join(root, "best.pth")
            if "best_psnr.pth" in files:
                return os.path.join(root, "best_psnr.pth")

    return None


def run_evaluation(
    input_dir,
    output_dir,
    checkpoint_path=None,
    gt_dir=None,
    batch_size=16,
    tile_size=128,
    tile_overlap=16,
    device_name=None,
    use_ema=True,
):
    # 1. Device selection
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Smart discovery of input_dir if needed
    candidates = [
        input_dir,
        os.path.join(input_dir, "NoisyLR"),
        "Test_NoisyLR",
        "Test_NoisyLR/NoisyLR",
        "./Test_NoisyLR",
        "./Test_NoisyLR/NoisyLR",
        "NoisyLR",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and len(get_clean_npy_filelist(c)) > 0:
            input_dir = c
            break

    print("=" * 65)
    print("KLA SEMICONDUCTOR RESTORATION — EVALUATION PIPELINE")
    print("=" * 65)
    print(f"Compute Device:      {device}")
    print(f"Input Directory:     {input_dir}")
    print(f"Output Directory:    {output_dir}")

    # 2. Checkpoint resolution
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        resolved_ckpt = find_default_checkpoint()
        if resolved_ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint provided and no saved checkpoint found in checkpoints/ or experiments/.\n"
                f"Please provide --checkpoint <path_to_checkpoint.pth>"
            )
        checkpoint_path = resolved_ckpt

    print(f"Loading Checkpoint:  {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 3. Model setup
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    model = FastNAFSR_V5(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        channels=model_cfg.get("channels", 48),
        num_lr_blocks=model_cfg.get("num_lr_blocks", 8),
        num_hr_blocks=model_cfg.get("num_hr_blocks", 4),
        upscale_factor=model_cfg.get("upscale_factor", 2),
    ).to(device)

    # Load weights (prefer EMA if available)
    if use_ema and "ema_state" in ckpt and ckpt["ema_state"] is not None:
        model.load_state_dict(ckpt["ema_state"])
        print("Using Exponential Moving Average (EMA) model weights.")
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print("Using standard model weights.")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # 4. Discover input files
    input_files = get_clean_npy_filelist(input_dir)
    assert len(input_files) > 0, f"No valid .npy files found in {input_dir}"
    print(f"Found {len(input_files)} valid input images to restore.")
    print("-" * 65)

    # 5. Process images
    inference_times = []
    psnr_scores = []
    ssim_scores = []

    # Warmup
    dummy = torch.randn(1, 1, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()

    num_files = len(input_files)
    for idx in tqdm(range(0, num_files, batch_size), desc="Restoring images"):
        batch_filenames = input_files[idx : idx + batch_size]
        batch_raws = [np.load(os.path.join(input_dir, fn)).astype(np.float32) for fn in batch_filenames]

        all_128 = all(img.shape == (128, 128) for img in batch_raws)

        if all_128:
            batch_norm = np.stack([normalize(img) for img in batch_raws], axis=0)
            batch_tensor = torch.from_numpy(batch_norm).unsqueeze(1).float().to(device)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                pred_tensors = model(batch_tensor)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            per_img_time = ((t1 - t0) * 1000.0) / len(batch_filenames)
            for _ in batch_filenames:
                inference_times.append(per_img_time)

            pred_raws = denormalize(pred_tensors).squeeze(1).cpu().numpy().astype(np.float32)

            for i, fn in enumerate(batch_filenames):
                out_path = os.path.join(output_dir, fn)
                np.save(out_path, pred_raws[i])

                if gt_dir is not None:
                    gt_path = os.path.join(gt_dir, fn)
                    if os.path.exists(gt_path):
                        gt_raw = np.load(gt_path).astype(np.float32)
                        p = calculate_psnr(pred_raws[i], gt_raw, data_range=1.0)
                        s = calculate_ssim(pred_raws[i], gt_raw, data_range=1.0)
                        psnr_scores.append(p)
                        ssim_scores.append(s)
        else:
            for fn, raw_lr in zip(batch_filenames, batch_raws):
                norm_lr = normalize(raw_lr)
                tensor_lr = torch.from_numpy(norm_lr).unsqueeze(0).unsqueeze(0).float().to(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    pred_tensor = tiled_inference(
                        model, tensor_lr, tile_size=tile_size, tile_overlap=tile_overlap, scale=2, device=device
                    )
                t1 = time.perf_counter()
                inference_times.append((t1 - t0) * 1000.0)

                pred_raw = denormalize(pred_tensor).squeeze().cpu().numpy().astype(np.float32)
                out_path = os.path.join(output_dir, fn)
                np.save(out_path, pred_raw)

                if gt_dir is not None:
                    gt_path = os.path.join(gt_dir, fn)
                    if os.path.exists(gt_path):
                        gt_raw = np.load(gt_path).astype(np.float32)
                        p = calculate_psnr(pred_raw, gt_raw, data_range=1.0)
                        s = calculate_ssim(pred_raw, gt_raw, data_range=1.0)
                        psnr_scores.append(p)
                        ssim_scores.append(s)

    # 6. Performance Summary
    mean_latency = float(np.mean(inference_times))
    median_latency = float(np.median(inference_times))
    p95_latency = float(np.percentile(inference_times, 95))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    print("\n" + "=" * 65)
    print("EVALUATION & INFERENCE SUMMARY")
    print("=" * 65)
    print(f"Total Images Processed: {len(input_files)}")
    print(f"Average Latency:        {mean_latency:.2f} ms/image")
    print(f"Median Latency:         {median_latency:.2f} ms/image")
    print(f"P95 Latency:            {p95_latency:.2f} ms/image")
    print(f"Throughput (FPS):       {fps:.1f} frames/sec")

    if psnr_scores:
        print("-" * 65)
        print(f"Mean Restored PSNR:     {np.mean(psnr_scores):.4f} dB")
        print(f"Mean Restored SSIM:     {np.mean(ssim_scores):.4f}")

    print(f"Outputs Saved to:       {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FastNAF-SR V5 on semiconductor images")
    parser.add_argument("--input_dir", required=True, help="Path to input directory containing .npy files")
    parser.add_argument("--output_dir", required=True, help="Path to output directory to save restored .npy files")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint .pth")
    parser.add_argument("--gt_dir", default=None, help="Optional GT directory for PSNR/SSIM evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--tile_size", type=int, default=128, help="Tile size for large images")
    parser.add_argument("--tile_overlap", type=int, default=16, help="Tile overlap in pixels")
    parser.add_argument("--overlap", type=int, default=None, help="Alias for tile_overlap in pixels")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 autocast during evaluation")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument("--no_ema", action="store_true", help="Do not use EMA weights")
    args = parser.parse_args()

    effective_overlap = args.overlap if args.overlap is not None else args.tile_overlap

    run_evaluation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        gt_dir=args.gt_dir,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        tile_overlap=effective_overlap,
        device_name=args.device,
        use_ema=not args.no_ema,
    )
