"""
Final Test Output Generator for KLA Semiconductor Restoration Submission.

Processes:
- Test_NoisyLR/NoisyLR/*.npy (400 images)
Saves:
- test_outputs/*.npy
- test_outputs/manifest.csv
"""

import os
import sys
import csv
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from tqdm import tqdm

from models.fastnaf_sr import FastNAFSR_V5
from datasets.preprocessing import normalize, denormalize
from datasets.kla_dataset import get_clean_npy_filelist
from utils.tiled_inference import tiled_inference
from evaluate import find_default_checkpoint


def generate_submission_outputs(
    input_dir="Test_NoisyLR/NoisyLR",
    output_dir="test_outputs",
    checkpoint_path=None,
    batch_size=16,
    device_name=None,
):
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("GENERATING FINAL TEST OUTPUTS & SUBMISSION MANIFEST")
    print("=" * 65)
    print(f"Device:           {device}")
    print(f"Test Input Dir:   {input_dir}")
    print(f"Test Output Dir:  {output_dir}")
    print(f"Batch Size:       {batch_size}")

    # Checkpoint
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_default_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found.")

    print(f"Using Checkpoint: {checkpoint_path}")
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
        print("Using EMA weights.")
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print("Using model state weights.")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    input_files = get_clean_npy_filelist(input_dir)
    assert len(input_files) > 0, f"No test files found in {input_dir}"
    print(f"Found {len(input_files)} test files.")
    print("-" * 65)

    manifest_rows = []

    # Warmup
    dummy = torch.randn(1, 1, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()

    # Process in batches
    num_files = len(input_files)
    for idx in tqdm(range(0, num_files, batch_size), desc="Generating test outputs"):
        batch_filenames = input_files[idx : idx + batch_size]
        batch_raws = [np.load(os.path.join(input_dir, fn)).astype(np.float32) for fn in batch_filenames]

        # Check if all images in batch are 128x128
        all_128 = all(img.shape == (128, 128) for img in batch_raws)

        if all_128:
            batch_norm = np.stack([normalize(img) for img in batch_raws], axis=0)  # [B, H, W]
            batch_tensor = torch.from_numpy(batch_norm).unsqueeze(1).float().to(device)  # [B, 1, H, W]

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                pred_tensors = model(batch_tensor)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            infer_ms_per_img = ((t1 - t0) * 1000.0) / len(batch_filenames)
            pred_raws = denormalize(pred_tensors).squeeze(1).cpu().numpy().astype(np.float32)

            for i, fn in enumerate(batch_filenames):
                out_path = os.path.join(output_dir, fn)
                np.save(out_path, pred_raws[i])
                manifest_rows.append({
                    "filename": fn,
                    "input_shape": f"{batch_raws[i].shape[0]}x{batch_raws[i].shape[1]}",
                    "output_shape": f"{pred_raws[i].shape[0]}x{pred_raws[i].shape[1]}",
                    "inference_time_ms": f"{infer_ms_per_img:.3f}",
                })
        else:
            # Individual tiled inference for irregular / large sizes
            for fn, raw_lr in zip(batch_filenames, batch_raws):
                norm_lr = normalize(raw_lr)
                tensor_lr = torch.from_numpy(norm_lr).unsqueeze(0).unsqueeze(0).float().to(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    pred_t = tiled_inference(model, tensor_lr, tile_size=128, tile_overlap=16, scale=2, device=device)
                t1 = time.perf_counter()

                infer_ms = (t1 - t0) * 1000.0
                pred_raw = denormalize(pred_t).squeeze().cpu().numpy().astype(np.float32)

                out_path = os.path.join(output_dir, fn)
                np.save(out_path, pred_raw)

                manifest_rows.append({
                    "filename": fn,
                    "input_shape": f"{raw_lr.shape[0]}x{raw_lr.shape[1]}",
                    "output_shape": f"{pred_raw.shape[0]}x{pred_raw.shape[1]}",
                    "inference_time_ms": f"{infer_ms:.3f}",
                })

    # Save manifest.csv
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "input_shape", "output_shape", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    latencies = [float(r["inference_time_ms"]) for r in manifest_rows]
    print("\n" + "=" * 65)
    print(f"Successfully generated {len(manifest_rows)} test output files.")
    print(f"Average Inference Latency: {np.mean(latencies):.2f} ms/image")
    print(f"Manifest written to:       {manifest_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test Outputs")
    parser.add_argument("--input_dir", default="Test_NoisyLR/NoisyLR", help="Input test folder")
    parser.add_argument("--output_dir", default="test_outputs", help="Output folder")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Device")
    args = parser.parse_args()

    generate_submission_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
