"""
Comprehensive Benchmarking Suite for FastNAF-SR V5.

Measures:
- Parameter count and Checkpoint size
- GPU VRAM allocation and Peak Memory
- FP32 and FP16 / AMP Latencies (Mean, Median, P95)
- FPS Throughput
- Breakdown: Preprocessing, Forward Inference, Postprocessing
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from models.fastnaf_sr import FastNAFSR_V5
from datasets.preprocessing import normalize, denormalize
from evaluate import find_default_checkpoint


def benchmark_model(
    checkpoint_path=None,
    device_name=None,
    num_runs=100,
    warmup_runs=20,
    input_size=(1, 1, 128, 128),
    save_csv="benchmark_results.csv",
):
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("FASTNAF-SR V5 BENCHMARKING SUITE")
    print("=" * 65)
    print(f"Compute Device:      {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU Name:            {torch.cuda.get_device_name(0)}")

    # 1. Model & Weights
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_default_checkpoint()

    model = FastNAFSR_V5(in_channels=1, out_channels=1, channels=48, num_lr_blocks=8, num_hr_blocks=4).to(device)

    ckpt_size_mb = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "ema_state" in ckpt and ckpt["ema_state"] is not None:
            model.load_state_dict(ckpt["ema_state"])
        elif "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Loaded Checkpoint:   {checkpoint_path} ({ckpt_size_mb:.2f} MB)")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Parameters:{trainable_params:,}")
    print(f"Input Shape:         {input_size} (LR Grayscale)")
    print(f"Output Shape:        (1, 1, {input_size[2]*2}, {input_size[3]*2}) (2x HR Restored)")
    print("-" * 65)

    precisions = ["FP32"]
    if torch.cuda.is_available() and device.type == "cuda":
        precisions.append("FP16/AMP")

    results = []

    for precision in precisions:
        use_fp16 = precision == "FP16/AMP"
        print(f"\nBenchmarking [{precision}] mode...")

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Dummy raw input
        raw_np = np.random.randn(input_size[2], input_size[3]).astype(np.float32)

        # Warmup
        for _ in range(warmup_runs):
            norm_lr = normalize(raw_np)
            tensor_in = torch.from_numpy(norm_lr).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out = model(tensor_in)
            _ = denormalize(out).cpu().numpy()

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()

        pre_times = []
        infer_times = []
        post_times = []
        total_times = []

        for _ in range(num_runs):
            # Preprocessing
            t0 = time.perf_counter()
            norm_lr = normalize(raw_np)
            tensor_in = torch.from_numpy(norm_lr).unsqueeze(0).unsqueeze(0).to(device)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Model Forward Pass
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out = model(tensor_in)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # Postprocessing (Denormalize & convert to numpy)
            out_raw = denormalize(out).squeeze().cpu().numpy()
            t3 = time.perf_counter()

            pre_times.append((t1 - t0) * 1000.0)
            infer_times.append((t2 - t1) * 1000.0)
            post_times.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)

        mean_infer = float(np.mean(infer_times))
        median_infer = float(np.median(infer_times))
        p95_infer = float(np.percentile(infer_times, 95))
        fps = 1000.0 / mean_infer if mean_infer > 0 else 0.0

        mean_pre = float(np.mean(pre_times))
        mean_post = float(np.mean(post_times))
        mean_total = float(np.mean(total_times))

        peak_mem_mb = 0.0
        curr_mem_mb = 0.0
        if torch.cuda.is_available() and device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            curr_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        res_entry = {
            "Precision": precision,
            "Parameters": total_params,
            "Checkpoint_MB": round(ckpt_size_mb, 2),
            "Pre_ms": round(mean_pre, 3),
            "Infer_Mean_ms": round(mean_infer, 3),
            "Infer_Median_ms": round(median_infer, 3),
            "Infer_P95_ms": round(p95_infer, 3),
            "Post_ms": round(mean_post, 3),
            "Total_Pipeline_ms": round(mean_total, 3),
            "FPS": round(fps, 1),
            "GPU_Peak_MB": round(peak_mem_mb, 2),
        }
        results.append(res_entry)

        print(f"--- Results for {precision} ---")
        print(f"Preprocessing Time:  {mean_pre:.3f} ms")
        print(f"Model Forward Mean:  {mean_infer:.3f} ms")
        print(f"Model Forward Median:{median_infer:.3f} ms")
        print(f"Model Forward P95:   {p95_infer:.3f} ms")
        print(f"Postprocessing Time: {mean_post:.3f} ms")
        print(f"Total Pipeline:      {mean_total:.3f} ms")
        print(f"Throughput:          {fps:.1f} FPS")
        if torch.cuda.is_available() and device.type == "cuda":
            print(f"Peak GPU VRAM:       {peak_mem_mb:.2f} MB")

    # Save to CSV
    if save_csv:
        import csv
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved benchmark metrics to {save_csv}")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FastNAF-SR V5")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs")
    parser.add_argument("--csv", default="benchmark_results.csv", help="Output CSV path")
    args = parser.parse_args()

    benchmark_model(
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        save_csv=args.csv,
    )
