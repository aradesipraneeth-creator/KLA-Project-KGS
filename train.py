"""
Training Pipeline for FastNAF-SR V5 Semiconductor AI Image Restoration.

Features:
- Fixed FastNAF-SR V5 Architecture
- Safe synchronous paired augmentations & paired cropping
- Global dataset statistics normalization
- CUDA / AMP / CPU automatic detection and support
- AdamW optimizer + Cosine Annealing with Linear Warmup
- Gradient clipping
- Model Exponential Moving Average (EMA)
- Deterministic Train/Val split with filelist saving
- Multi-metric checkpointing: latest.pth, best_psnr.pth, best_ssim.pth, best_overall.pth
- Experiment logging with CSV, config copy, and summary JSON
- Baseline comparison logging
"""

import os
import sys

# Ensure repository root is in python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import math
import yaml
import random
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets.kla_dataset import KLAPairedDataset, get_clean_npy_filelist
from datasets.preprocessing import normalize, denormalize, DATASET_MEAN, DATASET_STD
from models.fastnaf_sr import FastNAFSR_V5
from losses.combined_loss import RestorationLoss
from metrics.psnr import calculate_psnr, calculate_batch_psnr
from metrics.ssim import calculate_ssim, calculate_batch_ssim
from metrics.lpips_metric import LPIPSMetric
from utils.ema import ModelEMA
from utils.logger import CSVLogger, save_summary_json


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def compute_baseline_metrics(val_loader, lpips_fn=None):
    """
    Computes baseline metrics using simple bicubic upsampling of LR input against GT.
    """
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for batch in val_loader:
        lr_raw = batch["lr_raw"]  # [B, 1, 128, 128]
        gt_raw = batch["gt_raw"]  # [B, 1, 256, 256]

        # Bicubic 2x upsampling of raw LR
        lr_up = nn.functional.interpolate(lr_raw, scale_factor=2.0, mode="bicubic", align_corners=False)

        for i in range(lr_up.size(0)):
            p = calculate_psnr(lr_up[i], gt_raw[i], data_range=1.0)
            s = calculate_ssim(lr_up[i], gt_raw[i], data_range=1.0)
            psnr_list.append(p)
            ssim_list.append(s)
            if lpips_fn is not None and lpips_fn.available:
                l_val = lpips_fn(lr_up[i], gt_raw[i])
                if not math.isnan(l_val):
                    lpips_list.append(l_val)

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")

    return mean_psnr, mean_ssim, mean_lpips


def validate(model, val_loader, criterion, device, lpips_fn=None):
    """
    Validates model on the validation set.
    Metrics (PSNR, SSIM, LPIPS) are computed on restored raw space (denormalized).
    """
    model.eval()
    val_losses = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            gt_raw = batch["gt_raw"].to(device)

            pred = model(lr)
            loss, _ = criterion(pred, gt)
            val_losses.append(loss.item())

            # Convert prediction back to original data range for evaluation
            pred_raw = denormalize(pred)

            for i in range(pred_raw.size(0)):
                p = calculate_psnr(pred_raw[i], gt_raw[i], data_range=1.0)
                s = calculate_ssim(pred_raw[i], gt_raw[i], data_range=1.0)
                psnr_list.append(p)
                ssim_list.append(s)

                if lpips_fn is not None and lpips_fn.available:
                    l_val = lpips_fn(pred_raw[i], gt_raw[i])
                    if not math.isnan(l_val):
                        lpips_list.append(l_val)

    mean_loss = float(np.mean(val_losses))
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")

    return {
        "val_loss": mean_loss,
        "val_psnr": mean_psnr,
        "val_ssim": mean_ssim,
        "val_lpips": mean_lpips,
    }


def train_pipeline(
    config_path="configs/fastnaf_sr.yaml",
    resume_ckpt=None,
    override_epochs=None,
    override_batch_size=None,
    override_train_root=None,
    override_lr_dir=None,
    override_gt_dir=None,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI Overrides
    if override_epochs is not None:
        cfg["training"]["epochs"] = override_epochs
    if override_batch_size is not None:
        cfg["data"]["batch_size"] = override_batch_size
    if override_train_root is not None:
        cfg["data"]["train_root"] = override_train_root

    seed = cfg["data"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"].get("amp", True) and torch.cuda.is_available()

    # Experiment run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(cfg["training"].get("log_dir", "experiments"), timestamp)
    save_dir = cfg["training"].get("save_dir", "checkpoints")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Save copy of configuration
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Robust discovery of NoisyLR and GT directories
    raw_train_root = cfg["data"].get("train_root", "train")

    possible_lr_dirs = []
    if override_lr_dir:
        possible_lr_dirs.append(override_lr_dir)
    possible_lr_dirs.extend([
        os.path.join(raw_train_root, "NoisyLR"),
        os.path.join(raw_train_root, "train", "NoisyLR"),
        "train/NoisyLR",
        "train/train/NoisyLR",
        "./train/NoisyLR",
        "./train/train/NoisyLR",
        "NoisyLR",
    ])

    possible_gt_dirs = []
    if override_gt_dir:
        possible_gt_dirs.append(override_gt_dir)
    possible_gt_dirs.extend([
        os.path.join(raw_train_root, "GT"),
        os.path.join(raw_train_root, "train", "GT"),
        "train/GT",
        "train/train/GT",
        "./train/GT",
        "./train/train/GT",
        "GT",
    ])

    lr_dir = None
    for p in possible_lr_dirs:
        if os.path.isdir(p) and len(get_clean_npy_filelist(p)) > 0:
            lr_dir = p
            break

    gt_dir = None
    for p in possible_gt_dirs:
        if os.path.isdir(p) and len(get_clean_npy_filelist(p)) > 0:
            gt_dir = p
            break

    if lr_dir is None or gt_dir is None:
        raise FileNotFoundError(
            f"Could not locate paired NoisyLR and GT folders.\n"
            f"Searched LR candidates: {possible_lr_dirs}\n"
            f"Searched GT candidates: {possible_gt_dirs}\n"
            f"Please specify --train_root (e.g. --train_root train) or --lr_dir and --gt_dir."
        )

    lr_files = set(get_clean_npy_filelist(lr_dir))
    gt_files = set(get_clean_npy_filelist(gt_dir))
    common_files = sorted(list(lr_files.intersection(gt_files)))

    assert len(common_files) > 0, (
        f"No paired training data found between:\n"
        f"  LR dir: {lr_dir} ({len(lr_files)} files)\n"
        f"  GT dir: {gt_dir} ({len(gt_files)} files)"
    )

    # Deterministic split
    val_ratio = cfg["data"].get("val_split", 0.1)
    rng = random.Random(seed)
    shuffled = list(common_files)
    rng.shuffle(shuffled)

    num_val = int(len(shuffled) * val_ratio)
    val_files = sorted(shuffled[:num_val])
    train_files = sorted(shuffled[num_val:])

    # Save split lists
    with open(os.path.join(exp_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(exp_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_files))

    print("=" * 65)
    print("FASTNAF-SR V5 SEMICONDUCTOR RESTORATION TRAINING")
    print("=" * 65)
    print(f"Device:               {device}")
    print(f"AMP Enabled:          {use_amp}")
    print(f"Total Paired Samples: {len(common_files)}")
    print(f"Train Samples:        {len(train_files)}")
    print(f"Val Samples:          {len(val_files)}")
    print(f"Batch Size:           {cfg['data']['batch_size']}")
    print(f"Epochs:               {cfg['training']['epochs']}")
    print(f"Learning Rate:        {cfg['training']['lr']}")
    print(f"Experiment Directory: {exp_dir}")
    print("-" * 65)

    # Datasets and Loaders
    train_dataset = KLAPairedDataset(
        lr_dir,
        gt_dir,
        file_list=train_files,
        is_train=True,
        patch_size=cfg["data"].get("patch_size", None),
        augment=cfg["data"].get("augment", True),
    )
    val_dataset = KLAPairedDataset(
        lr_dir,
        gt_dir,
        file_list=val_files,
        is_train=False,
        augment=False,
    )

    num_workers = cfg["data"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"].get("val_batch_size", 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = FastNAFSR_V5(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=cfg["model"].get("out_channels", 1),
        channels=cfg["model"].get("channels", 48),
        num_lr_blocks=cfg["model"].get("num_lr_blocks", 8),
        num_hr_blocks=cfg["model"].get("num_hr_blocks", 4),
        upscale_factor=cfg["model"].get("upscale_factor", 2),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"FastNAF-SR V5 Parameters: {total_params:,}")

    # EMA
    ema = ModelEMA(model, decay=cfg["training"].get("ema_decay", 0.999)).to(device)

    # Loss
    loss_cfg = cfg["loss"]
    criterion = RestorationLoss(
        recon_type=loss_cfg.get("recon_type", "charbonnier"),
        lambda_recon=loss_cfg.get("lambda_recon", 1.0),
        lambda_ssim=loss_cfg.get("lambda_ssim", 0.5),
        lambda_edge=loss_cfg.get("lambda_edge", 0.05),
        edge_mode=loss_cfg.get("edge_mode", "sobel"),
    ).to(device)

    # Optimizer & Scheduler
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    min_lr = cfg["training"].get("min_lr", 1e-6)
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    warmup_epochs = cfg["training"].get("warmup_epochs", 3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=use_amp)

    # LPIPS evaluator
    lpips_fn = LPIPSMetric(device=device)

    # Baseline evaluation
    print("\nComputing Baseline Metrics on Validation Split (Bicubic 2x)...")
    base_psnr, base_ssim, base_lpips = compute_baseline_metrics(val_loader, lpips_fn)
    print(f"Validation Baseline => PSNR: {base_psnr:.4f} dB | SSIM: {base_ssim:.4f} | LPIPS: {base_lpips:.4f}")
    print("-" * 65)

    # Logging
    train_logger = CSVLogger(
        os.path.join(exp_dir, "train.csv"),
        ["epoch", "lr", "train_loss", "recon_loss", "ssim_loss", "edge_loss", "epoch_time"],
    )
    val_logger = CSVLogger(
        os.path.join(exp_dir, "validation.csv"),
        ["epoch", "val_loss", "val_psnr", "val_ssim", "val_lpips", "ema_psnr", "ema_ssim", "ema_lpips"],
    )

    # Resume if requested
    start_epoch = 1
    best_psnr = -1.0
    best_ssim = -1.0
    best_overall_score = -1.0

    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming training from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("val_psnr", -1.0)
        best_ssim = ckpt.get("val_ssim", -1.0)

    # Training Loop
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    grad_accum_steps = cfg["training"].get("grad_accum_steps", 1)

    print("\nStarting Training Loop...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        start_t = time.time()
        epoch_losses = []
        epoch_recon = []
        epoch_ssim = []
        epoch_edge = []

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for step, batch in enumerate(pbar):
            lr_imgs = batch["lr"].to(device)
            gt_imgs = batch["gt"].to(device)

            with autocast(enabled=use_amp):
                pred = model(lr_imgs)
                loss, loss_dict = criterion(pred, gt_imgs)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            epoch_losses.append(loss.item())
            epoch_recon.append(loss_dict.get("loss_recon", 0.0))
            epoch_ssim.append(loss_dict.get("loss_ssim", 0.0))
            epoch_edge.append(loss_dict.get("loss_edge", 0.0))

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{get_lr(optimizer):.6f}",
            })

        scheduler.step()
        epoch_time = time.time() - start_t

        mean_train_loss = float(np.mean(epoch_losses))
        mean_recon = float(np.mean(epoch_recon))
        mean_ssim_l = float(np.mean(epoch_ssim))
        mean_edge = float(np.mean(epoch_edge))

        train_logger.log({
            "epoch": epoch,
            "lr": get_lr(optimizer),
            "train_loss": mean_train_loss,
            "recon_loss": mean_recon,
            "ssim_loss": mean_ssim_l,
            "edge_loss": mean_edge,
            "epoch_time": epoch_time,
        })

        # Validation
        if epoch % cfg["training"].get("eval_freq", 1) == 0:
            val_res = validate(model, val_loader, criterion, device, lpips_fn)
            ema_res = validate(ema.ema_model, val_loader, criterion, device, lpips_fn)

            val_logger.log({
                "epoch": epoch,
                "val_loss": val_res["val_loss"],
                "val_psnr": val_res["val_psnr"],
                "val_ssim": val_res["val_ssim"],
                "val_lpips": val_res["val_lpips"],
                "ema_psnr": ema_res["val_psnr"],
                "ema_ssim": ema_res["val_ssim"],
                "ema_lpips": ema_res["val_lpips"],
            })

            # Print comparison
            print(
                f"[Epoch {epoch:03d}] Train Loss: {mean_train_loss:.4f} | "
                f"Model Val PSNR: {val_res['val_psnr']:.4f} dB, SSIM: {val_res['val_ssim']:.4f} | "
                f"EMA Val PSNR: {ema_res['val_psnr']:.4f} dB, SSIM: {ema_res['val_ssim']:.4f}"
            )

            # Determine best model between active and EMA
            best_curr_psnr = max(val_res["val_psnr"], ema_res["val_psnr"])
            best_curr_ssim = max(val_res["val_ssim"], ema_res["val_ssim"])
            overall_score = best_curr_psnr + 20.0 * best_curr_ssim  # Composite quality index

            # Checkpoint payload
            save_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "val_psnr": val_res["val_psnr"],
                "val_ssim": val_res["val_ssim"],
                "ema_psnr": ema_res["val_psnr"],
                "ema_ssim": ema_res["val_ssim"],
                "config": cfg,
                "seed": seed,
            }

            # Save latest
            torch.save(save_payload, os.path.join(save_dir, "latest.pth"))
            torch.save(save_payload, os.path.join(exp_dir, "latest.pth"))

            # Save best PSNR
            if best_curr_psnr > best_psnr:
                best_psnr = best_curr_psnr
                torch.save(save_payload, os.path.join(save_dir, "best_psnr.pth"))
                torch.save(save_payload, os.path.join(exp_dir, "best_psnr.pth"))
                print(f"  --> Saved NEW Best PSNR Checkpoint: {best_psnr:.4f} dB")

            # Save best SSIM
            if best_curr_ssim > best_ssim:
                best_ssim = best_curr_ssim
                torch.save(save_payload, os.path.join(save_dir, "best_ssim.pth"))
                torch.save(save_payload, os.path.join(exp_dir, "best_ssim.pth"))
                print(f"  --> Saved NEW Best SSIM Checkpoint: {best_ssim:.4f}")

            # Save best overall
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                torch.save(save_payload, os.path.join(save_dir, "best_overall.pth"))
                torch.save(save_payload, os.path.join(exp_dir, "best.pth"))

    # Summary report
    summary = {
        "timestamp": timestamp,
        "epochs_completed": epochs,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "baseline_psnr": base_psnr,
        "baseline_ssim": base_ssim,
        "psnr_improvement_db": best_psnr - base_psnr,
        "ssim_improvement": best_ssim - base_ssim,
        "model_parameters": total_params,
        "config": cfg,
    }
    save_summary_json(os.path.join(exp_dir, "summary.json"), summary)
    print("\n" + "=" * 65)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Baseline PSNR:      {base_psnr:.4f} dB | SSIM: {base_ssim:.4f}")
    print(f"Best Restored PSNR: {best_psnr:.4f} dB | SSIM: {best_ssim:.4f}")
    print(f"Improvement:        +{best_psnr - base_psnr:.4f} dB PSNR, +{best_ssim - base_ssim:.4f} SSIM")
    print(f"Artifacts Saved to: {exp_dir} and {save_dir}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastNAF-SR V5")
    parser.add_argument("--config", default="configs/fastnaf_sr.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--train_root", default=None, help="Path to train data directory containing NoisyLR and GT")
    parser.add_argument("--lr_dir", default=None, help="Direct path to NoisyLR folder")
    parser.add_argument("--gt_dir", default=None, help="Direct path to GT folder")
    args = parser.parse_args()

    train_pipeline(
        config_path=args.config,
        resume_ckpt=args.resume,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_train_root=args.train_root,
        override_lr_dir=args.lr_dir,
        override_gt_dir=args.gt_dir,
    )
