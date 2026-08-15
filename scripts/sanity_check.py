"""
Automated Sanity Check & Single-Batch Test for FastNAF-SR V5 Pipeline.

Verifies:
1. Dataset discovery & valid pair alignment
2. Test image discovery
3. Model instantiation & forward pass shape ([B, 1, 128, 128] -> [B, 1, 256, 256])
4. No NaNs or Infs in activations
5. RestorationLoss computation
6. Backward pass & gradient flow
7. Optimizer step & EMA update
8. Checkpoint saving & loading
9. Standalone tiled inference test
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.kla_dataset import KLAPairedDataset, KLATestDataset, get_clean_npy_filelist
from datasets.preprocessing import normalize, denormalize
from models.fastnaf_sr import FastNAFSR_V5
from losses.combined_loss import RestorationLoss
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from utils.ema import ModelEMA
from utils.tiled_inference import tiled_inference


def run_sanity_checks():
    print("=" * 65)
    print("RUNNING AUTOMATED SANITY CHECKS")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/9] Active Compute Device: {device}")

    # 1. Dataset discovery
    train_lr_dir = "train/train/NoisyLR"
    train_gt_dir = "train/train/GT"
    test_lr_dir = "Test_NoisyLR/NoisyLR"

    lr_files = get_clean_npy_filelist(train_lr_dir)
    gt_files = get_clean_npy_filelist(train_gt_dir)
    test_files = get_clean_npy_filelist(test_lr_dir)

    assert len(lr_files) > 0, "No train LR files found!"
    assert len(gt_files) > 0, "No train GT files found!"
    assert len(test_files) > 0, "No test files found!"
    print(f"[2/9] Discovered {len(lr_files)} train LR, {len(gt_files)} train GT, {len(test_files)} test files. (PASS)")

    # 2. Dataset & DataLoader verification
    dataset = KLAPairedDataset(train_lr_dir, train_gt_dir, file_list=lr_files[:16], is_train=True, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    lr = batch["lr"].to(device)
    gt = batch["gt"].to(device)

    assert lr.shape == (4, 1, 128, 128), f"Unexpected LR shape: {lr.shape}"
    assert gt.shape == (4, 1, 256, 256), f"Unexpected GT shape: {gt.shape}"
    assert not torch.isnan(lr).any(), "NaN found in LR input!"
    assert not torch.isnan(gt).any(), "NaN found in GT target!"
    print(f"[3/9] DataLoader batch shapes: LR={list(lr.shape)}, GT={list(gt.shape)}. (PASS)")

    # 3. Model instantiation & Forward pass
    model = FastNAFSR_V5(in_channels=1, out_channels=1, channels=48, num_lr_blocks=8, num_hr_blocks=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[4/9] Model FastNAF-SR V5 created. Parameter count: {total_params:,}. (PASS)")

    pred = model(lr)
    assert pred.shape == (4, 1, 256, 256), f"Expected [4, 1, 256, 256], got {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN found in model output!"
    assert not torch.isinf(pred).any(), "Inf found in model output!"
    print(f"[5/9] Forward pass output shape: {list(pred.shape)}, No NaNs/Infs. (PASS)")

    # 4. Loss computation
    criterion = RestorationLoss(recon_type="charbonnier", lambda_recon=1.0, lambda_ssim=0.5, lambda_edge=0.05).to(device)
    loss, loss_dict = criterion(pred, gt)
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss is non-positive!"
    print(f"[6/9] Loss computed: Total={loss.item():.4f}, Recon={loss_dict['loss_recon']:.4f}, SSIM={loss_dict['loss_ssim']:.4f}, Edge={loss_dict['loss_edge']:.4f}. (PASS)")

    # 5. Backward pass & Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()

    # Verify gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
            if param.grad.abs().sum() > 0:
                has_grad = True
    assert has_grad, "No gradients flowed through the model!"

    optimizer.step()
    print("[7/9] Backward pass & gradient step completed successfully. (PASS)")

    # 6. EMA update & Checkpointing
    ema = ModelEMA(model, decay=0.999)
    ema.update(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
        torch.save(
            {
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": 1,
                "val_psnr": 30.5,
            },
            ckpt_path,
        )
        assert os.path.exists(ckpt_path), "Checkpoint file was not created!"

        # Load back
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        print("[8/9] Checkpoint save & load verified. (PASS)")

    # 7. Tiled inference test
    sample_img = torch.randn(1, 1, 128, 128).to(device)
    tiled_out = tiled_inference(model, sample_img, tile_size=64, tile_overlap=16, scale=2, device=device)
    assert tiled_out.shape == (1, 1, 256, 256), f"Unexpected tiled output shape: {tiled_out.shape}"
    print(f"[9/9] Tiled inference verified: Input [1, 1, 128, 128] (tile_size=64) -> Output {list(tiled_out.shape)}. (PASS)", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("ALL SANITY CHECKS PASSED PERFECTLY!", flush=True)
    print("=" * 65, flush=True)
    return True


if __name__ == "__main__":
    success = run_sanity_checks()
    sys.exit(0 if success else 1)
