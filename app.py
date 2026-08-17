"""
FastNAF-SR V5 — Interactive Semiconductor Image Restoration Dashboard.
Built with Streamlit for visual inspection, quality metrics, and performance analysis.
"""

import os
import sys
import time
import numpy as np
import torch
import streamlit as st
from PIL import Image

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.fastnaf_sr import FastNAFSR_V5
from datasets.preprocessing import normalize, denormalize, DATASET_MEAN, DATASET_STD
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from utils.tiled_inference import tiled_inference
from evaluate import find_default_checkpoint


def load_model(checkpoint_path=None, device="cpu"):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_default_checkpoint()
    
    model = FastNAFSR_V5(
        in_channels=1,
        out_channels=1,
        channels=48,
        num_lr_blocks=8,
        num_hr_blocks=4,
        upscale_factor=2,
    ).to(device)

    ckpt_info = "Default Initialized Weights"
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "ema_state" in ckpt and ckpt["ema_state"] is not None:
            model.load_state_dict(ckpt["ema_state"])
            ckpt_info = f"Loaded EMA weights ({checkpoint_path})"
        elif "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            ckpt_info = f"Loaded Model weights ({checkpoint_path})"
        else:
            model.load_state_dict(ckpt)
            ckpt_info = f"Loaded raw checkpoint ({checkpoint_path})"

    model.eval()
    return model, ckpt_info


def array_to_display_image(arr):
    """Converts a float array to uint8 image for display without changing underlying data."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(arr)
    return Image.fromarray((norm * 255).astype(np.uint8))


def main():
    st.set_page_config(
        page_title="KLA Semiconductor AI Image Restoration",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 FastNAF-SR: Semiconductor AI Image Restoration")
    st.markdown(
        "**Grayscale Inspection Image Restoration & 2× Super-Resolution** "
        "(Speckle Noise + Gaussian Noise Removal + 2× SR: 128×128 → 256×256)"
    )

    # Sidebar
    st.sidebar.header("Configuration")
    device_choice = st.sidebar.selectbox("Compute Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    ckpt_path = st.sidebar.text_input("Checkpoint Path", value="checkpoints/best_overall.pth")
    if not os.path.exists(ckpt_path):
        default_found = find_default_checkpoint()
        if default_found:
            ckpt_path = default_found

    model, ckpt_info = load_model(ckpt_path, device=device_choice)
    total_params = sum(p.numel() for p in model.parameters())

    st.sidebar.success(f"Status: {ckpt_info}")
    st.sidebar.info(
        f"**Architecture Info:**\n"
        f"- Model: FastNAF-SR\n"
        f"- Channels: 48\n"
        f"- LR NAF Blocks: 8\n"
        f"- HR NAF Blocks: 4\n"
        f"- Upsampling: 2× PixelShuffle\n"
        f"- Parameters: {total_params:,} (~{total_params/1e6:.2f}M)\n"
        f"- Norm Mean: {DATASET_MEAN:.4f}\n"
        f"- Norm Std: {DATASET_STD:.4f}"
    )

    # Main UI Tabs
    tab1, tab2 = st.tabs(["Single Image Restoration", "Dataset Inspection"])

    with tab1:
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_lr = st.file_uploader("Upload NoisyLR Image (.npy)", type=["npy"])
        with col_up2:
            uploaded_gt = st.file_uploader("Optional: Upload CleanGT Image (.npy)", type=["npy"])

        if uploaded_lr is not None:
            lr_arr = np.load(uploaded_lr).astype(np.float32)
            gt_arr = np.load(uploaded_gt).astype(np.float32) if uploaded_gt is not None else None

            # Normalization & Inference
            norm_lr = normalize(lr_arr)
            tensor_lr = torch.from_numpy(norm_lr).unsqueeze(0).unsqueeze(0).float().to(device_choice)

            t0 = time.perf_counter()
            with torch.no_grad():
                if lr_arr.shape == (128, 128):
                    pred_tensor = model(tensor_lr)
                else:
                    pred_tensor = tiled_inference(model, tensor_lr, tile_size=128, tile_overlap=16, scale=2, device=device_choice)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            pred_arr = denormalize(pred_tensor).squeeze().cpu().numpy().astype(np.float32)

            # Display
            st.subheader("Restoration Results")
            if gt_arr is not None:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**NoisyLR Input** ({lr_arr.shape[0]}×{lr_arr.shape[1]})")
                    st.image(array_to_display_image(lr_arr), use_container_width=True)
                    st.caption(f"Min: {lr_arr.min():.4f} | Max: {lr_arr.max():.4f} | Mean: {lr_arr.mean():.4f}")
                with c2:
                    st.markdown(f"**FastNAF-SR Output** ({pred_arr.shape[0]}×{pred_arr.shape[1]})")
                    st.image(array_to_display_image(pred_arr), use_container_width=True)
                    st.caption(f"Min: {pred_arr.min():.4f} | Max: {pred_arr.max():.4f} | Mean: {pred_arr.mean():.4f}")
                with c3:
                    st.markdown(f"**CleanGT Ground Truth** ({gt_arr.shape[0]}×{gt_arr.shape[1]})")
                    st.image(array_to_display_image(gt_arr), use_container_width=True)
                    st.caption(f"Min: {gt_arr.min():.4f} | Max: {gt_arr.max():.4f} | Mean: {gt_arr.mean():.4f}")

                psnr_val = calculate_psnr(pred_arr, gt_arr, data_range=1.0)
                ssim_val = calculate_ssim(pred_arr, gt_arr, data_range=1.0)
                m1, m2, m3 = st.columns(3)
                m1.metric("PSNR (dB)", f"{psnr_val:.4f} dB")
                m2.metric("SSIM", f"{ssim_val:.4f}")
                m3.metric("Latency", f"{latency_ms:.2f} ms")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**NoisyLR Input** ({lr_arr.shape[0]}×{lr_arr.shape[1]})")
                    st.image(array_to_display_image(lr_arr), use_container_width=True)
                    st.caption(f"Min: {lr_arr.min():.4f} | Max: {lr_arr.max():.4f} | Mean: {lr_arr.mean():.4f}")
                with c2:
                    st.markdown(f"**FastNAF-SR Output** ({pred_arr.shape[0]}×{pred_arr.shape[1]})")
                    st.image(array_to_display_image(pred_arr), use_container_width=True)
                    st.caption(f"Min: {pred_arr.min():.4f} | Max: {pred_arr.max():.4f} | Mean: {pred_arr.mean():.4f}")

                st.info(f"⚡ Inference Latency: **{latency_ms:.2f} ms** on {device_choice.upper()}")

    with tab2:
        st.subheader("Dataset Report & Statistics")
        st.markdown(
            """
            - **Input Resolution:** 128 × 128 (1 channel float32)
            - **Ground Truth Resolution:** 256 × 256 (1 channel float32)
            - **Scale Factor:** 2× Spatial Super-Resolution
            - **Noise Profile:** Mixed Speckle & Gaussian Semiconductor Degradation
            - **Target Domain Statistics:** Mean ≈ 0.4335, Std ≈ 0.2848
            """
        )


if __name__ == "__main__":
    main()
