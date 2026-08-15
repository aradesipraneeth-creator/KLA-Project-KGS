# KLA Semiconductor AI Image Restoration — FastNAF-SR

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, robust, and mathematically grounded deep learning system for restoring degraded semiconductor inspection images using the **FastNAF-SR V5** architecture.

---

## 1. Problem Overview

Semiconductor wafer inspection images captured during electron beam or optical inspection suffer from severe composite degradations:
- **Speckle Noise**: Multiplicative noise caused by coherent laser/beam interference, generating signal-dependent intensity fluctuations that exceed standard $[0, 1]$ bounds.
- **Gaussian Noise**: Thermal and electronic sensor noise inherent to high-throughput nanoscale imaging.
- **Spatial Resolution Reduction**: Optical diffraction limits requiring a $2\times$ spatial super-resolution mapping ($128 \times 128 \to 256 \times 256$).
- **Simultaneous Degradation**: Non-linear interaction of noise and optical blur occurring simultaneously.
- **Out-of-Distribution (OOD) Generalization**: The network must generalize reliably to unseen wafer patterns, varied pitch lines, and unknown defect geometries without hallucinating non-existent features.
- **Inference-Speed Requirement**: Inspection pipelines demand real-time throughput ($< 10$ ms latency) to keep pace with fab fabrication rates.

---

## 2. Architecture: FastNAF-SR V5

FastNAF-SR V5 utilizes Nonlinear Activation Free (NAF) computational blocks, eliminating expensive non-linear activations in favor of simple channel multiplications (`SimpleGate`) and spatial depthwise convolutions for maximal inference throughput.

```text
Input 1×128×128
       ↓
3×3 Conv 1→48
       ↓
NAF ×8
       ↓
Residual Feature Fusion
       ↓
3×3 Conv 48→192
       ↓
PixelShuffle ×2
       ↓
NAF ×4
       ↓
3×3 Conv 48→1
       ↓
Output 1×256×256
```

### Architectural Key Characteristics
- **Shallow Feature Extraction**: 3×3 Conv mapping 1 grayscale channel to 48 intermediate feature channels.
- **Low-Resolution Restoration Trunk (8 NAF Blocks)**: Denoising and deblurring performed at the lower $128 \times 128$ resolution to keep compute complexity $O(HW)$ low.
- **Residual Feature Fusion**: Skip connection combining shallow features with restored low-resolution features to preserve fine low-frequency structures.
- **Sub-Pixel Convolution**: 3×3 Conv ($48 \to 192$) followed by `PixelShuffle(2)` to upsample features to $256 \times 256$.
- **High-Resolution Refinement Trunk (4 NAF Blocks)**: Refines edges and eliminates artifacts at the full $256 \times 256$ resolution.
- **Output Head**: Final 3×3 Conv projecting 48 channels back to 1 grayscale channel.

---

## 3. Dataset Setup

> **IMPORTANT**: The raw KLA proprietary dataset is **NOT included** in this public repository.

To run training or evaluation on your local machine, place your locally provided dataset in the project root as follows:

```text
Project KLA KSG/
├── train/
│   └── train/
│       ├── NoisyLR/    # 3,200 degraded float32 .npy files (128×128)
│       └── GT/         # 3,200 clean ground truth float32 .npy files (256×256)
└── Test_NoisyLR/
    └── NoisyLR/        # 400 test degraded float32 .npy files (128×128)
```

### Data Preprocessing & Non-Clipping Protocol
- **Dynamic Range Preservation**: Input pixel values naturally range outside $[0, 1]$ (min $\approx -0.003$, max $\approx 1.54$). We do **not** apply naive clipping, preserving true sensor physical values.
- **Global Z-Score Normalization**:
  $$\mu = 0.43353602, \quad \sigma = 0.28478748$$
  $$z = \frac{x - \mu}{\sigma}, \quad x = z \cdot \sigma + \mu$$
- **Metadata Filtering**: Automatic exclusion of macOS metadata (`__MACOSX`, `._*`) across all dataset and evaluation loaders.

---

## 4. Installation

```bash
# Clone the repository
git clone https://github.com/aradesipraneeth-creator/KLA-Project-KGS.git
cd KLA-Project-KGS

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Validation

Run the automated dataset validation script to verify file pairs, data shapes, dynamic ranges, and ensure zero corrupted files:

```bash
python scripts/validate_dataset.py
```

---

## 6. Pipeline Sanity Check

Run automated end-to-end unit and integration tests (forward pass, backward pass, loss computation, EMA weight updates, checkpoint saving/loading, and tiled inference):

```bash
python scripts/sanity_check.py
```

---

## 7. Model Training

Train the FastNAF-SR V5 network using the default configuration (AMP FP16, AdamW optimizer, Cosine Annealing scheduler, Model EMA, and multi-component Charbonnier + SSIM + Edge loss):

```bash
python train.py --config configs/fastnaf_sr.yaml
```

---

## 8. Evaluation & Inference (Reviewer CLI)

`evaluate.py` is the primary evaluation and inference entry point. It is completely standalone and accepts any arbitrary input directory containing test `.npy` files:

```bash
python evaluate.py \
    --input_dir ./Test_NoisyLR/NoisyLR \
    --output_dir ./test_outputs
```

### Full Argument Options

```bash
python evaluate.py --help
```

| Argument | Description | Default |
|:---|:---|:---|
| `--input_dir` | Path to input directory containing `.npy` images | **Required** |
| `--output_dir` | Path to destination directory to save restored `.npy` files | **Required** |
| `--checkpoint` | Path to specific `.pth` model checkpoint | Auto-resolves (`checkpoints/best_overall.pth`) |
| `--gt_dir` | Optional path to ground truth directory for metric calculation | `None` |
| `--batch_size` | Inference batch size for uniform $128 \times 128$ images | `16` |
| `--tile_size` | Tile size for large non-standard input images | `128` |
| `--tile_overlap` | Pixel overlap between adjacent tiles | `16` |
| `--device` | Compute device (`cuda` or `cpu`) | Auto-detect |
| `--no_ema` | Disable EMA weights and use raw model weights | `False` |

---

## 9. Benchmark

Measure model latency, peak GPU VRAM, parameter counts, and FPS throughput across FP32 and FP16:

```bash
python benchmark.py
```

---

## 10. ONNX Model Export

Export the trained model to ONNX format with dynamic spatial and batch axes, and numerically verify outputs against PyTorch:

```bash
python export_onnx.py --output fastnaf_sr.onnx
```

---

## 11. Model Checkpoints & Weights

The FastNAF-SR V5 model is lightweight (~378K parameters, ~2.5 MB). The best pre-trained checkpoint is provided directly in the repository at:

```text
checkpoints/best_overall.pth
```

When running `evaluate.py` or `infer.py`, the scripts will automatically find and load `checkpoints/best_overall.pth` without requiring manual path specification. You can also supply a custom checkpoint via `--checkpoint <path>`.

---

## 12. Repository Structure

```text
KLA-Project-KGS/
│
├── README.md                 # Comprehensive project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusions
├── LICENSE                   # MIT License
│
├── train.py                  # Full training pipeline with AMP, EMA & Cosine LR
├── evaluate.py               # Standalone evaluation & batch restoration script
├── infer.py                  # Single-image quick inference CLI
├── benchmark.py              # Latency, FPS, VRAM & parameter benchmarking suite
├── export_onnx.py            # Dynamic ONNX export & verification
│
├── models/
│   ├── __init__.py           # Model exports
│   ├── fastnaf_sr.py         # FastNAF-SR V5 architecture & build_model()
│   └── naf_block.py          # NAFBlock, LayerNorm2d, SimpleGate, SCA
│
├── datasets/
│   ├── __init__.py           # Dataset exports
│   ├── kla_dataset.py        # KLAPairedDataset & KLATestDataset
│   └── preprocessing.py      # Non-clipping global normalization functions
│
├── losses/
│   ├── __init__.py           # Loss module exports
│   ├── charbonnier.py        # Charbonnier robust reconstruction loss
│   ├── ssim.py               # Differentiable SSIM loss
│   ├── edge.py               # Sobel edge-aware preservation loss
│   └── combined_loss.py      # Composite RestorationLoss
│
├── metrics/
│   ├── __init__.py           # Metric exports
│   ├── psnr.py               # Peak Signal-to-Noise Ratio (PSNR)
│   ├── ssim.py               # Structural Similarity Index (SSIM)
│   └── lpips_metric.py       # Learned Perceptual Patch Similarity
│
├── configs/
│   └── fastnaf_sr.yaml       # Hyperparameters and model configuration
│
├── scripts/
│   ├── validate_dataset.py   # Dataset validation and integrity check
│   ├── sanity_check.py       # Automated forward/backward pipeline test
│   └── generate_test_outputs.py # Batch test output and manifest generator
│
├── docs/
│   └── architecture.md       # In-depth architectural design specification
│
├── examples/
│   └── inference_example.md  # Python and CLI inference walkthroughs
│
└── utils/
    ├── __init__.py           # Utility exports
    ├── ema.py                # Exponential Moving Average (EMA) helper
    ├── logger.py             # Logging and metric tracking
    └── tiled_inference.py    # Overlapped windowed tiled inference
```

---

## 13. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
