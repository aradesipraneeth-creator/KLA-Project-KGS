# Clean Installation and Zero-Dependency Reproduction Guide

This guide describes how to replicate and run the **FastNAF-SR** image restoration pipeline from a completely clean machine with zero hard-coded paths.

---

## 1. Environment Setup

Create and activate a fresh Python virtual environment (Python 3.9+ recommended):

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

Install the core production requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If GPU acceleration is available, install the matching PyTorch CUDA build:

```bash
# Example: PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Dataset Structure

Ensure the evaluation or training dataset is placed in the workspace root:

```text
Project KLA KSG/
│
├── train/
│   └── train/
│       ├── NoisyLR/    # [128×128 float32 .npy]
│       └── GT/         # [256×256 float32 .npy]
│
└── Test_NoisyLR/
    └── NoisyLR/        # [128×128 float32 .npy]
```

---

## 4. Run Dataset Validation & Sanity Check

Verify dataset integrity, 2× spatial dimensions, and end-to-end tensor flow:

```bash
python scripts/validate_dataset.py
python scripts/sanity_check.py
```

---

## 5. Official Submission Evaluation

To run standalone evaluation on the test set:

```bash
python evaluate.py \
    --input_dir Test_NoisyLR/NoisyLR \
    --output_dir outputs \
    --checkpoint checkpoints/best_overall.pth
```

All 400 test images (`000000.npy` to `000399.npy`) will be restored from 1×128×128 to 1×256×256 and saved in `outputs/` preserving exact filenames.

---

## 6. Benchmarking Throughput & Latency

Run the production benchmark suite (H100 / CUDA / CPU):

```bash
python benchmark.py --checkpoint checkpoints/best_overall.pth --csv benchmark_results.csv
```

---

## 7. Interactive Streamlit Dashboard

Launch the visual inspection interface:

```bash
streamlit run app.py
```
