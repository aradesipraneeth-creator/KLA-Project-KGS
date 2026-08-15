# FastNAF-SR V5 Inference Examples

This guide provides practical examples for running inference using the trained **FastNAF-SR V5** model.

---

## 1. Quick Python API Inference

You can load and run the model in Python using standard PyTorch:

```python
import numpy as np
import torch
from models.fastnaf_sr import FastNAFSR_V5
from datasets.preprocessing import normalize, denormalize

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Instantiate model & load checkpoint
model = FastNAFSR_V5(
    in_channels=1,
    out_channels=1,
    channels=48,
    num_lr_blocks=8,
    num_hr_blocks=4,
    upscale_factor=2,
).to(device)

checkpoint = torch.load("checkpoints/best_overall.pth", map_location=device)
if "ema_state" in checkpoint and checkpoint["ema_state"] is not None:
    model.load_state_dict(checkpoint["ema_state"])
else:
    model.load_state_dict(checkpoint["model_state"])
model.eval()

# 3. Load single raw .npy input (128x128 float32 array)
raw_input = np.load("path/to/input.npy").astype(np.float32)

# 4. Preprocess / Normalize
norm_input = normalize(raw_input)
tensor_in = torch.from_numpy(norm_input).unsqueeze(0).unsqueeze(0).to(device)

# 5. Forward inference
with torch.no_grad():
    tensor_out = model(tensor_in)

# 6. Denormalize to float32 raw range
restored_image = denormalize(tensor_out).squeeze().cpu().numpy().astype(np.float32)

# 7. Save output
np.save("restored_output.npy", restored_image)
print(f"Restored output shape: {restored_image.shape}")  # (256, 256)
```

---

## 2. Command Line CLI Inference

### Single Image Inference

Use `infer.py` to restore a single `.npy` image file:

```bash
python infer.py \
    --input sample_lr.npy \
    --output restored_hr.npy \
    --checkpoint checkpoints/best_overall.pth
```

### Full Folder Batch Evaluation

Use `evaluate.py` to evaluate an entire folder of test images:

```bash
python evaluate.py \
    --input_dir ./Test_NoisyLR/NoisyLR \
    --output_dir ./test_outputs \
    --checkpoint checkpoints/best_overall.pth \
    --batch_size 16
```

If ground truth images are available for quantitative benchmarking:

```bash
python evaluate.py \
    --input_dir ./train/train/NoisyLR \
    --output_dir ./val_outputs \
    --gt_dir ./train/train/GT \
    --checkpoint checkpoints/best_overall.pth
```

---

## 3. Large Image Tiled Inference

For arbitrary large images exceeding standard dimensions (e.g. 512×512, 1024×1024), seamless tiled inference with Hann window blending is supported out of the box:

```python
from utils.tiled_inference import tiled_inference

with torch.no_grad():
    tensor_out = tiled_inference(
        model=model,
        tensor=tensor_in,
        tile_size=128,
        tile_overlap=16,
        scale=2,
        device=device,
    )
```
