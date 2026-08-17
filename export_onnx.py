"""
ONNX Export and Numerical Verification Suite for FastNAF-SR V5.

Features:
- Dynamic batch and spatial dimensions
- Verification against PyTorch output (Maximum Absolute Difference)
- Optional FP16 export
"""

import os
import sys
import argparse
import numpy as np
import torch

from models.fastnaf_sr import FastNAFSR_V5
from evaluate import find_default_checkpoint


def export_to_onnx(
    checkpoint_path=None,
    output_onnx="fastnaf_sr.onnx",
    opset_version=14,
    device="cpu",
):
    print("=" * 65)
    print("FASTNAF-SR V5 ONNX EXPORT")
    print("=" * 65)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_default_checkpoint()

    model = FastNAFSR_V5(in_channels=1, out_channels=1, channels=48, num_lr_blocks=8, num_hr_blocks=4).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "ema_state" in ckpt and ckpt["ema_state"] is not None:
            model.load_state_dict(ckpt["ema_state"])
            print("Loaded EMA weights into model.")
        elif "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print("Loaded model state weights.")
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()

    dummy_input = torch.randn(1, 1, 128, 128, device=device)

    print(f"Exporting ONNX model to {output_onnx} (Opset {opset_version})...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height_2x", 3: "width_2x"},
            },
        )
        print(f"Exported successfully! File size: {os.path.getsize(output_onnx) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"ONNX export encountered issue: {e}")
        print("Retrying with standard torch.jit trace export...")
        traced = torch.jit.trace(model, dummy_input)
        traced_path = output_onnx.replace(".onnx", ".pt")
        traced.save(traced_path)
        print(f"Saved TorchScript traced model to {traced_path} ({os.path.getsize(traced_path) / (1024 * 1024):.2f} MB)")

    # Verify with onnx / onnxruntime if installed
    try:
        import onnx
        onnx_model = onnx.load(output_onnx)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure checked: VALID.")
    except ImportError:
        print("[Note] 'onnx' library not installed. Skipping structural validation.")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_onnx, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outs = session.run(None, ort_inputs)

        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        max_diff = float(np.max(np.abs(torch_out - ort_outs[0])))
        print(f"ONNXRuntime numerical comparison: Max absolute difference = {max_diff:.6e}")
        if max_diff < 1e-4:
            print("[SUCCESS] ONNX outputs match PyTorch within numerical tolerance!")
        else:
            print(f"[WARNING] Discrepancy observed: {max_diff}")
    except ImportError:
        print("[Note] 'onnxruntime' not installed. Skipping runtime numerical validation.")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FastNAF-SR V5 to ONNX")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--output", default="fastnaf_sr.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(checkpoint_path=args.checkpoint, output_onnx=args.output, opset_version=args.opset)
