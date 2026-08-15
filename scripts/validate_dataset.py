"""
Dataset Validation Script for KLA Semiconductor AI Image Restoration.

Checks:
- File counts & naming alignment
- Ignores __MACOSX and ._* metadata files
- Checks for corrupt, NaN, Inf, missing pairs
- Verifies shapes and 2x spatial relationship (LR 128x128 -> GT 256x256)
- Calculates and verifies dataset min, max, mean, std
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm


def find_valid_npy_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    valid_files = []
    for root, dirs, files in os.walk(folder_path):
        # Ignore __MACOSX directories
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.endswith(".npy") and not f.startswith("._"):
                valid_files.append(f)
    return sorted(list(set(valid_files)))


def validate_dataset(train_root="train/train", test_root="Test_NoisyLR/NoisyLR", max_inspect=None):
    train_lr_dir = os.path.join(train_root, "NoisyLR")
    train_gt_dir = os.path.join(train_root, "GT")
    test_lr_dir = test_root

    print("=" * 60)
    print("KLA Semiconductor Dataset Validation")
    print("=" * 60)

    lr_files = find_valid_npy_files(train_lr_dir)
    gt_files = find_valid_npy_files(train_gt_dir)
    test_files = find_valid_npy_files(test_lr_dir)

    lr_set = set(lr_files)
    gt_set = set(gt_files)

    common_pairs = sorted(list(lr_set.intersection(gt_set)))
    missing_gt = sorted(list(lr_set - gt_set))
    missing_lr = sorted(list(gt_set - lr_set))

    print(f"Discovered Train LR files: {len(lr_files)}")
    print(f"Discovered Train GT files: {len(gt_files)}")
    print(f"Paired Train samples:     {len(common_pairs)}")
    print(f"Missing GT for LR:        {len(missing_gt)}")
    print(f"Missing LR for GT:        {len(missing_lr)}")
    print(f"Discovered Test LR files:  {len(test_files)}")
    print("-" * 60)

    # Inspect paired files
    valid_2x_pairs = 0
    invalid_pairs = 0
    nan_count = 0
    inf_count = 0
    corrupt_count = 0

    lr_shapes = set()
    gt_shapes = set()
    lr_dtypes = set()
    gt_dtypes = set()

    all_lr_means = []
    all_lr_stds = []
    all_gt_means = []
    all_gt_stds = []

    global_lr_min = float("inf")
    global_lr_max = float("-inf")
    global_gt_min = float("inf")
    global_gt_max = float("-inf")

    inspect_list = common_pairs if max_inspect is None else common_pairs[:max_inspect]

    print(f"Inspecting {len(inspect_list)} paired samples...")
    for filename in tqdm(inspect_list, desc="Validating train pairs"):
        lr_path = os.path.join(train_lr_dir, filename)
        gt_path = os.path.join(train_gt_dir, filename)

        try:
            lr_arr = np.load(lr_path)
            gt_arr = np.load(gt_path)
        except Exception as e:
            corrupt_count += 1
            continue

        lr_shapes.add(lr_arr.shape)
        gt_shapes.add(gt_arr.shape)
        lr_dtypes.add(str(lr_arr.dtype))
        gt_dtypes.add(str(gt_arr.dtype))

        # Check NaN / Inf
        if np.isnan(lr_arr).any() or np.isnan(gt_arr).any():
            nan_count += 1
        if np.isinf(lr_arr).any() or np.isinf(gt_arr).any():
            inf_count += 1

        # Check 2x relationship
        if (
            len(lr_arr.shape) == 2
            and len(gt_arr.shape) == 2
            and gt_arr.shape[0] == 2 * lr_arr.shape[0]
            and gt_arr.shape[1] == 2 * lr_arr.shape[1]
        ):
            valid_2x_pairs += 1
        else:
            invalid_pairs += 1

        global_lr_min = min(global_lr_min, float(lr_arr.min()))
        global_lr_max = max(global_lr_max, float(lr_arr.max()))
        global_gt_min = min(global_gt_min, float(gt_arr.min()))
        global_gt_max = max(global_gt_max, float(gt_arr.max()))

        all_lr_means.append(float(lr_arr.mean()))
        all_lr_stds.append(float(lr_arr.std()))
        all_gt_means.append(float(gt_arr.mean()))
        all_gt_stds.append(float(gt_arr.std()))

    # Inspect test files
    test_shapes = set()
    test_dtypes = set()
    test_nan = 0
    test_inf = 0
    global_test_min = float("inf")
    global_test_max = float("-inf")
    all_test_means = []

    print(f"Inspecting {len(test_files)} test samples...")
    for filename in tqdm(test_files, desc="Validating test set"):
        test_path = os.path.join(test_lr_dir, filename)
        try:
            arr = np.load(test_path)
            test_shapes.add(arr.shape)
            test_dtypes.add(str(arr.dtype))
            if np.isnan(arr).any():
                test_nan += 1
            if np.isinf(arr).any():
                test_inf += 1
            global_test_min = min(global_test_min, float(arr.min()))
            global_test_max = max(global_test_max, float(arr.max()))
            all_test_means.append(float(arr.mean()))
        except Exception:
            corrupt_count += 1

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY REPORT")
    print("=" * 60)
    print(f"Training pairs checked:  {len(inspect_list)}")
    print(f"Valid 2x pairs:          {valid_2x_pairs}")
    print(f"Invalid pairs:           {invalid_pairs}")
    print(f"Missing GT files:        {len(missing_gt)}")
    print(f"Missing LR files:        {len(missing_lr)}")
    print(f"Corrupt files:           {corrupt_count}")
    print(f"NaN files:               {nan_count}")
    print(f"Inf files:               {inf_count}")
    print("-" * 60)
    print(f"Train LR shape(s):       {list(lr_shapes)}")
    print(f"Train GT shape(s):       {list(gt_shapes)}")
    print(f"Train LR dtype(s):       {list(lr_dtypes)}")
    print(f"Train GT dtype(s):       {list(gt_dtypes)}")
    print(f"Train LR Min / Max:      {global_lr_min:.4f} / {global_lr_max:.4f}")
    print(f"Train GT Min / Max:      {global_gt_min:.4f} / {global_gt_max:.4f}")
    print(f"Train LR Mean / Std:     {np.mean(all_lr_means):.8f} / {np.mean(all_lr_stds):.8f}")
    print(f"Train GT Mean / Std:     {np.mean(all_gt_means):.8f} / {np.mean(all_gt_stds):.8f}")
    print("-" * 60)
    print(f"Test LR shape(s):        {list(test_shapes)}")
    print(f"Test LR dtype(s):        {list(test_dtypes)}")
    print(f"Test LR Min / Max:       {global_test_min:.4f} / {global_test_max:.4f}")
    print(f"Test LR Mean:            {np.mean(all_test_means):.8f}")
    print(f"Test NaN / Inf files:    {test_nan} / {test_inf}")
    print("=" * 60)

    # Return success flag
    success = (
        len(common_pairs) > 0
        and invalid_pairs == 0
        and nan_count == 0
        and inf_count == 0
        and corrupt_count == 0
        and len(test_files) > 0
    )
    if success:
        print("[SUCCESS] All dataset integrity checks passed cleanly!")
    else:
        print("[WARNING] Dataset validation found anomalies.")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate KLA Dataset")
    parser.add_argument("--train_root", default="train/train", help="Train dataset folder containing NoisyLR/ and GT/")
    parser.add_argument("--test_root", default="Test_NoisyLR/NoisyLR", help="Test dataset folder containing NoisyLR/")
    parser.add_argument("--max_inspect", type=int, default=None, help="Max samples to inspect (None for all)")
    args = parser.parse_args()

    success = validate_dataset(args.train_root, args.test_root, args.max_inspect)
    sys.exit(0 if success else 1)
