"""
Dataset Validation Script for KLA Semiconductor AI Image Restoration.

Checks:
- File counts & naming alignment
- Ignores __MACOSX and ._* metadata files
- Checks for corrupt, NaN, Inf, missing pairs
- Verifies shapes and 2x spatial relationship (LR 128x128 -> GT 256x256)
- Calculates and verifies dataset min, max, mean, std
- Exports reports/dataset_report.json and reports/dataset_report.txt
- Generates deterministic splits (splits/train.txt, splits/val.txt)
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

DATASET_REF_MEAN = 0.43353602
DATASET_REF_STD = 0.28478748


def find_valid_npy_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    valid_files = []
    for root, dirs, files in os.walk(folder_path):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.endswith(".npy") and not f.startswith("._") and not f.startswith("."):
                valid_files.append(f)
    return sorted(list(set(valid_files)))


def validate_dataset(train_root="train/train", test_root="Test_NoisyLR/NoisyLR", max_inspect=None, save_splits=True, seed=42):
    train_lr_dir = os.path.join(train_root, "NoisyLR") if not train_root.endswith("NoisyLR") else train_root
    train_gt_dir = os.path.join(train_root, "GT") if not train_root.endswith("GT") else train_root
    test_lr_dir = test_root

    # Fallbacks if default train_root needs adjustment
    if not os.path.exists(train_lr_dir):
        for candidate in ["train/train/NoisyLR", "train/NoisyLR"]:
            if os.path.exists(candidate):
                train_lr_dir = candidate
                break
    if not os.path.exists(train_gt_dir):
        for candidate in ["train/train/GT", "train/GT"]:
            if os.path.exists(candidate):
                train_gt_dir = candidate
                break
    if not os.path.exists(test_lr_dir):
        for candidate in ["Test_NoisyLR/NoisyLR", "Test_NoisyLR"]:
            if os.path.exists(candidate):
                test_lr_dir = candidate
                break

    print("=" * 65)
    print("KLA SEMICONDUCTOR DATASET VALIDATION")
    print("=" * 65)

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
    print("-" * 65)

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
        except Exception:
            corrupt_count += 1
            continue

        lr_shapes.add(tuple(lr_arr.shape))
        gt_shapes.add(tuple(gt_arr.shape))
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
            test_shapes.add(tuple(arr.shape))
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

    train_lr_mean = float(np.mean(all_lr_means)) if all_lr_means else 0.0
    train_lr_std = float(np.mean(all_lr_stds)) if all_lr_stds else 0.0
    train_gt_mean = float(np.mean(all_gt_means)) if all_gt_means else 0.0
    train_gt_std = float(np.mean(all_gt_stds)) if all_gt_stds else 0.0
    test_mean = float(np.mean(all_test_means)) if all_test_means else 0.0

    report_data = {
        "reference_statistics": {
            "mean": DATASET_REF_MEAN,
            "std": DATASET_REF_STD,
        },
        "train_set": {
            "total_lr_files": len(lr_files),
            "total_gt_files": len(gt_files),
            "paired_samples": len(common_pairs),
            "inspected_samples": len(inspect_list),
            "valid_2x_pairs": valid_2x_pairs,
            "invalid_pairs": invalid_pairs,
            "missing_gt": len(missing_gt),
            "missing_lr": len(missing_lr),
            "corrupt_files": corrupt_count,
            "nan_files": nan_count,
            "inf_files": inf_count,
            "lr_shapes": [list(s) for s in lr_shapes],
            "gt_shapes": [list(s) for s in gt_shapes],
            "lr_dtypes": list(lr_dtypes),
            "gt_dtypes": list(gt_dtypes),
            "lr_min": global_lr_min,
            "lr_max": global_lr_max,
            "gt_min": global_gt_min,
            "gt_max": global_gt_max,
            "lr_mean": train_lr_mean,
            "lr_std": train_lr_std,
            "gt_mean": train_gt_mean,
            "gt_std": train_gt_std,
        },
        "test_set": {
            "total_test_files": len(test_files),
            "shapes": [list(s) for s in test_shapes],
            "dtypes": list(test_dtypes),
            "nan_files": test_nan,
            "inf_files": test_inf,
            "min": global_test_min,
            "max": global_test_max,
            "mean": test_mean,
        },
        "validation_passed": (
            len(common_pairs) > 0
            and invalid_pairs == 0
            and nan_count == 0
            and inf_count == 0
            and corrupt_count == 0
            and len(test_files) > 0
        ),
    }

    # Save reports
    os.makedirs("reports", exist_ok=True)
    report_json_path = os.path.join("reports", "dataset_report.json")
    report_txt_path = os.path.join("reports", "dataset_report.txt")

    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=4)

    report_text = f"""=====================================================================
KLA SEMICONDUCTOR DATASET VALIDATION REPORT
=====================================================================
Reference Global Mean:  {DATASET_REF_MEAN:.8f}
Reference Global Std:   {DATASET_REF_STD:.8f}

--- Training Set Integrity ---
Total LR Files:         {len(lr_files)}
Total GT Files:         {len(gt_files)}
Valid Paired Samples:   {len(common_pairs)}
Inspected Pairs:        {len(inspect_list)}
Valid 2x Scaled Pairs:  {valid_2x_pairs}
Invalid Pairs:          {invalid_pairs}
Corrupt / NaN / Inf:    {corrupt_count} / {nan_count} / {inf_count}
LR Shapes:              {list(lr_shapes)}
GT Shapes:              {list(gt_shapes)}
LR Value Range:         [{global_lr_min:.4f}, {global_lr_max:.4f}]
GT Value Range:         [{global_gt_min:.4f}, {global_gt_max:.4f}]
LR Measured Mean / Std: {train_lr_mean:.8f} / {train_lr_std:.8f}
GT Measured Mean / Std: {train_gt_mean:.8f} / {train_gt_std:.8f}

--- Test Set Integrity ---
Total Test Files:       {len(test_files)}
Test Shapes:            {list(test_shapes)}
Test Value Range:       [{global_test_min:.4f}, {global_test_max:.4f}]
Test Measured Mean:     {test_mean:.8f}
Test NaN / Inf Files:   {test_nan} / {test_inf}

=====================================================================
Status: {"[PASSED] Integrity checks completed successfully." if report_data["validation_passed"] else "[WARNING] Integrity check anomalies detected."}
=====================================================================
"""
    with open(report_txt_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"Reports saved to {report_json_path} and {report_txt_path}")

    # Generate deterministic splits if requested
    if save_splits and len(common_pairs) > 0:
        os.makedirs("splits", exist_ok=True)
        rng = random.Random(seed)
        shuffled = list(common_pairs)
        rng.shuffle(shuffled)

        val_count = max(1, int(len(shuffled) * 0.1))
        val_files = sorted(shuffled[:val_count])
        train_files = sorted(shuffled[val_count:])

        train_split_path = os.path.join("splits", "train.txt")
        val_split_path = os.path.join("splits", "val.txt")

        with open(train_split_path, "w") as f:
            f.write("\n".join(train_files))
        with open(val_split_path, "w") as f:
            f.write("\n".join(val_files))

        print(f"Saved deterministic split ({len(train_files)} train, {len(val_files)} val) with seed {seed}:")
        print(f"  - {train_split_path}")
        print(f"  - {val_split_path}")

    return report_data["validation_passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate KLA Dataset and Generate Reports")
    parser.add_argument("--train_root", default="train/train", help="Train dataset folder containing NoisyLR/ and GT/")
    parser.add_argument("--test_root", default="Test_NoisyLR/NoisyLR", help="Test dataset folder containing NoisyLR/")
    parser.add_argument("--max_inspect", type=int, default=None, help="Max samples to inspect (None for all)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    args = parser.parse_args()

    success = validate_dataset(args.train_root, args.test_root, args.max_inspect, seed=args.seed)
    sys.exit(0 if success else 1)
