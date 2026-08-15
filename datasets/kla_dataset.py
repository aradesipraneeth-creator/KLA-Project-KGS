"""
PyTorch Dataset implementations for KLA Semiconductor AI Image Restoration.

Features:
- Paired loading of NoisyLR (128x128) and Clean GT (256x256)
- Strict filtering of __MACOSX and ._* metadata files
- Safe synchronous paired data augmentations (H-flip, V-flip, 90-deg rotation, paired crop)
- Normalization using global dataset statistics (datasets.preprocessing)
- Unpaired test dataset for evaluation and inference
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.preprocessing import normalize


def get_clean_npy_filelist(directory):
    """
    Returns sorted list of valid .npy filenames, explicitly ignoring __MACOSX and ._*
    Uses fast os.scandir for near-instant discovery.
    """
    if not os.path.exists(directory):
        return []
    valid_files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            name = entry.name
            if entry.is_file() and name.endswith(".npy") and not name.startswith("._") and not name.startswith("."):
                valid_files.append(name)
    return sorted(valid_files)


class KLAPairedDataset(Dataset):
    """
    Paired Dataset for training and validation.
    """
    def __init__(
        self,
        lr_dir,
        gt_dir,
        file_list=None,
        is_train=True,
        patch_size=None,  # LR patch size (GT will be 2 * patch_size), e.g. 64 or 128
        augment=True,
        normalize_data=True,
    ):
        super().__init__()
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.is_train = is_train
        self.patch_size = patch_size
        self.augment = augment and is_train
        self.normalize_data = normalize_data

        if file_list is not None:
            self.file_list = sorted(list(file_list))
        else:
            lr_files = set(get_clean_npy_filelist(lr_dir))
            gt_files = set(get_clean_npy_filelist(gt_dir))
            self.file_list = sorted(list(lr_files.intersection(gt_files)))

    def __len__(self):
        return len(self.file_list)

    def _paired_crop(self, lr_img, gt_img):
        """
        Crop LR by patch_size and GT by 2 * patch_size at matching coordinates.
        """
        if self.patch_size is None:
            return lr_img, gt_img

        lr_h, lr_w = lr_img.shape
        if lr_h < self.patch_size or lr_w < self.patch_size:
            return lr_img, gt_img

        lr_top = random.randint(0, lr_h - self.patch_size)
        lr_left = random.randint(0, lr_w - self.patch_size)

        lr_crop = lr_img[lr_top : lr_top + self.patch_size, lr_left : lr_left + self.patch_size]
        gt_top = lr_top * 2
        gt_left = lr_left * 2
        gt_crop_size = self.patch_size * 2
        gt_crop = gt_img[gt_top : gt_top + gt_crop_size, gt_left : gt_left + gt_crop_size]

        return lr_crop, gt_crop

    def _paired_augment(self, lr_img, gt_img):
        """
        Synchronous spatial augmentation on LR and GT.
        """
        # Horizontal flip
        if random.random() < 0.5:
            lr_img = np.fliplr(lr_img)
            gt_img = np.fliplr(gt_img)

        # Vertical flip
        if random.random() < 0.5:
            lr_img = np.flipud(lr_img)
            gt_img = np.flipud(gt_img)

        # 90-degree rotations (k in [0, 1, 2, 3])
        k = random.randint(0, 3)
        if k > 0:
            lr_img = np.rot90(lr_img, k)
            gt_img = np.rot90(gt_img, k)

        return lr_img.copy(), gt_img.copy()

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        lr_arr = np.load(lr_path).astype(np.float32)
        gt_arr = np.load(gt_path).astype(np.float32)

        # Paired crop if training with sub-patches
        if self.is_train and self.patch_size is not None:
            lr_arr, gt_arr = self._paired_crop(lr_arr, gt_arr)

        # Paired augmentation
        if self.augment:
            lr_arr, gt_arr = self._paired_augment(lr_arr, gt_arr)

        # Normalization
        if self.normalize_data:
            lr_norm = normalize(lr_arr)
            gt_norm = normalize(gt_arr)
        else:
            lr_norm = lr_arr
            gt_norm = gt_arr

        # To tensor [1, H, W]
        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).float()

        return {
            "lr": lr_tensor,
            "gt": gt_tensor,
            "filename": filename,
            "lr_raw": torch.from_numpy(lr_arr).unsqueeze(0).float(),
            "gt_raw": torch.from_numpy(gt_arr).unsqueeze(0).float(),
        }


class KLATestDataset(Dataset):
    """
    Dataset for test inference.
    """
    def __init__(self, input_dir, normalize_data=True):
        super().__init__()
        self.input_dir = input_dir
        self.normalize_data = normalize_data
        self.file_list = get_clean_npy_filelist(input_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.input_dir, filename)
        lr_arr = np.load(file_path).astype(np.float32)
        orig_shape = lr_arr.shape

        if self.normalize_data:
            lr_norm = normalize(lr_arr)
        else:
            lr_norm = lr_arr

        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).float()

        return {
            "lr": lr_tensor,
            "filename": filename,
            "orig_shape": orig_shape,
            "lr_raw": torch.from_numpy(lr_arr).unsqueeze(0).float(),
        }
