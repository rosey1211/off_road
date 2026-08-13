# ─── off_road/dataset.py ──────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import config


class OffRoadDataset(Dataset):
    """
    Reads offroad_labels.csv from a train/ or test/ split directory.

    CSV layout:
        filename, r0_p0..r0_p49, r1_p0..r1_p49, r2_p0..r2_p49, r3_p0..r3_p49
    Row 0 = nearest scan line (largest y / bottom of image).
    Row 3 = farthest scan line (smallest y / top of image).
    Values are continuous floats 0.0–1.0 (1.0 = most traversable).

    Returns (image_tensor, labels_tensor) where:
        image_tensor  : (3, IMAGE_HEIGHT, IMAGE_WIDTH) normalised float32
        labels_tensor : (N_ROWS, N_POINTS) float32, values in [0, 1]
    """

    def __init__(self, csv_path: str, img_dir: str, augment: bool = False):
        self.df      = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment

        normalize = T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

        if augment:
            self.transform = T.Compose([
                T.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.transform = T.Compose([
                T.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                T.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img      = Image.open(img_path).convert("RGB")
        img_t    = self.transform(img)

        # Build (N_ROWS, N_POINTS) label tensor
        labels = np.zeros((config.N_ROWS, config.N_POINTS), dtype=np.float32)
        for r in range(config.N_ROWS):
            cols = [f"r{r}_p{p}" for p in range(config.N_POINTS)]
            labels[r] = row[cols].values.astype(np.float32)

        return img_t, torch.from_numpy(labels)
