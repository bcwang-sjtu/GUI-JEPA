"""Unlabelled screenshots for JEPA pretraining (any directory tree of images)."""
import os
from pathlib import Path
from typing import List

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def find_images(root: str) -> List[str]:
    paths = []
    for dp, _, files in os.walk(root, followlinks=True):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                paths.append(os.path.join(dp, f))
    return sorted(paths)


class ScreenshotDataset(Dataset):
    """224x224 random-resized crops; no horizontal flip (GUIs are not mirror-symmetric)."""

    def __init__(self, root: str, image_size: int = 224):
        self.paths = find_images(root)
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.tf = T.Compose([
            T.Lambda(lambda im: im if im.mode == "RGB" else im.convert("RGB")),
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            return self.tf(Image.open(self.paths[idx]))
        except Exception:  # unreadable file: fall back to the next image
            return self.__getitem__((idx + 1) % len(self))
