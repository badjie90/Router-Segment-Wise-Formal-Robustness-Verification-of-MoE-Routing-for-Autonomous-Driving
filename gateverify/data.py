from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import resolve_path


class PixelDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], image_size: int):
        self.rows = list(rows)
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image = self.transform(Image.open(row["image_path"]).convert("RGB"))
        objects = [row["car_present"], row["pedestrian_present"], row["traffic_sign_present"]]
        return {"image": image, "objects": torch.tensor(objects, dtype=torch.float32),
                "image_path": row["image_path"], "index": index}


def load_rows(cfg: Dict[str, Any], filename: str, max_samples: int | None = None):
    metadata_dir = resolve_path(cfg, cfg["baseline"]["metadata_dir"])
    path = metadata_dir / filename
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    limit = max_samples if max_samples is not None else cfg["data"].get("max_samples")
    return rows[: int(limit)] if limit else rows


def load_test_rows(cfg: Dict[str, Any], max_samples: int | None = None):
    return load_rows(cfg, cfg["data"].get("split_file", "test_fixed.json"), max_samples)
