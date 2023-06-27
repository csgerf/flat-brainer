import os
import zarr
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils import data


ZARR_DATA_FILE = "D:\\data\\kaggle\\vesuvius-challenge-ink-detection\\processed\\my_data.zarr"


class ZarrBaseDataSet(data.Dataset):
    def __init__(self, zarr_path, experiment_ids=("1", "2"), tile_size: int = 256, mode="train", level: int = 0, transform=None):
        self.mode = mode
        self.transform = transform
        self.zarr_path = zarr_path
        self.level = level
        self.zarr_data = zarr.open(zarr_path, mode="r")

    def get_image_grid(self, dataset_id: int):
        mask_image = self.zarr_data.masks[f"{dataset_id}_level_{self.level}"]


