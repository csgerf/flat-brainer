import os
from typing import Any, Optional
import pandas as pd
import torch
from torch.utils import data
import cv2
import numpy as np

IMAGE_DATA_DIR = "D:\\data\\kaggle\\hubmap-2023\\raw\\train"
MASK_DATA_DIR = "D:\\data\kaggle\\hubmap-2023\\processed\\masks\\composite"
DEFAULT_CSV_PATH = "D:\\data\\kaggle\\hubmap-2023\\tile_meta_train_split_01.csv"

__ALL__ = ["SimpleDataset"]


def read_tiff_file(file_path: str) -> np.ndarray:
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return image


def read_mask_file(file_path: str) -> np.ndarray:
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    image = image[:, :, 0]
    image = np.where(image > 0, 1, image).astype(np.uint8)
    image = np.expand_dims(image, axis=0)
    return image


class SimpleDataset(data.Dataset):
    def __init__(self, csv_path=DEFAULT_CSV_PATH,
                 is_train: bool = True,
                 data_dir: str = IMAGE_DATA_DIR,
                 mask_dir: str = MASK_DATA_DIR,
                 transform: Optional[Any] = None, **_):
        df = pd.read_csv(csv_path)
        df = df[df["is_train"] == is_train]
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["id"]
        num_labels = self.df.iloc[idx]["blood_vessel_count"]
        labels = np.zeros((1, 512, 512), dtype=np.uint8)
        if num_labels > 0:
            label_file_path = os.path.join(self.mask_dir, image_id + ".png")
            labels = read_mask_file(label_file_path)

        image_file_path = os.path.join(self.data_dir, image_id + ".tif")
        image = read_tiff_file(image_file_path)

        if self.transform:
            transformed_image = self.transform(image=image, mask=labels)
            image = transformed_image["image"]
            labels = transformed_image["mask"]

        return {"image": image, "label": labels}
