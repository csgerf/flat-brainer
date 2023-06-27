import sys

working_dir = "C:\\dev\\working\\cv-train"
sys.path.append(working_dir)

import os
import random
from typing import List, Tuple, Dict, Union, Any, Optional

import tifffile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import json
import cv2
import skimage
from tqdm import tqdm
import shutil

# my imports
from src.core.utils import fs

ROOT_DATA_DIR = "D:\\data\\kaggle\\hubmap-2023"
BASE_DATA_DIR = os.path.join(ROOT_DATA_DIR, "raw")
TRAIN_IMAGE_DIR = os.path.join(BASE_DATA_DIR, "train")
TRAIN_MASKS_FILE = os.path.join(BASE_DATA_DIR, "polygons.jsonl")
TILE_METADATA_FILE = os.path.join(BASE_DATA_DIR, "tile_meta.csv")
WSI_METADATA_FILE = os.path.join(BASE_DATA_DIR, "wsi_meta.csv")
TILE_SIZE = 512

LABEL_TYPES = ["glomerulus", "blood_vessel", "unsure"]

train_file_list = fs.find_in_dir_with_ext(TRAIN_IMAGE_DIR, [".tiff", ".tif"])
print(len(train_file_list))
print(train_file_list[0])
image = cv2.imread(train_file_list[0], cv2.IMREAD_UNCHANGED)


def read_polygon_file(file_path: str = TRAIN_MASKS_FILE) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]

    return json_labels


def get_polys_from_list(json_labels: List[Dict[str, Any]], labels: List[str] = LABEL_TYPES) -> Dict[str, Any]:
    poly_dict: Dict[str, Any] = {}

    for json_label in tqdm(json_labels):
        image_id = json_label['id']
        annotations = json_label['annotations']
        label_dict: Dict[str, List[np.ndarray]] = {key: [] for key in labels}
        for item in annotations:
            label_type = item["type"]
            if label_type not in labels:
                raise ValueError(f"Unknown label type: {label_type}")

            coords = np.array(item["coordinates"])
            label_dict[label_type].append(coords)

        poly_dict[image_id] = label_dict

    return poly_dict


def add_label_counts(df: pd.DataFrame, poly_dict: Dict[str, Any], labels: List[str] = LABEL_TYPES) -> pd.DataFrame:
    for label in labels:
        df[f"{label}_count"] = 0

    for image_id, label_dict in tqdm(poly_dict.items()):
        for label in labels:
            df.loc[df["id"] == image_id, f"{label}_count"] = len(label_dict[label])

    return df


def add_tile_indicies(df: pd.DataFrame, tile_size: int) -> pd.DataFrame:
    df["tile_row"] = df['i'] // tile_size
    df["tile_col"] = df['j'] // tile_size
    df["tile_id"] = df["tile_col"].astype(str) + "_" + df["tile_row"].astype(str)
    return df


def create_simple_train_test_split(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    image_ids = df["id"].unique()
    random.shuffle(image_ids)
    train_image_ids = image_ids[:int(train_frac * len(image_ids))]
    df["is_train"] = df["id"].isin(train_image_ids)
    return df


def main(output_base_dir: str = ROOT_DATA_DIR, output_filename: str = "preproceessed_train.csv"):
    output_path = os.path.join(output_base_dir, output_filename)
    df = pd.read_csv(TILE_METADATA_FILE)
    json_labels = read_polygon_file()
    poly_dict = get_polys_from_list(json_labels)
    df = add_label_counts(df, poly_dict)
    df = add_tile_indicies(df, TILE_SIZE)
    df = create_simple_train_test_split(df)

    df.to_csv(TILE_METADATA_FILE, index=False)


if __name__ == "__main__":
    main()
