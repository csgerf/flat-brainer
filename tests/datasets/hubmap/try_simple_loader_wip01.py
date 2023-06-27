import sys

working_dir = "C:\\dev\\working\\cv-train"
sys.path.append(working_dir)

import os

# import src
from src.data.datasets.factory import get_dataloader


def test_get_dataloader():
    name = "hubmap"
    params = {
        "name": "SimpleDataset",
        "num_workers": 0,
    }

    data_loaders = get_dataloader(name, params)
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]

    print(f"train_loader size: {len(train_loader)}")

    batch = next(iter(train_loader))
    print(f"batch type: {type(batch)}")
    # print(f"batch shape: {len(batch)}")
    images = batch["image"]
    labels = batch["label"]
    print(f"input tensor shape: {images.shape}")
    print(f"label tensor shape: {labels.shape}")


if __name__ == "__main__":
    test_get_dataloader()
