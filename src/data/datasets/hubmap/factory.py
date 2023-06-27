from typing import Dict, Any
from .simple_dataset import SimpleDataset

__ALL__ = ["get_dataset", " get_train_test_dataset"]


def get_dataset(params, split: str = "train", transform=None, ):
    name = params.get("name", "SimpleDataset")
    if name == "SimpleDataset":
        is_train = split == "train"
        return SimpleDataset(transform=transform, is_train=is_train, **params)


def get_train_test_dataset(params, transform=None):
    train_dataset = get_dataset(params, split="train", transform=transform)
    test_dataset = get_dataset(params, split="test", transform=transform)
    return train_dataset, test_dataset
