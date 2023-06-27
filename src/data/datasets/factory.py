from typing import Dict, Any, Optional
import os
import torch
# import medmnist
import torchvision.transforms as transforms

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from .hubmap.factory import get_train_test_dataset as get_hubmap_dataset


# def get_medmnist_dataset(data_flag='pathmnist', download=True, as_rgb=True, transform=None, **_):
#     info = medmnist.INFO[data_flag]
#     DataClass = getattr(medmnist, info['python_class'])
#     train_dataset = DataClass(split='train', transform=transform, download=download, as_rgb=as_rgb)
#     val_dataset = DataClass(split='val', transform=transform, download=download, as_rgb=as_rgb)
#
#     return train_dataset, val_dataset


def get_dataset(config, transform=None):
    # if name == medmnist:
    #     return get_medmnist_dataset(transform=transform, **params)
    if config.name == "hubmap":
        return get_hubmap_dataset(transform=transform, params=config.params)


def get_dataloader(config):
    data_transform = albu.Compose(
        [
            # albu.HorizontalFlip(p=0.5),
            # albu.VerticalFlip(p=0.5),
            albu.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()])
    
    batch_size = config.trainer.batch_size
    num_workers = config.dataset.params.get("num_workers", 0)

    train_dataset, val_dataset = get_dataset(config.dataset, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_loaders = {"train": train_loader, "val": val_loader}

    return data_loaders
