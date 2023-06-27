from typing import Dict, Tuple, Union, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(config: dict, split: str):
    """Get the transforms for the given split."""
    pass