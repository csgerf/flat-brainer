import cv2
import numpy as np
import torch
from torchvision.transforms import Lambda


class NormalizePerImage3D:
    def __init__(self, min_val: int=0, max_val: int=2**16 - 1, per_plane: bool=False):
        self.min_val = min_val
        self.max_val = max_val
        self.per_plane = per_plane

    def __call__(self, tensor) -> torch.Tensor:
        if tensor.dtype == torch.uint16 or tensor.dtype == torch.int16:
            # 16-bit image
            max_val = 2**16 - 1
        elif tensor.dtype == torch.uint8:
            # 8-bit image
            max_val = 2**8 - 1
        else:
            raise ValueError('Unsupported image type')

        if self.per_plane:
            for plane in tensor:
                plane -= torch.min(plane)
                plane /= torch.max(plane)
        else:
            tensor -= torch.min(tensor)
            tensor /= torch.max(tensor)
        
        # Scale to desired range
        tensor = tensor * (self.max_val - self.min_val) + self.min_val
        return tensor


def resize_image(image:np.ndarray, size, mode=cv2.INTER_LINEAR):
    resized_image = cv2.resize(image, size, interpolation=mode)
    return resized_image
