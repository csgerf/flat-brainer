import numpy as np

__all__ = ["is_3d_rgb_image", "is_3d_grayscale_image", "is_2d_rgb_image", "is_2d_grayscale_image"]


def is_3d_rgb_image(image: np.ndarray) -> bool:
    return len(image.shape) == 4 and image.shape[-1] == 3


def is_3d_grayscale_image(image: np.ndarray) -> bool:
    return (len(image.shape) == 3) or (len(image.shape) == 4 and image.shape[-1] == 1)


def is_2d_rgb_image(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[-1] == 3


def is_2d_grayscale_image(image: np.ndarray) -> bool:
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)
