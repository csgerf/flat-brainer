from enum import Enum
from typing import Union
from pathlib import Path

import numpy as np
import cv2

from src.core.utils import fs

COMMON_IMAGE_EXTENSIONS = [".bmp", ".png", ".jpeg", ".jpg", ".tiff", ".tif", ".nrrd", ".dcm", ".npy"]


class ImageReadMode(Enum):
    """
    Support for various modes while reading images.

    Use ``ImageReadMode.UNCHANGED`` for loading the image as-is,
    ``ImageReadMode.GRAY`` for converting to grayscale,
    ``ImageReadMode.GRAY_ALPHA`` for grayscale with transparency,
    ``ImageReadMode.RGB`` for RGB and ``ImageReadMode.RGB_ALPHA`` for
    RGB with transparency.
    """
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


# def read_png_file(path: str, mode: ImageReadMode) -> np.ndarray:
#
#     # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#     #     _log_api_usage_once(read_file)
#     data = torch.ops.image.read_file(path)
#     return data

# cv.IMREAD_UNCHANGED If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
# cv.IMREAD_GRAYSCALE If set, always convert image to the single channel grayscale image (codec internal conversion).
# cv.IMREAD_COLOR If set, always convert image to the 3 channel BGR color image.
# cv.IMREAD_ANYDEPTH If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
# cv.IMREAD_ANYCOLOR If set, the image is read in any possible color format.
# cv.IMREAD_LOAD_GDAL If set, use the gdal driver for loading the image.
# cv.IMREAD_REDUCED_GRAYSCALE_2 If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
# cv.IMREAD_REDUCED_COLOR_2 If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
# cv.IMREAD_REDUCED_GRAYSCALE_4 If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
# cv.IMREAD_REDUCED_COLOR_4 If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
# cv.IMREAD_REDUCED_GRAYSCALE_8 If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
# cv.IMREAD_REDUCED_COLOR_8 If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
# cv.IMREAD_IGNORE_ORIENTATION If set, do not rotate the image according to EXIF's orientation flag.



def read_rgb_image(file_name: Union[str, Path]) -> np.ndarray:
    if type(file_name) != str:
        file_name = str(file_name)

    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f'Cannot read image "{file_name}"')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image


def read_image_unchanged(file_name: Union[str, Path]) -> np.ndarray:
    if type(file_name) != str:
        file_name = str(file_name)

    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f'Cannot read image "{file_name}"')

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image
