from typing import Tuple
import numpy as np
import cv2


__ALL__ = ['mask_from_polygon']


def mask_from_polygon(polygon: np.ndarray, image_shape: Tuple[int, int], fill_value: int = 1) -> np.ndarray:
    """
    Create a mask from a polygon.

    :param polygon: a polygon defined by a list of points
    :param image_shape: the shape of the image
    :return: a mask
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], fill_value)
    return mask
