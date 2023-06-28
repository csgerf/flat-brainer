from typing import Dict, Tuple
import numpy as np
import time
import src.flatmap.ccf_streamlines.projection as ccfproj
from src.flatmap.ccf_streamlines.morphology import transform_coordinates_to_volume
import matplotlib.pyplot as plt

from skimage.morphology import binary_dilation, disk, dilation
from skimage.segmentation import expand_labels


def get_projection_images_from_dict(proj_butterfly_slab: ccfproj.Isocortex3dProjector, markers: Dict[str, np.ndarray]):
    morphological_list = []
    names = markers.keys()

    for i, points in enumerate(markers.values()):
        tic = time.perf_counter()
        point_vol = transform_coordinates_to_volume(points, resolution=(10,-10,-10))
        morphological_list.append(proj_butterfly_slab.project_volume(point_vol, thickness_type="unnormalized"))
        toc = time.perf_counter()
        print(f"Finished in {toc - tic:0.4f} seconds")

    return morphological_list, names


def setup_main_plot(main_shape: Tuple[int, int] = (1360, 2352), top_shape: Tuple[int, int] = (200, 2352), left_shape: Tuple[int, int] = (1360, 200)):
    # Set up a figure to plot them together
    fig, axes = plt.subplots(2, 2,
                 gridspec_kw=dict(
                 width_ratios=(left_shape[1], main_shape[1]),
                 height_ratios=(top_shape[0], main_shape[0]),
                 hspace=0.01,
                 wspace=0.01),
                 figsize=(19.4, 12))
    axes[1,1].invert_yaxis()
    axes[0, 0].set(xticks=[], yticks=[])
    axes[0, 1].set(xticks=[], yticks=[], anchor="SW")
    axes[1, 1].set(xticks=[], yticks=[], anchor="NW")
    axes[1, 0].set(xticks=[], yticks=[], anchor="NE")
    return fig, axes


def plot_boundaries(axes, bf_left_boundaries, bf_right_boundaries, cmap: str = "black", alpha=0.5):
    # We can plot the boundaries of the regions we found
    for k, boundary_coords in bf_left_boundaries.items():
        axes[1, 1].plot(*boundary_coords.T, c=cmap, lw=0.5, alpha=alpha)

    for k, boundary_coords in bf_right_boundaries.items():
        axes[1, 1].plot(*boundary_coords.T, c=cmap, lw=0.5, alpha=alpha)

    return axes


def expand_label_image(image: np.ndarray, radius: int = 12) -> np.ndarray:
    image = expand_labels(image, radius)

    return image


def dilate_image(image: np.ndarray, radius: int = 12) -> np.ndarray:
    image = dilation(image, disk(radius))

    return image


def dilate_binary_image(image: np.ndarray, radius: int = 12) -> np.ndarray:
    image = binary_dilation(image, disk(radius))
    return image
