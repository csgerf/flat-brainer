from typing import Tuple

import h5py
import numpy as np
from src.flatmap.ccf_streamlines.projection import _matching_voxel_indices
from scipy.spatial.distance import euclidean


__ALL__ = ["vector_to_3d_affine_matrix",
           "find_closest_streamline",
           "determine_angle_between_streamline_and_plane",
           "coordinates_to_voxels",
           "upscale_ish_volume",
           "transform_coordinates_to_volume",
           "remove_duplicate_voxels_from_paths",
           ]


def vector_to_3d_affine_matrix(vec):
    M = np.array([[vec[0], vec[1], vec[2], vec[9]],
                  [vec[3], vec[4], vec[5], vec[10]],
                  [vec[6], vec[7], vec[8], vec[11]],
                  ])
    return M


def find_closest_streamline(
        coord,
        closest_surface_voxel_reference,
        surface_paths,
        resolution=(10, 10, 10),
        volume_shape=(1320, 800, 1140),
):
    """ Find the nearest streamline of a CCF coordinate.

    If voxel is not in the reference file, returns None.

    Parameters
    ----------
    coord : (3, ) array
        Coordinates of point (microns)
    closest_surface_voxel_reference : str or array
        Either a file path to a HDF5 file containing information about the closest
        streamlines for voxels within the isocortex, or the already loaded
        array from that file.
    surface_paths : str or h5py file object
        Either a file path to an HDF5 file containing information about the paths between
        the top and bottom of cortex or an open h5py file object referring to that information.
    resolution : tuple, default (10, 10, 10)
        3-tuple of voxel dimensions in microns
    volume_shape : tuple, default (1320, 800, 1140)
        3-tuple of volume shape in voxels

    Returns
    -------
    streamline_coords : (N, 3) array
        3-D coordinates of streamline in microns
    """

    coord = np.array(coord)
    if len(coord.shape) == 1:
        coord = coord.reshape(1, -1)

    if isinstance(closest_surface_voxel_reference, str):
        with h5py.File(closest_surface_voxel_reference, "r") as f:
            closest_dset = f["closest surface voxel"]
            closest_surface_voxels = closest_dset[:]
    else:
        closest_surface_voxels = closest_surface_voxel_reference

    voxel = np.squeeze(coordinates_to_voxels(coord))

    # Reference file data only present on left side, so flip to left side
    # if voxel is on the right
    z_size = volume_shape[2]
    z_midline = z_size / 2
    flip_hemisphere = False
    if voxel[2] > z_midline:
        voxel[2] = z_size - voxel[2]
        flip_hemisphere = True

    voxel_ind = np.ravel_multi_index(
        tuple(voxel),
        volume_shape
    )
    matching_surface_voxel_ind = _matching_voxel_indices(
        np.array([voxel_ind]),
        closest_surface_voxels)[0]

    # Pull path from surface paths file
    if isinstance(surface_paths, str):
        with h5py.File(surface_paths, "r") as f:
            path_dset = f['paths']
            volume_lookup_dset = f['volume lookup flat']
            path_ind = volume_lookup_dset[matching_surface_voxel_ind]
            path = path_dset[path_ind, :]
    else:
        path_dset = surface_paths['paths']
        volume_lookup_dset = surface_paths['volume lookup flat']
        path_ind = volume_lookup_dset[matching_surface_voxel_ind]
        path = path_dset[path_ind, :]

    path = path[path > 0]

    # Convert path to coordinates in microns
    streamline_coords = np.unravel_index(
        path, volume_shape)
    streamline_coords = np.array(streamline_coords).T
    if flip_hemisphere:
        # Put streamline on same hemisphere as original voxel
        streamline_coords[:, 2] = z_size - streamline_coords[:, 2]

    # Scale to microns
    streamline_coords = streamline_coords * np.array(resolution)
    return streamline_coords


def determine_angle_between_streamline_and_plane(
        streamline_coords,
        plane_transform):
    """
    Find angle between the surface of a given plane and a streamline.

    Parameters
    ----------
    streamline_coords : (N, 3) array
        Coordinates in microns of streamline
    plane_transform : (3, 4) array
        3-D affine transform matrix for putting a plane into CCF space.

    Returns
    -------
    angle : float
        Angle between streamline and plane (in degrees)
    """

    # Find normal vector for plane transformed to CCF space
    a2 = np.dot(plane_transform, np.array([0, 0, 0, 1]))
    b2 = np.dot(plane_transform, np.array([1, 0, 0, 1]))
    c2 = np.dot(plane_transform, np.array([0, 1, 0, 1]))

    norm_vec = np.cross(b2 - a2, c2 - a2)
    norm_unit = norm_vec / euclidean(norm_vec, [0, 0, 0])

    streamline_wm = streamline_coords[-1, :]
    streamline_pia = streamline_coords[0, :]
    streamline_unit = (streamline_pia - streamline_wm) / euclidean(streamline_pia, streamline_wm)

    angle_with_norm = np.arctan2(
        np.linalg.norm(np.cross(norm_unit, streamline_unit)),
        np.dot(norm_unit, streamline_unit))

    # Take complement and convert from radians to degrees
    return 90. - (angle_with_norm * 180. / np.pi)


def coordinates_to_voxels(coords: np.ndarray, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    """ Find the voxel coordinates of spatial coordinates

    Parameters
    ----------
    coords : array
        (n, m) coordinate array. m must match the length of `resolution`
    resolution : tuple, default (10, 10, 10)
        Size of voxels in each dimension

    Returns
    -------
    voxels : array
        Integer voxel coordinates corresponding to `coords`
    """

    if len(resolution) != coords.shape[1]:
        raise ValueError(
            f"second dimension of `coords` must match length of `resolution`; "
            f"{len(resolution)} != {coords.shape[1]}")

    if not np.issubdtype(coords.dtype, np.number):
        raise ValueError(f"coords must have a numeric dtype (dtype is '{coords.dtype}')")

    voxels = np.floor(coords / resolution).astype(int)
    return voxels


def upscale_ish_volume(
        volume,
        orig_voxel_size=200,
        target_voxel_size=10,
        target_volume_shape=(1320, 800, 1140),
        rotate_axes=True,
):
    """ Upscale a lower-resolution ISH volume for projection.

    Parameters
    ----------
    volume : array
        Array of ISH expression data
    orig_voxel_size : int, default 200
        Size in microns of voxels of original data
    target_voxel_size : int, default 10
        Size of microns of voxels in desired volume
    target_volume_shape : tuple, default (1320, 800, 1140)
        Shape of target upscaled volume
    rotate_axes : bool, default True
        Whether to swap the x and z axes of the input volume. Volumes downloaded
        directly from the ISH atlas API have the anterior-posterior axis in the
        z-axis and the left-right axis in the x-axis, while the CCFv3 has those
        swapped. The default assumes that the input volume has not been adjusted
        to match the CCF yet.

    Returns
    -------
    upscaled_volume : array
        Upscaled ISH expression data
    """

    if rotate_axes:
        volume = np.swapaxes(volume, 0, 2)

    voxel_ratio = orig_voxel_size // target_voxel_size
    ind = np.indices(target_volume_shape, sparse=True)
    downscaled_ind = tuple(ind_dim // voxel_ratio for ind_dim in ind)
    upscaled_volume = np.zeros(target_volume_shape)
    upscaled_volume[ind] = volume[downscaled_ind]

    return upscaled_volume


def transform_coordinates_to_volume(
        coords,
        volume_shape: Tuple[int, int, int] = (1320, 800, 1140),
        resolution: Tuple[int, int, int] = (10, 10, 10),
) -> np.ndarray:
    """ Create a volume with counts of points.

    Parameters
    ----------
    coords : (N, 3) array
        Coordinates of points
    volume_shape : 3-tuple of ints, default (1320, 800, 1140)
        Shape of target volume in voxels
    resolution : 3-tuple of ints, default (10, 10, 10)
        Size of voxels in microns

    Returns
    -------
    volume : array
        Volume with shape `volume_shape` with counts of points in each voxel
    """
    voxels = coordinates_to_voxels(coords, resolution=resolution)

    # Count the nodes in each voxel
    populated_voxels, counts = np.unique(voxels, axis=0, return_counts=True)

    # Place counts into volume
    volume = np.zeros(volume_shape, dtype=np.uint32)
    volume[populated_voxels[:, 0], populated_voxels[:, 1], populated_voxels[:, 2]] = counts

    return volume


def remove_duplicate_voxels_from_paths(paths: np.ndarray) -> np.ndarray:
    """ Remove duplicate consecutive voxels from the paths

    Duplicate consecutive voxels are identified and removed - the list of
    voxels is then shifted to fill in the gaps left by the duplicates, so the
    format is consistent with the original.

    Parameters
    ----------
    paths : array
        Set of paths containing some duplicated voxels

    Returns
    -------
    new_paths : array
        Set of paths with duplicate voxels removed
    """

    # Remove duplicate consecutive voxels from the paths:
    # First identify places where voxels change (i.e. no duplicates) as
    # values to keep
    paths_diff = np.diff(paths, axis=1)

    # Determine how many remaining entries per line and how many zeros to
    # add at the end
    nonzero_per_row = np.count_nonzero(paths_diff, axis=1)
    n_paths, max_len = paths.shape
    zeros_at_end = max_len - nonzero_per_row

    # Insert the zeros at the ends of each line, then put back together
    # into 2D array
    insert_locs = np.cumsum(nonzero_per_row)
    paths = np.insert(
        paths[np.nonzero(paths_diff)],
        np.repeat(insert_locs, zeros_at_end),
        0).reshape(n_paths, -1)

    return paths
