from typing import Dict, Tuple
import json
import numpy as np
import nrrd

from src.flatmap.ccf_streamlines import projection as ccfproj
from src.flatmap import external_data_registry as data_registry

DEFAULT_DATA_DIRECTORY = "C:\\dev\\working\\flat-map\\flatmap\\data"
AVERAGE_TEMPLATE_PATH = (
    "D:\\data\\neuro\\Allen\\average_template_10.nrrd"
)
BoundaryType = Dict[str, np.ndarray]

__ALL__ = [
    "get_boundaries",
    "get_average_template_projection",
    "get_isocortex_3d_projector",
    "get_coordinate_projector",
    "get_avg_normalized_layers_butterfly",
]


def get_layer_depth_from_json(json_file: str):
    with open(json_file, "r") as f:
        layer_tops = json.load(f)

    layer_thicknesses = {
        "Isocortex layer 1": layer_tops["2/3"],
        "Isocortex layer 2/3": layer_tops["4"] - layer_tops["2/3"],
        "Isocortex layer 4": layer_tops["5"] - layer_tops["4"],
        "Isocortex layer 5": layer_tops["6a"] - layer_tops["5"],
        "Isocortex layer 6a": layer_tops["6b"] - layer_tops["6a"],
        "Isocortex layer 6b": layer_tops["wm"] - layer_tops["6b"],
    }

    return layer_thicknesses


def get_boundaries(
        view_lookup_file: str = "flatmap_butterfly",
        hemisphere: str = "right_for_both",
        view_space_for_other_hemisphere: str = "flatmap_butterfly",
) -> Tuple[BoundaryType, BoundaryType]:
    projected_atlas_file_path = data_registry.get_atlas_file_path(view_lookup_file)
    label_file_path = data_registry.get_other_file_path(
        "labelDescription_ITKSNAPColor.txt"
    )

    bf_boundary_finder = ccfproj.BoundaryFinder(
        projected_atlas_file=projected_atlas_file_path, labels_file=label_file_path
    )

    # We get the left hemisphere region boundaries with the default arguments
    bf_left_boundaries = bf_boundary_finder.region_boundaries()

    # And we can get the right hemisphere boundaries that match up with
    # our projection if we specify the same configuration
    bf_right_boundaries = bf_boundary_finder.region_boundaries(
        # we want the right hemisphere boundaries, but located in the right place
        # to plot both hemispheres at the same time
        hemisphere=hemisphere,
        # we also want the hemispheres to be adjacent
        view_space_for_other_hemisphere=view_space_for_other_hemisphere,
    )

    return bf_left_boundaries, bf_right_boundaries


def get_average_template_projection(view_lookup_key: str = "flatmap_butterfly"):
    view_lookup_file = data_registry.get_view_lookup_file_path(view_lookup_key)
    surface_paths_file = data_registry.get_streamline_path("surface_paths_10_v3")

    proj_bf = ccfproj.Isocortex2dProjector(
        projection_file=view_lookup_file,
        surface_paths_file=surface_paths_file,
        # Specify that we want to project both hemispheres
        hemisphere="both",
        # The butterfly view doesn't contain space for the right hemisphere,
        # but the projector knows where to put the right hemisphere data so
        # the two hemispheres are adjacent if we specify that we're using the
        # butterfly flatmap
        view_space_for_other_hemisphere=view_lookup_key,
    )

    average_template_path = data_registry.get_atlas_files("average_template_10")
    template, _ = nrrd.read(average_template_path)
    bf_projection_max = proj_bf.project_volume(template)
    return bf_projection_max


def get_isocortex_3d_projector(
        view_lookup_file: str = "flatmap_butterfly",
        surface_paths_file: str = "surface_paths_10_v3",
        hemisphere: str = "both",
        view_space_for_other_hemisphere: str = "flatmap_butterfly",
        thickness_type: str = "normalized_layers",  # each layer will have the same thickness everywhere
) -> ccfproj.Isocortex3dProjector:
    view_lookup_file = data_registry.get_view_lookup_file_path(view_lookup_file)
    surface_paths_file = data_registry.get_streamline_path(surface_paths_file)
    layer_depth_file = data_registry.get_isocortex_metric_path("avg_layer_depths")
    streamline_layer_thickness_file_path = data_registry.get_isocortex_metric_path(
        "cortical_layers_10_v2"
    )

    layer_thicknesses = get_layer_depth_from_json(layer_depth_file)

    proj_butterfly_slab = ccfproj.Isocortex3dProjector(
        # Similar inputs as the 2d version...
        projection_file=view_lookup_file,
        surface_paths_file=surface_paths_file,
        hemisphere=hemisphere,
        view_space_for_other_hemisphere=view_space_for_other_hemisphere,
        # Additional information for thickness calculations
        thickness_type=thickness_type,  # each layer will have the same thickness everywhere
        layer_thicknesses=layer_thicknesses,
        streamline_layer_thickness_file=streamline_layer_thickness_file_path,
    )

    return proj_butterfly_slab


def get_avg_normalized_layers_butterfly():
    view_lookup_file: str = "flatmap_butterfly"
    proj_butterfly_slab = get_isocortex_3d_projector(view_lookup_file=view_lookup_file,
                                                     view_space_for_other_hemisphere=view_lookup_file)

    average_template_path = data_registry.get_atlas_files("average_template_10")
    template, _ = nrrd.read(average_template_path)
    bf_projection_max = proj_butterfly_slab.project_volume(template, thickness_type="unnormalized")
    return bf_projection_max


def get_coordinate_projector(view_lookup_file: str = "flatmap_butterfly",
                             surface_paths_file: str = "surface_paths_10_v3", ):
    view_lookup_file = data_registry.get_view_lookup_file_path(view_lookup_file)
    surface_paths_file = data_registry.get_streamline_path(surface_paths_file)
    closest_surface_voxel_reference_file = data_registry.get_streamline_path("closest_surface_voxel_lookup")
    layer_depth_file = data_registry.get_isocortex_metric_path("avg_layer_depths")
    streamline_layer_thickness_file_path = data_registry.get_isocortex_metric_path(
        "cortical_layers_10_v2"
    )

    layer_thicknesses = get_layer_depth_from_json(layer_depth_file)
    resolution = (10, 10, 10)
    proj = ccfproj.IsocortexCoordinateProjector(
        surface_paths_file=surface_paths_file,
        closest_surface_voxel_reference_file=closest_surface_voxel_reference_file,
        layer_thicknesses=layer_thicknesses,
        streamline_layer_thickness_file=streamline_layer_thickness_file_path,
        resolution=resolution,
        projection_file=view_lookup_file,
    )

    return proj


def get_isocrotex_entire_projector(resolution=(10, 10, 10)):
    proj = ccfproj.IsocortexEntireProjector(
        resolution=resolution,
        surface_paths_file=AVERAGE_TEMPLATE_PATH,
    )
    return proj
