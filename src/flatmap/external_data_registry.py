# mypy: disable-error-code="misc"
import os
from pathlib import Path

from dotenv import load_dotenv  


# Load environment variables from .env file
load_dotenv()


enviornment_data_dir = os.getenv("ALLEN_CCF_STREAMLINE_DATA_DIR")

if enviornment_data_dir is not None and isinstance(enviornment_data_dir, str):
    DEFAULT_DATA_DIRECTORY = Path(enviornment_data_dir)
else:
    DEFAULT_DATA_DIRECTORY = Path("C:\\dev\\working\\flat-map\\flatmap\\data")


# STREAMLINE_DOWNLOAD_URL = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/streamlines/" 
# ISOCORTEX_METRIC_DOWNLOAD_URL = "https://download.alleninstitute.org/informatihttps://mypy.readthedocs.io/en/latest/_refs.html#code-misccs-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/cortical_metrics/" 
# ATLAS_FILE_DOWNLOAD_URL_BASE = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/master_updated/" 
# VIEW_LOOKUP_DOWNLOAD_URL_BASE = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/view_lookup/" 


ATLAS_FILES_2D = {
    "back": "back.nrrd",
    "bottom": "bottom.nrrd",
    "flatmap_butterfly": "flatmap_butterfly.nrrd",
    "flatmap_dorsal": "flatmap_dorsal.nrrd",
    "front": "front.nrrd",
    "medial": "medial.nrrd",  
    "rotated": "rotated.nrrd",
    "side": "side.nrrd",
    "top": "top.nrrd",
    "labelDescription": "labelDescription_ITKSNAPColor.txt", 
} 

VIEW_LOOKUP_FILES = {  
    "back": "back.h5",  
    "bottom": "bottom.h5",  
    "flatmap_butterfly": "flatmap_butterfly.h5",  
    "flatmap_dorsal": "flatmap_dorsal.h5",  
    "front": "front.h5",  
    "medial": "medial.h5",  
    "rotated": "rotated.h5",  
    "side": "side.h5",  
    "top": "top.h5",  
}  


STREAMLINE_FILES = {
    "surface_paths_10_v3": "surface_paths_10_v3.h5",
    "closest_surface_voxel_lookup": "closest_surface_voxel_lookup.h5",  
}


ISOCORTEX_METRIC_FILES = {  
    "avg_layer_depths": "avg_layer_depths.json",  
    "cortical_layers_10_v2": "cortical_layers_10_v2.h5",  
}  


def get_other_file_path(file_name: str) -> str:
    return os.path.join(DEFAULT_DATA_DIRECTORY, file_name)


def get_streamline_path(streamline_name: str) -> str:
    return os.path.join(DEFAULT_DATA_DIRECTORY, STREAMLINE_FILES[streamline_name])


def get_isocortex_metric_path(metric_name: str) -> str:
    return os.path.join(DEFAULT_DATA_DIRECTORY, ISOCORTEX_METRIC_FILES[metric_name])


def get_atlas_file_path(atlas_name: str) -> str:
    return os.path.join(DEFAULT_DATA_DIRECTORY, ATLAS_FILES_2D[atlas_name])


def get_view_lookup_file_path(view_name: str) -> str:
    return os.path.join(DEFAULT_DATA_DIRECTORY, VIEW_LOOKUP_FILES[view_name])
