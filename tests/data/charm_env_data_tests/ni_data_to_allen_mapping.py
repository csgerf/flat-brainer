import os
from dotenv import load_dotenv
import pprint
import nrrd

from src.core.utils import fs

# Load environment variables from .env file
if load_dotenv() is False:
    print("Failed to load .env file")


ALLEN_DATA_DIR = os.getenv("ALLEN_ATLAS_FILES")
CORONAL_DATA_DIR = os.getenv("MBF_ALLEN_CORONAL_FILES")


def main():
    if load_dotenv() is False:
        print("Failed to load .env file")
        return

    example_case_file_dir = os.getenv("CASE_FILE_DATA_DIR")
    if example_case_file_dir is None:
        print("Failed to load CASE_FILE_DATA_DIR from .env file")
        return

    example_case_file_dir = os.path.join(example_case_file_dir, "working")

    case_files = fs.find_in_dir_with_ext(example_case_file_dir, ".xml")
    if len(case_files) == 0:
        print("No case files found in directory")
        return

    case_file_path = case_files[0]
    print(f"Reading case file: {os.path.basename(case_file_path)}")

    coronal_annotation_path = os.path.join(CORONAL_DATA_DIR, "AllenCoronal_10um_anatomy.nrrd")
    allen_annotation_path = os.path.join(ALLEN_DATA_DIR, "annotation_10.nrrd")

    if not os.path.exists(coronal_annotation_path):
        print(f"Failed to find coronal annotation file: {coronal_annotation_path}")
        return

    if not os.path.exists(allen_annotation_path):
        print(f"Failed to find allen annotation file: {allen_annotation_path}")
        return

    coronal_data, coronal_header = nrrd.read(coronal_annotation_path)
    allen_data, allen_header = nrrd.read(allen_annotation_path)

    print(f"Coronal data shape: {coronal_data.shape}")
    print(f"Allen data shape: {allen_data.shape}")

    print("Coronal data type: ", coronal_header)
    print("Allen data type: ", allen_header)


if __name__ == "__main__":
    main()