from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET


def read_atlas_coordinates_from_csv(file_path: str = "") -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    headers = [df.columns.get_loc("Experiment"), df.columns.get_loc("Atlas")]
    df = pd.read_csv(file_path, header=headers)
    print(headers)

    return df


def get_coordinates_from_df(df: pd.DataFrame, scale_value: float = 1.0) -> List[Tuple[float, float, float]]:
    coords = []
    for index, row in df.iterrows():
        if row["id"][0] == 0:
            continue
        x_point = round(row["z"][1] * scale_value)
        y_point = round(row["y"][1] * scale_value)
        z_point = round(row["x"][1] * scale_value)
        coords.append((x_point, y_point, z_point))

    return coords


def get_xml_root(xml_file_path: str) -> ET.Element:
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    return root


def get_coordinates_from_csv(file_path: str = "") -> List[Tuple[float, float, float]]:
    df = read_atlas_coordinates_from_csv(file_path)
    coords = get_coordinates_from_df(df)
    return coords


def read_points(root) -> List[Tuple[float, float, float]]:
    points_list = []
    for child in root:
        if child.tag == "point" or child.tag == "{https://www.mbfbioscience.com/filespecification}point":
            x = float(child.attrib["x"])
            y = float(child.attrib["y"])
            z = float(child.attrib["z"])
            point = (z, y, x)
            points_list.append(point)

    return points_list


def get_marker_dict(root) -> Dict[str, np.ndarray]:
    marker_dict = {}
    for child in root:
        if child.tag == "marker" or child.tag == "{https://www.mbfbioscience.com/filespecification}marker":
            points = read_points(child)
            points = np.array(points)
            marker_dict[child.attrib["name"]] = points
    return marker_dict


def get_marker_points_from_xml(xml_file_path: str) -> Dict[str, np.ndarray]:
    root = get_xml_root(xml_file_path)
    marker_dict = get_marker_dict(root)
    return marker_dict
