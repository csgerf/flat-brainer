from typing import List, Tuple, Dict
from dataclasses import dataclass

import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class MarkerPoint:
    x: float
    y: float
    z: float


MarkerList = List[MarkerPoint]


def get_xml_root(xml_file_path: str) -> ET.Element:
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    return root


def read_points(root) -> MarkerList:
    points_list = []
    for child in root:
        if child.tag == "point" or child.tag == "{https://www.mbfbioscience.com/filespecification}point":
            point = MarkerPoint(x=float(child.attrib["x"]), y=float(child.attrib["y"]), z=float(child.attrib["z"]))
            points_list.append(point)

    return points_list


def get_marker_dict(root) -> Dict[str, MarkerList]:
    marker_dict = {}
    for child in root:
        if child.tag == "marker" or child.tag == "{https://www.mbfbioscience.com/filespecification}marker":
            points = read_points(child)
            marker_dict[child.attrib["name"]] = points
    return marker_dict


def get_marker_points_from_xml(xml_file_path: str) -> Dict[str, MarkerList]:
    root = get_xml_root(xml_file_path)
    marker_dict = get_marker_dict(root)
    return marker_dict
