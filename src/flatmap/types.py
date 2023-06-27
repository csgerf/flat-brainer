from typing import List, Optional, Tuple, Union, Dict
from os import PathLike

import numpy.typing as npt
# import SimpleITK as sitk
# import itk


OnePath = Union[str, PathLike]
# ImakeLike = Union[sitk.Image, itk.Image, np.ndarray]
ImakeLike = npt.NDArray
ShapeLike = Union[Tuple[int, int, int], List[int], npt.NDArray]
SpacingLike = Union[Tuple[float, float, float], List[float], npt.NDArray]


