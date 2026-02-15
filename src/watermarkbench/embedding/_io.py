from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
import cv2

PathLike = Union[str, Path]

def imread_rgb(path: PathLike) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
