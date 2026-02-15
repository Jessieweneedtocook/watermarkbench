from __future__ import annotations

from typing import Iterable, Union, Sequence
import numpy as np

Bits = Union[str, Sequence[int], np.ndarray]

def _to_bits(x: Bits) -> np.ndarray:
    if isinstance(x, str):
        arr = np.fromiter((1 if c == "1" else 0 for c in x.strip()), dtype=np.uint8)
        return arr
    arr = np.asarray(x)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr != 0).astype(np.uint8)
    raise ValueError(f"Unsupported bits type: {type(x)} / dtype={arr.dtype}")

def BER(groundtruth: Bits, extracted: Bits) -> float:
    gt = _to_bits(groundtruth)
    ex = _to_bits(extracted)
    if gt.shape != ex.shape:
        raise ValueError(f"Shape mismatch: groundtruth {gt.shape} vs extracted {ex.shape}")
    return float(np.mean(gt != ex))

