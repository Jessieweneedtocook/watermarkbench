from __future__ import annotations

from typing import Union
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_fn

from ._io import imread_rgb, PathLike


ArrayLike = Union[np.ndarray, PathLike]


def _as_rgb(img: ArrayLike) -> np.ndarray:
    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        return imread_rgb(img)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unexpected image shape: {arr.shape}")


def _to_gray01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arrf = arr.astype(np.float64) / 255.0
    else:
        arrf = arr.astype(np.float64)
        if arrf.max() > 1.5:
            arrf = np.clip(arrf, 0.0, 255.0) / 255.0
        else:
            arrf = np.clip(arrf, 0.0, 1.0)

    if arrf.ndim == 2:
        gray = arrf
    else:
        r, g, b = arrf[..., 0], arrf[..., 1], arrf[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.clip(gray, 0.0, 1.0).astype(np.float64)


def _match_shapes_center_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x):
        y0 = (x.shape[0] - h) // 2
        x0 = (x.shape[1] - w) // 2
        return x[y0:y0 + h, x0:x0 + w, :]

    return crop(a), crop(b)


def MSE(ref: ArrayLike, tst: ArrayLike) -> float:
    a = _to_gray01(_as_rgb(ref))
    b = _to_gray01(_as_rgb(tst))
    if a.shape != b.shape:
        ar = _as_rgb(ref); br = _as_rgb(tst)
        ar, br = _match_shapes_center_crop(ar, br)
        a = _to_gray01(ar); b = _to_gray01(br)
    return float(np.mean((a - b) ** 2))


def PSNR(ref: ArrayLike, tst: ArrayLike, max_val: float = 1.0) -> float:
    m = MSE(ref, tst)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / m))


def WPSNR(ref: ArrayLike, tst: ArrayLike, eps: float = 1e-6) -> float:
    ref_rgb = _as_rgb(ref)
    tst_rgb = _as_rgb(tst)
    if ref_rgb.shape != tst_rgb.shape:
        ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

    ref8 = (np.clip(_to_gray01(ref_rgb) * 255, 0, 255)).astype(np.uint8)
    blur = cv2.GaussianBlur(ref8, (7, 7), 1.5).astype(np.float64) / 255.0
    a = _to_gray01(ref_rgb)
    b = _to_gray01(tst_rgb)

    w = 1.0 / (1.0 + np.abs(a - blur) + eps)
    wmse = np.sum(w * (a - b) ** 2) / np.sum(w)
    if wmse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / wmse))


def SSIM(ref: ArrayLike, tst: ArrayLike) -> float:
    ref_rgb = _as_rgb(ref)
    tst_rgb = _as_rgb(tst)
    if ref_rgb.shape != tst_rgb.shape:
        ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

    a = _to_gray01(ref_rgb)
    b = _to_gray01(tst_rgb)
    return float(ssim_fn(a, b, data_range=1.0))


def JNDPassRate(ref: ArrayLike, tst: ArrayLike) -> float:
    ref_rgb = _as_rgb(ref)
    tst_rgb = _as_rgb(tst)
    if ref_rgb.shape != tst_rgb.shape:
        ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

    a = _to_gray01(ref_rgb)
    b = _to_gray01(tst_rgb)
    err = np.abs(a - b)

    k = 7
    mean = cv2.boxFilter(a.astype(np.float32), -1, (k, k), normalize=True).astype(np.float64)
    sqmean = cv2.boxFilter((a * a).astype(np.float32), -1, (k, k), normalize=True).astype(np.float64)
    std = np.sqrt(np.maximum(sqmean - mean ** 2, 0.0))

    base = 2 / 255.0
    alpha = 0.08
    beta = 0.6
    jnd = base + alpha * mean + beta * std

    pass_map = (err < jnd)
    return float(np.mean(pass_map))
