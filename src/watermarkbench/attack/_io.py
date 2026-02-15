from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import torch
from PIL import Image


PathLike = Union[str, Path]


def _load_image(path: PathLike) -> Tuple[torch.Tensor, Image.Image, Path]:
    p = Path(path)
    pil = Image.open(p)
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")

    arr = np.array(pil)
    if arr.ndim == 2:  
        x = torch.from_numpy(arr).unsqueeze(0)  
    else:             
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  

    if x.dtype != torch.uint8:
        x = x.to(torch.uint8)

    return x, pil, p


def _tensor_to_pil(x: torch.Tensor, mode_hint: str) -> Image.Image:
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")

    if torch.is_floating_point(x):
        y = x.detach().cpu()
        if float(y.max()) <= 1.5:
            y = (y.clamp(0, 1) * 255.0).round()
        else:
            y = y.clamp(0, 255).round()
        y = y.to(torch.uint8)
    else:
        y = x.detach().cpu().to(torch.uint8)

    if y.shape[0] == 1:
        arr = y[0].numpy()
        return Image.fromarray(arr, mode="L")
    else:
        arr = y.permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")


def _format_param(param) -> str:
    if isinstance(param, float):
        s = f"{param:.6g}"
        return s.replace(".", "p")
    return str(param).replace(".", "p")


def _output_path(
    input_path: Path,
    attack_name: str,
    strength,
    output_dir: Optional[PathLike] = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    ext = input_path.suffix  
    strength_s = _format_param(strength)
    out_name = f"{stem}_{attack_name}_{strength_s}{ext}"
    return out_dir / out_name

