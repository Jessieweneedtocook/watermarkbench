from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

import torch

from ._io import _load_image, _tensor_to_pil, _output_path, PathLike
from ._registry import ATTACKS


@dataclass(frozen=True)
class _AttackNamespace:
    """
    Usage:
      wb.attack.rotate("img.png", 15, output_dir="out/")
    """

    def list(self):
        return sorted(ATTACKS.keys())

    def __getattr__(self, name: str):
        if name not in ATTACKS:
            raise AttributeError(f"No such attack '{name}'. Available: {self.list()}")

        core_fn = ATTACKS[name]

        def _call(input_image: PathLike, strength, output_dir: Optional[PathLike] = None) -> str:
            x_chw_u8, pil, p = _load_image(input_image)

            # core expects torch tensor; attacks.py handles layout/range preservation.
            # Provide CHW uint8; it will be preserved by _apply_attack_preserve in your core.
            y = core_fn(x_chw_u8, strength)

            # Convert to PIL and save with required naming pattern
            out_path = _output_path(p, name, strength, output_dir=output_dir)
            out_pil = _tensor_to_pil(y, mode_hint=pil.mode)
            out_pil.save(out_path)
            return str(out_path)

        return _call


attack = _AttackNamespace()

__all__ = ["attack"]
