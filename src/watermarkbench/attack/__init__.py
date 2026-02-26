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
            y = core_fn(x_chw_u8, strength)
            out_path = _output_path(p, name, strength, output_dir=output_dir)
            out_pil = _tensor_to_pil(y, mode_hint=pil.mode)
            out_pil.save(out_path)
            return str(out_path)

        AI_ATTACKS = {"remove_ai", "replace_ai", "create_ai"}

        if name in AI_ATTACKS:
            def _call_ai(input_image: PathLike, output_dir: Optional[PathLike] = None, **kwargs) -> str:
                x_chw_u8, pil, p = _load_image(input_image)
                y = core_fn(x_chw_u8, None, **kwargs)
                tag = kwargs.get("tag", "ai")
                out_path = _output_path(p, name, tag, output_dir=output_dir)
                out_pil = _tensor_to_pil(y, mode_hint=pil.mode)
                out_pil.save(out_path)
                return str(out_path)

            return _call_ai

        return _call


attack = _AttackNamespace()

__all__ = ["attack"]


