from __future__ import annotations
from dataclasses import dataclass
from .metrics import BER

@dataclass(frozen=True)
class _ExtractingNamespace:
    BER = staticmethod(BER)

extracting = _ExtractingNamespace()
__all__ = ["extracting"]
