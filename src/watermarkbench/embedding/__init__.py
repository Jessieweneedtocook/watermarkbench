from __future__ import annotations
from dataclasses import dataclass
from .metrics import MSE, PSNR, WPSNR, SSIM, JNDPassRate

@dataclass(frozen=True)
class _EmbeddingNamespace:
    MSE = staticmethod(MSE)
    PSNR = staticmethod(PSNR)
    WPSNR = staticmethod(WPSNR)
    SSIM = staticmethod(SSIM)
    JNDPassRate = staticmethod(JNDPassRate)

embedding = _EmbeddingNamespace()
__all__ = ["embedding"]
