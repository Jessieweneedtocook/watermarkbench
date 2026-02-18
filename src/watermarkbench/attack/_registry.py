from __future__ import annotations

from typing import Callable, Dict

from . import _core

ATTACKS: Dict[str, Callable] = {
    "rotate": _core.rotate_tensor,
    "crop": _core.crop,
    "scaled": _core.scaled,
    "flipping": _core.flipping,
    "resized": _core.resized,
    "jpeg": _core.jpeg_compression,
    "jpeg2000": _core.jpeg2000_compression,
    "jpegai": _core.jpegai_compression,
    "jpegxl": _core.jpegxl_compression,
    "jpegxs": _core.jpegxs_compression,
    "gaussian_noise": _core.gaussian_noise,
    "speckle_noise": _core.speckle_noise,
    "blurring": _core.blurring,
    "brightness": _core.brightness,
    "sharpness": _core.sharpness,
    "median_filtering": _core.median_filtering,
}


