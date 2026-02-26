from __future__ import annotations

import os
from pathlib import Path
import requests


SAM_URLS = {
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def _default_weights_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "watermarkbench" / "weights"


def ensure_sam_checkpoint(
    filename: str = "sam_vit_h_4b8939.pth",
    *,
    weights_dir: str | Path | None = None,
) -> str:

    if filename not in SAM_URLS:
        raise ValueError(
            f"Unknown SAM checkpoint '{filename}'. Known: {list(SAM_URLS)}"
        )

    wdir = Path(weights_dir) if weights_dir is not None else _default_weights_dir()
    wdir.mkdir(parents=True, exist_ok=True)
    out = wdir / filename

    if out.exists() and out.stat().st_size > 0:
        return str(out)

    url = SAM_URLS[filename]
    tmp = out.with_suffix(out.suffix + ".tmp")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(out)

    return str(out)
