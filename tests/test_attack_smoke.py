import os
from pathlib import Path

import numpy as np
from PIL import Image

import watermarkbench as wb


def _make_test_image(path: Path):
    # small RGB image (fast test)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = [255, 0, 0]  # red square
    Image.fromarray(img).save(path)


def test_attack_rotate_smoke(tmp_path):
    # Create input image
    input_path = tmp_path / "test.png"
    _make_test_image(input_path)

    # Call attack
    out_path = wb.attack.rotate(input_path, 15)

    # 1) Returned path exists
    assert os.path.exists(out_path)

    # 2) Filename matches required pattern
    out_file = Path(out_path)
    assert out_file.name == "test_rotate_15.png"

    # 3) Output is a valid image
    img = Image.open(out_path)
    assert img.size[0] > 0
    assert img.size[1] > 0
