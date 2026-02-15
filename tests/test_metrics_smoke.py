import numpy as np
import watermarkbench as wb

def test_ber():
    assert wb.extracting.BER("0101", "0101") == 0.0
    assert wb.extracting.BER("0101", "1101") == 0.25

def test_ssim_runs():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.zeros((64, 64, 3), dtype=np.uint8)
    assert wb.embedding.SSIM(a, b) == 1.0
