"""Synthetic tests for DAVIR (no data download needed). Run: python -m pytest tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import davir  # noqa: E402

ZOOMS = (1.0, 1.0, 1.5)


def _phantom():
    """24 cylindrical bodies (L5 at the bottom) separated by 5 mm discs, each with an arch behind it."""
    pitch = [davir.GAP_MU.get(k, 17.0) for k in range(1, 25)]
    nz = int(sum(pitch) / ZOOMS[2]) + 20
    lab = np.zeros((70, 80, nz), np.uint8)
    x, y = np.meshgrid(np.arange(70), np.arange(80), indexing="ij")
    body = (x - 35) ** 2 + (y - 30) ** 2 <= 12 ** 2
    arch = (np.abs(x - 35) <= 4) & (y >= 42) & (y <= 60)
    pedicle = (np.abs(x - 35) <= 7) & (y > 39) & (y < 44)
    z = 10.0
    for k in range(1, 25):
        z0, z1 = int(z / ZOOMS[2]), int((z + pitch[k - 1] - 5.0) / ZOOMS[2])
        lab[:, :, z0:z1][body | pedicle | arch] = k
        z += pitch[k - 1]
    affine = np.diag(list(ZOOMS) + [1.0])
    return lab, affine


def _shift(lab):
    """The error seen on BDMAP_00000031: L1..T9 named one level too low."""
    out = lab.copy()
    for k in range(5, 10):
        out[lab == k] = k + 1
    return out


def _mean_dice(a, b):
    ds = [2 * ((a == k) & (b == k)).sum() / ((a == k).sum() + (b == k).sum()) for k in np.unique(b) if k]
    return float(np.mean(ds))


def test_clean_spine_is_unchanged():
    lab, affine = _phantom()
    out = davir.refine(lab, affine)
    assert _mean_dice(out, lab) > 0.999


def test_shifted_names_are_corrected():
    lab, affine = _phantom()
    log = {}
    out = davir.refine(_shift(lab), affine, log=log)
    assert log["mode"] == "full" and log["n_instances"] == 24
    assert _mean_dice(_shift(lab), lab) < 0.85
    assert _mean_dice(out, lab) > 0.99
