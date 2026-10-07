"""Synthetic stress tests for davir.py.

A clean label map is used as pseudo ground truth; known error types (level shift, merged and
fragmented labels, cropped field of view, flipped orientation) are injected and must be undone.

    python tools/robustness_tests.py --gt <clean case folder> --ct <case>/ct.nii.gz --out outputs/stress
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import davir as pv  # noqa: E402

INV = {v: k for k, v in pv.NAMES.items()}


def mean_dice(pred, gt):
    ds = []
    for k in np.unique(gt):
        if k == 0:
            continue
        a, b = pred == k, gt == k
        ds.append(2 * (a & b).sum() / (a.sum() + b.sum()))
    return float(np.mean(ds))


def save(img_like, data, path, affine=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = img_like.header.copy()
    hdr.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, img_like.affine if affine is None else affine, hdr), path)


def corrupt_shift(gt):
    """BDMAP_00000031-like: L1..T9 named one level too low, T8 absorbs a second body, L1 disappears."""
    x = gt.copy()
    for k in [INV["L1"], INV["T12"], INV["T11"], INV["T10"], INV["T9"]]:
        x[gt == k] = k + 1
    return x


def corrupt_merge(gt):
    """T6 body/arch labelled T5 (one label spanning two vertebrae)."""
    x = gt.copy()
    x[gt == INV["T6"]] = INV["T5"]
    return x


def corrupt_fragments(gt, zooms, rng):
    """Islands of wrong labels (radius 3-5 mm) inside vertebrae, +-1..3 levels off."""
    x = gt.copy()
    idx = np.argwhere(gt > 0)
    for _ in range(40):
        c = idx[rng.integers(len(idx))]
        r = rng.uniform(3, 5)
        rad = np.ceil(r / zooms).astype(int)
        sl = tuple(slice(max(ci - ri, 0), ci + ri + 1) for ci, ri in zip(c, rad))
        g = np.meshgrid(*[np.arange(s.start, s.stop) for s in sl], indexing="ij")
        ball = sum(((gi - ci) * z) ** 2 for gi, ci, z in zip(g, c, zooms)) <= r * r
        sub = x[sl]
        k = int(gt[tuple(c)])
        new = int(np.clip(k + rng.choice([-3, -2, -1, 1, 2, 3]), 1, 24))
        sub[ball & (gt[sl] > 0)] = new
    return x


def run(name, pred, ct_img, gt, ref_img, out_root, affine=None, use_ct=True):
    d = os.path.join(out_root, name)
    save(ref_img, pred.astype(np.uint8), os.path.join(d, "pred", "case", "combined_labels.nii.gz"), affine)
    ctp = None
    if use_ct:
        ctp = os.path.join(d, "ct", "case", "ct.nii.gz")
        save(ct_img, np.asarray(ct_img.dataobj).astype(np.int16), ctp, affine)
    log = pv.process_case(os.path.join(d, "pred", "case"), ctp, os.path.join(d, "out", "case"), p=pv.PARAMS,
                          qc_path=os.path.join(d, "qc.json"))
    out = np.asarray(nib.load(os.path.join(d, "out", "case", "combined_labels.nii.gz")).dataobj)
    return dict(test=name, dice_before=round(mean_dice(pred, gt), 4), dice_after=round(mean_dice(out, gt), 4),
                instances=log.get("n_instances"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="case folder with a clean combined_labels.nii.gz")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--out", default="outputs/stress")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    gimg = nib.load(os.path.join(a.gt, "combined_labels.nii.gz"))
    gt = np.asarray(gimg.dataobj).astype(np.uint8)
    cimg = nib.load(a.ct)
    zooms = np.array(gimg.header.get_zooms()[:3], float)
    res = []
    res.append(run("identity", gt, cimg, gt, gimg, a.out))
    res.append(run("shift_031_like", corrupt_shift(gt), cimg, gt, gimg, a.out))
    res.append(run("merge_T5_T6", corrupt_merge(gt), cimg, gt, gimg, a.out))
    res.append(run("fragments", corrupt_fragments(gt, zooms, rng), cimg, gt, gimg, a.out))
    res.append(run("shift_no_ct", corrupt_shift(gt), cimg, gt, gimg, a.out, use_ct=False))
    # field-of-view crops (cut through the middle of a vertebra) -- native axis 2 is S-I for this data
    zr = lambda lv: np.argwhere(gt == INV[lv])[:, 2]
    for name, top, bot in (("fov_lumbar_T11_L5", "T11", "L5"), ("fov_thoracic_T2_T11", "T2", "T11")):
        z0 = int(np.median(zr(bot)))  # cut through the middle of the boundary vertebrae
        z1 = int(np.median(zr(top)))
        lo, hi = min(z0, z1), max(z0, z1) + 1
        gi = gimg.slicer[:, :, lo:hi]
        ci = cimg.slicer[:, :, lo:hi]
        g2 = np.asarray(gi.dataobj).astype(np.uint8)
        ci_img = nib.Nifti1Image(np.asarray(ci.dataobj).astype(np.int16), ci.affine, ci.header)
        r = run(name, g2, ci_img, g2, gi, a.out)
        r2 = run(name + "_shifted", corrupt_shift(g2), ci_img, g2, gi, a.out)
        res += [r, r2]
    # orientation: flip x and z (data + affine consistently)
    shape = np.array(gt.shape)
    fl = np.eye(4)
    fl[0, 0], fl[2, 2] = -1, -1
    fl[0, 3], fl[2, 3] = shape[0] - 1, shape[2] - 1
    aff = gimg.affine @ fl
    g3 = gt[::-1, :, ::-1].copy()
    c3 = nib.Nifti1Image(np.asarray(cimg.dataobj)[::-1, :, ::-1].astype(np.int16).copy(), aff, cimg.header)
    res.append(run("orientation_flip_xz_shifted", corrupt_shift(g3), c3, g3, gimg, a.out, affine=aff))
    print(json.dumps(res, indent=1))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
