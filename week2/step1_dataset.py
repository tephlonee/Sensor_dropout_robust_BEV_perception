"""
Week 2 - Step 1: Dataset, training targets and sensor-failure simulation.

What this step teaches
----------------------
1. How the Week 1 cache becomes (input, target) pairs for a CenterNet-style
   detector: a heatmap that peaks at every object centre, plus per-object
   regression targets (sub-cell offset, size, heading).
2. How we SIMULATE sensor failures on the cached BEV - the core of the
   missing-modality thesis:
     no_radar     radar channels zeroed, and the model is TOLD (mask = 0).
                  Real systems know when a sensor is down (no messages arrive).
     noisy_radar  radar is present but degraded: real detections dropped,
                  velocities jittered, ghost detections added. The model is
                  NOT told - corrupted data usually looks valid to the system.
     no_camera    camera channels zeroed (mask = 0).
3. Training-time augmentation: left-right flip, and modality DROPOUT (randomly
   removing radar during training so the model can't rely on it completely).

Note on "camera only": our camera channels are LiDAR points painted with image
scores (PointPainting), so they need LiDAR geometry. A true camera-only
fallback needs a camera-to-BEV network (e.g. Lift-Splat). Here the
"without radar" condition is LiDAR + camera.

Run:  python step1_dataset.py      (prints stats and saves a figure of targets + corruptions)
"""

import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (CACHE_DIR, MAX_OBJECTS, MIN_RADIUS, MODALITIES, OUT_DIR,
                    REG_DIMS, V_SCALE, ensure_dir)


# ---------------------------------------------------------------------------
# 1. Grid geometry (mirrors week1/step4_bev.py: row 0 = forward, col 0 = left)
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self, spec):
        self.x_min, self.x_max = spec["x_range"]
        self.y_min, self.y_max = spec["y_range"]
        self.res = spec["res"]
        self.H = round((self.x_max - self.x_min) / self.res)
        self.W = round((self.y_max - self.y_min) / self.res)

    def to_pixel(self, x, y):
        """Metric ego (x, y) -> continuous (row, col). Cell i spans [i, i+1)."""
        return (self.x_max - x) / self.res, (self.y_max - y) / self.res

    def to_metric(self, row, col):
        return self.x_max - row * self.res, self.y_max - col * self.res


def load_index(cache_dir=CACHE_DIR):
    path = os.path.join(cache_dir, "index.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run week1/step6_build_dataset.py first, "
                                f"or set BEV_CACHE to the cache folder.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Targets: Gaussian heatmaps + regression values (CenterNet / CenterPoint)
# ---------------------------------------------------------------------------
def gaussian_radius(height, width, min_overlap=0.1):
    """CenterNet's radius: how far a predicted centre can be off while a box of
    this size still overlaps the true box by `min_overlap` IoU. Sizes in cells."""
    a1, b1 = 1, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap, row, col, radius):
    """Splat a Gaussian with peak EXACTLY 1.0 at integer cell (row, col).
    The focal loss treats cells equal to 1 as positives, everything else as
    negatives down-weighted by how close to a centre they are."""
    diameter = 2 * radius + 1
    sigma = diameter / 6
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32)
    H, W = heatmap.shape
    top, bottom = min(row, radius), min(H - row, radius + 1)
    left, right = min(col, radius), min(W - col, radius + 1)
    region = heatmap[row - top:row + bottom, col - left:col + right]
    np.maximum(region, g[radius - top:radius + bottom, radius - left:radius + right], out=region)


def build_targets(boxes, grid, n_classes):
    """boxes (K, 8): class_id, x, y, z, w, l, h, yaw (ego frame, from Week 1).

    Returns
      heatmap (n_classes, H, W)  - what the network should light up
      ind     (MAX,)             - flat cell index of each object centre
      reg_mask(MAX,)             - 1 for real objects, 0 for padding
      reg     (MAX, 6)           - off_row, off_col, log w, log l, sin yaw, cos yaw
      gt      (MAX, 4)           - class, x, y, yaw for evaluation (class -1 = padding)
    """
    heatmap = np.zeros((n_classes, grid.H, grid.W), dtype=np.float32)
    ind = np.zeros(MAX_OBJECTS, dtype=np.int64)
    reg_mask = np.zeros(MAX_OBJECTS, dtype=np.float32)
    reg = np.zeros((MAX_OBJECTS, len(REG_DIMS)), dtype=np.float32)
    gt = np.full((MAX_OBJECTS, 4), -1.0, dtype=np.float32)

    k = 0
    for cls_id, x, y, _z, w, l, _h, yaw in boxes:
        row_f, col_f = grid.to_pixel(x, y)
        r, c = int(math.floor(row_f)), int(math.floor(col_f))
        if not (0 <= r < grid.H and 0 <= c < grid.W) or k >= MAX_OBJECTS:
            continue
        radius = max(MIN_RADIUS, int(gaussian_radius(l / grid.res, w / grid.res)))
        draw_gaussian(heatmap[int(cls_id)], r, c, radius)
        ind[k] = r * grid.W + c
        reg_mask[k] = 1.0
        reg[k] = [row_f - r, col_f - c, math.log(max(w, 0.1)), math.log(max(l, 0.1)),
                  math.sin(yaw), math.cos(yaw)]
        gt[k] = [cls_id, x, y, yaw]
        k += 1
    return heatmap, ind, reg_mask, reg, gt


# ---------------------------------------------------------------------------
# 3. Sensor-failure simulation (works on the cached BEV, channels-first numpy)
# ---------------------------------------------------------------------------
def corrupt_radar(bev, radar_slice, severity, rng):
    """Degrade radar like a dirty / interfered / failing sensor.
    severity 0 = untouched, 1 = heavy. Three effects scale with severity:
      * drop    : up to 50 % of real radar detections disappear
      * jitter  : velocity noise up to 3 m/s std
      * ghosts  : up to as many fake detections as real ones (clutter)
    Radar channel order (from Week 1): occ, rcs, vx, vy.
    """
    if severity <= 0:
        return bev
    r = bev[radar_slice]                          # view: edits write into bev
    occ = r[0] > 0
    n_real = int(occ.sum())

    drop = occ & (rng.random(occ.shape) < 0.5 * severity)
    r[:, drop] = 0.0

    keep = occ & ~drop
    r[2:4, keep] += rng.normal(0, 3.0 * severity / V_SCALE, size=(2, int(keep.sum())))

    n_ghost = int(round(severity * max(n_real, 20)))
    H, W = occ.shape
    flat = rng.choice(H * W, size=min(n_ghost, H * W), replace=False)
    gr, gc = np.unravel_index(flat, (H, W))
    r[0, gr, gc] = 1.0
    r[1, gr, gc] = rng.uniform(0.05, 0.6, size=len(flat))
    r[2:4, gr, gc] = rng.normal(0, 5.0 / V_SCALE, size=(2, len(flat)))
    return bev


def apply_condition(bev, mask, slices, condition, severity, rng):
    """Apply one test condition in place. mask is (3,) presence flags."""
    if condition == "all":
        pass
    elif condition == "no_radar":
        bev[slices["radar"]] = 0.0
        mask[MODALITIES.index("radar")] = 0.0
    elif condition == "no_camera":
        bev[slices["camera"]] = 0.0
        mask[MODALITIES.index("camera")] = 0.0
    elif condition == "noisy_radar":
        corrupt_radar(bev, slices["radar"], severity, rng)   # mask stays 1: undetected
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return bev, mask


# ---------------------------------------------------------------------------
# 4. The Dataset
# ---------------------------------------------------------------------------
class BEVDataset(Dataset):
    """Reads the Week 1 cache.

    train=True : random flip + optional radar dropout / noise augmentation.
    train=False: a fixed test `condition` (deterministic per sample, so every
                 model sees exactly the same corrupted inputs).
    """

    def __init__(self, split, train=False, condition="all", severity=1.0,
                 radar_dropout=0.0, radar_noise_p=0.0, cache_dir=CACHE_DIR, limit=None):
        index = load_index(cache_dir)
        self.cache_dir = cache_dir
        self.items = [s for s in index["samples"] if s["split"] == split][:limit]
        if not self.items:
            raise ValueError(f"no samples with split={split!r} in {cache_dir}")
        self.channels = index["channels"]
        self.classes = index["classes"]
        self.slices = {k: slice(*v) for k, v in index["modality_slices"].items()}
        self.grid = Grid(index["grid"])
        self.train, self.condition, self.severity = train, condition, severity
        self.radar_dropout, self.radar_noise_p = radar_dropout, radar_noise_p
        self.vy_channel = self.channels.index("radar_vy")

    def __len__(self):
        return len(self.items)

    def _flip_left_right(self, bev, boxes):
        """Mirror across the car's forward axis: y -> -y. Velocities and headings mirror too."""
        bev = bev[:, :, ::-1].copy()
        bev[self.vy_channel] *= -1
        boxes = boxes.copy()
        boxes[:, 2] *= -1          # y
        boxes[:, 7] *= -1          # yaw
        return bev, boxes

    def __getitem__(self, i):
        item = self.items[i]
        data = np.load(os.path.join(self.cache_dir, f"{item['token']}.npz"))
        bev = data["bev"].astype(np.float32)
        boxes = data["boxes"].astype(np.float32).reshape(-1, 8)
        mask = np.ones(len(MODALITIES), dtype=np.float32)

        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                bev, boxes = self._flip_left_right(bev, boxes)
            if rng.random() < self.radar_dropout:
                apply_condition(bev, mask, self.slices, "no_radar", 0, rng)
            elif rng.random() < self.radar_noise_p:
                apply_condition(bev, mask, self.slices, "noisy_radar", rng.uniform(0.2, 1.0), rng)
        else:
            rng = np.random.default_rng(i)          # same corruption for every model
            apply_condition(bev, mask, self.slices, self.condition, self.severity, rng)

        heatmap, ind, reg_mask, reg, gt = build_targets(boxes, self.grid, len(self.classes))
        return {
            "bev": torch.from_numpy(bev), "mask": torch.from_numpy(mask),
            "heatmap": torch.from_numpy(heatmap), "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask), "reg": torch.from_numpy(reg),
            "gt": torch.from_numpy(gt), "idx": i,
        }


# ---------------------------------------------------------------------------
# 5. Look at it
# ---------------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt

    train = BEVDataset("mini_train", train=True)
    val = BEVDataset("mini_val")
    print(f"Samples: train {len(train)}, val {len(val)}")
    print(f"Input channels ({len(train.channels)}): {train.channels}")
    print(f"Modality slices: { {k: (s.start, s.stop) for k, s in train.slices.items()} }")

    counts = np.zeros(len(train.classes), int)
    for j in range(len(val)):
        gt = val[j]["gt"].numpy()
        for c in range(len(counts)):
            counts[c] += int((gt[:, 0] == c).sum())
    print(f"Val objects in grid: {dict(zip(val.classes, counts.tolist()))}")

    j = len(val) // 2
    clean = val[j]
    variants = [("all", 0), ("no_radar", 0), ("noisy_radar", 0.5), ("noisy_radar", 1.0)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    ax = axes[0, 0]
    ax.imshow(clean["bev"][val.channels.index("lidar_max_z")], cmap="gray", vmin=0, vmax=1)
    ax.set_title("LiDAR height (input)")
    ax = axes[0, 1]
    ax.imshow(clean["heatmap"].max(0).values, cmap="inferno", vmin=0, vmax=1)
    ax.set_title("Target heatmap (car + ped)")
    ax = axes[0, 2]
    hm = clean["heatmap"].max(0).values.numpy()
    ax.imshow(clean["bev"][val.channels.index("lidar_max_z")], cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_less(hm, 0.05), cmap="autumn", alpha=0.8)
    ax.set_title("Targets on top of LiDAR")
    axes[0, 3].axis("off")
    axes[0, 3].text(0.0, 0.5, "Bottom row: the radar occupancy channel\n"
                    "under each test condition.\n\n"
                    "no_radar: everything gone, model told (mask=0)\n"
                    "noisy_radar: dropped + ghost detections,\n"
                    "model NOT told", fontsize=12, va="center")

    occ = val.channels.index("radar_occ")
    for ax, (cond, sev) in zip(axes[1], variants):
        ds = BEVDataset("mini_val", condition=cond, severity=sev)
        s = ds[j]
        ys, xs = np.nonzero(s["bev"][occ].numpy())
        ax.imshow(np.zeros((ds.grid.H, ds.grid.W)), cmap="gray", vmin=0, vmax=1)
        ax.scatter(xs, ys, s=3, c="yellow")
        ax.set_title(f"radar occupancy: {cond}" + (f" (sev {sev})" if cond == "noisy_radar" else "")
                     + f"\nmask={s['mask'].numpy().tolist()}  points={len(xs)}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    out = os.path.join(ensure_dir(OUT_DIR), "step1_targets_and_conditions.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"Saved figure -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
