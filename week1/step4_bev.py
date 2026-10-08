"""
Week 1 - Step 4: Rasterise the point clouds into bird's-eye-view (BEV) grids,
and build the ground-truth heatmaps the detector will learn in Week 2.

Why BEV
-------
A CNN needs a regular grid, and point clouds are unordered sets. Looking from
above, the road is roughly flat and objects rarely overlap, so a 2D top-down
grid keeps almost everything a detector needs. LiDAR, radar and (Step 5)
camera information all land in the SAME grid, which makes fusion as simple
as stacking channels. Volvo's thesis detectors work in BEV space for the
same reason.

Grid convention (used everywhere from now on)
---------------------------------------------
* ego frame, 102.4 m x 102.4 m centred on the car, 0.4 m cells -> 256 x 256
* row 0 = furthest FORWARD, col 0 = furthest LEFT, so imshow() shows
  forward = up, left = left.

Channels
--------
LiDAR (4):  occupancy, max height, point density, mean intensity
Radar (4):  occupancy, max RCS, mean vx, mean vy (ego-motion-compensated)
Target (2): Gaussian heatmaps at the centres of cars and pedestrians

Run:  python step4_bev.py
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from nusc_utils import LIDAR_CHANNEL, ensure_out_dir, get_nusc
from step3_multisweep import load_lidar_multisweep, load_radar_multisweep

SCENE_INDEX = 0
SAMPLE_INDEX = 10

# Normalisation constants - keep every channel roughly in [0, 1] (or [-1, 1]).
Z_MIN, Z_MAX = -1.0, 4.0           # height band kept (m, ego frame; ground ~ 0)
DENSITY_SAT = 64                    # points per cell considered "full"
RCS_MIN, RCS_MAX = -10.0, 40.0      # dBsm
V_SCALE = 20.0                      # m/s

LIDAR_BEV_CHANNELS = ["lidar_occ", "lidar_max_z", "lidar_density", "lidar_intensity"]
RADAR_BEV_CHANNELS = ["radar_occ", "radar_rcs", "radar_vx", "radar_vy"]

# nuScenes category prefix -> our detection class
DET_CLASSES = ["car", "pedestrian"]
CATEGORY_MAP = {"vehicle.car": "car", "human.pedestrian": "pedestrian"}


@dataclass(frozen=True)
class BEVGrid:
    x_range: tuple = (-51.2, 51.2)   # forward / backward (m)
    y_range: tuple = (-51.2, 51.2)   # right / left (m)
    res: float = 0.4                 # metres per cell

    @property
    def shape(self):
        return (round((self.x_range[1] - self.x_range[0]) / self.res),
                round((self.y_range[1] - self.y_range[0]) / self.res))

    def to_cells(self, x, y):
        """Metric (x, y) -> integer (row, col) plus a mask of points inside the grid."""
        rows = np.floor((self.x_range[1] - x) / self.res).astype(int)
        cols = np.floor((self.y_range[1] - y) / self.res).astype(int)
        H, W = self.shape
        inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        return rows, cols, inside

    def to_pixel_float(self, x, y):
        """Continuous pixel coordinates (for drawing boxes and Gaussian centres)."""
        return (self.x_range[1] - x) / self.res, (self.y_range[1] - y) / self.res


GRID = BEVGrid()


# ---------------------------------------------------------------------------
# Scatter helpers: many points can fall in one cell, so we reduce per cell.
# ---------------------------------------------------------------------------
def _flat_index(grid, x, y):
    rows, cols, inside = grid.to_cells(x, y)
    return rows[inside] * grid.shape[1] + cols[inside], inside


def _reduce(flat, values, size, how):
    out = np.zeros(size, dtype=np.float32)
    if how == "count":
        np.add.at(out, flat, 1.0)
    elif how == "max":
        out[:] = -np.inf
        np.maximum.at(out, flat, values)
        out[np.isinf(out)] = 0.0
    elif how == "mean":
        cnt = np.zeros(size, dtype=np.float32)
        np.add.at(out, flat, values)
        np.add.at(cnt, flat, 1.0)
        out = np.divide(out, cnt, out=np.zeros_like(out), where=cnt > 0)
    return out


def rasterize_lidar(lidar, grid=GRID):
    """lidar: (5, N) from Step 3 -> (4, H, W)."""
    z = lidar[2]
    band = (z >= Z_MIN) & (z <= Z_MAX)
    pts = lidar[:, band]
    flat, inside = _flat_index(grid, pts[0], pts[1])
    size, H, W = grid.shape[0] * grid.shape[1], *grid.shape

    count = _reduce(flat, None, size, "count")
    max_z = _reduce(flat, (pts[2, inside] - Z_MIN) / (Z_MAX - Z_MIN), size, "max")
    intensity = _reduce(flat, pts[3, inside] / 255.0, size, "mean")
    occ = (count > 0).astype(np.float32)
    density = np.minimum(np.log1p(count) / np.log1p(DENSITY_SAT), 1.0)
    return np.stack([occ, max_z, density, intensity]).reshape(4, H, W)


def rasterize_radar(radar, grid=GRID):
    """radar: (8, M) from Step 3 -> (4, H, W). Radar z is ignored (it is ~mount height)."""
    flat, inside = _flat_index(grid, radar[0], radar[1])
    size, H, W = grid.shape[0] * grid.shape[1], *grid.shape
    pts = radar[:, inside]

    occ = (_reduce(flat, None, size, "count") > 0).astype(np.float32)
    # Offset by +1e-3 so a real RCS at the minimum is not confused with "empty".
    rcs = _reduce(flat, np.clip((pts[3] - RCS_MIN) / (RCS_MAX - RCS_MIN), 0, 1) + 1e-3,
                  size, "max")
    vx = _reduce(flat, pts[4] / V_SCALE, size, "mean")
    vy = _reduce(flat, pts[5] / V_SCALE, size, "mean")
    return np.stack([occ, rcs, vx, vy]).reshape(4, H, W)


# ---------------------------------------------------------------------------
# Ground truth: annotated boxes -> ego frame -> Gaussian centre heatmaps
# ---------------------------------------------------------------------------
def det_class(category_name):
    for prefix, cls in CATEGORY_MAP.items():
        if category_name.startswith(prefix):
            return cls
    return None


def get_gt_boxes(nusc, sample, ref_sd):
    """Annotated boxes in the ego frame at the LiDAR keyframe time.

    Boxes are stored in the GLOBAL frame, so we apply global -> ego (the inverse
    of the ego pose). Boxes with no LiDAR or radar hit are skipped (the sensors
    cannot see them, so the detector should not be punished for missing them).
    """
    pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    boxes = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls = det_class(ann["category_name"])
        if cls is None or ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
            continue
        box = nusc.get_box(ann_token)
        box.translate(-np.array(pose["translation"]))
        box.rotate(Quaternion(pose["rotation"]).inverse)
        box.name = cls
        boxes.append(box)
    return boxes


def draw_gaussian(heatmap, row, col, sigma):
    H, W = heatmap.shape
    r = int(3 * sigma) + 1
    r0, r1 = max(0, int(row) - r), min(H, int(row) + r + 1)
    c0, c1 = max(0, int(col) - r), min(W, int(col) + r + 1)
    if r0 >= r1 or c0 >= c1:
        return
    rr, cc = np.mgrid[r0:r1, c0:c1]
    g = np.exp(-((rr + 0.5 - row) ** 2 + (cc + 0.5 - col) ** 2) / (2 * sigma ** 2))
    np.maximum(heatmap[r0:r1, c0:c1], g, out=heatmap[r0:r1, c0:c1])


def make_heatmaps(boxes, grid=GRID):
    """(n_classes, H, W) with a peak of 1.0 at every object centre.

    Sigma grows with object size so a 4.5 m car gets a wider blob than a
    0.6 m pedestrian - the CenterNet / CenterPoint idea.
    """
    hm = np.zeros((len(DET_CLASSES), *grid.shape), dtype=np.float32)
    for box in boxes:
        row, col = grid.to_pixel_float(box.center[0], box.center[1])
        width, length = box.wlh[0], box.wlh[1]
        sigma = max(1.0, min(width, length) / grid.res / 4)
        draw_gaussian(hm[DET_CLASSES.index(box.name)], row, col, sigma)
    return hm


def boxes_to_array(boxes):
    """(K, 8): class_id, x, y, z, w, l, h, yaw - regression targets for Week 2."""
    rows = [[DET_CLASSES.index(b.name), *b.center, *b.wlh,
             b.orientation.yaw_pitch_roll[0]] for b in boxes]
    return np.array(rows, dtype=np.float32).reshape(-1, 8)


def build_bev(nusc, sample, grid=GRID):
    lidar, ref_sd = load_lidar_multisweep(nusc, sample)
    radar, _ = load_radar_multisweep(nusc, sample)
    boxes = get_gt_boxes(nusc, sample, ref_sd)
    return {
        "lidar_bev": rasterize_lidar(lidar, grid),
        "radar_bev": rasterize_radar(radar, grid),
        "heatmap": make_heatmaps(boxes, grid),
        "boxes": boxes_to_array(boxes),
        "_boxes": boxes, "_lidar": lidar, "_radar": radar, "_ref_sd": ref_sd,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def draw_boxes(ax, boxes, grid=GRID):
    colors = {"car": "lime", "pedestrian": "cyan"}
    for b in boxes:
        corners = b.bottom_corners()[:2]                       # (2, 4) metric x, y
        r, c = grid.to_pixel_float(corners[0], corners[1])
        ax.plot(np.append(c, c[0]) - 0.5, np.append(r, r[0]) - 0.5,
                color=colors[b.name], lw=1)


def main():
    nusc = get_nusc()
    scene = nusc.scene[SCENE_INDEX]
    token = scene["first_sample_token"]
    for _ in range(SAMPLE_INDEX):
        token = nusc.get("sample", token)["next"] or token
    sample = nusc.get("sample", token)

    bev = build_bev(nusc, sample)
    print(f"Grid {GRID.shape} at {GRID.res} m/cell")
    for name, arr in [("lidar_bev", bev["lidar_bev"]), ("radar_bev", bev["radar_bev"]),
                      ("heatmap", bev["heatmap"])]:
        print(f"  {name:10s} {arr.shape}  range [{arr.min():+.2f}, {arr.max():+.2f}]")
    print(f"  occupied cells: LiDAR {bev['lidar_bev'][0].mean():.1%}, "
          f"radar {bev['radar_bev'][0].mean():.2%}")
    n_per_class = {c: int((bev['boxes'][:, 0] == i).sum()) for i, c in enumerate(DET_CLASSES)}
    print(f"  GT objects: {n_per_class}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panels = [
        (bev["lidar_bev"][1], "LiDAR max height", "viridis", (0, 1)),
        (bev["lidar_bev"][2], "LiDAR density", "magma", (0, 1)),
        (bev["lidar_bev"][3], "LiDAR intensity", "gray", (0, 0.3)),
        (bev["lidar_bev"][0], "LiDAR occupancy", "gray", (0, 1)),
        (bev["radar_bev"][1], "Radar RCS (sparse!)", "hot", (0, 1)),
        (np.zeros(GRID.shape), "Radar |velocity| (m/s, enlarged dots)", "gray", (0, 1)),
        (bev["heatmap"].max(0), "GT heatmap (car + ped)", "inferno", (0, 1)),
        (bev["lidar_bev"][1], "LiDAR height + GT boxes", "gray", (0, 1)),
    ]
    for ax, (img, title, cmap, (vmin, vmax)) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title); ax.axis("off")
        ax.plot(GRID.shape[1] / 2 - 0.5, GRID.shape[0] / 2 - 0.5, "r^", ms=6)
    # Radar points are 1 cell each - enlarge them so they're visible.
    ys, xs = np.nonzero(bev["radar_bev"][0])
    axes[1, 0].scatter(xs, ys, s=6, c=bev["radar_bev"][1][ys, xs], cmap="hot", vmin=0, vmax=1)
    speed = np.hypot(bev["radar_bev"][2], bev["radar_bev"][3])[ys, xs] * V_SCALE
    sc = axes[1, 1].scatter(xs, ys, s=10, c=speed, cmap="cool", vmin=0, vmax=15)
    fig.colorbar(sc, ax=axes[1, 1], fraction=0.046)
    draw_boxes(axes[1, 3], bev["_boxes"])

    out = os.path.join(ensure_out_dir(), "step4_bev.png")
    fig.suptitle("Step 4: BEV channels (forward = up, red triangle = ego car)")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\nSaved figure -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
