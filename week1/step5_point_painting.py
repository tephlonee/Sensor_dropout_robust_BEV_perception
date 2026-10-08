"""
Week 1 - Step 5: PointPainting - give the BEV a camera channel.

Idea (Vora et al., "PointPainting", CVPR 2020)
---------------------------------------------
1. Run a pretrained 2D semantic-segmentation network on each camera image.
   Every pixel gets class probabilities (car, person, bicycle, ...).
2. Project every LiDAR point into every camera image.
3. Each point that lands inside an image takes the class scores of the pixel
   it hits - the point is "painted".
4. Rasterise the painted scores into the same BEV grid as Step 4.

That gives us camera semantics in BEV without training a camera-to-BEV network
like Lift-Splat-Shoot. It also matters for the missing-RADAR thesis: the
"camera-only" fallback has to come from somewhere, and in our setup it is
these channels plus LiDAR.

The projection chain (the calibration skill Ericsson asks for)
--------------------------------------------------------------
  point in ref ego frame (LiDAR time)
    -> global                (ego pose at LiDAR time)
    -> ego at CAMERA time    (inverse ego pose at camera time)
    -> camera frame          (inverse camera calibration; z = depth along optical axis)
    -> pixels                (intrinsic matrix K, then divide by depth)

Run:  pip install torch torchvision
      python step5_point_painting.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nusc_utils import (CAMERA_CHANNELS, LIDAR_CHANNEL, T_ego_from_sensor,
                        T_global_from_ego, apply_transform, ensure_out_dir,
                        get_nusc, sd_path)
from step3_multisweep import load_lidar_multisweep
from step4_bev import GRID, _flat_index, _reduce

SCENE_INDEX = 0
SAMPLE_INDEX = 10
MIN_DEPTH = 1.0          # ignore points closer than 1 m to the camera

# Pascal-VOC classes (what the torchvision model predicts) grouped into our channels.
PAINT_GROUPS = {
    "cam_vehicle": ["car", "bus"],
    "cam_pedestrian": ["person"],
    "cam_cycle": ["bicycle", "motorbike"],
}
CAMERA_BEV_CHANNELS = list(PAINT_GROUPS)


# ---------------------------------------------------------------------------
# 1. Segmentation model
# ---------------------------------------------------------------------------
class Segmenter:
    """Pretrained DeepLabV3 (MobileNetV3 backbone), 21 Pascal-VOC classes.

    Small enough for CPU (~0.3-1 s per image). Swap in a model trained on
    driving data (e.g. Cityscapes) later for better scores.
    """

    def __init__(self, device=None, pretrained=True):
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        if pretrained:
            model = deeplabv3_mobilenet_v3_large(weights=weights)
        else:   # random weights - only for testing the pipeline without a download
            model = deeplabv3_mobilenet_v3_large(
                weights=None, weights_backbone=None,
                num_classes=len(weights.meta["categories"]))
        self.model = model.eval().to(self.device)
        self.preprocess = weights.transforms()          # resize (short side 520) + normalise
        categories = weights.meta["categories"]
        self.group_idx = [[categories.index(c) for c in names] for names in PAINT_GROUPS.values()]

    def __call__(self, image: Image.Image):
        """Returns (n_groups, h, w) probabilities at the model's input resolution."""
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probs = self.model(x)["out"].softmax(dim=1)[0].cpu().numpy()
        return np.stack([probs[idx].sum(0) for idx in self.group_idx])


# ---------------------------------------------------------------------------
# 2. Projection
# ---------------------------------------------------------------------------
def T_cam_from_refego(nusc, cam_sd, ref_sd):
    """Ego frame at the LiDAR time -> camera frame at the camera's time."""
    T_ego_from_cam = T_ego_from_sensor(nusc, cam_sd)
    T_global_from_camego = T_global_from_ego(nusc, cam_sd)
    T_global_from_refego = T_global_from_ego(nusc, ref_sd)
    return (np.linalg.inv(T_ego_from_cam) @ np.linalg.inv(T_global_from_camego)
            @ T_global_from_refego)


def project_to_image(nusc, xyz_ego, cam_sd, ref_sd, min_depth=MIN_DEPTH):
    """Project (3, N) ego-frame points into one camera.

    Returns uv (2, N) pixel coords, depth (N,), and a mask of points that land
    inside the image and in front of the camera.
    """
    pts_cam = apply_transform(T_cam_from_refego(nusc, cam_sd, ref_sd), xyz_ego)
    depth = pts_cam[2]
    K = np.array(nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
                 ["camera_intrinsic"])
    uvw = K @ pts_cam
    uv = uvw[:2] / np.where(np.abs(uvw[2]) < 1e-6, 1e-6, uvw[2])   # perspective divide
    W, H = cam_sd["width"], cam_sd["height"]
    mask = ((depth > min_depth) & (uv[0] > 1) & (uv[0] < W - 1)
            & (uv[1] > 1) & (uv[1] < H - 1))
    return uv, depth, mask


# ---------------------------------------------------------------------------
# 3. Painting
# ---------------------------------------------------------------------------
def paint_points(nusc, sample, xyz_ego, ref_sd, segmenter, return_debug=False):
    """Returns scores (n_groups, N) and a 'seen by any camera' mask (N,).

    A point seen by two overlapping cameras gets the average of both.
    Known limitations (worth naming in an interview):
    * Points from OLD sweeps of moving objects paint slightly wrong - the object
      has moved since. The dt channel lets the network learn to trust them less.
    * No occlusion test: a wall point hidden behind a car projects onto the car's
      pixels and gets painted "vehicle". A depth (z-buffer) check would fix it.
    """
    n = xyz_ego.shape[1]
    score_sum = np.zeros((len(PAINT_GROUPS), n), dtype=np.float32)
    hits = np.zeros(n, dtype=np.float32)
    debug = {}
    for ch in CAMERA_CHANNELS:
        cam_sd = nusc.get("sample_data", sample["data"][ch])
        image = Image.open(sd_path(nusc, cam_sd)).convert("RGB")
        probs = segmenter(image)                                    # (G, h, w)
        uv, depth, mask = project_to_image(nusc, xyz_ego, cam_sd, ref_sd)
        # Model output is smaller than the image -> rescale pixel coordinates.
        h, w = probs.shape[1:]
        u = np.clip((uv[0, mask] * w / image.width).astype(int), 0, w - 1)
        v = np.clip((uv[1, mask] * h / image.height).astype(int), 0, h - 1)
        score_sum[:, mask] += probs[:, v, u]
        hits[mask] += 1
        if return_debug:
            debug[ch] = dict(image=image, probs=probs, uv=uv[:, mask], depth=depth[mask],
                             idx=np.nonzero(mask)[0])
    seen = hits > 0
    scores = np.divide(score_sum, hits, out=np.zeros_like(score_sum), where=seen)
    return (scores, seen, debug) if return_debug else (scores, seen)


def rasterize_painted(xyz_ego, scores, grid=GRID):
    """Per BEV cell, the MAX painted score of each group -> (n_groups, H, W)."""
    flat, inside = _flat_index(grid, xyz_ego[0], xyz_ego[1])
    size, H, W = grid.shape[0] * grid.shape[1], *grid.shape
    chans = [_reduce(flat, s[inside], size, "max") for s in scores]
    return np.stack(chans).reshape(len(chans), H, W)


def build_camera_bev(nusc, sample, segmenter, lidar=None, ref_sd=None, grid=GRID):
    if lidar is None:
        lidar, ref_sd = load_lidar_multisweep(nusc, sample)
    scores, _ = paint_points(nusc, sample, lidar[:3], ref_sd, segmenter)
    return rasterize_painted(lidar[:3], scores, grid)


# ---------------------------------------------------------------------------
# 4. Check our projection against the devkit
# ---------------------------------------------------------------------------
def verify_projection(nusc, sample, cam="CAM_FRONT"):
    lidar, ref_sd = load_lidar_multisweep(nusc, sample, nsweeps=1, min_dist=0.0)
    cam_sd = nusc.get("sample_data", sample["data"][cam])
    uv, _, mask = project_to_image(nusc, lidar[:3], cam_sd, ref_sd)
    theirs, _, _ = nusc.explorer.map_pointcloud_to_image(ref_sd["token"], cam_sd["token"])
    ours = uv[:, mask]
    # The devkit transforms float32 points through the global frame (coordinates in
    # the hundreds of metres), so expect ~0.01 px of rounding noise - and a point
    # sitting exactly on the image border may be kept by one and dropped by the other.
    if ours.shape[1] != theirs.shape[1]:
        print(f"Check vs devkit ({cam}): {ours.shape[1]} vs {theirs.shape[1]} points "
              f"(differ by {abs(ours.shape[1] - theirs.shape[1])}; border rounding if tiny)")
        return
    err = np.abs(ours - theirs[:2]).max() if ours.size else 0.0
    print(f"Check vs devkit map_pointcloud_to_image ({cam}): {ours.shape[1]} points, "
          f"max pixel error {err:.2e} -> {'OK' if err < 0.5 else 'MISMATCH'}")


def main():
    nusc = get_nusc()
    scene = nusc.scene[SCENE_INDEX]
    token = scene["first_sample_token"]
    for _ in range(SAMPLE_INDEX):
        token = nusc.get("sample", token)["next"] or token
    sample = nusc.get("sample", token)

    verify_projection(nusc, sample)

    segmenter = Segmenter()
    lidar, ref_sd = load_lidar_multisweep(nusc, sample)
    scores, seen, debug = paint_points(nusc, sample, lidar[:3], ref_sd, segmenter,
                                       return_debug=True)
    cam_bev = rasterize_painted(lidar[:3], scores)

    print(f"\nPainted {seen.sum():,} of {seen.size:,} points ({seen.mean():.0%} seen by a camera)")
    for name, s in zip(PAINT_GROUPS, scores):
        print(f"  {name:15s} points with score > 0.5: {(s > 0.5).sum():6,}")
    print(f"Camera BEV shape: {cam_bev.shape}")

    # ---- plot ----
    d = debug["CAM_FRONT"]
    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(d["image"]); ax.set_title("CAM_FRONT + projected LiDAR (colour = depth)")
    ax.scatter(d["uv"][0], d["uv"][1], c=d["depth"], s=0.5, cmap="jet", vmin=0, vmax=50)
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(d["probs"][0], cmap="inferno", vmin=0, vmax=1)
    ax.set_title("Segmentation: vehicle probability"); ax.axis("off")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(d["image"]); ax.axis("off")
    ax.scatter(d["uv"][0], d["uv"][1], c=scores[0, d["idx"]], s=0.8, cmap="inferno",
               vmin=0, vmax=1)
    ax.set_title("Painted points: vehicle score")

    for i, (name, title) in enumerate(zip(PAINT_GROUPS, ["vehicle", "pedestrian", "cycle"])):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.imshow(cam_bev[i], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(GRID.shape[1] / 2 - 0.5, GRID.shape[0] / 2 - 0.5, "c^", ms=6)
        ax.set_title(f"BEV camera channel: {title}"); ax.axis("off")

    out = os.path.join(ensure_out_dir(), "step5_point_painting.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\nSaved figure -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
