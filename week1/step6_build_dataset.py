"""
Week 1 - Step 6: Run Steps 3-5 over every keyframe and cache the result.

Output: outputs/bev_cache/<sample_token>.npz, each containing
    bev      (11, 256, 256) float16   4 LiDAR + 4 radar + 3 camera channels
    heatmap  (2, 256, 256)  float16   car, pedestrian centre heatmaps
    boxes    (K, 8)         float32   class_id, x, y, z, w, l, h, yaw (ego frame)
plus outputs/bev_cache/index.json with the channel names and the official
mini_train / mini_val split for every sample.

Week 2 trains directly from this cache, so modality dropout is just
zeroing channel slices: RADAR missing = bev[4:8] = 0.

Run:  python step6_build_dataset.py              # with camera painting
      python step6_build_dataset.py --no-camera  # LiDAR + radar only, much faster
"""

import argparse
import json
import os
import time

import numpy as np
from nuscenes.utils.splits import create_splits_scenes

from nusc_utils import ensure_out_dir, get_nusc
from step4_bev import (DET_CLASSES, GRID, LIDAR_BEV_CHANNELS, RADAR_BEV_CHANNELS,
                       boxes_to_array, get_gt_boxes, make_heatmaps,
                       rasterize_lidar, rasterize_radar)
from step3_multisweep import load_lidar_multisweep, load_radar_multisweep
from step5_point_painting import (CAMERA_BEV_CHANNELS, Segmenter, paint_points,
                                  rasterize_painted)

CHANNELS = LIDAR_BEV_CHANNELS + RADAR_BEV_CHANNELS + CAMERA_BEV_CHANNELS
MODALITY_SLICES = {"lidar": [0, 4], "radar": [4, 8], "camera": [8, 11]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true", help="skip PointPainting")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    nusc = get_nusc()
    out_dir = ensure_out_dir("bev_cache")
    segmenter = None if args.no_camera else Segmenter()

    splits = create_splits_scenes()
    scene_split = {name: split for split in ("mini_train", "mini_val")
                   for name in splits[split]}

    index = {"channels": CHANNELS, "modality_slices": MODALITY_SLICES,
             "classes": DET_CLASSES, "grid": {"x_range": GRID.x_range,
             "y_range": GRID.y_range, "res": GRID.res}, "samples": []}

    t0 = time.time()
    n_total = len(nusc.sample)
    for i, sample in enumerate(nusc.sample):
        scene = nusc.get("scene", sample["scene_token"])
        path = os.path.join(out_dir, f"{sample['token']}.npz")
        index["samples"].append({"token": sample["token"], "scene": scene["name"],
                                 "split": scene_split.get(scene["name"], "unknown"),
                                 "timestamp": sample["timestamp"]})
        if os.path.exists(path) and not args.overwrite:
            continue

        lidar, ref_sd = load_lidar_multisweep(nusc, sample)
        radar, _ = load_radar_multisweep(nusc, sample)
        if segmenter is not None:
            scores, _ = paint_points(nusc, sample, lidar[:3], ref_sd, segmenter)
            cam_bev = rasterize_painted(lidar[:3], scores)
        else:
            cam_bev = np.zeros((len(CAMERA_BEV_CHANNELS), *GRID.shape), dtype=np.float32)
        boxes = get_gt_boxes(nusc, sample, ref_sd)

        bev = np.concatenate([rasterize_lidar(lidar), rasterize_radar(radar), cam_bev])
        np.savez_compressed(path, bev=bev.astype(np.float16),
                            heatmap=make_heatmaps(boxes).astype(np.float16),
                            boxes=boxes_to_array(boxes))

        if (i + 1) % 20 == 0 or i + 1 == n_total:
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i + 1}/{n_total} samples  ({rate:.1f} s/sample, "
                  f"~{rate * (n_total - i - 1) / 60:.0f} min left)")

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    counts = {s: sum(1 for x in index["samples"] if x["split"] == s)
              for s in ("mini_train", "mini_val")}
    print(f"\nDone. {n_total} samples cached in {out_dir}  {counts}")


if __name__ == "__main__":
    main()
