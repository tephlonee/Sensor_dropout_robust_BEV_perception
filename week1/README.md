# Week 1 — nuScenes data & geometry pipeline

Turns raw nuScenes-mini sensor files into a cached BEV (bird's-eye-view) dataset
that the Week 2 detector trains on. Each step imports the previous one, so a
fix in an early step flows through automatically.

## Setup

```bash
pip install -r requirements.txt
export NUSCENES_DATAROOT=/path/to/your/dataroot   # folder with v1.0-mini/, samples/, sweeps/
```

(Or edit `DATAROOT` once in `nusc_utils.py`.)

## Run order

| Step | File | What it does | Output |
|---|---|---|---|
| 1 | `step1_load_lidar_radar.py` | Load raw LiDAR (4×N) and radar (18×M) in their own sensor frames | `outputs/step1_raw_frames.png` |
| 2 | `step2_to_ego_frame.py` | Calibration + pose transforms → all sensors in one ego frame | `outputs/step2_ego_frame.png` |
| 3 | `step3_multisweep.py` | Stack 10 LiDAR / 6 radar past sweeps with ego-motion compensation; radar points propagated by v·dt. Checks itself against the devkit. | `outputs/step3_multisweep.png` |
| 4 | `step4_bev.py` | Rasterise to a 256×256 grid (0.4 m cells): LiDAR occupancy/height/density/intensity, radar occupancy/RCS/vx/vy, plus GT centre heatmaps | `outputs/step4_bev.png` |
| 5 | `step5_point_painting.py` | Project LiDAR into the 6 cameras, paint points with DeepLabV3 class scores, rasterise to BEV. Checks projection against the devkit. | `outputs/step5_point_painting.png` |
| 6 | `step6_build_dataset.py` | Run steps 3–5 over all 404 keyframes and cache | `outputs/bev_cache/*.npz` + `index.json` |

Step 6 on CPU takes roughly 30–60 min with camera painting (6 segmentation
passes per sample); `--no-camera` takes a few minutes. It skips samples
already cached, so you can stop and resume.

## Cached format (what Week 2 reads)

```
bev      (11, 256, 256) float16   channels 0-3 LiDAR, 4-7 radar, 8-10 camera
heatmap  (2, 256, 256)  float16   car, pedestrian
boxes    (K, 8)         float32   class_id, x, y, z, w, l, h, yaw  (ego frame)
```

`index.json` holds channel names, `modality_slices`, grid spec and the official
`mini_train` / `mini_val` split per sample. Simulating a lost RADAR in Week 2 is
`bev[4:8] = 0`.

## Conventions

* Reference frame = ego frame at the keyframe LiDAR timestamp.
* Transforms are 4×4 matrices named `T_<to>_from_<from>`.
* BEV row 0 = furthest forward, column 0 = furthest left, so `imshow` shows forward = up.
