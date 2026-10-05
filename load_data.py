"""
Week 1 - Step 1: Load raw LiDAR and RADAR point clouds from nuScenes-mini.

What this step teaches
----------------------
1. How nuScenes is organised:  scene -> sample -> sample_data (one per sensor).
2. What a raw LiDAR point cloud looks like (x, y, z, intensity).
3. What a raw RADAR point cloud looks like (18 fields: position, RCS, velocity, quality flags).
4. That every sensor reports points in ITS OWN coordinate frame - so the raw
   clouds do NOT line up with each other yet. Fixing that is Step 2.

Run:
    pip install nuscenes-devkit matplotlib
    python step1_load_lidar_radar.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
DATAROOT = ".\data\sets\\nuscenes"   # folder that contains v1.0-mini/, samples/, sweeps/, maps/
VERSION = "v1.0-mini"
SCENE_INDEX = 0                        # nuScenes-mini has 10 scenes (0..9)
OUT_DIR = "outputs"

LIDAR_CHANNEL = "LIDAR_TOP"
RADAR_CHANNELS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

# Row indices into the 18-row radar array (order comes from the .pcd header).
RADAR_FIELDS = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy",
                "vx_comp", "vy_comp", "is_quality_valid", "ambig_state",
                "x_rms", "y_rms", "invalid_state", "pdh0", "vx_rms", "vy_rms"]
R = {name: i for i, name in enumerate(RADAR_FIELDS)}


# ---------------------------------------------------------------------------
# 1. Open the dataset and walk the hierarchy
# ---------------------------------------------------------------------------
def open_dataset():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    print("\nScenes in this split:")
    nusc.list_scenes()
    return nusc


def get_first_sample(nusc, scene_index):
    """scene -> its first keyframe sample.

    A *scene* is a ~20 s drive. It contains ~40 *samples* (keyframes at 2 Hz).
    Each sample holds one *sample_data* token per sensor, pointing to the
    measurement closest in time to that keyframe.
    """
    scene = nusc.scene[scene_index]
    sample = nusc.get("sample", scene["first_sample_token"])
    print(f"\nScene '{scene['name']}': {scene['description']}")
    print(f"  samples in scene : {scene['nbr_samples']}")
    print(f"  sensors in sample: {sorted(sample['data'].keys())}")
    return scene, sample


# ---------------------------------------------------------------------------
# 2. Load LiDAR
# ---------------------------------------------------------------------------
def load_lidar(nusc, sample):
    """Returns (points[4, N], sample_data record).

    Rows: x, y, z, intensity.  Frame: the LIDAR_TOP sensor frame
    (x right, y forward, z up for nuScenes' lidar - note: NOT the same axes as the car).
    The .pcd.bin file actually stores 5 values per point (the 5th is the laser
    ring index); LidarPointCloud keeps the first 4.
    """
    sd_token = sample["data"][LIDAR_CHANNEL]
    sd = nusc.get("sample_data", sd_token)
    path = os.path.join(nusc.dataroot, sd["filename"])
    pc = LidarPointCloud.from_file(path)
    return pc.points, sd


def describe_lidar(points):
    x, y, z, intensity = points
    rng = np.sqrt(x**2 + y**2)
    print(f"\n[LIDAR_TOP] {points.shape[1]:,} points, array shape {points.shape}")
    print(f"  x range        : {x.min():7.1f} .. {x.max():6.1f} m")
    print(f"  y range        : {y.min():7.1f} .. {y.max():6.1f} m")
    print(f"  z range        : {z.min():7.1f} .. {z.max():6.1f} m")
    print(f"  horizontal rng : median {np.median(rng):.1f} m, max {rng.max():.1f} m")
    print(f"  intensity      : {intensity.min():.0f} .. {intensity.max():.0f}")


# ---------------------------------------------------------------------------
# 3. Load RADAR
# ---------------------------------------------------------------------------
def load_radar(nusc, sample, channel, filtered=True):
    """Returns (points[18, M], sample_data record).

    filtered=True uses the devkit's default quality filters
    (valid clusters only, ambiguity state 3, dyn_prop 0..6) - what most papers use.
    filtered=False keeps every return, including noise and artefacts.
    """
    sd = nusc.get("sample_data", sample["data"][channel])
    path = os.path.join(nusc.dataroot, sd["filename"])
    if filtered:
        pc = RadarPointCloud.from_file(path)
    else:
        all_states = list(range(18))
        pc = RadarPointCloud.from_file(path, invalid_states=all_states,
                                       dynprop_states=all_states,
                                       ambig_states=all_states)
    return pc.points, sd


def describe_radar(channel, pts, n_unfiltered):
    print(f"\n[{channel}] {pts.shape[1]} points after filtering "
          f"({n_unfiltered} raw) - array shape {pts.shape}")
    if pts.shape[1] == 0:
        return
    speed = np.hypot(pts[R["vx_comp"]], pts[R["vy_comp"]])
    print(f"  x (forward) : {pts[R['x']].min():6.1f} .. {pts[R['x']].max():6.1f} m")
    print(f"  y (left)    : {pts[R['y']].min():6.1f} .. {pts[R['y']].max():6.1f} m")
    print(f"  z           : all zero? {np.allclose(pts[R['z']], 0)}  <- radar is effectively 2D")
    print(f"  RCS         : {pts[R['rcs']].min():.1f} .. {pts[R['rcs']].max():.1f} dBsm")
    print(f"  |v| ego-compensated: max {speed.max():.1f} m/s, "
          f"{(speed > 1.0).sum()} points moving > 1 m/s")


# ---------------------------------------------------------------------------
# 4. Plot: each cloud in its OWN sensor frame
# ---------------------------------------------------------------------------
def plot_raw_frames(lidar_pts, radar_by_channel, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Left: LiDAR top-down, coloured by height.
    ax = axes[0]
    keep = np.hypot(lidar_pts[0], lidar_pts[1]) < 50
    sc = ax.scatter(lidar_pts[0, keep], lidar_pts[1, keep], c=lidar_pts[2, keep],
                    s=0.3, cmap="viridis", vmin=-2, vmax=3)
    fig.colorbar(sc, ax=ax, label="z (m)")
    ax.scatter(0, 0, marker="*", s=200, c="red", label="sensor origin")
    ax.set_title("LIDAR_TOP - lidar sensor frame")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper right")

    # Right: all 5 radars, each in its OWN frame, overlaid on the same axes.
    ax = axes[1]
    colors = plt.cm.tab10.colors
    for (channel, pts), color in zip(radar_by_channel.items(), colors):
        if pts.shape[1] == 0:
            continue
        ax.scatter(pts[R["x"]], pts[R["y"]], s=12, color=color, label=channel)
        # Ego-motion-compensated velocity arrows (scaled 1 s ahead).
        ax.quiver(pts[R["x"]], pts[R["y"]], pts[R["vx_comp"]], pts[R["vy_comp"]],
                  color=color, angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.scatter(0, 0, marker="*", s=200, c="red")
    ax.set_title("All 5 radars, each in its OWN frame\n"
                 "(they pile up 'in front' - not aligned yet -> Step 2)")
    ax.set_xlabel("x forward of each radar (m)"); ax.set_ylabel("y left of each radar (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure -> {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Peek ahead: where each sensor is mounted (used in Step 2)
# ---------------------------------------------------------------------------
def print_mounting(nusc, sample):
    print("\nSensor mounting (calibrated_sensor, relative to the car's ego frame):")
    for channel in [LIDAR_CHANNEL] + RADAR_CHANNELS:
        sd = nusc.get("sample_data", sample["data"][channel])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        t = np.round(cs["translation"], 2)
        print(f"  {channel:18s} translation {t}  rotation(quat) {np.round(cs['rotation'], 3)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nusc = open_dataset()
    _, sample = get_first_sample(nusc, SCENE_INDEX)

    lidar_pts, lidar_sd = load_lidar(nusc, sample)
    describe_lidar(lidar_pts)
    print(f"  timestamp      : {lidar_sd['timestamp']} (microseconds)")

    radar_by_channel = {}
    for channel in RADAR_CHANNELS:
        pts, sd = load_radar(nusc, sample, channel, filtered=True)
        raw, _ = load_radar(nusc, sample, channel, filtered=False)
        describe_radar(channel, pts, raw.shape[1])
        dt_ms = (sd["timestamp"] - lidar_sd["timestamp"]) / 1000
        print(f"  time offset vs LiDAR: {dt_ms:+.1f} ms  <- sensors are not perfectly synchronised")
        radar_by_channel[channel] = pts

    total = sum(p.shape[1] for p in radar_by_channel.values())
    print(f"\nLiDAR points: {lidar_pts.shape[1]:,}   vs   radar points (all 5): {total}")

    print_mounting(nusc, sample)
    plot_raw_frames(lidar_pts, radar_by_channel, os.path.join(OUT_DIR, "step1_raw_frames.png"))


if __name__ == "__main__":
    main()