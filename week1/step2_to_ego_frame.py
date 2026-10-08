"""
Week 1 - Step 2: Put LiDAR and all 5 radars into ONE frame (the car's ego frame).

In Step 1 every radar plotted its points "in front" because each used its own
axes. Here we fix that with the calibration + pose transforms.

Two levels of correctness:
  naive  : sensor -> ego using only the mounting calibration.
  full   : sensor -> ego(t_sensor) -> global -> ego(t_ref).
           Each radar fires a few tens of ms before/after the LiDAR. At 10 m/s
           that is tens of cm of car motion. The global-frame hop removes it.

Reference time = the LiDAR keyframe. We use this convention in every later step.

Run:  python step2_to_ego_frame.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

from nusc_utils import (LIDAR_CHANNEL, RADAR_CHANNELS, R, T_ego_from_sensor,
                        T_refego_from_sensor, apply_transform, ensure_out_dir,
                        get_nusc, rotate_vectors, sd_path)

SCENE_INDEX = 0


def lidar_in_ego(nusc, sample):
    """Keyframe LiDAR sweep in the ego frame. Returns (4, N): x, y, z, intensity."""
    ref_sd = nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
    pts = LidarPointCloud.from_file(sd_path(nusc, ref_sd)).points
    T = T_refego_from_sensor(nusc, ref_sd, ref_sd)  # same time -> just calibration
    pts[:3] = apply_transform(T, pts[:3])
    return pts, ref_sd


def radar_in_ego(nusc, sample, channel, ref_sd, full=True):
    """One radar's keyframe sweep in the ego frame at the LiDAR time.

    Returns (6, M): x, y, z, rcs, vx, vy.
    Velocities are rotated into the ego frame too (they are vectors, so no translation).
    """
    sd = nusc.get("sample_data", sample["data"][channel])
    raw = RadarPointCloud.from_file(sd_path(nusc, sd)).points
    T = T_refego_from_sensor(nusc, sd, ref_sd) if full else T_ego_from_sensor(nusc, sd)

    xyz = apply_transform(T, raw[[R["x"], R["y"], R["z"]]])
    vel = np.vstack([raw[R["vx_comp"]], raw[R["vy_comp"]], np.zeros(raw.shape[1])])
    vel = rotate_vectors(T, vel)
    out = np.vstack([xyz, raw[R["rcs"]], vel[:2]])   # rows: x, y, z, rcs, vx, vy
    return out, sd


def main():
    nusc = get_nusc()
    scene = nusc.scene[SCENE_INDEX]
    sample = nusc.get("sample", scene["first_sample_token"])

    lidar, ref_sd = lidar_in_ego(nusc, sample)
    print(f"LiDAR in ego frame: {lidar.shape[1]:,} points, "
          f"z range {lidar[2].min():.1f}..{lidar[2].max():.1f} m (ground is ~0)")

    radars = {}
    print("\nRadar: naive (calibration only) vs full (with ego-motion between timestamps)")
    for ch in RADAR_CHANNELS:
        full, sd = radar_in_ego(nusc, sample, ch, ref_sd, full=True)
        naive, _ = radar_in_ego(nusc, sample, ch, ref_sd, full=False)
        dt_ms = (sd["timestamp"] - ref_sd["timestamp"]) / 1000
        shift = np.linalg.norm(full[:2] - naive[:2], axis=0)
        shift_txt = f"{shift.mean() * 100:5.1f} cm" if shift.size else "  n/a"
        print(f"  {ch:18s} dt {dt_ms:+6.1f} ms -> mean correction {shift_txt}")
        radars[ch] = full

    # ---- plot: everything in one frame ----
    fig, ax = plt.subplots(figsize=(9, 9))
    keep = (np.abs(lidar[0]) < 50) & (np.abs(lidar[1]) < 50)
    ax.scatter(lidar[1, keep], lidar[0, keep], s=0.2, c="0.6", label="LIDAR_TOP")
    for (ch, pts), color in zip(radars.items(), plt.cm.tab10.colors):
        ax.scatter(pts[1], pts[0], s=18, color=color, label=ch)
        ax.quiver(pts[1], pts[0], pts[5], pts[4], color=color, angles="xy",
                  scale_units="xy", scale=1, width=0.003)
    ax.add_patch(plt.Rectangle((-1.0, -1.0), 2.0, 4.5, fill=False, ec="red", lw=2))
    # Plot y on the horizontal axis (flipped) so forward is UP and left is LEFT.
    ax.set_xlim(50, -50); ax.set_ylim(-50, 50); ax.set_aspect("equal")
    ax.set_xlabel("y  (left +)  [m]"); ax.set_ylabel("x  (forward +)  [m]")
    ax.set_title("Step 2: LiDAR + 5 radars in the ego frame (forward = up)\n"
                 "arrows = ego-motion-compensated radar velocity (1 s)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    out = os.path.join(ensure_out_dir(), "step2_ego_frame.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
