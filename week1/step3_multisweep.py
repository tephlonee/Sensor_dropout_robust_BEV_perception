"""
Week 1 - Step 3: Accumulate past sweeps with ego-motion compensation.
This is the spatiotemporal part of the pipeline.

Why
---
* One LiDAR sweep is sparse far away. One radar sweep has only tens of points.
* nuScenes stores the sweeps BETWEEN keyframes too (LiDAR 20 Hz, radar ~13 Hz),
  linked by `prev` / `next` tokens on each sample_data record.
* Stacking the last N sweeps gives much denser input. But the car moved
  between sweeps, so each old sweep must be moved into the reference ego frame
  (the keyframe LiDAR time) with the same chain as Step 2.

What ego-motion compensation does NOT fix
-----------------------------------------
Static things (buildings, parked cars) line up perfectly. Things that were
MOVING leave a smeared trail, because they really were somewhere else 0.5 s ago.
* LiDAR cannot fix that - it measures position only.
* Radar can: each point has a velocity, so we can push it forward in time by
  v * dt ("propagation"). This is one concrete reason radar helps fusion.

Every point also gets a `dt` channel (seconds before the keyframe), so the
network can tell new points from old ones.

Run:  python step3_multisweep.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

from nusc_utils import (LIDAR_CHANNEL, RADAR_CHANNELS, R, T_refego_from_sensor,
                        apply_transform, ensure_out_dir, get_nusc, rotate_vectors,
                        sd_path)

SCENE_INDEX = 0
SAMPLE_INDEX = 10        # a keyframe in the middle of the scene has past sweeps
LIDAR_SWEEPS = 10        # 10 x 50 ms = 0.5 s of history
RADAR_SWEEPS = 6         # 6 x ~75 ms = ~0.45 s of history

# Row layouts of the arrays returned below.
LIDAR_ROWS = ["x", "y", "z", "intensity", "dt"]
RADAR_ROWS = ["x", "y", "z", "rcs", "vx", "vy", "dt", "sensor_id"]


def sweep_chain(nusc, sd_token, nsweeps):
    """The keyframe sample_data record followed by up to nsweeps-1 previous ones."""
    chain = []
    sd = nusc.get("sample_data", sd_token)
    for _ in range(nsweeps):
        chain.append(sd)
        if sd["prev"] == "":
            break                       # start of the scene: fewer sweeps available
        sd = nusc.get("sample_data", sd["prev"])
    return chain


def load_lidar_multisweep(nusc, sample, nsweeps=LIDAR_SWEEPS, min_dist=1.0):
    """Returns (5, N): x, y, z, intensity, dt  in the ego frame at the keyframe LiDAR time."""
    ref_sd = nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
    out = []
    for sd in sweep_chain(nusc, ref_sd["token"], nsweeps):
        pts = LidarPointCloud.from_file(sd_path(nusc, sd)).points       # sensor frame
        # Drop returns from the car's own roof (inside +/- min_dist of the sensor).
        own_car = (np.abs(pts[0]) < min_dist) & (np.abs(pts[1]) < min_dist)
        pts = pts[:, ~own_car]

        T = T_refego_from_sensor(nusc, sd, ref_sd)
        xyz = apply_transform(T, pts[:3])
        dt = (ref_sd["timestamp"] - sd["timestamp"]) * 1e-6
        out.append(np.vstack([xyz, pts[3], np.full(pts.shape[1], dt)]))
    return np.hstack(out), ref_sd


def load_radar_multisweep(nusc, sample, nsweeps=RADAR_SWEEPS, propagate=True):
    """Returns (8, M): x, y, z, rcs, vx, vy, dt, sensor_id  in the ego frame at the LiDAR time.

    propagate=True moves each old point by v * dt so moving objects line up
    with where they are NOW instead of leaving a trail.
    """
    ref_sd = nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
    out = []
    for sensor_id, ch in enumerate(RADAR_CHANNELS):
        for sd in sweep_chain(nusc, sample["data"][ch], nsweeps):
            raw = RadarPointCloud.from_file(sd_path(nusc, sd)).points
            if raw.shape[1] == 0:
                continue
            T = T_refego_from_sensor(nusc, sd, ref_sd)
            xyz = apply_transform(T, raw[[R["x"], R["y"], R["z"]]])
            vel = np.vstack([raw[R["vx_comp"]], raw[R["vy_comp"]], np.zeros(raw.shape[1])])
            vel = rotate_vectors(T, vel)[:2]
            # Radar can fire slightly AFTER the LiDAR, so dt may be a little negative.
            dt = (ref_sd["timestamp"] - sd["timestamp"]) * 1e-6
            if propagate:
                xyz[:2] += vel * dt
            n = raw.shape[1]
            out.append(np.vstack([xyz, raw[R["rcs"]], vel, np.full(n, dt),
                                  np.full(n, sensor_id)]))
    if not out:
        return np.zeros((len(RADAR_ROWS), 0)), ref_sd
    return np.hstack(out), ref_sd


def verify_against_devkit(nusc, sample, nsweeps=LIDAR_SWEEPS):
    """Sanity check: our hand-written chain must equal the devkit's implementation.

    The devkit returns points in the LiDAR SENSOR frame; we return the ego frame,
    so we move ours into the LiDAR frame before comparing.
    """
    ours, ref_sd = load_lidar_multisweep(nusc, sample, nsweeps)
    theirs, their_dt = LidarPointCloud.from_file_multisweep(
        nusc, sample, LIDAR_CHANNEL, LIDAR_CHANNEL, nsweeps=nsweeps)
    T_sensor_from_ego = np.linalg.inv(T_refego_from_sensor(nusc, ref_sd, ref_sd))
    ours_in_lidar = apply_transform(T_sensor_from_ego, ours[:3])
    assert ours.shape[1] == theirs.points.shape[1], "point counts differ"
    err_xyz = np.abs(ours_in_lidar - theirs.points[:3]).max()
    err_dt = np.abs(ours[4] - their_dt[0]).max()
    print(f"Check vs devkit from_file_multisweep: max |dxyz| = {err_xyz:.2e} m, "
          f"max |d dt| = {err_dt:.2e} s  -> {'OK' if err_xyz < 1e-3 else 'MISMATCH'}")


def main():
    nusc = get_nusc()
    scene = nusc.scene[SCENE_INDEX]
    token = scene["first_sample_token"]
    for _ in range(SAMPLE_INDEX):
        nxt = nusc.get("sample", token)["next"]
        token = nxt or token
    sample = nusc.get("sample", token)

    verify_against_devkit(nusc, sample)

    lidar1, _ = load_lidar_multisweep(nusc, sample, nsweeps=1)
    lidarN, _ = load_lidar_multisweep(nusc, sample, nsweeps=LIDAR_SWEEPS)
    radar1, _ = load_radar_multisweep(nusc, sample, nsweeps=1)
    radarN_raw, _ = load_radar_multisweep(nusc, sample, nsweeps=RADAR_SWEEPS, propagate=False)
    radarN, _ = load_radar_multisweep(nusc, sample, nsweeps=RADAR_SWEEPS, propagate=True)

    print(f"\nLiDAR points: 1 sweep {lidar1.shape[1]:,}  ->  {LIDAR_SWEEPS} sweeps {lidarN.shape[1]:,}")
    print(f"Radar points: 1 sweep {radar1.shape[1]}  ->  {RADAR_SWEEPS} sweeps {radarN.shape[1]}")
    print(f"History covered: LiDAR {lidarN[4].max():.2f} s, radar {radarN[6].max():.2f} s")
    moving = np.hypot(radarN[4], radarN[5]) > 1.0
    shift = np.hypot(*(radarN[:2] - radarN_raw[:2]))
    if moving.any():
        print(f"Propagation moved {moving.sum()} moving radar points by "
              f"{shift[moving].mean():.2f} m on average")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    lim = 40

    def lidar_panel(ax, pts, title):
        keep = (np.abs(pts[0]) < lim) & (np.abs(pts[1]) < lim)
        pts = pts[:, keep][:, np.argsort(-pts[4, keep])]   # draw newest points on top
        sc = ax.scatter(pts[1], pts[0], c=pts[4], s=0.2,
                        cmap="plasma_r", vmin=0, vmax=0.5)
        ax.set_title(title)
        return sc

    lidar_panel(axes[0], lidar1, "LiDAR - 1 sweep")
    sc = lidar_panel(axes[1], lidarN, f"LiDAR - {LIDAR_SWEEPS} sweeps (moving objects smear)")
    fig.colorbar(sc, ax=axes[1], label="dt: seconds before keyframe")

    ax = axes[2]
    ax.scatter(radarN_raw[1], radarN_raw[0], s=10, c="0.75", label="no propagation")
    sc = ax.scatter(radarN[1], radarN[0], s=14, c=radarN[6], cmap="plasma_r",
                    vmin=0, vmax=0.5, label="propagated by v*dt")
    ax.set_title(f"Radar - {RADAR_SWEEPS} sweeps x 5 sensors")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("y (left +) [m]"); ax.set_ylabel("x (forward +) [m]")
        ax.add_patch(plt.Rectangle((-1.0, -1.0), 2.0, 4.5, fill=False, ec="red", lw=1.5))

    out = os.path.join(ensure_out_dir(), "step3_multisweep.png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\nSaved figure -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
