"""
Shared configuration and geometry helpers for Week 1.

The one idea everything else builds on
--------------------------------------
Every point we load is expressed in some coordinate frame. nuScenes stores
two transforms per measurement (per `sample_data` record):

    calibrated_sensor : sensor frame  -> ego frame    (fixed mounting, from calibration)
    ego_pose          : ego frame     -> global frame (where the car was at that timestamp)

Frames used here
    sensor frame : the sensor's own axes (each radar: x = its boresight)
    ego frame    : the car. Origin at the rear axle on the ground,
                   x forward, y left, z up. Changes as the car drives.
    global frame : a fixed map frame for the whole log.

All transforms are 4x4 homogeneous matrices named T_<to>_from_<from>, so that
chaining reads right-to-left like the maths:
    T_ref_from_sensor = T_ref_from_global @ T_global_from_ego @ T_ego_from_sensor
"""

import os
from functools import lru_cache

import numpy as np
from pyquaternion import Quaternion

# ---------------------------------------------------------------------------
# Configuration - edit DATAROOT once here; every step imports it.
# ---------------------------------------------------------------------------
DATAROOT = os.environ.get("NUSCENES_DATAROOT", ".\data\sets\\nuscenes")
VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini")
OUT_DIR = "outputs"

LIDAR_CHANNEL = "LIDAR_TOP"
RADAR_CHANNELS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
                  "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
CAMERA_CHANNELS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

# Row indices into the raw 18-row radar array (order from the .pcd header).
RADAR_FIELDS = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy",
                "vx_comp", "vy_comp", "is_quality_valid", "ambig_state",
                "x_rms", "y_rms", "invalid_state", "pdh0", "vx_rms", "vy_rms"]
R = {name: i for i, name in enumerate(RADAR_FIELDS)}


@lru_cache(maxsize=1)
def get_nusc():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)


# ---------------------------------------------------------------------------
# Transform building blocks
# ---------------------------------------------------------------------------
def make_transform(translation, rotation_wxyz):
    """4x4 matrix from a translation [x, y, z] and a quaternion [w, x, y, z]."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    T[:3, 3] = translation
    return T


def T_ego_from_sensor(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return make_transform(cs["translation"], cs["rotation"])


def T_global_from_ego(nusc, sd):
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return make_transform(pose["translation"], pose["rotation"])


def T_refego_from_sensor(nusc, sd, ref_sd):
    """Sensor frame at sd's time  ->  ego frame at ref_sd's time.

    The middle hop through the global frame is what compensates for the car
    having moved between the two timestamps (ego-motion compensation).
    """
    T_ref_from_global = np.linalg.inv(T_global_from_ego(nusc, ref_sd))
    return T_ref_from_global @ T_global_from_ego(nusc, sd) @ T_ego_from_sensor(nusc, sd)


def apply_transform(T, xyz):
    """Apply a 4x4 transform to points of shape (3, N). Returns (3, N)."""
    return T[:3, :3] @ xyz + T[:3, 3:4]


def rotate_vectors(T, v):
    """Velocities/directions only rotate - translation does not apply to them."""
    return T[:3, :3] @ v


def sd_path(nusc, sd):
    return os.path.join(nusc.dataroot, sd["filename"])


def ensure_out_dir(*parts):
    path = os.path.join(OUT_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path
