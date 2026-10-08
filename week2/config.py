"""Shared settings for Week 2. Everything else imports from here."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Where Week 1 (step6_build_dataset.py) wrote its cache.
CACHE_DIR = os.environ.get("BEV_CACHE",
                           os.path.join(HERE, "..", "outputs", "bev_cache"))
RUNS_DIR = os.path.join(HERE, "runs")        # checkpoints + training logs
OUT_DIR = os.path.join(HERE, "outputs")      # figures + result tables

MODALITIES = ["lidar", "radar", "camera"]    # order of the modality mask everywhere

# Must match the normalisation used in week1/step4_bev.py.
V_SCALE = 20.0                               # radar velocity channels = v / 20 m/s

# Training targets
MAX_OBJECTS = 128                            # padded object slots per sample
MIN_RADIUS = 2                               # min Gaussian radius in cells (0.8 m)
REG_DIMS = ["off_row", "off_col", "log_w", "log_l", "sin_yaw", "cos_yaw"]

# Evaluation: nuScenes-style matching by centre distance (metres)
MATCH_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
