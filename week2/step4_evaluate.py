"""
Week 2 - Step 4: Evaluate every trained model under every sensor condition.

This produces the thesis result: a table of
    rows    = training strategy (baseline / dropout / aware)
    columns = test condition    (all sensors, radar missing, radar noisy at
                                 3 severities, camera missing)
plus the relative drop vs. the all-sensor case.

Metric - nuScenes-style centre-distance mAP (simplified)
-------------------------------------------------------
A detection counts as correct (true positive) if its centre lies within d
metres of a not-yet-matched ground-truth object of the same class, for
d in {0.5, 1, 2, 4} m. Detections are matched in order of confidence.
AP = area under the precision-recall curve (101-point interpolation);
mAP = mean over classes and distance thresholds.
(The official nuScenes metric also clips low recall/precision and adds
box-error terms; this keeps the same matching idea in ~30 lines.)

Run:  python step4_evaluate.py              # all runs in runs/
      python step4_evaluate.py --runs baseline aware
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import MATCH_THRESHOLDS, OUT_DIR, RUNS_DIR, ensure_dir
from step1_dataset import BEVDataset
from step2_model import build_model, decode

CONDITIONS = [("all", 0.0), ("no_radar", 0.0), ("noisy_radar", 0.25),
              ("noisy_radar", 0.5), ("noisy_radar", 1.0), ("no_camera", 0.0)]


def cond_name(cond, sev):
    return f"{cond}@{sev:g}" if cond == "noisy_radar" else cond


# ---------------------------------------------------------------------------
# 1. Average precision by centre distance
# ---------------------------------------------------------------------------
def average_precision(dets, gts, threshold):
    """dets: (N, 4) array of sample_id, score, x, y   (one class)
       gts : dict sample_id -> (M, 2) array of x, y  (same class)"""
    n_gt = sum(len(g) for g in gts.values())
    if n_gt == 0:
        return float("nan")
    if len(dets) == 0:
        return 0.0
    dets = dets[np.argsort(-dets[:, 1], kind="stable")]
    taken = {s: np.zeros(len(g), bool) for s, g in gts.items()}
    tp = np.zeros(len(dets))
    for j, (s, _score, x, y) in enumerate(dets):
        g = gts.get(int(s))
        if g is None or len(g) == 0:
            continue
        d = np.hypot(g[:, 0] - x, g[:, 1] - y)
        d[taken[int(s)]] = np.inf
        m = int(np.argmin(d))
        if d[m] <= threshold:
            tp[j] = 1
            taken[int(s)][m] = True
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]     # best precision at >= this recall
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(recall, r, side="left")
        ap += envelope[k] if k < len(envelope) else 0.0
    return ap / 101


def compute_map(all_dets, all_gts, classes):
    """all_dets: list over samples of (N, 7) tensors from decode().
       all_gts : list over samples of (MAX, 4) gt arrays (class, x, y, yaw; -1 = padding)."""
    per_class = {}
    for c, name in enumerate(classes):
        dets = [np.column_stack([np.full(int((d[:, 0] == c).sum()), s), d[d[:, 0] == c][:, 1:4].numpy()])
                for s, d in enumerate(all_dets)]
        dets = np.concatenate(dets) if dets else np.zeros((0, 4))
        gts = {s: g[g[:, 0] == c][:, 1:3] for s, g in enumerate(all_gts)}
        per_class[name] = {t: average_precision(dets, gts, t) for t in MATCH_THRESHOLDS}
    values = [v for d in per_class.values() for v in d.values() if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan"), per_class


# ---------------------------------------------------------------------------
# 2. Run a model over a test condition
# ---------------------------------------------------------------------------
def load_run(path, ds, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"], len(ds.channels), len(ds.classes),
                        ds.slices, base=ckpt.get("base", 32)).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return ckpt["name"], model


@torch.no_grad()
def run_condition(model, cond, sev, device, batch_size=8, workers=2):
    ds = BEVDataset("mini_val", condition=cond, severity=sev)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    dets, gts = [], []
    for batch in loader:
        out = model(batch["bev"].to(device), batch["mask"].to(device))
        dets += decode(out, ds.grid)
        gts += [g.numpy() for g in batch["gt"]]
    return compute_map(dets, gts, ds.classes)


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def plot_robustness(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    sevs = [0.0, 0.25, 0.5, 1.0]
    for name, row in results.items():
        ys = [row["all"]] + [row[cond_name("noisy_radar", s)] for s in sevs[1:]]
        line, = ax.plot(sevs, ys, "o-", label=name)
        ax.plot([1.25], [row["no_radar"]], "s", color=line.get_color(), ms=9)
    ax.set_xticks(sevs + [1.25]); ax.set_xticklabels(["0\n(clean)", "0.25", "0.5", "1.0", "radar\nmissing"])
    ax.set_xlabel("radar corruption severity"); ax.set_ylabel("mAP")
    ax.set_title("Detection robustness to degraded / missing radar")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


@torch.no_grad()
def plot_qualitative(models, device, path, sample_idx=None):
    """One val sample: predicted heatmaps for each model x condition, GT centres as circles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [("all", 0.0), ("no_radar", 0.0), ("noisy_radar", 1.0)]
    fig, axes = plt.subplots(len(models), len(conds), figsize=(5 * len(conds), 5 * len(models)),
                             squeeze=False)
    for ci, (cond, sev) in enumerate(conds):
        ds = BEVDataset("mini_val", condition=cond, severity=sev)
        j = len(ds) // 2 if sample_idx is None else sample_idx
        s = ds[j]
        gt = s["gt"].numpy(); gt = gt[gt[:, 0] >= 0]
        rows, cols = ds.grid.to_pixel(gt[:, 1], gt[:, 2])
        lidar = s["bev"][ds.channels.index("lidar_max_z")].numpy()
        for mi, (name, model) in enumerate(models.items()):
            out = model(s["bev"][None].to(device), s["mask"][None].to(device))
            heat = torch.sigmoid(out["hm"][0]).max(0).values.cpu().numpy()
            ax = axes[mi, ci]
            ax.imshow(lidar, cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_less(heat, 0.1), cmap="autumn", vmin=0, vmax=1, alpha=0.85)
            ax.scatter(cols - 0.5, rows - 0.5, s=60, facecolors="none", edgecolors="cyan", lw=1)
            ax.set_title(f"{name} | {cond_name(cond, sev)}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Predicted centre heatmap (orange) vs ground truth (cyan circles)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


@torch.no_grad()
def plot_gates(model, device, path):
    """For the modality-aware model: where does it trust each sensor?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [("all", 0.0), ("noisy_radar", 1.0), ("no_radar", 0.0)]
    fig, axes = plt.subplots(len(conds), 3, figsize=(15, 5 * len(conds)))
    for ri, (cond, sev) in enumerate(conds):
        ds = BEVDataset("mini_val", condition=cond, severity=sev)
        s = ds[len(ds) // 2]
        gates = model(s["bev"][None].to(device), s["mask"][None].to(device))["gates"][0].cpu().numpy()
        for mi, mod in enumerate(["lidar", "radar", "camera"]):
            ax = axes[ri, mi]
            im = ax.imshow(gates[mi], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{cond_name(cond, sev)}: trust in {mod} (mean {gates[mi].mean():.2f})")
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(path, dpi=90); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", help="run names in runs/ (default: all except *_overfit)")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*", "model.pt")))
    paths = [q for q in paths if not q.split(os.sep)[-2].endswith("_overfit")]
    if args.runs:
        paths = [q for q in paths if q.split(os.sep)[-2] in args.runs]
    if not paths:
        raise SystemExit("No trained runs found - run step3_train.py first.")

    ref_ds = BEVDataset("mini_val")
    models = dict(load_run(q, ref_ds, device) for q in paths)
    names = [cond_name(c, s) for c, s in CONDITIONS]

    results, per_class = {}, {}
    for name, model in models.items():
        results[name] = {}
        for cond, sev in CONDITIONS:
            m, pc = run_condition(model, cond, sev, device, workers=args.workers)
            results[name][cond_name(cond, sev)] = m
            per_class[(name, cond_name(cond, sev))] = pc
            print(f"  {name:10s} {cond_name(cond, sev):17s} mAP {m:.3f}")

    out_dir = ensure_dir(OUT_DIR)
    # --- table: absolute mAP and relative change vs all sensors ---
    lines = ["| model | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for name, row in results.items():
        cells = [f"{row['all']:.3f}"]
        for n in names[1:]:
            rel = (row[n] - row["all"]) / row["all"] * 100 if row["all"] > 0 else float("nan")
            cells.append(f"{row[n]:.3f} ({rel:+.0f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Per-class AP (averaged over distance thresholds):", "",
              "| model | condition | " + " | ".join(ref_ds.classes) + " |",
              "|---" * (2 + len(ref_ds.classes)) + "|"]
    for (name, cond), pc in per_class.items():
        lines.append(f"| {name} | {cond} | " + " | ".join(
            f"{np.nanmean(list(pc[c].values())):.3f}" for c in ref_ds.classes) + " |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + names)
        for name, row in results.items():
            w.writerow([name] + [f"{row[n]:.4f}" for n in names])
    print("\n" + table)

    plot_robustness(results, os.path.join(out_dir, "robustness.png"))
    plot_qualitative(models, device, os.path.join(out_dir, "qualitative.png"))
    if "aware" in models:
        plot_gates(models["aware"], device, os.path.join(out_dir, "aware_gates.png"))
    print(f"\nSaved results.md, results.csv and figures -> {out_dir}")


if __name__ == "__main__":
    main()
