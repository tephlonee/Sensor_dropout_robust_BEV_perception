"""
Week 2 - Step 3: Train the detector under three strategies.

  baseline : EarlyFusionNet, always sees all sensors.
             -> what most detectors do; expect a big drop when radar vanishes.
  dropout  : EarlyFusionNet, radar removed in 30 % of training samples.
             -> "modality dropout": forces the network to also work without radar.
  aware    : MidFusionNet (per-sensor encoders + gate + presence mask),
             also trained with 30 % radar dropout.
             -> "modality-aware learning": the model KNOWS when radar is gone.

The thesis question: can we make detection robust to missing radar WITHOUT
hurting normal all-sensor performance? Step 4 answers it from these 3 runs.

Run:
  python step3_train.py --config baseline
  python step3_train.py --config dropout
  python step3_train.py --config aware
  python step3_train.py --config baseline --overfit     # sanity check, see below

GPU strongly recommended (a free Colab/Kaggle GPU is fine): minutes per run.
On CPU the defaults take several hours per run; use --base 16 --epochs 20
(roughly 20 min for baseline/dropout, ~1 h for aware).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import RUNS_DIR, ensure_dir
from step1_dataset import BEVDataset
from step2_model import build_model, compute_loss, count_parameters

CONFIGS = {
    "baseline": dict(model="early", radar_dropout=0.0, radar_noise_p=0.0),
    "dropout": dict(model="early", radar_dropout=0.3, radar_noise_p=0.0),
    "aware": dict(model="mid", radar_dropout=0.3, radar_noise_p=0.0),
}


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        _, logs = compute_loss(model(batch["bev"], batch["mask"]), batch)
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v * len(batch["bev"])
        n += len(batch["bev"])
    return {k: v / n for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base", type=int, default=32, help="U-Net width (16 = faster, 64 = bigger)")
    p.add_argument("--radar-noise-p", type=float, default=None,
                   help="override: also train with noisy radar in this fraction of samples")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", action="store_true",
                   help="train on 4 samples only; the loss must fall close to 0 or something is broken")
    args = p.parse_args()

    cfg = dict(CONFIGS[args.config])
    if args.radar_noise_p is not None:
        cfg["radar_noise_p"] = args.radar_noise_p
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = BEVDataset("mini_train", train=not args.overfit,
                          radar_dropout=0.0 if args.overfit else cfg["radar_dropout"],
                          radar_noise_p=0.0 if args.overfit else cfg["radar_noise_p"])
    val_all = BEVDataset("mini_val", condition="all")
    val_noradar = BEVDataset("mini_val", condition="no_radar")
    if args.overfit:
        train_ds = Subset(train_ds, range(4))
        args.epochs, args.batch_size = 150, 4

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=not args.overfit,
                        num_workers=args.workers, pin_memory=device == "cuda",
                        persistent_workers=args.workers > 0)
    val_loaders = {"val_all": DataLoader(val_all, batch_size=args.batch_size, num_workers=args.workers),
                   "val_no_radar": DataLoader(val_noradar, batch_size=args.batch_size,
                                              num_workers=args.workers)}

    base_ds = val_all
    model = build_model(cfg["model"], len(base_ds.channels), len(base_ds.classes),
                        base_ds.slices, base=args.base).to(device)
    print(f"Config {args.config}: {cfg} | {count_parameters(model) / 1e6:.2f} M params | device {device}")
    print(f"Train samples {len(train_ds)}, val samples {len(val_all)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(loader), pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    run_dir = ensure_dir(os.path.join(RUNS_DIR, args.config + ("_overfit" if args.overfit else "")))
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, n = {}, 0
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                out = model(batch["bev"], batch["mask"])
            loss, logs = compute_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        row = {"epoch": epoch, **{f"train_{k}": v / n for k, v in sums.items()}}

        last = epoch == args.epochs
        if not args.overfit and (epoch % 5 == 0 or last):
            for name, vl in val_loaders.items():
                row[f"{name}_loss"] = validate(model, vl, device)["loss"]
        history.append(row)
        if args.overfit and epoch % 10 and not last:
            continue
        msg = " | ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "epoch")
        print(f"epoch {epoch:3d} | {msg} | {time.time() - t0:5.0f}s")

    torch.save({"config": cfg, "name": args.config, "base": args.base,
                "state_dict": model.state_dict()}, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    plot_history(history, os.path.join(run_dir, "loss_curves.png"), args.config)
    print(f"Saved model + history -> {run_dir}")
    if args.overfit:
        final = history[-1]["train_loss"]
        print("Overfit check:", "PASS - the pipeline can learn" if final < 0.5 * history[0]["train_loss"]
              else "loss barely moved - check data/targets before training for real")


def plot_history(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, [h["train_loss"] for h in history], label="train")
    for key, label in [("val_all_loss", "val: all sensors"), ("val_no_radar_loss", "val: radar missing")]:
        pts = [(h["epoch"], h[key]) for h in history if key in h]
        if pts:
            ax.plot(*zip(*pts), "o-", label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.set_title(f"Training curves - {title}"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
