# Week 2 — BEV detector + the missing-RADAR experiment

Trains a CenterNet-style BEV object detector on the Week 1 cache and answers
the question from the Volvo thesis *"Robust Multimodal 3D Object Detection with
RADAR under Missing-Modality Conditions"*:

> Can detection be made robust to missing or corrupted RADAR **without**
> hurting normal all-sensor performance?

## Prerequisite

`week1/step6_build_dataset.py` has produced `week1/outputs/bev_cache/`.
If your cache is elsewhere: `export BEV_CACHE=/path/to/bev_cache`.

## Run order

| Step | File | What it does |
|---|---|---|
| 1 | `step1_dataset.py` | Cache → (input, target) pairs. Builds Gaussian centre heatmaps + regression targets, and simulates sensor failures (`no_radar`, `noisy_radar`, `no_camera`). Saves a figure of targets and conditions. |
| 2 | `step2_model.py` | The two architectures, the focal loss and the decoder. Prints shapes and parameter counts. |
| 3 | `step3_train.py` | Trains one strategy. Run `--overfit` first as a sanity check. |
| 4 | `step4_evaluate.py` | Every trained model × every condition → `outputs/results.md`, `robustness.png`, `qualitative.png`, `aware_gates.png`. |

```bash
python step1_dataset.py
python step2_model.py
python step3_train.py --config baseline --overfit   # must print PASS
bash run_week2.sh                                     # trains all 3 + evaluates
```

GPU strongly recommended (a free Colab/Kaggle GPU is fine, a few minutes per run).
On CPU, use `bash run_week2.sh --base 16 --epochs 20` (about 20 min per early-fusion run, ~1 h for `aware`).

## The three training strategies

| Config | Architecture | Training | Idea |
|---|---|---|---|
| `baseline` | Early fusion: 11 channels → one U-Net | always all sensors | What most detectors do |
| `dropout` | Early fusion | radar removed in 30 % of samples | Modality dropout |
| `aware` | Mid fusion: per-sensor encoders → learned per-cell gate + presence mask → U-Net | radar removed in 30 % | Modality-aware learning: the model *knows* radar is gone |

Optional: `--radar-noise-p 0.3` also trains with corrupted radar.

## Test conditions

| Condition | What happens | Model told? |
|---|---|---|
| `all` | clean input | — |
| `no_radar` | radar channels zeroed | yes (mask = 0) — a dead sensor is detectable |
| `noisy_radar@s` | s ∈ {0.25, 0.5, 1}: real detections dropped, velocities jittered, ghost detections added | **no** — corrupted data usually looks valid |
| `no_camera` | camera channels zeroed | yes |

Corruption is seeded per sample, so every model sees identical inputs.

## Reading the results

* **all-sensor column** — did robustness training cost normal performance?
  (The thesis requires it not to.)
* **no_radar drop** — the headline number. Expect `baseline` to drop most.
* **noisy_radar curve** — often worse than missing radar for the baseline:
  it trusts ghost detections it can't distinguish from real ones.
* **aware_gates.png** — where the gated model trusts each sensor. Look for
  higher radar trust far away (where LiDAR is sparse) and lower trust when
  radar is noisy.

## Honest caveats (say these before an interviewer does)

* nuScenes-mini has ~320 train / ~80 val samples from 10 scenes. Absolute mAP
  will be low and noisy; **compare models relative to each other**, and repeat
  with 2–3 seeds (`--seed`) before trusting a difference of a few points.
* The metric is a simplified nuScenes-style centre-distance mAP, not the
  official NDS.
* "Without radar" here means LiDAR + camera: the camera channels are painted
  LiDAR points, so there is no true camera-only fallback in this setup.
* Results are reported on `mini_val` with the last checkpoint (no model selection on val).

## Implementation note worth knowing

The per-sensor encoders in `MidFusionNet` use **GroupNorm, not BatchNorm**.
With BatchNorm the gated model trained normally but its all-sensor loss was
4× worse at evaluation: radar BEV is >99 % empty, so its feature variance is
tiny, and BatchNorm's running statistics (used at test time) drift far from
the per-batch statistics seen in training — made worse by modality dropout
feeding all-zero radar batches. GroupNorm normalises per sample, so training
and test behave the same.

## What Week 3 reuses

`MidFusionNet.encode()` returns per-sensor BEV features. Week 3 trains a
generative model to predict the radar features from LiDAR + camera features,
and feeds them to the frozen detector when radar is missing.
