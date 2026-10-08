"""
Week 2 - Step 2: The detector networks, the loss, and decoding detections.

Two architectures (both end in the same CenterNet-style head)
------------------------------------------------------------
EarlyFusionNet  - stacks all 11 BEV channels and feeds them to one U-Net.
                  Fusion = "the first convolution mixes the sensors".
                  If radar is missing, its input channels are just zeros and
                  the network has no idea whether that means "no objects" or
                  "no sensor".

MidFusionNet    - "modality-aware": each sensor gets its own small encoder,
                  then a learned GATE decides, per BEV cell, how much to trust
                  each sensor's features. The network is also told which
                  sensors are present (the modality mask). Missing radar is
                  then an explicit, known state instead of silent zeros.
                  The per-sensor features from `encode()` are exactly what
                  Week 3's generative model will learn to reconstruct.

Head (CenterNet / CenterPoint idea)
-----------------------------------
For every BEV cell the network predicts
  * heatmap  : probability that an object CENTRE is in this cell (per class)
  * regression (6 numbers): sub-cell offset (2), log width/length (2), sin/cos heading (2)
At test time we take the local maxima of the heatmap as detections and read
the regression values at those cells.

Run:  python step2_model.py      (shape check + parameter counts)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODALITIES, REG_DIMS


# ---------------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------------
def make_norm(kind, channels):
    return nn.BatchNorm2d(channels) if kind == "bn" else nn.GroupNorm(8, channels)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, norm="bn"):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), make_norm(norm, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), make_norm(norm, cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = ConvBlock(2 * cout, cout)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNet(nn.Module):
    """256x256 -> 32x32 bottleneck -> back to 256x256.
    Downsampling gives each cell a large receptive field (context: "this blob
    sits on a road"), skip connections keep fine localisation."""

    def __init__(self, cin, base=32, depth=3):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.inc = ConvBlock(cin, ch[0])
        self.downs = nn.ModuleList(nn.Sequential(nn.MaxPool2d(2), ConvBlock(ch[i], ch[i + 1]))
                                   for i in range(depth))
        self.ups = nn.ModuleList(Up(ch[i + 1], ch[i]) for i in reversed(range(depth)))
        self.out_channels = ch[0]

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return x


class CenterHead(nn.Module):
    def __init__(self, cin, n_classes):
        super().__init__()
        self.hm = nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(cin, n_classes, 1))
        self.reg = nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(cin, len(REG_DIMS), 1))
        # Start by predicting ~10 % everywhere (CenterNet trick): almost every
        # cell is background, so a 50 % start would explode the initial loss.
        nn.init.constant_(self.hm[-1].bias, -math.log((1 - 0.1) / 0.1))

    def forward(self, x):
        return {"hm": self.hm(x), "reg": self.reg(x)}


# ---------------------------------------------------------------------------
# 2. The two fusion architectures
# ---------------------------------------------------------------------------
class EarlyFusionNet(nn.Module):
    def __init__(self, in_channels, n_classes, base=32, **_):
        super().__init__()
        self.backbone = UNet(in_channels, base)
        self.head = CenterHead(self.backbone.out_channels, n_classes)

    def forward(self, bev, mask=None):            # mask ignored: the model can't use it
        return self.head(self.backbone(bev))


class MidFusionNet(nn.Module):
    def __init__(self, in_channels, n_classes, slices, base=32, feat=32, **_):
        super().__init__()
        self.slices = slices
        # GroupNorm, not BatchNorm, in the per-sensor encoders. Radar input is
        # >99 % empty cells, so its feature variance is tiny and BatchNorm's
        # running statistics (used at test time) end up far from the per-batch
        # statistics used in training - especially with modality dropout feeding
        # all-zero batches. The result is a model that trains fine and fails at
        # evaluation. GroupNorm normalises within each sample: no train/test gap.
        self.encoders = nn.ModuleDict({m: ConvBlock(slices[m].stop - slices[m].start, feat, norm="gn")
                                       for m in MODALITIES})
        n_mod = len(MODALITIES)
        # Gate: sees all (masked) features + which sensors are present,
        # outputs one trust weight per sensor per cell.
        self.gate = nn.Sequential(nn.Conv2d(n_mod * feat + n_mod, 32, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, n_mod, 1))
        self.backbone = UNet(n_mod * feat + n_mod, base)
        self.head = CenterHead(self.backbone.out_channels, n_classes)

    def encode(self, bev, mask):
        """Per-sensor BEV features. A missing sensor's features are forced to zero."""
        return {m: self.encoders[m](bev[:, self.slices[m]]) * mask[:, i, None, None, None]
                for i, m in enumerate(MODALITIES)}

    def fuse(self, feats, mask):
        B, _, H, W = next(iter(feats.values())).shape
        planes = mask[:, :, None, None].expand(B, len(MODALITIES), H, W)
        f = [feats[m] for m in MODALITIES]
        gates = torch.sigmoid(self.gate(torch.cat(f + [planes], dim=1))) * planes
        fused = torch.cat([fm * gates[:, i:i + 1] for i, fm in enumerate(f)] + [planes], dim=1)
        return fused, gates

    def forward(self, bev, mask):
        fused, gates = self.fuse(self.encode(bev, mask), mask)
        out = self.head(self.backbone(fused))
        out["gates"] = gates
        return out


def build_model(kind, in_channels, n_classes, slices, base=32):
    if kind == "early":
        return EarlyFusionNet(in_channels, n_classes, base=base)
    if kind == "mid":
        return MidFusionNet(in_channels, n_classes, slices, base=base)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 3. Loss
# ---------------------------------------------------------------------------
def focal_loss(logits, target):
    """CenterNet's penalty-reduced focal loss.
    Positives: cells where target == 1 (object centres).
    Negatives: all other cells, down-weighted by (1 - target)^4 so cells right
    next to a centre are barely punished for being 'almost' a centre."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg_weight = (1 - target) ** 4
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * neg_weight * (1 - pos)
    num_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def gather_at(feature_map, ind):
    """(B, C, H, W) + flat indices (B, K) -> (B, K, C)."""
    B, C = feature_map.shape[:2]
    flat = feature_map.view(B, C, -1)
    return flat.gather(2, ind[:, None, :].expand(B, C, ind.shape[1])).permute(0, 2, 1)


REG_WEIGHTS = torch.tensor([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])


def compute_loss(out, batch, reg_weight=1.0):
    hm_loss = focal_loss(out["hm"], batch["heatmap"])
    pred = gather_at(out["reg"].float(), batch["ind"])
    m = batch["reg_mask"][..., None]
    l1 = (pred - batch["reg"]).abs() * m * REG_WEIGHTS.to(pred.device)
    reg_loss = l1.sum() / (m.sum() * len(REG_DIMS)).clamp(min=1)
    total = hm_loss + reg_weight * reg_loss
    return total, {"loss": total.item(), "hm": hm_loss.item(), "reg": reg_loss.item()}


# ---------------------------------------------------------------------------
# 4. Decoding: network output -> list of detections in metres
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(out, grid, top_k=100, score_thresh=0.01):
    """Returns, per sample, a tensor (N, 7): class, score, x, y, w, l, yaw (ego frame).

    A 3x3 max-pool keeps only local peaks - a cheap replacement for NMS that
    works because each object is ONE peak in the heatmap.
    """
    heat = torch.sigmoid(out["hm"].float())
    peaks = heat == F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * peaks
    B, C, H, W = heat.shape
    scores, idx = heat.view(B, -1).topk(min(top_k, C * H * W))
    cls = idx // (H * W)
    pix = idx % (H * W)
    rows, cols = (pix // W).float(), (pix % W).float()
    reg = gather_at(out["reg"].float(), pix)                       # (B, K, 6)
    row_f, col_f = rows + reg[..., 0], cols + reg[..., 1]
    x = grid.x_max - row_f * grid.res
    y = grid.y_max - col_f * grid.res
    w, l = reg[..., 2].exp(), reg[..., 3].exp()
    yaw = torch.atan2(reg[..., 4], reg[..., 5])
    dets = torch.stack([cls.float(), scores, x, y, w, l, yaw], dim=-1)
    return [d[d[:, 1] >= score_thresh].cpu() for d in dets]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    slices = {"lidar": slice(0, 4), "radar": slice(4, 8), "camera": slice(8, 11)}
    bev = torch.randn(2, 11, 256, 256)
    mask = torch.tensor([[1., 1., 1.], [1., 0., 1.]])        # 2nd sample: radar missing
    for kind in ["early", "mid"]:
        model = build_model(kind, 11, 2, slices).eval()
        out = model(bev, mask)
        print(f"{kind:5s}: {count_parameters(model) / 1e6:.2f} M params | "
              + " | ".join(f"{k} {tuple(v.shape)}" for k, v in out.items()))
        if kind == "mid":
            g = out["gates"]
            print(f"       radar gate mean: sample 1 {g[0, 1].mean():.2f}, "
                  f"sample 2 (radar missing) {g[1, 1].mean():.2f}  <- forced to 0")


if __name__ == "__main__":
    main()
