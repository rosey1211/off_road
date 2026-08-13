# ─── off_road/model.py ────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Spatial size of the projected feature map (after bilinear upsampling from
# the backbone's native stride-32 output).  Chosen so all four row fractions
# map to distinct row indices (see comments in forward()).
_FEAT_H = 24
_FEAT_W = 32
_FEAT_C = 64   # channels after the 1×1 projection


class TraversabilityCNN(nn.Module):
    """
    Camera-only traversability estimator.

    Input : (B, 3, H, W) — normalised RGB image (any resolution; resized internally)
    Output: (B, N_ROWS, N_POINTS) — per-point traversability scores in [0, 1]
            1.0 = most traversable, 0.0 = least traversable.

    Architecture
    ────────────
    1. MobileNetV3-Small backbone → (B, 576, ~H/32, ~W/32)
    2. Bilinear upsample to (_FEAT_H, _FEAT_W) for stable row indexing
    3. 1×1 conv  576 → _FEAT_C=64  (channel reduction, shared)
    4. Per-row extraction: feat[:, :, row_idx, :] → (B, 64, _FEAT_W)
    5. Bilinear upsample along width  _FEAT_W → N_POINTS
    6. Per-row Conv1d heads → (B, 1, N_POINTS) → sigmoid → (B, N_POINTS)
    7. Stack rows → (B, N_ROWS, N_POINTS)
    """

    def __init__(self, image_width, image_height, n_rows, n_points,
                 row_fractions, norm_mean, norm_std):
        super().__init__()

        self.image_width   = image_width
        self.image_height  = image_height
        self.n_rows        = n_rows
        self.n_points      = n_points
        self.row_fractions = row_fractions
        self.norm_mean     = norm_mean
        self.norm_std      = norm_std

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = backbone.features   # (B, 576, ~H/32, ~W/32)

        self.chan_proj = nn.Sequential(
            nn.Conv2d(576, _FEAT_C, 1, bias=False),
            nn.BatchNorm2d(_FEAT_C),
            nn.ReLU(inplace=True),
        )

        # Per-row 1-D conv heads.
        # Input width at this stage = N_POINTS (after bilinear upsampling).
        self.row_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(_FEAT_C, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            for _ in range(n_rows)
        ])

    def forward(self, x):
        # Resize to model's training resolution if needed
        if x.shape[-2:] != (self.image_height, self.image_width):
            x = F.interpolate(x, size=(self.image_height, self.image_width),
                              mode='bilinear', align_corners=False)

        feat = self.backbone(x)   # (B, 576, fH, fW)

        # Upsample to a fixed spatial grid so row_fractions map to distinct rows.
        # row fraction 0.45→row7, 0.55→row8, 0.65→row10, 0.75→row12  (all distinct)
        feat = F.interpolate(feat, size=(_FEAT_H, _FEAT_W),
                             mode='bilinear', align_corners=False)

        feat = self.chan_proj(feat)   # (B, 64, _FEAT_H, _FEAT_W)

        outs = []
        for r in range(self.n_rows):
            ri = min(int(self.row_fractions[r] * _FEAT_H), _FEAT_H - 1)
            row = feat[:, :, ri, :]   # (B, 64, _FEAT_W)

            # Upsample width _FEAT_W → N_POINTS to preserve lateral correspondence
            row_up = F.interpolate(
                row.unsqueeze(2),                        # (B, 64, 1, _FEAT_W)
                size=(1, self.n_points),
                mode='bilinear', align_corners=False,
            ).squeeze(2)                                 # (B, 64, N_POINTS)

            out = self.row_heads[r](row_up)   # (B, 1, N_POINTS)
            outs.append(out.squeeze(1))        # (B, N_POINTS)

        return torch.stack(outs, dim=1)   # (B, N_ROWS, N_POINTS)


def traversability_loss(pred, target):
    """
    MSE loss between predicted and ground-truth traversability maps.

    pred   : (B, N_ROWS, N_POINTS)  model output, values in [0, 1]
    target : (B, N_ROWS, N_POINTS)  ground truth, values in [0, 1]
    Returns scalar loss.
    """
    return F.mse_loss(pred, target)
