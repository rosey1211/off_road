# ─── off_road/live_infer.py ───────────────────────────────────────────────────
"""
Visualise traversability predictions and best paths on dataset images.

Controls:
  Space / Right-arrow  — next image
  Left-arrow           — previous image
  q / Esc              — quit
  g                    — toggle ground-truth dot overlay
  p                    — toggle path overlay
"""
import os
import re
import sys
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd

import config
config.init()

import infer_config as cfg

from model import TraversabilityCNN

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=cfg.CHECKPOINT)
parser.add_argument("--dataset",    default=cfg.DATASET_DIR)
parser.add_argument("--delay",      type=float, default=cfg.DELAY,
                    help="Seconds between frames (0 = wait for keypress)")
parser.add_argument("--no-gt",      action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
model = TraversabilityCNN(
    image_width   = ckpt["image_width"],
    image_height  = ckpt["image_height"],
    n_rows        = ckpt["n_rows"],
    n_points      = ckpt["n_points"],
    row_fractions = ckpt["row_fractions"],
    norm_mean     = ckpt["norm_mean"],
    norm_std      = ckpt["norm_std"],
)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()

ROW_FRACTIONS = ckpt["row_fractions"]
N_ROWS        = ckpt["n_rows"]
N_POINTS      = ckpt["n_points"]

print(f"Loaded checkpoint: epoch {ckpt['epoch']}  →  {args.checkpoint}")
print(f"  N_ROWS={N_ROWS}  N_POINTS={N_POINTS}  fractions={ROW_FRACTIONS}")

transform = T.Compose([
    T.Resize((ckpt["image_height"], ckpt["image_width"])),
    T.ToTensor(),
    T.Normalize(mean=ckpt["norm_mean"], std=ckpt["norm_std"]),
])

# ── Load dataset ──────────────────────────────────────────────────────────────
def _is_augmented(fname):
    stem = os.path.splitext(fname)[0]
    # Match suffix optionally followed by digits at end of stem,
    # e.g. catches both "image_sl.png" and "image_sl0040.png"
    return any(re.search(re.escape(s) + r'\d*$', stem) for s in cfg.EXCLUDE_SUFFIXES)

labels_csv = os.path.join(args.dataset, "offroad_labels.csv")
if os.path.exists(labels_csv):
    df        = pd.read_csv(labels_csv)
    df        = df[~df["filename"].apply(_is_augmented)].reset_index(drop=True)
    filenames = df["filename"].tolist()
    has_labels = True
else:
    filenames  = sorted(f for f in os.listdir(args.dataset)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                        and not _is_augmented(f))
    df         = None
    has_labels = False

print(f"Dataset: {args.dataset}  ({len(filenames)} images)")
if not filenames:
    print("No images found. Exiting.")
    sys.exit(1)


# ── Colour helper ─────────────────────────────────────────────────────────────
def _trav_to_bgr(score: float):
    """Traversability 0→1 maps red→yellow→green (BGR)."""
    score = float(np.clip(score, 0.0, 1.0))
    if score <= 0.5:
        t = score * 2.0
        return (0, int(255 * t), 255)
    else:
        t = (score - 0.5) * 2.0
        return (0, 255, int(255 * (1.0 - t)))


# ── Ground-truth dot overlay ──────────────────────────────────────────────────
def _draw_gt_dots(canvas, gt, dot_radius=5):
    """Filled coloured dots on each scan line — green=traversable, red=not."""
    h, w = canvas.shape[:2]
    for r in range(N_ROWS):
        row_y = int(ROW_FRACTIONS[r] * h)
        for p in range(N_POINTS):
            x = int((p + 0.5) / N_POINTS * w)
            cv2.circle(canvas, (x, row_y), dot_radius,
                       _trav_to_bgr(float(gt[r, p])), -1, cv2.LINE_AA)


# ── Prediction bar-chart overlay ──────────────────────────────────────────────
def _draw_pred_bars(canvas, pred):
    """
    For each scan-line point draw a vertical bar centred on the scan line.
      score > 0.5 → green bar rising  above the scan line
      score < 0.5 → red   bar falling below  the scan line
    Bar height is proportional to |score − 0.5| × 2, scaled so a score of
    1.0 or 0.0 reaches BAR_MAX_HEIGHT_FRAC × image height.
    """
    h, w      = canvas.shape[:2]
    max_bar_h = int(cfg.BAR_MAX_HEIGHT_FRAC * h)
    slot_w    = max(1, w // N_POINTS)
    bar_w     = max(1, slot_w - 1)   # 1-px gap between adjacent bars

    # Draw bars on a blended overlay
    overlay = canvas.copy()
    for r in range(N_ROWS):
        row_y = int(ROW_FRACTIONS[r] * h)
        for p in range(N_POINTS):
            score  = float(pred[r, p])
            bar_h  = int(abs(score - 0.5) * 2.0 * max_bar_h)
            if bar_h == 0:
                continue
            cx = int((p + 0.5) / N_POINTS * w)
            x1 = max(0, cx - bar_w // 2)
            x2 = min(w - 1, x1 + bar_w)
            if score > 0.5:
                y1, y2 = max(0, row_y - bar_h), row_y
                color  = (0, 210, 0)
            else:
                y1, y2 = row_y, min(h - 1, row_y + bar_h)
                color  = (0, 0, 210)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, cfg.BAR_ALPHA, canvas, 1.0 - cfg.BAR_ALPHA, 0, canvas)

    # Draw scan lines on top of bars, thickness scaled with image width
    scale      = max(1.0, w / cfg.PATH_THICKNESS_REF_WIDTH)
    line_thick = max(1, round(scale))
    for r in range(N_ROWS):
        row_y = int(ROW_FRACTIONS[r] * h)
        cv2.line(canvas, (0, row_y), (w, row_y), (128, 0, 128), line_thick)


# ── Path finding ──────────────────────────────────────────────────────────────
def find_best_paths(pred):
    """
    Enumerate paths from the middle 20% of the lowest scan line (r0) upward
    through all scan lines, respecting lateral continuity.

    Scoring: mean traversability − CENTER_BIAS_WEIGHT × mean centre distance.
    Traversability is the primary constraint; centre preference is a weak tie-
    breaker that never overrides a higher-traversability path.

    Returns list of (indices, mean_trav, score) sorted best-first, length ≤ N_PATHS.
    indices = [p0, p1, p2, p3], one point index per row (r0→r3).
    """
    max_step = max(1, int(cfg.MAX_LATERAL_STEP_FRAC * N_POINTS))

    # Middle START_MIDDLE_FRAC of r0
    half_margin = (1.0 - cfg.START_MIDDLE_FRAC) / 2.0
    p_lo = int(half_margin * N_POINTS)
    p_hi = int((1.0 - half_margin) * N_POINTS)

    def _sgn(x): return (1 if x > 0 else -1) if x != 0 else 0

    def _enumerate(require_r3_traversable):
        results = []
        for p0 in range(p_lo, p_hi):
            t0  = float(pred[0, p0])
            cx0 = abs((p0 + 0.5) / N_POINTS - 0.5)
            if t0 < cfg.TRAV_THRESHOLD:
                continue

            for p1 in range(max(0, p0 - max_step), min(N_POINTS - 1, p0 + max_step) + 1):
                t1  = float(pred[1, p1])
                cx1 = abs((p1 + 0.5) / N_POINTS - 0.5)
                if t1 < cfg.TRAV_THRESHOLD:
                    continue

                for p2 in range(max(0, p1 - max_step), min(N_POINTS - 1, p1 + max_step) + 1):
                    t2  = float(pred[2, p2])
                    cx2 = abs((p2 + 0.5) / N_POINTS - 0.5)
                    if t2 < cfg.TRAV_THRESHOLD:
                        continue

                    for p3 in range(max(0, p2 - max_step), min(N_POINTS - 1, p2 + max_step) + 1):
                        t3  = float(pred[3, p3])
                        cx3 = abs((p3 + 0.5) / N_POINTS - 0.5)
                        if require_r3_traversable and t3 < cfg.TRAV_THRESHOLD:
                            continue

                        mean_trav = (t0 + t1 + t2 + t3) / N_ROWS
                        mean_cx   = (cx0 + cx1 + cx2 + cx3) / N_ROWS
                        lat_dev   = (abs(p1-p0) + abs(p2-p1) + abs(p3-p2)) / (3.0 * max_step)
                        d1, d2, d3 = p1-p0, p2-p1, p3-p2
                        rev = sum(1 for a, b in ((_sgn(d1), _sgn(d2)), (_sgn(d2), _sgn(d3)))
                                  if a != 0 and b != 0 and a != b)
                        dir_change       = rev / 2.0
                        start_off_center = cx0 * 2.0
                        score = (mean_trav
                                 - cfg.CENTER_BIAS_WEIGHT       * mean_cx
                                 - cfg.STRAIGHT_BIAS_WEIGHT     * lat_dev
                                 - cfg.DIRECTION_CHANGE_WEIGHT  * dir_change
                                 - cfg.START_CENTER_BIAS_WEIGHT * start_off_center)
                        results.append(([p0, p1, p2, p3], mean_trav, score))
        return results

    # Strict pass: all 4 rows must be traversable
    paths = _enumerate(require_r3_traversable=True)

    # Fallback: relax r3 if strict pass found nothing
    if not paths:
        paths = _enumerate(require_r3_traversable=False)

    if not paths:
        return []

    # Sort by score (traversability-dominant)
    paths.sort(key=lambda x: x[2], reverse=True)

    # Deduplicate into a ranked list, then stride by PATH_RANK_SKIP
    seen   = set()
    ranked = []
    needed = cfg.N_PATHS * max(1, cfg.PATH_RANK_SKIP)
    for indices, mean_trav, score in paths:
        key = tuple(indices)
        if key not in seen:
            seen.add(key)
            ranked.append((indices, mean_trav, score))
            if len(ranked) >= needed:
                break

    stride = max(1, cfg.PATH_RANK_SKIP)
    return ranked[::stride][:cfg.N_PATHS]


# ── Path drawing ──────────────────────────────────────────────────────────────
def _path_pixel_pts(indices, img_h, img_w, n_rows_to_use, n_samples=80):
    """
    Fit a polynomial through the first n_rows_to_use scan-line waypoints and
    return a dense array of (x, y) pixel coordinates for cv2.polylines.
    """
    n = n_rows_to_use
    ys = np.array([ROW_FRACTIONS[r] * img_h for r in range(n)], dtype=float)
    xs = np.array([(indices[r] + 0.5) / N_POINTS * img_w for r in range(n)], dtype=float)

    coeffs   = np.polyfit(ys, xs, min(3, n - 1))
    y_sample = np.linspace(ys[0], ys[-1], n_samples)
    x_sample = np.polyval(coeffs, y_sample).clip(0, img_w - 1)

    pts = np.stack([x_sample, y_sample], axis=1).astype(np.int32)
    return pts


def _draw_paths(canvas, top_paths, pred):
    """
    Draw N ranked paths onto canvas using alpha blending.
    Best path (rank 0) is most opaque; subsequent paths fade toward ALPHA_MIN.
    If the r3 (farthest) point is untraversable the curve is only drawn up to r2.
    """
    if not top_paths:
        return

    h, w  = canvas.shape[:2]
    n     = len(top_paths)
    scale = max(1.0, w / cfg.PATH_THICKNESS_REF_WIDTH)
    best_thickness = max(1, round(cfg.PATH_LINE_THICKNESS * scale))
    sub_thickness  = max(1, round(scale))

    # Draw worst → best so the best path lands on top
    for rank in range(n - 1, -1, -1):
        indices, mean_trav, score = top_paths[rank]

        # If r3 is untraversable, draw only r0→r2
        r3_trav = float(pred[N_ROWS - 1, indices[N_ROWS - 1]])
        n_rows_to_use = (N_ROWS - 1) if r3_trav < cfg.TRAV_THRESHOLD else N_ROWS

        if n_rows_to_use < 2:
            continue

        pts = _path_pixel_pts(indices, h, w, n_rows_to_use).reshape(-1, 1, 2)

        if rank == 0:
            # Best path: dark blue, fully opaque
            cv2.polylines(canvas, [pts], False, (180, 0, 0), best_thickness, cv2.LINE_AA)
        else:
            # Suboptimal paths: light blue, fading toward ALPHA_MIN
            alpha = cfg.ALPHA_BEST - (cfg.ALPHA_BEST - cfg.ALPHA_MIN) * rank / max(n - 1, 1)
            overlay = canvas.copy()
            cv2.polylines(overlay, [pts], False, (235, 206, 135), sub_thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


# ── Main display loop ─────────────────────────────────────────────────────────
idx        = 0
show_gt    = has_labels and not args.no_gt
show_paths = True
auto_mode  = True   # starts in continuous mode; space toggles to step mode
win        = "Off-Road Traversability"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

while True:
    filename = filenames[idx]
    img_path = os.path.join(args.dataset, filename)

    pil_img       = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Inference
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(img_t).squeeze(0).cpu().numpy()   # (N_ROWS, N_POINTS)

    # Ground truth labels (loaded whenever available; displayed only if show_gt)
    gt = None
    if has_labels:
        row = df.iloc[idx]
        gt  = np.zeros((N_ROWS, N_POINTS), dtype=np.float32)
        for r in range(N_ROWS):
            cols  = [f"r{r}_p{p}" for p in range(N_POINTS)]
            gt[r] = row[cols].values.astype(np.float32)
        if not show_gt:
            gt = None

    display = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    dh, dw  = display.shape[:2]

    # 1 — ground-truth coloured dots on scan lines (drawn first, underneath)
    if gt is not None:
        _draw_gt_dots(display, gt, dot_radius=cfg.GT_DOT_RADIUS)

    # 2 — prediction bar chart (score 0.5 = scan line; above=green, below=red)
    _draw_pred_bars(display, pred)

    # 3 — best paths
    if show_paths:
        top_paths = find_best_paths(pred)
        _draw_paths(display, top_paths, pred)

    # HUD
    mode_label = "AUTO" if auto_mode else "STEP"
    cv2.putText(display,
                f"[{idx+1}/{len(filenames)}] {filename}  [{mode_label}]",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1, cv2.LINE_AA)
    hint = "spc: toggle mode  n: next (step)  ←: prev  g: GT  p: paths  q: quit"
    cv2.putText(display, hint,
                (8, dh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.imshow(win, display)

    if auto_mode:
        # Non-blocking wait; advance automatically each frame
        frame_delay = max(1, int(args.delay * 1000)) if args.delay > 0 else 1
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord(' '):
            auto_mode = False          # switch to step mode
        elif key in (ord('q'), 27):
            break
        elif key == ord('g'):
            show_gt = not show_gt and has_labels
        elif key == ord('p'):
            show_paths = not show_paths
        else:
            idx = (idx + 1) % len(filenames)
    else:
        # Block until keypress
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):
            auto_mode = True           # switch back to continuous mode
        elif key == ord('n'):
            idx = (idx + 1) % len(filenames)
        elif key == 81:                # left arrow — go back
            idx = (idx - 1) % len(filenames)
        elif key in (ord('q'), 27):
            break
        elif key == ord('g'):
            show_gt = not show_gt and has_labels
        elif key == ord('p'):
            show_paths = not show_paths

cv2.destroyAllWindows()
