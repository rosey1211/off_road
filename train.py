# ─── off_road/train.py ────────────────────────────────────────────────────────
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import config
config.init()

from dataset import OffRoadDataset
from model   import TraversabilityCNN, traversability_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(current, total, bar_len=28, **stats):
    filled   = int(bar_len * current / total)
    bar      = "█" * filled + "░" * (bar_len - filled)
    stat_str = "  ".join(f"{k}: {v}" for k, v in stats.items())
    sys.stdout.write(f"\r  [{bar}] {current}/{total}  {stat_str}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


# ── Data ──────────────────────────────────────────────────────────────────────
full_ds    = OffRoadDataset(config.TRAIN_LABELS_CSV, config.TRAIN_DIR, augment=False)
val_size   = max(1, int(len(full_ds) * config.VAL_SPLIT))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(
    full_ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(42))

# Augmentation only on the training split
train_aug_ds = torch.utils.data.Subset(
    OffRoadDataset(config.TRAIN_LABELS_CSV, config.TRAIN_DIR, augment=True),
    train_ds.indices)

_PIN = torch.cuda.is_available()
train_loader = DataLoader(train_aug_ds, batch_size=config.BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=_PIN)
val_loader   = DataLoader(val_ds,       batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=_PIN)


# ── Model, optimiser, scheduler ───────────────────────────────────────────────
model = TraversabilityCNN(
    image_width   = config.IMAGE_WIDTH,
    image_height  = config.IMAGE_HEIGHT,
    n_rows        = config.N_ROWS,
    n_points      = config.N_POINTS,
    row_fractions = config.ROW_FRACTIONS,
    norm_mean     = config.NORM_MEAN,
    norm_std      = config.NORM_STD,
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=config.LR,
                       weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

n_params = sum(p.numel() for p in model.parameters())
flat = (config.TRAIN_DIR == config.DATASET_DIR)
print(f"\n{'='*60}")
print(f"  Device     : {DEVICE}")
print(f"  Parameters : {n_params:,}")
print(f"  Dataset    : {'flat (all data used for train+val split)' if flat else 'train/test split'}")
print(f"  Train / Val: {train_size} / {val_size} samples")
print(f"  Epochs     : {config.EPOCHS}  |  Batch: {config.BATCH_SIZE}")
print(f"{'='*60}\n")


# ── Train / eval loops ────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_samples  = 0
    n_batches  = len(loader)
    t0         = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (images, labels) in enumerate(loader, 1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images)
            loss = traversability_loss(pred, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs          = images.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            elapsed = time.time() - t0
            sps     = n_samples / elapsed if elapsed > 0 else 0.0
            eta_s   = (n_batches - i) * (elapsed / i)
            progress_bar(i, n_batches,
                         loss=f"{total_loss/n_samples:.5f}",
                         img_s=f"{sps:.0f}",
                         eta=f"{eta_s:.0f}s")

    return total_loss / n_samples


# ── Visualise predictions: colored dots overlaid on images ───────────────────
import matplotlib.cm as _cm
_cmap = _cm.get_cmap("RdYlGn")   # red=0 (impassable) → green=1 (traversable)


def _save_vis(epoch, save_path):
    model.eval()
    torch.cuda.empty_cache()
    n_show  = min(10, len(val_ds))
    n_cols  = 5
    n_rows  = (n_show + n_cols - 1) // n_cols

    indices = torch.randperm(len(val_ds))[:n_show].tolist()
    samples = [val_ds[i] for i in indices]
    imgs    = torch.stack([s[0] for s in samples]).to(DEVICE)

    with torch.no_grad():
        preds = model(imgs).cpu().numpy()   # (n_show, N_ROWS, N_POINTS)

    mean      = torch.tensor(config.NORM_MEAN).view(3, 1, 1)
    std       = torch.tensor(config.NORM_STD ).view(3, 1, 1)
    imgs_disp = (imgs.cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    # imgs_disp: (n_show, H, W, 3)

    ih, iw = imgs_disp.shape[1:3]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.0))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.axis("off")
        if i >= n_show:
            continue

        ax.imshow(imgs_disp[i])

        for r in range(config.N_ROWS):
            y_img = config.ROW_FRACTIONS[r] * ih
            xs    = [(p + 0.5) / config.N_POINTS * iw for p in range(config.N_POINTS)]
            colors = [_cmap(float(v)) for v in preds[i, r]]
            ax.scatter(xs, [y_img] * config.N_POINTS,
                       c=colors, s=8, linewidths=0,
                       edgecolors="none", zorder=5)

    fig.suptitle(f"Epoch {epoch} — traversability predictions", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    torch.cuda.empty_cache()
    model.train()


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
best_epoch       = None
no_improve_count = 0
total_start      = time.time()

log_dir = os.path.join(os.path.dirname(config.CHECKPOINT), "graphic_images")
os.makedirs(log_dir, exist_ok=True)
history = {"tr_loss": [], "va_loss": []}

for epoch in range(1, config.EPOCHS + 1):
    t_epoch     = time.time()
    current_lr  = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:03d}/{config.EPOCHS}  (lr={current_lr:.6f})")
    print("  Training:")
    tr_loss = run_epoch(train_loader, train=True)

    print("  Validating:")
    va_loss = run_epoch(val_loader, train=False)
    scheduler.step()

    epoch_time = time.time() - t_epoch
    is_best    = va_loss < best_val_loss

    if is_best:
        best_val_loss    = va_loss
        best_epoch       = epoch
        no_improve_count = 0
        torch.save({
            "model_state":   model.state_dict(),
            "epoch":         epoch,
            "image_width":   model.image_width,
            "image_height":  model.image_height,
            "n_rows":        model.n_rows,
            "n_points":      model.n_points,
            "row_fractions": model.row_fractions,
            "norm_mean":     model.norm_mean,
            "norm_std":      model.norm_std,
        }, config.CHECKPOINT)
    else:
        no_improve_count += 1

    # Trend indicator
    if len(history["va_loss"]) == 0:
        trend = ""
    elif va_loss < history["va_loss"][-1] - 1e-6:
        trend = "  ↓ improving"
    elif va_loss > history["va_loss"][-1] + 1e-6:
        trend = "  ↑ rising"
    else:
        trend = "  → plateau"

    marker          = "  ★ best" if is_best else ""
    early_stop_note = ""
    if config.EARLY_STOP_PATIENCE > 0 and not is_best:
        early_stop_note = f"  (no improvement {no_improve_count}/{config.EARLY_STOP_PATIENCE})"

    print(f"  Train  →  MSE: {tr_loss:.5f}")
    print(f"  Val    →  MSE: {va_loss:.5f}{marker}{trend}{early_stop_note}")
    print(f"  Time   →  {epoch_time:.1f}s  |  elapsed: {time.time()-total_start:.0f}s")

    history["tr_loss"].append(tr_loss)
    history["va_loss"].append(va_loss)

    # Loss curve
    metrics_path = os.path.join(log_dir, "metrics.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["tr_loss"], label="train")
    ax.plot(history["va_loss"], label="val")
    if best_epoch is not None:
        ax.axvline(best_epoch - 1, color="green", linestyle="--", alpha=0.6,
                   label=f"best (ep {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.set_title("Traversability MSE")
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=90)
    plt.close(fig)
    print(f"  Metrics → {metrics_path}")

    # Visual snapshot
    vis_path = os.path.join(log_dir, f"vis_epoch_{epoch:03d}.png")
    _save_vis(epoch, vis_path)
    print(f"  Vis     → {vis_path}")
    print()

    # Early stopping
    if config.EARLY_STOP_PATIENCE > 0 and no_improve_count >= config.EARLY_STOP_PATIENCE:
        print(f"  Early stopping: val MSE has not improved for "
              f"{config.EARLY_STOP_PATIENCE} consecutive epochs.")
        print(f"  Best checkpoint is epoch {best_epoch}  "
              f"(val MSE {best_val_loss:.5f})  →  {config.CHECKPOINT}")
        print()
        break

total_time = time.time() - total_start
print(f"{'='*60}")
print(f"  Training complete in {total_time/60:.1f} min")
print(f"  Best val MSE : {best_val_loss:.5f}  (epoch {best_epoch})")
print(f"  Checkpoint   : {config.CHECKPOINT}")
print(f"{'='*60}")
