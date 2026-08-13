# ─── off_road/config.py ───────────────────────────────────────────────────────

DATASET_NAME = "offroad_grayling_composite"

# ── Model input size ──────────────────────────────────────────────────────────
IMAGE_WIDTH  = 320
IMAGE_HEIGHT = 240

# ── Scan line geometry ────────────────────────────────────────────────────────
# Row 0 = nearest to vehicle (largest y / bottom of image).
# Row 3 = farthest (smallest y / higher in image).
N_ROWS   = 4
N_POINTS = 50   # uniformly spaced traversability points per scan line

# Populated by init() from offroad_dataset.info — do not set manually.
ROW_FRACTIONS = None   # [scan_r0, scan_r1, scan_r2, scan_r3]
ROW_INDICES   = None   # corresponding pixel rows in IMAGE_HEIGHT space

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE          = 16
EPOCHS              = 200
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
VAL_SPLIT           = 0.15
VIS_INTERVAL        = 0      # write mid-epoch snapshot every N batches (0 = off)
EARLY_STOP_PATIENCE = 15     # 0 = disabled

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
DATA_DIR           = os.path.join(os.path.dirname(__file__), "data")
DATASET_DIR        = os.path.join(DATA_DIR, DATASET_NAME)
TRAIN_DIR          = os.path.join(DATASET_DIR, "train")
TEST_DIR           = os.path.join(DATASET_DIR, "test")
TRAIN_LABELS_CSV   = os.path.join(TRAIN_DIR, "offroad_labels.csv")
TEST_LABELS_CSV    = os.path.join(TEST_DIR,  "offroad_labels.csv")
TRAIN_DATASET_INFO = os.path.join(TRAIN_DIR, "offroad_dataset.info")
TEST_DATASET_INFO  = os.path.join(TEST_DIR,  "offroad_dataset.info")

MODEL_NAME = "offroad_traversability_best"
CHECKPOINT = os.path.join(os.path.dirname(__file__), MODEL_NAME + ".pth")

# ── ImageNet normalisation ────────────────────────────────────────────────────
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# ── dataset.info parser ───────────────────────────────────────────────────────
def _parse_dataset_info(path):
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = float(val.strip())
    return result


# ── Runtime init ──────────────────────────────────────────────────────────────
def init():
    """
    Call once at startup.  Reads offroad_dataset.info and populates
    ROW_FRACTIONS and ROW_INDICES.

    If the dataset has no train/ subdirectory, all files are read directly
    from DATASET_DIR and all data is used for training.
    """
    global ROW_FRACTIONS, ROW_INDICES
    global TRAIN_DIR, TEST_DIR
    global TRAIN_LABELS_CSV, TEST_LABELS_CSV
    global TRAIN_DATASET_INFO, TEST_DATASET_INFO

    # Flat dataset: no train/ subdirectory — use dataset root for everything
    if not os.path.isdir(TRAIN_DIR):
        TRAIN_DIR          = DATASET_DIR
        TEST_DIR           = DATASET_DIR
        TRAIN_LABELS_CSV   = os.path.join(DATASET_DIR, "offroad_labels.csv")
        TEST_LABELS_CSV    = os.path.join(DATASET_DIR, "offroad_labels.csv")
        TRAIN_DATASET_INFO = os.path.join(DATASET_DIR, "offroad_dataset.info")
        TEST_DATASET_INFO  = os.path.join(DATASET_DIR, "offroad_dataset.info")

    info_path = TRAIN_DATASET_INFO
    if not os.path.exists(info_path):
        info_path = TEST_DATASET_INFO
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"offroad_dataset.info not found in {TRAIN_DIR} or {TEST_DIR}.\n"
            "Expected keys: scan_r0, scan_r1, scan_r2, scan_r3, points_per_line")

    info = _parse_dataset_info(info_path)

    required = ["scan_r0", "scan_r1", "scan_r2", "scan_r3", "points_per_line"]
    missing = [k for k in required if k not in info]
    if missing:
        raise KeyError(f"offroad_dataset.info is missing keys: {missing}")

    n_pts = int(info["points_per_line"])
    if n_pts != N_POINTS:
        raise ValueError(
            f"offroad_dataset.info has points_per_line={n_pts}, but config.N_POINTS={N_POINTS}."
            " Update N_POINTS in config.py to match.")

    ROW_FRACTIONS = [info[f"scan_r{i}"] for i in range(N_ROWS)]
    ROW_INDICES   = [int(f * IMAGE_HEIGHT) for f in ROW_FRACTIONS]

    print("Config ready:")
    print(f"  Dataset      : {DATASET_NAME}")
    print(f"  Input size   : {IMAGE_WIDTH} × {IMAGE_HEIGHT}")
    print(f"  N_ROWS={N_ROWS}  N_POINTS={N_POINTS}")
    print(f"  Row fractions: {ROW_FRACTIONS}")
    print(f"  Row pixel rows: {ROW_INDICES}")
