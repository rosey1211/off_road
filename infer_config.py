# ─── off_road/infer_config.py ────────────────────────────────────────────────
# All parameters for live_infer.py.  Edit here — no other files need changing.

import os
import config as _cfg

# ── Model and dataset ─────────────────────────────────────────────────────────
CHECKPOINT   = _cfg.CHECKPOINT
#DATASET_NAME = "offroad_grayling_composite"
#DATASET_NAME = "thinly_vegetated_terrain_w_small_trees2320x240"
#DATASET_NAME = "thinly_vegetated_terrain_w_small_trees320x240"
#DATASET_NAME = "grass_trail_w_tree_overhangs_for_offroad_lr"
#DATASET_NAME = "vegetated_field_w_trees2_lr"
#DATASET_NAME = "vegetation_w_fence_and_trees"
#DATASET_NAME = "vegetated_field_w_trees320x240"
#DATASET_NAME = "vegetated_field_w_trees2320x240"
#DATASET_NAME = "trail_w_vehicle_in_front"
#DATASET_NAME = "grass_trail_w_tree_overhangs2"
#DATASET_NAME = "grass_trail_w_tree_overhangs"
#DATASET_NAME = "flat_grass_w_mounds_and_sandbags"
#DATASET_NAME = "flat_grass_w_mounds_and_sandbags320x240"
#DATASET_NAME = "grass_trail_w_tree_overhangs320x240"
#DATASET_NAME = "grass_trail_w_tree_overhangs2320x240"
DATASET_NAME = "offroad_grayling_concatenated"
DATASET_DIR  = os.path.join(os.path.dirname(__file__), "data", DATASET_NAME)

# ── Display ───────────────────────────────────────────────────────────────────
# Seconds between frames.  0 = wait for a keypress between each image.
DELAY = 0.0

# Skip augmented images whose filename stem ends with any of these suffixes.
# e.g. img_001_c.png, img_001_sl.jpg are excluded by default.
# Set to [] to use all images.
EXCLUDE_SUFFIXES = ["_f", "_c", "_sn", "_sp", "_sl", "_sw"]

# ── Ground-truth dot size ─────────────────────────────────────────────────────
# Radius in pixels of the ground-truth traversability dots drawn on scan lines.
GT_DOT_RADIUS = 1

# ── Prediction bar chart ──────────────────────────────────────────────────────
# Maximum height of a prediction bar (above or below the scan line) as a
# fraction of the display image height.
BAR_MAX_HEIGHT_FRAC = 0.025

# Opacity of the prediction bars (0.0 = invisible, 1.0 = fully solid).
BAR_ALPHA = 0.50

# ── Path finding ──────────────────────────────────────────────────────────────
# Number of paths to display.
N_PATHS = 100

# Stride through the ranked path list when selecting which paths to display.
# 1 = show ranks 1, 2, 3, … N_PATHS  (consecutive best paths).
# 10 = show ranks 1, 11, 21, 31, …   (every 10th path).
PATH_RANK_SKIP = 20

# Traversability score below which a point is avoided when better options exist.
TRAV_THRESHOLD = 0.3

# Soft preference for paths closer to the image centre.
# Intentionally small — traversability always dominates.  0.0 = disabled.
CENTER_BIAS_WEIGHT = 0.05

# Penalty applied per unit of normalised lateral deviation between consecutive
# scan lines.  Higher values prefer straighter paths, but traversability always
# dominates when the weight is small.
STRAIGHT_BIAS_WEIGHT = 0.10

# Penalty for direction reversals (left→right→left or right→left→right).
# Applied per reversal, normalised so 2 reversals (max for 4 rows) = 1.0.
# Traversability always dominates when this weight is small.
DIRECTION_CHANGE_WEIGHT = 0.10

# Negative bias for paths whose starting point (r0) is off-centre.
# Penalty = weight × normalised_distance_from_centre (0 at centre, 1 at edge).
# The further from centre, the larger the penalty.  Traversability always dominates.
START_CENTER_BIAS_WEIGHT = 0.10

# Maximum lateral movement between consecutive scan lines as a fraction of
# N_POINTS.  0.15 → path may shift at most 15 % of the scan-line width per step.
MAX_LATERAL_STEP_FRAC = 0.15

# Fraction of the lowest scan line (r0) used as path starting points.
# 0.20 → middle 20 % (points 20–29 for N_POINTS=50).
START_MIDDLE_FRAC = 0.050

# ── Path drawing ──────────────────────────────────────────────────────────────
# Opacity of the best (most traversable) path.
ALPHA_BEST = 0.70

# Opacity of the worst displayed path.  Keep close to ALPHA_BEST for a subtle
# gradient across the N paths.
ALPHA_MIN  = 0.40

# Line thickness in pixels at the reference resolution below.
# Scales up proportionally for higher-resolution displays.
PATH_LINE_THICKNESS = 1

# Image width (px) at which PATH_LINE_THICKNESS applies.
# Wider images get proportionally thicker lines.
PATH_THICKNESS_REF_WIDTH = 640
