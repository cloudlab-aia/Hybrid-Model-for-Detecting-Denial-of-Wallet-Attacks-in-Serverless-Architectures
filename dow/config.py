"""
dow/config.py
=============
Centralised constants, model registries, and default augmentation
configuration.  All other modules import from here.
"""

# Dataset
DATASET_PATH  = "dataset.csv"
TARGET_COL    = "bot"
RANDOM_STATE  = 42
TRAIN_RATIO   = 0.70

# Columns excluded from the feature matrix.
# 'IP' is intentionally KEPT: bot traffic uses a fixed pool of 100 IPs
# with zero overlap to legitimate traffic — strong discriminating signal.
NON_FEATURE_COLS = ["Id", "timestamp", TARGET_COL]

# Deep-learning hyper-parameters
DEFAULT_EPOCHS = 20
BATCH_SIZE     = 256

# Model registries
ALL_ML_MODELS = [
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "naive_bayes",
    "kneighbors",
]

ALL_DL_MODELS = ["mlp", "lstm", "bilstm", "gru"]

ALL_MODELS = ALL_ML_MODELS + ALL_DL_MODELS

# Augmentation defaults (mirrors AugmentConfig in dow/augment.py)
AUG_N_SYNTHETIC_FUNCTIONS = 50
AUG_N_BOT_PER_FUNC        = 1_500
AUG_N_LEGIT_PER_FUNC      = 15
AUG_NOISE_SCALE           = 0.05
AUG_BOT_LEGIT_IP_FRACTION = 0.40
AUG_IP_MERGE_FRACTION     = 0.15
