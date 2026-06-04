"""
dow/data.py
===========
Data loading, preprocessing and train/test splitting.

Two entry points
----------------
load_and_split(path, chronological)
    Load the raw CSV, encode categoricals, split 70/30.
    Used when --augment is NOT active.

load_raw(path)
    Load the raw CSV without encoding or splitting.
    Used by the augmentation pipeline (augment.py) which needs the
    original string values (IP addresses, trigger names, vm categories)
    before encoding.

scale(X_tr, X_te)
    Fit StandardScaler on training set, apply to both splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dow.config import NON_FEATURE_COLS, RANDOM_STATE, TARGET_COL, TRAIN_RATIO

# Raw load (used by augment.py)
def load_raw(path: str) -> pd.DataFrame:
    """Return the raw DataFrame without any encoding or splitting."""
    return pd.read_csv(path)


# Full load + split (used without augmentation)
def load_and_split(
    path: str,
    chronological: bool = False,
) -> tuple:
    """
    Load the CSV, encode categoricals and split into train / test sets.

    Parameters
    ----------
    path          : path to dataset.csv
    chronological : True  → strict time-order split (first 70% / last 30%)
                    False → stratified random split (default; better class
                            balance in both sets)

    Returns
    -------
    X_tr, X_te : float32 ndarray
    y_tr, y_te : int ndarray
    feat_cols  : list[str]  — ordered feature column names

    Notes
    -----
    The 'IP' column is encoded (ordinal) and KEPT as a feature.
    Bot traffic originates from a fixed pool of 100 IPs with zero overlap
    to legitimate IPs — a strong discriminating signal for all classifiers.
    """
    print(f"\n[DATA] Loading '{path}' …", flush=True)
    df = pd.read_csv(path)
    n, ncols = df.shape
    n_attack  = int(df[TARGET_COL].sum())

    print(f"[DATA] {n:,} rows  ·  {ncols} columns")
    print(f"[DATA] Attack ratio : {n_attack:,} / {n:,} = {n_attack/n*100:.1f}%")

    # encode all string / object columns (IP, functionTrigger, vmcategory …)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y         = df[TARGET_COL].astype(int).values
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X         = df[feat_cols].values.astype(np.float32)

    print(f"[DATA] Features ({len(feat_cols)}) : {feat_cols}")

    if chronological:
        sp = int(n * TRAIN_RATIO)
        X_tr, X_te = X[:sp], X[sp:]
        y_tr, y_te = y[:sp], y[sp:]
        mode_str = "chronological"
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size    = 1 - TRAIN_RATIO,
            stratify     = y,
            random_state = RANDOM_STATE,
        )
        mode_str = "stratified random"

    bc = np.bincount(y_te)
    print(
        f"[DATA] Split ({mode_str}) — "
        f"train: {len(X_tr):,}  |  "
        f"test: {len(X_te):,}  "
        f"(benign {bc[0]:,} / attack {bc[1]:,})"
    )
    return X_tr, X_te, y_tr, y_te, feat_cols


# Scaler helper (shared by ml_models and dl_models)
def scale(X_tr: np.ndarray, X_te: np.ndarray) -> tuple:
    """
    Fit StandardScaler on X_tr, apply to both splits.

    Returns
    -------
    X_tr_scaled, X_te_scaled : ndarray
    scaler                   : fitted StandardScaler
    """
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), sc
