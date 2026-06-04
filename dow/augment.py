"""
dow/augment.py
==============
GAN-style dataset augmentation pipeline that replicates the approach
described in Section 5.1 of the research.

Public API
----------
build_augmented_dataset(df_raw, cfg, verbose) -> (X, y, feature_names)
AugmentConfig                                 -> dataclass with all parameters
DEFAULT_CONFIG                                -> validated best configuration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dow.config import (
    AUG_BOT_LEGIT_IP_FRACTION,
    AUG_IP_MERGE_FRACTION,
    AUG_N_BOT_PER_FUNC,
    AUG_N_LEGIT_PER_FUNC,
    AUG_N_SYNTHETIC_FUNCTIONS,
    AUG_NOISE_SCALE,
    RANDOM_STATE,
    TRAIN_RATIO,
)

# Column definitions

# Numerical columns used to build function prototypes and apply noise
_NUM_COLS = [
    "SubmitTime", "RTT", "InvocationDelay", "ResponseDelay",
    "FunctionDuration", "ActiveFunctionsAtRequest",
    "ActiveFunctionsAtResponse", "maxcpu", "avgcpu", "p95maxcpu",
    "vmcorecountbucket", "vmmemorybucket",
]

# Output feature matrix column order (must match data.py / NON_FEATURE_COLS)
FEATURE_COLS = [
    "IP", "FunctionId", "functionTrigger", "SubmitTime", "RTT",
    "InvocationDelay", "ResponseDelay", "FunctionDuration",
    "ActiveFunctionsAtRequest", "ActiveFunctionsAtResponse",
    "maxcpu", "avgcpu", "p95maxcpu",
    "vmcategory", "vmcorecountbucket", "vmmemorybucket",
]


# Configuration
@dataclass
class AugmentConfig:
    """
    Tunable parameters for the GAN-style augmentation pipeline.

    n_synthetic_functions : int
        Number of new virtual serverless functions to create by interpolating
        between pairs of existing FunctionId parameter vectors.  Each
        generates n_bot_per_func attack rows + n_legit_per_func legit rows.

    n_bot_per_func : int
        Attack transactions per synthetic function (~1,500 keeps total size
        manageable while matching original attack density of ~2,900/function).

    n_legit_per_func : int
        Legitimate transactions per synthetic function (scaled to original
        ~27 legit rows per function).

    noise_scale : float
        Gaussian noise as a fraction of each feature's std, applied to all
        numerical columns of synthetic rows.  Breaks exact vector duplication.

    bot_legit_ip_fraction : float
        Fraction of SYNTHETIC attack rows assigned a legitimate IP.
        These "stealth attacks" are the primary source of new ROC operating
        points at low FPR.  Range [0, 1].  0.40 = 40% of new attacks look
        like legit traffic.

    ip_merge_fraction : float
        Fraction of ORIGINAL bot rows whose IP is replaced by a random
        legitimate IP.  Removes the perfect IP-pool separator in the
        existing data.  Range [0, 0.30].

    random_state : int
        Master random seed for reproducibility.
    """
    n_synthetic_functions: int   = AUG_N_SYNTHETIC_FUNCTIONS
    n_bot_per_func:        int   = AUG_N_BOT_PER_FUNC
    n_legit_per_func:      int   = AUG_N_LEGIT_PER_FUNC
    noise_scale:           float = AUG_NOISE_SCALE
    bot_legit_ip_fraction: float = AUG_BOT_LEGIT_IP_FRACTION
    ip_merge_fraction:     float = AUG_IP_MERGE_FRACTION
    random_state:          int   = RANDOM_STATE


DEFAULT_CONFIG = AugmentConfig()


# Internal helpers
def _prototypes(df: pd.DataFrame) -> np.ndarray:
    """Return (n_functions, len(_NUM_COLS)) array of per-FunctionId vectors."""
    return df.groupby("FunctionId")[_NUM_COLS].first().values


def _make_row(fid, trigger, vmcat, v, ip, is_bot: bool) -> dict:
    """Build one record dict from a numerical value vector."""
    return {
        "FunctionId":                fid,
        "functionTrigger":           trigger,
        "vmcategory":                vmcat,
        "SubmitTime":                int(v[0]),
        "RTT":                       int(v[1]),
        "InvocationDelay":           float(v[2]),
        "ResponseDelay":             float(v[3]),
        "FunctionDuration":          float(v[4]),
        "ActiveFunctionsAtRequest":  int(v[5]),
        "ActiveFunctionsAtResponse": int(v[6]),
        "maxcpu":                    float(v[7]),
        "avgcpu":                    float(v[8]),
        "p95maxcpu":                 float(v[9]),
        "vmcorecountbucket":         int(v[10]),
        "vmmemorybucket":            float(v[11]),
        "IP":                        ip,
        "bot":                       is_bot,
    }


def _synthesise(
    proto_vals:     np.ndarray,
    n_funcs:        int,
    n_bot:          int,
    n_legit:        int,
    noise:          float,
    bot_legit_frac: float,
    legit_ips:      np.ndarray,
    bot_ips:        np.ndarray,
    triggers:       np.ndarray,
    vmcats:         np.ndarray,
    rng:            np.random.Generator,
) -> pd.DataFrame:
    """
    Create n_funcs synthetic serverless functions.

    For each function:
    - Pick two prototype indices at random, mix with alpha ~ Uniform(0.1, 0.9).
    - Sample n_bot attack rows: each perturbed by Normal(0, noise * std).
      bot_legit_frac of them receive a legitimate IP (stealth attacks).
    - Sample n_legit legitimate rows with legit IPs.
    """
    rows = []
    for i in range(n_funcs):
        a, b  = rng.integers(0, len(proto_vals), 2)
        alpha = rng.uniform(0.1, 0.9)
        base  = (1.0 - alpha) * proto_vals[a] + alpha * proto_vals[b]
        fid   = 100 + i
        tr    = rng.choice(triggers)
        vm    = rng.choice(vmcats)

        for _ in range(n_bot):
            v  = base * (1.0 + rng.normal(0.0, noise, len(_NUM_COLS)))
            ip = rng.choice(legit_ips if rng.random() < bot_legit_frac else bot_ips)
            rows.append(_make_row(fid, tr, vm, v, ip, True))

        for _ in range(n_legit):
            v  = base * (1.0 + rng.normal(0.0, noise, len(_NUM_COLS)))
            rows.append(_make_row(fid, tr, vm, v, rng.choice(legit_ips), False))

    return pd.DataFrame(rows)


def _merge_ips(df: pd.DataFrame, frac: float,
               legit_ips: np.ndarray,
               rng: np.random.Generator) -> pd.DataFrame:
    """Replace `frac` of original (FunctionId < 100) bot rows' IPs."""
    df    = df.copy()
    mask  = (df["bot"] == True) & (df["FunctionId"] < 100)
    idx   = df.index[mask].tolist()
    n     = int(len(idx) * frac)
    if n > 0:
        df.loc[rng.choice(idx, n, replace=False), "IP"] = rng.choice(legit_ips, n)
    return df


def _clip(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Clip numerical features to the original dataset's [min, max] range."""
    df = df.copy()
    for col in _NUM_COLS:
        if col in df.columns and col in ref.columns:
            df[col] = df[col].clip(ref[col].min(), ref[col].max())
    return df


def _to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode categoricals and return (X float32, y int, feature_names)."""
    df = df.copy()
    for col in ("functionTrigger", "vmcategory", "IP"):
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    feat = [c for c in FEATURE_COLS if c in df.columns]
    X    = df[feat].values.astype(np.float32)
    y    = df["bot"].astype(int).values
    return X, y, feat


# Public API
def build_augmented_dataset(
    df_raw:  pd.DataFrame,
    cfg:     AugmentConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Apply the full GAN-style augmentation pipeline.

    Steps
    -----
    1. Extract per-FunctionId numerical parameter prototypes.
    2. Synthesise cfg.n_synthetic_functions new virtual functions by
       pairwise prototype interpolation.  Each yields n_bot_per_func
       attack rows + n_legit_per_func legit rows.
    3. Assign IPs: synthetic attack rows receive a legitimate IP with
       probability cfg.bot_legit_ip_fraction (stealth attacks).
    4. Merge original bot-IP pool: replace cfg.ip_merge_fraction of
       original attack rows' IPs with legitimate IPs.
    5. Clip all numerical features to the original [min, max] range.
    6. Label-encode categorical columns and return (X, y, feature_names).

    Parameters
    ----------
    df_raw  : raw DataFrame from load_raw() — before any encoding
    cfg     : AugmentConfig (default = DEFAULT_CONFIG)
    verbose : print a per-step summary

    Returns
    -------
    X    : float32 ndarray  (n_samples, n_features)
    y    : int ndarray      (n_samples,)
    feat : list[str]        feature column names
    """
    rng = np.random.default_rng(cfg.random_state)

    if verbose:
        n_orig = len(df_raw)
        ar     = df_raw["bot"].mean() * 100
        print(f"\n[AUG] Starting GAN-style augmentation pipeline …")
        print(f"[AUG] Original  : {n_orig:>8,} rows  "
              f"({ar:.1f}% attack, {df_raw['FunctionId'].nunique()} functions)")

    # Prototype vectors and IP pools
    proto   = _prototypes(df_raw)
    l_ips   = df_raw.loc[df_raw["bot"] == False, "IP"].unique()
    b_ips   = df_raw.loc[df_raw["bot"] == True,  "IP"].unique()
    trigs   = df_raw["functionTrigger"].unique()
    vmcats  = df_raw["vmcategory"].unique()

    # Step 2–3: synthesise new functions
    df_synth = _synthesise(
        proto, cfg.n_synthetic_functions,
        cfg.n_bot_per_func, cfg.n_legit_per_func,
        cfg.noise_scale, cfg.bot_legit_ip_fraction,
        l_ips, b_ips, trigs, vmcats, rng,
    )
    shared = list(set(df_raw.columns) & set(df_synth.columns))
    df_aug = pd.concat([df_raw[shared], df_synth[shared]], ignore_index=True)

    if verbose:
        n_new = len(df_synth)
        print(f"[AUG] Synthetic : {n_new:>8,} rows  "
              f"({cfg.n_synthetic_functions} functions × "
              f"{cfg.n_bot_per_func} attack + {cfg.n_legit_per_func} legit, "
              f"{cfg.bot_legit_ip_fraction*100:.0f}% stealth IP)")

    # Step 4: IP merge on original bot rows
    df_aug = _merge_ips(df_aug, cfg.ip_merge_fraction, l_ips, rng)
    if verbose:
        shared_ips = len(
            set(df_aug.loc[df_aug["bot"] == True,  "IP"].unique()) &
            set(df_aug.loc[df_aug["bot"] == False, "IP"].unique())
        )
        print(f"[AUG] IP merge  : {cfg.ip_merge_fraction*100:.0f}% of original bot rows → "
              f"shared IP pool = {shared_ips:,}")

    # Step 5: clip to valid ranges
    df_aug = _clip(df_aug, df_raw)

    # Step 6: encode and build matrix
    X, y, feat = _to_matrix(df_aug)
    if verbose:
        print(f"[AUG] Final     : {len(X):>8,} rows  "
              f"({y.mean()*100:.1f}% attack, "
              f"{df_aug['FunctionId'].nunique()} functions, "
              f"{len(feat)} features)")

    return X, y, feat


def augmented_split(
    df_raw:        pd.DataFrame,
    cfg:           AugmentConfig = DEFAULT_CONFIG,
    chronological: bool          = False,
    verbose:       bool          = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build the augmented dataset and split into train / test sets.

    Returns
    -------
    X_tr, X_te : float32 ndarray
    y_tr, y_te : int ndarray
    feat_cols  : list[str]
    """
    X, y, feat = build_augmented_dataset(df_raw, cfg, verbose=verbose)

    if chronological:
        sp = int(len(X) * TRAIN_RATIO)
        X_tr, X_te = X[:sp], X[sp:]
        y_tr, y_te = y[:sp], y[sp:]
        mode = "chronological"
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=1-TRAIN_RATIO, stratify=y, random_state=RANDOM_STATE
        )
        mode = "stratified random"

    bc = np.bincount(y_te)
    print(f"[AUG] Split ({mode}) — "
          f"train: {len(X_tr):,}  |  "
          f"test: {len(X_te):,}  "
          f"(benign {bc[0]:,} / attack {bc[1]:,})")

    return X_tr, X_te, y_tr, y_te, feat
