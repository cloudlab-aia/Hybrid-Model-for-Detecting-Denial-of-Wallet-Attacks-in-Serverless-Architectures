"""
dow/cross_validation.py
=======================
Stratified K-Fold cross-validation for ML and DL classifiers.

For each metric the output shows:  mean ± std  (per fold + summary table)

Public API
----------
run_cv_ml(name, X, y, n_splits, use_scale, verbose)  -> cv_results_dict
run_cv_dl(name, X, y, n_splits, epochs, verbose)      -> cv_results_dict
run_cv_all(names, X, y, n_splits, use_scale, epochs)  -> {model: cv_results}
print_cv_summary(cv_all_results)

cv_results_dict keys
--------------------
  folds    : list[dict]   — per-fold metric dicts
  mean     : dict         — mean across folds
  std      : dict         — std  across folds
  elapsed  : float        — total wall-clock seconds
"""

import time
import warnings
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dow.config import RANDOM_STATE
from dow.metrics import (
    _tpr_at_fpr,   # private helper — same module, same package
    evaluate,
    print_model_report,
    print_summary,
)

# Console helpers
_W = 74

def _sep(char: str = "=") -> None:
    print(char * _W)


def _print_cv_header(name: str, n_splits: int, n_samples: int) -> None:
    _sep()
    print(f"  CROSS-VALIDATION : {name.upper()}")
    print(f"  Folds   : {n_splits}-fold Stratified K-Fold")
    print(f"  Samples : {n_samples:,}")
    _sep("-")


def _print_fold_row(fold: int, m: dict) -> None:
    print(
        f"  Fold {fold:>2} │ "
        f"Prec {m['Precision']:.4f}  Rec {m['Recall']:.4f}  "
        f"F1 {m['F1']:.4f}  Acc {m['Accuracy']:.4f}  "
        f"ROC {m['ROC_AUC']:.4f}  PR {m['PR_AUC']:.4f}  "
        f"T@1% {m['TPR_1FPR']:.4f}  T@5% {m['TPR_5FPR']:.4f}"
    )


def _aggregate(fold_metrics: list[dict]) -> tuple[dict, dict]:
    """Return (mean_dict, std_dict) over a list of per-fold metric dicts."""
    keys = list(fold_metrics[0].keys())
    mean = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    std  = {k: float(np.std( [f[k] for f in fold_metrics])) for k in keys}
    return mean, std


def _print_cv_summary_single(name: str, mean: dict, std: dict,
                               elapsed: float) -> None:
    _sep("-")
    print(f"  {'Metric':<22} {'Mean':>10}  {'±Std':>10}")
    _sep("-")
    rows = [
        ("Precision",  "Precision"),
        ("Recall",     "Recall"),
        ("F1-score",   "F1"),
        ("Accuracy",   "Accuracy"),
        ("ROC-AUC",    "ROC_AUC"),
        ("PR-AUC",     "PR_AUC"),
        ("TPR@1%FPR",  "TPR_1FPR"),
        ("TPR@5%FPR",  "TPR_5FPR"),
    ]
    for label, key in rows:
        print(
            f"  {label:<22} {mean[key]:>10.4f}  ±{std[key]:>8.4f}"
        )
    _sep("-")
    print(f"  Total CV time : {elapsed:.1f}s")
    _sep()


# ML cross-validation
def _build_ml_for_cv(name: str):
    """Return a sklearn estimator"""
    if name == "decision_tree":
        base = DecisionTreeClassifier(
            max_depth=15, min_samples_leaf=2, random_state=RANDOM_STATE,
        )
        return CalibratedClassifierCV(base, method="isotonic", cv=3)

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )

    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=RANDOM_STATE,
        )

    if name == "naive_bayes":
        return GaussianNB()

    if name == "kneighbors":
        return KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)

    raise ValueError(f"Unknown ML model '{name}'.")


# Per-model decision thresholds
_THRESHOLDS = {
    "decision_tree":     0.65,
    "random_forest":     0.50,
    "gradient_boosting": 0.50,
    "naive_bayes":       0.50,
    "kneighbors":        0.50,
}


def run_cv_ml(
    name:      str,
    X:         np.ndarray,
    y:         np.ndarray,
    n_splits:  int  = 5,
    use_scale: bool = True,
    verbose:   bool = True,
) -> dict:
    """
    Stratified K-Fold CV for one ML classifier.

    Parameters
    ----------
    name      : model identifier  (decision_tree, random_forest, …)
    X, y      : full feature matrix and label array  (NOT pre-split)
    n_splits  : number of CV folds  (default 5)
    use_scale : apply StandardScaler per fold  (fitted on train, applied to val)
    verbose   : print per-fold rows

    Returns
    -------
    dict with keys: folds, mean, std, elapsed
    """
    _print_cv_header(name, n_splits, len(X))

    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=RANDOM_STATE)
    threshold = _THRESHOLDS.get(name, 0.50)
    fold_metrics: list[dict] = []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if use_scale:
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)

        clf = _build_ml_for_cv(name)
        clf.fit(X_tr, y_tr)

        y_prob = (
            clf.predict_proba(X_va)[:, 1]
            if hasattr(clf, "predict_proba")
            else clf.predict(X_va).astype(float)
        )
        y_pred = (y_prob >= threshold).astype(int)

        m, _ = evaluate(y_va, y_pred, y_prob)
        fold_metrics.append(m)

        if verbose:
            _print_fold_row(fold, m)

    elapsed      = time.time() - t0
    mean, std    = _aggregate(fold_metrics)

    if verbose:
        _print_cv_summary_single(name, mean, std, elapsed)

    return {"folds": fold_metrics, "mean": mean, "std": std, "elapsed": elapsed}


# DL cross-validation
def _build_dl_for_cv(name: str, n_features: int):
    """Import lazily to avoid TF import when only ML CV is needed."""
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(RANDOM_STATE)

    inp = keras.Input(shape=(n_features,))

    if name == "mlp":
        x   = keras.layers.Dense(256, activation="relu")(inp)
        x   = keras.layers.BatchNormalization()(x)
        x   = keras.layers.Dropout(0.3)(x)
        x   = keras.layers.Dense(128, activation="relu")(x)
        x   = keras.layers.BatchNormalization()(x)
        x   = keras.layers.Dropout(0.2)(x)
        x   = keras.layers.Dense(64, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)

    elif name == "lstm":
        x   = keras.layers.Reshape((n_features, 1))(inp)
        x   = keras.layers.LSTM(128, return_sequences=True)(x)
        x   = keras.layers.Dropout(0.3)(x)
        x   = keras.layers.LSTM(64)(x)
        x   = keras.layers.Dropout(0.2)(x)
        x   = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)

    elif name == "bilstm":
        x   = keras.layers.Reshape((n_features, 1))(inp)
        x   = keras.layers.Bidirectional(
                  keras.layers.LSTM(128, return_sequences=True))(x)
        x   = keras.layers.Dropout(0.3)(x)
        x   = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
        x   = keras.layers.Dropout(0.2)(x)
        x   = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)

    elif name == "gru":
        x   = keras.layers.Reshape((n_features, 1))(inp)
        x   = keras.layers.GRU(128, return_sequences=True)(x)
        x   = keras.layers.Dropout(0.3)(x)
        x   = keras.layers.GRU(64)(x)
        x   = keras.layers.Dropout(0.2)(x)
        x   = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)

    else:
        raise ValueError(f"Unknown DL model '{name}'.")

    return keras.Model(inp, out, name=name)


def _focal_loss_cv(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss (same as dl_models.py) — duplicated to avoid circular import."""
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce  = -(y_true * tf.math.log(y_pred)
                 + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t  = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        fl   = bce * tf.pow(1.0 - p_t, gamma)
        a_t  = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(a_t * fl)
    loss.__name__ = "focal_loss"
    return loss


def _best_threshold_cv(y_val: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.05):
        f1 = f1_score(y_val, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def run_cv_dl(
    name:     str,
    X:        np.ndarray,
    y:        np.ndarray,
    n_splits: int  = 5,
    epochs:   int  = 20,
    verbose:  bool = True,
) -> dict:
    """
    Stratified K-Fold CV for one DL architecture.

    Notes
    -----
    · StandardScaler fitted per fold (train only), applied to val.
    · Uses focal loss (γ=2, α=0.25)
    · Decision threshold tuned on the last 10 % of each train fold.
    · EarlyStopping (patience=5) and ReduceLROnPlateau applied per fold.
    · Keras model is rebuilt fresh for each fold to avoid weight leakage.
    """
    from tensorflow import keras
    from dow.config import BATCH_SIZE

    _print_cv_header(name, n_splits, len(X))

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=RANDOM_STATE)
    fold_metrics: list[dict] = []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n  ── Fold {fold}/{n_splits} ──", flush=True)

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_va = sc.transform(X_va).astype(np.float32)
        y_tr_f = y_tr.astype(np.float32)

        # Rebuild model each fold (fresh weights)
        model = _build_dl_for_cv(name, X_tr.shape[1])
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=5e-4),
            loss      = _focal_loss_cv(gamma=2.0, alpha=0.25),
            metrics   = ["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-6, verbose=0,
            ),
        ]

        model.fit(
            X_tr, y_tr_f,
            validation_split = 0.10,
            epochs           = epochs,
            batch_size       = BATCH_SIZE,
            callbacks        = callbacks,
            verbose          = 0,
        )

        # Threshold tuning on last 10 % of train fold
        n_val      = max(1, int(len(X_tr) * 0.10))
        y_prob_val = model.predict(X_tr[-n_val:], verbose=0).ravel()
        threshold  = _best_threshold_cv(y_tr_f[-n_val:], y_prob_val)

        y_prob = model.predict(X_va, verbose=0).ravel()
        y_pred = (y_prob > threshold).astype(int)

        m, _ = evaluate(y_va, y_pred, y_prob)
        fold_metrics.append(m)

        if verbose:
            _print_fold_row(fold, m)

        # Free GPU/CPU memory between folds
        keras.backend.clear_session()

    elapsed   = time.time() - t0
    mean, std = _aggregate(fold_metrics)

    if verbose:
        _print_cv_summary_single(name, mean, std, elapsed)

    return {"folds": fold_metrics, "mean": mean, "std": std, "elapsed": elapsed}


# Multi-model runner
def run_cv_all(
    names:     list[str],
    X:         np.ndarray,
    y:         np.ndarray,
    n_splits:  int  = 5,
    use_scale: bool = True,
    epochs:    int  = 20,
) -> dict:
    """
    Run CV for every model in *names* and return a nested results dict.

    Returns
    -------
    {model_name: cv_results_dict}
    """
    from dow.config import ALL_ML_MODELS

    all_results: dict = {}

    for name in names:
        try:
            if name in ALL_ML_MODELS:
                r = run_cv_ml(name, X, y,
                              n_splits=n_splits, use_scale=use_scale)
            else:
                r = run_cv_dl(name, X, y,
                              n_splits=n_splits, epochs=epochs)
            all_results[name] = r

        except KeyboardInterrupt:
            print("\n[INFO] CV interrupted — showing partial results …")
            break

        except Exception as exc:
            import traceback
            print(f"\n[ERROR] CV for '{name}' failed: {exc}")
            traceback.print_exc()
            print("[INFO] Continuing with next model …\n")

    return all_results


# Summary table
def print_cv_summary(cv_results: dict) -> None:
    """
    Print a consolidated table

    For each model: mean ± std across folds for all 8 metrics.
    """
    if not cv_results:
        return

    _sep()
    print("  CROSS-VALIDATION SUMMARY  (mean ± std across folds)")
    _sep("-")
    print(
        f"  {'Model':<20} "
        f"{'Precision':>12} {'Recall':>12} {'F1':>12} "
        f"{'Acc(Test)':>12} {'ROC-AUC':>12} {'PR-AUC':>12} "
        f"{'TPR@1%FPR':>12} {'TPR@5%FPR':>12}"
    )
    _sep("-")

    metric_keys = [
        ("Precision", "Precision"),
        ("Recall",    "Recall"),
        ("F1",        "F1"),
        ("Accuracy",  "Accuracy"),
        ("ROC_AUC",   "ROC_AUC"),
        ("PR_AUC",    "PR_AUC"),
        ("TPR_1FPR",  "TPR_1FPR"),
        ("TPR_5FPR",  "TPR_5FPR"),
    ]

    for model_name, r in cv_results.items():
        mean = r["mean"]
        std  = r["std"]
        cells = [f"{mean[k]:.4f}±{std[k]:.4f}" for _, k in metric_keys]
        print(f"  {model_name:<20} " + "  ".join(f"{c:>12}" for c in cells))

    _sep()
    print(
        "  Note: values formatted as mean±std  "
        "(e.g. 0.9360±0.0041 means mean=0.9360, std=0.0041)"
    )
    _sep()
