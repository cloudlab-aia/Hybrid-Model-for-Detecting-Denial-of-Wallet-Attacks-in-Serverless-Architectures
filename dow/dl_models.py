"""
dow/dl_models.py
================
Keras / TensorFlow neural network classifiers (Keras 3 / TF 2.16+ compatible).

Architectures
-------------
  mlp    : Multi-Layer Perceptron  (Dense → BatchNorm → Dropout × 2)
  lstm   : Stacked LSTM  (128 → 64 units, 2 layers)
  bilstm : Bidirectional LSTM  (128×2 → 64×2, 2 layers)
  gru    : Stacked GRU  (128 → 64 units, 2 layers)

Implementation notes
--------------------
All models use the Keras Functional API (keras.Input → layer chain →
keras.Model).  Using Sequential with input_shape inside Reshape/RNN
layers raises a dtype-promotion error in Keras 3.x and is avoided.

RNN models treat each transaction as a single-timestep sequence of
(n_features,) values, consistent with the paper's window approach.
The flat feature vector is reshaped to (n_features, 1) before the
recurrent layers, so each feature dimension is one "timestep".

Public API
----------
run_dl(name, X_tr, X_te, y_tr, y_te, epochs) -> metrics_dict
"""

import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras

from dow.config import BATCH_SIZE, RANDOM_STATE
from dow.data import scale
from dow.metrics import (
    evaluate,
    print_classification_report,
    print_confusion,
    print_model_report,
)

tf.random.set_seed(RANDOM_STATE)


# Focal loss
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Binary focal loss  FL(p_t) = −α_t · (1−p_t)^γ · log(p_t).

    Prevents majority-class collapse on the non-augmented original dataset
    by down-weighting easy, high-confidence predictions.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = -(y_true * tf.math.log(y_pred)
                   + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        fl     = bce * tf.pow(1.0 - p_t, gamma)
        a_t    = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(a_t * fl)
    loss.__name__ = "focal_loss"
    return loss


# Callbacks
class _EpochPrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        l = logs or {}
        print(
            f"    Epoch {epoch+1:>2} — "
            f"loss: {l.get('loss', 0):.4f}  "
            f"accuracy: {l.get('accuracy', 0):.4f}  "
            f"val_loss: {l.get('val_loss', 0):.4f}  "
            f"val_accuracy: {l.get('val_accuracy', 0):.4f}",
            flush=True,
        )


def _get_callbacks() -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=0,
        ),
    ]


# Model builders — Functional API (Keras 3 compatible)
def _build_mlp(n: int) -> keras.Model:
    """Dense(256)→BN→Drop(0.3) → Dense(128)→BN→Drop(0.2) → Dense(64) → Dense(1)"""
    inp = keras.Input(shape=(n,))
    x   = keras.layers.Dense(256, activation="relu")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="mlp")


def _build_lstm(n: int) -> keras.Model:
    """Reshape(n,1) → LSTM(128)→Drop → LSTM(64)→Drop → Dense(32) → Dense(1)"""
    inp = keras.Input(shape=(n,))
    x   = keras.layers.Reshape((n, 1))(inp)
    x   = keras.layers.LSTM(128, return_sequences=True)(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.LSTM(64)(x)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="lstm")


def _build_bilstm(n: int) -> keras.Model:
    """Reshape → BiLSTM(128×2)→Drop → BiLSTM(64×2)→Drop → Dense(32) → Dense(1)"""
    inp = keras.Input(shape=(n,))
    x   = keras.layers.Reshape((n, 1))(inp)
    x   = keras.layers.Bidirectional(
              keras.layers.LSTM(128, return_sequences=True))(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="bilstm")


def _build_gru(n: int) -> keras.Model:
    """Reshape(n,1) → GRU(128)→Drop → GRU(64)→Drop → Dense(32) → Dense(1)"""
    inp = keras.Input(shape=(n,))
    x   = keras.layers.Reshape((n, 1))(inp)
    x   = keras.layers.GRU(128, return_sequences=True)(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.GRU(64)(x)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="gru")


_BUILDERS = {
    "mlp":    _build_mlp,
    "lstm":   _build_lstm,
    "bilstm": _build_bilstm,
    "gru":    _build_gru,
}


# Helpers
def _is_augmented(n_rows: int) -> bool:
    """Augmented datasets (from --augment) are > 200 k training rows."""
    return n_rows > 200_000


def _best_threshold(y_val: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the decision threshold maximising F1 on the held-out validation slice.
    Corrects the precision/recall imbalance caused by the class distribution.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.05):
        f1 = f1_score(y_val, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


# Public API
def run_dl(
    name:   str,
    X_tr:   np.ndarray,
    X_te:   np.ndarray,
    y_tr:   np.ndarray,
    y_te:   np.ndarray,
    epochs: int = 20,
) -> dict:
    """
    Build, train and evaluate one DL architecture.

    Loss is selected automatically:
      • original dataset (< 200 k training rows) → focal loss (γ=2, α=0.25)
      • augmented dataset (≥ 200 k training rows) → binary cross-entropy

    After training, the decision threshold is tuned on the held-out
    validation slice (10% of training data) to maximise F1-score.

    Parameters
    ----------
    name          : 'mlp' | 'lstm' | 'bilstm' | 'gru'
    X_tr / X_te  : raw feature matrices  (StandardScaler applied internally)
    y_tr / y_te  : label arrays  (0 = benign, 1 = attack)
    epochs        : maximum epochs  (early stopping may halt sooner)

    Returns
    -------
    dict — Precision, Recall, F1, Accuracy, ROC_AUC, PR_AUC,
           TPR_1FPR, TPR_5FPR
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown DL model '{name}'. Valid: {list(_BUILDERS)}"
        )

    print(f"\n[DL] Training {name.upper()}  ({epochs} epochs) …", flush=True)

    # Scale and cast to float32
    X_tr_s, X_te_s, _ = scale(X_tr, X_te)
    X_tr_s = np.asarray(X_tr_s, dtype=np.float32)
    X_te_s = np.asarray(X_te_s, dtype=np.float32)
    y_tr_f = np.asarray(y_tr,   dtype=np.float32)
    n_feat  = X_tr_s.shape[1]

    # Build model
    model     = _BUILDERS[name](n_feat)
    augmented = _is_augmented(len(X_tr_s))
    loss      = ("binary_crossentropy" if augmented
                 else focal_loss(gamma=2.0, alpha=0.25))
    batch     = BATCH_SIZE * 2 if augmented else BATCH_SIZE

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=5e-4),
        loss      = loss,
        metrics   = ["accuracy"],
    )
    model.summary(print_fn=lambda s: print(f"    {s}"))

    loss_name = ("binary_crossentropy" if augmented
                 else "focal_loss(γ=2, α=0.25)")
    print(
        f"    Loss: {loss_name}  |  Batch: {batch}  |  "
        f"Dataset: {'augmented' if augmented else 'original'}",
        flush=True,
    )

    # Train
    val_split = 0.10
    t0 = time.time()
    model.fit(
        X_tr_s, y_tr_f,
        validation_split = val_split,
        epochs           = epochs,
        batch_size       = batch,
        callbacks        = _get_callbacks() + [_EpochPrinter()],
        verbose          = 0,
    )
    elapsed = time.time() - t0

    # Adaptive threshold tuning on held-out validation slice
    n_val      = int(len(X_tr_s) * val_split)
    y_prob_val = model.predict(X_tr_s[-n_val:], verbose=0).ravel()
    threshold  = _best_threshold(y_tr_f[-n_val:], y_prob_val)
    print(f"    Optimal threshold (val F1): {threshold:.2f}", flush=True)

    # Final test evaluation
    y_prob = model.predict(X_te_s, verbose=0).ravel()
    y_pred = (y_prob > threshold).astype(int)

    metrics, cm = evaluate(y_te, y_pred, y_prob)
    print_model_report(name, metrics, elapsed)
    print_confusion(cm)
    print_classification_report(y_te, y_pred)

    return metrics
