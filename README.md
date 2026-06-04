# DoW Attack Detector — Stage-2 Classifier

A modular command-line tool for training and evaluating Machine Learning and
Deep Learning classifiers on the **Denial of Wallet (DoW)** attack detection
dataset for serverless architectures, with an optional **GAN-style dataset
augmentation** pipeline.

---

## Table of Contents

1. [Project structure](#1-project-structure)
2. [Prerequisites](#2-prerequisites)
3. [Dependencies](#3-dependencies)
4. [Virtual environment setup](#4-virtual-environment-setup)
5. [Dataset](#5-dataset)
6. [General usage](#6-general-usage)
7. [Available models and CLI options](#7-available-models-and-cli-options)
8. [Running with augmentation](#8-running-with-augmentation)
9. [Augmentation parameters in detail](#9-augmentation-parameters-in-detail)
10. [Reproducibility notes](#10-reproducibility-notes)


---

## 1. Project structure

```
dow-detector/
│
├── main.py                  ← CLI entry point and orchestration
│
├── dow/
│   ├── __init__.py
│   ├── config.py            ← constants, model registries, paper values,
│   │                           augmentation defaults
│   ├── data.py              ← CSV loading, encoding, train/test split
│   ├── augment.py           ← GAN-style augmentation pipeline
│   ├── metrics.py           ← metric computation and console output
│   ├── ml_models.py         ← scikit-learn classifiers
│   └── dl_models.py         ← Keras / TensorFlow neural networks
│
├── requirements.txt
├── README.md
└── dataset.csv              ← place the dataset here before running
```

---

## 2. Prerequisites

- Python **3.10 – 3.12** (3.11 recommended)
- `dataset.csv` at the project root (target column: `bot`)

---

## 3. Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tensorflow>=2.13
```

## 4. Virtual environment setup

### Create and activate

```bash
# Linux / macOS
$ python3 -m venv .venv
$ source .venv/bin/activate

# Windows — Command Prompt
$ python -m venv .venv
$ .venv\Scripts\activate.bat

# Windows — PowerShell
$ python -m venv .venv
$ .venv\Scripts\Activate.ps1
```

The prompt changes to `(.venv) $` when active.

### Install dependencies

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** replace `tensorflow` in requirements.txt with:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### Deactivate when done

```bash
deactivate
```

---

## 5. Dataset

Place `dataset.csv` at the project root.
The file contains **187,087 serverless transactions** with 19 columns.

| Column | Type | Description |
|---|---|---|
| `bot` | bool | **Target** — `True` = DoW attack, `False` = legitimate |
| `IP` | str | Source IP address |
| `FunctionId` | int | Serverless function (45 unique functions) |
| `functionTrigger` | str | http · storage · sql · notification · stream |
| `FunctionDuration` | float | Execution duration (ms) |
| `maxcpu` / `avgcpu` | float | CPU usage metrics |
| … | | 16 features total after encoding |

**Class balance:** 131,072 attacks (70.1 %) · 56,015 benign (29.9 %)

---

## 6. General usage

```bash
# Always activate the virtual environment first
$ source .venv/bin/activate        # Linux / macOS
$ .venv\Scripts\activate.bat       # Windows

# General syntax
$ python main.py --model <name_or_group> [options]
```

---

## 7. Available models and CLI options

### Models

| Group | Name | Algorithm |
|---|---|---|
| `all_ml` | `decision_tree` | Decision Tree + isotonic calibration |
| `all_ml` | `random_forest` | Random Forest |
| `all_ml` | `gradient_boosting` | Gradient Boosting |
| `all_ml` | `naive_bayes` | Gaussian Naive Bayes |
| `all_ml` | `kneighbors` | K-Nearest Neighbours |
| `all_dl` | `mlp` | Multi-Layer Perceptron |
| `all_dl` | `lstm` | Stacked LSTM (2 layers) |
| `all_dl` | `bilstm` | Bidirectional LSTM |
| `all_dl` | `gru` | Stacked GRU (2 layers) |

Groups: `all_ml` · `all_dl` · `all`

### Core options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model name or group |
| `--dataset` | `dataset.csv` | Path to the CSV file |
| `--epochs` | `20` | DL training epochs |
| `--no-scale` | off | Disable StandardScaler for ML models |
| `--chronological` | off | Strict time-order split instead of stratified |

### Augmentation options

| Option | Default | Description |
|---|---|---|
| `--augment` | off | **Enable GAN-style augmentation** |
| `--aug-n-funcs` | `50` | Synthetic serverless functions to create |
| `--aug-n-bot` | `1500` | Attack rows per synthetic function |
| `--aug-n-legit` | `15` | Legit rows per synthetic function |
| `--aug-noise` | `0.05` | Gaussian noise scale (fraction of feature std) |
| `--aug-stealth-ip` | `0.40` | Fraction of synthetic attacks with legitimate IP |
| `--aug-ip-merge` | `0.15` | Fraction of original bot rows re-assigned legit IP |


---

##8. Running with augmentation

Adding `--augment` activates the GAN-style pipeline before training.
It synthesises new serverless function profiles and introduces stealth
attacks with legitimate IPs.

```bash
# Default augmentation (recommended starting point for ML models)
python main.py --model decision_tree  --augment
python main.py --model all_ml         --augment
python main.py --model all            --augment --epochs 20

# Larger augmentation — more synthetic functions and attack rows
python main.py --model decision_tree --augment --aug-n-funcs 80 --aug-n-bot 2000

# DL models benefit less from augmentation but run without issues
python main.py --model bilstm --augment --epochs 20
```

**Expected output (augmentation n-funcs=80, n-bot=2000):**

```
$ python main.py --model all_ml --augment --aug-n-funcs 80 --aug-n-bot 2000
==========================================================================
  DoW Attack Detector – Stage 2 Classifier
  Research: 'Hybrid Model for Detecting Denial of Wallet Attacks'
==========================================================================
  Models       : ['decision_tree', 'random_forest', 'gradient_boosting', 'naive_bayes', 'kneighbors']
  Dataset      : dataset.csv
  Split mode   : Stratified random  (70% train / 30% test)
  DL epochs    : 20
  Augmentation : ON
    synthetic functions : 80
    bot rows / func     : 2000
    legit rows / func   : 15
    noise scale         : 0.05
    stealth IP fraction : 0.4
    IP merge fraction   : 0.15
==========================================================================

[AUG] Starting GAN-style augmentation pipeline …
[AUG] Original  :  187,087 rows  (70.1% attack, 45 functions)
[AUG] Synthetic :  161,200 rows  (80 functions × 2000 attack + 15 legit, 40% stealth IP)
[AUG] IP merge  : 15% of original bot rows → shared IP pool = 7,487
[AUG] Final     :  348,287 rows  (83.6% attack, 125 functions, 16 features)
[AUG] Split (stratified random) — train: 243,800  |  test: 104,487  (benign 17,165 / attack 87,322)

[ML] Training DECISION_TREE …
    Decision threshold : 0.65
==========================================================================
  MODEL : DECISION_TREE
--------------------------------------------------------------------------
  Metric                      Value
--------------------------------------------------------------------------
  Precision                  0.9391
  Recall                     0.9144
  F1-score                   0.9266
  Accuracy                   0.8789
  ROC-AUC                    0.9288
  PR-AUC                     0.9826
  TPR@1%FPR                  0.5785
  TPR@5%FPR                  0.8043
--------------------------------------------------------------------------
  Training / inference time : 6.18s
==========================================================================
--------------------------------------------------------------------------
  Confusion Matrix
                      Pred Benign    Pred Attack
  Actual Benign            11,985          5,180
  Actual Attack             7,475         79,847
--------------------------------------------------------------------------
              precision    recall  f1-score   support

      Benign       0.62      0.70      0.65     17165
      Attack       0.94      0.91      0.93     87322

    accuracy                           0.88    104487
   macro avg       0.78      0.81      0.79    104487
weighted avg       0.89      0.88      0.88    104487


[ML] Training RANDOM_FOREST …
==========================================================================
  MODEL : RANDOM_FOREST
--------------------------------------------------------------------------
  Metric                      Value
--------------------------------------------------------------------------
  Precision                  0.8565
  Recall                     0.9929
  F1-score                   0.9197
  Accuracy                   0.8551
  ROC-AUC                    0.8867
  PR-AUC                     0.9739
  TPR@1%FPR                  0.2725
  TPR@5%FPR                  0.6923
--------------------------------------------------------------------------
  Training / inference time : 4.16s
==========================================================================
--------------------------------------------------------------------------
  Confusion Matrix
                      Pred Benign    Pred Attack
  Actual Benign             2,637         14,528
  Actual Attack               616         86,706
--------------------------------------------------------------------------
              precision    recall  f1-score   support

      Benign       0.81      0.15      0.26     17165
      Attack       0.86      0.99      0.92     87322

    accuracy                           0.86    104487
   macro avg       0.83      0.57      0.59    104487
weighted avg       0.85      0.86      0.81    104487


[ML] Training GRADIENT_BOOSTING …
==========================================================================
  MODEL : GRADIENT_BOOSTING
--------------------------------------------------------------------------
  Metric                      Value
--------------------------------------------------------------------------
  Precision                  0.9069
  Recall                     0.9782
  F1-score                   0.9412
  Accuracy                   0.8979
  ROC-AUC                    0.9447
  PR-AUC                     0.9877
  TPR@1%FPR                  0.4322
  TPR@5%FPR                  0.7659
--------------------------------------------------------------------------
  Training / inference time : 74.31s
==========================================================================
--------------------------------------------------------------------------
  Confusion Matrix
                      Pred Benign    Pred Attack
  Actual Benign             8,394          8,771
  Actual Attack             1,902         85,420
--------------------------------------------------------------------------
              precision    recall  f1-score   support

      Benign       0.82      0.49      0.61     17165
      Attack       0.91      0.98      0.94     87322

    accuracy                           0.90    104487
   macro avg       0.86      0.73      0.78    104487
weighted avg       0.89      0.90      0.89    104487


[ML] Training NAIVE_BAYES …
==========================================================================
  MODEL : NAIVE_BAYES
--------------------------------------------------------------------------
  Metric                      Value
--------------------------------------------------------------------------
  Precision                  0.9429
  Recall                     0.7754
  F1-score                   0.8510
  Accuracy                   0.7730
  ROC-AUC                    0.8365
  PR-AUC                     0.9628
  TPR@1%FPR                  0.0000
  TPR@5%FPR                  0.6550
--------------------------------------------------------------------------
  Training / inference time : 0.04s
==========================================================================
--------------------------------------------------------------------------
  Confusion Matrix
                      Pred Benign    Pred Attack
  Actual Benign            13,061          4,104
  Actual Attack            19,613         67,709
--------------------------------------------------------------------------
              precision    recall  f1-score   support

      Benign       0.40      0.76      0.52     17165
      Attack       0.94      0.78      0.85     87322

    accuracy                           0.77    104487
   macro avg       0.67      0.77      0.69    104487
weighted avg       0.85      0.77      0.80    104487


[ML] Training KNEIGHBORS …
==========================================================================
  MODEL : KNEIGHBORS
--------------------------------------------------------------------------
  Metric                      Value
--------------------------------------------------------------------------
  Precision                  0.9386
  Recall                     0.9628
  F1-score                   0.9505
  Accuracy                   0.9163
  ROC-AUC                    0.9097
  PR-AUC                     0.9702
  TPR@1%FPR                  0.0000
  TPR@5%FPR                  0.0000
--------------------------------------------------------------------------
  Training / inference time : 0.01s
==========================================================================
--------------------------------------------------------------------------
  Confusion Matrix
                      Pred Benign    Pred Attack
  Actual Benign            11,667          5,498
  Actual Attack             3,250         84,072
--------------------------------------------------------------------------
              precision    recall  f1-score   support

      Benign       0.78      0.68      0.73     17165
      Attack       0.94      0.96      0.95     87322

    accuracy                           0.92    104487
   macro avg       0.86      0.82      0.84    104487
weighted avg       0.91      0.92      0.91    104487

==========================================================================
  FULL RESULTS SUMMARY
--------------------------------------------------------------------------
  Model                  Prec    Rec     F1    Acc    ROC     PR   T@1%   T@5%
--------------------------------------------------------------------------
  decision_tree        0.9391 0.9144 0.9266 0.8789 0.9288 0.9826 0.5785 0.8043
  random_forest        0.8565 0.9929 0.9197 0.8551 0.8867 0.9739 0.2725 0.6923
  gradient_boosting    0.9069 0.9782 0.9412 0.8979 0.9447 0.9877 0.4322 0.7659
  naive_bayes          0.9429 0.7754 0.8510 0.7730 0.8365 0.9628 0.0000 0.6550
  kneighbors           0.9386 0.9628 0.9505 0.9163 0.9097 0.9702 0.0000 0.0000
==========================================================================

```

---

## 9. Augmentation parameters in detail

### `--aug-n-funcs` — Number of synthetic functions

Controls how many new virtual serverless functions are interpolated from
existing prototypes. More functions → more diverse feature vectors → smoother
ROC curve with more operating points.

```bash
python main.py --model decision_tree --augment --aug-n-funcs 30   # minimal
python main.py --model decision_tree --augment --aug-n-funcs 50   # default
python main.py --model decision_tree --augment --aug-n-funcs 80   # recommended
python main.py --model decision_tree --augment --aug-n-funcs 150  # large
```

### `--aug-n-bot` — Attack rows per synthetic function

Scales the attack density in the augmented dataset. Higher values increase
total dataset size and improve the classifier's ability to generalise on
attack patterns.

```bash
python main.py --model all_ml --augment --aug-n-bot 1000  # light
python main.py --model all_ml --augment --aug-n-bot 1500  # default
python main.py --model all_ml --augment --aug-n-bot 2000  # recommended
python main.py --model all_ml --augment --aug-n-bot 2900  # matches original density
```

### `--aug-stealth-ip` — Stealth attack IP fraction

The fraction of synthetic attack rows assigned a **legitimate IP** instead
of a bot IP.

```bash
python main.py --model decision_tree --augment --aug-stealth-ip 0.20  # few stealth
python main.py --model decision_tree --augment --aug-stealth-ip 0.40  # default
python main.py --model decision_tree --augment --aug-stealth-ip 0.60  # more stealth
```

### `--aug-ip-merge` — IP merge fraction

Fraction of **original** bot rows re-assigned a legitimate IP.

```bash
python main.py --model decision_tree --augment --aug-ip-merge 0.10  # light merge
python main.py --model decision_tree --augment --aug-ip-merge 0.15  # default
python main.py --model decision_tree --augment --aug-ip-merge 0.25  # heavy merge
```

### `--aug-noise` — Feature noise scale

Gaussian noise as a fraction of each feature's standard deviation.

```bash
python main.py --model decision_tree --augment --aug-noise 0.03  # subtle
python main.py --model decision_tree --augment --aug-noise 0.05  # default
python main.py --model decision_tree --augment --aug-noise 0.10  # stronger
```

---

## 10. Reproducibility notes

| Aspect | Detail |
|---|---|
| **Split** | Default: stratified random (seed 42). `--chronological` activates the time-order split. |
| **IP feature** | Label-encoded ordinal, kept as a feature. |
| **DT calibration** | Wrapped in `CalibratedClassifierCV(isotonic, cv=5)` in `ml_models.py`. Adds ~2 s. |
| **Augmentation seed** | Fixed at `RANDOM_STATE=42` in `config.py`. Override via `AugmentConfig.random_state`. |
| **Early stopping (DL)** | `patience=3` on `val_loss`, 10% of training set as validation. |

---


