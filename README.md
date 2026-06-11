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
10. [Stratified K-Fold cross-validation for ML and DL classifiers](#10-Stratified K-Fold cross-validation for ML and DL classifiers)
11. [Reproducibility notes](#11-reproducibility-notes)


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
|   ├── cross_validation.py  ← Stratified K-Fold cross-validation for ML and DL classifiers
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

## 10. Stratified K-Fold cross-validation for ML and DL classifiers

# All ML classifiers with 10-fold CV
```bash
python main.py --model all_ml --cv --cv-folds 10

==========================================================================
  DoW Attack Detector – Stage 2 Classifier
  Research: 'Hybrid Model for Detecting Denial of Wallet Attacks'
==========================================================================
  Models       : ['decision_tree', 'random_forest', 'gradient_boosting', 'naive_bayes', 'kneighbors']
  Dataset      : dataset.csv
  Split mode   : Stratified random  (70% train / 30% test)
  DL epochs    : 20
  Augmentation : OFF  (use --augment to enable)
==========================================================================

[DATA] Loading 'dataset.csv' for cross-validation …
[DATA] 187,087 rows  ·  16 features  ·  attack ratio 70.1%
==========================================================================
  DoW Attack Detector – Cross-Validation Mode
  Models  : ['decision_tree', 'random_forest', 'gradient_boosting', 'naive_bayes', 'kneighbors']
  Folds   : 10-fold Stratified K-Fold
  Dataset : dataset.csv
==========================================================================
==========================================================================
  CROSS-VALIDATION : DECISION_TREE
  Folds   : 10-fold Stratified K-Fold
  Samples : 187,087
--------------------------------------------------------------------------
  Fold  1 │ Prec 0.8448  Rec 0.8364  F1 0.8406  Acc 0.7777  ROC 0.8676  PR 0.9390  T@1% 0.4847  T@5% 0.5769
  Fold  2 │ Prec 0.8342  Rec 0.8566  F1 0.8453  Acc 0.7803  ROC 0.8612  PR 0.9350  T@1% 0.5052  T@5% 0.5512
  Fold  3 │ Prec 0.8212  Rec 0.8647  F1 0.8424  Acc 0.7733  ROC 0.8565  PR 0.9351  T@1% 0.4798  T@5% 0.5716
  Fold  4 │ Prec 0.8317  Rec 0.8218  F1 0.8267  Acc 0.7586  ROC 0.8603  PR 0.9372  T@1% 0.4297  T@5% 0.5938
  Fold  5 │ Prec 0.8459  Rec 0.8395  F1 0.8427  Acc 0.7804  ROC 0.8702  PR 0.9387  T@1% 0.4730  T@5% 0.5624
  Fold  6 │ Prec 0.8356  Rec 0.8627  F1 0.8489  Acc 0.7849  ROC 0.8623  PR 0.9360  T@1% 0.4626  T@5% 0.5632
  Fold  7 │ Prec 0.8513  Rec 0.8334  F1 0.8423  Acc 0.7813  ROC 0.8690  PR 0.9374  T@1% 0.4684  T@5% 0.5400
  Fold  8 │ Prec 0.8553  Rec 0.7792  F1 0.8155  Acc 0.7529  ROC 0.8665  PR 0.9392  T@1% 0.4707  T@5% 0.5855
  Fold  9 │ Prec 0.8432  Rec 0.8534  F1 0.8482  Acc 0.7861  ROC 0.8735  PR 0.9406  T@1% 0.4664  T@5% 0.5419
  Fold 10 │ Prec 0.8402  Rec 0.8109  F1 0.8253  Acc 0.7595  ROC 0.8556  PR 0.9292  T@1% 0.4281  T@5% 0.5373
--------------------------------------------------------------------------
  Metric                       Mean        ±Std
--------------------------------------------------------------------------
  Precision                  0.8403  ±  0.0095
  Recall                     0.8359  ±  0.0252
  F1-score                   0.8378  ±  0.0107
  Accuracy                   0.7735  ±  0.0114
  ROC-AUC                    0.8643  ±  0.0057
  PR-AUC                     0.9367  ±  0.0031
  TPR@1%FPR                  0.4669  ±  0.0222
  TPR@5%FPR                  0.5624  ±  0.0187
--------------------------------------------------------------------------
  Total CV time : 6.2s
==========================================================================
==========================================================================
  CROSS-VALIDATION : RANDOM_FOREST
  Folds   : 10-fold Stratified K-Fold
  Samples : 187,087
--------------------------------------------------------------------------
  Fold  1 │ Prec 0.7509  Rec 0.9988  F1 0.8573  Acc 0.7670  ROC 0.8392  PR 0.9225  T@1% 0.2908  T@5% 0.4592
  Fold  2 │ Prec 0.7582  Rec 0.9989  F1 0.8620  Acc 0.7760  ROC 0.8459  PR 0.9258  T@1% 0.2856  T@5% 0.4600
  Fold  3 │ Prec 0.7568  Rec 0.9997  F1 0.8615  Acc 0.7748  ROC 0.8458  PR 0.9260  T@1% 0.2943  T@5% 0.4633
  Fold  4 │ Prec 0.7530  Rec 0.9996  F1 0.8590  Acc 0.7701  ROC 0.8427  PR 0.9240  T@1% 0.2844  T@5% 0.4508
  Fold  5 │ Prec 0.7578  Rec 0.9996  F1 0.8621  Acc 0.7759  ROC 0.8471  PR 0.9258  T@1% 0.2708  T@5% 0.4549
  Fold  6 │ Prec 0.7576  Rec 0.9996  F1 0.8620  Acc 0.7757  ROC 0.8442  PR 0.9254  T@1% 0.2894  T@5% 0.4553
  Fold  7 │ Prec 0.7558  Rec 0.9991  F1 0.8606  Acc 0.7733  ROC 0.8460  PR 0.9253  T@1% 0.2808  T@5% 0.4458
  Fold  8 │ Prec 0.7585  Rec 0.9986  F1 0.8622  Acc 0.7763  ROC 0.8486  PR 0.9255  T@1% 0.2873  T@5% 0.4607
  Fold  9 │ Prec 0.7621  Rec 0.9996  F1 0.8648  Acc 0.7811  ROC 0.8442  PR 0.9242  T@1% 0.2930  T@5% 0.4449
  Fold 10 │ Prec 0.7574  Rec 0.9997  F1 0.8619  Acc 0.7755  ROC 0.8459  PR 0.9257  T@1% 0.2647  T@5% 0.4617
--------------------------------------------------------------------------
  Metric                       Mean        ±Std
--------------------------------------------------------------------------
  Precision                  0.7568  ±  0.0029
  Recall                     0.9993  ±  0.0004
  F1-score                   0.8613  ±  0.0019
  Accuracy                   0.7746  ±  0.0036
  ROC-AUC                    0.8450  ±  0.0025
  PR-AUC                     0.9250  ±  0.0011
  TPR@1%FPR                  0.2841  ±  0.0091
  TPR@5%FPR                  0.4557  ±  0.0062
--------------------------------------------------------------------------
  Total CV time : 10.4s
==========================================================================
==========================================================================
  CROSS-VALIDATION : GRADIENT_BOOSTING
  Folds   : 10-fold Stratified K-Fold
  Samples : 187,087
--------------------------------------------------------------------------
  Fold  1 │ Prec 0.8810  Rec 1.0000  F1 0.9368  Acc 0.9054  ROC 0.9688  PR 0.9834  T@1% 0.4573  T@5% 0.7704
  Fold  2 │ Prec 0.8900  Rec 0.9986  F1 0.9412  Acc 0.9126  ROC 0.9753  PR 0.9869  T@1% 0.4687  T@5% 0.8371
  Fold  3 │ Prec 0.8924  Rec 1.0000  F1 0.9431  Acc 0.9155  ROC 0.9776  PR 0.9882  T@1% 0.5268  T@5% 0.8305
  Fold  4 │ Prec 0.8763  Rec 1.0000  F1 0.9341  Acc 0.9011  ROC 0.9751  PR 0.9869  T@1% 0.5126  T@5% 0.8398
  Fold  5 │ Prec 0.8842  Rec 1.0000  F1 0.9386  Acc 0.9083  ROC 0.9752  PR 0.9865  T@1% 0.4893  T@5% 0.8582
  Fold  6 │ Prec 0.8959  Rec 1.0000  F1 0.9451  Acc 0.9186  ROC 0.9756  PR 0.9872  T@1% 0.5153  T@5% 0.8254
  Fold  7 │ Prec 0.8899  Rec 1.0000  F1 0.9418  Acc 0.9134  ROC 0.9759  PR 0.9873  T@1% 0.4894  T@5% 0.8414
  Fold  8 │ Prec 0.8867  Rec 1.0000  F1 0.9399  Acc 0.9105  ROC 0.9745  PR 0.9865  T@1% 0.4845  T@5% 0.8132
  Fold  9 │ Prec 0.8876  Rec 1.0000  F1 0.9405  Acc 0.9113  ROC 0.9733  PR 0.9857  T@1% 0.4675  T@5% 0.8151
  Fold 10 │ Prec 0.8835  Rec 0.9992  F1 0.9378  Acc 0.9072  ROC 0.9753  PR 0.9871  T@1% 0.5075  T@5% 0.8201
--------------------------------------------------------------------------
  Metric                       Mean        ±Std
--------------------------------------------------------------------------
  Precision                  0.8868  ±  0.0054
  Recall                     0.9998  ±  0.0005
  F1-score                   0.9399  ±  0.0030
  Accuracy                   0.9104  ±  0.0048
  ROC-AUC                    0.9747  ±  0.0022
  PR-AUC                     0.9866  ±  0.0012
  TPR@1%FPR                  0.4919  ±  0.0220
  TPR@5%FPR                  0.8251  ±  0.0224
--------------------------------------------------------------------------
  Total CV time : 192.9s
==========================================================================
==========================================================================
  CROSS-VALIDATION : NAIVE_BAYES
  Folds   : 10-fold Stratified K-Fold
  Samples : 187,087
--------------------------------------------------------------------------
  Fold  1 │ Prec 0.8182  Rec 0.5790  F1 0.6781  Acc 0.6149  ROC 0.6780  PR 0.8453  T@1% 0.1086  T@5% 0.2890
  Fold  2 │ Prec 0.8197  Rec 0.5803  F1 0.6795  Acc 0.6165  ROC 0.6830  PR 0.8474  T@1% 0.1096  T@5% 0.2999
  Fold  3 │ Prec 0.8220  Rec 0.5708  F1 0.6738  Acc 0.6128  ROC 0.6777  PR 0.8446  T@1% 0.1113  T@5% 0.2935
  Fold  4 │ Prec 0.8179  Rec 0.5709  F1 0.6724  Acc 0.6103  ROC 0.6760  PR 0.8454  T@1% 0.1159  T@5% 0.2976
  Fold  5 │ Prec 0.8224  Rec 0.5815  F1 0.6813  Acc 0.6188  ROC 0.6824  PR 0.8464  T@1% 0.1115  T@5% 0.2944
  Fold  6 │ Prec 0.8186  Rec 0.5721  F1 0.6735  Acc 0.6114  ROC 0.6794  PR 0.8461  T@1% 0.1121  T@5% 0.2959
  Fold  7 │ Prec 0.8216  Rec 0.5743  F1 0.6761  Acc 0.6144  ROC 0.6802  PR 0.8448  T@1% 0.1086  T@5% 0.2842
  Fold  8 │ Prec 0.8189  Rec 0.5682  F1 0.6709  Acc 0.6094  ROC 0.6778  PR 0.8462  T@1% 0.1143  T@5% 0.3025
  Fold  9 │ Prec 0.8258  Rec 0.5725  F1 0.6762  Acc 0.6159  ROC 0.6800  PR 0.8453  T@1% 0.1098  T@5% 0.2919
  Fold 10 │ Prec 0.8201  Rec 0.5736  F1 0.6750  Acc 0.6131  ROC 0.6824  PR 0.8493  T@1% 0.1082  T@5% 0.3056
--------------------------------------------------------------------------
  Metric                       Mean        ±Std
--------------------------------------------------------------------------
  Precision                  0.8205  ±  0.0023
  Recall                     0.5743  ±  0.0042
  F1-score                   0.6757  ±  0.0031
  Accuracy                   0.6138  ±  0.0028
  ROC-AUC                    0.6797  ±  0.0022
  PR-AUC                     0.8461  ±  0.0013
  TPR@1%FPR                  0.1110  ±  0.0024
  TPR@5%FPR                  0.2955  ±  0.0060
--------------------------------------------------------------------------
  Total CV time : 0.6s
==========================================================================
==========================================================================
  CROSS-VALIDATION : KNEIGHBORS
  Folds   : 10-fold Stratified K-Fold
  Samples : 187,087
--------------------------------------------------------------------------
  Fold  1 │ Prec 0.9352  Rec 1.0000  F1 0.9665  Acc 0.9514  ROC 0.9652  PR 0.9711  T@1% 0.0000  T@5% 0.9998
  Fold  2 │ Prec 0.9366  Rec 1.0000  F1 0.9673  Acc 0.9526  ROC 0.9648  PR 0.9708  T@1% 0.0000  T@5% 0.9998
  Fold  3 │ Prec 0.9393  Rec 1.0000  F1 0.9687  Acc 0.9547  ROC 0.9671  PR 0.9727  T@1% 0.0000  T@5% 0.9998
  Fold  4 │ Prec 0.9373  Rec 1.0000  F1 0.9676  Acc 0.9531  ROC 0.9629  PR 0.9693  T@1% 0.0000  T@5% 0.9997
  Fold  5 │ Prec 0.9377  Rec 1.0000  F1 0.9678  Acc 0.9534  ROC 0.9669  PR 0.9725  T@1% 0.0000  T@5% 0.9998
  Fold  6 │ Prec 0.9362  Rec 1.0000  F1 0.9671  Acc 0.9523  ROC 0.9671  PR 0.9727  T@1% 0.0000  T@5% 0.9998
  Fold  7 │ Prec 0.9364  Rec 1.0000  F1 0.9672  Acc 0.9524  ROC 0.9642  PR 0.9703  T@1% 0.0000  T@5% 0.9995
  Fold  8 │ Prec 0.9377  Rec 1.0000  F1 0.9678  Acc 0.9534  ROC 0.9675  PR 0.9730  T@1% 0.0000  T@5% 0.9997
  Fold  9 │ Prec 0.9339  Rec 1.0000  F1 0.9658  Acc 0.9504  ROC 0.9649  PR 0.9709  T@1% 0.0000  T@5% 0.9996
  Fold 10 │ Prec 0.9339  Rec 1.0000  F1 0.9658  Acc 0.9504  ROC 0.9622  PR 0.9687  T@1% 0.0000  T@5% 0.9995
--------------------------------------------------------------------------
  Metric                       Mean        ±Std
--------------------------------------------------------------------------
  Precision                  0.9364  ±  0.0016
  Recall                     1.0000  ±  0.0000
  F1-score                   0.9672  ±  0.0009
  Accuracy                   0.9524  ±  0.0013
  ROC-AUC                    0.9653  ±  0.0018
  PR-AUC                     0.9712  ±  0.0014
  TPR@1%FPR                  0.0000  ±  0.0000
  TPR@5%FPR                  0.9997  ±  0.0001
--------------------------------------------------------------------------
  Total CV time : 15.0s
==========================================================================
==========================================================================
  CROSS-VALIDATION SUMMARY  (mean ± std across folds)
--------------------------------------------------------------------------
  Model                   Precision       Recall           F1    Acc(Test)      ROC-AUC       PR-AUC    TPR@1%FPR    TPR@5%FPR
--------------------------------------------------------------------------
  decision_tree        0.8403±0.0095  0.8359±0.0252  0.8378±0.0107  0.7735±0.0114  0.8643±0.0057  0.9367±0.0031  0.4669±0.0222  0.5624±0.0187
  random_forest        0.7568±0.0029  0.9993±0.0004  0.8613±0.0019  0.7746±0.0036  0.8450±0.0025  0.9250±0.0011  0.2841±0.0091  0.4557±0.0062
  gradient_boosting    0.8868±0.0054  0.9998±0.0005  0.9399±0.0030  0.9104±0.0048  0.9747±0.0022  0.9866±0.0012  0.4919±0.0220  0.8251±0.0224
  naive_bayes          0.8205±0.0023  0.5743±0.0042  0.6757±0.0031  0.6138±0.0028  0.6797±0.0022  0.8461±0.0013  0.1110±0.0024  0.2955±0.0060
  kneighbors           0.9364±0.0016  1.0000±0.0000  0.9672±0.0009  0.9524±0.0013  0.9653±0.0018  0.9712±0.0014  0.0000±0.0000  0.9997±0.0001
==========================================================================
  Note: values formatted as mean±std  (e.g. 0.9360±0.0041 means mean=0.9360, std=0.0041)
==========================================================================


# All DL classifiers with 5-fold CV(with early stopping)
python main.py --model all_dl --cv --epochs 20

# All ML and DL classifiers with 10-fold CV
python main.py --model all --cv --cv-folds 10

# One model with 5-fold CV
python main.py --model bilstm --cv
python main.py --model decision_tree --cv --cv-folds 5
```

## 11. Reproducibility notes

| Aspect | Detail |
|---|---|
| **Split** | Default: stratified random (seed 42). `--chronological` activates the time-order split. |
| **IP feature** | Label-encoded ordinal, kept as a feature. |
| **DT calibration** | Wrapped in `CalibratedClassifierCV(isotonic, cv=5)` in `ml_models.py`. Adds ~2 s. |
| **Augmentation seed** | Fixed at `RANDOM_STATE=42` in `config.py`. Override via `AugmentConfig.random_state`. |
| **Early stopping (DL)** | `patience=3` on `val_loss`, 10% of training set as validation. |

---

