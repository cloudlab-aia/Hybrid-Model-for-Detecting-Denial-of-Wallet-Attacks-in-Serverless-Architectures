#!/usr/bin/env python3
"""
main.py
=======
Command-line entry point for the DoW Attack Detection Stage-2 Classifier.

Project structure
-----------------
  main.py          ← CLI entry point and orchestration  (this file)
  dow/
    config.py      ← constants, model registries, paper reference values
    data.py        ← CSV loading, encoding, train/test split
    augment.py     ← GAN-style dataset augmentation pipeline
    metrics.py     ← evaluation metrics + all console-output helpers
    ml_models.py   ← scikit-learn classifiers  (ML)
    dl_models.py   ← Keras / TensorFlow neural networks  (DL)

Usage
-----
  # Without augmentation
  python main.py --model decision_tree
  python main.py --model bilstm --epochs 20
  python main.py --model all_ml
  python main.py --model all_dl
  python main.py --model all
  python main.py --model lstm --chronological
  python main.py --model kneighbors --no-scale
  python main.py --model all --dataset /path/to/data.csv

  # With GAN-style augmentation (recommended for ML models)
  python main.py --model decision_tree --augment
  python main.py --model all_ml --augment
  python main.py --model all --augment --epochs 20

  # Customise augmentation parameters
  python main.py --model all_ml --augment --aug-n-funcs 80 --aug-n-bot 2000
  python main.py --model decision_tree --augment --aug-ip-merge 0.20
"""

import argparse
import sys
import traceback

import pandas as pd

from dow.augment import AugmentConfig, augmented_split
from dow.config import (
    ALL_DL_MODELS,
    ALL_ML_MODELS,
    ALL_MODELS,
    AUG_BOT_LEGIT_IP_FRACTION,
    AUG_IP_MERGE_FRACTION,
    AUG_N_BOT_PER_FUNC,
    AUG_N_LEGIT_PER_FUNC,
    AUG_N_SYNTHETIC_FUNCTIONS,
    AUG_NOISE_SCALE,
    DATASET_PATH,
    DEFAULT_EPOCHS,
    TRAIN_RATIO,
)
from dow.data import load_and_split, load_raw
from dow.dl_models import run_dl
from dow.metrics import print_summary
from dow.ml_models import run_ml


# Display helpers
def _sep(char: str = "=", width: int = 74) -> None:
    print(char * width)


def _print_header(
    models:       list,
    dataset:      str,
    split_mode:   str,
    epochs:       int,
    augment:      bool,
    aug_cfg:      AugmentConfig | None,
) -> None:
    _sep()
    print("  DoW Attack Detector – Stage 2 Classifier")
    print("  Research: 'Hybrid Model for Detecting Denial of Wallet Attacks'")
    _sep()
    print(f"  Models       : {models}")
    print(f"  Dataset      : {dataset}")
    print(
        f"  Split mode   : {split_mode}  "
        f"({int(TRAIN_RATIO*100)}% train / "
        f"{int((1-TRAIN_RATIO)*100)}% test)"
    )
    print(f"  DL epochs    : {epochs}")
    if augment and aug_cfg:
        print(f"  Augmentation : ON")
        print(f"    synthetic functions : {aug_cfg.n_synthetic_functions}")
        print(f"    bot rows / func     : {aug_cfg.n_bot_per_func}")
        print(f"    legit rows / func   : {aug_cfg.n_legit_per_func}")
        print(f"    noise scale         : {aug_cfg.noise_scale}")
        print(f"    stealth IP fraction : {aug_cfg.bot_legit_ip_fraction}")
        print(f"    IP merge fraction   : {aug_cfg.ip_merge_fraction}")
    else:
        print(f"  Augmentation : OFF  (use --augment to enable)")
    _sep()


# Model resolution
def _resolve_models(key: str) -> list:
    key = key.lower().strip()
    if key == "all":
        return ALL_MODELS
    if key == "all_ml":
        return ALL_ML_MODELS
    if key == "all_dl":
        return ALL_DL_MODELS
    if key in ALL_MODELS:
        return [key]

    valid = ALL_MODELS + ["all", "all_ml", "all_dl"]
    print(f"[ERROR] Unknown model/group '{key}'.\n        Valid: {valid}")
    sys.exit(1)


# CLI definition
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = "DoW Attack Detection – Stage 2 Classifier",
        formatter_class = argparse.RawTextHelpFormatter,
        epilog="""
Available models
----------------
  ML  : decision_tree  random_forest  gradient_boosting  naive_bayes  kneighbors
  DL  : mlp  lstm  bilstm  gru
  Groups: all  all_ml  all_dl

Quick examples
--------------
  # Standard run (original dataset)
  python main.py --model all_ml
  python main.py --model bilstm --epochs 20

  # With GAN augmentation (improves ML precision and TPR@5%FPR)
  python main.py --model all_ml --augment
  python main.py --model all --augment --epochs 20

  # Custom augmentation parameters
  python main.py --model decision_tree --augment --aug-n-funcs 80 --aug-n-bot 2000
  python main.py --model all_ml --augment --aug-ip-merge 0.10 --aug-stealth-ip 0.50
        """,
    )

    # Core options
    p.add_argument("--model",   required=True, metavar="NAME",
                   help="Model name or group (see above).")
    p.add_argument("--dataset", default=DATASET_PATH, metavar="PATH",
                   help=f"Path to the CSV dataset  (default: {DATASET_PATH})")
    p.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS, metavar="N",
                   help=f"DL training epochs  (default: {DEFAULT_EPOCHS})")
    p.add_argument("--no-scale", action="store_true",
                   help="Disable StandardScaler for ML models.")
    p.add_argument("--chronological", action="store_true",
                   help="Strict time-order 70/30 split instead of stratified random.")

    # Augmentation flag
    p.add_argument("--augment", action="store_true",
                   help=(
                       "Enable GAN-style augmentation.\n"
                       "Synthesises new serverless function profiles and\n"
                       "introduces stealth attacks with legitimate IPs.\n"
                   ))

    # Augmentation parameters (only active when --augment is set)
    aug = p.add_argument_group("augmentation parameters (require --augment)")
    aug.add_argument("--aug-n-funcs", type=int,
                     default=AUG_N_SYNTHETIC_FUNCTIONS, metavar="N",
                     help=f"Number of synthetic serverless functions  "
                          f"(default: {AUG_N_SYNTHETIC_FUNCTIONS})")
    aug.add_argument("--aug-n-bot", type=int,
                     default=AUG_N_BOT_PER_FUNC, metavar="N",
                     help=f"Attack rows per synthetic function  "
                          f"(default: {AUG_N_BOT_PER_FUNC})")
    aug.add_argument("--aug-n-legit", type=int,
                     default=AUG_N_LEGIT_PER_FUNC, metavar="N",
                     help=f"Legit rows per synthetic function  "
                          f"(default: {AUG_N_LEGIT_PER_FUNC})")
    aug.add_argument("--aug-noise", type=float,
                     default=AUG_NOISE_SCALE, metavar="F",
                     help=f"Gaussian noise scale  (default: {AUG_NOISE_SCALE})")
    aug.add_argument("--aug-stealth-ip", type=float,
                     default=AUG_BOT_LEGIT_IP_FRACTION, metavar="F",
                     help=f"Fraction of synthetic attacks with legitimate IP  "
                          f"(default: {AUG_BOT_LEGIT_IP_FRACTION})")
    aug.add_argument("--aug-ip-merge", type=float,
                     default=AUG_IP_MERGE_FRACTION, metavar="F",
                     help=f"Fraction of original bot rows re-assigned legit IP  "
                          f"(default: {AUG_IP_MERGE_FRACTION})")

    return p


# Main
def main() -> None:
    args      = build_parser().parse_args()
    to_run    = _resolve_models(args.model)
    use_scale = not args.no_scale

    # Build augmentation config if requested
    aug_cfg = None
    if args.augment:
        aug_cfg = AugmentConfig(
            n_synthetic_functions = args.aug_n_funcs,
            n_bot_per_func        = args.aug_n_bot,
            n_legit_per_func      = args.aug_n_legit,
            noise_scale           = args.aug_noise,
            bot_legit_ip_fraction = args.aug_stealth_ip,
            ip_merge_fraction     = args.aug_ip_merge,
        )

    split_label = "Chronological" if args.chronological else "Stratified random"
    _print_header(to_run, args.dataset, split_label, args.epochs, args.augment, aug_cfg)

    # Load dataset (augmented or original)
    if args.augment:
        df_raw = load_raw(args.dataset)
        X_tr, X_te, y_tr, y_te, feat_cols = augmented_split(
            df_raw,
            cfg           = aug_cfg,
            chronological = args.chronological,
            verbose       = True,
        )
    else:
        X_tr, X_te, y_tr, y_te, feat_cols = load_and_split(
            args.dataset,
            chronological = args.chronological,
        )

    # Train and evaluate each model
    results: dict = {}
    for name in to_run:
        try:
            if name in ALL_ML_MODELS:
                m = run_ml(name, X_tr, X_te, y_tr, y_te, use_scale=use_scale)
            else:
                m = run_dl(name, X_tr, X_te, y_tr, y_te, epochs=args.epochs)
            results[name] = m

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted — showing partial results …")
            break

        except Exception as exc:
            print(f"\n[ERROR] Model '{name}' failed: {exc}")
            traceback.print_exc()
            print("[INFO] Continuing with next model …\n")

    # Summary (shown when ≥ 2 models ran)
    print_summary(results)


if __name__ == "__main__":
    main()
