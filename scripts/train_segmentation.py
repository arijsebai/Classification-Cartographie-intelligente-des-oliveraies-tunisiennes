"""Scaffold training script for segmentation (U-Net with ResNet backbone).

This file is a scaffold: fill dataset loading (Sentinel-2 patches), preprocessing
and model hyperparameters. It uses `segmentation_models_pytorch` if available.

Usage example (scaffold):
  python -m scripts.train_segmentation --train data/splits/train.geojson --val data/splits/val.geojson --epochs 30
"""
from __future__ import annotations

import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()

    try:
        import segmentation_models_pytorch as smp
    except Exception:
        print("segmentation_models_pytorch not installed. Install it to train the U-Net.")
        print("pip install segmentation-models-pytorch torch torchvision")
        sys.exit(1)

    print("Scaffold: load train/val splits, build dataset of Sentinel-2 patches and train U-Net here.")
    print(f"Train file: {args.train}")
    print(f"Val file: {args.val}")


if __name__ == "__main__":
    main()
