from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.segmentation_core import binary_dice_loss, binary_iou_from_logits, build_model, to_tensor_batch
from training.segmentation_dataset import EzzayraSegmentationDataset, load_split_records

logger = logging.getLogger("train_segmentation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)


@dataclass
class TrainConfig:
    train_split: str
    val_split: str
    image_root: str
    image_pattern: str
    bands: Sequence[int]
    tile_size: int
    tiles_per_polygon: int
    epochs: int
    batch_size: int
    lr: float
    patience: int
    cv_folds: int
    encoder_name: str
    out_dir: str
    num_workers: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train U-Net segmentation with spatial CV")
    parser.add_argument("--train-split", default="data_splits/ezzayra_oliviers/train.json")
    parser.add_argument("--val-split", default="data_splits/ezzayra_oliviers/val.json")
    parser.add_argument(
        "--image-root",
        default=r"C:\Users\SP\Desktop\Hack The Harvest\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\sentinel2_l2a\2025_05_06",
        help="Root folder containing train/val/test per-parcel multi-band TIFF stacks.",
    )
    parser.add_argument("--image-pattern", default="{cultivation_system}__{id}.tif")
    parser.add_argument("--bands", default="2,3,4,8,11")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tiles-per-polygon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--out-dir", default="models/segmentation")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    bands = tuple(int(x.strip()) for x in args.bands.split(",") if x.strip())
    return TrainConfig(
        train_split=args.train_split,
        val_split=args.val_split,
        image_root=args.image_root,
        image_pattern=args.image_pattern,
        bands=bands,
        tile_size=args.tile_size,
        tiles_per_polygon=args.tiles_per_polygon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        cv_folds=args.cv_folds,
        encoder_name=args.encoder_name,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
    )


def _group_records(records: List[Dict], group_key: str = "zone_id"):
    groups = [record.get(group_key) or record.get("governorate") or record.get("id") for record in records]
    return groups


def _build_datasets(records: List[Dict], cfg: TrainConfig, augment: bool, seed: int):
    return EzzayraSegmentationDataset(
        records=records,
        image_root=cfg.image_root,
        image_pattern=cfg.image_pattern,
        bands=cfg.bands,
        tile_size=cfg.tile_size,
        tiles_per_polygon=cfg.tiles_per_polygon,
        augment=augment,
        seed=seed,
    )


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for training") from exc

    def collate_fn(batch):
        images = [item[0] for item in batch]
        masks = [item[1] for item in batch]
        records = [item[2] for item in batch]
        image_tensor, mask_tensor = to_tensor_batch(images, masks)
        return image_tensor, mask_tensor, records

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )


def _train_epoch(model, loader, optimizer, device):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    model.train()
    total_loss = 0.0
    total_iou = 0.0
    loss_bce = torch.nn.BCEWithLogitsLoss()

    for images, masks, _records in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = loss_bce(logits, masks) + binary_dice_loss(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        total_iou += float(binary_iou_from_logits(logits.detach(), masks).detach())

    return total_loss / max(1, len(loader)), total_iou / max(1, len(loader))


def _evaluate_epoch(model, loader, device):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    loss_bce = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks, _records in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = loss_bce(logits, masks) + binary_dice_loss(logits, masks)
            total_loss += float(loss.detach())
            total_iou += float(binary_iou_from_logits(logits, masks).detach())

    return total_loss / max(1, len(loader)), total_iou / max(1, len(loader))


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_score = None
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _train_with_early_stopping(train_records: List[Dict], val_records: List[Dict], cfg: TrainConfig, fold_name: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for training") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(encoder_name=cfg.encoder_name, in_channels=len(cfg.bands), classes=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=cfg.patience)

    train_ds = _build_datasets(train_records, cfg, augment=True, seed=42)
    val_ds = _build_datasets(val_records, cfg, augment=False, seed=1337)
    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = _make_loader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{fold_name}_best.pth"
    best_iou = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_iou = _train_epoch(model, train_loader, optimizer, device)
        val_loss, val_iou = _evaluate_epoch(model, val_loader, device)
        scheduler.step(val_iou)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
        )
        logger.info(
            "%s epoch %d/%d train_loss=%.4f val_loss=%.4f val_iou=%.4f",
            fold_name,
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            val_iou,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "bands": list(cfg.bands),
                    "encoder_name": cfg.encoder_name,
                    "tile_size": cfg.tile_size,
                    "fold": fold_name,
                },
                best_path,
            )

        if early_stopping.step(val_iou):
            logger.info("Early stopping triggered for %s at epoch %d", fold_name, epoch)
            break

    return {
        "best_path": str(best_path),
        "best_val_iou": best_iou,
        "history": history,
    }


def run_spatial_cv(records: List[Dict], cfg: TrainConfig):
    try:
        from sklearn.model_selection import GroupKFold
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for spatial CV") from exc

    groups = _group_records(records, "zone_id")
    unique_groups = len(set(groups))
    n_splits = min(cfg.cv_folds, unique_groups)
    if n_splits < 2:
        logger.info("Skipping CV: not enough unique spatial groups")
        return []

    splitter = GroupKFold(n_splits=n_splits)
    results = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(records, groups=groups), start=1):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        logger.info("CV fold %d/%d train=%d val=%d", fold_index, n_splits, len(train_records), len(val_records))
        results.append(_train_with_early_stopping(train_records, val_records, cfg, fold_name=f"fold_{fold_index}"))
    return results


def final_train(train_records: List[Dict], val_records: List[Dict], cfg: TrainConfig):
    return _train_with_early_stopping(train_records, val_records, cfg, fold_name="final")


def main() -> None:
    cfg = parse_args()
    train_records = load_split_records(cfg.train_split)
    val_records = load_split_records(cfg.val_split)

    logger.info("Loaded train=%d val=%d records", len(train_records), len(val_records))
    cv_results = run_spatial_cv(train_records, cfg)
    final_result = final_train(train_records, val_records, cfg)

    summary = {
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "cv_results": cv_results,
        "final_result": final_result,
        "config": {
            "bands": list(cfg.bands),
            "tile_size": cfg.tile_size,
            "tiles_per_polygon": cfg.tiles_per_polygon,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "patience": cfg.patience,
            "cv_folds": cfg.cv_folds,
            "encoder_name": cfg.encoder_name,
        },
    }

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved summary to %s", out_dir / "training_summary.json")


if __name__ == "__main__":
    main()
