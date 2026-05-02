import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SPLITS = ("train", "val", "test")
BAND_COUNT = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net ResNet on EZZAYRA Sentinel-2 parcel segmentation.")
    parser.add_argument("--sentinel-root", default="sentinel2_l2a/2025_05_06")
    parser.add_argument("--geojson-root", default="data_splits/ezzayra_oliviers_geojson")
    parser.add_argument("--output-dir", default="models/unet_resnet_sentinel2")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=["auto", "rasterize", "valid"],
        default="auto",
        help="Mask source: rasterize GeoJSON, valid pixels from clipped GeoTIFF, or auto fallback.",
    )
    return parser.parse_args()


def load_features_by_id(geojson_root):
    features = {}
    for split in SPLITS:
        geojson_path = Path(geojson_root) / f"{split}.geojson"
        with geojson_path.open("r", encoding="utf-8") as file:
            collection = json.load(file)

        for feature in collection["features"]:
            parcel_id = feature["properties"]["id"]
            features[parcel_id] = feature

    return features


def parcel_id_from_tif(tif_path):
    stem = tif_path.stem
    parts = stem.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected GeoTIFF name: {tif_path.name}")
    return parts[1]


def build_items(sentinel_root, geojson_root, split):
    features = load_features_by_id(geojson_root)
    split_dir = Path(sentinel_root) / split
    tif_paths = sorted(split_dir.glob("*.tif"))
    items = []

    for tif_path in tif_paths:
        parcel_id = parcel_id_from_tif(tif_path)
        if parcel_id not in features:
            raise KeyError(f"Missing GeoJSON feature for {parcel_id}")
        items.append({"image_path": tif_path, "feature": features[parcel_id]})

    if not items:
        raise FileNotFoundError(f"No GeoTIFF files found in {split_dir}")

    return items


def percentile_normalize(image):
    image = image.astype(np.float32)
    normalized = np.zeros_like(image, dtype=np.float32)

    for band_index in range(image.shape[0]):
        band = image[band_index]
        valid = np.isfinite(band) & (band > 0)
        if valid.sum() == 0:
            continue

        lo, hi = np.percentile(band[valid], [2, 98])
        if math.isclose(float(lo), float(hi)):
            continue

        normalized[band_index] = np.clip((band - lo) / (hi - lo), 0, 1)

    return normalized


class ParcelSegmentationDataset(Dataset):
    def __init__(self, sentinel_root, geojson_root, split, image_size=256, mask_mode="auto"):
        self.items = build_items(sentinel_root, geojson_root, split)
        self.image_size = image_size
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image_path = item["image_path"]
        feature = item["feature"]

        with rasterio.open(image_path) as src:
            image = src.read(list(range(1, min(BAND_COUNT, src.count) + 1))).astype(np.float32)
            if image.shape[0] != BAND_COUNT:
                raise ValueError(f"{image_path} has {image.shape[0]} bands, expected {BAND_COUNT}")

            # Project GeoJSON geometry (WGS84) into the raster CRS before rasterizing.
            try:
                projected_geometry = transform_geom("EPSG:4326", src.crs, feature["geometry"])
            except Exception:
                # Fallback: if projection fails, use original geometry (may produce empty mask)
                projected_geometry = feature["geometry"]

            rasterized_mask = rasterize(
                [(projected_geometry, 1)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype="uint8",
            )

            finite_mask = np.all(np.isfinite(image), axis=0)
            positive_mask = np.any(image > 0, axis=0)
            valid_mask = (finite_mask & positive_mask).astype(np.float32)
            if src.nodata is not None:
                valid_mask = (valid_mask * np.all(image != src.nodata, axis=0)).astype(np.float32)

            if self.mask_mode == "rasterize":
                mask = rasterized_mask
            elif self.mask_mode == "valid":
                mask = valid_mask.astype("uint8")
            else:
                mask = rasterized_mask if rasterized_mask.sum() > 0 else valid_mask.astype("uint8")

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = percentile_normalize(image)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        image_tensor = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(self.image_size, self.image_size), mode="nearest").squeeze(0)
        valid_tensor = F.interpolate(valid_tensor, size=(self.image_size, self.image_size), mode="nearest").squeeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "valid": valid_tensor,
            "id": feature["properties"]["id"],
            "path": str(image_path),
        }


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, valid=None):
        probs = torch.sigmoid(logits)
        if valid is not None:
            probs = probs * valid
            targets = targets * valid

        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss()

    def forward(self, logits, targets, valid=None):
        bce = self.bce(logits, targets)
        if valid is not None:
            bce = bce * valid
            bce = bce.sum() / valid.sum().clamp_min(1.0)
        else:
            bce = bce.mean()

        dice = self.dice(logits, targets, valid=valid)
        return self.bce_weight * bce + self.dice_weight * dice


def make_model(encoder, encoder_weights):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=BAND_COUNT,
        classes=1,
        activation=None,
    )


@torch.no_grad()
def compute_metrics(logits, targets, valid=None, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    if valid is not None:
        preds = preds * valid
        targets = targets * valid

    dims = (1, 2, 3)
    tp = torch.sum(preds * targets, dims)
    fp = torch.sum(preds * (1 - targets), dims)
    fn = torch.sum((1 - preds) * targets, dims)
    target_pixels = torch.sum(targets, dims)
    pred_pixels = torch.sum(preds, dims)

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = torch.where(dice_denominator > 0, (2 * tp) / dice_denominator.clamp_min(1e-7), torch.zeros_like(tp))
    iou = torch.where(iou_denominator > 0, tp / iou_denominator.clamp_min(1e-7), torch.zeros_like(tp))

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "target_pixels": target_pixels.mean().item(),
        "pred_pixels": pred_pixels.mean().item(),
    }


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "count": 0}

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        valid = batch["valid"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks, valid=valid)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics = compute_metrics(logits.detach(), masks, valid=valid)
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["dice"] += metrics["dice"] * batch_size
        totals["iou"] += metrics["iou"] * batch_size
        totals["count"] += batch_size

        progress.set_postfix(loss=loss.item(), dice=metrics["dice"], iou=metrics["iou"])

    return {key: value / totals["count"] for key, value in totals.items() if key != "count"}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ParcelSegmentationDataset(
        args.sentinel_root, args.geojson_root, "train", args.image_size, mask_mode=args.mask_mode
    )
    val_dataset = ParcelSegmentationDataset(
        args.sentinel_root, args.geojson_root, "val", args.image_size, mask_mode=args.mask_mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    encoder_weights = None if args.no_pretrained else args.encoder_weights
    model = make_model(args.encoder, encoder_weights).to(args.device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = -1.0
    history = []

    config = vars(args)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, args.device, train=False)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Best val Dice: {best_dice:.4f}")
    print(f"Saved checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
