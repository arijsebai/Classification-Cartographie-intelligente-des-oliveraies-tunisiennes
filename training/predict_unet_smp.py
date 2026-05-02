import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.transform import Affine
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_unet_smp import (
    BAND_COUNT,
    ParcelSegmentationDataset,
    compute_metrics,
    make_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on an EZZAYRA split.")
    parser.add_argument("--checkpoint", default="models/unet_resnet_sentinel2/best.pt")
    parser.add_argument("--sentinel-root", default="sentinel2_l2a/2025_05_06")
    parser.add_argument("--geojson-root", default="data_splits/ezzayra_oliviers_geojson")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="predictions/unet_resnet_sentinel2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-mode", choices=["auto", "rasterize", "valid"], default="auto")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional comma-separated thresholds to evaluate, for example 0.2,0.3,0.4,0.5,0.6.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save predicted mask GeoTIFFs. Metrics are always computed.",
    )
    return parser.parse_args()


def resized_transform(src, image_size):
    scale_x = src.width / image_size
    scale_y = src.height / image_size
    return src.transform * Affine.scale(scale_x, scale_y)


def save_prediction(mask, source_path, output_path, image_size):
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            width=image_size,
            height=image_size,
            transform=resized_transform(src, image_size),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    image_size = int(config["image_size"])

    model = make_model(
        encoder=config["encoder"],
        encoder_weights=None,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ParcelSegmentationDataset(args.sentinel_root, args.geojson_root, args.split, image_size, mask_mode=args.mask_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    thresholds = [args.threshold]
    if args.thresholds:
        thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]

    metrics_by_threshold = {
        threshold: {"dice": 0.0, "iou": 0.0, "target_pixels": 0.0, "pred_pixels": 0.0, "count": 0, "items": []}
        for threshold in thresholds
    }
    split_output = Path(args.output_dir) / args.split

    for batch in tqdm(loader, desc=f"predict {args.split}"):
        images = batch["image"].to(args.device)
        masks = batch["mask"].to(args.device)
        valid = batch["valid"].to(args.device)

        logits = model(images)
        batch_size = images.size(0)

        for threshold in thresholds:
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            metrics = compute_metrics(logits, masks, valid=valid, threshold=threshold)
            bucket = metrics_by_threshold[threshold]
            bucket["dice"] += metrics["dice"] * batch_size
            bucket["iou"] += metrics["iou"] * batch_size
            bucket["target_pixels"] += metrics["target_pixels"] * batch_size
            bucket["pred_pixels"] += metrics["pred_pixels"] * batch_size
            bucket["count"] += batch_size

            for i, parcel_id in enumerate(batch["id"]):
                source_path = batch["path"][i]
                output_path = split_output / f"{parcel_id}.tif"
                if args.save_masks and threshold == args.threshold:
                    pred_mask = preds[i, 0].cpu().numpy()
                    save_prediction(pred_mask, source_path, output_path, image_size)

                bucket["items"].append(
                    {
                        "id": parcel_id,
                        "source_path": source_path,
                        "prediction_path": str(output_path) if args.save_masks and threshold == args.threshold else None,
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "target_pixels": metrics["target_pixels"],
                        "pred_pixels": metrics["pred_pixels"],
                    }
                )

    threshold_summaries = []
    for threshold, bucket in metrics_by_threshold.items():
        threshold_summaries.append(
            {
                "threshold": threshold,
                "dice": bucket["dice"] / bucket["count"],
                "iou": bucket["iou"] / bucket["count"],
                "target_pixels": bucket["target_pixels"] / bucket["count"],
                "pred_pixels": bucket["pred_pixels"] / bucket["count"],
                "count": bucket["count"],
                "items": bucket["items"],
            }
        )

    threshold_summaries = sorted(threshold_summaries, key=lambda row: row["iou"], reverse=True)
    best = threshold_summaries[0]
    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "requested_threshold": args.threshold,
        "best_threshold": best["threshold"],
        "best_dice": best["dice"],
        "best_iou": best["iou"],
        "count": best["count"],
        "thresholds": threshold_summaries,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / f"{args.split}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "split": summary["split"],
                "best_threshold": summary["best_threshold"],
                "best_dice": summary["best_dice"],
                "best_iou": summary["best_iou"],
                "target_pixels": best["target_pixels"],
                "pred_pixels": best["pred_pixels"],
                "count": summary["count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
