from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.segmentation_core import binary_dice_loss, binary_iou_from_logits, build_model, to_tensor_batch
from training.segmentation_dataset import EzzayraSegmentationDataset, load_split_records

logger = logging.getLogger("evaluate_segmentation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation checkpoint on test split")
    parser.add_argument("--test-split", default="data_splits/ezzayra_oliviers/test.json")
    parser.add_argument(
        "--image-root",
        default=r"C:\Users\SP\Desktop\Hack The Harvest\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\sentinel2_l2a\2025_05_06",
        help="Root folder containing train/val/test per-parcel multi-band TIFF stacks.",
    )
    parser.add_argument("--image-pattern", default="{cultivation_system}__{id}.tif")
    parser.add_argument("--bands", default="2,3,4,8,11")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="models/segmentation/test_metrics.json")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _make_loader(dataset, batch_size: int, num_workers: int):
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    def collate_fn(batch):
        images = [item[0] for item in batch]
        masks = [item[1] for item in batch]
        records = [item[2] for item in batch]
        image_tensor, mask_tensor = to_tensor_batch(images, masks)
        return image_tensor, mask_tensor, records

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)


def main() -> None:
    args = parse_args()
    records = load_split_records(args.test_split)
    bands = tuple(int(x.strip()) for x in args.bands.split(",") if x.strip())
    dataset = EzzayraSegmentationDataset(
        records=records,
        image_root=args.image_root,
        image_pattern=args.image_pattern,
        bands=bands,
        tile_size=args.tile_size,
        tiles_per_polygon=1,
        augment=False,
        seed=99,
    )

    loader = _make_loader(dataset, args.batch_size, args.num_workers)

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for evaluation") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder_name = checkpoint.get("encoder_name", "resnet34")
    model = build_model(encoder_name=encoder_name, in_channels=len(bands), classes=1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    loss_bce = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _records in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = loss_bce(logits, masks) + binary_dice_loss(logits, masks)
            total_loss += float(loss.detach())
            total_iou += float(binary_iou_from_logits(logits, masks).detach())
            total_dice += float(1.0 - binary_dice_loss(logits, masks).detach())
            count += 1

    metrics = {
        "test_samples": len(records),
        "mean_loss": total_loss / max(1, count),
        "mean_iou": total_iou / max(1, count),
        "mean_dice": total_dice / max(1, count),
        "checkpoint": args.checkpoint,
        "bands": list(bands),
        "tile_size": args.tile_size,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved metrics to %s", output_path)
    logger.info("Test IoU %.4f Dice %.4f Loss %.4f", metrics["mean_iou"], metrics["mean_dice"], metrics["mean_loss"])


if __name__ == "__main__":
    main()
