from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.segmentation_core import build_model
from training.segmentation_dataset import (
    load_split_records,
    record_to_polygon_geojson_in_crs,
    resolve_image_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual segmentation predictions for sanity checking.")
    parser.add_argument("--test-split", default="data_splits/ezzayra_oliviers/test.json")
    parser.add_argument(
        "--image-root",
        default=r"C:\Users\SP\Desktop\Hack The Harvest\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\sentinel2_l2a\2025_05_06",
    )
    parser.add_argument("--image-pattern", default="{cultivation_system}__{id}.tif")
    parser.add_argument("--bands", default="2,3,4,8,11")
    parser.add_argument("--checkpoint", default="models/segmentation_wiem_import/final_best.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="models/segmentation_wiem_import/visual_test_predictions")
    return parser.parse_args()


def _resolve_band_indexes(src, bands):
    if src.count == len(bands):
        return list(range(1, src.count + 1))
    if max(bands) <= src.count:
        return list(bands)
    raise ValueError(f"Raster has {src.count} bands, cannot map requested bands {bands}.")


def _stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for idx in range(rgb.shape[2]):
        channel = rgb[:, :, idx]
        valid = channel[channel > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        hi = max(hi, lo + 1)
        stretched = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
        out[:, :, idx] = np.round(stretched * 255.0).astype(np.uint8)
    return out


def _overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    mask_bool = mask > 0
    if np.any(mask_bool):
        out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = pred > 0
    target_bool = target > 0
    inter = np.logical_and(pred_bool, target_bool).sum()
    union = np.logical_or(pred_bool, target_bool).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = pred > 0
    target_bool = target > 0
    inter = np.logical_and(pred_bool, target_bool).sum()
    denom = pred_bool.sum() + target_bool.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / denom)


def _make_panel(record_id: str, rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, iou: float, dice: float) -> Image.Image:
    base = Image.fromarray(rgb)
    gt = Image.fromarray(_overlay_mask(rgb, gt_mask, color=(60, 220, 60)))
    pred = Image.fromarray(_overlay_mask(rgb, pred_mask, color=(230, 50, 50)))

    diff = rgb.copy()
    false_pos = np.logical_and(pred_mask > 0, gt_mask == 0)
    false_neg = np.logical_and(gt_mask > 0, pred_mask == 0)
    true_pos = np.logical_and(pred_mask > 0, gt_mask > 0)
    diff[true_pos] = np.array([255, 210, 0], dtype=np.uint8)
    diff[false_pos] = np.array([255, 0, 0], dtype=np.uint8)
    diff[false_neg] = np.array([0, 220, 255], dtype=np.uint8)
    diff_img = Image.fromarray(diff)

    width, height = base.size
    panel = Image.new("RGB", (width * 4, height + 44), color=(18, 18, 18))
    panel.paste(base, (0, 44))
    panel.paste(gt, (width, 44))
    panel.paste(pred, (width * 2, 44))
    panel.paste(diff_img, (width * 3, 44))

    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{record_id} | IoU={iou:.4f} | Dice={dice:.4f}", fill=(255, 255, 255))
    draw.text((8, 24), "RGB | GT overlay | Pred overlay | Diff (yellow TP, red FP, cyan FN)", fill=(190, 190, 190))
    return panel


def _predict_with_padding(model, image_chw: np.ndarray, device, threshold: float) -> np.ndarray:
    import torch

    _, height, width = image_chw.shape
    pad_h = (4 - (height % 4)) % 4
    pad_w = (4 - (width % 4)) % 4
    padded = np.pad(image_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    tensor = torch.from_numpy(padded[None, :, :, :]).contiguous().float().to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probs = probs[:height, :width]
    return (probs >= threshold).astype(np.uint8)


def main():
    args = parse_args()

    import torch

    records = load_split_records(args.test_split)
    bands = tuple(int(x.strip()) for x in args.bands.split(",") if x.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        encoder_name=checkpoint.get("encoder_name", "resnet34"),
        in_channels=len(bands),
        classes=1,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    summary_rows = []

    with torch.no_grad():
        for record in records:
            image_path = resolve_image_path(record, args.image_root, args.image_pattern)
            with rasterio.open(image_path) as src:
                band_indexes = _resolve_band_indexes(src, bands)
                image = src.read(indexes=band_indexes).astype(np.float32) / 10000.0
                polygon_geojson = record_to_polygon_geojson_in_crs(record, src.crs)
                gt_mask = rasterize(
                    [(polygon_geojson, 1)],
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    dtype="uint8",
                )

            pred_mask = _predict_with_padding(model, image, device, args.threshold)

            rgb = np.moveaxis(image[:3], 0, -1)
            rgb = rgb[:, :, [2, 1, 0]]
            rgb = _stretch_rgb(rgb)

            iou = _iou(pred_mask, gt_mask)
            dice = _dice(pred_mask, gt_mask)
            panel = _make_panel(record["id"], rgb, gt_mask, pred_mask, iou, dice)
            out_path = output_dir / f"{record['id']}.png"
            panel.save(out_path)

            summary_rows.append(
                {
                    "id": record["id"],
                    "split": record.get("split"),
                    "cultivation_system": record.get("cultivation_system"),
                    "image_path": str(image_path),
                    "png_path": str(out_path),
                    "iou": iou,
                    "dice": dice,
                    "pred_positive_pixels": int(pred_mask.sum()),
                    "gt_positive_pixels": int(gt_mask.sum()),
                }
            )
            print(json.dumps(summary_rows[-1], ensure_ascii=False))

    summary = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "count": len(summary_rows),
        "mean_iou": float(np.mean([row["iou"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_dice": float(np.mean([row["dice"] for row in summary_rows])) if summary_rows else 0.0,
        "items": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
