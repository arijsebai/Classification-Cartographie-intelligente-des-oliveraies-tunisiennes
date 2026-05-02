import argparse
import json

from train_unet_smp import ParcelSegmentationDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Check generated segmentation masks.")
    parser.add_argument("--sentinel-root", default="sentinel2_l2a/2025_05_06")
    parser.add_argument("--geojson-root", default="data_splits/ezzayra_oliviers_geojson")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-mode", choices=["auto", "rasterize", "valid"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ParcelSegmentationDataset(
        args.sentinel_root,
        args.geojson_root,
        args.split,
        args.image_size,
        mask_mode=args.mask_mode,
    )

    rows = []
    for item in dataset:
        rows.append(
            {
                "id": item["id"],
                "path": item["path"],
                "mask_pixels": float(item["mask"].sum().item()),
                "valid_pixels": float(item["valid"].sum().item()),
            }
        )

    summary = {
        "split": args.split,
        "mask_mode": args.mask_mode,
        "count": len(rows),
        "empty_masks": sum(1 for row in rows if row["mask_pixels"] == 0),
        "mean_mask_pixels": sum(row["mask_pixels"] for row in rows) / len(rows),
        "mean_valid_pixels": sum(row["valid_pixels"] for row in rows) / len(rows),
        "items": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
