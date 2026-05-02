import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser(description="Check downloaded Sentinel-2 GeoTIFF files.")
    parser.add_argument("--sentinel-root", default="sentinel2_l2a/2025_05_06")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.sentinel_root)
    rows = []

    for tif_path in sorted(root.glob("*/*.tif")):
        with rasterio.open(tif_path) as src:
            sample = src.read()
            finite = np.isfinite(sample)
            positive = sample > 0
            valid = finite & positive
            rows.append(
                {
                    "path": str(tif_path),
                    "split": tif_path.parent.name,
                    "bands": src.count,
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
                    "nodata": src.nodata,
                    "valid_ratio": float(valid.sum() / valid.size),
                    "min": float(np.nanmin(sample)),
                    "max": float(np.nanmax(sample)),
                    "mean": float(np.nanmean(sample)),
                }
            )

    by_split = {}
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1

    summary = {
        "root": str(root),
        "count": len(rows),
        "by_split": by_split,
        "bad_band_count": [row for row in rows if row["bands"] != 5],
        "empty_or_invalid": [row for row in rows if row["valid_ratio"] == 0],
        "items": rows,
    }

    output_path = root / "geotiff_quality_report.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "root": summary["root"],
                "count": summary["count"],
                "by_split": summary["by_split"],
                "bad_band_count": len(summary["bad_band_count"]),
                "empty_or_invalid": len(summary["empty_or_invalid"]),
                "report": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
