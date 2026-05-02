import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops


BANDS = {
    "B02": 0,
    "B03": 1,
    "B04": 2,
    "B08": 3,
    "B11": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract parcel-level spectral features from Sentinel-2 GeoTIFFs.")
    parser.add_argument("--sentinel-root", default="sentinel2_l2a/2025_05_06")
    parser.add_argument("--splits-root", default="data_splits/ezzayra_oliviers")
    parser.add_argument("--output", default="features/parcel_sentinel2_features.csv")
    return parser.parse_args()


def load_labels(splits_root):
    labels = {}
    for split in ("train", "val", "test"):
        path = Path(splits_root) / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for parcel in data["parcels"]:
            labels[parcel["id"]] = {
                "split": split,
                "cultivation_system": parcel.get("cultivation_system"),
                "governorate": parcel.get("governorate"),
                "zone_id": parcel.get("zone_id"),
                "area_ha": parcel.get("area_ha"),
                **geometry_features(parcel),
            }
    return labels


def distance_deg(a, b):
    return math.hypot(float(a["lng"]) - float(b["lng"]), float(a["lat"]) - float(b["lat"]))


def shoelace_area_deg2(points):
    area = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        area += float(point["lng"]) * float(nxt["lat"]) - float(nxt["lng"]) * float(point["lat"])
    return abs(area) / 2.0


def geometry_features(parcel):
    points = parcel.get("coordinates") or []
    if len(points) < 3:
        return {
            "vertex_count": 0,
            "bbox_width_deg": np.nan,
            "bbox_height_deg": np.nan,
            "bbox_aspect_ratio": np.nan,
            "perimeter_deg": np.nan,
            "polygon_area_deg2": np.nan,
            "compactness_deg": np.nan,
        }

    lats = [float(point["lat"]) for point in points]
    lngs = [float(point["lng"]) for point in points]
    width = max(lngs) - min(lngs)
    height = max(lats) - min(lats)
    perimeter = sum(distance_deg(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))
    area_deg2 = shoelace_area_deg2(points)
    compactness = (4.0 * math.pi * area_deg2 / (perimeter ** 2)) if perimeter > 0 else np.nan

    return {
        "vertex_count": len(points),
        "bbox_width_deg": width,
        "bbox_height_deg": height,
        "bbox_aspect_ratio": width / height if height > 0 else np.nan,
        "perimeter_deg": perimeter,
        "polygon_area_deg2": area_deg2,
        "compactness_deg": compactness,
    }


def parcel_id_from_tif(path):
    stem = path.stem
    parts = stem.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected GeoTIFF file name: {path.name}")
    return parts[1]


def safe_divide(numerator, denominator):
    return numerator / np.where(np.abs(denominator) < 1e-6, np.nan, denominator)


def valid_pixels(data):
    finite = np.all(np.isfinite(data), axis=0)
    positive = np.any(data > 0, axis=0)
    return finite & positive


def summarize(values, prefix):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def quantize_for_glcm(values, mask, levels=16):
    valid_values = values[mask & np.isfinite(values)]
    quantized = np.zeros(values.shape, dtype=np.uint8)

    if valid_values.size == 0:
        return quantized, False

    lo, hi = np.percentile(valid_values, [2, 98])
    if np.isclose(lo, hi):
        return quantized, False

    scaled = np.clip((values - lo) / (hi - lo), 0, 1)
    quantized = np.floor(scaled * (levels - 1)).astype(np.uint8)
    quantized[~mask] = 0
    return quantized, True


def glcm_texture_features(values, mask, prefix, levels=16):
    quantized, ok = quantize_for_glcm(values, mask, levels=levels)
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    if not ok:
        return {f"{prefix}_glcm_{prop}": np.nan for prop in props}

    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    return {
        f"{prefix}_glcm_{prop}": float(graycoprops(glcm, prop).mean())
        for prop in props
    }


def extract_features(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        if data.shape[0] < 5:
            raise ValueError(f"{tif_path} has {data.shape[0]} bands, expected at least 5")

    mask = valid_pixels(data)
    b02 = data[BANDS["B02"]]
    b03 = data[BANDS["B03"]]
    b04 = data[BANDS["B04"]]
    b08 = data[BANDS["B08"]]
    b11 = data[BANDS["B11"]]

    ndvi = safe_divide(b08 - b04, b08 + b04)
    ndwi = safe_divide(b03 - b08, b03 + b08)

    # Sentinel-2 NDRE ideally uses B05/B06/B07 red-edge bands. The current extraction
    # has no red-edge band, so this is a conservative red-edge proxy using red/NIR.
    ndre_proxy = safe_divide(b08 - b04, b08 + b04)

    # Lightweight proxies for canopy density. True FAPAR/LAI products should be added
    # later from a biophysical processor or Copernicus vegetation products.
    fapar_proxy = np.clip((ndvi - 0.2) / 0.6, 0, 1)
    lai_proxy = np.clip(3.618 * ndvi - 0.118, 0, 6)

    row = {}
    for prefix, index in (
        ("ndvi", ndvi),
        ("ndwi", ndwi),
        ("ndre_proxy", ndre_proxy),
        ("fapar_proxy", fapar_proxy),
        ("lai_proxy", lai_proxy),
    ):
        row.update(summarize(index[mask], prefix))

    # Texture captures regular planting patterns. At 10 m, hyper-intensive hedgerows
    # tend to be more regular than extensive dry orchards, especially in NIR/NDVI.
    for prefix, index in (
        ("b08", b08),
        ("b11", b11),
        ("ndvi", ndvi),
        ("ndwi", ndwi),
    ):
        row.update(glcm_texture_features(index, mask, prefix))

    row["ndvi_range"] = row["ndvi_max"] - row["ndvi_min"]
    row["ndvi_p90_p10_amplitude"] = row["ndvi_p90"] - row["ndvi_p10"]
    row["seasonal_ndvi_amplitude"] = np.nan
    row["seasonal_amplitude_available"] = False
    row["valid_pixel_count"] = int(mask.sum())
    row["valid_pixel_ratio"] = float(mask.sum() / mask.size)
    return row


def main():
    args = parse_args()
    labels = load_labels(args.splits_root)
    rows = []

    for tif_path in sorted(Path(args.sentinel_root).glob("*/*.tif")):
        parcel_id = parcel_id_from_tif(tif_path)
        label = labels.get(parcel_id)
        if label is None:
            raise KeyError(f"No label found for {parcel_id}")

        row = {
            "id": parcel_id,
            "image_path": str(tif_path),
            **label,
            **extract_features(tif_path),
        }
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "count": len(rows),
                "splits": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
