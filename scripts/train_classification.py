from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
from joblib import dump
from rasterio.enums import Resampling
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.segmentation_dataset import load_split_records, record_to_polygon_geojson_in_crs, resolve_image_path


BAND_NAMES = ("blue", "green", "red", "nir", "swir1")


@dataclass
class DatasetBundle:
    records: List[Dict]
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    feature_names: List[str]


def align_dataset_features(bundle: DatasetBundle, target_feature_names: Sequence[str]) -> DatasetBundle:
    index_by_name = {name: idx for idx, name in enumerate(bundle.feature_names)}
    aligned = np.zeros((bundle.features.shape[0], len(target_feature_names)), dtype=np.float32)
    for target_idx, name in enumerate(target_feature_names):
        source_idx = index_by_name.get(name)
        if source_idx is not None:
            aligned[:, target_idx] = bundle.features[:, source_idx]
    return DatasetBundle(
        records=bundle.records,
        features=aligned,
        labels=bundle.labels,
        groups=bundle.groups,
        feature_names=list(target_feature_names),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train olive cultivation classification from parcel TIFF composites.")
    parser.add_argument("--train", default="data_splits/ezzayra_oliviers/train.json")
    parser.add_argument("--val", default="data_splits/ezzayra_oliviers/val.json")
    parser.add_argument("--test", default="data_splits/ezzayra_oliviers/test.json")
    parser.add_argument(
        "--image-root",
        default=r"C:\Users\SP\Desktop\Hack The Harvest\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\sentinel2_l2a\2025_05_06",
        help="Root folder containing train/val/test per-parcel TIFF composites.",
    )
    parser.add_argument(
        "--temporal-root",
        default=r"C:\Users\SP\Desktop\Hack The Harvest\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\sentinel2_l2a\from_wiem_branch",
        help="Optional root containing per-date Sentinel-2 parcel folders for seasonal features.",
    )
    parser.add_argument("--image-pattern", default="{cultivation_system}__{id}.tif")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="models/classification")
    parser.add_argument("--feature-csv", default="models/classification/features.csv")
    parser.add_argument("--summary-json", default="models/classification/training_summary.json")
    parser.add_argument("--model-path", default="models/classification/random_forest.joblib")
    return parser.parse_args()


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)


def _quantize_uint8(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def _glcm_features(values_2d: np.ndarray, mask_2d: np.ndarray, prefix: str) -> Dict[str, float]:
    valid_values = values_2d[mask_2d > 0]
    if valid_values.size < 16:
        return {
            f"{prefix}_glcm_contrast": 0.0,
            f"{prefix}_glcm_dissimilarity": 0.0,
            f"{prefix}_glcm_homogeneity": 0.0,
            f"{prefix}_glcm_energy": 0.0,
            f"{prefix}_glcm_correlation": 0.0,
        }

    fill_value = float(np.median(valid_values))
    filled = np.where(mask_2d > 0, values_2d, fill_value)
    quantized = _quantize_uint8(filled)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0.0, np.pi / 4.0, np.pi / 2.0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    return {
        f"{prefix}_glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        f"{prefix}_glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        f"{prefix}_glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        f"{prefix}_glcm_energy": float(graycoprops(glcm, "energy").mean()),
        f"{prefix}_glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def _summary_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p90": 0.0,
        }

    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def _circular_diff_radians(a: float, b: float) -> float:
    diff = abs(a - b) % np.pi
    return min(diff, np.pi - diff)


def _extract_mask_shape_spacing_features(mask: np.ndarray, pixel_size_x: float, pixel_size_y: float, prefix: str) -> Dict[str, float]:
    pixel_area_m2 = abs(pixel_size_x * pixel_size_y)
    mask_bool = mask.astype(bool)
    labeled = label(mask_bool, connectivity=2)
    props = regionprops(labeled)

    features = {
        f"{prefix}_object_count": 0.0,
        f"{prefix}_coverage_fraction": float(mask_bool.mean()) if mask_bool.size else 0.0,
        f"{prefix}_object_density_per_ha": 0.0,
        f"{prefix}_area_mean_m2": 0.0,
        f"{prefix}_area_std_m2": 0.0,
        f"{prefix}_area_max_m2": 0.0,
        f"{prefix}_perimeter_mean_m": 0.0,
        f"{prefix}_compactness_mean": 0.0,
        f"{prefix}_eccentricity_mean": 0.0,
        f"{prefix}_major_axis_mean_m": 0.0,
        f"{prefix}_minor_axis_mean_m": 0.0,
        f"{prefix}_bbox_fill_mean": 0.0,
        f"{prefix}_nearest_neighbor_mean_m": 0.0,
        f"{prefix}_nearest_neighbor_std_m": 0.0,
        f"{prefix}_alignment_strength": 0.0,
        f"{prefix}_orientation_coherence": 0.0,
    }

    if not props:
        return features

    areas_m2 = np.array([prop.area * pixel_area_m2 for prop in props], dtype=np.float32)
    perimeters_m = np.array(
        [prop.perimeter * ((abs(pixel_size_x) + abs(pixel_size_y)) / 2.0) for prop in props],
        dtype=np.float32,
    )
    compactness = np.array(
        [
            float((4.0 * np.pi * area) / (perimeter**2)) if perimeter > 0 else 0.0
            for area, perimeter in zip(areas_m2, perimeters_m)
        ],
        dtype=np.float32,
    )
    eccentricities = np.array([float(prop.eccentricity) for prop in props], dtype=np.float32)
    major_axes_m = np.array([float(prop.axis_major_length * abs(pixel_size_x)) for prop in props], dtype=np.float32)
    minor_axes_m = np.array([float(prop.axis_minor_length * abs(pixel_size_x)) for prop in props], dtype=np.float32)
    bbox_fill = np.array(
        [
            float(prop.area / max(1.0, (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])))
            for prop in props
        ],
        dtype=np.float32,
    )

    centroids = np.array([prop.centroid for prop in props], dtype=np.float32)
    centroids_xy_m = np.stack(
        [
            centroids[:, 1] * abs(pixel_size_x),
            centroids[:, 0] * abs(pixel_size_y),
        ],
        axis=1,
    )

    if len(centroids_xy_m) >= 2:
        diff = centroids_xy_m[:, None, :] - centroids_xy_m[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        features[f"{prefix}_nearest_neighbor_mean_m"] = float(nearest.mean()) if nearest.size else 0.0
        features[f"{prefix}_nearest_neighbor_std_m"] = float(nearest.std()) if nearest.size else 0.0

        centered = centroids_xy_m - centroids_xy_m.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[0] > 0:
            features[f"{prefix}_alignment_strength"] = float(eigvals[0] / max(1e-6, eigvals.sum()))

    orientations = np.array([float(prop.orientation) for prop in props], dtype=np.float32)
    if orientations.size >= 2:
        weights = areas_m2 / max(1e-6, areas_m2.sum())
        weighted_mean_cos = float(np.sum(np.cos(2.0 * orientations) * weights))
        weighted_mean_sin = float(np.sum(np.sin(2.0 * orientations) * weights))
        features[f"{prefix}_orientation_coherence"] = float(
            np.sqrt(weighted_mean_cos**2 + weighted_mean_sin**2)
        )

    parcel_area_ha = (mask_bool.sum() * pixel_area_m2) / 10000.0
    features.update(
        {
            f"{prefix}_object_count": float(len(props)),
            f"{prefix}_object_density_per_ha": float(len(props) / max(parcel_area_ha, 1e-6)),
            f"{prefix}_area_mean_m2": float(areas_m2.mean()),
            f"{prefix}_area_std_m2": float(areas_m2.std()),
            f"{prefix}_area_max_m2": float(areas_m2.max()),
            f"{prefix}_perimeter_mean_m": float(perimeters_m.mean()),
            f"{prefix}_compactness_mean": float(compactness.mean()),
            f"{prefix}_eccentricity_mean": float(eccentricities.mean()),
            f"{prefix}_major_axis_mean_m": float(major_axes_m.mean()),
            f"{prefix}_minor_axis_mean_m": float(minor_axes_m.mean()),
            f"{prefix}_bbox_fill_mean": float(bbox_fill.mean()),
        }
    )
    return features


def _compute_fapar_proxy(ndvi: np.ndarray) -> np.ndarray:
    return np.clip(1.4 * ndvi - 0.1, 0.0, 1.0)


def _compute_lai_proxy(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    evi = 2.5 * _safe_divide(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0)
    return np.clip(3.618 * evi - 0.118, 0.0, 8.0)


def _compute_ndre(nir: np.ndarray, red_edge: np.ndarray | None, red: np.ndarray) -> Tuple[np.ndarray, bool]:
    if red_edge is not None:
        return _safe_divide(nir - red_edge, nir + red_edge), True
    return _safe_divide(nir - red, nir + red), False


def _find_temporal_parcel_dir(temporal_root: str, parcel_id: str) -> Path | None:
    root = Path(temporal_root)
    if not root.exists():
        return None

    matches = [path for path in root.rglob(parcel_id) if path.is_dir()]
    if not matches:
        return None
    return matches[0]


def _read_single_band(path: Path, out_shape=None, resampling=Resampling.nearest) -> np.ndarray:
    with rasterio.open(path) as src:
        if out_shape is None:
            return src.read(1)
        return src.read(1, out_shape=out_shape, resampling=resampling)


def _load_temporal_observations(parcel_dir: Path) -> List[Dict]:
    date_band_map: Dict[str, Dict[str, Path]] = {}
    dated_pattern = re.compile(r"^(\d{8})_(B02|B03|B04|B05|B06|B07|B08|B8A|B11|SCL)\.tif$", re.IGNORECASE)
    for tif_path in parcel_dir.glob("*.tif"):
        match = dated_pattern.match(tif_path.name)
        if not match:
            continue
        date_key, band_name = match.group(1), match.group(2).upper()
        date_band_map.setdefault(date_key, {})[band_name] = tif_path

    observations = []
    for date_key, band_paths in sorted(date_band_map.items()):
        required = {"B02", "B03", "B04", "B08", "B11"}
        if not required.issubset(band_paths):
            continue

        blue = _read_single_band(band_paths["B02"]).astype(np.float32) / 10000.0
        green = _read_single_band(band_paths["B03"]).astype(np.float32) / 10000.0
        red = _read_single_band(band_paths["B04"]).astype(np.float32) / 10000.0
        nir = _read_single_band(band_paths["B08"]).astype(np.float32) / 10000.0
        swir1 = _read_single_band(
            band_paths["B11"],
            out_shape=blue.shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32) / 10000.0

        if "SCL" in band_paths:
            scl = _read_single_band(
                band_paths["SCL"],
                out_shape=blue.shape,
                resampling=Resampling.nearest,
            )
            valid_mask = np.isin(scl, [4, 5])
        else:
            valid_mask = np.ones_like(blue, dtype=bool)

        red_edge = None
        for band_name in ["B8A", "B05", "B06", "B07"]:
            if band_name in band_paths:
                red_edge = _read_single_band(
                    band_paths[band_name],
                    out_shape=blue.shape,
                    resampling=Resampling.bilinear,
                ).astype(np.float32) / 10000.0
                break

        ndvi = _safe_divide(nir - red, nir + red)
        ndwi = _safe_divide(green - nir, green + nir)
        ndre, ndre_real = _compute_ndre(nir, red_edge, red)
        fapar = _compute_fapar_proxy(ndvi)
        lai = _compute_lai_proxy(blue, red, nir)

        observations.append(
            {
                "date": date_key,
                "blue": blue,
                "green": green,
                "red": red,
                "nir": nir,
                "swir1": swir1,
                "valid_mask": valid_mask,
                "ndvi": ndvi,
                "ndwi": ndwi,
                "ndre": ndre,
                "ndre_real": ndre_real,
                "fapar": fapar,
                "lai": lai,
            }
        )
    return observations


def _pick_high_season_observation(observations: List[Dict]) -> Dict | None:
    if not observations:
        return None

    def _score(obs: Dict) -> Tuple[float, str]:
        valid_mask = obs["valid_mask"]
        if not np.any(valid_mask):
            return (-9999.0, obs["date"])
        ndvi_mean = float(obs["ndvi"][valid_mask].mean())
        return (ndvi_mean, obs["date"])

    return max(observations, key=_score)


def _extract_temporal_features(record: Dict, temporal_root: str) -> Dict[str, float]:
    parcel_dir = _find_temporal_parcel_dir(temporal_root, record["id"])
    if parcel_dir is None:
        return {
            "seasonal_obs_count": 0.0,
            "ndvi_seasonal_amplitude": 0.0,
            "ndwi_seasonal_amplitude": 0.0,
            "ndre_seasonal_amplitude": 0.0,
            "fapar_seasonal_amplitude": 0.0,
            "lai_seasonal_amplitude": 0.0,
            "ndre_real_available": 0.0,
            "high_season_texture_available": 0.0,
        }

    observations = _load_temporal_observations(parcel_dir)
    if not observations:
        return {
            "seasonal_obs_count": 0.0,
            "ndvi_seasonal_amplitude": 0.0,
            "ndwi_seasonal_amplitude": 0.0,
            "ndre_seasonal_amplitude": 0.0,
            "fapar_seasonal_amplitude": 0.0,
            "lai_seasonal_amplitude": 0.0,
            "ndre_real_available": 0.0,
            "high_season_texture_available": 0.0,
        }

    ndvi_means = []
    ndwi_means = []
    ndre_means = []
    fapar_means = []
    lai_means = []
    real_ndre_count = 0

    for obs in observations:
        if obs["ndre_real"]:
            real_ndre_count += 1

        valid_pixels = obs["valid_mask"]
        if not np.any(valid_pixels):
            valid_pixels = np.ones_like(valid_pixels, dtype=bool)

        ndvi_means.append(float(obs["ndvi"][valid_pixels].mean()))
        ndwi_means.append(float(obs["ndwi"][valid_pixels].mean()))
        ndre_means.append(float(obs["ndre"][valid_pixels].mean()))
        fapar_means.append(float(obs["fapar"][valid_pixels].mean()))
        lai_means.append(float(obs["lai"][valid_pixels].mean()))

    def _amplitude(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(max(values) - min(values))

    features: Dict[str, float] = {
        "seasonal_obs_count": float(len(ndvi_means)),
        "ndvi_seasonal_amplitude": _amplitude(ndvi_means),
        "ndwi_seasonal_amplitude": _amplitude(ndwi_means),
        "ndre_seasonal_amplitude": _amplitude(ndre_means),
        "fapar_seasonal_amplitude": _amplitude(fapar_means),
        "lai_seasonal_amplitude": _amplitude(lai_means),
        "ndre_real_available": float(real_ndre_count > 0),
        "high_season_texture_available": 0.0,
    }

    for prefix, values in [
        ("seasonal_ndvi_mean", ndvi_means),
        ("seasonal_ndwi_mean", ndwi_means),
        ("seasonal_ndre_mean", ndre_means),
        ("seasonal_fapar_mean", fapar_means),
        ("seasonal_lai_mean", lai_means),
    ]:
        array = np.asarray(values, dtype=np.float32)
        features.update(_summary_stats(array, prefix))

    high_season = _pick_high_season_observation(observations)
    if high_season is not None:
        features["high_season_texture_available"] = 1.0
        features.update(_glcm_features(high_season["nir"], high_season["valid_mask"], "high_season_nir"))
        features.update(_glcm_features(high_season["ndvi"], high_season["valid_mask"], "high_season_ndvi"))

    return features


def extract_record_features(record: Dict, image_root: str, image_pattern: str, temporal_root: str | None = None) -> Dict[str, float]:
    image_path = resolve_image_path(record, image_root, image_pattern)
    with rasterio.open(image_path) as src:
        if src.count < 5:
            raise ValueError(f"Raster {image_path} has {src.count} bands, expected at least 5.")

        image = src.read(indexes=[1, 2, 3, 4, 5]).astype(np.float32) / 10000.0
        polygon_geojson = record_to_polygon_geojson_in_crs(record, src.crs)
        pixel_size_x = src.transform.a
        pixel_size_y = src.transform.e
        mask = rasterize(
            [(polygon_geojson, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )

    valid_mask = mask > 0
    if not np.any(valid_mask):
        raise ValueError(f"Rasterized parcel mask is empty for {record.get('id')}")

    blue, green, red, nir, swir1 = image
    ndvi = _safe_divide(nir - red, nir + red)
    ndwi = _safe_divide(green - nir, green + nir)
    ndmi = _safe_divide(nir - swir1, nir + swir1)
    bsi = _safe_divide((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    ndre, _ = _compute_ndre(nir, None, red)
    fapar = _compute_fapar_proxy(ndvi)
    lai = _compute_lai_proxy(blue, red, nir)

    features: Dict[str, float] = {
        "parcel_area_ha": float(record.get("area_ha", 0.0) or 0.0),
        "mask_pixel_count": float(valid_mask.sum()),
        "canopy_fraction_ndvi_035": float((ndvi[valid_mask] > 0.35).mean()),
        "canopy_fraction_ndvi_05": float((ndvi[valid_mask] > 0.5).mean()),
    }

    for band_name, band_values in zip(BAND_NAMES, [blue, green, red, nir, swir1]):
        features.update(_summary_stats(band_values[valid_mask], band_name))

    for index_name, index_values in [
        ("ndvi", ndvi),
        ("ndwi", ndwi),
        ("ndre", ndre),
        ("ndmi", ndmi),
        ("fapar", fapar),
        ("lai", lai),
        ("bsi", bsi),
    ]:
        features.update(_summary_stats(index_values[valid_mask], index_name))

    features.update(_glcm_features(nir, valid_mask, "nir"))
    features.update(_glcm_features(ndvi, valid_mask, "ndvi"))
    canopy_mask = np.logical_and(valid_mask, ndvi > 0.35).astype(np.uint8)
    features.update(
        _extract_mask_shape_spacing_features(
            canopy_mask,
            pixel_size_x=float(pixel_size_x),
            pixel_size_y=float(pixel_size_y),
            prefix="canopy_mask",
        )
    )
    if temporal_root:
        features.update(_extract_temporal_features(record, temporal_root))
    return features


def build_dataset(records: Sequence[Dict], image_root: str, image_pattern: str, temporal_root: str | None = None) -> DatasetBundle:
    filtered_records: List[Dict] = []
    rows: List[Dict[str, float]] = []
    labels: List[str] = []
    groups: List[str] = []

    for record in records:
        if record.get("cultivation_system") not in {"extensif", "intensif"}:
            continue

        feature_row = extract_record_features(
            record,
            image_root=image_root,
            image_pattern=image_pattern,
            temporal_root=temporal_root,
        )
        filtered_records.append(record)
        rows.append(feature_row)
        labels.append(record["cultivation_system"])
        groups.append(record.get("zone_id") or record.get("governorate") or record.get("id"))

    if not rows:
        raise ValueError("No usable records found for classification training.")

    feature_names = sorted({key for row in rows for key in row.keys()})
    feature_matrix = np.array([[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float32)
    return DatasetBundle(
        records=filtered_records,
        features=feature_matrix,
        labels=np.array(labels),
        groups=np.array(groups),
        feature_names=feature_names,
    )


def make_classifier(args):
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=args.n_jobs,
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> Dict:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": list(labels),
        "classification_report": report,
    }


def run_group_cv(dataset: DatasetBundle, args, labels: Sequence[str]) -> List[Dict]:
    unique_groups = np.unique(dataset.groups)
    n_splits = min(args.cv_folds, len(unique_groups))
    if n_splits < 2:
        return []

    splitter = GroupKFold(n_splits=n_splits)
    results = []
    for fold_index, (train_idx, val_idx) in enumerate(
        splitter.split(dataset.features, dataset.labels, groups=dataset.groups),
        start=1,
    ):
        model = make_classifier(args)
        model.fit(dataset.features[train_idx], dataset.labels[train_idx])
        pred = model.predict(dataset.features[val_idx])
        metrics = evaluate_predictions(dataset.labels[val_idx], pred, labels=labels)
        metrics["fold"] = fold_index
        metrics["train_size"] = int(len(train_idx))
        metrics["val_size"] = int(len(val_idx))
        results.append(metrics)
    return results


def write_feature_csv(path: Path, datasets: Dict[str, DatasetBundle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = next(iter(datasets.values())).feature_names

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "record_id", "label", "group", *feature_names])

        for dataset_name, bundle in datasets.items():
            for record, feature_row, label, group in zip(bundle.records, bundle.features, bundle.labels, bundle.groups):
                writer.writerow([dataset_name, record.get("id"), label, group, *map(float, feature_row)])


def main():
    args = parse_args()

    train_records = load_split_records(args.train)
    val_records = load_split_records(args.val)
    test_records = load_split_records(args.test) if args.test else []

    train_dataset = build_dataset(
        train_records,
        image_root=args.image_root,
        image_pattern=args.image_pattern,
        temporal_root=args.temporal_root,
    )
    val_dataset = build_dataset(
        val_records,
        image_root=args.image_root,
        image_pattern=args.image_pattern,
        temporal_root=args.temporal_root,
    )
    datasets_for_csv = {"train": train_dataset, "val": val_dataset}

    labels = sorted(set(train_dataset.labels.tolist()) | set(val_dataset.labels.tolist()))
    if test_records:
        test_dataset = build_dataset(
            test_records,
            image_root=args.image_root,
            image_pattern=args.image_pattern,
            temporal_root=args.temporal_root,
        )
        datasets_for_csv["test"] = test_dataset
        labels = sorted(set(labels) | set(test_dataset.labels.tolist()))
    else:
        test_dataset = None

    all_feature_names = sorted(set(train_dataset.feature_names) | set(val_dataset.feature_names))
    if test_dataset is not None:
        all_feature_names = sorted(set(all_feature_names) | set(test_dataset.feature_names))

    train_dataset = align_dataset_features(train_dataset, all_feature_names)
    val_dataset = align_dataset_features(val_dataset, all_feature_names)
    datasets_for_csv = {"train": train_dataset, "val": val_dataset}
    if test_dataset is not None:
        test_dataset = align_dataset_features(test_dataset, all_feature_names)
        datasets_for_csv["test"] = test_dataset

    cv_results = run_group_cv(train_dataset, args, labels)

    val_model = make_classifier(args)
    val_model.fit(train_dataset.features, train_dataset.labels)
    val_pred = val_model.predict(val_dataset.features)
    holdout_val_metrics = evaluate_predictions(val_dataset.labels, val_pred, labels=labels)

    final_features = np.vstack([train_dataset.features, val_dataset.features])
    final_labels = np.concatenate([train_dataset.labels, val_dataset.labels])
    final_model = make_classifier(args)
    final_model.fit(final_features, final_labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": final_model,
            "feature_names": train_dataset.feature_names,
            "labels": labels,
            "image_pattern": args.image_pattern,
            "image_root": args.image_root,
        },
        args.model_path,
    )

    test_metrics = None
    if test_dataset is not None:
        test_pred = final_model.predict(test_dataset.features)
        test_metrics = evaluate_predictions(test_dataset.labels, test_pred, labels=labels)

    feature_importances = [
        {"feature": name, "importance": float(score)}
        for name, score in sorted(
            zip(train_dataset.feature_names, final_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    summary = {
        "train_records": len(train_dataset.labels),
        "val_records": len(val_dataset.labels),
        "test_records": int(len(test_dataset.labels)) if test_dataset is not None else 0,
        "labels": labels,
        "config": {
            "image_root": args.image_root,
            "temporal_root": args.temporal_root,
            "image_pattern": args.image_pattern,
            "cv_folds": args.cv_folds,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "cv_results": cv_results,
        "holdout_val_metrics": holdout_val_metrics,
        "test_metrics": test_metrics,
        "feature_importances": feature_importances,
        "model_path": args.model_path,
        "feature_csv": args.feature_csv,
    }

    write_feature_csv(Path(args.feature_csv), datasets_for_csv)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
