from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
from joblib import dump
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from skimage.feature import graycomatrix, graycoprops

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


def extract_record_features(record: Dict, image_root: str, image_pattern: str) -> Dict[str, float]:
    image_path = resolve_image_path(record, image_root, image_pattern)
    with rasterio.open(image_path) as src:
        if src.count < 5:
            raise ValueError(f"Raster {image_path} has {src.count} bands, expected at least 5.")

        image = src.read(indexes=[1, 2, 3, 4, 5]).astype(np.float32) / 10000.0
        polygon_geojson = record_to_polygon_geojson_in_crs(record, src.crs)
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
        ("ndmi", ndmi),
        ("bsi", bsi),
    ]:
        features.update(_summary_stats(index_values[valid_mask], index_name))

    features.update(_glcm_features(nir, valid_mask, "nir"))
    features.update(_glcm_features(ndvi, valid_mask, "ndvi"))
    return features


def build_dataset(records: Sequence[Dict], image_root: str, image_pattern: str) -> DatasetBundle:
    filtered_records: List[Dict] = []
    rows: List[Dict[str, float]] = []
    labels: List[str] = []
    groups: List[str] = []

    for record in records:
        if record.get("cultivation_system") not in {"extensif", "intensif"}:
            continue

        feature_row = extract_record_features(record, image_root=image_root, image_pattern=image_pattern)
        filtered_records.append(record)
        rows.append(feature_row)
        labels.append(record["cultivation_system"])
        groups.append(record.get("zone_id") or record.get("governorate") or record.get("id"))

    if not rows:
        raise ValueError("No usable records found for classification training.")

    feature_names = sorted(rows[0].keys())
    feature_matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
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

    train_dataset = build_dataset(train_records, image_root=args.image_root, image_pattern=args.image_pattern)
    val_dataset = build_dataset(val_records, image_root=args.image_root, image_pattern=args.image_pattern)
    datasets_for_csv = {"train": train_dataset, "val": val_dataset}

    labels = sorted(set(train_dataset.labels.tolist()) | set(val_dataset.labels.tolist()))
    if test_records:
        test_dataset = build_dataset(test_records, image_root=args.image_root, image_pattern=args.image_pattern)
        datasets_for_csv["test"] = test_dataset
        labels = sorted(set(labels) | set(test_dataset.labels.tolist()))
    else:
        test_dataset = None

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
