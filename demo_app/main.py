import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_POLYGON_KM2 = 100.0
EARTH_RADIUS_M = 6_371_008.8
MODEL_PATH = ROOT / "models" / "cultivation_classifier" / "random_forest.joblib"
FEATURES_PATH = ROOT / "features" / "parcel_sentinel2_features.csv"
LOCAL_CLASSIFY_SCRIPT = ROOT / "scripts" / "classify_polygon_local.py"
SPATIAL_CV_PREDICTIONS_PATH = ROOT / "models" / "spatial_cv_classifier" / "spatial_cv_predictions.csv"
CULTIVATION_PREDICTIONS_PATH = ROOT / "models" / "cultivation_classifier" / "predictions.csv"
SUBMISSIONS_DIR = ROOT / "demo_app" / "submissions"
LATENCY_LOG = ROOT / "demo_app" / "latency.log"
SENTINEL_ROOT = ROOT / "sentinel2_l2a" / "2025_05_06"
INTENSIF_THRESHOLD = 0.2
OPENEO_ACTIONS = {"full-workflow"}

app = FastAPI(title="EZZAYRA Olive Mapping Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
try:
    import joblib
except ImportError:
    joblib = None

model = joblib.load(MODEL_PATH) if joblib is not None and MODEL_PATH.exists() else None
openeo_processes: dict[str, subprocess.Popen] = {}


class PolygonRequest(BaseModel):
    geometry: dict[str, Any] = Field(..., description="GeoJSON Polygon geometry")


def load_prediction_table() -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}

    if SPATIAL_CV_PREDICTIONS_PATH.exists():
        with SPATIAL_CV_PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                predictions[row["id"]] = {
                    "model_prediction": row.get("prediction"),
                    "model_correct_cv": row.get("correct"),
                    "model_source": "spatial_cv_random_forest",
                }

    if CULTIVATION_PREDICTIONS_PATH.exists():
        with CULTIVATION_PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                prediction = row.get("prediction_threshold") or row.get("prediction")
                current = predictions.get(row["id"], {})
                current.update(
                    {
                        "model_prediction": prediction,
                        "model_prediction_raw": row.get("prediction"),
                        "model_correct": row.get("correct_threshold") or row.get("correct"),
                        "prob_intensif": float(row["prob_intensif"]) if row.get("prob_intensif") else None,
                        "model_source": "thresholded_random_forest",
                    }
                )
                predictions[row["id"]] = current

    return predictions


def load_parcels_with_predictions() -> dict[str, Any]:
    path = ROOT / "data_splits" / "ezzayra_oliviers_geojson" / "all_splits.geojson"
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}

    data = json.loads(path.read_text(encoding="utf-8"))
    predictions = load_prediction_table()

    for feature in data.get("features", []):
        parcel_id = feature.get("properties", {}).get("id")
        if parcel_id in predictions:
            feature["properties"].update(predictions[parcel_id])

    return data


def ring_area_m2(ring: list[list[float]]) -> float:
    if len(ring) < 4:
        return 0.0

    total = 0.0
    for i in range(len(ring) - 1):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[i + 1]
        lon1 = math.radians(lon1)
        lat1 = math.radians(lat1)
        lon2 = math.radians(lon2)
        lat2 = math.radians(lat2)
        total += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))

    return abs(total * EARTH_RADIUS_M * EARTH_RADIUS_M / 2.0)


def polygon_area_m2(geometry: dict[str, Any]) -> float:
    if geometry.get("type") != "Polygon":
        raise HTTPException(status_code=400, detail="Only GeoJSON Polygon geometries are supported.")

    coordinates = geometry.get("coordinates")
    if not coordinates or not isinstance(coordinates, list):
        raise HTTPException(status_code=400, detail="Polygon has no coordinates.")

    outer = coordinates[0]
    holes = coordinates[1:]
    area = ring_area_m2(outer)
    for hole in holes:
        area -= ring_area_m2(hole)

    return max(area, 0.0)


def geometry_centroid(geometry: dict[str, Any]) -> dict[str, float]:
    ring = geometry["coordinates"][0]
    points = ring[:-1] if ring[0] == ring[-1] else ring
    lng = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    return {"lat": lat, "lng": lng}


def point_in_ring(point: list[float], ring: list[list[float]]) -> bool:
    lng, lat = point
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        lng_i, lat_i = ring[i]
        lng_j, lat_j = ring[j]
        intersects = (lat_i > lat) != (lat_j > lat) and (
            lng < (lng_j - lng_i) * (lat - lat_i) / ((lat_j - lat_i) or 1e-12) + lng_i
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_polygon(point: list[float], geometry: dict[str, Any]) -> bool:
    if geometry.get("type") != "Polygon":
        return False
    rings = geometry.get("coordinates") or []
    if not rings or not point_in_ring(point, rings[0]):
        return False
    return not any(point_in_ring(point, hole) for hole in rings[1:])


def find_matching_offline_parcel(geometry: dict[str, Any]) -> dict[str, Any] | None:
    
    
    drawn_centroid = geometry_centroid(geometry)
    drawn_point = [drawn_centroid["lng"], drawn_centroid["lat"]]
    parcels_data = load_parcels_with_predictions()

    best = None
    best_distance = float("inf")
    for feature in parcels_data.get("features", []):
        parcel_geometry = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        parcel_centroid = geometry_centroid(parcel_geometry)

        parcel_point = [parcel_centroid["lng"], parcel_centroid["lat"]]
        if point_in_polygon(drawn_point, parcel_geometry) or point_in_polygon(parcel_point, geometry):
            distance = math.hypot(parcel_point[0] - drawn_point[0], parcel_point[1] - drawn_point[1])
            if distance < best_distance:
                best = feature
                best_distance = distance

    return best


def distance_deg(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def shoelace_area_deg2(points: list[list[float]]) -> float:
    area = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        area += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return abs(area) / 2.0


def demo_geometry_features(geometry: dict[str, Any], area_ha: float) -> dict[str, float]:
    ring = geometry["coordinates"][0]
    points = ring[:-1] if ring[0] == ring[-1] else ring
    lngs = [float(point[0]) for point in points]
    lats = [float(point[1]) for point in points]
    width = max(lngs) - min(lngs)
    height = max(lats) - min(lats)
    perimeter = sum(distance_deg(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))
    area_deg2 = shoelace_area_deg2(points)
    compactness = (4.0 * math.pi * area_deg2 / (perimeter**2)) if perimeter > 0 else 0.0
    aspect_ratio = width / height if height > 0 else 0.0

    return {
        "area_ha": area_ha,
        "vertex_count": float(len(points)),
        "bbox_width_deg": width,
        "bbox_height_deg": height,
        "bbox_aspect_ratio": aspect_ratio,
        "perimeter_deg": perimeter,
        "polygon_area_deg2": area_deg2,
        "compactness_deg": compactness,
    }


def geometry_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    ring = geometry["coordinates"][0]
    lngs = [float(point[0]) for point in ring]
    lats = [float(point[1]) for point in ring]
    return min(lngs), min(lats), max(lngs), max(lats)


def bboxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def local_sentinel_intersects(geometry: dict[str, Any]) -> bool:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Missing local Sentinel dependencies in the demo environment: {error.name}",
        ) from error

    if not SENTINEL_ROOT.exists():
        return False

    geometry_bounds_wgs84 = geometry_bbox(geometry)
    for tif_path in SENTINEL_ROOT.glob("*/*.tif"):
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                continue
            geometry_bounds = transform_bounds(
                "EPSG:4326",
                src.crs,
                *geometry_bounds_wgs84,
                densify_pts=21,
            )
            image_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            if bboxes_intersect(geometry_bounds, image_bounds):
                return True
    return False


def predict_feature_row(feature_row: dict[str, float]) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Classification model is not available.")

    import pandas as pd

    feature_names = list(model.feature_names_in_)
    row = pd.DataFrame([{name: feature_row.get(name, math.nan) for name in feature_names}])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    intensif_probability = float(probabilities[classes.index("intensif")]) if "intensif" in classes else 0.0
    label = "intensif" if intensif_probability >= INTENSIF_THRESHOLD else "extensif"
    confidence = intensif_probability if label == "intensif" else 1.0 - intensif_probability

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "prob_intensif": round(intensif_probability, 3),
        "threshold_intensif": INTENSIF_THRESHOLD,
    }


def classify_local_sentinel_subprocess(geometry: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps({"geometry": geometry})
    completed = subprocess.run(
        [sys.executable, str(LOCAL_CLASSIFY_SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=120,
    )

    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(completed.stderr or completed.stdout or "Local Sentinel-2 classification process failed.").strip(),
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Local Sentinel-2 classification returned invalid JSON: {completed.stdout[:500]}",
        ) from error

    if "error" in result:
        error = result["error"]
        raise HTTPException(status_code=int(error.get("status_code", 500)), detail=error.get("detail"))

    return result["classification"]


def validate_demo_job_id(job_id: str) -> str:
    if not re.fullmatch(r"demo_[0-9]+_[0-9]+", job_id):
        raise HTTPException(status_code=400, detail="Invalid demo job id.")
    return job_id


def demo_submission_path(job_id: str) -> Path:
    validate_demo_job_id(job_id)
    return SUBMISSIONS_DIR / f"{job_id}.json"


def demo_geojson_path(job_id: str) -> Path:
    validate_demo_job_id(job_id)
    return SUBMISSIONS_DIR / f"{job_id}.geojson"


def demo_manifest_path(job_id: str) -> Path:
    validate_demo_job_id(job_id)
    return ROOT / f"openeo_jobs_manifest_{job_id}.json"


def demo_log_path(job_id: str, action: str) -> Path:
    validate_demo_job_id(job_id)
    if action not in OPENEO_ACTIONS:
        raise HTTPException(status_code=400, detail="Unknown openEO action.")
    return SUBMISSIONS_DIR / f"{job_id}_{action}.log"


def load_demo_submission(job_id: str) -> dict[str, Any]:
    path = demo_submission_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def export_demo_submission_geojson(job_id: str) -> Path:
    submission = load_demo_submission(job_id)
    feature = {
        "type": "Feature",
        "properties": {
            "job_id": submission["job_id"],
            "status": submission.get("status"),
            "created_at": submission.get("created_at"),
            "area_km2": submission.get("area_km2"),
            "area_ha": submission.get("area_ha"),
            "centroid_lat": (submission.get("centroid") or {}).get("lat"),
            "centroid_lng": (submission.get("centroid") or {}).get("lng"),
        },
        "geometry": submission["geometry"],
    }
    output = demo_geojson_path(job_id)
    output.write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2),
        encoding="utf-8",
    )
    return output


def start_openeo_process(job_id: str, action: str) -> dict[str, Any]:
    if action not in OPENEO_ACTIONS:
        raise HTTPException(status_code=400, detail="Unknown openEO action.")

    key = f"{job_id}:{action}"
    existing = openeo_processes.get(key)
    if existing is not None and existing.poll() is None:
        return openeo_status(job_id, action)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    geojson_path = export_demo_submission_geojson(job_id)
    manifest_path = demo_manifest_path(job_id)
    log_path = demo_log_path(job_id, action)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_demo_openeo_workflow.py"),
        "--job-id",
        job_id,
    ]

    log_file = log_path.open("w", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    openeo_processes[key] = process
    return openeo_status(job_id, action)


def openeo_status(job_id: str, action: str) -> dict[str, Any]:
    if action not in OPENEO_ACTIONS:
        raise HTTPException(status_code=400, detail="Unknown openEO action.")

    key = f"{job_id}:{action}"
    process = openeo_processes.get(key)
    return_code = process.poll() if process is not None else None
    status = "not_started"
    if process is not None:
        status = "running" if return_code is None else ("finished" if return_code == 0 else "error")

    log_path = demo_log_path(job_id, action)
    log_tail = ""
    if log_path.exists():
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]

    submission: dict[str, Any] = {}
    submission_path = demo_submission_path(job_id)
    if submission_path.exists():
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        if submission.get("analysis_status") == "completed":
            status = "finished" if status != "error" else status

    return {
        "job_id": job_id,
        "action": action,
        "status": status,
        "return_code": return_code,
        "geojson": str(demo_geojson_path(job_id).relative_to(ROOT)),
        "manifest": str(demo_manifest_path(job_id).relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "log_tail": log_tail,
        "analysis_status": submission.get("analysis_status"),
        "analysis_completed_at": submission.get("analysis_completed_at"),
        "analysis_result": submission.get("analysis_result"),
        "analysis_error": submission.get("analysis_error"),
        "result_file": submission.get("result_file"),
        "result_source": submission.get("result_source"),
    }


def extract_local_sentinel_features(geometry: dict[str, Any], area_ha: float) -> dict[str, Any]:
    try:
        import numpy as np
        import rasterio
        from rasterio.mask import mask as raster_mask
        from rasterio.warp import transform_bounds, transform_geom
        from skimage.feature import graycomatrix, graycoprops
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Missing local Sentinel dependencies in the demo environment: {error.name}",
        ) from error

    if not SENTINEL_ROOT.exists():
        raise HTTPException(status_code=404, detail="Local Sentinel-2 directory was not found.")

    band_indexes = {"B02": 0, "B03": 1, "B04": 2, "B08": 3, "B11": 4}
    geometry_bounds_wgs84 = geometry_bbox(geometry)

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
                f"{prefix}_mean": math.nan,
                f"{prefix}_min": math.nan,
                f"{prefix}_max": math.nan,
                f"{prefix}_std": math.nan,
                f"{prefix}_p10": math.nan,
                f"{prefix}_p50": math.nan,
                f"{prefix}_p90": math.nan,
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

    def quantize_for_glcm(values, valid_mask, levels=16):
        valid_values = values[valid_mask & np.isfinite(values)]
        quantized = np.zeros(values.shape, dtype=np.uint8)
        if valid_values.size == 0:
            return quantized, False
        lo, hi = np.percentile(valid_values, [2, 98])
        if np.isclose(lo, hi):
            return quantized, False
        scaled = np.nan_to_num(np.clip((values - lo) / (hi - lo), 0, 1), nan=0.0)
        quantized = np.floor(scaled * (levels - 1)).astype(np.uint8)
        quantized[~valid_mask] = 0
        return quantized, True

    def glcm_texture_features(values, valid_mask, prefix, levels=16):
        props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
        quantized, ok = quantize_for_glcm(values, valid_mask, levels=levels)
        if not ok:
            return {f"{prefix}_glcm_{prop}": math.nan for prop in props}

        glcm = graycomatrix(
            quantized,
            distances=[1, 2],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )
        return {f"{prefix}_glcm_{prop}": float(graycoprops(glcm, prop).mean()) for prop in props}

    candidates = []
    for tif_path in SENTINEL_ROOT.glob("*/*.tif"):
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                continue

            geometry_bounds = transform_bounds(
                "EPSG:4326",
                src.crs,
                *geometry_bounds_wgs84,
                densify_pts=21,
            )
            image_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            if not bboxes_intersect(geometry_bounds, image_bounds):
                continue

            projected_geometry = transform_geom("EPSG:4326", src.crs, geometry)
            data, _ = raster_mask(src, [projected_geometry], crop=True, filled=True, nodata=0)
            data = data.astype(np.float32)
            if data.shape[0] < 5:
                continue

        valid_mask = valid_pixels(data)
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            continue

        b03 = data[band_indexes["B03"]]
        b04 = data[band_indexes["B04"]]
        b08 = data[band_indexes["B08"]]
        b11 = data[band_indexes["B11"]]

        ndvi = safe_divide(b08 - b04, b08 + b04)
        ndwi = safe_divide(b03 - b08, b03 + b08)
        ndre_proxy = safe_divide(b08 - b04, b08 + b04)
        fapar_proxy = np.clip((ndvi - 0.2) / 0.6, 0, 1)
        lai_proxy = np.clip(3.618 * ndvi - 0.118, 0, 6)

        row = demo_geometry_features(geometry, area_ha)
        for prefix, index in (
            ("ndvi", ndvi),
            ("ndwi", ndwi),
            ("ndre_proxy", ndre_proxy),
            ("fapar_proxy", fapar_proxy),
            ("lai_proxy", lai_proxy),
        ):
            row.update(summarize(index[valid_mask], prefix))

        for prefix, index in (
            ("b08", b08),
            ("b11", b11),
            ("ndvi", ndvi),
            ("ndwi", ndwi),
        ):
            row.update(glcm_texture_features(index, valid_mask, prefix))

        row["ndvi_range"] = row["ndvi_max"] - row["ndvi_min"]
        row["ndvi_p90_p10_amplitude"] = row["ndvi_p90"] - row["ndvi_p10"]
        row["seasonal_ndvi_amplitude"] = math.nan
        row["seasonal_amplitude_available"] = False
        row["valid_pixel_count"] = valid_count
        row["valid_pixel_ratio"] = float(valid_count / valid_mask.size)
        candidates.append((valid_count, tif_path, row))

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No local Sentinel-2 GeoTIFF intersects this polygon. Submit it for openEO extraction first.",
        )

    valid_count, tif_path, row = max(candidates, key=lambda item: item[0])
    prediction = predict_feature_row(row)
    prediction.update(
        {
            "method": "local_sentinel2_random_forest",
            "image_path": str(tif_path.relative_to(ROOT)),
            "valid_pixel_count": valid_count,
            "valid_pixel_ratio": round(float(row["valid_pixel_ratio"]), 3),
            "note": "Prediction uses Sentinel-2 pixels from a local GeoTIFF and the trained Random Forest.",
        }
    )
    return prediction


def demo_classify(geometry: dict[str, Any], area_ha: float) -> dict[str, Any]:
    matching_parcel = find_matching_offline_parcel(geometry)
    if matching_parcel is not None:
        props = matching_parcel.get("properties", {})
        label = props.get("model_prediction") or props.get("cultivation_system")
        prob_intensif = props.get("prob_intensif")
        if prob_intensif is not None:
            confidence = prob_intensif if label == "intensif" else 1.0 - prob_intensif
        else:
            confidence = 0.9
        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "prob_intensif": round(float(prob_intensif), 3) if prob_intensif is not None else None,
            "method": "offline_random_forest_matched_existing_parcel",
            "matched_parcel_id": props.get("id"),
            "ground_truth": props.get("cultivation_system"),
            "note": "Drawn polygon overlaps an EZZAYRA parcel; returned the offline Sentinel-2 Random Forest prediction.",
        }

    features = demo_geometry_features(geometry, area_ha)

    elongated = features["bbox_aspect_ratio"] >= 2.5 or features["bbox_aspect_ratio"] <= 0.4
    label = "intensif" if area_ha <= 250 or (area_ha <= 700 and elongated) else "extensif"
    return {
        "label": label,
        "confidence": 0.62,
        "method": "demo_rule_geometry_precheck",
        "note": "Fast precheck only. The automatic Sentinel-2 route runs immediately after this validation.",
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "max_polygon_km2": MAX_POLYGON_KM2,
        "default_center": {"lat": 34.8, "lng": 9.8},
        "default_zoom": 7,
    }


@app.get("/api/parcels")
def parcels() -> dict[str, Any]:
    return load_parcels_with_predictions()


@app.get("/api/sentinel-footprints")
def sentinel_footprints() -> dict[str, Any]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Missing raster dependency in the demo environment: {error.name}",
        ) from error

    features = []
    if not SENTINEL_ROOT.exists():
        return {"type": "FeatureCollection", "features": features}

    for tif_path in sorted(SENTINEL_ROOT.glob("*/*.tif")):
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                continue
            left, bottom, right, top = transform_bounds(
                src.crs,
                "EPSG:4326",
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
                densify_pts=21,
            )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "image_path": str(tif_path.relative_to(ROOT)),
                    "split": tif_path.parent.name,
                    "name": tif_path.name,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [left, bottom],
                            [right, bottom],
                            [right, top],
                            [left, top],
                            [left, bottom],
                        ]
                    ],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/model-summary")
def model_summary() -> dict[str, Any]:
    metrics_path = ROOT / "models" / "spatial_cv_classifier" / "spatial_cv_metrics.json"
    if not metrics_path.exists():
        return {"available": False}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "model": metrics.get("model"),
        "group_column": metrics.get("group_column"),
        "classes": metrics.get("classes"),
        "feature_count": metrics.get("feature_count"),
        "spatial_cv_accuracy": metrics.get("overall", {}).get("accuracy"),
        "spatial_cv_macro_f1": metrics.get("overall", {}).get("macro_f1"),
    }


@app.post("/api/analyze-polygon")
def analyze_polygon(payload: PolygonRequest) -> dict[str, Any]:
    start_ts = time.time()
    area_m2 = polygon_area_m2(payload.geometry)
    area_km2 = area_m2 / 1_000_000.0
    area_ha = area_m2 / 10_000.0

    if area_km2 > MAX_POLYGON_KM2:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Polygon exceeds demo limit.",
                "area_km2": round(area_km2, 3),
                "max_polygon_km2": MAX_POLYGON_KM2,
            },
        )

    centroid = geometry_centroid(payload.geometry)
    return {
        "accepted": True,
        "area_km2": round(area_km2, 3),
        "area_ha": round(area_ha, 2),
        "max_polygon_km2": MAX_POLYGON_KM2,
        "centroid": centroid,
        "classification": demo_classify(payload.geometry, area_ha),
        "message": "Polygon accepted for live demo inference.",
        "latency_s": round(time.time() - start_ts, 3),
    }


@app.post("/api/submit-sentinel-analysis")
def submit_sentinel_analysis(payload: PolygonRequest) -> dict[str, Any]:
    area_m2 = polygon_area_m2(payload.geometry)
    area_km2 = area_m2 / 1_000_000.0
    area_ha = area_m2 / 10_000.0

    if area_km2 > MAX_POLYGON_KM2:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Polygon exceeds demo limit.",
                "area_km2": round(area_km2, 3),
                "max_polygon_km2": MAX_POLYGON_KM2,
            },
        )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"demo_{int(time.time())}_{abs(hash(json.dumps(payload.geometry, sort_keys=True))) % 1_000_000}"
    centroid = geometry_centroid(payload.geometry)
    submission = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "area_km2": round(area_km2, 3),
        "area_ha": round(area_ha, 2),
        "centroid": centroid,
        "geometry": payload.geometry,
        "next_step": "Export this submission to GeoJSON, create/start openEO jobs, download GeoTIFFs, then rerun automatic analysis.",
        "commands": [
            f"python scripts\\export_demo_submissions_geojson.py --job-id {job_id} --output demo_app\\submissions\\{job_id}.geojson",
            f"python scripts\\download_sentinel2_openeo.py --input-geojson demo_app\\submissions\\{job_id}.geojson --manifest openeo_jobs_manifest_{job_id}.json --start-jobs --auth-method device --auth-timeout 1800 --no-browser",
            f"python scripts\\download_openeo_results.py --manifest openeo_jobs_manifest_{job_id}.json --output sentinel2_l2a\\2025_05_06 --auth-method device --auth-timeout 1800 --no-browser",
        ],
    }
    (SUBMISSIONS_DIR / f"{job_id}.json").write_text(json.dumps(submission, indent=2), encoding="utf-8")

    return {
        "accepted": True,
        "job_id": job_id,
        "status": "queued",
        "area_km2": round(area_km2, 3),
        "area_ha": round(area_ha, 2),
        "commands": submission["commands"],
        "message": "Polygon queued for Sentinel-2 feature extraction.",
    }


@app.post("/api/classify-sentinel-local")
async def classify_sentinel_local(payload: PolygonRequest) -> dict[str, Any]:
    area_m2 = polygon_area_m2(payload.geometry)
    area_km2 = area_m2 / 1_000_000.0
    area_ha = area_m2 / 10_000.0

    if area_km2 > MAX_POLYGON_KM2:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Polygon exceeds demo limit.",
                "area_km2": round(area_km2, 3),
                "max_polygon_km2": MAX_POLYGON_KM2,
            },
        )

    return {
        "accepted": True,
        "area_km2": round(area_km2, 3),
        "area_ha": round(area_ha, 2),
        "sentinel_local_available": True,
        "classification": classify_local_sentinel_subprocess(payload.geometry),
    }


@app.post("/api/fast-analyze")
def fast_analyze(payload: PolygonRequest) -> dict[str, Any]:
    """Fast path: return cached prediction if the drawn polygon overlaps an existing parcel."""
    matching = find_matching_offline_parcel(payload.geometry)
    if matching is None:
        raise HTTPException(status_code=404, detail="No cached parcel matches this geometry.")

    props = matching.get("properties", {})
    prediction = {
        "method": "fast_cached",
        "matched_parcel_id": props.get("id"),
        "label": props.get("model_prediction") or props.get("cultivation_system"),
        "prob_intensif": props.get("prob_intensif"),
        "confidence": round(float(props.get("prob_intensif")), 3) if props.get("prob_intensif") is not None else None,
        "ground_truth": props.get("cultivation_system"),
        "note": "Returned cached prediction for overlapping parcel.",
    }

    return {"found": True, "classification": prediction}


@app.get("/api/sentinel-analysis/{job_id}")
def sentinel_analysis_status(job_id: str) -> dict[str, Any]:
    path = SUBMISSIONS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/openeo/{job_id}/export")
def export_openeo_request(job_id: str) -> dict[str, Any]:
    output = export_demo_submission_geojson(job_id)
    return {
        "job_id": job_id,
        "status": "exported",
        "geojson": str(output.relative_to(ROOT)),
    }


@app.post("/api/openeo/{job_id}/{action}")
def run_openeo_action(job_id: str, action: str) -> dict[str, Any]:
    return start_openeo_process(job_id, action)


@app.get("/api/openeo/{job_id}/{action}")
def get_openeo_action_status(job_id: str, action: str) -> dict[str, Any]:
    return openeo_status(job_id, action)


@app.get("/api/benchmark")
def benchmark_endpoint(n_runs: int = 10) -> dict[str, Any]:
    """Run N latency measurements on /api/fast-analyze and /api/analyze-polygon."""
    if n_runs < 1 or n_runs > 100:
        raise HTTPException(status_code=400, detail="n_runs must be between 1 and 100")

    # Get a test geometry: centroid of first parcel
    parcels_data = load_parcels_with_predictions()
    if not parcels_data.get("features"):
        raise HTTPException(status_code=404, detail="No parcels loaded for benchmark")

    test_geom = parcels_data["features"][0]["geometry"]

    fast_latencies = []
    full_latencies = []

    for i in range(n_runs):
        # Measure fast-analyze
        t0 = time.time()
        matching = find_matching_offline_parcel(test_geom)
        fast_lat = time.time() - t0
        fast_latencies.append(fast_lat)

        # Measure analyze-polygon (demo_classify)
        area_m2 = polygon_area_m2(test_geom)
        area_ha = area_m2 / 10_000.0
        t0 = time.time()
        demo_classify(test_geom, area_ha)
        full_lat = time.time() - t0
        full_latencies.append(full_lat)

    fast_sorted = sorted(fast_latencies)
    full_sorted = sorted(full_latencies)

    def percentile(arr, p):
        idx = int(len(arr) * (p / 100.0))
        return arr[min(idx, len(arr) - 1)]

    result = {
        "n_runs": n_runs,
        "fast_analyze": {
            "p50_ms": round(percentile(fast_sorted, 50) * 1000, 2),
            "p95_ms": round(percentile(fast_sorted, 95) * 1000, 2),
            "p99_ms": round(percentile(fast_sorted, 99) * 1000, 2),
            "max_ms": round(max(fast_latencies) * 1000, 2),
        },
        "full_analyze": {
            "p50_ms": round(percentile(full_sorted, 50) * 1000, 2),
            "p95_ms": round(percentile(full_sorted, 95) * 1000, 2),
            "p99_ms": round(percentile(full_sorted, 99) * 1000, 2),
            "max_ms": round(max(full_latencies) * 1000, 2),
        },
    }

    # Log result
    LATENCY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LATENCY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | n={n_runs} | fast_p99={result['fast_analyze']['p99_ms']}ms | full_p99={result['full_analyze']['p99_ms']}ms\n"
        )

    return result
