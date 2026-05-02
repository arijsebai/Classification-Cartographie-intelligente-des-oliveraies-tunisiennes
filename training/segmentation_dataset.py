from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def load_split_records(split_json_path: str) -> List[Dict[str, Any]]:
    with open(split_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    split_name = None
    if isinstance(data, dict) and "parcels" in data:
        records = data["parcels"]
        split_name = data.get("split")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported split JSON structure in {split_json_path}")

    normalized: List[Dict[str, Any]] = []
    for record in records:
        coords = record.get("coordinates") or record.get("polygon")
        if not coords:
            continue
        normalized_record = dict(record)
        if split_name and "split" not in normalized_record:
            normalized_record["split"] = split_name
        normalized.append(normalized_record)
    return normalized


def record_to_ring(record: Dict[str, Any]) -> List[Tuple[float, float]]:
    coords = record.get("coordinates") or []
    ring: List[Tuple[float, float]] = []
    for point in coords:
        if isinstance(point, dict):
            lat = float(point["lat"])
            lng = float(point["lng"])
        else:
            # accept [lat, lng] or [lng, lat] if user provided a raw array
            if len(point) != 2:
                raise ValueError("Invalid coordinate pair in record")
            lat = float(point[0])
            lng = float(point[1])
        ring.append((lng, lat))

    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def record_to_polygon_geojson(record: Dict[str, Any]) -> Dict[str, Any]:
    ring = record_to_ring(record)
    return {"type": "Polygon", "coordinates": [[list(pt) for pt in ring]]}


def record_to_polygon_geojson_in_crs(record: Dict[str, Any], dst_crs) -> Dict[str, Any]:
    polygon = record_to_polygon_geojson(record)
    if not dst_crs:
        return polygon

    try:
        from rasterio.warp import transform_geom
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("rasterio is required for geometry reprojection") from exc

    return transform_geom("EPSG:4326", dst_crs, polygon)


def resolve_image_path(record: Dict[str, Any], image_root: str | os.PathLike[str], image_pattern: str) -> Path:
    candidate = record.get("image_path") or record.get("raster_path") or record.get("tif_path")
    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            path = Path(image_root) / path
        return path

    sample_id = record.get("id") or record.get("parcel_id")
    if not sample_id:
        raise ValueError("Cannot resolve image path: record missing id")

    split = str(record.get("split", "") or "").strip()
    cultivation_system = str(record.get("cultivation_system", "") or "").strip()
    base_root = Path(image_root)
    formatted_name = image_pattern.format(
        id=sample_id,
        governorate=record.get("governorate", ""),
        split=split,
        cultivation_system=cultivation_system,
    )

    candidate_paths = []
    for root in [base_root, base_root / split if split else None]:
        if root is None:
            continue
        candidate_paths.append(root / formatted_name)
        candidate_paths.append(root / f"{sample_id}.tif")
        if cultivation_system:
            candidate_paths.append(root / f"{cultivation_system}__{sample_id}.tif")

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Missing image stack for record "
        f"{sample_id}. Tried: {', '.join(str(path) for path in candidate_paths)}"
    )


def build_augmentation_pipeline(tile_size: int):
    try:
        import albumentations as A
    except Exception as exc:  # pragma: no cover - import guard for minimal environments
        # Fallback: provide a minimal augmentation pipeline using numpy + cv2
        # when albumentations is not available. This keeps the training path
        # usable for quick local smoke tests without requiring heavy build
        # toolchains.
        import numpy as np
        import cv2

        def _noop(image, mask):
            return image, mask

        def _augment(image, mask):
            # Random horizontal/vertical flips
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

            # Random 90 deg rotation
            k = np.random.randint(0, 4)
            if k != 0:
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()

            # Small brightness/contrast jitter
            if np.random.rand() < 0.3:
                alpha = 1.0 + (np.random.rand() - 0.5) * 0.2
                beta = (np.random.rand() - 0.5) * 0.1
                image = np.clip(image * alpha + beta, 0.0, 1.0)

            # Ensure crop/pad to tile_size
            h, w = mask.shape[:2]
            th = tile_size
            tw = tile_size
            if h != th or w != tw:
                image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

            return image, mask

        def _wrapper(image, mask):
            out_image, out_mask = (_augment(image, mask) if np.random.rand() < 0.9 else _noop(image, mask))
            return {"image": out_image, "mask": out_mask}

        return _wrapper

    # Preferred: return albumentations pipeline if available
    return A.Compose(
        [
            A.RandomCrop(height=tile_size, width=tile_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10, rotate_limit=20, border_mode=0, p=0.6),
            A.GaussNoise(var_limit=(5.0, 35.0), p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.2),
        ]
    )


def _center_window(center_row: int, center_col: int, tile_size: int, height: int, width: int):
    try:
        from rasterio.windows import Window
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("rasterio is required for window sampling") from exc

    half = tile_size // 2
    row_off = max(0, center_row - half)
    col_off = max(0, center_col - half)
    return Window(col_off, row_off, tile_size, tile_size)


def _sample_center(record: Dict[str, Any], src, tile_size: int, jitter: int, seed: int) -> Any:
    ring = record_to_ring(record)
    if not ring:
        return tile_size // 2, tile_size // 2

    lngs = [p[0] for p in ring[:-1]]
    lats = [p[1] for p in ring[:-1]]
    center_lng = sum(lngs) / len(lngs)
    center_lat = sum(lats) / len(lats)

    if src.crs:
        try:
            from rasterio.warp import transform
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("rasterio is required for coordinate reprojection") from exc

        xs, ys = transform("EPSG:4326", src.crs, [center_lng], [center_lat])
        center_x, center_y = xs[0], ys[0]
    else:
        center_x, center_y = center_lng, center_lat

    row, col = src.index(center_x, center_y)

    rng = random.Random(seed)
    if jitter > 0:
        row += rng.randint(-jitter, jitter)
        col += rng.randint(-jitter, jitter)
    return row, col


def _sample_random_center(src, tile_size: int, seed: int) -> Tuple[int, int]:
    rng = random.Random(seed)
    height = src.height
    width = src.width
    row = rng.randint(tile_size // 2, max(tile_size // 2, height - tile_size // 2 - 1))
    col = rng.randint(tile_size // 2, max(tile_size // 2, width - tile_size // 2 - 1))
    return row, col


class EzzayraSegmentationDataset:
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        image_root: str,
        image_pattern: str = "{id}.tif",
        bands: Sequence[int] = (2, 3, 4, 8, 11),
        tile_size: int = 256,
        tiles_per_polygon: int = 1,
        augment: bool = False,
        positive_tile_ratio: float = 0.85,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.image_root = Path(image_root)
        self.image_pattern = image_pattern
        self.bands = tuple(int(b) for b in bands)
        self.tile_size = int(tile_size)
        self.tiles_per_polygon = max(1, int(tiles_per_polygon))
        self.augment = augment
        self.positive_tile_ratio = float(positive_tile_ratio)
        self.seed = int(seed)
        self._augmenter = build_augmentation_pipeline(self.tile_size) if augment else None

    def __len__(self) -> int:
        return len(self.records) * self.tiles_per_polygon

    def _resolve_band_indexes(self, src) -> List[int]:
        if src.count == len(self.bands):
            return list(range(1, src.count + 1))
        if max(self.bands) <= src.count:
            return list(self.bands)
        raise ValueError(
            f"Raster has {src.count} bands, but requested bands {self.bands} cannot be mapped. "
            "Use a raster with stacked requested bands or pass raster band indexes."
        )

    def _load_sample(self, record: Dict[str, Any], sample_index: int):
        try:
            import cv2
            import numpy as np
            import rasterio
            from rasterio.features import rasterize
            from rasterio.windows import Window
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("Missing runtime dependencies for segmentation dataset") from exc

        image_path = resolve_image_path(record, self.image_root, self.image_pattern)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image stack: {image_path}")

        rng = random.Random(self.seed + sample_index)

        with rasterio.open(image_path) as src:
            band_indexes = self._resolve_band_indexes(src)

            polygon_geojson = record_to_polygon_geojson_in_crs(record, src.crs)

            positive = rng.random() < self.positive_tile_ratio or not self.augment
            if positive:
                center_row, center_col = _sample_center(
                    record,
                    src,
                    self.tile_size,
                    jitter=max(1, self.tile_size // 12) if self.augment else 0,
                    seed=self.seed + sample_index,
                )
            else:
                center_row, center_col = _sample_random_center(src, self.tile_size, seed=self.seed + sample_index)

            window = Window(center_col - self.tile_size // 2, center_row - self.tile_size // 2, self.tile_size, self.tile_size)
            image = src.read(indexes=band_indexes, window=window, boundless=True, fill_value=0)
            image = np.moveaxis(image, 0, -1)

            transform = src.window_transform(window)
            mask = rasterize(
                [(polygon_geojson, 1)],
                out_shape=(self.tile_size, self.tile_size),
                transform=transform,
                fill=0,
                dtype="uint8",
            )

        image = image.astype(np.float32)
        # Copernicus L2A reflectance is usually scaled by 10000.
        image = image / 10000.0

        if self._augmenter is not None:
            augmented = self._augmenter(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # ensure channel-first tensors later in training loop
        return image, mask.astype(np.float32), record

    def __getitem__(self, index: int):
        record = self.records[index % len(self.records)]
        image, mask, record = self._load_sample(record, index)
        return image, mask, record
