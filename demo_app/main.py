import json
import math
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

app = FastAPI(title="EZZAYRA Olive Mapping Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PolygonRequest(BaseModel):
    geometry: dict[str, Any] = Field(..., description="GeoJSON Polygon geometry")


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
    path = ROOT / "data_splits" / "ezzayra_oliviers_geojson" / "all_splits.geojson"
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/analyze-polygon")
def analyze_polygon(payload: PolygonRequest) -> dict[str, Any]:
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
        "message": "Polygon accepted for live demo inference.",
    }
