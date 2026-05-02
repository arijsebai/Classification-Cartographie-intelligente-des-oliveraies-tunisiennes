from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import date as date_cls
from typing import Any, Dict, List, Literal, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

app = FastAPI(
        title="Cartographie intelligente des oliveraies tunisiennes",
        version="0.1.0",
)

MAX_AREA_KM2 = 100.0
EARTH_RADIUS_M = 6_371_008.8


class CartographieRequest(BaseModel):
        polygone_perimetre: Dict[str, Any]
        date: str = Field(..., description="Date ISO AAAA-MM-JJ")

        @field_validator("date")
        @classmethod
        def validate_date(cls, v: str) -> str:
                try:
                        date_cls.fromisoformat(v)
                except ValueError as exc:
                        raise ValueError("date doit etre au format AAAA-MM-JJ") from exc
                return v


class OliveGrovePrediction(BaseModel):
        polygone: Dict[str, Any]
        systeme: Literal["extensif", "intensif", "hyper_intensif"]
        confiance: float
        surface_ha: float


class StatsResponse(BaseModel):
        total_oliveraies: int
        surface_totale_ha: float
        repartition: Dict[str, int]
        surface_moyenne_ha: float


class CartographieResponse(BaseModel):
        oliveraies: List[OliveGrovePrediction]
        stats: StatsResponse


def _ensure_geojson_polygon(geo: Dict[str, Any]) -> List[List[float]]:
        if not isinstance(geo, dict) or geo.get("type") != "Polygon":
                raise HTTPException(status_code=422, detail="polygone_perimetre doit etre un GeoJSON Polygon")

        coordinates = geo.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) == 0:
                raise HTTPException(status_code=422, detail="Polygon invalide: coordinates manquantes")

        exterior = coordinates[0]
        if not isinstance(exterior, list) or len(exterior) < 4:
                raise HTTPException(status_code=422, detail="Polygon invalide: contour externe trop court")

        ring: List[List[float]] = []
        for p in exterior:
                if not isinstance(p, list) or len(p) != 2:
                        raise HTTPException(status_code=422, detail="Polygon invalide: chaque sommet doit etre [lon, lat]")
                lon, lat = float(p[0]), float(p[1])
                if lon < -180 or lon > 180 or lat < -90 or lat > 90:
                        raise HTTPException(status_code=422, detail="Polygon invalide: coordonnees hors limites")
                ring.append([lon, lat])

        if ring[0] != ring[-1]:
                ring.append(ring[0])

        return ring


def _spherical_polygon_area_m2(ring_lon_lat: List[List[float]]) -> float:
        if len(ring_lon_lat) < 4:
                return 0.0

        area_component = 0.0
        for i in range(len(ring_lon_lat) - 1):
                lon1, lat1 = ring_lon_lat[i]
                lon2, lat2 = ring_lon_lat[i + 1]
                lon1_r = math.radians(lon1)
                lon2_r = math.radians(lon2)
                lat1_r = math.radians(lat1)
                lat2_r = math.radians(lat2)
                area_component += (lon2_r - lon1_r) * (2 + math.sin(lat1_r) + math.sin(lat2_r))

        return abs(area_component) * (EARTH_RADIUS_M**2) / 2.0


def _polygon_centroid(ring: List[List[float]]) -> Tuple[float, float]:
        xs = [p[0] for p in ring[:-1]]
        ys = [p[1] for p in ring[:-1]]
        return sum(xs) / len(xs), sum(ys) / len(ys)


def _point_in_polygon(lon: float, lat: float, ring: List[List[float]]) -> bool:
        inside = False
        for i in range(len(ring) - 1):
                x1, y1 = ring[i]
                x2, y2 = ring[i + 1]
                intersects = ((y1 > lat) != (y2 > lat)) and (
                        lon < (x2 - x1) * (lat - y1) / ((y2 - y1) + 1e-12) + x1
                )
                if intersects:
                        inside = not inside
        return inside


def _square_polygon(center_lon: float, center_lat: float, half_size_deg: float) -> Dict[str, Any]:
        ring = [
                [center_lon - half_size_deg, center_lat - half_size_deg],
                [center_lon + half_size_deg, center_lat - half_size_deg],
                [center_lon + half_size_deg, center_lat + half_size_deg],
                [center_lon - half_size_deg, center_lat + half_size_deg],
                [center_lon - half_size_deg, center_lat - half_size_deg],
        ]
        return {"type": "Polygon", "coordinates": [ring]}


def _simulate_stage1_segmentation(perimeter_ring: List[List[float]], iso_date: str) -> List[Dict[str, Any]]:
        seed_key = json.dumps(perimeter_ring, sort_keys=True) + iso_date
        seed_int = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)

        lons = [p[0] for p in perimeter_ring[:-1]]
        lats = [p[1] for p in perimeter_ring[:-1]]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        area_ha = _spherical_polygon_area_m2(perimeter_ring) / 10_000.0
        estimated_count = max(6, min(120, int(area_ha / 18)))
        results: List[Dict[str, Any]] = []

        attempts = 0
        while len(results) < estimated_count and attempts < estimated_count * 12:
                attempts += 1
                lon = rng.uniform(min_lon, max_lon)
                lat = rng.uniform(min_lat, max_lat)
                if not _point_in_polygon(lon, lat, perimeter_ring):
                        continue
                half_size = rng.uniform(0.0007, 0.0024)
                candidate = _square_polygon(lon, lat, half_size)
                results.append(candidate)

        return results


def _fake_features_for_polygon(poly: Dict[str, Any], seed_suffix: str) -> Dict[str, float]:
        key = json.dumps(poly, sort_keys=True) + seed_suffix
        rng = random.Random(int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16))
        return {
                "ndvi_mean": rng.uniform(0.2, 0.85),
                "ndvi_amp": rng.uniform(0.03, 0.35),
                "ndwi": rng.uniform(-0.2, 0.4),
                "texture_regularity": rng.uniform(0.1, 0.95),
        }


def _simulate_stage2_classification(polygons: List[Dict[str, Any]], iso_date: str) -> List[OliveGrovePrediction]:
        predictions: List[OliveGrovePrediction] = []

        for poly in polygons:
                f = _fake_features_for_polygon(poly, iso_date)

                if f["texture_regularity"] > 0.74 and f["ndwi"] > 0.06:
                        systeme = "hyper_intensif"
                        confiance = 0.78 + (f["texture_regularity"] - 0.74) * 0.6
                elif f["ndvi_mean"] > 0.55:
                        systeme = "intensif"
                        confiance = 0.66 + (f["ndvi_mean"] - 0.55) * 0.7
                else:
                        systeme = "extensif"
                        confiance = 0.64 + (0.55 - f["ndvi_mean"]) * 0.35

                ring = poly["coordinates"][0]
                area_ha = _spherical_polygon_area_m2(ring) / 10_000.0

                predictions.append(
                        OliveGrovePrediction(
                                polygone=poly,
                                systeme=systeme,
                                confiance=round(max(0.51, min(0.98, confiance)), 3),
                                surface_ha=round(area_ha, 3),
                        )
                )

        return predictions


def _compute_stats(items: List[OliveGrovePrediction]) -> StatsResponse:
        repartition = {"extensif": 0, "intensif": 0, "hyper_intensif": 0}
        total_surface = 0.0

        for x in items:
                repartition[x.systeme] += 1
                total_surface += x.surface_ha

        total = len(items)
        return StatsResponse(
                total_oliveraies=total,
                surface_totale_ha=round(total_surface, 3),
                repartition=repartition,
                surface_moyenne_ha=round(total_surface / total, 3) if total else 0.0,
        )


@app.get("/api/health")
def health() -> Dict[str, str]:
        return {"status": "ok"}


@app.post("/api/cartographier", response_model=CartographieResponse)
def cartographier(data: CartographieRequest) -> CartographieResponse:
        try:
                perimeter_ring = _ensure_geojson_polygon(data.polygone_perimetre)
        except ValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        area_km2 = _spherical_polygon_area_m2(perimeter_ring) / 1_000_000.0
        if area_km2 <= 0:
                raise HTTPException(status_code=422, detail="Le polygone doit avoir une surface > 0")
        if area_km2 > MAX_AREA_KM2:
                raise HTTPException(
                        status_code=413,
                        detail=f"Zone trop grande pour la demo ({area_km2:.2f} km2). Maximum autorise: {MAX_AREA_KM2:.0f} km2",
                )

        stage1_polygons = _simulate_stage1_segmentation(perimeter_ring, data.date)
        stage2_predictions = _simulate_stage2_classification(stage1_polygons, data.date)
        stats = _compute_stats(stage2_predictions)

        return CartographieResponse(oliveraies=stage2_predictions, stats=stats)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
        return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Cartographie des Oliveraies</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
    <style>
        :root {
            --bg: #f7f8f1;
            --panel: #ffffff;
            --ink: #1f2a1f;
            --muted: #5b6656;
            --accent: #2f6d3c;
            --accent-soft: #d9e9cf;
            --danger: #a5281b;
        }
        body {
            margin: 0;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            background: radial-gradient(circle at 20% 20%, #ffffff, var(--bg));
            color: var(--ink);
        }
        .layout {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 14px;
            height: 100vh;
            padding: 14px;
            box-sizing: border-box;
        }
        .panel {
            background: var(--panel);
            border: 1px solid #d9dfd4;
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 8px 24px rgba(50, 60, 50, 0.09);
            overflow: auto;
        }
        .badge {
            display: inline-block;
            font-size: 12px;
            background: var(--accent-soft);
            color: var(--accent);
            padding: 4px 8px;
            border-radius: 999px;
            margin-bottom: 8px;
        }
        h1 {
            font-size: 19px;
            margin: 0 0 10px;
        }
        p {
            color: var(--muted);
            font-size: 14px;
            margin-top: 0;
        }
        #map {
            width: 100%;
            height: 100%;
            border-radius: 14px;
            border: 1px solid #d9dfd4;
        }
        .btn {
            width: 100%;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 10px;
        }
        .btn:disabled {
            opacity: 0.55;
            cursor: not-allowed;
        }
        .btn-outline {
            background: white;
            color: var(--accent);
            border: 1px solid var(--accent);
        }
        .stats {
            margin-top: 12px;
            font-size: 14px;
            line-height: 1.6;
        }
        .legend {
            margin-top: 10px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 6px;
            font-size: 13px;
        }
        .dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .err {
            color: var(--danger);
            font-weight: 700;
            margin-top: 8px;
            font-size: 13px;
            min-height: 18px;
        }
        @media (max-width: 980px) {
            .layout {
                grid-template-columns: 1fr;
                grid-template-rows: auto 1fr;
            }
            .panel {
                max-height: 42vh;
            }
        }
    </style>
</head>
<body>
    <div class="layout">
        <section class="panel">
            <span class="badge">Hackathon Demo</span>
            <h1>Cartographie intelligente des oliveraies</h1>
            <p>Dessine un polygone en Tunisie, puis lance le pipeline detection + classification.</p>

            <label for="dateInput">Date d'analyse (mai-juin recommande)</label>
            <input id="dateInput" type="date" value="2026-06-15" style="width:100%;padding:8px;margin-top:6px;border-radius:8px;border:1px solid #cfd7ca;" />

            <button id="analyserBtn" class="btn">Analyser cette zone</button>
            <button id="telechargerBtn" class="btn btn-outline" disabled>Telecharger GeoJSON</button>
            <div id="error" class="err"></div>

            <div class="legend">
                <div><span class="dot" style="background:#2e8b57"></span>Extensif</div>
                <div><span class="dot" style="background:#d4a017"></span>Intensif</div>
                <div><span class="dot" style="background:#c33f2e"></span>Hyper-intensif</div>
            </div>

            <div id="stats" class="stats">Aucune analyse lancee.</div>
        </section>
        <div id="map"></div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <script>
        const map = L.map('map').setView([34.2, 9.4], 7);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: 'OpenStreetMap' }).addTo(map);

        const drawLayer = new L.FeatureGroup().addTo(map);
        const resultLayer = new L.FeatureGroup().addTo(map);
        let lastGeoJson = null;

        const drawControl = new L.Control.Draw({
            draw: {
                polygon: true,
                rectangle: false,
                circle: false,
                circlemarker: false,
                marker: false,
                polyline: false
            },
            edit: {
                featureGroup: drawLayer,
                edit: true,
                remove: true
            }
        });
        map.addControl(drawControl);

        map.on(L.Draw.Event.CREATED, (e) => {
            drawLayer.clearLayers();
            drawLayer.addLayer(e.layer);
            resultLayer.clearLayers();
            lastGeoJson = null;
            document.getElementById('telechargerBtn').disabled = true;
            document.getElementById('stats').innerText = 'Polygone pret. Lancez analyse.';
            document.getElementById('error').innerText = '';
        });

        map.on(L.Draw.Event.EDITED, () => {
            resultLayer.clearLayers();
            lastGeoJson = null;
            document.getElementById('telechargerBtn').disabled = true;
            document.getElementById('stats').innerText = 'Polygone modifie. Relancez analyse.';
            document.getElementById('error').innerText = '';
        });

        map.on(L.Draw.Event.DELETED, () => {
            resultLayer.clearLayers();
            lastGeoJson = null;
            document.getElementById('telechargerBtn').disabled = true;
            document.getElementById('stats').innerText = 'Aucun polygone. Dessinez une zone.';
            document.getElementById('error').innerText = '';
        });

        function colorFor(systeme) {
            if (systeme === 'extensif') return '#2e8b57';
            if (systeme === 'intensif') return '#d4a017';
            return '#c33f2e';
        }

        document.getElementById('analyserBtn').addEventListener('click', async () => {
            const errorEl = document.getElementById('error');
            const statsEl = document.getElementById('stats');
            errorEl.innerText = '';

            if (drawLayer.getLayers().length === 0) {
                errorEl.innerText = 'Dessinez d abord un polygone.';
                return;
            }

            const poly = drawLayer.getLayers()[0].toGeoJSON().geometry;
            const dateValue = document.getElementById('dateInput').value || '2026-06-15';
            const btn = document.getElementById('analyserBtn');
            btn.disabled = true;
            btn.innerText = 'Analyse en cours...';

            try {
                const resp = await fetch('/api/cartographier', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ polygone_perimetre: poly, date: dateValue })
                });

                const data = await resp.json();
                if (!resp.ok) {
                    throw new Error(data.detail || 'Erreur API');
                }

                resultLayer.clearLayers();
                const fc = { type: 'FeatureCollection', features: [] };

                data.oliveraies.forEach((o) => {
                    const feature = {
                        type: 'Feature',
                        properties: {
                            systeme: o.systeme,
                            confiance: o.confiance,
                            surface_ha: o.surface_ha
                        },
                        geometry: o.polygone
                    };
                    fc.features.push(feature);

                    L.geoJSON(feature, {
                        style: {
                            color: colorFor(o.systeme),
                            weight: 1,
                            fillOpacity: 0.45
                        }
                    })
                    .bindPopup('Systeme: ' + o.systeme + '<br>Confiance: ' + o.confiance + '<br>Surface: ' + o.surface_ha + ' ha')
                    .addTo(resultLayer);
                });

                lastGeoJson = fc;
                document.getElementById('telechargerBtn').disabled = false;

                const s = data.stats;
                statsEl.innerHTML =
                    'Total oliveraies: <b>' + s.total_oliveraies + '</b><br>' +
                    'Surface totale: <b>' + s.surface_totale_ha + ' ha</b><br>' +
                    'Surface moyenne: <b>' + s.surface_moyenne_ha + ' ha</b><br>' +
                    'Repartition: extensif=' + s.repartition.extensif +
                    ', intensif=' + s.repartition.intensif +
                    ', hyper_intensif=' + s.repartition.hyper_intensif;
            } catch (err) {
                errorEl.innerText = err.message;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Analyser cette zone';
            }
        });

        document.getElementById('telechargerBtn').addEventListener('click', () => {
            if (!lastGeoJson) return;
            const blob = new Blob([JSON.stringify(lastGeoJson, null, 2)], { type: 'application/geo+json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'oliveraies_annotees.geojson';
            a.click();
            URL.revokeObjectURL(url);
        });
    </script>
</body>
</html>
"""