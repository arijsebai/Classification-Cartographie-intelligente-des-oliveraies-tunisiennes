# Demo FastAPI + Leaflet

Demo live avec :

```text
FastAPI
Leaflet
Leaflet.draw
limite polygone: 100 km2
```

## Installation

```powershell
pip install -r requirements-demo.txt
```

## Lancement

```powershell
uvicorn demo_app.main:app --reload --host 127.0.0.1 --port 8000
```

Puis ouvrir :

```text
http://127.0.0.1:8000
```

Sous Windows, tu peux aussi garder ce script ouvert :

```powershell
.\demo_app\run_demo.ps1
```

## API

```text
GET  /api/config
GET  /api/parcels
POST /api/analyze-polygon
```

Le frontend calcule la surface pour feedback instantane. Le backend recalcule et applique aussi la limite pour eviter de faire confiance au navigateur.
