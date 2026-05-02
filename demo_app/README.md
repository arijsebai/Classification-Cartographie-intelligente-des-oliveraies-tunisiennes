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
POST /api/classify-sentinel-local
POST /api/submit-sentinel-analysis
GET  /api/sentinel-analysis/{job_id}
```

Le frontend calcule la surface pour feedback instantane. Le backend recalcule et applique aussi la limite pour eviter de faire confiance au navigateur.

## Classification live

Les parcelles EZZAYRA deja chargees sur la carte affichent la prediction reelle du modele offline quand `models/spatial_cv_classifier/spatial_cv_predictions.csv` ou `models/cultivation_classifier/predictions.csv` existe.

Clique sur une parcelle pour voir :

```text
verite terrain
prediction modele
probabilite intensif si disponible
source du modele
```

Apres dessin d'un nouveau polygone, l'API renvoie une classe demo :

```text
extensif
intensif
```

Important : un polygone dessine en live n'a pas automatiquement ses pixels Sentinel-2. Apres dessin, la page tente automatiquement de classifier avec les GeoTIFFs deja telecharges dans `sentinel2_l2a/2025_05_06`. Si le polygone intersecte une image locale, elle extrait les pixels, calcule les indices/features et applique le Random Forest entraine.

Si la zone n'est pas couverte localement, la page soumet automatiquement l'extraction openEO et affiche un `job_id`. La classification complete utilise le pipeline :

```text
Sentinel-2 -> features spectrales + texture -> Random Forest
```

## Soumission Sentinel-2

Apres avoir dessine un polygone valide sans couverture Sentinel locale, la demo cree automatiquement une demande dans :

```text
demo_app/submissions/
```

Chaque demande contient la geometrie GeoJSON, la surface, le centroid et un `job_id`. Pour exporter toutes les demandes en GeoJSON :

```powershell
python scripts\export_demo_submissions_geojson.py
```

Le fichier obtenu peut ensuite servir d'entree pour une extraction Sentinel-2/openEO asynchrone. C'est le chemin correct pour une zone non-vue : la demo repond vite, puis les vraies bandes Sentinel-2 sont calculees hors requete HTTP.
