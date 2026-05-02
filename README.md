# Cartographie intelligente des oliveraies — Hackathon MVP

Contenu minimal pour la démo hackathon :

- `main.py` : FastAPI app + front Leaflet (route `/`).
- `requirements.txt` : dépendances Python.
- `scripts/data_split.py` : utilitaire pour découper spatialement EZZAYRA en train/val/test.
- `scripts/train_segmentation.py` : scaffold pour entraîner U-Net (à remplir).
- `scripts/train_classification.py` : scaffold pour entraîner classifieur (à remplir).

Quick start (depuis le dossier `Classification-Cartographie-intelligente-des-oliveraies-tunisiennes`):

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Ouvrir `http://127.0.0.1:8000/` pour la carte et la démo.

Data split example:

```powershell
..\.venv\Scripts\python.exe -m scripts.data_split --input data/ezzayra.json --out_dir data/splits --seed 42
```

Next actions:
- Remplir `train_segmentation.py` pour charger tes patches Sentinel-2 et entraîner U-Net.
- Remplir `train_classification.py` pour extraire features et entraîner RandomForest/XGBoost.
- Brancher les modèles exportés dans `main.py` (remplacer simulate_* par inference réelle).
