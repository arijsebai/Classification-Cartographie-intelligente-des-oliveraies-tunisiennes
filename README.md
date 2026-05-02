# Classification & Cartographie Intelligente des Oliviers Tunisiens

## Résumé
Projet de recherche et démonstration pour la classification des systèmes de culture d'oliviers (intensif vs extensif) à partir d'images Sentinel‑2 et d'un pipeline de segmentation + classification. Le dépôt contient : prétraitement, entraînement de modèles (U‑Net pour segmentation, Random Forest pour classification), scripts d'inférence, jeu de données augmenté et une démo web interactive.

## Points clés
- Données : patches Sentinel‑2 L2A (multi‑bands) et géométries de parcelles.
- Segmentation : U‑Net (encoder ResNet34 via `segmentation_models_pytorch`).
- Classification : Random Forest sur caractéristiques extraites (NDVI, statistiques, textures GLCM).
- Validation spatiale : séparation par zones pour éviter fuite spatiale.
- Démo : interface Leaflet + FastAPI pour inspection interactive et cache rapide.

## Structure du dépôt (high‑level)
- `demo_app/` : application FastAPI + front-end (Leaflet) pour démonstration et exploration.
  - `demo_app/main.py` — API et logique demo
  - `demo_app/static/app.js` — frontend JS (cartographie, popups)
- `training/` : scripts d'entraînement et prédiction pour U‑Net
  - `training/train_unet_smp.py` — entraînement U‑Net
  - `training/predict_unet_smp.py` — inférence / métriques
- `models/` : modèles entraînés (checkpoints, joblib, etc.)
- `scripts/` : utilitaires et scripts d'analyse
  - `scripts/classify_polygon_local.py` — wrapper pour classification locale
- `data_splits/` : geojsons des splits (train/val/test), versions augmentées
- `sentinel2_l2a/` : arborescence locale des GeoTIFF Sentinel‑2 (utilisée par la démo)
- `predictions/` : sorties d'inférence enregistrées
- `training/` et `models/` contiennent les artefacts d'entraînement et résultats.

## Résultats clés (résumé)
- Segmentation (U‑Net, données augmentées): Test IoU ≈ 0.9975, Dice ≈ 0.9987 (checkpoint: `models/unet_resnet_sentinel2_augmented/best.pt`).
- Classification (Random Forest): F1 macro optimisée = 0.8444 avec seuil `prob_intensif >= 0.2` (voir `f1_macro_results.json` et `F1_MACRO_REPORT.md`).

> Note : Les métriques ci‑dessus proviennent des jeux de test/validation indiqués dans `data_splits/ezzayra_oliviers_geojson_augmented/`.

## Prérequis & environnement
- Systèmes testés : Windows / Linux
- Python 3.10+ recommandé
- Installer dépendances (exemples) :

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements-classification.txt
pip install -r requirements-training.txt
```

Les fichiers `requirements-*.txt` fournis contiennent les bibliothèques nécessaires pour chaque sous‑module (training, demo, openeo, etc.).

## Démarrage rapide — démo locale
1. S'assurer que les fichiers Sentinel‑2 sont disponibles sous `sentinel2_l2a/2025_05_06/` (ou pointer `SENTINEL_ROOT` dans `demo_app/main.py`).
2. Lancer l'API demo :

```bash
cd demo_app
pip install -r ../requirements-demo.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Ouvrir la démo dans le navigateur : `http://localhost:8000`
4. Dessiner un polygone pour obtenir une prédiction immédiate (fast‑cache) ou lancer l'extraction Sentinel‑2 si pas d'image locale.

## Entraînement — segmentation (U‑Net)
Exemple rapide pour relancer l'entraînement U‑Net sur le dataset augmenté :

```bash
python training/train_unet_smp.py \
  --sentinel-root sentinel2_l2a/2025_05_06 \
  --geojson-root data_splits/ezzayra_oliviers_geojson_augmented \
  --output-dir models/unet_resnet_sentinel2_augmented \
  --epochs 30 --batch-size 4 --lr 1e-4
```

Résultats et checkpoints sont sauvegardés dans le dossier `--output-dir`.

## Inférence segmentation

```bash
python training/predict_unet_smp.py --checkpoint models/unet_resnet_sentinel2_augmented/best.pt --split test --save-masks
```

Les métriques sont écrites sous `predictions/unet_resnet_sentinel2_augmented/metrics.json`.

## Entraînement & réentrainement Random Forest (classification)
Un script de génération d'extraits et d'entraînement RF existe — ajuster `train_rf_augmented.py` si nécessaire :

```bash
python scripts/train_rf_augmented.py \
  --geojson data_splits/ezzayra_oliviers_geojson_augmented/all_splits.geojson \
  --sentinel-root sentinel2_l2a/2025_05_06 \
  --output-dir models/cultivation_classifier_augmented \
  --n-estimators 500
```

Sorties attendues : `random_forest_augmented.joblib` et `metrics_augmented.json`.

## API & points d'extension
- `demo_app/main.py` expose les endpoints suivants :
  - `POST /api/fast-analyze` — recherche cache locale
  - `POST /api/analyze-polygon` — analyse complète + extraction openEO (si nécessaire)
  - `POST /api/classify-sentinel-local` — classification sur GeoTIFF local
  - `POST /api/cache-add`, `GET /api/cache-list` — gestion cache session

Voir le code pour les détails d'implémentation : [demo_app/main.py](demo_app/main.py)

## Tests front-end
Un fichier de test statique a été ajouté pour valider l'affichage des probabilités :
- `test_frontend_display.html` — ouvre localement pour vérifier l'affichage [successful tests].

## Bonnes pratiques et considérations
- Valeurs de seuils : la classification binaire utilise un seuil optimisé (`INTENSIF_THRESHOLD`, documenté dans le code et dans `F1_MACRO_REPORT.md`).
- Validation spatiale obligatoire : regrouper par `zone_id` pour éviter fuite spatiale dans CV.
- Augmentation des parcelles : nous avons appliqué une augmentation géométrique contrôlée (5×) et conservé `original_id`/`variant` pour traçabilité.

## Fichiers importants
- Modèles segmentation : `models/unet_resnet_sentinel2_augmented/`
- Modèles classification : `models/cultivation_classifier/` (ou `_augmented`)
- Data splits augmentés : `data_splits/ezzayra_oliviers_geojson_augmented/`
- Scripts d'appui : `scripts/` (ex. `classify_polygon_local.py`)

## Dépannage rapide
- Si `SENTINEL_ROOT` introuvable → vérifier le chemin et les droits de lecture.
- Si l'entraînement échoue pour manque de dépendances → installer `requirements-training.txt`.
- Si la démo renvoie `No local Sentinel-2 GeoTIFF` → lancer `submit-sentinel-analysis` pour extraction openEO.

## Licence & citation
Consulter `LICENSE` pour les termes d'utilisation. Si vous utilisez ce travail dans un article, citez l'équipe/projet conformément aux règles du dépôt.

## Contribuer
- Ouvrir un issue pour bugs/requests.
- Fork & PR : tests et documentation requis.

## Contact
- Mainteneur principal : voir métadonnées du dépôt ou `demo_app/README.md` pour contacts.

---

Merci d'utiliser le projet — dites-moi si vous voulez que je personnalise ce README (version anglaise, badges CI, templates de contribution, ou ajout d'exemples de résultats).