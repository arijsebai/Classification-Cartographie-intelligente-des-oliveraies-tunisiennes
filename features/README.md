# Features Etage 2

Ce dossier extrait des features par parcelle a partir des GeoTIFF Sentinel-2 L2A.

## Commande

```powershell
python features\extract_parcel_features.py
```

Sortie :

```text
features/parcel_sentinel2_features.csv
```

## Features calculees avec les bandes actuelles

Les GeoTIFF actuels contiennent :

```text
B02, B03, B04, B08, B11
```

Features disponibles :

```text
NDVI mean/min/max/std/p10/p50/p90
NDWI mean/min/max/std/p10/p50/p90
NDVI range
NDVI p90-p10 amplitude intra-parcelle
FAPAR proxy depuis NDVI
LAI proxy depuis NDVI
valid_pixel_count
valid_pixel_ratio
```

Features geometriques :

```text
area_ha
vertex_count
bbox_width_deg
bbox_height_deg
bbox_aspect_ratio
perimeter_deg
polygon_area_deg2
compactness_deg
```

Textures GLCM haute saison :

```text
b08_glcm_contrast / dissimilarity / homogeneity / energy / correlation / ASM
b11_glcm_contrast / dissimilarity / homogeneity / energy / correlation / ASM
ndvi_glcm_contrast / dissimilarity / homogeneity / energy / correlation / ASM
ndwi_glcm_contrast / dissimilarity / homogeneity / energy / correlation / ASM
```

Ces textures mesurent la regularite spatiale des reflectances. Les systemes hyper-intensifs en haies peuvent produire des motifs plus repetitifs et homogenes, visibles dans NIR/NDVI meme a 10 m.

## Features approximatives

`NDRE` exige normalement une bande red-edge Sentinel-2 :

```text
B05, B06 ou B07
```

Comme les exports actuels n'ont pas de red-edge, le CSV contient :

```text
ndre_proxy
```

Pour un vrai NDRE, re-exporter au minimum :

```text
B05
```

Puis calculer :

```text
NDRE = (B08 - B05) / (B08 + B05)
```

## Features non disponibles avec un seul composite

L'amplitude saisonniere exige plusieurs dates ou plusieurs composites mensuels. Avec un seul composite mai-juin, elle est marquee :

```text
seasonal_ndvi_amplitude = NaN
seasonal_amplitude_available = False
```

Pour la produire, exporter par exemple :

```text
mars, avril, mai, juin, juillet, aout, septembre
```

Puis calculer :

```text
seasonal_ndvi_amplitude = max(NDVI_mensuel) - min(NDVI_mensuel)
```

## FAPAR / LAI

Les colonnes actuelles sont des proxys empiriques depuis NDVI :

```text
fapar_proxy
lai_proxy
```

Pour des valeurs biophysiques officielles, utiliser un produit Copernicus biophysical/FAPAR/LAI ou ajouter un processeur biophysique dedie.

## Classification extensif / intensif

Installer les dependances :

```powershell
pip install -r requirements-classification.txt
```

Entrainer le classifieur :

```powershell
python features\train_cultivation_classifier.py
```

Sorties :

```text
models/cultivation_classifier/
  random_forest.joblib
  metrics.json
  predictions.csv
  feature_importance.csv
```

Le modele utilise `train`, mesure `val`, puis donne le score final sur `test`.

## Validation croisee spatiale obligatoire

Pour evaluer la generalisation spatiale, utiliser `zone_id` comme groupe :

```powershell
python features\spatial_cv_classifier.py --model rf --group-column zone_id
```

Sorties :

```text
models/spatial_cv_classifier/
  rf_final.joblib
  spatial_cv_metrics.json
  spatial_cv_predictions.csv
```

Le script est multi-classe et accepte automatiquement les labels presents dans `cultivation_system` :

```text
extensif
intensif
hyper_intensif
```

Le dataset actuel ne contient que `extensif` et `intensif`; `hyper_intensif` sera pris en compte des que des parcelles labelisees seront ajoutees.

Option XGBoost :

```powershell
pip install xgboost
python features\spatial_cv_classifier.py --model xgboost --group-column zone_id
```
