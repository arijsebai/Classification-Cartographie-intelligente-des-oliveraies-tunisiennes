# Entrainement U-Net Sentinel-2

Cette etape entraine un U-Net avec backbone ResNet pre-entraine via `segmentation_models_pytorch`.

## Entrees attendues

Images Sentinel-2 :

```text
sentinel2_l2a/2025_05_06/
  train/*.tif
  val/*.tif
  test/*.tif
```

Polygones labels :

```text
data_splits/ezzayra_oliviers_geojson/
  train.geojson
  val.geojson
  test.geojson
```

Chaque masque binaire est genere automatiquement en rasterisant le polygone de la parcelle sur la grille du GeoTIFF.

## Installation

```powershell
pip install -r requirements-training.txt
```

Si tu n'as pas de GPU CUDA, PyTorch utilisera le CPU, mais ce sera plus lent.

## Entrainement

Commande rapide pour tester :

```powershell
python training\train_unet_smp.py --epochs 2 --batch-size 2 --image-size 256
```

Commande plus serieuse :

```powershell
python training\train_unet_smp.py --epochs 30 --batch-size 4 --image-size 256 --encoder resnet34
```

La loss est :

```text
0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

Sorties :

```text
models/unet_resnet_sentinel2/
  best.pt
  last.pt
  config.json
  history.json
```

## Evaluation test

```powershell
python training\predict_unet_smp.py --split test
```

Sorties :

```text
predictions/unet_resnet_sentinel2/
  test_metrics.json
  test/*.tif
```

## Note importante

Les GeoTIFF decoupes strictement par parcelle ne permettent pas une vraie evaluation detection : le masque peut devenir toute l'image. Pour viser le critere IoU detection, exporter des chips avec buffer :

```powershell
python scripts\download_sentinel2_openeo.py --start-jobs --export-region bbox --buffer-deg 0.01
```

Puis entrainer/evaluer avec le masque rasterise :

```powershell
python training\train_unet_smp.py --epochs 60 --batch-size 4 --image-size 256 --encoder resnet34 --mask-mode rasterize
python training\predict_unet_smp.py --split test --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --mask-mode rasterize
```
