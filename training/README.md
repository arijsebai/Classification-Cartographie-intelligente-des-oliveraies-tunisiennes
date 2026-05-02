Training README

1) Purpose
- Train a U-Net (ResNet backbone) for olive grove segmentation on EZZAYRA polygons.
- Rasterize polygon masks from the split JSON, sample tiles from multi-band TIFF stacks, apply augmentations, run spatial CV, and evaluate IoU on the test split.

2) Install (recommended in a venv)

```bash
python -m venv .venv
source .venv/bin/activate    # or .\.venv\Scripts\activate on Windows
pip install -r training/requirements-train.txt
```

3) Expected inputs
- `data_splits/ezzayra_oliviers/train.json`, `val.json`, and `test.json` from the hackathon split.
- A folder of georeferenced multi-band TIFF stacks, one per parcel or one per prepared sample.
- By default the code expects TIFF files named with the parcel id: `{image_root}/{id}.tif`.
- If your files use another naming scheme, pass `--image-pattern` or add `image_path` to the JSON records.
- The raster and polygon coordinates should use the same CRS, or the rasters should already be clipped in the polygon CRS.

4) Run spatial CV + final fit

```bash
python training/train_segmentation.py --image-root "C:\\Users\\SP\\Desktop\\Hack The Harvest\\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\\sentinel2_l2a\\2025_05_06\\train" \
  --train-split data_splits/ezzayra_oliviers/train.json \
  --val-split data_splits/ezzayra_oliviers/val.json \
  --epochs 25 --batch-size 4 --tile-size 256 --tiles-per-polygon 4 --cv-folds 3
```

5) Evaluate on test split

```bash
python training/evaluate_segmentation.py --image-root "C:\\Users\\SP\\Desktop\\Hack The Harvest\\Classification-Cartographie-intelligente-des-oliveraies-tunisiennes\\sentinel2_l2a\\2025_05_06\\train" \
  --test-split data_splits/ezzayra_oliviers/test.json \
  --checkpoint models/segmentation/final_best.pth \
  --output-json models/segmentation/test_metrics.json
```

6) What is implemented
- Real dataset class: `training/segmentation_dataset.py`
- Shared model/loss/metric helpers: `training/segmentation_core.py`
- Trainer with spatial CV, early stopping, and `ReduceLROnPlateau`: `training/train_segmentation.py`
- Test evaluation with IoU/Dice metrics: `training/evaluate_segmentation.py`

7) Next practical follow-up
- If your TIFF naming differs, tell me the pattern and I’ll wire the resolver directly.
- If you already have patch-level image/mask files, I can simplify the dataset to read them faster for the demo.
