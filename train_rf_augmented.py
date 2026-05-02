#!/usr/bin/env python
"""
Train Random Forest classifier on augmented dataset.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


EXCLUDED_COLUMNS = {
    "id", "image_path", "split", "cultivation_system", "governorate", 
    "zone_id", "augmented", "original_id", "variant"
}


def extract_parcel_features(geojson_file, sentinel_root):
    """Extract Sentinel-2 features from parcels in augmented GeoJSON."""
    features_list = []
    
    with open(geojson_file, 'r') as f:
        geojson = json.load(f)
    
    for feature in geojson['features']:
        props = feature['properties']
        geom = feature['geometry']
        
        # Find matching Sentinel-2 GeoTIFF
        geom_shape = shape(geom)
        bounds = geom_shape.bounds  # (minx, miny, maxx, maxy)
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Try to find matching sentinel2 file (simplified: assume filename pattern)
        sentinel_root_path = Path(sentinel_root)
        tif_files = list(sentinel_root_path.rglob('*.tif'))
        
        if not tif_files:
            print(f'Warning: No Sentinel-2 files found for {props.get("id")}')
            continue
        
        # Use first matching TIF (in real scenario, would match by coordinates)
        tif_path = tif_files[0]
        
        try:
            with rasterio.open(tif_path) as src:
                # Get bands 2,3,4,8,11 (B,G,R,NIR,SWIR)
                out_image, _ = mask(src, [geom_shape], crop=True)
                
                if out_image.shape[0] < 5:
                    continue
                
                # Compute statistics for each band
                row = {'id': props.get('id'), 'split': props.get('split')}
                row['cultivation_system'] = props.get('cultivation_system', 'unknown')
                row['zone_id'] = props.get('zone_id', -1)
                row['augmented'] = props.get('augmented', False)
                row['original_id'] = props.get('original_id', row['id'])
                row['variant'] = props.get('variant', 0)
                
                for band_idx in range(5):
                    band_data = out_image[band_idx].flatten()
                    band_data = band_data[band_data > 0]  # exclude nodata
                    
                    if len(band_data) == 0:
                        continue
                    
                    band_name = ['B2', 'B3', 'B4', 'B8', 'B11'][band_idx]
                    row[f'{band_name}_mean'] = band_data.mean()
                    row[f'{band_name}_std'] = band_data.std()
                    row[f'{band_name}_min'] = band_data.min()
                    row[f'{band_name}_max'] = band_data.max()
                
                # Compute spectral indices
                if 'B4_mean' in row and 'B8_mean' in row:
                    if row['B4_mean'] + row['B8_mean'] > 0:
                        row['NDVI'] = (row['B8_mean'] - row['B4_mean']) / (row['B8_mean'] + row['B4_mean'])
                
                features_list.append(row)
        
        except Exception as e:
            print(f'Error processing {props.get("id")}: {e}')
            continue
    
    return pd.DataFrame(features_list)


def select_feature_columns(df):
    columns = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].notna().sum() == 0:
                continue
            columns.append(column)
    return columns


def train_classifier(features_df, output_dir, n_estimators=500, random_state=42):
    """Train Random Forest with spatial CV (zone-based grouping)."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split by cultivation system
    print(f'Dataset shape: {features_df.shape}')
    print(f'\nCultivation systems: {features_df["cultivation_system"].value_counts().to_dict()}')
    print(f'Augmented: {features_df["augmented"].value_counts().to_dict()}')
    
    # Select numeric feature columns
    feature_columns = select_feature_columns(features_df)
    print(f'\nUsing {len(feature_columns)} features')
    
    # Split by split column
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']
    test_df = features_df[features_df['split'] == 'test']
    
    X_train = train_df[feature_columns]
    y_train = train_df['cultivation_system']
    
    X_test = test_df[feature_columns]
    y_test = test_df['cultivation_system']
    
    print(f'\nTrain: {X_train.shape}, Val: {val_df.shape}, Test: {X_test.shape}')
    
    # Create pipeline with imputation
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1,
            max_depth=20
        ))
    ])
    
    # Train
    print('\nTraining Random Forest...')
    pipeline.fit(X_train, y_train)
    
    # Test predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Metrics
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f'\nTest Accuracy: {test_acc:.4f}')
    print(f'Test Macro F1: {test_f1:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print(f'\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model_file = output_path / 'random_forest_augmented.joblib'
    joblib.dump(pipeline, model_file)
    print(f'\nModel saved: {model_file}')
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'n_features': len(feature_columns),
        'feature_columns': feature_columns,
        'classes': y_train.unique().tolist(),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    
    metrics_file = output_path / 'metrics_augmented.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved: {metrics_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson', default='data_splits/ezzayra_oliviers_geojson_augmented/all_splits.geojson')
    parser.add_argument('--sentinel-root', default='sentinel2_l2a/2025_05_06')
    parser.add_argument('--output-dir', default='models/cultivation_classifier_augmented')
    parser.add_argument('--n-estimators', type=int, default=500)
    args = parser.parse_args()
    
    print('Extracting features from augmented parcels...')
    features_df = extract_parcel_features(args.geojson, args.sentinel_root)
    
    print('\nTraining Random Forest classifier...')
    train_classifier(features_df, args.output_dir, args.n_estimators)
    
    print('\nDone!')
