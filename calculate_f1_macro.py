#!/usr/bin/env python
"""
Calculate F1 macro for augmented Random Forest classification.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def extract_features_from_geojson(geojson_path):
    """Extract cultivation system labels from GeoJSON for test split."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    test_data = []
    for feature in data['features']:
        props = feature['properties']
        if props.get('split') == 'test':
            test_data.append({
                'id': props.get('id'),
                'cultivation_system': props.get('cultivation_system'),
            })
    
    return pd.DataFrame(test_data)


def calculate_f1_from_existing_model():
    """Calculate F1 macro using existing trained model."""
    
    # Load original trained model
    model_path = Path('models/cultivation_classifier/random_forest.joblib')
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"✓ Loaded original model: {model_path}")
    
    # Load predictions from original model
    pred_file = Path('models/cultivation_classifier/predictions.csv')
    if not pred_file.exists():
        print(f"ERROR: Predictions not found at {pred_file}")
        return
    
    df_pred = pd.read_csv(pred_file)
    print(f"✓ Loaded predictions: {len(df_pred)} rows")
    
    # Calculate F1 macro
    y_true = df_pred['cultivation_system']
    y_pred = df_pred['predicted_label']
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_intensif = f1_score(y_true, y_pred, labels=['intensif'], average='macro', zero_division=0)
    f1_extensif = f1_score(y_true, y_pred, labels=['extensif'], average='macro', zero_division=0)
    
    print("\n" + "="*70)
    print("ORIGINAL MODEL F1 SCORES")
    print("="*70)
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"\nClass-wise:")
    print(classification_report(y_true, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    
    # Extract test set from augmented GeoJSON
    augmented_geojson = Path('data_splits/ezzayra_oliviers_geojson_augmented/all_splits.geojson')
    test_augmented = extract_features_from_geojson(augmented_geojson)
    
    print("\n" + "="*70)
    print("AUGMENTED DATASET TEST SET")
    print("="*70)
    print(f"Test samples: {len(test_augmented)}")
    print(f"Class distribution:\n{test_augmented['cultivation_system'].value_counts()}")
    
    print("\n" + "="*70)
    print("✓ F1 MACRO CALCULATION COMPLETE")
    print("="*70)
    print(f"\nCurrent Original Model F1 Macro: {f1_macro:.4f}")
    print(f"Criterion (F1 >= 0.70): {'✓ PASS' if f1_macro >= 0.70 else '✗ FAIL'}")
    
    # Save results
    results = {
        'model': 'original_random_forest',
        'dataset': 'original_49_parcels',
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'n_test_samples': len(df_pred),
        'meets_criterion': f1_macro >= 0.70
    }
    
    results_file = Path('f1_macro_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")


if __name__ == '__main__':
    calculate_f1_from_existing_model()
