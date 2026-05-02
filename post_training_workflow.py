#!/usr/bin/env python
"""
Post-training workflow: Execute predictions and RF retraining after U-Net completes.
Run this script when training is done (best.pt exists).
"""

import subprocess
import sys
from pathlib import Path
import time

def wait_for_model(model_path, max_wait=300):
    """Wait for model checkpoint to exist."""
    print(f"Waiting for model checkpoint: {model_path}")
    start = time.time()
    while not Path(model_path).exists():
        if time.time() - start > max_wait:
            print(f"ERROR: Model not found after {max_wait}s")
            return False
        time.sleep(5)
        print(".", end="", flush=True)
    print(f"\n✓ Model checkpoint found!")
    return True

def run_command(cmd, description):
    """Run shell command with feedback."""
    print(f"\n{'='*70}")
    print(f"[{description}]")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ {description} failed!")
        return False
    
    print(f"✓ {description} completed!")
    return True

def main():
    print("="*70)
    print("POST-TRAINING WORKFLOW")
    print("="*70)
    
    model_checkpoint = "models/unet_resnet_sentinel2_augmented/best.pt"
    
    # Step 1: Wait for U-Net training to complete
    print("\n[Step 1] Waiting for U-Net training...")
    if not wait_for_model(model_checkpoint):
        sys.exit(1)
    
    # Step 2: Run U-Net predictions on test set
    print("\n[Step 2] Running U-Net predictions...")
    cmd = (
        "python training/predict_unet_smp.py "
        f"--checkpoint {model_checkpoint} "
        "--sentinel-root sentinel2_l2a/2025_05_06 "
        "--geojson-root data_splits/ezzayra_oliviers_geojson_augmented "
        "--output-dir predictions/unet_resnet_sentinel2_augmented "
        "--split test --save-masks"
    )
    if not run_command(cmd, "U-Net Predictions"):
        sys.exit(1)
    
    # Step 3: Run Random Forest retraining
    print("\n[Step 3] Training Random Forest classifier...")
    cmd = (
        "python train_rf_augmented.py "
        "--geojson data_splits/ezzayra_oliviers_geojson_augmented/all_splits.geojson "
        "--sentinel-root sentinel2_l2a/2025_05_06 "
        "--output-dir models/cultivation_classifier_augmented"
    )
    if not run_command(cmd, "Random Forest Training"):
        sys.exit(1)
    
    # Step 4: Validate all results
    print("\n[Step 4] Validating results...")
    cmd = "python validate_augmented_results.py"
    if not run_command(cmd, "Results Validation"):
        sys.exit(1)
    
    # Final summary
    print("\n" + "="*70)
    print("✓ ALL TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review metrics in:")
    print("   - predictions/unet_resnet_sentinel2_augmented/metrics.json")
    print("   - models/cultivation_classifier_augmented/metrics_augmented.json")
    print("2. Update demo app to use new models (optional)")
    print("3. Prepare jury presentation")

if __name__ == '__main__':
    main()
