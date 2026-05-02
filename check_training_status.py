#!/usr/bin/env python
"""
Validate re-trained models with augmented dataset.
"""

import json
from pathlib import Path

# Check if training completed
model_dir = Path('models/unet_resnet_sentinel2_augmented')
print(f'Checking {model_dir}...')

if not model_dir.exists():
    print('ERROR: Model directory not found!')
    exit(1)

# Check for checkpoints
best_pt = model_dir / 'best.pt'
last_pt = model_dir / 'last.pt'
history_json = model_dir / 'history.json'

print(f'✓ best.pt exists: {best_pt.exists()}')
print(f'✓ last.pt exists: {last_pt.exists()}')
print(f'✓ history.json exists: {history_json.exists()}')

if history_json.exists():
    with open(history_json, 'r') as f:
        history = json.load(f)
    
    print(f'\nTraining History:')
    print(f'  Epochs trained: {len(history)}')
    
    if history:
        last_epoch = history[-1]
        print(f'  Last epoch metrics:')
        print(f'    Train loss: {last_epoch["train"]["loss"]:.4f}')
        print(f'    Train Dice: {last_epoch["train"]["dice"]:.4f}')
        print(f'    Val loss: {last_epoch["val"]["loss"]:.4f}')
        print(f'    Val Dice: {last_epoch["val"]["dice"]:.4f}')
        print(f'    Val IoU: {last_epoch["val"]["iou"]:.6f}')

print('\nNext steps:')
print('1. Run predictions: python training/predict_unet_smp.py --model-dir models/unet_resnet_sentinel2_augmented')
print('2. Validate metrics vs jury criteria')
print('3. Update demo with new model if metrics are good')
