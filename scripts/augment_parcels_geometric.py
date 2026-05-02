#!/usr/bin/env python
"""
Data Augmentation: Generate geometric variants of existing parcels
by perturbing coordinates slightly (±5% jitter on each vertex).
Keeps the same label (intensif/extensif) and properties.
"""

import argparse
import json
import random
from pathlib import Path


def perturb_coordinate(coord, max_jitter=0.0005):
    """
    Perturb a single [lng, lat] coordinate with small random noise.
    max_jitter ≈ 0.0005 degrees ≈ 55m at equator.
    """
    lng, lat = coord
    jitter_lng = random.uniform(-max_jitter, max_jitter)
    jitter_lat = random.uniform(-max_jitter, max_jitter)
    return [lng + jitter_lng, lat + jitter_lat]


def perturb_ring(ring, max_jitter=0.0005):
    """Perturb all coordinates in a ring (keep closure)."""
    if len(ring) < 2:
        return ring
    
    perturbed = [perturb_coordinate(coord, max_jitter) for coord in ring[:-1]]
    perturbed.append(perturbed[0])  # Close the ring
    return perturbed


def perturb_geometry(geometry, max_jitter=0.0005):
    """Perturb coordinates in a Polygon geometry."""
    if geometry["type"] != "Polygon":
        return geometry
    
    coordinates = geometry["coordinates"]
    perturbed_coords = []
    
    for ring in coordinates:
        perturbed_coords.append(perturb_ring(ring, max_jitter))
    
    return {
        "type": "Polygon",
        "coordinates": perturbed_coords,
    }


def augment_parcels(input_path, output_path, variants_per_parcel=4, max_jitter=0.0005, seed=42):
    """
    Load parcels GeoJSON, generate variants, save augmented GeoJSON.
    
    Args:
        input_path: Path to input GeoJSON (train.geojson, val.geojson, or all_splits.geojson)
        output_path: Path to save augmented GeoJSON
        variants_per_parcel: Number of variants per original parcel (3-5 recommended)
        max_jitter: Max perturbation in degrees (0.0005 ≈ 55m)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_features = data.get("features", [])
    print(f"Loaded {len(original_features)} original parcels")
    
    augmented_features = list(original_features)  # Start with originals
    variant_count = 0
    
    for original_feature in original_features:
        props = original_feature.get("properties", {})
        original_id = props.get("id", "unknown")
        
        for variant_idx in range(1, variants_per_parcel + 1):
            # Create new feature with perturbed geometry
            variant_feature = {
                "type": "Feature",
                "id": f"{original_id}_aug{variant_idx}",
                "properties": dict(props),  # Copy properties
                "geometry": perturb_geometry(original_feature.get("geometry", {}), max_jitter),
            }
            
            # Mark as augmented in properties
            variant_feature["properties"]["id"] = f"{original_id}_aug{variant_idx}"
            variant_feature["properties"]["original_id"] = original_id
            variant_feature["properties"]["augmented"] = True
            variant_feature["properties"]["variant_index"] = variant_idx
            
            augmented_features.append(variant_feature)
            variant_count += 1
    
    # Create output GeoJSON
    output_data = {
        "type": "FeatureCollection",
        "generated_at": data.get("generated_at", "augmented"),
        "original_count": len(original_features),
        "variant_count": variant_count,
        "total_count": len(augmented_features),
        "augmentation_method": "geometric_perturbation",
        "max_jitter_degrees": max_jitter,
        "variants_per_parcel": variants_per_parcel,
        "features": augmented_features,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Augmentation complete!")
    print(f"  Original: {len(original_features)} parcels")
    print(f"  Variants: {variant_count} ({variants_per_parcel} per parcel)")
    print(f"  Total:    {len(augmented_features)} parcels (×{len(augmented_features)/len(original_features):.1f})")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment parcels GeoJSON by geometric perturbation"
    )
    parser.add_argument(
        "--input",
        default="data_splits/ezzayra_oliviers_geojson/all_splits.geojson",
        help="Input GeoJSON path"
    )
    parser.add_argument(
        "--output",
        default="data_splits/ezzayra_oliviers_geojson/all_splits_augmented.geojson",
        help="Output augmented GeoJSON path"
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=4,
        help="Number of variants per original parcel (default: 4)"
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0005,
        help="Max coordinate perturbation in degrees (default: 0.0005 ≈ 55m)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    augment_parcels(
        input_path,
        output_path,
        variants_per_parcel=args.variants,
        max_jitter=args.jitter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
