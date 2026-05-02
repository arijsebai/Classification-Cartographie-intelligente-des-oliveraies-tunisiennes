"""Split spatial des polygones EZZAYRA en train/val/test par gouvernorat.

Usage:
  python -m scripts.data_split --input data/ezzayra.json --out_dir data/splits --seed 42

Le script attend un GeoJSON FeatureCollection (ou une liste de features).
Il identifie la propriété du gouvernorat automatiquement (gouvernorat/governorat/governorate/gov)
et affecte des gouvernorats entiers à train/val/test (70/15/15) pour éviter la fuite spatiale.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


POSSIBLE_GOV_KEYS = ["gouvernorat", "gouvernorat_name", "gov", "governorat", "governorate", "governor", "region"]


def detect_gov_key(props: Dict) -> str | None:
    for k in POSSIBLE_GOV_KEYS:
        if k in props:
            return k
    # fallback: any key containing 'gov' or 'gouv' or 'gouver'
    for k in props.keys():
        lk = k.lower()
        if "gov" in lk or "gouv" in lk or "gouver" in lk:
            return k
    return None


def load_features(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        return data.get("features", [])
    if isinstance(data, list):
        return data
    # try single feature
    if isinstance(data, dict) and data.get("type") == "Feature":
        return [data]
    raise ValueError("Input JSON must be a FeatureCollection or list of features")


def group_by_governorat(features: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for feat in features:
        props = feat.get("properties", {}) or {}
        key = detect_gov_key(props)
        gov = None
        if key:
            gov = str(props.get(key) or "unknown").strip()
        # if still none, try to use admin tag
        if not gov:
            gov = "unknown"
        groups[gov].append(feat)
    return groups


def assign_groups(groups: Dict[str, List[Dict]], seed: int = 42) -> Dict[str, str]:
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    total = sum(len(v) for _, v in items)
    target_train = total * 0.7
    target_val = total * 0.85

    assign = {}
    c = 0
    for gov, feats in items:
        if c < target_train:
            assign[gov] = "train"
        elif c < target_val:
            assign[gov] = "val"
        else:
            assign[gov] = "test"
        c += len(feats)
    return assign


def write_split(assign: Dict[str, str], groups: Dict[str, List[Dict]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    parts = {"train": [], "val": [], "test": []}
    for gov, part in assign.items():
        parts[part].extend(groups[gov])

    for name, feats in parts.items():
        path = os.path.join(out_dir, f"{name}.geojson")
        fc = {"type": "FeatureCollection", "features": feats}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        print(f"Wrote {len(feats)} features to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input EZZAYRA GeoJSON")
    p.add_argument("--out_dir", default="data/splits", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    feats = load_features(args.input)
    print(f"Loaded {len(feats)} features from {args.input}")
    groups = group_by_governorat(feats)
    print(f"Detected {len(groups)} gouvernorats (groups)")
    assign = assign_groups(groups, seed=args.seed)
    write_split(assign, groups, args.out_dir)


if __name__ == "__main__":
    main()
