import argparse
import os
import shutil


def load_ids(geojson_path: str) -> set[str]:
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    if "id" not in gdf.columns:
        raise ValueError(f"Missing 'id' column in {geojson_path}")
    return set(map(str, gdf["id"].tolist()))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_parcel_dir(src_dir: str, dst_dir: str, dry_run: bool) -> bool:
    if not os.path.isdir(src_dir):
        return False
    if os.path.exists(dst_dir):
        # Already organized (or conflict). Skip to avoid overwriting.
        return False
    if dry_run:
        return True
    ensure_dir(os.path.dirname(dst_dir))
    shutil.move(src_dir, dst_dir)
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="Organize sentinel2_data/<parcel_id> into train/val/test subfolders using split_data/*.geojson"
    )
    p.add_argument("--sentinel2-dir", default="sentinel2_data", help="Base directory containing parcel folders")
    p.add_argument("--split-dir", default="split_data", help="Directory containing train/val/test geojson files")
    p.add_argument("--dry-run", action="store_true", help="Print what would be moved, without moving")
    args = p.parse_args()

    sentinel2_dir = args.sentinel2_dir
    split_dir = args.split_dir

    splits = {
        "train": os.path.join(split_dir, "train.geojson"),
        "val": os.path.join(split_dir, "val.geojson"),
        "test": os.path.join(split_dir, "test.geojson"),
    }

    ids_by_split: dict[str, set[str]] = {}
    for split_name, path in splits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")
        ids_by_split[split_name] = load_ids(path)

    all_ids = set().union(*ids_by_split.values())
    # Ensure split directories exist even if nothing is moved yet
    for split_name in ids_by_split.keys():
        ensure_dir(os.path.join(sentinel2_dir, split_name))

    moved = 0
    skipped_missing = 0
    skipped_existing = 0

    for split_name, ids in ids_by_split.items():
        for parcel_id in sorted(ids):
            src = os.path.join(sentinel2_dir, parcel_id)
            dst = os.path.join(sentinel2_dir, split_name, parcel_id)
            ok = move_parcel_dir(src, dst, args.dry_run)
            if ok:
                moved += 1
                if args.dry_run:
                    print(f"WOULD MOVE: {src} -> {dst}")
                else:
                    print(f"MOVED: {src} -> {dst}")
            else:
                if os.path.isdir(dst):
                    skipped_existing += 1
                elif not os.path.isdir(src):
                    skipped_missing += 1

    # Any parcel folders not present in splits are left untouched
    unassigned = []
    if os.path.isdir(sentinel2_dir):
        for name in os.listdir(sentinel2_dir):
            full = os.path.join(sentinel2_dir, name)
            if os.path.isdir(full) and name not in ("train", "val", "test") and name not in all_ids:
                unassigned.append(name)

    print(
        f"\nSummary: moved={moved}, skipped_existing={skipped_existing}, skipped_missing_src={skipped_missing}, unassigned_dirs={len(unassigned)}"
    )
    if unassigned:
        print("Unassigned (left in place):")
        for x in sorted(unassigned)[:50]:
            print(f"- {x}")
        if len(unassigned) > 50:
            print(f"... and {len(unassigned) - 50} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

