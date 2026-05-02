import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "demo_app" / "submissions"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "submissions.geojson"


def load_submission(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "type": "Feature",
        "properties": {
            "job_id": data.get("job_id"),
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "area_km2": data.get("area_km2"),
            "area_ha": data.get("area_ha"),
            "centroid_lat": (data.get("centroid") or {}).get("lat"),
            "centroid_lng": (data.get("centroid") or {}).get("lng"),
        },
        "geometry": data["geometry"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export demo Sentinel-2 submissions to a GeoJSON FeatureCollection."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--job-id",
        default=None,
        help="Export only one demo submission, for example demo_1777742534_27532.",
    )
    args = parser.parse_args()

    features = []
    if args.input_dir.exists():
        pattern = f"{args.job_id}.json" if args.job_id else "demo_*.json"
        for path in sorted(args.input_dir.glob(pattern)):
            features.append(load_submission(path))

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(args.output), "count": len(features)}, indent=2))


if __name__ == "__main__":
    main()
