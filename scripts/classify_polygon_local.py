import json
import sys
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_app.main import extract_local_sentinel_features, polygon_area_m2


def main() -> int:
    payload = json.loads(sys.stdin.read())
    geometry = payload["geometry"]
    area_ha = polygon_area_m2(geometry) / 10_000.0

    try:
        classification = extract_local_sentinel_features(geometry, area_ha)
    except HTTPException as error:
        print(json.dumps({"error": {"status_code": error.status_code, "detail": error.detail}}))
        return 0

    print(json.dumps({"classification": classification}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
