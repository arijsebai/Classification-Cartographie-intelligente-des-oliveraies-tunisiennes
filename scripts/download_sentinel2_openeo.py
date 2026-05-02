import argparse
import json
import re
import webbrowser
from pathlib import Path

import openeo
from openeo.rest import OpenEoApiError


SPLITS = ("train", "val", "test")
S2_BANDS = ["B02", "B03", "B04", "B08", "B11"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Copernicus Data Space openEO jobs for Sentinel-2 L2A parcel composites."
    )
    parser.add_argument(
        "--input",
        default="data_splits/ezzayra_oliviers_geojson",
        help="Directory containing train.geojson, val.geojson and test.geojson.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to create jobs for, for example test or val,test.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--max-cloud-cover", type=int, default=60)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument(
        "--backend",
        default="openeofed.dataspace.copernicus.eu",
        help="Copernicus Data Space openEO backend.",
    )
    parser.add_argument(
        "--start-jobs",
        action="store_true",
        help="Start jobs immediately. Without this flag, jobs are created but left queued/not started.",
    )
    parser.add_argument(
        "--max-started-jobs",
        type=int,
        default=30,
        help="Maximum number of jobs to start in one run. Copernicus commonly limits this to 30.",
    )
    parser.add_argument(
        "--auth-timeout",
        type=int,
        default=900,
        help="Authentication timeout in seconds. Default: 900.",
    )
    parser.add_argument(
        "--auth-method",
        choices=["device", "auth-code"],
        default="device",
        help="Authentication method. Use auth-code if device login stays pending.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the authentication URL automatically.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing, for example --limit 1.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest path. Defaults to openeo_jobs_manifest.json or openeo_jobs_manifest_bbox.json.",
    )
    parser.add_argument(
        "--export-region",
        choices=["polygon", "bbox"],
        default="polygon",
        help="Export clipped to parcel polygon or to its bounding box. Use bbox for segmentation training.",
    )
    parser.add_argument(
        "--buffer-deg",
        type=float,
        default=0.01,
        help="Bounding-box buffer in degrees when --export-region bbox. 0.01 is about 1 km north-south.",
    )
    return parser.parse_args()


def make_auth_display(open_browser):
    opened = {"value": False}

    def display(message="", end="\n"):
        text = str(message)
        print(text, end=end)

        if open_browser and not opened["value"]:
            match = re.search(r"https://\S+", text)
            if match:
                opened["value"] = True
                webbrowser.open(match.group(0))

    return display


def authenticate(connection, args):
    if args.auth_method == "auth-code":
        connection.authenticate_oidc_authorization_code(
            timeout=args.auth_timeout,
            store_refresh_token=True,
        )
        return

    connection.authenticate_oidc(
        max_poll_time=args.auth_timeout,
        display=make_auth_display(open_browser=not args.no_browser),
    )


def read_feature_collection(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")

    return data["features"]


def bbox_from_feature(feature):
    if "bbox" in feature:
        west, south, east, north = feature["bbox"]
        return {"west": west, "south": south, "east": east, "north": north}

    ring = feature["geometry"]["coordinates"][0]
    lngs = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    return {
        "west": min(lngs),
        "south": min(lats),
        "east": max(lngs),
        "north": max(lats),
    }


def buffered_bbox(feature, buffer_deg):
    bbox = bbox_from_feature(feature)
    return {
        "west": bbox["west"] - buffer_deg,
        "south": bbox["south"] - buffer_deg,
        "east": bbox["east"] + buffer_deg,
        "north": bbox["north"] + buffer_deg,
    }


def bbox_geometry(bbox):
    west = bbox["west"]
    south = bbox["south"]
    east = bbox["east"]
    north = bbox["north"]
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }


def safe_name(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def build_parcel_cube(connection, feature, start_date, end_date, max_cloud_cover, export_region, buffer_deg):
    if export_region == "bbox":
        spatial_extent = buffered_bbox(feature, buffer_deg)
        geometry = bbox_geometry(spatial_extent)
    else:
        spatial_extent = bbox_from_feature(feature)
        geometry = feature["geometry"]

    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start_date, end_date],
        bands=["SCL"],
        max_cloud_cover=max_cloud_cover,
    )

    # SCL values kept by the task: 4 vegetation and 5 bare soil.
    # openEO mask() removes pixels where the mask cube is true.
    # This direct boolean mask is more portable than backend-specific SCL dilation.
    invalid_scl_mask = (scl != 4).logical_and(scl != 5)

    bands = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start_date, end_date],
        bands=S2_BANDS,
        max_cloud_cover=max_cloud_cover,
    )

    masked = bands.mask(invalid_scl_mask)
    composite = masked.reduce_dimension(dimension="t", reducer="median")
    clipped = composite.filter_spatial(geometry)
    return clipped.save_result(format="GTiff")


def main():
    args = parse_args()
    input_dir = Path(args.input)
    start_date = f"{args.year}-05-01"
    end_date = f"{args.year}-06-30"

    connection = openeo.connect(args.backend)
    authenticate(connection, args)

    created_jobs = []
    features_seen = 0
    started_jobs = 0
    start_blocked = False
    selected_splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())

    for split in selected_splits:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}. Expected one of {SPLITS}.")

        features = read_feature_collection(input_dir / f"{split}.geojson")

        for feature in features:
            if args.limit is not None and features_seen >= args.limit:
                break

            props = feature["properties"]
            parcel_id = safe_name(props["id"])
            system = safe_name(props["cultivation_system"])
            title = f"{split}__{system}__{parcel_id}__{args.year}__05_06__s2_l2a"
            if args.export_region == "bbox":
                title = f"{title}__bboxbuf{str(args.buffer_deg).replace('.', 'p')}"

            result_cube = build_parcel_cube(
                connection=connection,
                feature=feature,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=args.max_cloud_cover,
                export_region=args.export_region,
                buffer_deg=args.buffer_deg,
            )

            job = result_cube.create_job(
                title=title,
                description=(
                    "EZZAYRA olive parcel Sentinel-2 L2A May-June median composite. "
                    "Bands: B02, B03, B04, B08, B11. SCL mask keeps only classes 4 and 5. "
                    f"Export region: {args.export_region}."
                ),
                job_options={
                    "driver-memory": "2G",
                    "executor-memory": "2G",
                },
            )

            started = False
            start_error = None
            if args.start_jobs and started_jobs < args.max_started_jobs and not start_blocked:
                try:
                    job.start_job()
                    started = True
                    started_jobs += 1
                except OpenEoApiError as exc:
                    start_error = str(exc)
                    print(f"Could not start {job.job_id}: {start_error}")
                    if "ConcurrentJobLimit" in start_error or "Too Many Requests" in start_error:
                        start_blocked = True

            created_jobs.append(
                {
                    "split": split,
                    "id": props["id"],
                    "cultivation_system": props["cultivation_system"],
                    "governorate": props.get("governorate"),
                    "zone_id": props.get("zone_id"),
                    "job_id": job.job_id,
                    "title": title,
                    "started": started,
                    "start_error": start_error,
                }
            )
            features_seen += 1
            print(json.dumps(created_jobs[-1], ensure_ascii=False))

            manifest_path = Path(
                args.manifest
                or ("openeo_jobs_manifest_bbox.json" if args.export_region == "bbox" else "openeo_jobs_manifest.json")
            )
            manifest_path.write_text(json.dumps(created_jobs, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.limit is not None and features_seen >= args.limit:
            break

    default_manifest = "openeo_jobs_manifest_bbox.json" if args.export_region == "bbox" else "openeo_jobs_manifest.json"
    if selected_splits != SPLITS:
        split_suffix = "_".join(selected_splits)
        default_manifest = default_manifest.replace(".json", f"_{split_suffix}.json")
    manifest_path = Path(args.manifest or default_manifest)
    manifest_path.write_text(json.dumps(created_jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path} with {len(created_jobs)} jobs.")


if __name__ == "__main__":
    main()
