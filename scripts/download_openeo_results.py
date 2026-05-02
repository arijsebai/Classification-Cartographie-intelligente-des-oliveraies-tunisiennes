import argparse
import json
import re
import webbrowser
from pathlib import Path

import openeo


def parse_args():
    parser = argparse.ArgumentParser(description="Download completed openEO Sentinel-2 parcel jobs.")
    parser.add_argument("--manifest", default="openeo_jobs_manifest.json")
    parser.add_argument("--output", default="sentinel2_l2a/2025_05_06")
    parser.add_argument("--backend", default="openeofed.dataspace.copernicus.eu")
    parser.add_argument("--auth-timeout", type=int, default=900)
    parser.add_argument("--auth-method", choices=["device", "auth-code"], default="device")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def safe_name(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


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


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    connection = openeo.connect(args.backend)
    authenticate(connection, args)

    for item in manifest:
        job = connection.job(item["job_id"])
        status = job.status()
        print(f"{item['job_id']} {status} {item['title']}")

        if status != "finished":
            continue

        split = safe_name(item["split"])
        system = safe_name(item["cultivation_system"])
        parcel_id = safe_name(item["id"])
        target_dir = Path(args.output) / split
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{system}__{parcel_id}.tif"

        if target_file.exists():
            print(f"Already exists: {target_file}")
            continue

        job.get_results().download_file(target_file)
        print(f"Downloaded: {target_file}")


if __name__ == "__main__":
    main()
