import argparse
import json
import re
import webbrowser
from pathlib import Path

import openeo
from openeo.rest import OpenEoApiError


def parse_args():
    parser = argparse.ArgumentParser(description="Delete openEO jobs listed in a manifest.")
    parser.add_argument("--manifest", default="openeo_jobs_manifest.json")
    parser.add_argument("--backend", default="openeofed.dataspace.copernicus.eu")
    parser.add_argument("--auth-timeout", type=int, default=900)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--statuses",
        default="finished,error,canceled",
        help="Comma-separated statuses to delete. Default: finished,error,canceled.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete jobs. Without this flag, only print what would be deleted.",
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


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    allowed_statuses = {status.strip() for status in args.statuses.split(",") if status.strip()}

    connection = openeo.connect(args.backend)
    connection.authenticate_oidc(
        max_poll_time=args.auth_timeout,
        display=make_auth_display(open_browser=not args.no_browser),
    )

    matched = 0
    deleted = 0
    for item in manifest:
        job = connection.job(item["job_id"])
        status = job.status()

        if status not in allowed_statuses:
            print(f"skip {item['job_id']} status={status} title={item['title']}")
            continue

        matched += 1
        if not args.yes:
            print(f"would delete {item['job_id']} status={status} title={item['title']}")
            continue

        try:
            job.delete()
            deleted += 1
            print(f"deleted {item['job_id']} status={status} title={item['title']}")
        except OpenEoApiError as exc:
            print(f"could not delete {item['job_id']}: {exc}")

    mode = "deleted" if args.yes else "matched"
    print(json.dumps({mode: deleted if args.yes else matched, "manifest": args.manifest}, indent=2))


if __name__ == "__main__":
    main()
