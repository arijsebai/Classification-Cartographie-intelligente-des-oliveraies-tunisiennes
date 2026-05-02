import argparse
import json
import re
import webbrowser
from pathlib import Path

import openeo
from openeo.rest import OpenEoApiError


def parse_args():
    parser = argparse.ArgumentParser(description="Start pending openEO jobs from openeo_jobs_manifest.json.")
    parser.add_argument("--manifest", default="openeo_jobs_manifest.json")
    parser.add_argument("--backend", default="openeofed.dataspace.copernicus.eu")
    parser.add_argument("--auth-timeout", type=int, default=900)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--max-start", type=int, default=30)
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
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    connection = openeo.connect(args.backend)
    connection.authenticate_oidc(
        max_poll_time=args.auth_timeout,
        display=make_auth_display(open_browser=not args.no_browser),
    )

    started_count = 0
    for item in manifest:
        if item.get("started"):
            continue
        if started_count >= args.max_start:
            break

        job = connection.job(item["job_id"])
        status = job.status()
        if status not in {"created", "queued"}:
            item["status"] = status
            continue

        try:
            job.start_job()
            item["started"] = True
            item["start_error"] = None
            item["status"] = "queued"
            started_count += 1
            print(f"Started {item['job_id']} {item['title']}")
        except OpenEoApiError as exc:
            item["start_error"] = str(exc)
            print(f"Could not start {item['job_id']}: {exc}")
            break

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Started {started_count} jobs. Updated {manifest_path}.")


if __name__ == "__main__":
    main()
