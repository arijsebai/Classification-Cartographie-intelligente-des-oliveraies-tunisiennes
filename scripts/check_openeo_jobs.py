import argparse
import json
import re
import webbrowser
from collections import Counter
from pathlib import Path

import openeo


def parse_args():
    parser = argparse.ArgumentParser(description="Check openEO job statuses from openeo_jobs_manifest.json.")
    parser.add_argument("--manifest", default="openeo_jobs_manifest.json")
    parser.add_argument("--backend", default="openeofed.dataspace.copernicus.eu")
    parser.add_argument("--auth-timeout", type=int, default=900)
    parser.add_argument("--no-browser", action="store_true")
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

    connection = openeo.connect(args.backend)
    connection.authenticate_oidc(
        max_poll_time=args.auth_timeout,
        display=make_auth_display(open_browser=not args.no_browser),
    )

    rows = []
    for item in manifest:
        job = connection.job(item["job_id"])
        status = job.status()
        row = {
            "job_id": item["job_id"],
            "split": item["split"],
            "id": item["id"],
            "cultivation_system": item["cultivation_system"],
            "status": status,
            "title": item["title"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    status_counts = Counter(row["status"] for row in rows)
    Path("openeo_jobs_status.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Summary:", json.dumps(dict(status_counts), ensure_ascii=False))
    print("Wrote openeo_jobs_status.json")


if __name__ == "__main__":
    main()
