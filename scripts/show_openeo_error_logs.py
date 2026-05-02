import argparse
import json
import re
import webbrowser
from pathlib import Path

import openeo


def parse_args():
    parser = argparse.ArgumentParser(description="Show logs for failed openEO jobs.")
    parser.add_argument("--manifest", default="openeo_jobs_manifest.json")
    parser.add_argument("--backend", default="openeofed.dataspace.copernicus.eu")
    parser.add_argument("--auth-timeout", type=int, default=900)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
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

    shown = 0
    for item in manifest:
        job = connection.job(item["job_id"])
        status = job.status()
        if status != "error":
            continue

        print("=" * 88)
        print(f"{item['job_id']} {item['title']}")
        print("-" * 88)
        for log in job.logs(level="error").logs():
            print(json.dumps(log, ensure_ascii=False, default=str))

        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()
