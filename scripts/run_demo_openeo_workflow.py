import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True)
    print(f"Exit code: {completed.returncode}", flush=True)
    return completed.returncode


def submission_path(job_id: str) -> Path:
    return ROOT / "demo_app" / "submissions" / f"{job_id}.json"


def load_submission(job_id: str) -> dict:
    return json.loads(submission_path(job_id).read_text(encoding="utf-8"))


def save_submission(job_id: str, data: dict) -> None:
    submission_path(job_id).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def expected_download_path(submission: dict) -> Path:
    split = submission.get("split") or "demo"
    cultivation_system = submission.get("cultivation_system") or "unknown"
    parcel_id = submission.get("id") or submission.get("job_id")
    return ROOT / "sentinel2_l2a" / "2025_05_06" / split / f"{cultivation_system}__{parcel_id}.tif"


def classify_downloaded_submission(job_id: str) -> dict | None:
    submission = load_submission(job_id)
    if submission.get("analysis_result"):
        return submission["analysis_result"]

    target_file = expected_download_path(submission)
    if not target_file.exists():
        return None

    payload = json.dumps({"geometry": submission["geometry"]})
    completed = subprocess.run(
        [sys.executable, "scripts/classify_polygon_local.py"],
        cwd=ROOT,
        input=payload,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "Local Sentinel-2 classification failed.").strip())

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON from local classification: {completed.stdout[:500]}") from error

    if "error" in result:
        error = result["error"]
        raise RuntimeError(str(error.get("detail") or error.get("message") or error))

    analysis_result = result["classification"]
    submission.update(
        {
            "status": "completed",
            "analysis_status": "completed",
            "analysis_completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "analysis_result": analysis_result,
            "result_source": "openEO_downloaded_geotiff_local_random_forest",
            "result_file": str(target_file.relative_to(ROOT)),
        }
    )
    save_submission(job_id, submission)
    return analysis_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the demo openEO workflow for one submitted polygon.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-download-attempts", type=int, default=20)
    args = parser.parse_args()

    geojson = Path("demo_app") / "submissions" / f"{args.job_id}.geojson"
    manifest = Path(f"openeo_jobs_manifest_{args.job_id}.json")

    steps = [
        [
            sys.executable,
            "scripts/export_demo_submissions_geojson.py",
            "--job-id",
            args.job_id,
            "--output",
            str(geojson),
        ],
        [
            sys.executable,
            "scripts/download_sentinel2_openeo.py",
            "--input-geojson",
            str(geojson),
            "--manifest",
            str(manifest),
            "--start-jobs",
            "--auth-method",
            "device",
            "--auth-timeout",
            "1800",
            "--no-browser",
        ],
    ]

    for command in steps:
        code = run_step(command)
        if code != 0:
            return code

    download_command = [
        sys.executable,
        "scripts/download_openeo_results.py",
        "--manifest",
        str(manifest),
        "--output",
        "sentinel2_l2a/2025_05_06",
        "--auth-method",
        "device",
        "--auth-timeout",
        "1800",
        "--no-browser",
    ]

    for attempt in range(1, args.max_download_attempts + 1):
        print(f"\nDownload attempt {attempt}/{args.max_download_attempts}", flush=True)
        code = run_step(download_command)
        if code != 0:
            return code

        try:
            analysis_result = classify_downloaded_submission(args.job_id)
        except Exception as error:
            print(f"Classification pending or failed: {error}", flush=True)
            analysis_result = None

        if analysis_result is not None:
            print(f"Classification completed: {json.dumps(analysis_result, ensure_ascii=False)}", flush=True)
            print("Workflow finished successfully.", flush=True)
            return 0

        print(
            "If the job is not finished yet, the downloader skips it. "
            f"Next check in {args.poll_seconds} seconds.",
            flush=True,
        )
        if attempt < args.max_download_attempts:
            time.sleep(args.poll_seconds)

    submission = load_submission(args.job_id)
    submission.update(
        {
            "analysis_status": "timeout",
            "analysis_error": "The openEO download finished polling before a GeoTIFF was available for local classification.",
        }
    )
    save_submission(args.job_id, submission)
    print("\nWorkflow finished polling without a final classification. Relance Analyse automatique dans la demo.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
