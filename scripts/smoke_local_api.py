from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx


CASES: dict[str, tuple[str, str]] = {
    "compound_3220": ("compound", "3220"),
    "herb_ejiao": ("herb", "\u963f\u80f6"),
    "herb_heshouwu": ("herb", "\u4f55\u9996\u4e4c"),
    "formula_shengxuebao": ("formula", "\u751f\u8840\u5b9d\u5408\u5242"),
}


def poll_job(client: httpx.Client, job_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").raise_for_status().json()
        print(f"{job_id} {job['status']} {job['stage']} {job['progress']}% {job.get('message', '')}")
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(5)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_seconds} seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local ToxHERB API smoke checks with UTF-8 JSON and no env proxy.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--case", choices=sorted(CASES), action="append", help="Case to run. Repeatable. Defaults to all cases.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    cases = args.case or list(CASES)
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        health = client.get("/health").raise_for_status().json()
        print(f"health ok: version={health.get('version', 'unknown')}")
        for case_name in cases:
            query_type, text = CASES[case_name]
            created = client.post(f"/api/jobs/predict/{query_type}", json={"text": text}).raise_for_status().json()
            job = poll_job(client, created["job_id"], args.timeout_seconds)
            if job["status"] != "succeeded":
                print(job, file=sys.stderr)
                return 1
            result = client.get(f"/api/jobs/{job['job_id']}/result").raise_for_status().json()
            summary = result.get("summary", {})
            print(
                f"{case_name} ok: risk={summary.get('risk_label')} "
                f"raw={summary.get('raw_max_toxic_probability')} "
                f"charts={len(result.get('chart_specs', []))}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
