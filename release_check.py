from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
DATA_MANIFEST = ROOT / "data_manifest.json"
CALIBRATION_MANIFEST = ROOT / "models" / "calibration" / "calibration_manifest.json"
REPORT_PATH = ROOT / "build" / "release_report.json"
TEMP_PATTERNS = ("*.tmp.sqlite", "*.pyc", "*.pid", "*.err.log", "*.out.log")
WARN_SIZE_BYTES = 25 * 1024 * 1024 * 1024


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def top_files(path: Path, limit: int = 20) -> list[dict[str, object]]:
    if not path.exists():
        return []
    files = sorted((file for file in path.rglob("*") if file.is_file()), key=lambda item: item.stat().st_size, reverse=True)
    return [
        {
            "path": str(file.relative_to(ROOT)),
            "size_bytes": file.stat().st_size,
        }
        for file in files[:limit]
    ]


def load_json(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    data_manifest, data_manifest_error = load_json(DATA_MANIFEST)
    if data_manifest_error:
        failures.append(f"data_manifest.json {data_manifest_error}")
    elif not data_manifest.get("files"):
        failures.append("data_manifest.json has no files entries")

    calibration_manifest, calibration_error = load_json(CALIBRATION_MANIFEST)
    if calibration_error:
        failures.append(f"calibration_manifest.json {calibration_error}")
    elif calibration_manifest.get("calibration_status") == "missing":
        warnings.append("calibration models are explicitly missing; calibrated probabilities must remain null")

    temp_files = []
    for pattern in TEMP_PATTERNS:
        temp_files.extend(ROOT.rglob(pattern))
    temp_files = [path for path in temp_files if "dist" not in path.parts and "build" not in path.parts]
    if temp_files:
        warnings.append(f"temporary files in source tree: {len(temp_files)}")

    dist_size = folder_size(DIST_DIR)
    if dist_size > WARN_SIZE_BYTES:
        warnings.append(f"dist folder is larger than {WARN_SIZE_BYTES} bytes")

    report = {
        "ok": not failures,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(ROOT),
        "dist_size_bytes": dist_size,
        "top20_project_files": top_files(ROOT),
        "top20_dist_files": top_files(DIST_DIR),
        "failures": failures,
        "warnings": warnings,
        "data_manifest": {
            "path": str(DATA_MANIFEST),
            "file_count": len(data_manifest.get("files", [])) if data_manifest else 0,
            "hash": data_manifest.get("hash") if data_manifest else None,
        },
        "calibration_manifest": calibration_manifest,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failures": failures, "warnings": warnings, "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
