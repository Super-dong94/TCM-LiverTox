from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "data_manifest.json"
BLOCK_SIZE = 1024 * 1024


DATA_CATEGORIES = {
    "00_ctd_lookup.sqlite": "index_cache",
    "00_ctd_lookup.tmp.sqlite": "run_outputs",
    "08_target_disease.csv": "large_optional",
    "09_chem_diseases.csv": "large_optional",
    "10_chem_go.csv": "large_optional",
    "11_chem_pathways.csv": "large_optional",
}


def quick_hash(path: Path) -> str:
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(path.name.encode("utf-8", errors="ignore"))
    hasher.update(str(stat.st_size).encode("ascii"))
    hasher.update(str(stat.st_mtime_ns).encode("ascii"))
    with path.open("rb") as handle:
        hasher.update(handle.read(BLOCK_SIZE))
        if stat.st_size > BLOCK_SIZE:
            handle.seek(max(0, stat.st_size - BLOCK_SIZE))
            hasher.update(handle.read(BLOCK_SIZE))
    return hasher.hexdigest()


def category_for(path: Path) -> str:
    if path.name in DATA_CATEGORIES:
        return DATA_CATEGORIES[path.name]
    if path.suffix.lower() in {".sqlite", ".db"}:
        return "index_cache"
    return "core"


def main() -> None:
    files = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "file_name": path.name,
                "category": category_for(path),
                "source": "local",
                "schema_version": "2",
                "hash": quick_hash(path),
                "hash_mode": "sha256_name_size_mtime_head_tail",
                "size_bytes": stat.st_size,
                "generated_at": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(timespec="seconds"),
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "required": path.name != "00_ctd_lookup.tmp.sqlite",
                "license_note": "Local research data bundled with ToxHERB; verify upstream licensing before redistribution.",
            }
        )
    overall = hashlib.sha256(
        json.dumps([(item["file_name"], item["hash"]) for item in files], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash": overall,
        "hash_mode": "sha256_of_file_hashes",
        "method_version": "manuscript-20260922",
        "blood_exposure_rule": "Reference_Match OR Bioavailability_Ma >= 0.3",
        "endpoint_thresholds": {"cell": 0.55, "animal": 0.96, "clinical": 0.64},
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(MANIFEST)


if __name__ == "__main__":
    main()
