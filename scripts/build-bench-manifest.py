#!/usr/bin/env python3
"""Build bench manifest + mirror bench-results/ into docs/public/.

Run during docs:build (and CI). Walks `bench-results/<modality>/
<dataset>/*.json`, emits `docs/public/bench-data/index.json` (flat
list of every BenchReport with summary fields), and mirrors the raw
JSON files under `docs/public/bench-results/` so the in-browser
Vue component can fetch them via fetch().

Output structure (under docs/public/):

  bench-data/index.json                          ← manifest
  bench-results/<modality>/<dataset>/*.json       ← raw reports

Schema of manifest entries — kept narrow on purpose; the dashboard
loads individual JSONs lazily for sweep details.

  {
    "schema_version": 1,
    "entries": [
      {
        "path": "bench-results/vector/.../abc-coordinode-20260521.json",
        "modality": "vector",
        "dataset": "sift-128-euclidean",
        "subject": "coordinode",
        "sha": "abc1234...",
        "sha_short": "abc1234",
        "timestamp": "2026-05-21T22:25:37Z"
      }
    ]
  }
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_RESULTS = REPO_ROOT / "bench-results"
DOCS_PUBLIC = REPO_ROOT / "docs" / "public"
MANIFEST_DIR = DOCS_PUBLIC / "bench-data"
MIRROR_DIR = DOCS_PUBLIC / "bench-results"


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe + recreate mirror so deleted bench JSONs don't linger.
    if MIRROR_DIR.exists():
        shutil.rmtree(MIRROR_DIR)
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    if BENCH_RESULTS.exists():
        for src in BENCH_RESULTS.rglob("*.json"):
            # Skip README.md / .gitkeep / non-report files just in case.
            if src.name.endswith(".json") is False:
                continue
            try:
                data = json.loads(src.read_text())
            except Exception as e:  # noqa: BLE001
                print(f"::warning::skipping unreadable {src}: {e}", file=sys.stderr)
                continue
            if not isinstance(data, dict) or "schema_version" not in data:
                continue
            rel = src.relative_to(REPO_ROOT).as_posix()
            # Mirror the file into docs/public/<rel> so fetch resolves.
            mirror_target = DOCS_PUBLIC / rel
            mirror_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, mirror_target)
            git = data.get("git") or {}
            entries.append(
                {
                    "path": rel,
                    "modality": data.get("modality"),
                    "dataset": data.get("dataset"),
                    "subject": data.get("subject"),
                    "sha": git.get("sha"),
                    "sha_short": git.get("sha_short"),
                    "timestamp": data.get("timestamp"),
                }
            )

    # Stable ordering — same input set → same manifest bytes
    # (so docs build is reproducible / cacheable).
    entries.sort(key=lambda e: (e.get("modality") or "", e.get("dataset") or "", e.get("timestamp") or "", e.get("subject") or ""))

    manifest = {"schema_version": 1, "entries": entries}
    out = MANIFEST_DIR / "index.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"build-bench-manifest: {len(entries)} entries → {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
