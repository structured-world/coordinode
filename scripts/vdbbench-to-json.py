#!/usr/bin/env python3
# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0.
"""Convert a VDBBench result JSON into the canonical bench-results schema.

VDBBench writes one JSON per task at
``vectordb_bench/results/CoordiNode/result_<date>_<task_id>_coordinode.json``
containing the full metric dictionary (qps, recall, p99 latency,
concurrent sweep, etc.). This script flattens those numbers into the
same per-(modality, dataset, subject) JSON that the docs / bench-data
branch consumes for the by-dimension chart.

Output: ``<out-dir>/vector/<dataset>/<sha>-coordinode-vdbbench-<ts>.json``

Each output entry follows the bench-results schema (v1):

    {
      "schema_version": 1,
      "timestamp":      "<ISO 8601>",
      "git":            { "sha": ..., "sha_short": ..., ... },
      "hardware":       { "cpu_brand": ..., "cpu_cores": ..., ... },
      "modality":       "vector",
      "benchmark":      "vdbbench",
      "dataset":        "openai-1536-50K",   # derived from case_id
      "subject":        "coordinode",
      "codec":          "none",
      "version":        "<engine version>",
      "metrics": {
        "dataset_dim":     1536,
        "dataset_n_train": 50000,
        "k":               100,
        "hnsw_m":          16,
        "hnsw_ef_construction": 200,
        "build_secs":      <insert_duration>,
        "recall_at_k":     <recall>,
        "ndcg":            <ndcg>,
        "qps":             <serial qps>,
        "latency_us_p99":  <serial p99 * 1e6>,
        "latency_us_p95":  <serial p95 * 1e6>,
        "concurrent_sweep": [
          {"workers": 1,  "qps": ..., "latency_us_p99": ..., ...},
          {"workers": 4,  ...},
          {"workers": 16, ...},
        ]
      }
    }
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# VDBBench's CaseType.value → (dim, n_train, dataset_name).
# Kept in sync with vectordb_bench/backend/cases.py — refresh when
# upstream introduces new cases.
CASE_TABLE: dict[int, tuple[int, int, str]] = {
    50: (1536, 50_000, "openai-1536-50K"),
    10: (1536, 500_000, "openai-1536-500K"),
    11: (1536, 5_000_000, "openai-1536-5M"),
    5:  (768, 1_000_000, "cohere-768-1M"),
    4:  (768, 10_000_000, "cohere-768-10M"),
    3:  (768, 100_000_000, "cohere-768-100M"),
    17: (1024, 1_000_000, "bioasq-1024-1M"),
    20: (1024, 10_000_000, "bioasq-1024-10M"),
}


def _git(repo: Path, *args: str) -> str:
    """Run ``git -C <repo> <args>`` and return stripped stdout."""
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _hardware() -> dict[str, object]:
    """Capture a small hardware fingerprint for the report.

    Best-effort — fields we can't read end up empty, not absent. Keeps
    the bench-data schema stable across hosts where /proc/cpuinfo or
    similar might differ.
    """
    cpu_brand = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_brand = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    ram_gb = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    ram_gb = kb // (1024 * 1024)
                    break
    except OSError:
        pass
    return {
        "cpu_brand":  cpu_brand,
        "cpu_cores":  os.cpu_count() or 0,
        "cpu_threads": os.cpu_count() or 0,
        "ram_gb":     ram_gb,
        "os_name":    platform.system(),
        "os_version": platform.release(),
        "arch":       platform.machine(),
    }


def _convert(src: Path, sha: str, repo: Path, version: str, out_dir: Path) -> Path:
    raw = json.loads(src.read_text(encoding="utf-8"))
    results = raw.get("results", [])
    if not results:
        raise ValueError(f"{src}: no results array")
    result = results[0]
    metrics = result["metrics"]
    task_config = result["task_config"]
    case_id = task_config["case_config"]["case_id"]
    if case_id not in CASE_TABLE:
        raise ValueError(
            f"{src}: unknown VDBBench case_id={case_id} — add to CASE_TABLE in scripts/vdbbench-to-json.py"
        )
    dim, n_train, dataset_name = CASE_TABLE[case_id]
    db_case = task_config["db_case_config"]

    # VDBBench timestamps are epoch seconds (float). Use UTC ISO.
    epoch = raw.get("timestamp")
    if isinstance(epoch, (int, float)):
        ts = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat()
    else:
        ts = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    stamp = ts.replace("-", "").replace(":", "").replace(".", "").split("+")[0][:15]

    sha_short = sha[:7]
    commit_date = ""
    try:
        commit_date = _git(repo, "log", "-1", "--format=%cI", sha)
    except subprocess.CalledProcessError:
        pass

    # Build concurrent_sweep entries.  VDBBench's three concurrency-lists
    # are aligned (i-th entry of each list belongs to the i-th workers
    # count from conc_num_list).
    conc_num = metrics.get("conc_num_list", []) or []
    conc_qps = metrics.get("conc_qps_list", []) or []
    conc_p99 = metrics.get("conc_latency_p99_list", []) or []
    conc_p95 = metrics.get("conc_latency_p95_list", []) or []
    conc_avg = metrics.get("conc_latency_avg_list", []) or []
    concurrent_sweep = [
        {
            "workers":        int(n),
            "qps":            float(q),
            "latency_us_p99": float(p99) * 1_000_000.0,
            "latency_us_p95": float(p95) * 1_000_000.0,
            "latency_us_avg": float(avg) * 1_000_000.0,
        }
        for n, q, p99, p95, avg in zip(conc_num, conc_qps, conc_p99, conc_p95, conc_avg)
    ]

    report = {
        "schema_version": 1,
        "timestamp":      ts,
        "git": {
            "sha":         sha,
            "sha_short":   sha_short,
            "branch":      "main",
            "dirty":       False,
            "commit_date": commit_date,
        },
        "hardware": _hardware(),
        "modality":   "vector",
        "benchmark":  "vdbbench",
        "dataset":    dataset_name,
        "subject":    "coordinode",
        "codec":      "none",
        "version":    version,
        "metrics": {
            "dataset_dim":          dim,
            "dataset_n_train":      n_train,
            "k":                    int(task_config["case_config"].get("k", 100)),
            "hnsw_m":               int(db_case.get("M", 0)),
            "hnsw_ef_construction": int(db_case.get("ef_construction", 0)),
            "build_secs":           float(metrics.get("insert_duration", 0.0)),
            "recall_at_k":          float(metrics.get("recall", 0.0)),
            "ndcg":                 float(metrics.get("ndcg", 0.0)),
            "qps":                  float(metrics.get("qps", 0.0)),
            "latency_us_p99":       float(metrics.get("serial_latency_p99", 0.0)) * 1_000_000.0,
            "latency_us_p95":       float(metrics.get("serial_latency_p95", 0.0)) * 1_000_000.0,
            "concurrent_sweep":     concurrent_sweep,
        },
    }

    dst_dir = out_dir / "vector" / dataset_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{sha_short}-coordinode-vdbbench-{stamp}.json"
    dst.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  required=True, type=Path,
                    help="VDBBench result.json (or directory of them)")
    ap.add_argument("--sha",  required=True, help="coordinode SHA")
    ap.add_argument("--coordinode-repo", required=True, type=Path,
                    help="coordinode checkout for git metadata")
    ap.add_argument("--version", default="HEAD", help="engine version label")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="where to write bench-results/vector/...")
    args = ap.parse_args()

    sources: list[Path]
    if args.src.is_dir():
        sources = sorted(args.src.glob("result_*_coordinode.json"))
    else:
        sources = [args.src]

    if not sources:
        print("no VDBBench results found", file=sys.stderr)
        return 1

    for src in sources:
        dst = _convert(src, args.sha, args.coordinode_repo, args.version, args.out_dir)
        print(f"  wrote {dst.relative_to(args.out_dir.parent)}")
    print(f"\nTotal: {len(sources)} bench-data JSON file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
