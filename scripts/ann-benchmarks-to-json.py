#!/usr/bin/env python3
# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0.
"""Flatten ann-benchmarks HDF5 results into the canonical bench-results JSON.

ann-benchmarks writes one HDF5 per (algorithm, dataset, M, ef_construction,
ef_search) tuple.  This script groups them into the schema CoordiNode
already uses on the ``bench-data`` branch — one JSON file per
(dataset, M) pair, with the ef sweep collapsed into the ``sweep`` array:

    bench-results/vector/<dataset>/<sha>-<subject>-<ts>.json

Each JSON looks like::

    {
      "schema_version": 1,
      "timestamp": "2026-05-22T23:00:00Z",
      "git": { "sha": "...", "sha_short": "...", "branch": "main",
               "dirty": false, "commit_date": "..." },
      "hardware": { "cpu_brand": "...", "cpu_cores": ..., ... },
      "modality": "vector",
      "benchmark": "ann-benchmarks",
      "dataset": "sift-128-euclidean",
      "subject": "coordinode",          # or hnswlib / faiss-hnsw / qdrant / ...
      "codec": "none",
      "version": "0.4.3",
      "metrics": {
        "build_secs": 22.3,
        "dataset_dim": 128,
        "dataset_n_test": 10000,
        "dataset_n_train": 1000000,
        "hnsw_ef_construction": 500,
        "hnsw_m": 16,
        "k": 10,
        "qps_at_recall_peak": ...,
        "recall_at_k_peak": ...,
        "sweep": [ {"ef_search": 10, "qps": ..., "recall_at_k": ..., "latency_us_mean": ...}, ...]
      }
    }
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DATASET_META = {
    # name -> (dim, n_train, n_test, distance, codec_default)
    "glove-25-angular":            (25,  1_183_514, 10_000, "angular",   "none"),
    "glove-50-angular":            (50,  1_183_514, 10_000, "angular",   "none"),
    "glove-100-angular":           (100, 1_183_514, 10_000, "angular",   "none"),
    "glove-200-angular":           (200, 1_183_514, 10_000, "angular",   "none"),
    "sift-128-euclidean":          (128, 1_000_000, 10_000, "euclidean", "none"),
    "nytimes-256-angular":         (256,   290_000, 10_000, "angular",   "none"),
    "fashion-mnist-784-euclidean": (784,    60_000, 10_000, "euclidean", "none"),
    "mnist-784-euclidean":         (784,    60_000, 10_000, "euclidean", "none"),
    "gist-960-euclidean":          (960, 1_000_000,  1_000, "euclidean", "none"),
    "deep-image-96-angular":       (96,  9_990_000, 10_000, "angular",   "none"),
    "lastfm-64-dot":               (64,    292_385, 50_000, "dot",       "none"),
}


def _parse_filename_params(name: str) -> dict[str, Any]:
    base = name[:-5] if name.endswith(".hdf5") else name
    parts = base.split("_")
    out: dict[str, Any] = {"distance": parts[0]}
    i = 1
    while i < len(parts):
        if parts[i] == "M":
            out["M"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "efConstruction":
            out["ef_construction"] = int(parts[i + 1])
            i += 2
        else:
            try:
                out["ef_search"] = int(parts[i])
            except ValueError:
                pass
            i += 1
    return out


def _load_truth_topk(dataset: str, annb_root: Path, k: int) -> np.ndarray:
    ds_path = annb_root / "data" / f"{dataset}.hdf5"
    with h5py.File(ds_path, "r") as f:
        return np.array(f["neighbors"])[:, :k]


def _recall_at_k(truth_topk: np.ndarray, found: np.ndarray) -> float:
    n = truth_topk.shape[0]
    if n == 0:
        return 0.0
    k = truth_topk.shape[1]
    total = 0
    for i in range(n):
        total += len(set(truth_topk[i].tolist()) & set(found[i, :k].tolist()))
    return total / (n * k)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo)] + list(args), text=True
    ).strip()


def _hardware_block() -> dict[str, Any]:
    out: dict[str, Any] = {"arch": platform.machine()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        out["cpu_brand"] = m.group(1).strip() if m else platform.processor()
        physical = len({line.split(":")[1].strip() for line in cpuinfo.splitlines() if line.startswith("physical id")})
        cores = len({line.split(":")[1].strip() for line in cpuinfo.splitlines() if line.startswith("core id")})
        threads = sum(1 for line in cpuinfo.splitlines() if line.startswith("processor"))
        out["cpu_cores"] = max(physical * cores, 1) if cores else threads
        out["cpu_threads"] = threads
    except FileNotFoundError:
        out["cpu_brand"] = platform.processor()
        out["cpu_cores"] = os.cpu_count() or 1
        out["cpu_threads"] = os.cpu_count() or 1
    try:
        meminfo = Path("/proc/meminfo").read_text()
        m = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
        out["ram_gb"] = round(int(m.group(1)) / 1_048_576) if m else 0
    except FileNotFoundError:
        out["ram_gb"] = 0
    try:
        with open("/etc/os-release") as f:
            os_kv = dict(line.strip().split("=", 1) for line in f if "=" in line)
        out["os_name"] = os_kv.get("NAME", "Linux").strip('"')
        out["os_version"] = os_kv.get("VERSION_ID", "").strip('"')
    except FileNotFoundError:
        out["os_name"] = platform.system()
        out["os_version"] = platform.release()
    return out


def _emit_one_dataset(
    annb_root: Path,
    algorithm: str,
    subject: str,
    dataset: str,
    k: int,
    coordinode_repo: Path,
    sha_override: str | None,
    version: str,
    codec: str,
    out_dir: Path,
    ts: str,
) -> list[Path]:
    """Emit one JSON per M for this dataset.  Returns paths written."""
    results_dir = annb_root / "results" / dataset / str(k) / algorithm
    if not results_dir.is_dir():
        print(f"  no results for {dataset} — skipping")
        return []
    dim, n_train, n_test, distance, _default_codec = DATASET_META.get(
        dataset, (None, None, None, None, "none")
    )
    truth_topk = _load_truth_topk(dataset, annb_root, k)

    # Group HDF5 files by M.
    by_m: dict[int, list[Path]] = {}
    for h5path in results_dir.glob("*.hdf5"):
        params = _parse_filename_params(h5path.name)
        by_m.setdefault(params.get("M", -1), []).append(h5path)

    # Git block (per CoordiNode SHA we're tagging the result with).
    if sha_override:
        sha = sha_override
    else:
        sha = _git(coordinode_repo, "rev-parse", "HEAD")
    short = sha[:7]
    branch = _git(coordinode_repo, "rev-parse", "--abbrev-ref", "HEAD")
    commit_date = _git(coordinode_repo, "show", "-s", "--format=%cI", sha)
    dirty = bool(_git(coordinode_repo, "status", "--porcelain"))

    written: list[Path] = []
    for m, files in sorted(by_m.items()):
        files = sorted(files)
        sweep: list[dict[str, Any]] = []
        ef_construction = None
        build_s = None
        for h5path in files:
            params = _parse_filename_params(h5path.name)
            with h5py.File(h5path, "r") as f:
                neighbours = np.array(f["neighbors"])
                times = np.array(f["times"])
                if "build_time" in f.attrs:
                    build_s = float(f.attrs["build_time"])
            recall = _recall_at_k(truth_topk, neighbours)
            qps = float(len(times) / times.sum()) if times.sum() > 0 else 0.0
            latency_us = times * 1e6
            sweep.append(
                {
                    "ef_search": params.get("ef_search"),
                    "qps": round(qps, 2),
                    "recall_at_k": round(recall, 5),
                    "latency_us_mean": round(float(np.mean(latency_us)), 2),
                    "latency_us_p50": round(float(np.percentile(latency_us, 50)), 2),
                    "latency_us_p95": round(float(np.percentile(latency_us, 95)), 2),
                    "latency_us_p99": round(float(np.percentile(latency_us, 99)), 2),
                }
            )
            if ef_construction is None:
                ef_construction = params.get("ef_construction")
        # Operating point: prefer the highest QPS among configurations that
        # cleared recall ≥ 0.95.  When NO configuration reaches 0.95 (typical
        # for low-M HNSW on hard datasets), report the highest-recall point
        # we DID achieve — that's the more useful summary number than the
        # fastest-but-blind row.
        good = [s for s in sweep if s["recall_at_k"] >= 0.95]
        if good:
            peak = max(good, key=lambda s: s["qps"])
        else:
            peak = max(sweep, key=lambda s: s["recall_at_k"])
        report = {
            "schema_version": 1,
            "timestamp": ts,
            "git": {
                "sha": sha,
                "sha_short": short,
                "branch": branch,
                "dirty": dirty,
                "commit_date": commit_date,
            },
            "hardware": _hardware_block(),
            "modality": "vector",
            "benchmark": "ann-benchmarks",
            "dataset": dataset,
            "subject": subject,
            "codec": codec,
            "version": version,
            "metrics": {
                "build_secs": build_s,
                "dataset_dim": dim,
                "dataset_n_test": n_test,
                "dataset_n_train": n_train,
                "hnsw_ef_construction": ef_construction,
                "hnsw_m": m,
                "k": k,
                "qps_at_recall_peak": peak["qps"],
                "recall_at_k_peak": peak["recall_at_k"],
                "sweep": sweep,
            },
        }
        # Filename format mirrors the existing bench-results convention:
        # <sha>-<subject>-M<m>-YYYYMMDD-HHMMSS.json
        ts_compact = re.sub(r"[^0-9T]", "", ts).replace("T", "-")[:15]
        fname = f"{short}-{subject}-M{m}-{ts_compact}.json"
        out_path = out_dir / "vector" / dataset / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        written.append(out_path)
        print(f"  wrote {out_path.relative_to(out_dir)}")
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annb-root", required=True, type=Path)
    p.add_argument("--algorithm", required=True,
                   help="ann-benchmarks algorithm name (results subdirectory)")
    p.add_argument("--subject", default=None,
                   help="bench JSON `subject` field — defaults to --algorithm")
    p.add_argument("--coordinode-repo", required=True, type=Path,
                   help="path to coordinode repo (for git metadata)")
    p.add_argument("--sha", default=None,
                   help="coordinode SHA to tag results with (default: HEAD of --coordinode-repo)")
    p.add_argument("--datasets", required=True, help="comma-separated list")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out-dir", required=True, type=Path,
                   help="where to write bench-results/vector/<dataset>/*.json")
    p.add_argument("--codec", default="none")
    p.add_argument("--version", default="0.4.3",
                   help="subject library version label")
    args = p.parse_args()

    subject = args.subject or args.algorithm
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    total = 0
    for ds in datasets:
        print(f"Processing dataset {ds}")
        files = _emit_one_dataset(
            annb_root=args.annb_root,
            algorithm=args.algorithm,
            subject=subject,
            dataset=ds,
            k=args.k,
            coordinode_repo=args.coordinode_repo,
            sha_override=args.sha,
            version=args.version,
            codec=args.codec,
            out_dir=args.out_dir,
            ts=ts,
        )
        total += len(files)
    print(f"\nTotal JSON files written: {total}")


if __name__ == "__main__":
    main()
