#!/usr/bin/env python3
"""hnswlib competitor runner — emits CoordiNode bench JSON schema.

Reads SIFT-style .fvecs / .ivecs files (same format as the CN
adapter), builds hnswlib's HNSW index over the training vectors,
runs the same ef_search sweep, writes a JSON in
`bench-results/vector/<dataset>/<sha>-hnswlib-<ts>.json` format.

The output schema MUST match `crates/coordinode-bench/src/lib.rs`
`BenchReport` so the gh-pages dashboard can render CN and
hnswlib side-by-side.

Invoke (on the runner, NOT in CI):

    docker run --rm \\
        -v <bench-data-root>/datasets:/data:ro \\
        -v $PWD/out:/out \\
        coordinode-bench-hnswlib:latest \\
        --train /data/sift/sift_base.fvecs \\
        --query /data/sift/sift_query.fvecs \\
        --groundtruth /data/sift/sift_groundtruth.ivecs \\
        --dataset-name sift-128-euclidean \\
        --output /out \\
        --cn-sha $(git rev-parse HEAD)        # CoordiNode commit being compared against

The --cn-sha argument is critical: the JSON filename uses it so
the gh-pages renderer pairs the hnswlib result with the matching
CN run (the dashboard line for "hnswlib at CN commit X" comes from
this naming). Without it, hnswlib results can't be correlated to a
specific CN snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hnswlib  # type: ignore[import-untyped]
import numpy as np
import psutil


SCHEMA_VERSION = 1
DEFAULT_EF_SWEEP = [16, 32, 64, 128, 256, 512]
REPLAY_ROUNDS = 10


def read_fvecs(path: Path) -> np.ndarray:
    """Read a Texmex INRIA .fvecs file → (N, D) float32 array."""
    with path.open("rb") as f:
        data = f.read()
    if not data:
        raise SystemExit(f"{path}: empty file")
    dim = struct.unpack("<i", data[:4])[0]
    stride = 4 + dim * 4
    if len(data) % stride != 0:
        raise SystemExit(f"{path}: file size {len(data)} not a multiple of stride {stride}")
    count = len(data) // stride
    arr = np.frombuffer(data, dtype=np.float32).reshape(count, dim + 1)
    # Strip the per-row dim prefix (column 0).
    return arr[:, 1:].copy()


def read_ivecs(path: Path) -> np.ndarray:
    """Read a Texmex INRIA .ivecs file → (N, K) int32 array."""
    with path.open("rb") as f:
        data = f.read()
    dim = struct.unpack("<i", data[:4])[0]
    stride = 4 + dim * 4
    count = len(data) // stride
    arr = np.frombuffer(data, dtype=np.int32).reshape(count, dim + 1)
    return arr[:, 1:].copy()


def hardware_fingerprint() -> dict:
    """Match Rust `coordinode_bench::HardwareFingerprint`."""
    mem_gb = psutil.virtual_memory().total // (1024 ** 3)
    return {
        "cpu_brand": platform.processor() or socket.gethostname(),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": int(mem_gb),
        "os_name": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
    }


def detect_metric(dataset_name: str) -> str:
    if dataset_name.endswith("-euclidean"):
        return "l2"
    if dataset_name.endswith("-angular"):
        return "cosine"
    raise SystemExit(
        f"cannot detect metric from dataset name {dataset_name!r}: "
        "expected suffix -euclidean or -angular"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--groundtruth", type=Path, required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-sweep", default=",".join(str(x) for x in DEFAULT_EF_SWEEP))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--codec", default="none", help="Recorded for symmetry; hnswlib has no codec dimension")
    p.add_argument("--output", type=Path, default=Path("bench-results"))
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread cap for both build (add_items) and search (knn_query). MUST match the CoordiNode side for apples-to-apples comparison. Bench host is 8C/16T i9-9900K; 1 = single-thread, 4 = stable multi-thread (leaves headroom for OS / OMP overhead — 8 saturates and produces noisy numbers).",
    )
    p.add_argument(
        "--cn-sha",
        required=True,
        help="CoordiNode commit SHA this run is paired with — used in the filename so the dashboard correlates timepoints across subjects.",
    )
    args = p.parse_args()

    # Pin thread count BEFORE importing/using hnswlib internals — the
    # library reads OMP_NUM_THREADS at first index construction. Setting
    # only set_num_threads() lets OpenMP spin up the default pool before
    # we get a chance to constrain it.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    print(f"[hnswlib] threads={args.threads}", flush=True)

    print(f"[hnswlib] loading {args.train}", flush=True)
    train = read_fvecs(args.train)
    print(f"[hnswlib] loading {args.query}", flush=True)
    query = read_fvecs(args.query)
    print(f"[hnswlib] loading {args.groundtruth}", flush=True)
    gt = read_ivecs(args.groundtruth)

    n_train, d = train.shape
    n_test, q_d = query.shape
    gt_n, gt_k = gt.shape
    if q_d != d:
        raise SystemExit(f"train dim {d} != query dim {q_d}")
    if gt_n != n_test:
        raise SystemExit(f"groundtruth rows {gt_n} != query rows {n_test}")
    if gt_k < args.k:
        raise SystemExit(f"groundtruth K {gt_k} < requested --k {args.k}")

    metric = detect_metric(args.dataset_name)
    print(f"[hnswlib] metric={metric} n_train={n_train} dim={d} n_test={n_test}", flush=True)

    # Build
    index = hnswlib.Index(space=metric, dim=d)
    index.init_index(max_elements=n_train, ef_construction=args.ef_construction, M=args.m)
    # Belt-and-braces: also pin via the runtime API in case the OMP env
    # vars are honoured later than expected by the bundled wheel.
    index.set_num_threads(args.threads)
    t0 = time.time()
    index.add_items(train, np.arange(n_train, dtype=np.int64), num_threads=args.threads)
    build_secs = time.time() - t0
    print(f"[hnswlib] build_secs={build_secs:.2f}", flush=True)

    # Sweep
    sweep_values = [int(x) for x in args.ef_sweep.split(",")]
    points = []
    for ef in sweep_values:
        index.set_ef(ef)
        latencies_us: list[float] = []
        hits = 0
        total = 0
        wall_start = time.time()
        if args.threads == 1:
            # Single-thread: per-query measurement gives p50/p95/p99 latency.
            for _round in range(REPLAY_ROUNDS):
                for q_idx in range(n_test):
                    q = query[q_idx]
                    t = time.time()
                    labels, _ = index.knn_query(q.reshape(1, -1), k=args.k, num_threads=1)
                    latencies_us.append((time.time() - t) * 1e6)
                    gt_row = set(int(x) for x in gt[q_idx, : args.k])
                    for lab in labels[0]:
                        if int(lab) in gt_row:
                            hits += 1
                    total += args.k
        else:
            # Multi-thread: batched knn_query so hnswlib's OpenMP loop
            # actually distributes queries across `args.threads` cores.
            # Per-query latency loses meaning here so we synthesise it
            # from total wall / n_queries; the QPS number is wall-honest.
            for _round in range(REPLAY_ROUNDS):
                t = time.time()
                labels_batch, _ = index.knn_query(
                    query, k=args.k, num_threads=args.threads
                )
                round_wall_us = (time.time() - t) * 1e6
                per_query_us = round_wall_us / n_test
                latencies_us.extend([per_query_us] * n_test)
                for q_idx in range(n_test):
                    gt_row = set(int(x) for x in gt[q_idx, : args.k])
                    for lab in labels_batch[q_idx]:
                        if int(lab) in gt_row:
                            hits += 1
                    total += args.k
        wall = time.time() - wall_start
        latencies_us.sort()
        n = len(latencies_us)
        points.append(
            {
                "ef_search": ef,
                "recall_at_k": hits / total,
                "qps": (n_test * REPLAY_ROUNDS) / wall,
                "latency_us_mean": sum(latencies_us) / n,
                "latency_us_p50": latencies_us[int(n * 0.50)],
                "latency_us_p95": latencies_us[int(n * 0.95)],
                "latency_us_p99": latencies_us[int(n * 0.99)],
            }
        )
        print(
            f"[hnswlib] ef={ef} recall={points[-1]['recall_at_k']:.4f} qps={points[-1]['qps']:.0f}",
            flush=True,
        )

    ts = datetime.now(timezone.utc)
    short_sha = args.cn_sha[:7]
    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts.isoformat(),
        "git": {
            "sha": args.cn_sha,
            "sha_short": short_sha,
            "branch": "main",  # competitor runs are pinned to a CN main commit by user
            "dirty": False,
            "commit_date": ts.isoformat(),  # we don't extract commit date for competitor runs
        },
        "hardware": hardware_fingerprint(),
        "modality": "vector",
        "benchmark": "ann-benchmarks",
        "dataset": args.dataset_name,
        "subject": "hnswlib",
        "codec": args.codec,
        "version": hnswlib.__version__ if hasattr(hnswlib, "__version__") else "unknown",
        "metrics": {
            "hnsw_m": args.m,
            "hnsw_ef_construction": args.ef_construction,
            "build_secs": build_secs,
            "dataset_n_train": n_train,
            "dataset_n_test": n_test,
            "dataset_dim": d,
            "k": args.k,
            "threads": args.threads,
            "sweep": points,
            "recall_at_k_peak": max(p["recall_at_k"] for p in points),
            "qps_at_recall_peak": max(p["qps"] for p in points if p["recall_at_k"] == max(q["recall_at_k"] for q in points)),
        },
        "notes": None,
    }
    over_95 = [p for p in points if p["recall_at_k"] >= 0.95]
    if over_95:
        report["metrics"]["qps_at_recall_0_95"] = over_95[0]["qps"]

    out_dir = args.output / "vector" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{short_sha}-hnswlib-t{args.threads}-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[hnswlib] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
