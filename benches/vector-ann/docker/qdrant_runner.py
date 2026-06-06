#!/usr/bin/env python3
"""qdrant competitor runner with proper HNSW configuration.

The previous qdrant numbers in bench-data showed a flat r=1.0
curve across all ef values at low QPS, which is the signature
of an adapter that fails to pass `params.hnsw_ef` per query and
lets qdrant fall back to exact-search on the small 100k subset.
This adapter explicitly:

  1. Sets HnswConfigDiff(m, ef_construct, full_scan_threshold=0)
     so the HNSW path is ALWAYS used (no exact-search fallback).
  2. Passes SearchParams(hnsw_ef=ef, exact=False) on every query
     so the recall/QPS curve actually responds to the ef sweep.
  3. Drives multi-threaded search via a ThreadPoolExecutor with
     `threads` workers issuing concurrent single-query calls,
     matching ann-benchmarks methodology and the hnswlib /
     chromadb adapters here.

Schema matches `crates/coordinode-bench/src/lib.rs` `BenchReport`
so the gh-pages dashboard renders CN, hnswlib, chromadb and
qdrant on the same axes. `subject` is set to `"qdrant"`.

Invoke (on the runner, NOT in CI):

    docker run --rm \\
        -v <bench-data-root>/datasets:/data:ro \\
        -v $PWD/out:/out \\
        coordinode-bench-qdrant:latest \\
        --train /data/sift/sift_base.fvecs \\
        --query /data/sift/sift_query.fvecs \\
        --groundtruth /data/sift/sift_groundtruth.ivecs \\
        --dataset-name sift-128-euclidean \\
        --m 16 --ef-construction 200 \\
        --threads 1 \\
        --output /out \\
        --cn-sha $(git rev-parse HEAD)

For the parallel-efficiency E_core(4) sweep, run twice:
  --threads 1  → ST baseline
  --threads 4  → MT4 baseline
The filename encodes the thread count, the dashboard uses it.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)


SCHEMA_VERSION = 1
DEFAULT_EF_SWEEP = [20, 40, 80, 200, 400, 800]
REPLAY_ROUNDS = 10


def read_fvecs(path: Path) -> np.ndarray:
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
    return arr[:, 1:].copy()


def read_ivecs(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = f.read()
    dim = struct.unpack("<i", data[:4])[0]
    stride = 4 + dim * 4
    count = len(data) // stride
    arr = np.frombuffer(data, dtype=np.int32).reshape(count, dim + 1)
    return arr[:, 1:].copy()


def hardware_fingerprint() -> dict:
    mem_gb = psutil.virtual_memory().total // (1024**3)
    return {
        "cpu_brand": platform.processor() or socket.gethostname(),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": int(mem_gb),
        "os_name": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
    }


def detect_metric(dataset_name: str) -> Distance:
    if dataset_name.endswith("-euclidean"):
        return Distance.EUCLID
    if dataset_name.endswith("-angular"):
        return Distance.COSINE
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
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-sweep", default=",".join(str(x) for x in DEFAULT_EF_SWEEP))
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--codec",
        default="none",
        help="Recorded for symmetry; this adapter does not exercise quantization.",
    )
    p.add_argument("--output", type=Path, default=Path("bench-results"))
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Concurrent search workers. 1 = ST baseline, 4 = MT4 budget matching the CN / hnswlib MT runs.",
    )
    p.add_argument(
        "--cn-sha",
        required=True,
        help="CoordiNode commit SHA the run is paired with (filename prefix for dashboard correlation).",
    )
    args = p.parse_args()

    print(f"[qdrant] loading {args.train}", flush=True)
    train = read_fvecs(args.train)
    print(f"[qdrant] loading {args.query}", flush=True)
    query = read_fvecs(args.query)
    print(f"[qdrant] loading {args.groundtruth}", flush=True)
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
    print(
        f"[qdrant] metric={metric.value} n_train={n_train} dim={d} n_test={n_test} m={args.m} ef_c={args.ef_construction} threads={args.threads}",
        flush=True,
    )

    # `:memory:` mode runs the qdrant store in-process, the same way
    # hnswlib and chromadb adapters operate, so the comparison stays
    # apples-to-apples (no network RTT, no separate server box).
    client = QdrantClient(":memory:")
    # `full_scan_threshold=0` is the critical knob: it forces qdrant
    # to use the HNSW index even on small collections where it would
    # otherwise fall back to exact search and produce the bogus flat
    # r=1.0 curve seen in the previous adapter's output.
    client.create_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=d, distance=metric),
        hnsw_config=HnswConfigDiff(
            m=args.m,
            ef_construct=args.ef_construction,
            full_scan_threshold=0,
            max_indexing_threads=args.threads,
        ),
    )
    t0 = time.time()
    batch = 10_000
    for start in range(0, n_train, batch):
        end = min(start + batch, n_train)
        points = [
            PointStruct(id=i, vector=train[i].tolist())
            for i in range(start, end)
        ]
        client.upsert(collection_name="bench", points=points)
    build_secs = time.time() - t0
    print(f"[qdrant] build_secs={build_secs:.2f}", flush=True)

    sweep_values = [int(x) for x in args.ef_sweep.split(",")]
    points_out = []
    for ef in sweep_values:
        params = SearchParams(hnsw_ef=ef, exact=False)

        def one_query(q_idx: int) -> tuple[float, list[int]]:
            t = time.perf_counter()
            result = client.search(
                collection_name="bench",
                query_vector=query[q_idx].tolist(),
                limit=args.k,
                search_params=params,
            )
            latency_us = (time.perf_counter() - t) * 1e6
            return latency_us, [int(r.id) for r in result]

        latencies_us: list[float] = []
        hits = 0
        total = 0
        wall_start = time.time()
        if args.threads == 1:
            for _round in range(REPLAY_ROUNDS):
                for q_idx in range(n_test):
                    lat, ids = one_query(q_idx)
                    latencies_us.append(lat)
                    gt_row = set(int(x) for x in gt[q_idx, : args.k])
                    hits += sum(1 for i in ids if i in gt_row)
                    total += args.k
        else:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                for _round in range(REPLAY_ROUNDS):
                    indices = list(range(n_test))
                    results = list(pool.map(one_query, indices))
                    for q_idx, (lat, ids) in zip(indices, results):
                        latencies_us.append(lat)
                        gt_row = set(int(x) for x in gt[q_idx, : args.k])
                        hits += sum(1 for i in ids if i in gt_row)
                        total += args.k
        wall = time.time() - wall_start
        latencies_us.sort()
        n = len(latencies_us)
        points_out.append(
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
            f"[qdrant] ef={ef} recall={points_out[-1]['recall_at_k']:.4f} qps={points_out[-1]['qps']:.0f}",
            flush=True,
        )

    ts = datetime.now(timezone.utc)
    short_sha = args.cn_sha[:7]
    qps_over_95 = [p for p in points_out if p["recall_at_k"] >= 0.95]
    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts.isoformat(),
        "git": {
            "sha": args.cn_sha,
            "sha_short": short_sha,
            "branch": "main",
            "dirty": False,
            "commit_date": ts.isoformat(),
        },
        "hardware": hardware_fingerprint(),
        "modality": "vector",
        "benchmark": "ann-benchmarks",
        "dataset": args.dataset_name,
        "subject": "qdrant",
        "codec": args.codec,
        "version": "1.13.0",
        "metrics": {
            "hnsw_m": args.m,
            "hnsw_ef_construction": args.ef_construction,
            "build_secs": build_secs,
            "dataset_n_train": n_train,
            "dataset_n_test": n_test,
            "dataset_dim": d,
            "k": args.k,
            "threads": args.threads,
            "sweep": points_out,
            "recall_at_k_peak": max(p["recall_at_k"] for p in points_out),
            "qps_at_recall_peak": max(
                p["qps"]
                for p in points_out
                if p["recall_at_k"] == max(q["recall_at_k"] for q in points_out)
            ),
        },
        "notes": None,
    }
    if qps_over_95:
        report["metrics"]["qps_at_recall_0_95"] = qps_over_95[0]["qps"]

    out_dir = args.output / "vector" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{short_sha}-qdrant-t{args.threads}-M{args.m}-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[qdrant] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
