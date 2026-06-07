#!/usr/bin/env python3
"""chromadb competitor runner — emits CoordiNode bench JSON schema.

chromadb is a multi-model vector DB that uses hnswlib under the
hood for ANN, but adds a serialisation layer, a metadata index,
and a query-time filter pipeline. Comparing against chromadb (vs
raw hnswlib) shows what the "vector DB tax" costs on the same
HNSW backbone — the gap is mostly the persistence layer plus the
per-query python round-trip.

Schema matches `crates/coordinode-bench/src/lib.rs` `BenchReport`
so the gh-pages dashboard renders CN, hnswlib and chromadb on the
same axes. `subject` is set to `"chromadb"`.

Invoke (on the runner, NOT in CI):

    docker run --rm \\
        -v <bench-data-root>/datasets:/data:ro \\
        -v $PWD/out:/out \\
        coordinode-bench-chromadb:latest \\
        --train /data/sift/sift_base.fvecs \\
        --query /data/sift/sift_query.fvecs \\
        --groundtruth /data/sift/sift_groundtruth.ivecs \\
        --dataset-name sift-128-euclidean \\
        --output /out \\
        --cn-sha $(git rev-parse HEAD)

The --cn-sha argument is critical: the JSON filename uses it so
the gh-pages renderer pairs the chromadb result with the matching
CN run. Same convention as the hnswlib adapter.
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

import chromadb  # type: ignore[import-untyped]
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


def detect_metric(dataset_name: str) -> str:
    """Map dataset suffix to chromadb's HNSW distance keyword."""
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
    p.add_argument(
        "--codec",
        default="none",
        help="Recorded for symmetry; chromadb has no codec dimension on the HNSW path.",
    )
    p.add_argument("--output", type=Path, default=Path("bench-results"))
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread cap for both build and search. chromadb's HNSW backend honours OMP_NUM_THREADS through the bundled hnswlib wheel. 1 = single-thread, 4 = the bench-host MT4 budget.",
    )
    p.add_argument(
        "--cn-sha",
        required=True,
        help="CoordiNode commit SHA this run is paired with — used in the filename so the dashboard correlates timepoints across subjects.",
    )
    args = p.parse_args()

    # Pin thread count BEFORE chromadb spins up its bundled hnswlib.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    print(f"[chromadb] threads={args.threads}", flush=True)

    print(f"[chromadb] loading {args.train}", flush=True)
    train = read_fvecs(args.train)
    print(f"[chromadb] loading {args.query}", flush=True)
    query = read_fvecs(args.query)
    print(f"[chromadb] loading {args.groundtruth}", flush=True)
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
    print(f"[chromadb] metric={metric} n_train={n_train} dim={d} n_test={n_test}", flush=True)

    # Build — in-memory client, single collection, HNSW config matching
    # the CN side (M, ef_construction, distance). chromadb stores ids
    # as strings on the wire; convert int → str at insert.
    client = chromadb.Client()
    collection = client.create_collection(
        name="bench",
        metadata={
            "hnsw:space": metric,
            "hnsw:M": args.m,
            "hnsw:construction_ef": args.ef_construction,
            "hnsw:num_threads": args.threads,
        },
    )
    t0 = time.time()
    # Batch inserts. chromadb v0.5 has a hard limit around 41_666 per
    # add() call (`MAX_BATCH_SIZE`); use 32_000 to stay safely below.
    batch_size = 32_000
    ids_all = [str(i) for i in range(n_train)]
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        collection.add(
            ids=ids_all[start:end],
            embeddings=train[start:end].tolist(),
        )
    build_secs = time.time() - t0
    print(f"[chromadb] build_secs={build_secs:.2f}", flush=True)

    # Sweep
    sweep_values = [int(x) for x in args.ef_sweep.split(",")]
    def single_search(q_vec):
        return collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=args.k,
        )

    points = []
    for ef in sweep_values:
        # chromadb exposes ef_search through collection.modify; the
        # metadata key is `hnsw:search_ef`.
        collection.modify(metadata={"hnsw:search_ef": ef})
        latencies_us: list[float] = []
        hits = 0
        total = 0
        wall_start = time.time()
        if args.threads == 1:
            for _round in range(REPLAY_ROUNDS):
                for q_idx in range(n_test):
                    q = query[q_idx]
                    t = time.time()
                    result = single_search(q)
                    latencies_us.append((time.time() - t) * 1e6)
                    gt_row = set(int(x) for x in gt[q_idx, : args.k])
                    for lab in result["ids"][0]:
                        if int(lab) in gt_row:
                            hits += 1
                    total += args.k
        else:
            # MT: ThreadPoolExecutor with N workers each issuing
            # independent collection.query calls. chromadb's HNSW
            # search releases the GIL inside C++ so real wall-clock
            # parallelism is achievable up to the runtime thread cap.
            from concurrent.futures import ThreadPoolExecutor

            for _round in range(REPLAY_ROUNDS):
                with ThreadPoolExecutor(max_workers=args.threads) as pool:
                    t_round = time.time()
                    results = list(pool.map(single_search, (query[i] for i in range(n_test))))
                    round_wall_us = (time.time() - t_round) * 1e6
                per_query_us = round_wall_us / n_test
                latencies_us.extend([per_query_us] * n_test)
                for q_idx, result in enumerate(results):
                    gt_row = set(int(x) for x in gt[q_idx, : args.k])
                    for lab in result["ids"][0]:
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
            f"[chromadb] ef={ef} recall={points[-1]['recall_at_k']:.4f} qps={points[-1]['qps']:.0f}",
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
            "branch": "main",
            "dirty": False,
            "commit_date": ts.isoformat(),
        },
        "hardware": hardware_fingerprint(),
        "modality": "vector",
        "benchmark": "ann-benchmarks",
        "dataset": args.dataset_name,
        "subject": "chromadb",
        "codec": args.codec,
        "version": getattr(chromadb, "__version__", "unknown"),
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
            "qps_at_recall_peak": max(
                p["qps"]
                for p in points
                if p["recall_at_k"] == max(q["recall_at_k"] for q in points)
            ),
        },
        "notes": None,
    }
    over_95 = [p for p in points if p["recall_at_k"] >= 0.95]
    if over_95:
        report["metrics"]["qps_at_recall_0_95"] = over_95[0]["qps"]

    out_dir = args.output / "vector" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{short_sha}-chromadb-t{args.threads}-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[chromadb] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
