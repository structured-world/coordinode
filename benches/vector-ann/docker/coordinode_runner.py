#!/usr/bin/env python3
"""CoordiNode-over-gRPC competitor-style runner — server-tier bench datapoint.

This is the SERVER-tier measurement of CoordiNode: it drives a running
`coordinode serve` through the same python-grpcio, one-query-per-call path the
qdrant adapter uses, so CoordiNode and qdrant are compared on equal transport
(both pay the per-query gRPC round-trip). It is the fair peer to the qdrant
number; the in-process `bench-vector-ann` figure is the embedded-tier
measurement (peer to hnswlib / chromadb), and the two tiers are not mixed.

Pipeline, mirroring `qdrant_runner.py` / `hnswlib_runner.py`:
  1. compile the gRPC stubs from the proto tree (the google.api REST
     annotations are stripped first — grpc_tools cannot resolve them and the
     gRPC services do not need them),
  2. ingest the (subset of the) corpus via `ExecuteCypher` with inline vector
     list literals, tagging each node with its corpus index as `pid`,
  3. for each `ef_search` in the sweep, (re)create the HNSW index with that
     search beam (ef_search is index config, not a per-query RPC argument, so
     the sweep recreates the index — the same shape as the chromadb adapter),
     wait until it serves at full recall, then issue every query through the
     `VectorSearch` RPC and score recall@k by the returned `pid` property.

Schema matches `crates/coordinode-bench` BenchReport; `subject` is
`coordinode-grpc` so the dashboard places it in the server tier next to qdrant.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

SCHEMA_VERSION = 1
DEFAULT_EF_SWEEP = [40, 80, 135, 200, 400]
REPLAY_ROUNDS = 10
LABEL = "Bench"
PROP = "embedding"
INDEX = "bench_idx"


def read_fvecs(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = f.read()
    if not data:
        raise SystemExit(f"{path}: empty file")
    dim = struct.unpack("<i", data[:4])[0]
    stride = 4 + dim * 4
    if len(data) % stride != 0:
        raise SystemExit(f"{path}: size {len(data)} not a multiple of stride {stride}")
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


def _brute_force_groundtruth(train, query, k, metric):
    n_test = query.shape[0]
    out = np.empty((n_test, k), dtype=np.int32)
    chunk = max(1, 10_000_000 // max(1, train.shape[0]))
    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        if metric == "cosine":
            tn = train / np.linalg.norm(train, axis=1, keepdims=True).clip(min=1e-12)
            qn = query[start:end] / np.linalg.norm(query[start:end], axis=1, keepdims=True).clip(min=1e-12)
            sim = qn @ tn.T
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1)
        else:
            d2 = (
                (query[start:end] ** 2).sum(1, keepdims=True)
                - 2 * (query[start:end] @ train.T)
                + (train ** 2).sum(1)[None, :]
            )
            top = np.argpartition(d2, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(d2, top, axis=1), axis=1)
        out[start:end] = np.take_along_axis(top, order, axis=1).astype(np.int32)
    return out


def metric_keyword(dataset_name: str) -> str:
    if dataset_name.endswith("-euclidean"):
        return "l2"
    if dataset_name.endswith("-angular"):
        return "cosine"
    raise SystemExit(f"cannot detect metric from {dataset_name!r}: expected -euclidean / -angular")


def compile_stubs(proto_src: Path, gen: Path):
    """Strip the google.api REST annotations, then compile python gRPC stubs."""
    stripped = gen.parent / "cn_proto_stripped"
    protos = []
    for p in proto_src.rglob("*.proto"):
        rel = p.relative_to(proto_src)
        if str(rel).startswith("google/"):
            continue  # well-known types ship with grpc_tools
        out = stripped / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = p.read_text()
        text = re.sub(r'^\s*import\s+"google/api/annotations\.proto";\s*$', "", text, flags=re.M)
        text = re.sub(r"option\s*\(google\.api\.http\)\s*=\s*\{.*?\};", "", text, flags=re.S)
        out.write_text(text)
        protos.append(str(rel))
    gen.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "grpc_tools.protoc", f"-I{stripped}",
           f"--python_out={gen}", f"--grpc_python_out={gen}", *protos]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"protoc failed:\n{r.stderr}")
    for d, _, _ in os.walk(gen):
        Path(d, "__init__.py").touch()


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
    p.add_argument("--subset-size", type=int, default=0)
    p.add_argument("--codec", default="none")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cn-sha", required=True)
    p.add_argument("--addr", default="localhost:17080")
    p.add_argument("--proto-dir", type=Path, default=Path("/tmp/cbench-main/proto"))
    p.add_argument("--gen-dir", type=Path, default=Path("/tmp/cn_gen"))
    p.add_argument("--output", type=Path, default=Path("bench-results"))
    args = p.parse_args()

    print(f"[coordinode-grpc] loading {args.train}", flush=True)
    train = read_fvecs(args.train)
    query = read_fvecs(args.query)
    gt = read_ivecs(args.groundtruth)

    if 0 < args.subset_size < train.shape[0]:
        print(f"[coordinode-grpc] subset {args.subset_size}/{train.shape[0]}; recomputing GT", flush=True)
        train = train[: args.subset_size]
        m_str = "cosine" if args.dataset_name.endswith("-angular") else "euclidean"
        gt = _brute_force_groundtruth(train, query, args.k, m_str)
        suffix = f"{args.subset_size // 1000}k"
        parts = args.dataset_name.rsplit("-", 1)
        args.dataset_name = f"{parts[0]}-{suffix}-{parts[1]}" if len(parts) == 2 else f"{args.dataset_name}-{suffix}"

    n_train, d = train.shape
    n_test, q_d = query.shape
    if q_d != d:
        raise SystemExit(f"train dim {d} != query dim {q_d}")
    metric = metric_keyword(args.dataset_name)
    print(f"[coordinode-grpc] metric={metric} n_train={n_train} dim={d} n_test={n_test} "
          f"m={args.m} ef_c={args.ef_construction} threads={args.threads}", flush=True)

    compile_stubs(args.proto_dir, args.gen_dir)
    sys.path.insert(0, str(args.gen_dir))
    import grpc
    from coordinode.v1.query import cypher_pb2, cypher_pb2_grpc
    from coordinode.v1.query import vector_pb2, vector_pb2_grpc
    from coordinode.v1.common import types_pb2

    metric_enum = (vector_pb2.DistanceMetric.DISTANCE_METRIC_COSINE if metric == "cosine"
                   else vector_pb2.DistanceMetric.DISTANCE_METRIC_L2)

    # Generous message + keepalive limits: a 200-node ingest batch with 128-dim
    # inline literals is well under the server's 16 MiB cap but over grpc's tiny
    # default; bump both directions.
    chan_opts = [("grpc.max_send_message_length", 64 * 1024 * 1024),
                 ("grpc.max_receive_message_length", 64 * 1024 * 1024)]
    ch = grpc.insecure_channel(args.addr, options=chan_opts)
    cy = cypher_pb2_grpc.CypherServiceStub(ch)
    vec = vector_pb2_grpc.VectorServiceStub(ch)

    def cypher(q: str):
        return cy.ExecuteCypher(cypher_pb2.ExecuteCypherRequest(query=q), timeout=120)

    # Idempotent start: clear any prior run's index + nodes so a re-invocation
    # (for example the t1 pass followed by the t4 pass against the same server)
    # begins from an empty label rather than double-ingesting.
    try:
        cypher(f"DROP VECTOR INDEX {INDEX}")
    except grpc.RpcError:
        pass
    try:
        cypher(f"MATCH (n:{LABEL}) DETACH DELETE n")
    except grpc.RpcError:
        pass

    # Ingest once: inline UNWIND batches, tagging each node with its corpus
    # index as `pid` so VectorSearch results map back to the groundtruth.
    print("[coordinode-grpc] ingesting", flush=True)
    t0 = time.time()
    batch = 200
    for start in range(0, n_train, batch):
        end = min(start + batch, n_train)
        rows = []
        for i in range(start, end):
            lit = "[" + ",".join(f"{x:.6f}" for x in train[i].tolist()) + "]"
            rows.append(f"{{pid:{i},e:{lit}}}")
        cypher(f"UNWIND [{','.join(rows)}] AS row CREATE (n:{LABEL} {{pid:row.pid, {PROP}:row.e}})")
    ingest_secs = time.time() - t0
    print(f"[coordinode-grpc] ingest_secs={ingest_secs:.2f}", flush=True)

    def drop_index():
        try:
            cypher(f"DROP VECTOR INDEX {INDEX}")
        except grpc.RpcError:
            pass

    def create_index(ef_search: int) -> float:
        t = time.time()
        cypher(f'CREATE VECTOR INDEX {INDEX} ON :{LABEL}({PROP}) '
               f'OPTIONS {{ m: {args.m}, ef_construction: {args.ef_construction}, '
               f'ef_search: {ef_search}, metric: "{metric}", dimensions: {d} }}')
        # Wait until the index serves at full recall (health READY).
        deadline = time.time() + 600
        qv = types_pb2.Vector(values=query[0].tolist())
        probe = vector_pb2.VectorSearchRequest(label=LABEL, property=PROP, query_vector=qv,
                                               top_k=args.k, metric=metric_enum)
        ready_state = vector_pb2.VectorIndexHealth.ServingState.SERVING_STATE_READY
        while time.time() < deadline:
            r = vec.VectorSearch(probe, timeout=60)
            if r.index_health.serving_state == ready_state:
                break
            time.sleep(0.5)
        return time.time() - t

    def pids_of(resp) -> list[int]:
        # Node.properties is a proto map<string, PropertyValue>.
        out = []
        for res in resp.results:
            props = res.node.properties
            if "pid" in props:
                out.append(props["pid"].int_value)
        return out

    sweep_values = [int(x) for x in args.ef_sweep.split(",")]
    points = []
    build_secs = 0.0
    for ef_i, ef in enumerate(sweep_values):
        drop_index()
        bs = create_index(ef)
        if ef_i == 0:
            build_secs = bs
            print(f"[coordinode-grpc] build_secs={build_secs:.2f}", flush=True)

        def one_query(qi: int):
            qv = types_pb2.Vector(values=query[qi].tolist())
            req = vector_pb2.VectorSearchRequest(label=LABEL, property=PROP, query_vector=qv,
                                                 top_k=args.k, metric=metric_enum)
            t = time.perf_counter()
            resp = vec.VectorSearch(req, timeout=60)
            return (time.perf_counter() - t) * 1e6, pids_of(resp)

        latencies, hits, total = [], 0, 0
        wall = time.time()
        if args.threads == 1:
            for _ in range(REPLAY_ROUNDS):
                for qi in range(n_test):
                    lat, ids = one_query(qi)
                    latencies.append(lat)
                    truth = set(int(x) for x in gt[qi, : args.k])
                    hits += sum(1 for i in ids if i in truth)
                    total += args.k
        else:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                for _ in range(REPLAY_ROUNDS):
                    for qi, (lat, ids) in enumerate(pool.map(one_query, range(n_test))):
                        latencies.append(lat)
                        truth = set(int(x) for x in gt[qi, : args.k])
                        hits += sum(1 for i in ids if i in truth)
                        total += args.k
        wall = time.time() - wall
        latencies.sort()
        n = len(latencies)
        points.append({
            "ef_search": ef,
            "recall_at_k": hits / total,
            "qps": (n_test * REPLAY_ROUNDS) / wall,
            "latency_us_mean": sum(latencies) / n,
            "latency_us_p50": latencies[int(n * 0.50)],
            "latency_us_p95": latencies[int(n * 0.95)],
            "latency_us_p99": latencies[int(n * 0.99)],
        })
        print(f"[coordinode-grpc] ef={ef} recall={points[-1]['recall_at_k']:.4f} "
              f"qps={points[-1]['qps']:.0f}", flush=True)

    ts = datetime.now(timezone.utc)
    short_sha = args.cn_sha[:7]
    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts.isoformat(),
        "git": {"sha": args.cn_sha, "sha_short": short_sha, "branch": "main",
                "dirty": False, "commit_date": ts.isoformat()},
        "hardware": hardware_fingerprint(),
        "modality": "vector",
        "benchmark": "ann-benchmarks",
        "dataset": args.dataset_name,
        "subject": "coordinode-grpc",
        "codec": args.codec,
        "version": short_sha,
        "metrics": {
            "hnsw_m": args.m,
            "hnsw_ef_construction": args.ef_construction,
            "build_secs": build_secs,
            "ingest_secs": ingest_secs,
            "dataset_n_train": n_train,
            "dataset_n_test": n_test,
            "dataset_dim": d,
            "k": args.k,
            "threads": args.threads,
            "transport": "grpc",
            "sweep": points,
            "recall_at_k_peak": max(p["recall_at_k"] for p in points),
            "qps_at_recall_peak": max(p["qps"] for p in points
                                      if p["recall_at_k"] == max(q["recall_at_k"] for q in points)),
        },
        "notes": None,
    }
    over_95 = [p for p in points if p["recall_at_k"] >= 0.95]
    if over_95:
        report["metrics"]["qps_at_recall_0_95"] = over_95[0]["qps"]

    out_dir = args.output / "vector" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{short_sha}-coordinode-grpc-t{args.threads}-M{args.m}-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[coordinode-grpc] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
