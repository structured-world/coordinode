#!/usr/bin/env python3
"""Prepare a high-dimension benchmark dataset for the bench-vector-ann
harness. Two source paths:

  1. **Public HDF5** (`--hdf5-url`): download the canonical ann-benchmarks
     / HuggingFace artefact, run the HDF5 → fvecs converter. Use this
     where a real-world embedding corpus exists (e.g. 1536D
     dbpedia-openai-1000k).

  2. **Synthetic** (no URL): draw N base + Q query vectors from N(0, 1)
     on the requested dimension, then compute exact top-K nearest
     neighbours via brute force. Use this for dimensions that do not
     have a published evaluation corpus yet (3072, 4096, 8192). The
     vectors are L2-normalised so the resulting distribution looks
     close to what a real angular / cosine model emits.

Output triplet lives in
`$DATASET_ROOT/high-d/<d>/<prefix>_{base,query,groundtruth}.{fvecs,ivecs}`
which is exactly where the workflow looks.

Idempotent: re-runs detect existing fvecs and skip work.

Usage:

    python3 prepare_high_d.py \\
        --d        1536 \\
        --output   /data/datasets/high-d/1536 \\
        --prefix   dbpedia-openai-1M-1536 \\
        --hdf5-url https://ann-benchmarks.com/dbpedia-openai-1000k-angular.hdf5

    python3 prepare_high_d.py \\
        --d       4096 \\
        --n-train 100000 \\
        --n-test  1000 \\
        --output  /data/datasets/high-d/4096 \\
        --prefix  synthetic-4096
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with path.open("wb") as f:
        for row in arr:
            f.write(struct.pack("<i", d))
            f.write(row.tobytes())
    print(f"wrote {path} ({n} x {d}, {path.stat().st_size / 1e6:.1f} MB)")


def write_ivecs(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    n, k = arr.shape
    with path.open("wb") as f:
        for row in arr:
            f.write(struct.pack("<i", k))
            f.write(row.tobytes())
    print(f"wrote {path} ({n} x {k}, {path.stat().st_size / 1e6:.1f} MB)")


def synthesize(
    d: int, n_train: int, n_test: int, k: int, out_dir: Path, prefix: str
) -> None:
    rng = np.random.default_rng(seed=42)
    print(f"sampling train {n_train} x {d}")
    train = rng.standard_normal((n_train, d), dtype=np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True).clip(min=1e-8)

    print(f"sampling test {n_test} x {d}")
    test = rng.standard_normal((n_test, d), dtype=np.float32)
    test /= np.linalg.norm(test, axis=1, keepdims=True).clip(min=1e-8)

    print(f"computing brute-force top-{k} neighbours (could take minutes)")
    # On normalised vectors L2 == 2 - 2*cos; ordering is preserved by
    # cosine similarity, so a single matmul gives us the sort key.
    chunk = max(1, 1024 * 1024 // (n_train * 4 // (1 << 20) + 1))
    neighbors = np.empty((n_test, k), dtype=np.int32)
    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        sim = test[start:end] @ train.T
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        # Order top by descending similarity for a stable recall metric.
        ordered = np.take_along_axis(
            top, np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1), axis=1
        )
        neighbors[start:end] = ordered.astype(np.int32)
        print(f"  {end}/{n_test}")

    write_fvecs(out_dir / f"{prefix}_base.fvecs", train)
    write_fvecs(out_dir / f"{prefix}_query.fvecs", test)
    write_ivecs(out_dir / f"{prefix}_groundtruth.ivecs", neighbors)


def fetch_hdf5(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"hdf5 cache hit: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    subprocess.run(
        ["curl", "-fL", "--retry", "3", "-o", str(dest), url], check=True
    )
    return dest


def convert_hdf5(hdf5: Path, out_dir: Path, prefix: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "hdf5_to_fvecs.py"),
            "--hdf5",
            str(hdf5),
            "--out-dir",
            str(out_dir),
            "--prefix",
            prefix,
        ],
        check=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, required=True, help="Vector dimension.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--hdf5-url", default=None)
    p.add_argument(
        "--n-train",
        type=int,
        default=100_000,
        help="Synthetic only: number of base vectors.",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=1_000,
        help="Synthetic only: number of query vectors.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-K stored in groundtruth.ivecs.",
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / f"{args.prefix}_base.fvecs"
    query = args.output / f"{args.prefix}_query.fvecs"
    truth = args.output / f"{args.prefix}_groundtruth.ivecs"
    if base.is_file() and query.is_file() and truth.is_file():
        print(f"triplet already present in {args.output}, skip")
        return 0

    if args.hdf5_url:
        hdf5_path = fetch_hdf5(args.hdf5_url, args.output / "source.hdf5")
        convert_hdf5(hdf5_path, args.output, args.prefix)
    else:
        synthesize(
            args.d, args.n_train, args.n_test, args.k, args.output, args.prefix
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
