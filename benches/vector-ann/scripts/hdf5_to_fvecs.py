#!/usr/bin/env python3
"""Convert an ann-benchmarks / VDBBench style HDF5 dataset to the
Texmex `.fvecs` / `.ivecs` triplet the bench-vector-ann harness
expects on disk.

HDF5 inputs carry four arrays:
  - `train`     (N, D) float32 — base vectors used to build the index
  - `test`      (Q, D) float32 — query vectors
  - `neighbors` (Q, K) int32   — ground-truth nearest-neighbour ids
  - `distances` (Q, K) float32 — unused (we measure recall by id, not
                                 by distance match)

Output triplet at `--out-dir`:
  - `<prefix>_base.fvecs`
  - `<prefix>_query.fvecs`
  - `<prefix>_groundtruth.ivecs`

`.fvecs` layout: each row is `int32 dim` followed by `dim` `float32`s.
`.ivecs` layout: each row is `int32 dim` followed by `dim` `int32`s.

Usage:

    python3 hdf5_to_fvecs.py \\
        --hdf5    /data/raw/dbpedia-openai-1M-1536.hdf5 \\
        --out-dir /data/datasets/dbpedia-openai-1M-1536 \\
        --prefix  dbpedia
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix (e.g. 'dbpedia' → dbpedia_base.fvecs).",
    )
    p.add_argument(
        "--train-key",
        default="train",
        help="HDF5 dataset name for the base vectors (default: train).",
    )
    p.add_argument(
        "--test-key",
        default="test",
        help="HDF5 dataset name for the query vectors (default: test).",
    )
    p.add_argument(
        "--neighbors-key",
        default="neighbors",
        help="HDF5 dataset name for ground-truth neighbour ids (default: neighbors).",
    )
    args = p.parse_args()

    if not args.hdf5.is_file():
        raise SystemExit(f"input not found: {args.hdf5}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as h:
        print(f"opening {args.hdf5}, keys = {list(h.keys())}")
        for key in (args.train_key, args.test_key, args.neighbors_key):
            if key not in h:
                raise SystemExit(f"missing key '{key}' in {args.hdf5}")

        train = np.asarray(h[args.train_key])
        test = np.asarray(h[args.test_key])
        neighbors = np.asarray(h[args.neighbors_key])

    if train.ndim != 2 or test.ndim != 2 or neighbors.ndim != 2:
        raise SystemExit(
            f"unexpected shapes: train {train.shape} test {test.shape} neighbors {neighbors.shape}"
        )
    if train.shape[1] != test.shape[1]:
        raise SystemExit(
            f"train dim {train.shape[1]} != test dim {test.shape[1]}"
        )
    if neighbors.shape[0] != test.shape[0]:
        raise SystemExit(
            f"neighbors rows {neighbors.shape[0]} != query rows {test.shape[0]}"
        )

    print(
        f"train {train.shape} {train.dtype}, "
        f"test {test.shape} {test.dtype}, "
        f"neighbors {neighbors.shape} {neighbors.dtype}"
    )

    write_fvecs(args.out_dir / f"{args.prefix}_base.fvecs", train)
    write_fvecs(args.out_dir / f"{args.prefix}_query.fvecs", test)
    write_ivecs(args.out_dir / f"{args.prefix}_groundtruth.ivecs", neighbors)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
